from __future__ import annotations

import json
import math
import os
import platform
import tempfile
import zipfile
import sys
import time
import urllib.request
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    mp = None
    MEDIAPIPE_ERROR = exc
else:
    MEDIAPIPE_ERROR = None

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover
    tf = None
    TENSORFLOW_ERROR = exc
else:
    TENSORFLOW_ERROR = None

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

IMAGE_MODEL_PATH = MODELS_DIR / "gesture_model.keras"
LANDMARK_MODEL_PATH = MODELS_DIR / "landmark_svm.joblib"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
ENSEMBLE_CONFIG_PATH = MODELS_DIR / "ensemble_config.json"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

IMG_SIZE = (224, 224)
DEFAULT_CLASS_NAMES = ["fist", "open_palm", "peace", "point_up", "thumbs_up"]
DISPLAY_NAMES = {
    "fist": "FIST ✊",
    "open_palm": "OPEN PALM ✋",
    "peace": "PEACE / VICTORY ✌️",
    "point_up": "POINT UP ☝️",
    "thumbs_up": "THUMBS UP 👍",
    "unknown": "UNKNOWN / LOW CONFIDENCE",
}
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]
TIP_IDS = [4, 8, 12, 16, 20]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_class_names() -> list[str]:
    value = load_json(CLASS_NAMES_PATH, DEFAULT_CLASS_NAMES)
    if isinstance(value, list) and len(value) == 5:
        return [str(item) for item in value]
    return DEFAULT_CLASS_NAMES.copy()


CLASS_NAMES = load_class_names()
ENSEMBLE_CONFIG = load_json(
    ENSEMBLE_CONFIG_PATH,
    {"image_weight": 0.7, "landmark_feature_set": "geometry"},
)


def category_name(category: Any) -> str:
    return str(
        getattr(category, "category_name", None)
        or getattr(category, "categoryName", None)
        or ""
    )


def ensure_hand_model() -> Path:
    if HAND_MODEL_PATH.exists() and HAND_MODEL_PATH.stat().st_size > 1_000_000:
        return HAND_MODEL_PATH
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the official MediaPipe Hand Landmarker model. "
            f"Place hand_landmarker.task in {MODELS_DIR}. Original error: {exc}"
        ) from exc
    return HAND_MODEL_PATH


@st.cache_resource(show_spinner=False)
def load_hand_landmarker():
    if mp is None:
        raise RuntimeError(f"MediaPipe import failed: {MEDIAPIPE_ERROR}")
    model_path = ensure_hand_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def _remove_null_quantization_config(value: Any) -> int:
    """Remove metadata written by newer Keras when no quantization was used."""
    removed = 0
    if isinstance(value, dict):
        if value.get("quantization_config", object()) is None:
            value.pop("quantization_config", None)
            removed += 1
        for child in value.values():
            removed += _remove_null_quantization_config(child)
    elif isinstance(value, list):
        for child in value:
            removed += _remove_null_quantization_config(child)
    return removed


def _make_keras_archive_compatible(source: Path) -> Path:
    """Create a compatible .keras copy without changing model weights."""
    destination = source.with_name(f"{source.stem}_compatible.keras")
    if (
        destination.exists()
        and destination.stat().st_mtime >= source.stat().st_mtime
        and destination.stat().st_size > 0
    ):
        return destination

    with zipfile.ZipFile(source, "r") as source_zip:
        try:
            config = json.loads(source_zip.read("config.json"))
        except KeyError as exc:
            raise RuntimeError(
                "The .keras archive does not contain config.json. "
                "Please export the model again from Colab."
            ) from exc

        removed = _remove_null_quantization_config(config)
        if removed == 0:
            raise RuntimeError(
                "The model could not be loaded, but no null quantization_config "
                "metadata was found. Match the local TensorFlow/Keras versions "
                "to the Colab versions."
            )

        fd, temporary_name = tempfile.mkstemp(
            prefix="gesture_model_", suffix=".keras", dir=str(source.parent)
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as target_zip:
                for info in source_zip.infolist():
                    data = source_zip.read(info.filename)
                    if info.filename == "config.json":
                        data = json.dumps(config).encode("utf-8")
                    target_zip.writestr(info, data)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    return destination


@st.cache_resource(show_spinner=False)
def load_image_model():
    if tf is None or not IMAGE_MODEL_PATH.exists():
        return None

    try:
        model = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)
    except (TypeError, ValueError) as error:
        # Colab may save with a newer Keras version whose Dense config contains
        # quantization_config=None. Older Keras releases reject that harmless
        # metadata even though this model was never quantized.
        if "quantization_config" not in str(error):
            raise
        compatible_path = _make_keras_archive_compatible(IMAGE_MODEL_PATH)
        model = tf.keras.models.load_model(compatible_path, compile=False)

    try:
        _ = model(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32), training=False)
    except Exception:
        pass
    return model


@st.cache_resource(show_spinner=False)
def load_landmark_model():
    if not LANDMARK_MODEL_PATH.exists():
        return None
    return joblib.load(LANDMARK_MODEL_PATH)


def transform_condition(image: Image.Image, condition: str) -> Image.Image:
    image = image.convert("RGB")
    if condition == "Original":
        return image
    if condition == "Low light":
        return ImageEnhance.Brightness(image).enhance(0.45)
    if condition == "Gaussian blur":
        return image.filter(ImageFilter.GaussianBlur(radius=3.0))
    if condition == "High contrast":
        return ImageEnhance.Contrast(image).enhance(1.8)
    if condition == "Rotate +15°":
        return image.rotate(
            15,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(127, 127, 127),
        )
    raise ValueError(f"Unknown condition: {condition}")


def prepare_model_input(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    padded = ImageOps.pad(
        image.convert("RGB"),
        IMG_SIZE,
        method=Image.Resampling.LANCZOS,
        color=(127, 127, 127),
    )
    return padded, np.asarray(padded, dtype=np.float32)[None, ...]


def detect_landmarks(image: Image.Image) -> dict[str, Any]:
    landmarker = load_hand_landmarker()
    image = image.convert("RGB")
    attempts = [("original", image), ("mirrored", ImageOps.mirror(image))]
    total_start = time.perf_counter()
    for attempt_name, candidate in attempts:
        array = np.ascontiguousarray(np.asarray(candidate, dtype=np.uint8))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
        result = landmarker.detect(mp_image)
        if result.hand_landmarks:
            image_points = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
                dtype=np.float32,
            )
            if attempt_name == "mirrored":
                image_points[:, 0] = 1.0 - image_points[:, 0]
            if result.hand_world_landmarks:
                world_points = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]],
                    dtype=np.float32,
                )
            else:
                world_points = image_points.copy()
            handedness = ""
            if result.handedness and result.handedness[0]:
                handedness = category_name(result.handedness[0][0])
            return {
                "image_points": image_points,
                "world_points": world_points,
                "handedness": handedness,
                "attempt": attempt_name,
                "latency_ms": (time.perf_counter() - total_start) * 1000,
            }
    return {
        "image_points": None,
        "world_points": None,
        "handedness": "",
        "attempt": "not_detected",
        "latency_ms": (time.perf_counter() - total_start) * 1000,
    }


def normalized_coordinates(detection: dict[str, Any]) -> np.ndarray:
    points = np.asarray(detection["world_points"], dtype=np.float32).copy()
    points -= points[0]
    if str(detection.get("handedness", "")).lower() == "left":
        points[:, 0] *= -1
    scale = float(np.max(np.linalg.norm(points, axis=1)))
    if scale < 1e-6:
        scale = 1.0
    return points / scale


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    denominator = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denominator < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0))
    return math.acos(cosine) / math.pi


def landmark_feature_vector(
    detection: dict[str, Any], feature_set: str = "geometry"
) -> np.ndarray:
    points = normalized_coordinates(detection)
    coordinates = points.reshape(-1)
    if feature_set == "coordinates":
        return coordinates.astype(np.float32)
    angles = np.array(
        [angle_at(points[a], points[b], points[c]) for a, b, c in ANGLE_TRIPLETS],
        dtype=np.float32,
    )
    fingertip_distances = np.array(
        [np.linalg.norm(points[a] - points[b]) for a, b in combinations(TIP_IDS, 2)],
        dtype=np.float32,
    )
    return np.concatenate([coordinates, angles, fingertip_distances]).astype(np.float32)


def draw_landmarks(image: Image.Image, image_points: np.ndarray | None) -> Image.Image:
    output = image.convert("RGB").copy()
    if image_points is None:
        return output
    draw = ImageDraw.Draw(output)
    width, height = output.size
    points = [
        (int(np.clip(x, 0, 1) * width), int(np.clip(y, 0, 1) * height))
        for x, y, _ in image_points
    ]
    for start, end in HAND_CONNECTIONS:
        draw.line(
            [points[start], points[end]],
            fill=(0, 220, 255),
            width=max(2, width // 180),
        )
    radius = max(3, width // 120)
    for x, y in points:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 70, 70),
        )
    return output


def call_layer(layer: Any, value: Any) -> Any:
    try:
        return layer(value, training=False)
    except TypeError:
        return layer(value)


def feature_layer_index(model: Any) -> int:
    candidates: list[int] = []
    for index, layer in enumerate(model.layers):
        if tf is not None and isinstance(layer, tf.keras.layers.InputLayer):
            continue
        shape = getattr(getattr(layer, "output", None), "shape", None)
        if shape is not None and len(shape) == 4:
            candidates.append(index)
    if not candidates:
        raise ValueError("No spatial feature layer was found for Grad-CAM.")
    return candidates[-1]


def make_gradcam_heatmap(
    model: Any, image_array: np.ndarray, class_index: int
) -> np.ndarray:
    target_index = feature_layer_index(model)
    value = tf.convert_to_tensor(image_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        features = None
        for index, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            value = call_layer(layer, value)
            if index == target_index:
                features = value
                tape.watch(features)
        predictions = value
        target = predictions[:, class_index]
    gradients = tape.gradient(target, features)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are unavailable for this model.")
    pooled = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(features[0] * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy()


def overlay_heatmap(
    image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42
) -> Image.Image:
    import matplotlib

    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(image.size)
    colors = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    colored = Image.fromarray(np.uint8(colors * 255)).convert("RGB")
    return Image.blend(image.convert("RGB"), colored, alpha)


def predict_image_model(image: Image.Image) -> dict[str, Any] | None:
    model = load_image_model()
    if model is None:
        return None
    processed, array = prepare_model_input(image)
    start = time.perf_counter()
    probabilities = np.asarray(model(array, training=False))[0]
    latency_ms = (time.perf_counter() - start) * 1000
    index = int(probabilities.argmax())
    try:
        heatmap = make_gradcam_heatmap(model, array, index)
        explanation = overlay_heatmap(processed, heatmap)
        gradcam_error = None
    except Exception as exc:
        explanation = processed
        gradcam_error = str(exc)
    return {
        "probabilities": probabilities,
        "prediction_index": index,
        "confidence": float(probabilities[index]),
        "latency_ms": latency_ms,
        "input": processed,
        "explanation": explanation,
        "gradcam_error": gradcam_error,
    }


def predict_landmark_model(
    detection: dict[str, Any]
) -> dict[str, Any] | None:
    model = load_landmark_model()
    if model is None or detection["image_points"] is None:
        return None
    feature_set = str(ENSEMBLE_CONFIG.get("landmark_feature_set", "geometry"))
    feature = landmark_feature_vector(detection, feature_set)[None, ...]
    start = time.perf_counter()
    raw_probabilities = model.predict_proba(feature)[0]
    classifier_latency = (time.perf_counter() - start) * 1000
    probabilities = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for source_index, class_value in enumerate(model.classes_):
        probabilities[int(class_value)] = raw_probabilities[source_index]
    index = int(probabilities.argmax())
    return {
        "probabilities": probabilities,
        "prediction_index": index,
        "confidence": float(probabilities[index]),
        "latency_ms": detection["latency_ms"] + classifier_latency,
    }


def combined_prediction(
    image_result: dict[str, Any] | None,
    landmark_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if image_result is None and landmark_result is None:
        return None
    if image_result is None:
        return {**landmark_result, "mode": "Landmark SVM"}
    if landmark_result is None:
        return {**image_result, "mode": "Image model fallback"}
    image_weight = float(ENSEMBLE_CONFIG.get("image_weight", 0.7))
    probabilities = (
        image_weight * image_result["probabilities"]
        + (1.0 - image_weight) * landmark_result["probabilities"]
    )
    index = int(probabilities.argmax())
    return {
        "probabilities": probabilities,
        "prediction_index": index,
        "confidence": float(probabilities[index]),
        "latency_ms": image_result["latency_ms"] + landmark_result["latency_ms"],
        "mode": f"Fusion ({image_weight:.1f} image / {1-image_weight:.1f} geometry)",
    }


st.set_page_config(page_title="GestureVision A2", page_icon="✋", layout="wide")
st.title("GestureVision A2 — Explainable Hand Gesture Recognition")
st.caption(
    "Five static classes: fist, open palm, peace, point up, and thumbs up. "
    "This demonstration is not a sign-language translator."
)

image_available = IMAGE_MODEL_PATH.exists() and tf is not None
landmark_available = LANDMARK_MODEL_PATH.exists() and mp is not None

with st.sidebar:
    st.header("Demo settings")
    condition = st.selectbox(
        "Environment stress test",
        ["Original", "Low light", "Gaussian blur", "High contrast", "Rotate +15°"],
    )
    confidence_threshold = st.slider(
        "Low-confidence threshold", 0.0, 0.95, 0.40, 0.05
    )
    preferred_mode = st.selectbox(
        "Prediction mode",
        ["Automatic fusion", "Image model", "Landmark SVM"],
    )
    st.divider()
    st.write("**Model files**")
    st.write("Image model:", "available" if image_available else "missing")
    st.write("Landmark SVM:", "available" if landmark_available else "missing")
    st.divider()
    st.write("**Server environment**")
    st.write(platform.system(), platform.machine())
    st.write("Python", platform.python_version())
    if tf is not None:
        st.write("TensorFlow", tf.__version__)
        st.write(
            "Inference device:",
            "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        )
    try:
        user_agent = st.context.headers.get("User-Agent", "Unavailable")
    except Exception:
        user_agent = "Unavailable"
    st.caption(f"Browser user agent: {user_agent}")

input_mode = st.radio("Input source", ["Camera", "Upload"], horizontal=True)
if input_mode == "Camera":
    source = st.camera_input("Take a clear photo of one hand", resolution="720p")
else:
    source = st.file_uploader(
        "Upload a JPG, JPEG, PNG, or WEBP image",
        type=["jpg", "jpeg", "png", "webp"],
    )

if source is None:
    st.info(
        "Keep the wrist and all fingers visible. The app will show prediction, "
        "confidence, latency, Grad-CAM, and 21 hand landmarks."
    )
    st.stop()

try:
    original = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
except Exception as exc:
    st.error(f"Could not read the image: {exc}")
    st.stop()

processed = transform_condition(original, condition)
if condition != "Original":
    st.warning(
        f"Synthetic stress-test transformation applied: **{condition}**. "
        "This is deterministic image processing, not generative AI."
    )

try:
    detection = detect_landmarks(processed)
except Exception as exc:
    detection = {
        "image_points": None,
        "world_points": None,
        "handedness": "",
        "attempt": "error",
        "latency_ms": 0.0,
    }
    st.warning(f"MediaPipe Hand Landmarker could not run: {exc}")

image_result = predict_image_model(processed) if image_available else None
landmark_result = (
    predict_landmark_model(detection) if landmark_available else None
)

if preferred_mode == "Image model":
    final_result = (
        {**image_result, "mode": "Image model"} if image_result is not None else None
    )
elif preferred_mode == "Landmark SVM":
    final_result = (
        {**landmark_result, "mode": "Landmark SVM"}
        if landmark_result is not None
        else None
    )
else:
    final_result = combined_prediction(image_result, landmark_result)

if final_result is None:
    st.error(
        "No trained model is available. Run the Colab notebook and copy its exported "
        "model files into the repository's models/ folder."
    )
    st.stop()

prediction_index = int(final_result["prediction_index"])
prediction_label = CLASS_NAMES[prediction_index]
confidence = float(final_result["confidence"])
if confidence < confidence_threshold:
    displayed_label = "unknown"
else:
    displayed_label = prediction_label

landmark_overlay = draw_landmarks(processed, detection.get("image_points"))
if image_result is not None:
    gradcam_overlay = image_result["explanation"]
    image_input = image_result["input"]
else:
    image_input = processed
    gradcam_overlay = processed

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Model input")
    st.image(image_input, use_container_width=True)
with col2:
    st.subheader("Grad-CAM")
    st.image(gradcam_overlay, use_container_width=True)
    if image_result and image_result.get("gradcam_error"):
        st.caption("Grad-CAM fallback: " + image_result["gradcam_error"])
with col3:
    st.subheader("21-point hand skeleton")
    st.image(landmark_overlay, use_container_width=True)
    if detection.get("image_points") is None:
        st.caption("No hand landmarks detected; fusion falls back to the image model.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Prediction", DISPLAY_NAMES.get(displayed_label, displayed_label))
m2.metric("Confidence", f"{confidence:.1%}")
m3.metric("Total latency", f"{float(final_result['latency_ms']):.1f} ms")
m4.metric("Mode", str(final_result.get("mode", preferred_mode)))

score_frame = pd.DataFrame(
    {
        "gesture": [DISPLAY_NAMES.get(name, name) for name in CLASS_NAMES],
        "probability": final_result["probabilities"],
    }
).sort_values("probability", ascending=False)
st.subheader("Final class scores")
st.bar_chart(score_frame.set_index("gesture"))

model_rows = []
if image_result is not None:
    model_rows.append(
        {
            "component": "Image model",
            "prediction": CLASS_NAMES[int(image_result["prediction_index"])],
            "confidence": float(image_result["confidence"]),
            "latency_ms": float(image_result["latency_ms"]),
        }
    )
if landmark_result is not None:
    model_rows.append(
        {
            "component": "MediaPipe landmarks + SVM",
            "prediction": CLASS_NAMES[int(landmark_result["prediction_index"])],
            "confidence": float(landmark_result["confidence"]),
            "latency_ms": float(landmark_result["latency_ms"]),
        }
    )
if model_rows:
    st.subheader("Component comparison")
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

if displayed_label == "unknown":
    st.warning(
        "The confidence is below the selected threshold. Try a clearer pose, better "
        "lighting, full finger visibility, and a simpler background."
    )

st.divider()
st.subheader("Ethics, consent, and limitations")
st.write(
    "Use photographs only with participant consent. The repository should not expose "
    "private raw images unless participants agreed to public release. Performance may "
    "change with skin tone, hand size, handedness, camera quality, background, lighting, "
    "motion blur, occlusion, and unfamiliar poses. The application processes a gesture "
    "image and does not identify the person. Stress-test outputs are explicitly labeled."
)