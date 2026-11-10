# Exported model files

Run the notebook and copy these generated files here:

- `gesture_model.keras`
- `class_names.json`
- `landmark_svm.joblib`
- `hand_landmarker.task`
- `ensemble_config.json`

Large trained weights are intentionally not committed automatically. The Streamlit application checks which components are available and can fall back to the image model when landmark inference is unavailable.
