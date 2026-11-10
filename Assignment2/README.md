# GestureVision — Assignment 2

Continuation of the A1 five-class hand-gesture recognition project.

## Goal

Recognize five static gestures: `fist`, `open_palm`, `peace`, `point_up`, and `thumbs_up`.

This is a static gesture classifier, not a full sign-language translator.

## A2 additions

- baseline CNN comparison;
- MobileNetV2 transfer learning;
- frozen-backbone vs fine-tuning ablation;
- MediaPipe Hand Landmarker;
- custom RBF SVM on landmark geometry;
- optional score-level fusion;
- Grad-CAM explainability;
- 21-point hand skeleton visualization;
- latency measurement;
- low-light, blur, contrast and rotation stress tests;
- Streamlit camera/upload demo.

## Current held-out-person results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline CNN | 0.3069 | 0.2522 |
| MobileNetV2 frozen | 0.5743 | 0.4889 |
| MobileNetV2 fine-tuned | **0.5941** | **0.5118** |
| Off-the-shelf MediaPipe gesture head | 0.0198 | 0.0286 |

The main result is the generalization gap: validation on familiar participants was much easier than testing on the completely unseen `person_003`.

## Run the notebook

1. Put the dataset at `MyDrive/CV_project/hand_gestures.zip`.
2. Open `notebooks/GestureVision_Assignment2_SOTA.ipynb` in Google Colab.
3. Enable a GPU runtime.
4. Run all cells.
5. Copy the exported model files into `models/`.

Expected exported files:
- `gesture_model.keras`
- `class_names.json`
- `landmark_svm.joblib`
- `hand_landmarker.task`
- `ensemble_config.json`

## Run the app on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Open `http://localhost:8501`.

## Privacy

The raw participant photos are intentionally not committed to this public repository. They should only be shared with participant consent.

## Synthetic data

Generative synthetic training share: **0%**. The project uses conventional training augmentation and deterministic stress-test transformations, not Step-Video-T2V or ArtAug-generated training samples.
