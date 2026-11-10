# 3–5 Minute Demo Script

## 0:00–0:25 — Introduction

Hello. My project is GestureVision, an explainable hand-gesture recognition system for five static classes: fist, open palm, peace, point up, and thumbs up. It continues my Assignment 1 dataset and adds transfer learning, MediaPipe hand landmarks, visual explanations, robustness tests, and a web demo. It is not a sign-language translator.

## 0:25–1:05 — Dataset and evaluation protocol

The audited dataset contains 510 usable images from three participants. I used 327 images for training, 82 for validation, and 101 images from person 003 as a completely unseen test participant. I kept the unseen participant separate because random image splitting can produce unrealistically high scores when the same hands and backgrounds appear in both training and testing.

## 1:05–1:45 — Model comparison

My A1-style CNN reached 30.69 percent accuracy and 0.2522 macro-F1 on the held-out participant. A frozen MobileNetV2 improved this to 57.43 percent accuracy and 0.4889 macro-F1. Fine-tuning the final 40 layers produced the best current image-model result: 59.41 percent accuracy and 0.5118 macro-F1.

The validation accuracy of the fine-tuned model was 100 percent, but the unseen-person result was much lower. This is not a contradiction. It shows domain shift: the validation images came from familiar participants and environments, while person 003 used different hands, orientations, lighting, and backgrounds.

## 1:45–2:25 — SOTA component

For the modern component, I use MediaPipe Hand Landmarker. It detects a palm and returns 21 three-dimensional hand landmarks. The generic MediaPipe gesture head performed poorly on my photographs and returned unknown for almost all held-out images. Instead of hiding this result, I replaced the generic head with a custom SVM trained on normalized coordinates, joint angles, and fingertip distances from my own dataset.

The final notebook also tests score-level fusion between MobileNetV2 appearance features and landmark geometry.

## 2:25–3:25 — Web application

Now I open the Streamlit application. I can upload an image or use the camera. The app shows the predicted class, confidence, inference latency, and the runtime environment.

On the left is the model input. In the center, Grad-CAM shows which image regions influenced the CNN prediction. On the right, MediaPipe shows the 21-point hand skeleton. I can switch between the image model, landmark SVM, and automatic fusion.

The stress-test menu applies low light, blur, high contrast, or rotation. Each transformed image is clearly labeled as synthetic deterministic processing, not generative AI.

## 3:25–4:15 — Error analysis and limitations

The strongest class was open palm, with high recall on the unseen participant. The most serious confusion was thumbs up versus fist: 20 of 26 thumbs-up test images were classified as fist by the fine-tuned image model. Peace was also confused with point up in 11 cases. These errors suggest that the model still depends on pose orientation, background, and participant-specific appearance.

The dataset contains only three participants, and the held-out fist class has only five examples. Therefore, macro-F1 and per-class metrics are more informative than accuracy alone.

## 4:15–4:40 — Ethics and conclusion

The repository should not expose raw participant photographs without consent. The system does not identify a person and is not intended for surveillance. It also must not be presented as a complete sign-language translator.

In conclusion, transfer learning clearly improved the A1 baseline, but the large validation-to-test gap shows that unseen-person generalization remains the main challenge. The landmark pipeline and robustness tests make that limitation visible rather than hiding it behind a single high validation score.