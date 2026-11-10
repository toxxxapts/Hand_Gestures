# Defense Notes and Likely Questions

## One-sentence project description

I built an explainable five-class hand-gesture recognizer that compares a CNN baseline, MobileNetV2 transfer learning, and MediaPipe landmark geometry on an unseen participant, then deploys the result in a Streamlit camera/upload application.

## Why is validation 100% but test only 59.41%?

Validation images came from the same two participants and similar collection environments as the training data. The test set contains a third participant who was never used during model selection. The difference is evidence of domain shift, not a calculation error.

## Why use macro-F1?

The held-out test distribution is uneven: fist has 5 images, while peace has 35. Accuracy can be dominated by larger classes. Macro-F1 gives equal importance to every class.

## Why did the generic MediaPipe Gesture Recognizer fail?

It was trained for its own gesture distribution and returned `unknown` for 99 of 101 held-out images. The Hand Landmarker itself remains useful, so I use its 21 landmarks and train a classifier on my own labels.

## Why SVM for landmarks?

The landmark dataset is small and tabular: each image becomes a compact vector of normalized coordinates, joint angles, and fingertip distances. An RBF SVM is suitable for a small nonlinear classification problem and is faster to train than another deep network.

## What is the ablation?

The primary ablation compares a frozen MobileNetV2 backbone with fine-tuning the final 40 layers. Accuracy increased by 1.98 percentage points and macro-F1 by 2.29 points on the held-out participant, with about 3.7 ms extra average latency in the recorded run.

The final notebook also compares coordinates-only landmarks against coordinates plus angles/distances, and selects the image/geometry fusion weight on validation data.

## What does Grad-CAM show?

It computes the gradient of the selected class score with respect to a spatial feature map, then weights and combines the feature channels. The heatmap indicates influential regions, but it is not a proof of causal reasoning.

## Why not call this sign-language recognition?

The model recognizes only five isolated static poses. Real sign language includes a much larger vocabulary, motion, timing, two-hand interactions, facial expressions, body pose, grammar, and sentence context.

## Main limitations

- only three participants;
- uneven held-out class counts;
- different backgrounds are correlated with some participants and classes;
- static images only;
- no explicit unknown-class training;
- confidence may be poorly calibrated;
- camera and lighting conditions are limited.

## Best next experiment

Collect more participants across multiple sessions and backgrounds, then use group-based cross-validation. For a future sign-language project, collect short labeled sequences and use hand, face, and pose landmarks with a temporal model such as a Transformer or TCN.