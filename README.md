# Hand_Gestures
This project investigates supervised classification of five static hand gestures: fist, open palm, peace, point up, and thumbs up. I collected the images using a mobile phone in several everyday, then compared a small convolutional neural network trained from scratch with transfer learning using MobileNetV2.  
The objective is to develop and evaluate a complete image-classification workflow using self-collected data, including preprocessing, augmentation, training, and error analysis. 
The available evidence shows poor generalization in this experiment. The baseline predicts only point up on its test set, while MobileNetV2 achieves much higher training accuracy than validation accuracy. These results motivate a careful discussion of participant diversity and collection conditions. The system recognizes predefined static gestures; it does not translate sign language or process gesture sequences. 
Dataset collection 
I collected the images using a mobile phone in several everyday settings: on a bus, at home, outdoors, and at university. Some photographs were taken in low light. I varied viewing directions and included bent or differently oriented hand poses. This collection introduces variation in background, illumination, perspective, and hand appearance. Per-image condition annotations were not available for this report, so the individual effects of these conditions cannot be measured here. 
The dataset is organized into five gesture folders, each containing person_001, person_002, and person_003. Each identifier represents the same participant across classes. Folder names provide the class labels. I collected approximately 100 or more images per class, but the exact usable totals after inspection need to be verified from the notebook. Augmented views do not count as additional self-collected photographs.
Subset	Participant	Purpose
Training	person_001	Learn model parameters
Validation	person_002	Select checkpoints and model
Test	person_003	Evaluate fixed models
The split separates participants rather than enforcing a 70/15/15 ratio. This tests transfer to a held-out person, but training on only one participant severely limits the diversity available to the models.
Preprocessing and augmentation 
The notebook checks whether image files can be decoded, corrects EXIF orientation, and converts images to RGB. SHA-256 hashes of decoded image dimensions and pixels identify exact duplicates. Duplicates with inconsistent class labels trigger manual review. This procedure does not detect all near-duplicates, such as slightly changed frames or recompressed copies. 
Working copies are resized and padded to 160 × 160 pixels while preserving aspect ratio. Original photographs remain unchanged. This reduces computation, although small hands in wide photographs may lose finger detail after resizing. No automatic hand detector or segmentation model is used. Errors and excluded files are recorded in data_issues.csv; exact exclusion counts were not available for this draft. 
Training images are shuffled and processed in batches of 16. On-the-fly augmentation includes rotations with factor 0.04, translations of up to 0.06 of the image dimensions, and zoom with factor 0.10. Random transformations are applied only during training. Horizontal flipping is omitted to avoid changing potentially meaningful gesture direction. The notebook provides a preview for checking that transformations preserve class identity. 
4 Models and training procedure 
The baseline CNN contains three convolutional layers with 32, 64, and 128 filters, respectively. Each uses a 3 × 3 kernel, ReLU activation, and max pooling. Global average pooling is followed by a 64-unit dense layer, dropout of 0.4, and a five-class softmax output. Input pixels are scaled to [0, 1]. All 101,829 parameters are trainable, and the network starts without pretrained weights. 
The second model uses MobileNetV2 with ImageNet weights and removes its original classification head [1]. The backbone is frozen and called with training=False. Global average pooling, dropout of 0.3, and a new five-class softmax layer form the classification head. Pixels are scaled to [-1, 1]. Only the new head is trained; this experiment does not fine-tune the backbone.
Setting	Both models
Optimizer	Adam; initial learning rate 0.001
Loss	Sparse categorical cross-entropy
Maximum epochs	25
Batch size	16
Checkpoint criterion	Lowest validation loss
Early stopping	Patience 5; restore best weights
Learning rate reduction	Factor 0.5; patience 2; minimum 0.000001
Random seed	42

Results from the available evidence 
The learning-curve screenshot shows seven baseline epochs and six MobileNetV2 epochs. Values read from curves below are approximate and describe epoch logs, not necessarily the restored checkpoint scores. The plots use zero-based epoch indices.
 
Figure 1. User-supplied learning curves for the baseline CNN and MobileNetV2. Values are interpreted visually; underlying history CSV files were not available. 
Baseline training accuracy rises from approximately 24% to 32%, while validation accuracy peaks near 18% and ends near 13%. Training loss decreases, but validation loss remains high and rises after its early minimum. Because training accuracy is also low, the result cannot be explained solely as successful learning followed by overfitting. 
MobileNetV2 training accuracy rises from approximately 43% to 94%, while validation accuracy peaks near 35–36% and ends near 33%. Training loss falls substantially, but validation loss remains around 1.8–2.1. This large gap is consistent with overfitting and/or differences between training and validation participants and collection conditions. The curves alone cannot isolate the cause. 
MobileNetV2 achieves a higher peak validation accuracy than the baseline in the supplied plots. This does not establish superior test performance or identify the winner under the notebook’s validation macro F1 selection rule. 

Baseline test evaluation and discussion 
The supplied test confusion matrix contains 101 images. Every prediction is point up: the class supports are 5 fist, 26 open palm, 35 peace, 9 point up, and 26 thumbs up. Thus, the model produces nine correct predictions and fails to discriminate the five classes on this test set.

Table values for the baseline are calculated from the screenshot counts using zero precision for classes with no predictions, matching zero_division=0 in the notebook. The 20% macro recall arises because point up has recall 1 and all other classes have recall 0; it does not indicate useful recognition across classes. The test set is imbalanced, with only five fist images, making per-class conclusions especially uncertain. 
Limitations and next steps 
Only one participant is present in each subset. Participant differences may be entangled with backgrounds, lighting, pose, or camera distance. Collection in varied settings is useful, but variation must also be represented within training rather than primarily between subsets. Neither the screenshots nor the code establish which factor caused the failures. 
The next priority is to inspect labels and hand visibility, collect additional training participants and independent sessions, and balance collection conditions across gestures. A consistent hand-region crop could be investigated if hands occupy little of the frame. These are proposed improvements, not completed experiments. Since test results have already been inspected, later improvements should be evaluated with a fresh held-out dataset or a clearly disclosed revised evaluation protocol. 
Conclusion 
The implemented workflow covers dataset preparation, augmentation, a baseline CNN, transfer learning, and evaluation. Available results show weak cross-participant recognition: the baseline reaches 8.91% test accuracy by always predicting point up, and MobileNetV2 shows a substantial training–validation gap. No MobileNetV2 test claim can be made without its results. The present evidence supports a limited exploratory experiment rather than a reliable gesture-recognition system. 
References
TensorFlow. Transfer learning and fine-tuning. https://www.tensorflow.org/tutorials/images/transfer_learning 
Scikit-learn. classification_report. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
