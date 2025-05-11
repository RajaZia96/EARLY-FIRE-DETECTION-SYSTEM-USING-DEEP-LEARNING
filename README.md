# EARLY-FIRE-DETECTION-SYSTEM-USING-DEEP-LEARNING
This project presents a deep learning-based method for detecting fire and smoke in surveillance videos using a combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. It was developed as part of the ONFIRE 2023 contest and tested on a video dataset containing annotated fire/smoke and no-fire scenarios.

📌 Objective
To build a system that can accurately identify the presence of fire or smoke in real-time using frames from fixed CCTV camera videos.

🧠 Methodology
1. Model Architecture
CNN (EfficientNet_b1): Used as a spatial feature extractor on individual video frames.

LSTM Layer: Captures temporal dependencies across 5 consecutive frames.

Fully Connected Layer: Applies sigmoid activation for binary classification (fire/smoke vs. no fire).

2. Training Workflow
Step 1: Fine-tune the last layer of a pre-trained EfficientNet_b1 to classify single frames.

Step 2: Use the CNN as a feature extractor by outputting 300-dim feature vectors.

Step 3: Feed sequences of 5-frame features into the LSTM to learn temporal patterns.

Step 4: Final prediction based on sigmoid output and thresholding.

🎯 Dataset
312 total videos:

218 with fire/smoke

94 without fire

Videos labeled with annotations for fire/smoke start time and type (smoke, fire, both).

Frame extraction using OpenCV + ffmpeg at 1 FPS.

Frames resized to 224×224 and normalized.

🧪 Preprocessing
Frame normalization using ImageNet mean and std.

Data augmentation using horizontal flips (50% chance).

Separate DataLoaders for CNN and LSTM training:

CNN: 1 random frame/video

LSTM: 5 consecutive frames/video

⚙️ Training Details
Optimizer: SGD with momentum 0.9

Loss Function: Binary Cross Entropy

Activation: Sigmoid

Early stopping: 5 epochs

Learning rates:

CNN Initial Fine-tune: 0.05 (10 epochs)

CNN Full Fine-tune: 0.0001 (200 epochs)

CNN+LSTM Joint Training: 0.001 (200 epochs)

📊 Results
Achieved ~71% accuracy and 0.54 loss on validation set before overfitting.

EfficientNet_b1 outperformed other backbones (ResNet50, ResNet101, GoogleNet).

Robust performance in detecting fire/smoke with limited data and compute.
