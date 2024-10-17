# Hand Gesture Recognition System

This repository contains a project to build a **Hand Gesture Recognition** model using Python, TensorFlow/Keras, and computer vision libraries such as **OpenCV** and **imageio**. The model is trained to classify different hand gestures from image data, which can be used for gesture-based control systems.

---

## Table of Contents
- [Overview](#overview)
- [Project Setup](#project-setup)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Installation](#installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project focuses on recognizing hand gestures using a **Convolutional Neural Network (CNN)** model. The gestures are captured from image data using a webcam or from pre-recorded video files. The dataset used for training consists of infrared images, and the model is designed to classify gestures into multiple categories (e.g., "palm", "fist", "L-sign", etc.).

The **main stages** of this project include:
1. **Data Collection/Preprocessing**: Using a webcam or a dataset of images to train the model.
2. **Model Building**: Building a CNN using TensorFlow/Keras.
3. **Training**: Training the model on gesture data.
4. **Real-time Inference**: Using the trained model to predict gestures from live video.

---

## Project Setup

### Requirements

The project requires the following Python libraries:
- `TensorFlow` for building the CNN model.
- `imageio` for webcam and video capture.
- `OpenCV` for image processing and frame manipulation.
- `Pillow` for image resizing and manipulation.
- `numpy` for array operations.
  
### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
Create and Activate a Virtual Environment
```
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
2 **Install Required Dependencies**

```bash
pip install -r requirements.txt
```
# Install OpenCV (if needed)

```bash
pip install opencv-python
```
 # Dataset
The project uses the LeapGestRecog dataset, which contains near-infrared images of hand gestures captured by the Leap Motion sensor. The dataset can be downloaded from Kaggle.

#Dataset Structure
The dataset is organized into folders, each representing a subject, with subfolders for different hand gestures:

```bash
/00 (subject 00)
   /01_palm
       frame_01.png
       frame_02.png
   /02_l
       frame_03.png
   ...
/01 (subject 01)
   /01_palm
   /02_l
   ...
/09 (subject 09)
```
# Model Architecture
The hand gesture recognition model is based on a Convolutional Neural Network (CNN), which is effective for image classification tasks. The architecture consists of several convolutional layers followed by fully connected layers.

CNN Model Summary:
Input: 128x128 grayscale images.
Layers:
3 Convolutional Layers (32, 64, 128 filters)
Max Pooling layers after each convolutional layer.
Fully connected Dense layers.
Dropout layer for regularization.
Softmax output layer for multi-class classification.
```bash
python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))  # Adjust based on the number of gesture classes

    return model
```
# Training Process
Once the model is built, we compile it using the Adam optimizer and the categorical crossentropy loss function, as it's a multi-class classification problem.

Model Compilation:

```bash
python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
Training Example:
```bash
python
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
The model is trained on the dataset using 70% of the data for training and 30% for validation.
```
## Usage
Run the Training Script

```bash
python train_model.py
```
Run the Real-time Inference (Optional) After training, you can use the model for real-time gesture recognition using a webcam or a video file.

Example:

```bash
python realtime_inference.py
Future Enhancements
Real-time Gesture Recognition: Integrate the trained model with a real-time webcam feed using imageio or OpenCV.
Improve Model Accuracy: Explore advanced architectures like ResNet or MobileNet for better accuracy.
Add More Gesture Classes: Expand the dataset to include more complex gestures.
Contributing
Contributions are welcome! If you'd like to contribute to this project, feel free to fork the repository and submit a pull request.
```
# License
This project is licensed under the MIT License - see the LICENSE file for details.

### Notes:
- Update the `git clone` URL in the **Installation** section with your actual GitHub repository link.
- If you have a `requirements.txt` file, make sure it includes all the necessary Python dependencies (e.g., TensorFlow, OpenCV, imageio, etc.).
- Make sure you provide the actual `LICENSE` file in the repository if you are referencing one.
- Adjust the number of gesture classes in the model if it differs from the example (5 classes).

This `README.md` is structured to help users understand your project, install dependencies
