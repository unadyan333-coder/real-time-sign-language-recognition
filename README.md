# Real-Time Hand Gesture Recognition

A real-time computer vision system that recognizes predefined hand gestures using a standard webcam and converts them into live text.

The system uses **MediaPipe** for hand landmark detection and a **Random Forest** machine learning classifier trained on a custom hand-gesture dataset.

## Project Overview

This project implements a complete pipeline for real-time hand gesture recognition:

1. **Data Collection** — capture labeled hand-landmark data from a webcam.
2. **Model Training** — train a Random Forest classifier using the collected landmark data.
3. **Real-Time Prediction** — detect the hand through the webcam, extract its landmarks, and predict the corresponding gesture in real time.

The current version supports **one hand** and focuses on recognizing **static hand gestures**. Each recognized gesture is displayed as live text on the video feed.

## How It Works

The system uses MediaPipe to detect the hand and extract its **21 landmarks**. Each landmark contains three coordinates (`x`, `y`, `z`), resulting in **63 numerical features** for each sample.

These features are then used to train a **Random Forest Classifier**, which learns the relationship between hand landmark patterns and their corresponding gesture labels.

During real-time prediction, the same landmark extraction process is applied to the live camera feed, and the trained classifier predicts the gesture shown by the user.

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
(x, y, z) → 63 Features
   ↓
Random Forest Classifier
   ↓
Predicted Gesture
   ↓

Live Text Overlay
```

## Project Pipeline

### 1. Data Collection

The `data_collection.py` script captures hand landmarks from the webcam and stores them as labeled samples in a CSV dataset.

For each captured sample:

* MediaPipe detects the hand.
* 21 hand landmarks are extracted.
* The `x`, `y`, and `z` coordinates are collected.
* The sample is assigned to its corresponding gesture label.
* The data is saved to a CSV file for model training.

The collection process can be started and controlled using keyboard commands.

### 2. Model Training

The `train_model.py` script loads the collected dataset and trains a **Random Forest Classifier**.

The dataset is divided into training and testing data, allowing the trained model to be evaluated on samples that were not used during training.

After training, the model is saved and can be used for real-time prediction.

### 3. Real-Time Prediction

The `predict.py` script opens the webcam and continuously processes the incoming frames.

For each frame:

* MediaPipe detects the hand.
* The 21 landmarks are extracted.
* The landmarks are converted into the same 63-feature format used during training.
* The trained Random Forest model predicts the gesture.
* The predicted label is displayed as live text on the screen.

This allows the system to recognize trained gestures directly from the webcam in real time.

## Demo

The system recognizes trained hand gestures in real time and displays the predicted gesture as live text on the camera feed.

[Real-Time Gesture Recognition](assets/demo.gif)

## Data Collection

The dataset is collected directly from the webcam using MediaPipe hand landmarks. Each captured sample is labeled according to its corresponding gesture.

During collection, the system extracts the 21 hand landmarks and stores their coordinates as labeled samples in a CSV dataset.

## Hand Landmark Detection

MediaPipe tracks **21 landmarks** on the detected hand. These landmarks provide the numerical features used by the machine learning model.

![Hand Landmarks](assets/hand_landmarks.jpg)

## Model Training Results

The Random Forest classifier was trained using the collected hand-landmark dataset and evaluated on a held-out test set.

The training process achieved a test accuracy of **XX%**.

![Model Training Results](assets/training_result.jpg)



## Tech Stack

* **Python** — main programming language
* **OpenCV** — webcam access and real-time video processing
* **MediaPipe** — hand detection and 21-point landmark extraction
* **scikit-learn** — Random Forest machine learning classifier
* **pandas** — dataset handling
* **joblib** — saving and loading the trained model

## Project Structure

```text
.
├── data_collection.py       # Collect labeled hand-landmark data
├── train_model.py           # Train and evaluate the Random Forest model
├── predict.py               # Real-time gesture prediction
├── hand_gestures_data.csv   # Generated dataset
└── assets/                  # Images and demo media
```

## Getting Started

### Requirements

Install the required Python libraries:

```bash
pip install opencv-python mediapipe pandas scikit-learn joblib
```

### 1. Collect Data

Run the data collection script:

```bash
python data_collection.py
```

Collect samples for each gesture you want the model to recognize.

### 2. Train the Model

After collecting the dataset, run:

```bash
python train_model.py
```

The script trains the Random Forest classifier and saves the trained model.

### 3. Run Real-Time Recognition

Run:

```bash
python predict.py
```

Show a supported gesture to the webcam. The predicted gesture will appear as live text on the screen.

## Current Limitations

The current version:

* Supports one hand at a time.
* Focuses on static hand gestures.
* Recognizes only gestures included in the training dataset.
* Performance can vary depending on lighting, camera angle, hand distance, and the quality and diversity of the collected data.

## Future Improvements

* [ ] Two-hand gesture recognition
* [ ] Dynamic gesture recognition based on hand movement over multiple frames
* [ ] Text-to-speech output
* [ ] Expanded gesture vocabulary
* [ ] Improved robustness across different environments and lighting conditions
