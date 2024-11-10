# Sign Language Recognition with KNN

This project is a real-time sign language recognition system that uses hand gesture data captured through a webcam. The system leverages a K-Nearest Neighbors (KNN) classifier to recognize hand gestures and provides audio output corresponding to each recognized gesture.

## Project Overview

This project captures hand gestures using the MediaPipe library and classifies them into sign language alphabets using a KNN model trained on a dataset of hand landmarks. The recognized gesture is then converted to speech using the Google Text-to-Speech (gTTS) library.

### Features

- **Real-time Gesture Detection**: Uses OpenCV and MediaPipe to capture and process hand gestures.
- **Gesture Classification**: Classifies gestures into sign language alphabets with a KNN model.
- **Audio Feedback**: Converts the recognized gesture into speech audio output using gTTS.
- **Dynamic KNN Error Visualization**: Plots the error rates for different K values to aid in model tuning.

## Dataset

The dataset used for training consists of hand landmarks for various gestures and is available at:
[Dataset Link](https://raw.githubusercontent.com/MinorvaFalk/KNN_Alphabet/main/Dataset/hand_dataset_1000_24.csv)

## Requirements

- **Python** 3.x
- **OpenCV** for real-time video capture
- **MediaPipe** for hand landmark detection
- **Pandas** and **NumPy** for data manipulation
- **scikit-learn** for the KNN model
- **gTTS** for text-to-speech conversion
- **pygame** for audio playback

### Installation

Install the required libraries with:
```bash
pip install opencv-python-headless mediapipe pandas numpy scikit-learn gTTS pygame matplotlib

## Run the project 
`sign.py`
