# Import required libraries
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from gtts import gTTS
import os
import pygame

# Dataset URL
dataset_url = 'https://raw.githubusercontent.com/MinorvaFalk/KNN_Alphabet/main/Dataset/hand_dataset_1000_24.csv'

# Load dataset
dataset = pd.read_csv(dataset_url)

# Define features and labels
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

# Standardize dataset
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN classifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate classifier
print(classification_report(y_test, y_pred))
print('Model accuracy:', accuracy_score(y_test, y_pred) * 100)

# Error for different K values
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# Plot error rate for different K values
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate for K Values')
plt.xlabel('K Value')
plt.ylabel('Error')
plt.show()

# Initialize mediapipe and pygame
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
pygame.mixer.init()
gesture_count = 0
output_dir = "audio_outputs"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Start gesture recognition with webcam
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    cap = cv2.VideoCapture(0)
    
    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and process image
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = hand_landmarks.landmark
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                coords = list(np.array([[landmark.x, landmark.y] for landmark in coords]).flatten())
                coords = scaler.transform([coords])

                # Predict gesture
                predicted = classifier.predict(coords)
                
                # Convert predicted class to speech
                tts = gTTS(text=str(predicted[0]), lang='en')
                output_file_path = os.path.join(output_dir, f"output_{gesture_count}.mp3")
                tts.save(output_file_path)

                # Load and play the audio file
                pygame.mixer.music.load(output_file_path)
                pygame.mixer.music.play()

                # Display prediction
                cv2.rectangle(image, (0, 0), (100, 60), (245, 90, 16), -1)
                cv2.putText(image, 'Gesture:', (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(predicted[0]), (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Increment gesture count
                gesture_count += 1

        cv2.imshow('Sign Language Recognition', image)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
