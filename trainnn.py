import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the dataset
dataset_path = "D:/utsav1/face_detection/face_dataset.csv"
df = pd.read_csv(dataset_path)

# Initialize lists for embeddings, names, ages, genders, and Aadhar numbers
X_embeddings = []
y_names = []
y_ages = []
y_genders = []
y_aadhar_numbers = []

# Load a pre-trained face embedding model (e.g., FaceNet or OpenCV DNN)
face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Loop through each row in the dataset
for _, row in df.iterrows():
    name = row["Name"]
    age = row["Age"]
    gender = row["Gender"]
    aadhar_number = row["Aadhar number"]
    image_path = row["Image Path"]

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}. Skipping...")
        continue

    # Detect face
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Extract the first detected face
    if detections.shape[2] == 0:
        print(f"No face detected in {image_path}. Skipping...")
        continue

    confidence = detections[0, 0, 0, 2]
    if confidence < 0.5:
        print(f"Low confidence ({confidence}) for {image_path}. Skipping...")
        continue

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    face = image[startY:endY, startX:endX]

    # Resize the face to a fixed size (e.g., 96x96)
    face_resized = cv2.resize(face, (96, 96))

    # Convert the face to grayscale (optional)
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

    # Flatten the face into a 1D array (simple embedding)
    embedding = face_gray.flatten()

    # Append data
    X_embeddings.append(embedding)
    y_names.append(name)
    y_ages.append(age)
    y_genders.append(gender)
    y_aadhar_numbers.append(aadhar_number)

# Convert lists to NumPy arrays
X_embeddings = np.array(X_embeddings)
y_names = np.array(y_names)
y_ages = np.array(y_ages)
y_genders = np.array(y_genders)
y_aadhar_numbers = np.array(y_aadhar_numbers)

# Split the data into training and testing sets
X_train, X_test, y_train_names, y_test_names = train_test_split(
    X_embeddings, y_names, test_size=0.2, random_state=42
)

# Train the name classifier
name_classifier = SVC(kernel="linear", probability=True)
name_classifier.fit(X_train, y_train_names)

# Evaluate the name classifier
y_pred_names = name_classifier.predict(X_test)
accuracy = accuracy_score(y_test_names, y_pred_names)
print(f"Name Prediction Accuracy: {accuracy * 100:.2f}%")

# Split the data into training and testing sets
X_train, X_test, y_train_ages, y_test_ages = train_test_split(
    X_embeddings, y_ages, test_size=0.2, random_state=42
)

# Define the age regressor
age_regressor = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(1, activation="linear")  # Output layer for age prediction
])
age_regressor.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

# Train the age regressor
age_regressor.fit(X_train, y_train_ages, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the age regressor
y_pred_ages = age_regressor.predict(X_test).flatten()
mae = np.mean(np.abs(y_pred_ages - y_test_ages))
print(f"Age Prediction MAE: {mae:.2f} years")

# Encode gender labels (Male: 0, Female: 1)
gender_mapping = {"Male": 0, "Female": 1}
y_genders_encoded = np.array([gender_mapping.get(g, -1) for g in y_genders])  # Handle invalid values

# Split the data into training and testing sets
X_train, X_test, y_train_genders, y_test_genders = train_test_split(
    X_embeddings, y_genders_encoded, test_size=0.2, random_state=42
)

# Train the gender classifier
gender_classifier = SVC(kernel="linear", probability=True)
gender_classifier.fit(X_train, y_train_genders)

# Evaluate the gender classifier
y_pred_genders = gender_classifier.predict(X_test)
accuracy = accuracy_score(y_test_genders, y_pred_genders)
print(f"Gender Prediction Accuracy: {accuracy * 100:.2f}%")

import joblib

# Save the name classifier
joblib.dump(name_classifier, "D:/utsav1/face_detection/model/name_classifier.pkl")

# Save the age regressor
age_regressor.save("D:/utsav1/face_detection/model/age_regressor.h5")

# Save the gender classifier
joblib.dump(gender_classifier, "D:/utsav1/face_detection/model/gender_classifier.pkl")