import cv2
import numpy as np
import joblib
from keras.models import load_model
import pandas as pd

# Load the trained models
name_classifier = joblib.load("D:/utsav1/face_detection/model/name_classifier.pkl")
gender_classifier = joblib.load("D:/utsav1/face_detection/model/gender_classifier.pkl")

# Load the main dataset
dataset_path = "D:/utsav1/face_detection/face_dataset.csv"
df = pd.read_csv(dataset_path)

# Create mappings between names and Aadhar numbers, ages, and genders
name_to_aadhar = dict(zip(df["Name"], df["Aadhar number"]))
name_to_age = dict(zip(df["Name"], df["Age"]))
name_to_gender = dict(zip(df["Name"], df["Gender"]))

# Load the pre-trained face detection model
face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Loop through all detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = frame[startY:endY, startX:endX]

        # Resize and preprocess the face
        try:
            face_resized = cv2.resize(face, (96, 96))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            embedding = face_gray.flatten().reshape(1, -1)
        except Exception as e:
            print(f"Error processing face: {e}")
            continue

        # Predict name and gender
        name_probs = name_classifier.predict_proba(embedding)[0]
        name = name_classifier.classes_[np.argmax(name_probs)]  # Get the predicted name
        confidence_score = np.max(name_probs)  # Confidence score for the predicted name

        gender_prob = gender_classifier.predict_proba(embedding)[0]
        gender = "Male" if gender_prob[0] > gender_prob[1] else "Female"

        # Check if the predicted name exists in the dataset and meets the confidence threshold
        if name in name_to_aadhar and confidence_score >= 0.6:  # Confidence threshold
            aadhar_number = name_to_aadhar[name]
            age = name_to_age[name]
            gender = name_to_gender[name]
        else:
            name = "Unknown Person"
            aadhar_number = "N/A"
            age = "N/A"
            gender = "N/A"

        # Draw bounding box around the face
        color = (0, 0, 255) if name == "Unknown Person" else (0, 255, 0)  # Red for unknown faces
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Display predictions on separate lines
        y_offset = startY - 10  # Start position for text
        line_height = 20  # Height of each line

        # Add "Unknown Person" label for unknown faces
        if name == "Unknown Person":
            cv2.putText(frame, "Unknown Person", (startX, startY - 4 * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Name: {name}", (startX, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Aadhar Number: {aadhar_number}", (startX, y_offset - line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Gender: {gender}", (startX, y_offset - 2 * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Age: {age}" if age != "N/A" else "Age: N/A", (startX, y_offset - 3 * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()