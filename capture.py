import cv2
import os
import time

# Ask for the person's name
person_name = input("Enter the name of the person: ").strip()

# Create a directory to save captured images if it doesn't exist
output_dir = os.path.join("D:/utsav1/Multiple/captured_images", person_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Capturing 50 images for {person_name}. Please stay in front of the camera...")

image_count = 0  # Counter for captured images

while image_count < 50:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from the webcam.")
        break

    # Display the frame in a window
    cv2.imshow("Webcam", frame)

    # Generate a unique filename
    image_name = f"{person_name}_image_{image_count + 1}.jpg"
    image_path = os.path.join(output_dir, image_name)

    try:
        # Save the captured image
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")
        image_count += 1

        # Wait for a short duration (e.g., 0.5 seconds) between captures
        time.sleep(0.5)

    except Exception as e:
        print(f"Error saving image: {e}")
        break

    # Break the loop if 'q' is pressed (optional, for manual exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting early...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Successfully captured {image_count} images for {person_name}.")