import os
import pandas as pd

# Step 1: Define the dataset directory
dataset_path = "D:/utsav1/face_detection/dataset"

# Step 2: Initialize an empty list to store metadata
metadata = []

# Step 3: Loop through each person's folder
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    # Skip if it's not a directory
    if not os.path.isdir(person_folder):
        continue
    
    # Prompt the user for age and gender
    print(f"Enter details for {person_name}:")
    age = input(f"Age of {person_name}: ")
    gender = input(f"Gender of {person_name} (Male/Female/Other): ").strip().capitalize()
    aadhar_number = input(f"Aadhar number of {person_name}:")
    
    # Loop through all images in the person's folder
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        
        # Append metadata for the current image
        metadata.append({
            "Name": person_name,
            "Image Path": image_path,
            "Age": age,
            "Gender": gender,
            "Aadhar number":aadhar_number
        })

# Step 4: Convert the metadata list to a DataFrame
df = pd.DataFrame(metadata)

# Step 5: Save the dataset to a CSV file
csv_file = "face_dataset.csv"
df.to_csv(csv_file, index=False)

print(f"Dataset saved successfully to {csv_file}!")