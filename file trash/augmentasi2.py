import cv2
import pandas as pd
import os
import numpy as np
import random

# Load data wajah
face_data = pd.read_csv("faces.csv")
output_dir = "augmented_faces"
os.makedirs(output_dir, exist_ok=True)

def augment_image(image):
    augmented_images = []

    # Flip Horizontal
    flip = cv2.flip(image, 1)
    augmented_images.append(flip)

    # Brightness Adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    augmented_images.append(bright)

    # Rotation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle=15, scale=1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented_images.append(rotated)

    return augmented_images

new_rows = []

# Loop semua data
for idx, row in face_data.iterrows():
    image_path = row['ImagePath']
    name = row['Name']
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {image_path}")
        continue

    augmented_images = augment_image(image)
    for i, aug_img in enumerate(augmented_images):
        new_image_path = os.path.join(output_dir, f"{name}_{idx}_{i}.jpg")
        cv2.imwrite(new_image_path, aug_img)
        new_rows.append({"Name": name, "ImagePath": new_image_path})

# Save hasil augmentasi ke CSV baru
augmented_df = pd.DataFrame(new_rows)
augmented_df.to_csv("augmented_faces.csv", index=False)
print("Augmentasi selesai dan disimpan di 'augmented_faces.csv'")
