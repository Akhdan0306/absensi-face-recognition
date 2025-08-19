import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Path dataset kamu
dataset_path = 'C:/Users/akhda/Documents/absensi/faces_dataset/'

# Ukuran gambar (resize semua gambar ke ukuran ini)
image_size = (100, 100)

# List untuk data dan label
data = []
labels = []

# Mapping nama folder ke label numerik
label_map = {}

# Mulai baca dataset
for idx, folder in enumerate(os.listdir(dataset_path)):
    label_map[idx] = folder  # Mapping angka ke nama orang
    folder_path = os.path.join(dataset_path, folder)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            # Baca gambar
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)  # Resize gambar
            img = img / 255.0  # Normalisasi (biar 0-1)
            data.append(img)
            labels.append(idx)  # Label numerik
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

# Ubah ke numpy array
data = np.array(data)
labels = np.array(labels)

# Bagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# One-hot encoding label
num_classes = len(label_map)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Model CNN sederhana
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Simpan model
model.save('model_wajah.h5')

print("Training selesai! Model disimpan sebagai 'model_wajah.h5'.")
print("Label mapping:", label_map)
