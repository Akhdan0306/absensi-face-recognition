import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load data (misalnya dari csv)
data = pd.read_csv('augmented_faces.csv')

images = []
labels = []

# Baca gambar
for index, row in data.iterrows():
    img = cv2.imread(row['ImagePath'])
    img = cv2.resize(img, (100, 100))  # Sesuaikan ukuran gambar
    images.append(img)
    labels.append(row['Name'])

# Konversi ke numpy array
X = np.array(images) / 255.0  # Normalisasi
y = np.array(labels)

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)

# Buat model CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')  # jumlah kelas sesuai data
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Simpan model
model.save('model.h5')
