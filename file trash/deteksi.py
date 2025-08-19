import cv2
import numpy as np
import json
from keras.models import load_model

# Load model
model = load_model('model_wajah.h5')

# Load label mapping dari file
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing (samakan dengan model training)
    face = cv2.resize(frame, (100, 100))  # Sesuaikan dengan input model
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    # Prediksi
    pred = model.predict(face)
    label = label_mapping[str(np.argmax(pred))]  # Gunakan str karena JSON key adalah string

    # Tampilkan hasil
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Deteksi Wajah', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
