import os
import json

dataset_path = 'C:/Users/akhda/Documents/absensi/faces_dataset/'

# Buat label mapping otomatis
label_mapping = {idx: folder for idx, folder in enumerate(os.listdir(dataset_path))}
print(label_mapping)  # Contoh output: {0: 'Akhdan', 1: 'Firman', 2: 'Havid', 3: 'Samudra'}

# Simpan ke file JSON biar bisa dipakai saat prediksi
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)
