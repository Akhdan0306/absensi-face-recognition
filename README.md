📌 README.md
# Absensi Face Recognition

Sistem absensi berbasis **Face Recognition + YOLOv8** dengan GUI menggunakan **PyQt5**.  
Program ini memungkinkan pengguna untuk:
- Merekam wajah peserta
- Melakukan absensi otomatis
- Menyimpan hasil absensi ke file Excel (`attendance.xlsx`) dengan status warna:
  - **Hijau** = Detected
  - **Kuning** = Late
  - **Merah** = Not Detected
- Menghapus data wajah tertentu

---

## 🚀 Instalasi

1. Clone repository:
   ```bash
   git clone https://github.com/Akhdan0306/absensi-face-recognition.git
   cd absensi-face-recognition


Install dependencies:

pip install -r requirements.txt


Pastikan file model YOLO (yolov8l.pt) sudah ada di folder project.
Jika belum, unduh dari:
YOLOv8 Releases

▶️ Menjalankan Program

Jalankan file utama:

python terbaru.py

📂 Struktur Project
.
├── ambil-data-wajah.py      # Script ambil data wajah
├── data-wajah-csv.py        # Script pengolahan data wajah ke CSV
├── pelatihan.py             # Training (jika ada)
├── terbaru.py               # Main Program (GUI Absensi)
├── faces.csv                # Dataset wajah (otomatis terisi)
├── attendance.xlsx          # File absensi
├── yolov8l.pt               # Model YOLOv8
├── requirements.txt
└── README.md

✨ Fitur Utama

Absensi otomatis dengan kamera

Status kehadiran berdasarkan waktu

Penyimpanan ke Excel dengan pewarnaan status

Data wajah dapat ditambah atau dihapus langsung lewat GUI
