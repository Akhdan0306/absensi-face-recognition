import cv2
import os

# Tentukan folder untuk menyimpan dataset wajah
base_folder = 'C:/Users/akhda/Documents/absensi/images'
if not os.path.exists(base_folder):
    os.makedirs(base_folder)  # Gunakan makedirs agar bisa membuat folder bersarang jika belum ada

# Inisialisasi kamera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Gagal membuka kamera. Pastikan kamera terhubung.")
    exit()

# Path ke file Haarcascade untuk deteksi wajah
cascade_path = 'C:/Users/akhda/Documents/absensi/models/haarcascade_frontalface_default.xml'

# Pastikan file Haarcascade ada sebelum digunakan
if not os.path.exists(cascade_path):
    print(f"Error: File Haarcascade tidak ditemukan di {cascade_path}")
    exit()

# Load model Haarcascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cascade_path)

# Cek apakah model Haarcascade berhasil dimuat
if face_cascade.empty():
    print("Error: Gagal memuat model Haarcascade.")
    exit()

# Input nama pengguna untuk dataset
user_name = input("Masukkan nama: ").strip()
user_folder = os.path.join(base_folder, user_name)

# Buat folder khusus untuk user jika belum ada
os.makedirs(user_folder, exist_ok=True)

# Variabel kontrol pengambilan gambar
img_id = 0
max_images = 50  # Jumlah maksimal gambar per user
fixed_size = (224, 224)  # Ukuran tetap untuk semua gambar wajah

print(f"Mengambil hingga {max_images} gambar untuk {user_name}. Tekan 'q' untuk keluar.")

while img_id < max_images:
    ret, frame = cam.read()
    if not ret:
        print("Gagal membaca frame dari kamera. Menghentikan proses.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        img_id += 1
        face = frame[y:y+h, x:x+w]

        # Ubah ukuran citra wajah ke ukuran tetap
        resized_face = cv2.resize(face, fixed_size)
        file_name_path = os.path.join(user_folder, f"{user_name}_{img_id}.jpg")
        cv2.imwrite(file_name_path, resized_face)

        # Tampilkan kotak deteksi wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        print(f"Gambar {img_id} disimpan di {file_name_path}")

        # Hentikan loop setelah mencapai jumlah maksimal gambar
        if img_id >= max_images:
            break

    # Tampilkan jendela dengan hasil deteksi wajah
    cv2.imshow('Face Capture', frame)

    # Hentikan dengan menekan 'q' atau setelah mencapai jumlah maksimal gambar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela OpenCV
cam.release()
cv2.destroyAllWindows()

print(f"Proses selesai. Total {img_id} gambar disimpan di folder: {user_folder}")
