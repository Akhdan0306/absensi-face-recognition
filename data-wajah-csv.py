import os
import pandas as pd

# Path ke folder gambar wajah
faces_folder = r"C:\Users\akhda\Documents\absensi\images"

# Periksa apakah folder faces_dataset ada
if not os.path.exists(faces_folder):
    print(f"Folder {faces_folder} tidak ditemukan!")
else:
    # Ambil daftar subfolder di dalam faces_folder
    subfolders = [f.path for f in os.scandir(faces_folder) if f.is_dir()]

    # Jika tidak ada subfolder
    if not subfolders:
        print(f"Tidak ada subfolder yang ditemukan di folder {faces_folder}.")
    else:
        # Daftar nama dan path gambar
        names = []
        paths = []

        # Load CSV untuk memeriksa nama yang sudah ada
        faces_csv_path = r"C:\Users\akhda\Documents\absensi\faces.csv"
        if os.path.exists(faces_csv_path):
            faces_df = pd.read_csv(faces_csv_path)
        else:
            faces_df = pd.DataFrame(columns=["Name", "ImagePath"])

        # Fungsi untuk menghasilkan nama gambar yang unik
        def generate_image_name(base_name):
            """Generate a unique name for the image by adding a number if it exists in the CSV"""
            name_count = sum(faces_df["Name"].str.contains(base_name))
            if name_count == 0:
                return base_name  # If no name found, return the base name
            else:
                return f"{base_name}{name_count + 1}"

        # Loop untuk membaca setiap folder (subfolder)
        for subfolder in subfolders:
            # Ambil daftar gambar di setiap subfolder dengan ekstensi .jpg, .png, .jpeg
            face_images = [f for f in os.listdir(subfolder) if f.endswith(('.jpg', '.png', '.jpeg'))]

            # Jika tidak ada gambar di subfolder
            if not face_images:
                print(f"Tidak ada gambar yang ditemukan di folder {subfolder}.")
            else:
                # Loop untuk mendapatkan nama dan path gambar
                for image in face_images:
                    base_name = image.split('.')[0]  # Nama berdasarkan nama file (misalnya john_doe.jpg -> john_doe)
                    name = generate_image_name(base_name)
                    names.append(name)
                    paths.append(os.path.join(subfolder, image))

        # Buat DataFrame dan simpan ke CSV
        faces_df = pd.DataFrame({"Name": names, "ImagePath": paths})

        # Simpan DataFrame ke CSV
        try:
            faces_df.to_csv(faces_csv_path, index=False)
            print("faces.csv telah dibuat!")
        except Exception as e:
            print(f"Terjadi kesalahan saat menyimpan CSV: {e}")
