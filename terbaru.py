import sys
import cv2
import pandas as pd
import datetime
import face_recognition
from ultralytics import YOLO
import os
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import re
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QInputDialog, QMessageBox
from PyQt5.QtCore import QTimer, QTime
from PyQt5.QtGui import QImage, QPixmap

# Load YOLOv8 model
model = YOLO('yolov8l.pt')
model.to('cuda')

# Attendance initialization
attendance_file = "attendance.xlsx"
columns = ["Name", "Time", "Status"]

# Jika file attendance.xlsx belum ada, buat file baru dengan kolom yang sudah ditentukan
if not os.path.exists(attendance_file):
    attendance_data = pd.DataFrame(columns=columns)
    try:
        # Membuat file Excel baru jika belum ada
        attendance_data.to_excel(attendance_file, index=False)
        print(f"File {attendance_file} berhasil dibuat.")
    except Exception as e:
        print(f"Error creating file: {e}")
else:
    attendance_data = pd.read_excel(attendance_file)

# Load known faces
face_data = pd.read_csv("faces.csv")
known_face_encodings = []
known_face_names = list(face_data["Name"])

for _, row in face_data.iterrows():
    try:
        image = face_recognition.load_image_file(row["ImagePath"])
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
    except Exception as e:
        print(f"Error loading/encoding image {row['ImagePath']}: {e}")

def extract_name(name):
    match = re.match(r"([a-zA-Z]+)", name)
    return match.group(1) if match else name

last_logged_times = {}

def log_attendance(name, time, status):
    global attendance_data, last_logged_times
    base_name = extract_name(name)
    current_date = time[:10]

    # Jika sudah tercatat dengan status lain, jangan timpa
    if ((attendance_data["Name"] == base_name) & attendance_data["Time"].str.startswith(current_date)).any():
        return

    # Cegah pencatatan berulang terlalu cepat
    if base_name in last_logged_times and (datetime.datetime.now() - last_logged_times[base_name]).seconds < 10:
        return

    new_entry = pd.DataFrame({"Name": [base_name], "Time": [time], "Status": [status]})
    attendance_data = pd.concat([attendance_data, new_entry], ignore_index=True)

    try:
        attendance_data.to_excel(attendance_file, index=False)
        last_logged_times[base_name] = datetime.datetime.now()
        print(f"{base_name} logged at {time} with status: {status}")
        apply_color_to_status()
    except Exception as e:
        print(f"Error saving attendance: {e}")

def apply_color_to_status():
    wb = load_workbook(attendance_file)
    ws = wb.active

    green_fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")  # Detected
    yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")  # Late
    red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")  # Not Detected

    for row in range(2, ws.max_row + 1):
        status = ws.cell(row=row, column=3).value
        if status == "Detected":
            ws.cell(row=row, column=3).fill = green_fill
        elif status == "Late":
            ws.cell(row=row, column=3).fill = yellow_fill
        else:
            ws.cell(row=row, column=3).fill = red_fill

    wb.save(attendance_file)

# Halaman Utama

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Halaman Utama")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()
        self.label = QLabel("Selamat Datang di Sistem Absensi")
        self.start_button = QPushButton("Mulai Absensi")
        self.capture_button = QPushButton("Ambil Data Wajah")
        self.delete_button = QPushButton("Hapus Data Wajah")

        layout.addWidget(self.label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.delete_button)
        self.setLayout(layout)

        self.start_button.clicked.connect(self.open_attendance)
        self.capture_button.clicked.connect(self.capture_face_data)
        self.delete_button.clicked.connect(self.delete_face_data)

    def open_attendance(self):
        self.attendance_window = AttendanceApp(self)
        self.attendance_window.show()
        self.hide()

    def capture_face_data(self):
        name, ok = QInputDialog.getText(self, "Input Nama", "Masukkan nama:")
        if not ok or not name.strip():
            return

        name = name.strip()
        save_dir = os.path.join("images", name)
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(1)
        captured = 0
        max_photos = 50
        image_paths = []

        while captured < max_photos:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            for top, right, bottom, left in face_locations:
                face_image = frame[top:bottom, left:right]
                file_path = os.path.join(save_dir, f"{name}_{captured}.jpg")
                cv2.imwrite(file_path, face_image)
                image_paths.append(file_path)
                captured += 1
                if captured >= max_photos:
                    break
            cv2.imshow("Ambil Data Wajah", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        for path in image_paths:
            augment_and_add_to_csv(name, path)

        QMessageBox.information(self, "Selesai", "Data wajah berhasil diambil dan disimpan.")

    def delete_face_data(self):
        name, ok = QInputDialog.getText(self, "Hapus Data", "Masukkan nama yang ingin dihapus:")
        if not ok or not name.strip():
            return

        name = name.strip()
        save_dir = os.path.join("images", name)

        if os.path.exists(save_dir):
            for file in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, file))
            os.rmdir(save_dir)

        if os.path.exists("faces.csv"):
            df = pd.read_csv("faces.csv")
            df = df[df["Name"] != name]
            df.to_csv("faces.csv", index=False)

        QMessageBox.information(self, "Sukses", f"Data wajah '{name}' berhasil dihapus.")


def augment_and_add_to_csv(name, image_path):
    img = cv2.imread(image_path)
    augmentations = [
        img,
        cv2.flip(img, 1),
        cv2.convertScaleAbs(img, alpha=1.1, beta=10),
        cv2.convertScaleAbs(img, alpha=0.9, beta=-10),
    ]

    face_df = pd.read_csv("faces.csv") if os.path.exists("faces.csv") else pd.DataFrame(columns=["Name", "ImagePath"])
    new_rows = []

    for i, aug in enumerate(augmentations):
        aug_path = image_path.replace(".jpg", f"_aug{i}.jpg")
        cv2.imwrite(aug_path, aug)
        try:
            image = face_recognition.load_image_file(aug_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                new_rows.append({"Name": name, "ImagePath": aug_path})
        except Exception as e:
            print(f"Error encoding {aug_path}: {e}")

    updated_df = pd.concat([face_df, pd.DataFrame(new_rows)], ignore_index=True)
    updated_df.to_csv("faces.csv", index=False)

  
# Halaman Absensi

class AttendanceApp(QWidget):
    def __init__(self, main_menu):
        super().__init__()
        self.setWindowTitle("Sistem Absensi")
        self.setGeometry(100, 100, 800, 600)
        self.main_menu = main_menu

        self.layout = QVBoxLayout()
        self.label = QLabel("Klik Mulai untuk memulai absensi")
        self.image_label = QLabel()
        self.start_button = QPushButton("Mulai")
        self.stop_button = QPushButton("Selesai")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)
        self.setLayout(self.layout)

        self.start_button.clicked.connect(self.start_attendance)
        self.stop_button.clicked.connect(self.stop_attendance)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.running = False
        self.attended_names = set()

        self.start_time = None  # Menyimpan waktu mulai absensi

    def start_attendance(self):
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.label.setText("Kamera tidak dapat diakses!")
            return
        self.running = True
        self.start_time = datetime.datetime.now()  # Set waktu mulai
        self.timer.start(30)
        self.label.setText("Deteksi wajah aktif...")

    def stop_attendance(self):
        self.running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        for name in known_face_names:
            base_name = extract_name(name)
            if base_name not in self.attended_names:
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                log_attendance(base_name, time_str, "Not Detected")

        self.label.setText("Absensi dihentikan.")
        self.main_menu.show()
        self.close()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds() / 10  # Waktu dalam menit

        # status berdasarkan waktu
        status = "Not Detected"
        status_color = (0, 0, 255)  # Merah (default)

        if elapsed_time <= 2:
            # Deteksi Kehadiran (Menit 1-2)
            status = "Detected"
            status_color = (0, 255, 0)  # Hijau
        elif 2 < elapsed_time <= 4:
            # Deteksi Telat (Menit 2-4)
            status = "Late"
            status_color = (0, 255, 255)  # Kuning
        elif 4 < elapsed_time <= 5:
            # Deteksi Tidak Terdeteksi (Menit 4-5)
            status = "Not Detected"
            status_color = (0, 0, 255)  # Merah

        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cls = int(box.cls[0])
                confidence = box.conf[0]
                label = result.names[cls] if hasattr(result, 'names') else f"Class {cls}"

                if confidence > 0.5 and label == "person":
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(distances)

                        name = "Unknown"
                        if distances[best_match_index] < 0.5:
                            name = known_face_names[best_match_index]
                            base_name = extract_name(name)

                            if base_name not in self.attended_names:
                                now = datetime.datetime.now()
                                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                                log_attendance(base_name, time_str, status)
                                self.attended_names.add(base_name)

                        # Tampilkan bounding box dan nama
                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), status_color, 2)
                        cv2.putText(frame, f"{name} ({status})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                        # Tampilkan frame di PyQt
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        height, width, channel = rgb_image.shape
                        bytes_per_line = channel * width
                        qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

                        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_BGR888)
                        self.image_label.setPixmap(QPixmap(image))
                        # Tampilkan frame ke QLabel (UI)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        height, width, channel = frame_rgb.shape
                        bytes_per_line = 3 * width
                        
                        name = known_face_names[best_match_index]
                        base_name = extract_name(name)
                        
        # Tampilkan frame pada label di UI
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480)
        self.image_label.setPixmap(QPixmap.fromImage(p))
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainMenu()
    window.show()
    sys.exit(app.exec())
