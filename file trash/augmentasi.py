import cv2

# Fungsi auto brightness & contrast
def auto_brightness_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

# Fungsi augmentasi real-time (flip dan zoom)
def augment_realtime(image):
    # Flip horizontal (mirror)
    image = cv2.flip(image, 1)

    # Resize (zoom in 10%)
    h, w = image.shape[:2]
    zoom_factor = 1.1
    image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))

    # Crop tengah untuk menyesuaikan ukuran asli
    center_h, center_w = image.shape[:2]
    startx = center_w // 2 - w // 2
    starty = center_h // 2 - h // 2
    image = image[starty:starty + h, startx:startx + w]

    return image

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membuka kamera.")
        break

    # Augmentasi real-time
    frame_aug = auto_brightness_contrast(frame)
    frame_aug = augment_realtime(frame_aug)

    # Tampilkan hasil augmentasi
    cv2.imshow("Kamera + Augmentasi Real-time", frame_aug)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
