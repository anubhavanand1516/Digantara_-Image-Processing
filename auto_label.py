import cv2
import numpy as np
import os

def normalize_bbox(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def auto_label_single_image(image_path, label_path, debug=False, debug_dir=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = img.shape

    # Preprocessing: Blur + CLAHE
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blurred)

    # Threshold + Noise cleanup
    _, thresh = cv2.threshold(clahe_img, 20, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if debug else None

    with open(label_path, 'w') as f:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Ignore tiny or noisy blobs
            if w < 2 or h < 2 or area < 4:
                continue

            aspect_ratio = w / float(h)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter ** 2 + 1e-6))

            if 0.8 < aspect_ratio < 1.2 and circularity > 0.5:
                cls = 0  # Star
                color = (0, 255, 0)
            elif aspect_ratio >= 1.5 or aspect_ratio <= 0.67:
                cls = 1  # Streak
                color = (0, 0, 255)
            else:
                continue  # ambiguous shape, skip

            x_center, y_center, w_norm, h_norm = normalize_bbox(x, y, w, h, img_w, img_h)
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            if debug:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 1)

    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_out = os.path.join(debug_dir, os.path.basename(image_path))
        cv2.imwrite(debug_out, debug_img)

def auto_label_folder(image_dir, label_dir, debug=False, debug_dir=None):
    os.makedirs(label_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, txt_file)
        auto_label_single_image(img_path, label_path, debug=debug, debug_dir=debug_dir)
        print(f"[INFO] Labeled {img_file} -> {txt_file}")

# === Example usage ===
if __name__ == "__main__":
    image_folder = "/Users/anubhavanand/Desktop/Datasets/Reference_Images"
    label_folder = "/Users/anubhavanand/Desktop/level/generated_labels1"
    debug_folder = "/Users/anubhavanand/Desktop/level/debug_overlays1"  # Optional

    auto_label_folder(image_folder, label_folder, debug=True, debug_dir=debug_folder)
