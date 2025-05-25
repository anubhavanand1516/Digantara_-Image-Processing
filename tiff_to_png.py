import os
import cv2
import numpy as np
from PIL import Image

# Paths
input_dir = "/Users/anubhavanand/Desktop/Datasets/Raw_Images"        # TIFF input
output_img_dir = "/Users/anubhavanand/Desktop/star_streak/yolo_datasets/converted_images"                # PNG output
output_label_dir = "/Users/anubhavanand/Desktop/star_streak/yolo_datasets/yolo_labels"                   # YOLO labels

# Create output folders
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

def detect_objects(image, filename, img_width, img_height):
    """Detect stars (blob) and streaks (lines) and save YOLO annotations."""
    labels = []

    # Normalize 16-bit image to 8-bit
    gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Detect Stars (Blob Detection) ---
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 1000
    params.filterByCircularity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(blurred)

    for kp in keypoints:
        x, y = kp.pt
        d = kp.size
        labels.append(f"1 {x/img_width:.6f} {y/img_height:.6f} {d/img_width:.6f} {d/img_height:.6f}")

    # --- Detect Streaks (Hough Line Transform) ---
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines[:10]:  # Limit to avoid excess
            x1, y1, x2, y2 = line[0]
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            w = abs(x2 - x1) / img_width
            h = abs(y2 - y1) / img_height
            labels.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Save YOLO annotations
    with open(os.path.join(output_label_dir, f"{filename}.txt"), 'w') as f:
        for label in labels:
            f.write(label + "\n")

# Process all .tiff images
for file in os.listdir(input_dir):
    if file.lower().endswith(".tiff"):
        name = os.path.splitext(file)[0]
        path = os.path.join(input_dir, file)
        
        # Load TIFF image
        img = Image.open(path)
        img_array = np.array(img)

        # Save as PNG (for YOLO input or visualization)
        output_png_path = os.path.join(output_img_dir, f"{name}.png")
        img_normalized = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        cv2.imwrite(output_png_path, img_normalized)

        # Generate YOLO labels
        detect_objects(img_normalized, name, img_array.shape[1], img_array.shape[0])
