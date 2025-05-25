from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import csv

# === Configuration ===
MODEL_PATH = "/Users/anubhavanand/Desktop/project/yolo_digantara_fast1/retry1/weights/best.pt"
IMAGE_PATH = "/Users/anubhavanand/Desktop/project/datasets/images/train/Raw_Observation_006_Set1.png"
CONFIDENCE_THRESHOLD = 0.25
DISPLAY = False
CLASS_NAMES = ['streak', 'star']  # Make sure these match your model's class order

# === Output folder for centroids ===
CENTROID_FOLDER = "centroids"
os.makedirs(CENTROID_FOLDER, exist_ok=True)

# === Load trained YOLOv8 model ===
model = YOLO(MODEL_PATH)

# === Predict on image ===
results = model.predict(
    source=IMAGE_PATH,
    save=True,
    save_txt=True,
    conf=CONFIDENCE_THRESHOLD
)

# === Output locations ===
output_dir = results[0].save_dir
label_dir = os.path.join(output_dir, "labels")
print(f"[INFO] Annotated images saved in: {output_dir}")
print(f"[INFO] YOLO label .txt files saved in: {label_dir}")

# === Prepare CSV filename based on image name ===
image_name = os.path.basename(IMAGE_PATH)                  # e.g. Raw_Observation_004_Set1.png
base_name = os.path.splitext(image_name)[0]                 # e.g. Raw_Observation_004_Set1
centroid_file = os.path.join(CENTROID_FOLDER, f"{base_name}.csv")

# === Save centroids to CSV ===
with open(centroid_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Class", "Confidence", "Centroid_X", "Centroid_Y"])

    print("\n[INFO] Detected objects with centroids:")
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            class_name = CLASS_NAMES[cls_id]
            print(f" - Class: {class_name}, Confidence: {conf:.2f}, Centroid: ({cx:.1f}, {cy:.1f})")
            writer.writerow([class_name, round(conf, 2), round(cx.item(), 1), round(cy.item(), 1)])

print(f"\n[INFO] Centroids saved to: {centroid_file}")

# === Optionally display predictions ===
if DISPLAY:
    for result in results:
        img = result.plot()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction - {image_name}")
        plt.axis("off")
        plt.show()
