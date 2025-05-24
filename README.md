# Auto Labeling Tool for Stars and Streaks

This Python script automatically labels astronomical images by detecting and classifying **stars** (circular blobs) and **streaks** (elongated shapes). The output is in YOLO format, suitable for training object detection models like YOLOv8.

---

## ✨ Features

- Converts grayscale images into YOLO format labels.
- Uses OpenCV for image preprocessing and contour detection.
- Classifies objects into:
  - **Class 0**: Stars (round and bright)
  - **Class 1**: Streaks (elongated shapes)
- Optional debug mode to generate visual overlays.
- Batch processing of entire image folders.

---

## 🧠 Classification Logic

| Feature        | Stars        | Streaks       |
|----------------|--------------|---------------|
| Aspect Ratio   | ~1 (0.8–1.2) | >1.5 or <0.67 |
| Circularity    | > 0.5        | —             |

---

## 🗂️ Folder Structure

```bash
project/
├── auto_label.py
├── README.md
├── Reference_Images/        # Input images (grayscale)
├── generated_labels/       # Output YOLO labels
└── debug_overlays/         # Optional visual overlays (if debug=True)
