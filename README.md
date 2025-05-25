# ⭐ YOLOv8 Star & Streak Detection Pipeline

This repository provides a complete end-to-end solution to automatically detect and classify **stars** and **streaks** in astronomical images using **YOLOv8**.

---

## 🛠️ Requirements

- Python 3.7+
- OpenCV
- NumPy
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

Install dependencies:
```bash
pip install opencv-python numpy ultralytics
```

## 📦 Project Structure

```
project/
├── auto_label.py              # Script to auto-generate YOLO labels
├── yolov8_config.yaml         # YOLOv8 dataset config
├── datasets/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
|----generated_images          #it contain label images
|----generated_labels          #it contain txt labels
|----runs/
|     |--detect/
|        |--predect/
|            |--lables         #predect labels with classes
|----yolo_digantara_fast1/     #it contain all the result file 
|    |--weights/
|       |-best.pt
|        |-last.pt
|----tiff_to_png.py            #script convert tiff to png 
|----train.py                  #script to train a model
|----test.py                   #script to test a model
├── README.md
└── runs/                      # YOLOv8 training results
```

---

## 📌 Classes

| Class ID | Name   | Description           |
|----------|--------|-----------------------|
| 0        | star   | Circular blobs        |
| 1        | streak | Long, narrow shapes   |

---

## 🧪 Step 1: Auto-label Images

Use `auto_label.py` to generate YOLO-format `.txt` files from grayscale images.

```bash
python auto_label.py
```

Update paths in the script:
```python
image_folder = "/path/to/Reference_Images"
label_folder = "/path/to/generated_labels"
debug_folder = "/path/to/generated_images"
```
![Raw_Observation_001_Set1](https://github.com/user-attachments/assets/044690df-8a99-48d6-bfad-a8f35daab794)


---

## 📁 Step 2: Prepare Dataset for YOLOv8

Organize the dataset like this:

```
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Make sure each image has a corresponding `.txt` label file in the `labels/` folder.

---

## 🧾 Step 3: Create Dataset Config

**`yolov8_config.yaml`**
```yaml
path: datasets/
train: images/train
val: images/val

names:
  0: star
  1: streak
```

---

## 🚀 Step 4: Train YOLOv8 Model

Install YOLOv8:
```bash
pip install ultralytics
```

Run training:
```bash
python train.py
```

> You can use `yolov8s.pt`, `yolov8m.pt` for larger models.

---

## 🔍 Step 6: Run Inference

```bash
python test.py
```

Predicted images will be saved to:
```
runs/detect/predict/
```

![Raw_Observation_006_Set1](https://github.com/user-attachments/assets/ff97528e-694a-453e-ac5f-99d2a7789998)

Centroid coordinates are stored in the centroids folder:
```
centroids/Raw_Observation_006_Set1.csv
```
---








