from ultralytics import YOLO
import torch

# ✅ Select device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# ✅ Load the YOLOv8n (Nano) model
model = YOLO("yolov8n.pt")

# ✅ Train the model
model.train(
    data="/Users/anubhavanand/Desktop/project/yolov8_config.yaml",
    model="yolov8n.pt",
    epochs=100,
    imgsz=1024,
    patience=300,
    project="yolo_digantara_fast4",
    device=device
)
