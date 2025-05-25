import os
import shutil
import random
from pathlib import Path
import yaml

# Source directories
source_img_dir = Path("/Users/anubhavanand/Desktop/level/converted_images")
source_label_dir = Path("/Users/anubhavanand/Desktop/level/generated_labels")

# Target structure
base_dir = Path("digantara_yolo")
dataset_dir = base_dir / "dataset"
image_train_dir = dataset_dir / "images/train"
image_val_dir = dataset_dir / "images/val"
label_train_dir = dataset_dir / "labels/train"
label_val_dir = dataset_dir / "labels/val"

# Create folders
for d in [image_train_dir, image_val_dir, label_train_dir, label_val_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Gather and shuffle image list
all_images = list(source_img_dir.glob("*.png"))
random.seed(42)
random.shuffle(all_images)

# 80-20 split
split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Helper to copy images and label files
def copy_data(images, image_out, label_out):
    for img_path in images:
        shutil.copy(img_path, image_out / img_path.name)
        label_path = source_label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, label_out / label_path.name)
        else:
            (label_out / (img_path.stem + ".txt")).touch()  # create empty label

copy_data(train_images, image_train_dir, label_train_dir)
copy_data(val_images, image_val_dir, label_val_dir)

# Write data.yaml
data_yaml = {
    "train": str(image_train_dir.relative_to(base_dir)),
    "val": str(image_val_dir.relative_to(base_dir)),
    "nc": 2,
    "names": ["star", "streak"]
}
with open(base_dir / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("✅ Dataset converted to digantara_yolo/ format.")
