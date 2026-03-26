# import os
# import shutil
# import random
# from tqdm import tqdm

# # Base paths
# base_path = r"C:\imagevision projects\tata\23sept_augumented\DATASET"
# train_images = os.path.join(base_path, "train", "images")
# train_labels = os.path.join(base_path, "train", "labels")

# valid_images = os.path.join(base_path, "valid", "images")
# valid_labels = os.path.join(base_path, "valid", "labels")

# # Create valid output folders
# os.makedirs(valid_images, exist_ok=True)

# os.makedirs(valid_labels, exist_ok=True)

# # List all image files
# image_files = [f for f in os.listdir(valid_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
# random.shuffle(image_files)

# # 20% split
# num_valid = int(len(image_files) * 0.5)
# print(num_valid)
# valid_files = image_files[:num_valid]

# # Move 20% to valid folder
# for img_file in tqdm(valid_files, desc="Moving to valid set"):
#     label_file = os.path.splitext(img_file)[0] + ".txt"

#     # Move image
#     src_img = os.path.join(train_images, img_file)
#     dst_img = os.path.join(valid_images, img_file)
#     shutil.move(src_img, dst_img)

#     # Move label
#     src_lbl = os.path.join(train_labels, label_file)
#     dst_lbl = os.path.join(valid_labels, label_file)
#     if os.path.exists(src_lbl):
#         shutil.move(src_lbl, dst_lbl)
#     else:
#         print(f"⚠️ Label missing for {img_file}")

# ----------------------------------------------------------------------------------------
# VALID DATA
# import os
# import shutil
# import random
# from tqdm import tqdm

# # Base paths
# base_path = r"c:\imagevision projects\tata\bottomcut_8oct"
# train_images = os.path.join(base_path, "train", "images")
# train_labels = os.path.join(base_path, "train", "labels")

# valid_images = os.path.join(base_path, "valid", "images")
# valid_labels = os.path.join(base_path, "valid", "labels")

# # Create valid output folders
# os.makedirs(valid_images, exist_ok=True)
# os.makedirs(valid_labels, exist_ok=True)

# # List all image files from TRAIN (not valid)
# image_files = [f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
# random.shuffle(image_files)

# # 20% split
# num_valid = int(len(image_files) * 0.2)
# print(f"Moving {num_valid} images to validation set...")
# valid_files = image_files[:num_valid]

# # Move 20% to valid folder
# for img_file in tqdm(valid_files, desc="Moving to valid set"):
#     label_file = os.path.splitext(img_file)[0] + ".txt"

#     # Move image
#     src_img = os.path.join(train_images, img_file)
#     dst_img = os.path.join(valid_images, img_file)
#     if not os.path.exists(dst_img):
#         shutil.move(src_img, dst_img)

#     # Move label
#     src_lbl = os.path.join(train_labels, label_file)
#     dst_lbl = os.path.join(valid_labels, label_file)
#     if os.path.exists(src_lbl):
#         shutil.move(src_lbl, dst_lbl)
#     else:
#         print(f"⚠️ Label missing for {img_file}")

# -------------------------------------------------------------------------------------
# test
# import os
# import shutil
# import random
# from tqdm import tqdm

# # Base paths
# base_path = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Crown\engineering\poc_\data_sets\dataset\dataset\images"
# train_images = os.path.join(base_path, "train", "images")
# train_labels = os.path.join(base_path, "train", "labels")

# test_images = os.path.join(base_path, "valid", "images")
# test_labels = os.path.join(base_path, "valid", "labels")

# # Create output folders
# os.makedirs(test_images, exist_ok=True)
# os.makedirs(test_labels, exist_ok=True)

# # ✅ List all image files from TRAIN (not TEST!)
# image_files = [f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
# random.shuffle(image_files)

# # % split for test
# test_split = 0.16    # 25% of data will go to test
# num_test = int(len(image_files) * test_split)

# print(f"Moving {num_test} images to test set...")

# test_files = image_files[:num_test]

# # ✅ Move test images/labels
# for img_file in tqdm(test_files, desc="Moving to test set"):
#     label_file = os.path.splitext(img_file)[0] + ".txt"

#     # Image
#     src_img = os.path.join(train_images, img_file)
#     dst_img = os.path.join(test_images, img_file)
#     if not os.path.exists(dst_img):
#         shutil.move(src_img, dst_img)

#     # Label
#     src_lbl = os.path.join(train_labels, label_file)
#     dst_lbl = os.path.join(test_labels, label_file)
#     if os.path.exists(src_lbl):
#         shutil.move(src_lbl, dst_lbl)
#     else:
#         print(f"⚠️ Label missing for {img_file}")


import os
import random
import shutil

from tqdm import tqdm

"""
========================================================
Dataset Split Script (Train / Validation)
========================================================

Purpose:
--------
This script splits an image dataset and its corresponding YOLO label files
into training and validation (test) sets.

It:
- Reads images from `images_` folder
- Reads labels from `labels_` folder
- Randomly shuffles the dataset
- Splits data into train and valid sets
- Moves files into YOLO-compatible directory structure
- Warns if any image does not have a corresponding label

Typical use case:
-----------------
Preparing datasets for YOLOv5 / YOLOv8 / YOLOv9 / YOLOv11 training.
"""

# =========================
# Base paths (Root dataset directory)
# =========================
BASE_PATH = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Crown\engineering\poc_\data_sets\dataset\dataset"

# Source directories (original data)
IMAGES_DIR = os.path.join(BASE_PATH, "images")
LABELS_DIR = os.path.join(BASE_PATH, "labels")

# Destination directories (after split)
TRAIN_IMG = os.path.join(BASE_PATH, "train", "images")
TRAIN_LBL = os.path.join(BASE_PATH, "train", "labels")
TEST_IMG = os.path.join(BASE_PATH, "valid", "images")
TEST_LBL = os.path.join(BASE_PATH, "valid", "labels")

# =========================
# Create output folders if they do not exist
# =========================
os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_LBL, exist_ok=True)
os.makedirs(TEST_IMG, exist_ok=True)
os.makedirs(TEST_LBL, exist_ok=True)

# =========================
# Collect all image files
# =========================
# Reads all images with valid extensions from the images folder
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Shuffle images to ensure random distribution
random.shuffle(image_files)

# =========================
# Split configuration
# =========================
test_split = 0.2  # 20% of data will go to validation set
num_test = int(len(image_files) * test_split)

# Split file lists
test_files = image_files[:num_test]
train_files = image_files[num_test:]

# Display dataset statistics
print(f"Total images : {len(image_files)}")
print(f"Train images : {len(train_files)}")
print(f"Test images  : {len(test_files)}")

# =========================
# Move TRAIN files
# =========================
for img_file in tqdm(train_files, desc="Moving TRAIN files"):
    # Create label filename from image filename
    lbl_file = os.path.splitext(img_file)[0] + ".txt"

    # Move image to train/images
    shutil.move(os.path.join(IMAGES_DIR, img_file), os.path.join(TRAIN_IMG, img_file))

    # Move label to train/labels if it exists
    lbl_src = os.path.join(LABELS_DIR, lbl_file)
    if os.path.exists(lbl_src):
        shutil.move(lbl_src, os.path.join(TRAIN_LBL, lbl_file))
    else:
        print(f"⚠️ Missing label for {img_file}")

# =========================
# Move TEST files
# =========================
for img_file in tqdm(test_files, desc="Moving TEST files"):
    # Create label filename from image filename
    lbl_file = os.path.splitext(img_file)[0] + ".txt"

    # Move image to valid/images
    shutil.move(os.path.join(IMAGES_DIR, img_file), os.path.join(TEST_IMG, img_file))

    # Move label to valid/labels if it exists
    lbl_src = os.path.join(LABELS_DIR, lbl_file)
    if os.path.exists(lbl_src):
        shutil.move(lbl_src, os.path.join(TEST_LBL, lbl_file))
    else:
        print(f"⚠️ Missing label for {img_file}")

# =========================
# Completion message
# =========================
print("✅ Dataset split completed successfully!")
