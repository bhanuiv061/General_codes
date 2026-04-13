# import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Fix OpenMP runtime error

# import cv2

# import albumentations as A

# # =========================

# # Paths

# # =========================

# IMAGE_DIR = r"C:\imagevision projects\tata\combine_i2_day4_base 1\combine_i2_day3_base\train\images"

# LABEL_DIR = r"C:\imagevision projects\tata\combine_i2_day4_base 1\combine_i2_day3_base\train\labels"

# AUG_IMAGE_DIR = r"C:\imagevision projects\tata\combine_i2_day4_base 1\combine_i2_day3_base\train\aug_images"
# AUG_LABEL_DIR = r"C:\imagevision projects\tata\combine_i2_day4_base 1\combine_i2_day3_base\train\aug_labels"

# # Create output directories

# os.makedirs(AUG_IMAGE_DIR, exist_ok=True)

# os.makedirs(AUG_LABEL_DIR, exist_ok=True)

# # Copy existing classes.txt if available

# if os.path.exists(os.path.join(LABEL_DIR, "classes.txt")):

#     with open(os.path.join(LABEL_DIR, "classes.txt"), "r") as f:

#         class_names = [line.strip() for line in f if line.strip()]

# else:

#     # fallback example

#     class_names = ["defect"]

# # Save classes.txt into aug_labels folder

# with open(os.path.join(AUG_LABEL_DIR, "classes.txt"), "w") as f:

#     f.write("\n".join(class_names) + "\n")

# # =========================

# # Augmentations (only new ones)

# # =========================

# transform = A.Compose([

#     # Flips

#     A.HorizontalFlip(p=0.5),

#     A.VerticalFlip(p=0.5),

#     # fixed rotations 30° or 60°

#     A.OneOf([

#         A.Affine(rotate=30, scale=(1.0, 1.0), fit_output=True, cval=(0, 0, 0)),

#         A.Affine(rotate=60, scale=(1.0, 1.0), fit_output=True, cval=(0, 0, 0)),

#     ], p=0.5),

#     # any random rotation (0–360°)

#     A.Rotate(limit=360, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=0.5),

#     # zoom in/out

#     A.OneOf([

#         A.Affine(scale=(1.2, 1.5), fit_output=True, cval=(0, 0, 0)),  # zoom-in

#         A.Affine(scale=(0.7, 0.9), fit_output=True, cval=(0, 0, 0)),  # zoom-out

#     ], p=0.5),

#     # brightness/contrast

#     A.RandomBrightnessContrast(p=0.4)

# ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2))


# # =========================

# # Helper functions

# # =========================

# def load_yolo_labels(txt_file):

#     """Read YOLO labels (supports multiple classes)."""

#     bboxes, labels = [], []

#     with open(txt_file, "r") as f:

#         for line in f:

#             parts = line.strip().split()

#             if len(parts) == 5:

#                 cls, x, y, w, h = parts

#                 bboxes.append([float(x), float(y), float(w), float(h)])

#                 labels.append(int(cls))  # force integer

#     return bboxes, labels


# def save_yolo_labels(txt_file, bboxes, labels):

#     """Save YOLO labels with integer class IDs."""

#     with open(txt_file, "w") as f:

#         for (bbox, cls) in zip(bboxes, labels):

#             x, y, w, h = bbox

#             f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# # =========================

# # Main Loop

# # =========================

# N_AUGS = 5  # number of augmentations per image

# for filename in os.listdir(IMAGE_DIR):

#     if filename.lower().endswith((".jpg", ".png", ".jpeg")):

#         img_path = os.path.join(IMAGE_DIR, filename)

#         label_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")

#         if not os.path.exists(label_path):

#             print(f"⚠ No label found for {filename}, skipping...")

#             continue

#         # Load image & labels

#         image = cv2.imread(img_path)

#         bboxes, class_labels = load_yolo_labels(label_path)

#         # Generate augmentations

#         for i in range(N_AUGS):

#             augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

#             aug_img = augmented["image"]

#             aug_bboxes = augmented["bboxes"]

#             aug_labels = [int(c) for c in augmented["class_labels"]]  # ensure int

#             # Save augmented image

#             aug_img_name = os.path.splitext(filename)[0] + f"_aug{i+1}.jpg"

#             cv2.imwrite(os.path.join(AUG_IMAGE_DIR, aug_img_name), aug_img)

#             # Save augmented labels

#             aug_label_name = os.path.splitext(filename)[0] + f"_aug{i+1}.txt"

#             save_yolo_labels(os.path.join(AUG_LABEL_DIR, aug_label_name), aug_bboxes, aug_labels)

#         print(f"✅ Augmented {filename} -> {N_AUGS} versions saved")

# print(f"🎉 Augmentation completed. Classes file saved in {AUG_LABEL_DIR}/classes.txt")


import os
import random

import albumentations as A
import cv2

# =========================
# Paths
# =========================
IMAGE_DIR = r"D:\crown\images\cam 1 (L30766761)\New folder\images"
LABEL_DIR = r"D:\crown\images\cam 1 (L30766761)\New folder\txt_labels"
AUG_IMAGE_DIR = os.path.join(os.path.dirname(IMAGE_DIR), "aug_imagesbp_")
AUG_LABEL_DIR = os.path.join(os.path.dirname(LABEL_DIR), "aug_labelsbp_")

# Create output directories
os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
os.makedirs(AUG_LABEL_DIR, exist_ok=True)

# Copy existing classes.txt if available
if os.path.exists(os.path.join(LABEL_DIR, "classes.txt")):
    with open(os.path.join(LABEL_DIR, "classes.txt")) as f:
        class_names = [line.strip() for line in f if line.strip()]
else:
    class_names = ["defect"]

with open(os.path.join(AUG_LABEL_DIR, "classes.txt"), "w") as f:
    f.write("\n".join(class_names) + "\n")


# =========================
# Helper functions
# =========================
def load_yolo_labels(txt_file):
    bboxes, labels = [], []
    with open(txt_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = parts
                bboxes.append([float(x), float(y), float(w), float(h)])
                labels.append(int(cls))
    return bboxes, labels


def save_yolo_labels(txt_file, bboxes, labels):
    with open(txt_file, "w") as f:
        for bbox, cls in zip(bboxes, labels):
            x, y, w, h = bbox
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# =========================
# Define individual transforms
# =========================
all_transforms = [
    A.HorizontalFlip(p=1.0),  # Optional horizontal flip
    # A.VerticalFlip(p=1.0),      # Flips the image vertically
    A.Affine(rotate=20, scale=(1.0, 1.0), fit_output=True, cval=(0, 0, 0)),  # Rotation with constant background
    # A.Affine(rotate=60, scale=(1.0, 1.0), fit_output=True, cval=(0, 0, 0)), # Optional
    # A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1.0),  # Alternative rotate
    A.Affine(scale=(1.2, 1.5), fit_output=True, cval=(0, 0, 0)),  # Zoom-in (enlarge image)
    A.Affine(scale=(0.7, 0.9), fit_output=True, cval=(0, 0, 0)),  # Zoom-out (shrink image)
    A.RandomBrightnessContrast(p=1.0),  # Adjust brightness & contrast
]

# =========================
# Main Loop (2 different random augs per image)
# =========================
N_AUGS = 4

for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(IMAGE_DIR, filename)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path):
            print(f"⚠ No label found for {filename}, skipping...")
            continue

        # Load image & labels
        image = cv2.imread(img_path)
        bboxes, class_labels = load_yolo_labels(label_path)

        # Pick 2 unique transforms for this image
        chosen_transforms = random.sample(all_transforms, N_AUGS)

        for i, t in enumerate(chosen_transforms, start=1):
            transform = A.Compose(
                [t], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2)
            )

            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = [int(c) for c in augmented["class_labels"]]

            # Save augmented image
            aug_img_name = os.path.splitext(filename)[0] + f"_aug{i}.jpg"
            cv2.imwrite(os.path.join(AUG_IMAGE_DIR, aug_img_name), aug_img)

            # Save augmented labels
            aug_label_name = os.path.splitext(filename)[0] + f"_aug{i}.txt"
            save_yolo_labels(os.path.join(AUG_LABEL_DIR, aug_label_name), aug_bboxes, aug_labels)

        print(f"✅ Augmented {filename} -> {N_AUGS} different versions saved")

print(f"🎉 Augmentation completed. Classes file saved in {AUG_LABEL_DIR}/classes.txt")


# import os
# import cv2
# import random
# import albumentations as A

# # =========================
# # Paths
# # =========================
# IMAGE_DIR = r"C:\imagevision projects\tata\bottomcut_8oct\train\images"
# LABEL_DIR = r"C:\imagevision projects\tata\bottomcut_8oct\train\labels"
# AUG_IMAGE_DIR = os.path.join(os.path.dirname(IMAGE_DIR), "aug_imagesv1")
# AUG_LABEL_DIR = os.path.join(os.path.dirname(LABEL_DIR), "aug_labelsv1")

# # Create output directories
# os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
# os.makedirs(AUG_LABEL_DIR, exist_ok=True)

# # Copy existing classes.txt if available
# if os.path.exists(os.path.join(LABEL_DIR, "classes.txt")):
#     with open(os.path.join(LABEL_DIR, "classes.txt"), "r") as f:
#         class_names = [line.strip() for line in f if line.strip()]
# else:
#     class_names = ["defect"]

# with open(os.path.join(AUG_LABEL_DIR, "classes.txt"), "w") as f:
#     f.write("\n".join(class_names) + "\n")

# # =========================
# # Helper functions
# # =========================
# def load_yolo_labels(txt_file):
#     bboxes, labels = [], []
#     with open(txt_file, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 5:
#                 cls, x, y, w, h = parts
#                 bboxes.append([float(x), float(y), float(w), float(h)])
#                 labels.append(int(cls))
#     return bboxes, labels

# def save_yolo_labels(txt_file, bboxes, labels):
#     with open(txt_file, "w") as f:
#         for (bbox, cls) in zip(bboxes, labels):
#             x, y, w, h = bbox
#             f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# # =========================
# # Define transforms (rotation 0–90 + zooms + brightness/contrast)
# # =========================
# all_transforms = [
#     A.Rotate(limit=(10, 40), border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1.0),  # random 0–90°
#     A.Affine(scale=(1.2, 1.5), fit_output=True, cval=(0, 0, 0)),  # zoom-in
#     A.Affine(scale=(0.7, 0.9), fit_output=True, cval=(0, 0, 0)),  # zoom-out
#     A.RandomBrightnessContrast(p=1.0)  # brightness/contrast
# ]

# # =========================
# # Main Loop
# # =========================
# N_AUGS = 5  # number of augmentations per image

# for filename in os.listdir(IMAGE_DIR):
#     if filename.lower().endswith((".jpg", ".png", ".jpeg")):
#         img_path = os.path.join(IMAGE_DIR, filename)
#         label_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")

#         if not os.path.exists(label_path):
#             print(f"⚠ No label found for {filename}, skipping...")
#             continue

#         # Load image & labels
#         image = cv2.imread(img_path)
#         bboxes, class_labels = load_yolo_labels(label_path)

#         # Randomly pick N_AUGS transforms for this image
#         chosen_transforms = random.choices(all_transforms, k=N_AUGS)

#         for i, t in enumerate(chosen_transforms, start=1):
#             transform = A.Compose(
#                 [t],
#                 bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2)
#             )

#             augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
#             aug_img = augmented["image"]
#             aug_bboxes = augmented["bboxes"]
#             aug_labels = [int(c) for c in augmented["class_labels"]]

#             # Save augmented image
#             aug_img_name = os.path.splitext(filename)[0] + f"_aug{i}_.jpg"
#             cv2.imwrite(os.path.join(AUG_IMAGE_DIR, aug_img_name), aug_img)

#             # Save augmented labels
#             aug_label_name = os.path.splitext(filename)[0] + f"_aug{i}_.txt"
#             save_yolo_labels(os.path.join(AUG_LABEL_DIR, aug_label_name), aug_bboxes, aug_labels)

#         print(f"✅ Augmented {filename} -> {N_AUGS} versions saved")

# print(f"🎉 Augmentation completed. Classes file saved in {AUG_LABEL_DIR}/classes.txt")
