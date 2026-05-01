import os

import albumentations as A
import cv2

IMAGE_DIR = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\personal_projects\iron_\data\images"
LABEL_DIR = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\personal_projects\iron_\data\labels_yolo"

AUG_IMAGE_DIR = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\personal_projects\iron_\data\aug_images"
AUG_LABEL_DIR = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\personal_projects\iron_\data\aug_labels"

os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
os.makedirs(AUG_LABEL_DIR, exist_ok=True)


# -------------------------
# Load YOLO segmentation
# -------------------------
def load_yolo_seg(txt_path):
    polys, labels = [], []

    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()

            labels.append(int(p[0]))

            coords = list(map(float, p[1:]))

            poly = []
            for i in range(0, len(coords), 2):
                poly.append((coords[i], coords[i + 1]))

            polys.append(poly)

    return polys, labels


# -------------------------
# Save YOLO segmentation
# -------------------------
def save_yolo_seg(path, polys, labels):

    with open(path, "w") as f:
        for poly, cls in zip(polys, labels):
            flat = []

            for x, y in poly:
                x = min(max(x, 0), 1)
                y = min(max(y, 0), 1)

                flat.extend([x, y])

            line = f"{cls} " + " ".join(f"{v:.6f}" for v in flat)
            f.write(line + "\n")


# -------------------------
# Augmentation pipeline
# -------------------------
transform = A.Compose(
    [A.HorizontalFlip(p=1), A.RandomBrightnessContrast(p=1)],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)


# -------------------------
# Process images
# -------------------------
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    base = os.path.splitext(img_name)[0]

    img_path = os.path.join(IMAGE_DIR, img_name)
    lbl_path = os.path.join(LABEL_DIR, base + ".txt")

    if not os.path.exists(lbl_path):
        continue

    image = cv2.imread(img_path)
    polygons, labels = load_yolo_seg(lbl_path)

    # convert polygons → keypoints
    keypoints = []
    poly_lengths = []

    for poly in polygons:
        poly_lengths.append(len(poly))
        keypoints.extend(poly)

    out = transform(image=image, keypoints=keypoints)

    new_image = out["image"]
    new_keypoints = out["keypoints"]

    # rebuild polygons
    new_polys = []
    idx = 0

    for l in poly_lengths:
        poly = []

        for _ in range(l):
            x, y = new_keypoints[idx]
            idx += 1

            x /= new_image.shape[1]
            y /= new_image.shape[0]

            poly.append((x, y))

        new_polys.append(poly)

    # save image
    aug_img_name = f"{base}_mirror_bright.jpg"

    cv2.imwrite(os.path.join(AUG_IMAGE_DIR, aug_img_name), new_image)

    # save label
    save_yolo_seg(os.path.join(AUG_LABEL_DIR, f"{base}_mirror_bright.txt"), new_polys, labels)

    print(f"✅ Augmented {img_name}")

print("\n🎯 Augmentation complete")
