# import cv2

# # Load input video
# cap = cv2.VideoCapture(r"C:\Users\IVTRNE26\Downloads\mold_300_22-09\mold_300_22-09")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(r"D:\bhanu\DOWNLOADS\op_floating_750ml\cam 7 (DA5392480)\op_flat_28.avi", fourcc, cap.get(cv2.CAP_PROP_FPS),
#                       (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     out.write(rotated)

# cap.release()
# out.release()


# "D:\bhanu\tata_paid_demo\gp_particles\raw_test\MV-CS032-60GC (DA5392478)\gp_paricle20250711152724053.avi"


# import cv2
# import os
# import glob

# # Input image folder
# input_folder = r"C:\Users\IVTRNE26\Downloads\cut_mark_2_22-09 (2)"
# image_paths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))  # or "*.png"

# # Process and overwrite each image
# for img_path in image_paths:
#     frame = cv2.imread(img_path)
#     rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     cv2.imwrite(img_path, rotated)


import glob
import os

import cv2

# Input image folder
input_folder = r"c:\Users\IVTRNE26\Downloads\cut_mark_2_22-09 (2)\cut_mark_2_22-09"

# Collect images (jpg, jpeg, png)
image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
image_paths = sorted(image_paths)

print(f"Found {len(image_paths)} images to rotate...")

# Process and overwrite each image
for img_path in image_paths:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"⚠️ Could not read {img_path}, skipping...")
        continue

    # Rotate 90° counterclockwise
    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Overwrite original file
    cv2.imwrite(img_path, rotated)
    print(f"✅ Rotated: {os.path.basename(img_path)}")

print("🎉 All images rotated successfully.")
