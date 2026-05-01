import cv2

# Load input video
cap = cv2.VideoCapture(
    r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\TATA_V3_HIMALAYA\image_data_\day_1_(05_12_25)\tata_meach_side_view_images\bp_images\testing_unseen\cam 7 (DA5392480)\hair_cabl_2.avi"
)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\TATA_V3_HIMALAYA\image_data_\day_1_(05_12_25)\tata_meach_side_view_images\bp_images\testing_unseen\cam 7 (DA5392480)\hair_cabl_2.avi",
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    out.write(rotated)

print(out)

cap.release()
out.release()


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


# import cv2
# import os
# import glob

# # Input image folder
# input_folder = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Spritzer_pet_water\spritizer_v3\cable_0.5mm\p1\cable_frames\images_25"

# # Collect images (jpg, jpeg, png)
# image_paths = []
# for ext in ("*.jpg", "*.jpeg", "*.png"):
#     image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
# image_paths = sorted(image_paths)

# print(f"Found {len(image_paths)} images to rotate...")

# # Process and overwrite each image
# for img_path in image_paths:
#     frame = cv2.imread(img_path)
#     if frame is None:
#         print(f"⚠️ Could not read {img_path}, skipping...")
#         continue

#     # Rotate 90° counterclockwise
#     rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

#     # Overwrite original file
#     cv2.imwrite(img_path, rotated)
#     print(f"✅ Rotated: {os.path.basename(img_path)}")

# print("🎉 All images rotated successfully.")


import os

import cv2

video_path = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Spritzer_pet_water\spritizer_v3\paper\cam 7 (DA5392480)\cable_0_5_32.avi"
output_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Spritzer_pet_water\spritizer_v3\paper\cam 7 (DA5392480)\cable_0_5_32"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0
saved_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % 2 == 0:
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate CCW
        frame_path = os.path.join(output_dir, f"gp_41{saved_idx:04d}.jpg")
        cv2.imwrite(frame_path, rotated_frame)
        print(f"CUT_MARK{saved_idx:04d}.jpg")
        saved_idx += 1

    frame_idx += 1

cap.release()
