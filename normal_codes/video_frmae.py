import os
from datetime import datetime

import cv2

# -----------------------------
# Input Video
# -----------------------------
video_path = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\personal_projects\iron_\input_video\v2.mp4"

# Custom frame skip
skip_frames = 5  # save every 5th frame

# -----------------------------
# Get video filename
# -----------------------------
video_name = os.path.splitext(os.path.basename(video_path))[0]

# -----------------------------
# Create Main Output Folder
# -----------------------------
main_output = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\personal_projects\iron_\output_frames"
os.makedirs(main_output, exist_ok=True)

# -----------------------------
# Create Video Specific Folder
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(main_output, f"{video_name}_{timestamp}")

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Read Video
# -----------------------------
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

print("Processing video...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % skip_frames == 0:
        filename = f"frame_{saved_count:05d}.jpg"
        save_path = os.path.join(output_folder, filename)

        cv2.imwrite(save_path, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print("Done!")
print(f"Total frames processed: {frame_count}")
print(f"Frames saved: {saved_count}")
print(f"Saved in: {output_folder}")
