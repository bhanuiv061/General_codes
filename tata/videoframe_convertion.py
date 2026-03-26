import os

import cv2

# Input video path
video_path = r"C:\Users\IVTRNE26\Downloads\op_flat_7.avi"

# Output folder
file_name = "op1"
output_folder = r"C:\imagevision projects\tata\floating_video\op"
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Save frame (only filename, not full path twice)
    frame_filename = os.path.join(output_folder, f"{file_name}_{frame_count:06d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"Done! Saved {frame_count} frames in '{output_folder}'")
