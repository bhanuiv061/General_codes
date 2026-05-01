# import cv2
# import os

# # Input video path
# video_path = r"C:\haldirams\input_images\IMG_8137.MOV"

# # Output folder for frames
# output_folder = r"C:\haldirams\output_folder"
# os.makedirs(output_folder, exist_ok=True)

# # Load video
# cap = cv2.VideoCapture(video_path)

# frame_count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # Save frame as image
#     frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
#     cv2.imwrite(frame_filename, frame)
#     frame_count += 1

# cap.release()
# print(f"Extracted {frame_count} frames to {output_folder}/")


import os

import cv2

# Input video path

video_path = r"C:\haldirams\input_images\IMG_8140.MOV"

# Output folder path (change this to yours)

output_folder = r"C:\haldirams\output_folder1"

os.makedirs(output_folder, exist_ok=True)

# Load video

cap = cv2.VideoCapture(video_path)

frame_count = 0

saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Save every 5th frame

    if frame_count % 5 == 0:
        print("gfui")

        frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")

        cv2.imwrite(frame_filename, frame)

        saved_count += 1

    frame_count += 1

cap.release()

print(f"Extracted {saved_count} frames (every 5th) to {output_folder}")
