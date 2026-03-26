import os

import cv2

input_file = (
    r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\RFQ_\SHG_municipal admin\documents\tree_counting.avi"
)
output_file = (
    r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\RFQ_\SHG_municipal admin\documents\output\output.mp4"
)

# Create output folder
os.makedirs(os.path.dirname(output_file), exist_ok=True)

cap = cv2.VideoCapture(input_file)

# Get original properties (keeps resolution)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# MP4 codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()

print("✅ Conversion done (no FFmpeg, resolution preserved)")
