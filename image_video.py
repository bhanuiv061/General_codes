import os
from glob import glob

import cv2


def help():
    print("""
========================================================
Images → Video Converter (OpenCV)
========================================================

This script converts a folder of images into a single video file.
Images are read in order, resized if needed, and written frame-by-frame
to create a video.

--------------------------------------------------------
WORKING FLOW
--------------------------------------------------------

1. Input Folder & Output Path
   - A folder containing images is provided.
   - An output video path is specified.
   - Output directory is created automatically if it does not exist.

2. Image Collection
   - Collects all image files with extensions:
       • .png
       • .jpg
       • .jpeg
   - Images are sorted alphabetically to preserve sequence order.

3. Frame Size Detection
   - Reads the first image to determine:
       • Video width
       • Video height
   - All subsequent images are resized to match this size.

4. Video Writer Initialization
   - Uses MJPG codec for wide compatibility.
   - Initializes VideoWriter with:
       • Output path
       • Frames per second (FPS)
       • Frame resolution

5. Frame Processing
   - Reads each image using OpenCV.
   - Skips unreadable or corrupted images.
   - Resizes frames if their resolution does not match.
   - Writes each frame to the output video.

6. Finalization
   - Releases the VideoWriter.
   - Saves the video file to disk.

--------------------------------------------------------
USAGE
--------------------------------------------------------

images_to_video(
    image_folder="path/to/image_folder",
    output_path="path/to/output_video.avi",
    fps=frames_per_second
)

--------------------------------------------------------
EXAMPLE
--------------------------------------------------------

images_to_video(
    "images/",
    "output/output.avi",
    fps=1
)

--------------------------------------------------------
OUTPUT
--------------------------------------------------------

• A single video file containing all images in sequence.

--------------------------------------------------------
NOTES
--------------------------------------------------------

• All images are forced to the same resolution.
• Image order depends on file naming.
• MJPG codec ensures good compatibility across systems.
• FPS controls how long each image appears in the video.

========================================================
""")


help()


def images_to_video(image_folder, output_path, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect images
    images = []
    images.extend(glob(os.path.join(image_folder, "*.png")))
    images.extend(glob(os.path.join(image_folder, "*.jpg")))
    images.extend(glob(os.path.join(image_folder, "*.jpeg")))
    images = sorted(images)

    if not images:
        print("No images found!")
        return

    first_frame = cv2.imread(images[0])
    if first_frame is None:
        print("Error reading first image")
        return

    height, width, _ = first_frame.shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    if not out.isOpened():
        print("VideoWriter failed to open!")
        return

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)

        out.write(frame)

    out.release()
    print("✅ Video saved:", output_path)


images_to_video(
    r"C:\Users\admin\Downloads\overlap_dataset\overlap_dataset",
    r"C:\Users\admin\Downloads\overlap_dataset\overlap_dataset\output.avi",
    fps=1,
)
