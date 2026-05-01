# import torch
# from yolov5 import YOLOv5
# import cv2
# import numpy as np
# import pathlib
# import os
# from tqdm import tqdm

# # Path configuration
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # Model configuration
# model_path = r"C:/Users/iv061/Downloads/sharp_bister.pt"
# model = YOLOv5(model_path, device="cpu")  # or 'cuda' for GPU

# # Detection parameters
# CONF_THRESHOLD = 0.8  # Adjust this value (0-1), higher = more conservative
# IOU_THRESHOLD = 0.45  # Intersection over Union threshold

# # Input/Output paths
# input_path = r"C:\Users\iv061\Downloads\mvs\mvs"
# output_path = r"D:\bhanu\sharp_blisters\output_videos\fp"
# os.makedirs(output_path, exist_ok=True)
# logo_path = r".\logo_black.png"

# # Create output directory if it doesn't exist
# os.makedirs(output_path, exist_ok=True)

# # Load logo (optional)
# logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED) if os.path.exists(logo_path) else None

# # Function to add watermark
# def add_watermark(frame, text="Demonstration Only", opacity=0.15):
#     overlay = frame.copy()
#     text_size = 1.2
#     thickness = 3
#     color = (255, 255, 255)

#     text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, text_size, thickness)[0]
#     text_x = (frame.shape[1] - text_width) // 2
#     text_y = (frame.shape[0] + text_height) // 2

#     watermark = np.zeros_like(frame, dtype=np.uint8)
#     cv2.putText(watermark, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, text_size, color, thickness, cv2.LINE_AA)

#     center = (frame.shape[1] // 2, frame.shape[0] // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle=30, scale=1)
#     rotated_watermark = cv2.warpAffine(watermark, rotation_matrix, (frame.shape[1], frame.shape[0]))

#     cv2.addWeighted(rotated_watermark, opacity, overlay, 1 - opacity, 0, overlay)
#     return overlay

# # Function to add logo
# def add_logo(frame, logo, logo_width=150):
#     if logo is None:
#         return frame

#     h_logo, w_logo = logo.shape[:2]
#     aspect_ratio = h_logo / w_logo
#     new_height = int(logo_width * aspect_ratio)
#     resized_logo = cv2.resize(logo, (logo_width, new_height))

#     x_offset = 10
#     y_offset = 35

#     if resized_logo.shape[2] == 4:
#         alpha = resized_logo[:, :, 3] / 255.0
#         for c in range(3):
#             frame[y_offset:y_offset + new_height, x_offset:x_offset + logo_width, c] = (
#                 alpha * resized_logo[:, :, c] + (1 - alpha) * frame[y_offset:y_offset + new_height, x_offset:x_offset + logo_width, c]
#             )
#     else:
#         frame[y_offset:y_offset + new_height, x_offset:x_offset + logo_width] = resized_logo[:, :, :3]

#     return frame

# # Function to process a single frame
# def process_frame(frame):
#     # Perform inference with confidence threshold
#     results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
#     predictions = results.pandas().xyxy[0]

#     # Draw bounding boxes
#     for _, pred in predictions.iterrows():
#         if pred['confidence'] >= CONF_THRESHOLD:  # Additional confidence check
#             xmin, ymin, xmax, ymax = map(int, [pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']])
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

#             # Optional: Add label and confidence
#             label = f"{pred['name']} {pred['confidence']:.2f}"
#             cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

#     frame = add_watermark(frame)
#     frame = add_logo(frame, logo)
#     return frame

# # Check if input is video or image folder
# if os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mov')):
#     # Process video
#     cap = cv2.VideoCapture(input_path)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     output_video = os.path.join(output_path, "processed_" + os.path.basename(input_path))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     with tqdm(total=frame_count, desc="Processing video") as pbar:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             processed_frame = process_frame(frame)
#             out.write(processed_frame)
#             pbar.update(1)

#     cap.release()
#     out.release()
#     print(f"Video processing complete. Saved to: {output_video}")

# elif os.path.isdir(input_path):
#     # Process images
#     image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
#     image_files = [f for f in os.listdir(input_path) if f.lower().endswith(image_extensions)]

#     if not image_files:
#         print("No images found in the specified directory")
#     else:
#         os.makedirs(os.path.join(output_path, "processed_images"), exist_ok=True)

#         for img_file in tqdm(image_files, desc="Processing images"):
#             img_path = os.path.join(input_path, img_file)
#             frame = cv2.imread(img_path)

#             if frame is not None:
#                 processed_frame = process_frame(frame)
#                 output_img_path = os.path.join(output_path, "processed_images", "processed_" + img_file)
#                 cv2.imwrite(output_img_path, processed_frame)

#         print(f"Processed {len(image_files)} images. Saved to: {os.path.join(output_path, 'processed_images')}")

# else:
#     print("Invalid input path. Please provide either a video file or a directory containing images")


# print(f"Using confidence threshold: {CONF_THRESHOLD}")
# print(f"Using IOU threshold: {IOU_THRESHOLD}")


import os
import pathlib

import cv2
import numpy as np
from tqdm import tqdm
from yolov5 import YOLOv5

# Path configuration for Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Model configuration
model_path = r"C:\Users\iv061\Downloads\savola_fp_V1.1 1 (1).pt"
model = YOLOv5(model_path, device="cpu")  # or 'cuda' for GPU

# Detection parameters
CONF_THRESHOLD = 0.01  # Confidence threshold (0-1)
IOU_THRESHOLD = 0.45  # Intersection over Union threshold

# Input/Output paths
input_path = r"C:\Users\iv061\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\CITIVA"
output_path = r"C:\Users\iv061\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\CITIVA\OUTPUT"
logo_path = r".\logo_black.png"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load logo (optional)
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED) if os.path.exists(logo_path) else None


def add_watermark(frame, text="Demonstration Only", opacity=0.15):
    """Add rotated watermark text to frame."""
    overlay = frame.copy()
    text_size = 1.2
    thickness = 3
    color = (255, 255, 255)

    # Calculate text position
    text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, text_size, thickness)[0]
    text_x = (frame.shape[1] - text_width) // 2
    text_y = (frame.shape[0] + text_height) // 2

    # Create watermark
    watermark = np.zeros_like(frame, dtype=np.uint8)
    cv2.putText(watermark, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, text_size, color, thickness, cv2.LINE_AA)

    # Rotate watermark
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle=30, scale=1)
    rotated_watermark = cv2.warpAffine(watermark, rotation_matrix, (frame.shape[1], frame.shape[0]))

    # Blend watermark with frame
    cv2.addWeighted(rotated_watermark, opacity, overlay, 1 - opacity, 0, overlay)
    return overlay


def add_logo(frame, logo, logo_width=150):
    """Add logo to the frame with transparency support."""
    if logo is None:
        return frame

    # Resize logo maintaining aspect ratio
    h_logo, w_logo = logo.shape[:2]
    aspect_ratio = h_logo / w_logo
    new_height = int(logo_width * aspect_ratio)
    resized_logo = cv2.resize(logo, (logo_width, new_height))

    # Position logo
    x_offset = 10  # Left margin
    y_offset = 35  # Top margin

    # Handle transparent PNGs
    if resized_logo.shape[2] == 4:  # If logo has alpha channel
        alpha = resized_logo[:, :, 3] / 255.0
        for c in range(3):
            frame[y_offset : y_offset + new_height, x_offset : x_offset + logo_width, c] = (
                alpha * resized_logo[:, :, c]
                + (1 - alpha) * frame[y_offset : y_offset + new_height, x_offset : x_offset + logo_width, c]
            )
    else:  # For non-transparent images
        frame[y_offset : y_offset + new_height, x_offset : x_offset + logo_width] = resized_logo[:, :, :3]

    return frame


def process_frame(frame):
    """Process a single frame with object detection."""
    # Perform inference (note: some YOLOv5 versions don't accept conf/iou in predict)
    try:
        # Try with parameters first
        results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    except TypeError:
        # Fallback if parameters not accepted
        results = model.predict(frame)

    # Get predictions in pandas format
    predictions = results.pandas().xyxy[0]

    # Draw bounding boxes
    for _, pred in predictions.iterrows():
        if pred["confidence"] >= CONF_THRESHOLD:
            xmin, ymin, xmax, ymax = map(int, [pred["xmin"], pred["ymin"], pred["xmax"], pred["ymax"]])

            # Draw rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            # Add label with confidence
            # label = f"{pred['name']} {pred['confidence']:.2f}"
            # cv2.putText(frame, label, (xmin, ymin-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Add watermark and logo
    frame = add_watermark(frame)
    frame = add_logo(frame, logo)

    return frame


def process_video(input_video, output_folder):
    """Process video file frame by frame."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video file {input_video}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output video
    output_file = os.path.join(output_folder, "processed_+21_" + os.path.basename(input_video))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Process frames with progress bar
    with tqdm(total=frame_count, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame)
            out.write(processed_frame)
            pbar.update(1)

    # Clean up
    cap.release()
    out.release()
    print(f"Video processing complete. Saved to: {output_file}")


def process_images(image_folder, output_folder):
    """Process all images in a folder."""
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

    if not image_files:
        print("No images found in the specified directory")
        return

    # Create output subfolder
    output_img_folder = os.path.join(output_folder, "processed_images")
    os.makedirs(output_img_folder, exist_ok=True)

    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_folder, img_file)
        frame = cv2.imread(img_path)

        if frame is not None:
            processed_frame = process_frame(frame)
            output_path = os.path.join(output_img_folder, "processed_" + img_file)
            cv2.imwrite(output_path, processed_frame)

    print(f"Processed {len(image_files)} images. Saved to: {output_img_folder}")


# Main execution
if __name__ == "__main__":
    print(f"Using confidence threshold: {CONF_THRESHOLD}")
    print(f"Using IOU threshold: {IOU_THRESHOLD}")

    if os.path.isfile(input_path) and input_path.lower().endswith((".mp4", ".avi", ".mov")):
        process_video(input_path, output_path)
    elif os.path.isdir(input_path):
        process_images(input_path, output_path)
    else:
        print("Invalid input path. Please provide either a video file or a directory containing images")
