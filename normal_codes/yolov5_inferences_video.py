# import cv2
# import torch
# import numpy as np
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
# # Load the YOLOv5 model
# from yolov5 import YOLOv5

# # Load model (replace with your correct path)
# model = YOLOv5("C:/Users/iv061/Downloads/sharp_bister.pt", device="cpu")  # or device="cuda" for GPU

# model.eval()  # Now it's a proper model and you can call eval()


# # Function to add watermark
# def add_watermark(frame, text="Demonstration Only", opacity=0.15):
#     """Adds a semi-transparent watermark diagonally across the video."""
#     overlay = frame.copy()
#     text_size = 1.2  # Slightly smaller watermark
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

#     cv2.addWeighted(rotated_watermark, opacity, frame, 1 - opacity, 0, frame)
#     return frame

# # Function to add logo
# def add_logo(frame, logo, logo_width=150):
#     """Adds the resized logo to the top-left corner of the black bar."""
#     if logo is None:
#         return frame  # Skip if logo is not found

#     # Resize the logo to the specified width, keeping aspect ratio
#     h_logo, w_logo = logo.shape[:2]
#     aspect_ratio = h_logo / w_logo
#     new_height = int(logo_width * aspect_ratio)
#     resized_logo = cv2.resize(logo, (logo_width, new_height))

#     x_offset = 10  # Left margin
#     y_offset = 35  # Adjusted to fit small black bar

#     # Ensure logo has alpha (transparency)
#     if resized_logo.shape[2] == 4:
#         alpha = resized_logo[:, :, 3] / 255.0  # Extract alpha channel
#         for c in range(3):
#             frame[y_offset:y_offset + new_height, x_offset:x_offset + logo_width, c] = (
#                 alpha * resized_logo[:, :, c] + (1 - alpha) * frame[y_offset:y_offset + new_height, x_offset:x_offset + logo_width, c]
#             )
#     else:  # If no transparency, just overlay
#         frame[y_offset:y_offset + new_height, x_offset:x_offset + logo_width] = resized_logo[:, :, :3]

#     return frame

# # Function to create a black bar with text
# def create_black_bar(width, bar_height, text=None, font_scale=0.5, font_thickness=1, color=(0, 255, 0)):
#     """Creates a black bar with optional text at the top."""
#     black_bar = np.zeros((bar_height, width, 3), dtype=np.uint8)
#     if text:
#         cv2.putText(black_bar, text, (10, 22), cv2.FONT_HERSHEY_TRIPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
#     return black_bar

# # Open the input video
# input_video = r"D:\bhanu\sharp_blisters\train\images\output_video.mp4"  # Replace with your input video file
# cap = cv2.VideoCapture(input_video)

# # Get video properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# # Output video writer setup
# output_video = "output_video_with_watermark.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
# out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# # Load logo (optional)
# logo = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)  # Replace with your logo path

# # Process each frame of the video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform inference with YOLOv5 model
#     img = [frame]  # Make the frame compatible with YOLOv5
#     results = model(img)  # Run inference

#     # Visualize the results on the frame
#     frame = results.render()[0]  # This will draw the boxes, labels, and confidences on the frame

#     # Add watermark
#     frame = add_watermark(frame)

#     # Add logo
#     frame = add_logo(frame, logo)

#     # Write the frame to the output video
#     out.write(frame)

# # Release the video capture and writer
# cap.release()
# out.release()

# print("Inference complete. Video saved as:", output_video)


##working_yolov5 infernces code


# import torch
# from yolov5 import YOLOv5  # Import the YOLOv5 class from the YOLOv5 package
# import cv2
# import numpy as np
# import pathlib

# # Switch to WindowsPath if needed (based on your platform)
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # Load the YOLOv5 model
# model = YOLOv5(r"C:/Users/iv061/Downloads/sharp_bister.pt", device="cpu")  # or use 'cuda' for GPU

# # Open the video for inference
# input_video = r"D:\bhanu\sharp_blisters\train\images\output_video.mp4"  # Replace with your input video file
# cap = cv2.VideoCapture(input_video)

# # Get video properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# # Output video writer setup
# output_video = "output_video_with_watermark.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
# out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# # Load logo (optional)
# logo = cv2.imread('.\logo_black.png', cv2.IMREAD_UNCHANGED)  # Replace with your logo path

# # Function to add watermark
# def add_watermark(frame, text="Demonstration Only", opacity=0.15):
#     overlay = frame.copy()  # Ensure we're working with a writable copy
#     text_size = 1.2  # Slightly smaller watermark
#     thickness = 3
#     color = (255, 255, 255)

#     text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, text_size, thickness)[0]
#     text_x = (frame.shape[1] - text_width) // 2
#     text_y = (frame.shape[0] + text_height) // 2

#     watermark = np.zeros_like(frame, dtype=np.uint8)
#     cv2.putText(watermark, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, text_size, color, thickness, cv2.LINE_AA)

#     # Rotate watermark
#     center = (frame.shape[1] // 2, frame.shape[0] // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle=30, scale=1)
#     rotated_watermark = cv2.warpAffine(watermark, rotation_matrix, (frame.shape[1], frame.shape[0]))

#     # Apply the watermark with appropriate blending
#     cv2.addWeighted(rotated_watermark, opacity, overlay, 1 - opacity, 0, overlay)
#     return overlay

# # Function to add logo
# def add_logo(frame, logo, logo_width=150):
#     if logo is None:
#         return frame

#     # Resize the logo to the specified width, keeping aspect ratio
#     h_logo, w_logo = logo.shape[:2]
#     aspect_ratio = h_logo / w_logo
#     new_height = int(logo_width * aspect_ratio)
#     resized_logo = cv2.resize(logo, (logo_width, new_height))

#     x_offset = 10  # Left margin
#     y_offset = 35  # Adjusted to fit small black bar

#     if resized_logo.shape[2] == 4:  # If logo has transparency
#         alpha = resized_logo[:, :, 3] / 255.0  # Extract alpha channel
#         for c in range(3):  # Blend the logo with the frame based on transparency
#             frame[y_offset:y_offset + new_height, x_offset:x_offset + logo_width, c] = (
#                 alpha * resized_logo[:, :, c] + (1 - alpha) * frame[y_offset:y_offset + new_height, x_offset:x_offset + logo_width, c]
#             )
#     else:  # If no transparency, just overlay the logo
#         frame[y_offset:y_offset + new_height, x_offset:x_offset + logo_width] = resized_logo[:, :, :3]

#     return frame

# # Process each frame of the video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform inference with YOLOv5 model using the 'predict' method
#     results = model.predict(frame)  # Use the 'predict' method for inference

#     # Render the results (boxes, labels, etc.)
#     frame = results.render()[0]  # This will draw the boxes, labels, and confidences on the frame

#     # Add watermark
#     frame = add_watermark(frame)

#     # Add logo
#     frame = add_logo(frame, logo)

#     # Write the frame to the output video
#     out.write(frame)

# # Release the video capture and writer
# cap.release()
# out.release()

# print("Inference complete. Video saved as:", output_video)


import pathlib

import cv2
import numpy as np
from yolov5 import YOLOv5  # Import the YOLOv5 class from the YOLOv5 package

# Switch to WindowsPath if needed (based on your platform)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the YOLOv5 model
model = YOLOv5(r"C:\Users\iv061\Downloads\sharpv2_500.pt", device="cpu")  # or use 'cuda' for GPU

# Open the video for inference
input_video = (
    r"D:\bhanu\sharp_blisters\output_videos\cracks\input_video\output_video.mp4"  # Replace with your input video file
)
cap = cv2.VideoCapture(input_video)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer setup
output_video = r"D:\bhanu\sharp_blisters\output_videos\cracks\cracks\output_video_with_watermark.mp4"  # Update the output path here
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Load logo (optional)
logo = cv2.imread(".\logo_black.png", cv2.IMREAD_UNCHANGED)  # Replace with your logo path


# Function to add watermark
def add_watermark(frame, text="Demonstration Only", opacity=0.15):
    overlay = frame.copy()  # Ensure we're working with a writable copy
    text_size = 1.2  # Slightly smaller watermark
    thickness = 3
    color = (255, 255, 255)

    text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, text_size, thickness)[0]
    text_x = (frame.shape[1] - text_width) // 2
    text_y = (frame.shape[0] + text_height) // 2

    watermark = np.zeros_like(frame, dtype=np.uint8)
    cv2.putText(watermark, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, text_size, color, thickness, cv2.LINE_AA)

    # Rotate watermark
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle=30, scale=1)
    rotated_watermark = cv2.warpAffine(watermark, rotation_matrix, (frame.shape[1], frame.shape[0]))

    # Apply the watermark with appropriate blending
    cv2.addWeighted(rotated_watermark, opacity, overlay, 1 - opacity, 0, overlay)
    return overlay


# Function to add logo
def add_logo(frame, logo, logo_width=150):
    if logo is None:
        return frame

    # Resize the logo to the specified width, keeping aspect ratio
    h_logo, w_logo = logo.shape[:2]
    aspect_ratio = h_logo / w_logo
    new_height = int(logo_width * aspect_ratio)
    resized_logo = cv2.resize(logo, (logo_width, new_height))

    x_offset = 10  # Left margin
    y_offset = 35  # Adjusted to fit small black bar

    if resized_logo.shape[2] == 4:  # If logo has transparency
        alpha = resized_logo[:, :, 3] / 255.0  # Extract alpha channel
        for c in range(3):  # Blend the logo with the frame based on transparency
            frame[y_offset : y_offset + new_height, x_offset : x_offset + logo_width, c] = (
                alpha * resized_logo[:, :, c]
                + (1 - alpha) * frame[y_offset : y_offset + new_height, x_offset : x_offset + logo_width, c]
            )
    else:  # If no transparency, just overlay the logo
        frame[y_offset : y_offset + new_height, x_offset : x_offset + logo_width] = resized_logo[:, :, :3]

    return frame


# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference with YOLOv5 model using the 'predict' method
    results = model.predict(frame)  # Use the 'predict' method for inference

    # Get the predictions in pandas dataframe format
    predictions = results.pandas().xyxy[0]

    # Draw bounding boxes on the frame
    for _, pred in predictions.iterrows():
        xmin, ymin, xmax, ymax = int(pred["xmin"]), int(pred["ymin"]), int(pred["xmax"]), int(pred["ymax"])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red color (0, 0, 255)

    # Add watermark
    frame = add_watermark(frame)

    # Add logo
    frame = add_logo(frame, logo)

    # Write the frame to the output video
    out.write(frame)

# Release the video capture and writer
cap.release()
out.release()

print("Inference complete. Video saved as:", output_video)
