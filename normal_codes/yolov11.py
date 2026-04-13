# import os
# import cv2
# from ultralytics import YOLO

# # Define paths
# output_folder = r"D:\bhanu\tata_consumers_gbottles\TATA_OPTOMECH\data\test\out"
# input_folder = r"D:\bhanu\tata_consumers_gbottles\TATA_OPTOMECH\data\test"
# model_path = r"C:\Users\iv061\Downloads\tata_optomechv11.pt"

# # Create the output directory if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Load the trained model
# model = YOLO(model_path)

# # Run detection on images or video, with a confidence threshold (conf)
# results = model.predict(input_folder, conf=0.25)

# # Loop through the results (one result per image/video)
# for result in results:
#     # Get the original image (numpy array)
#     img = result.orig_img  # This is the original image in numpy format

#     # Extract the bounding boxes and labels from the results
#     boxes = result.boxes  # Get the boxes (bounding boxes)

#     # Draw bounding boxes and labels on the image using OpenCV
#     for box in boxes:
#         # Get the box coordinates and label (accessing tensor values correctly)
#         x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list and then unpack

#         # Convert to integers
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers

#         label = int(box.cls.item())  # Convert tensor to integer for the class label
#         confidence = float(box.conf.item())  # Convert tensor to float for confidence score

#         # Draw the bounding box on the image (rectangle)
#         color = (0, 255, 0)  # Green color for the box
#         thickness = 2  # Thickness of the bounding box
#         img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

#         # Optionally, draw the label and confidence score
#         text = f"Class: {label}, Confidence: {confidence:.2f}"  # Format text with label and confidence
#         img = cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         cv2.imshow('Detection Result', img)

#     # Save the processed image with bounding boxes using cv2.imwrite
#     output_image_path = os.path.join(output_folder, f"image_{result.frame}.jpg")  # Output image path
#     cv2.imwrite(output_image_path, img)  # Save the image using OpenCV

#     # Optional: Show the image with bounding boxes (you can comment this line if not needed)
#     # cv2.imshow('Detection Result', img)
#     # cv2.waitKey(0)  # Press any key to close the displayed window

# # Optional: Close any OpenCV windows (comment this out if you don't use imshow)
# # cv2.destroyAllWindows()


import os

import cv2
from ultralytics import YOLO

# Define paths
output_folder = r"D:\bhanu\tata_consumers_gbottles\TATA_OPTOMECH\data\test\outv_pavan"
input_folder = r"D:\bhanu\tata_consumers_gbottles\TATA_OPTOMECH\data\test"
# model_path = r"C:\Users\iv061\Downloads\tata_optomech_v11v2.pt"
model_path = r"C:\Users\iv061\Downloads\best 7.pt"
# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the trained model
model = YOLO(model_path)

# Run detection on images or video, with a confidence threshold (conf)
results = model.predict(input_folder, conf=0.25)

# Loop through the results (one result per image/video)
for result in results:
    # Get the original image (numpy array)
    img = result.orig_img  # This is the original image in numpy format

    # Extract the bounding boxes and labels from the results
    boxes = result.boxes  # Get the boxes (bounding boxes)

    # Draw bounding boxes and labels on the image using OpenCV
    for box in boxes:
        # Get the box coordinates and label (accessing tensor values correctly)
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list and then unpack

        # Convert to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers

        label = int(box.cls.item())  # Convert tensor to integer for the class label
        confidence = float(box.conf.item())  # Convert tensor to float for confidence score

        # Draw the bounding box on the image (rectangle)
        color = (0, 255, 0)  # Green color for the box
        thickness = 2  # Thickness of the bounding box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Optionally, draw the label and confidence score
        text = f"Class: {label}, Confidence: {confidence:.2f}"  # Format text with label and confidence
        img = cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Extract the file name from the path and create output path
    input_image_path = result.path  # Path to the input image (from result)
    input_image_name = os.path.basename(input_image_path)  # Extract file name from the path
    output_image_path = os.path.join(output_folder, input_image_name)  # Use the same file name for output

    # Save the processed image with bounding boxes using cv2.imwrite
    cv2.imwrite(output_image_path, img)  # Save the image using OpenCV

    # Optional: Show the image with bounding boxes (you can comment this line if not needed)
    # cv2.imshow('Detection Result', img)
    # cv2.waitKey(0)  # Press any key to close the displayed window

# Optional: Close any OpenCV windows (comment this out if you don't use imshow)
# cv2.destroyAllWindows()
