import os
import sys

from PIL import Image

folder_path = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\TATA_V3_HIMALAYA\image_data_\masked_rcnn_op_+particle\train\images"


def help():
    print("""
========================================================
Image Resolution Analyzer (Lowest & Highest Resolution)
========================================================

This script scans a folder containing images and finds:
• The image with the LOWEST resolution
• The image with the HIGHEST resolution

Resolution is calculated using image width × height (pixel area).

--------------------------------------------------------
WORKING FLOW
--------------------------------------------------------

1. Folder Selection
   - A directory path containing images is provided.
   - The script reads all files inside this folder.

2. Image Validation
   - Each file is attempted to be opened using PIL (Pillow).
   - Non-image files or corrupted files are skipped safely.

3. Resolution Extraction
   - For each valid image:
       • Width and height are read
       • Pixel area is calculated (width × height)
       • Image name and resolution data are stored

4. Sorting by Resolution
   - All images are sorted based on pixel area.
   - This allows easy identification of:
       • Smallest image (lowest resolution)
       • Largest image (highest resolution)

5. Result Display
   - Prints:
       • File name
       • Image resolution (WxH)
       • Total pixel area
   - Shows both lowest and highest resolution images.

--------------------------------------------------------
USAGE
--------------------------------------------------------

python image_resolution_check.py

(No command-line arguments required.
Folder path is defined inside the script.)

--------------------------------------------------------
OUTPUT EXAMPLE
--------------------------------------------------------

Lowest Resolution Image:
File: img_012.jpg, Size: 640x480 (Area: 307200)

Highest Resolution Image:
File: img_098.jpg, Size: 4096x2160 (Area: 8847360)

--------------------------------------------------------
NOTES
--------------------------------------------------------

• Supports all image formats readable by PIL.
• Non-image files are automatically ignored.
• Useful for:
    - Dataset validation
    - Detecting inconsistent image sizes
    - Preprocessing before training

========================================================
""")
    sys.exit(0)


image_sizes = []

# Loop through all files
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Check if it is an image
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            area = width * height
            image_sizes.append((file_name, width, height, area))
    except:
        pass  # skip non-images

# Sort by area
image_sizes.sort(key=lambda x: x[3])

# Lowest resolution
lowest = image_sizes[0]
# Highest resolution
highest = image_sizes[-1]

print("Lowest Resolution Image:")
print(f"File: {lowest[0]}, Size: {lowest[1]}x{lowest[2]} (Area: {lowest[3]})")

print("\nHighest Resolution Image:")
print(f"File: {highest[0]}, Size: {highest[1]}x{highest[2]} (Area: {highest[3]})")
