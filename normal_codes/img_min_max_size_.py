import os

from PIL import Image

folder_path = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\TATA_V3_HIMALAYA\image_data_\masked_rcnn_op_+particle\train\images"

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
