import os

from PIL import Image

# Folder path
folder = r"c:\imagevision projects\tata\29sept_\CUTMARK_\cam 7 (DA5392480)"
output_folder = os.path.join(folder, "rotated")

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Rotation angle (change as needed: 90, 180, 270, etc.)
angle = 90

# Supported image formats
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for filename in os.listdir(folder):
    if filename.lower().endswith(valid_extensions):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path)
            rotated = img.rotate(angle, expand=True)  # expand=True keeps full image
            save_path = os.path.join(output_folder, filename)
            rotated.save(save_path)
            print(f"Rotated and saved: {save_path}")
        except Exception as e:
            print(f"❌ Could not process {filename}: {e}")
