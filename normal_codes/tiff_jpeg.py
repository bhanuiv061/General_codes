import os

from PIL import Image

# Input and output folders"
input_folder = r"C:\Users\admin\Downloads\122025 Citi\122025 Citi"
output_folder = r"C:\Users\admin\Downloads\122025 Citi\122025 Citi\jpg"

os.makedirs(output_folder, exist_ok=True)

supported_formats = (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")

for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(supported_formats):
        input_path = os.path.join(input_folder, file_name)

        # Output filename with .jpg extension
        output_name = os.path.splitext(file_name)[0] + ".jpg"
        output_path = os.path.join(output_folder, output_name)

        try:
            with Image.open(input_path) as img:
                # Convert to RGB (JPEG does not support alpha or grayscale properly)
                img = img.convert("RGB")

                # Save as JPEG without changing resolution
                img.save(output_path, "JPEG", quality=95)

                print(f"Converted: {file_name}")

        except Exception as e:
            print(f"Failed: {file_name} | Error: {e}")
