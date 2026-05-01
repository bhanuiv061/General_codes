import os

import cv2
import numpy as np

# --- Load image ---
image_path = r"D:\bhanu\tata_zaid_sunmission\All_defects_\glass\300ml\surface\gp_bis_317898.jpg"  # change this to your image filename
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check image load
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# --- Separate channels if RGBA ---
if image.shape[2] == 4:
    b, g, r, a = cv2.split(image)
else:
    b, g, r = cv2.split(image)
    a = None

# --- Define color bounds (BGR order for OpenCV) ---
lower = np.array([24, 45, 37], dtype=np.uint8)  # B, G, R
upper = np.array([106, 127, 116], dtype=np.uint8)  # B, G, R

# --- Create mask ---
mask = cv2.inRange(image[:, :, :3], lower, upper)

# --- Apply mask ---
masked = cv2.bitwise_and(image, image, mask=mask)

# --- Create output folder if not exists ---
output_folder = "masked"
os.makedirs(output_folder, exist_ok=True)

# --- Build output path ---
filename = os.path.basename(image_path)
output_path = os.path.join(output_folder, f"masked_{filename}")

# --- Save masked image ---
cv2.imwrite(output_path, masked)
print(f"Masked image saved at: {output_path}")

# --- Optional display ---
# cv2.imshow("Masked Output", masked)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
