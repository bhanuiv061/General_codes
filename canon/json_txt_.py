import json
import os

from PIL import Image

# ---------- paths ----------
json_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\personal_projects\iron_\data\labels"
image_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\personal_projects\iron_\data\images"
output_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\personal_projects\iron_\data\labels_yolo"

os.makedirs(output_dir, exist_ok=True)

# ---------- class mapping ----------
class_map = {"ironing": 0, "iron_board": 1, "iron_box": 2}

# ---------- process json ----------
for json_file in os.listdir(json_dir):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(json_dir, json_file)
    base_name = os.path.splitext(json_file)[0]

    # detect image
    image_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        temp = os.path.join(image_dir, base_name + ext)
        if os.path.exists(temp):
            image_path = temp
            break

    if image_path is None:
        print(f"⚠ Image missing for {json_file}")
        continue

    # read image size
    img = Image.open(image_path)
    img_w, img_h = img.size

    # load json
    with open(json_path) as f:
        data = json.load(f)

    yolo_lines = []

    for shape in data.get("shapes", []):
        label = shape["label"]

        if label not in class_map:
            continue

        class_id = class_map[label]

        if shape["shape_type"] != "polygon":
            continue

        points = shape["points"]

        seg_points = []

        for x, y in points:
            x = x / img_w
            y = y / img_h
            seg_points.append(f"{x:.6f} {y:.6f}")

        line = f"{class_id} " + " ".join(seg_points)

        yolo_lines.append(line)

    # save txt
    txt_path = os.path.join(output_dir, base_name + ".txt")

    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

    print(f"✅ {json_file} → {base_name}.txt")

print("\n🎯 Conversion complete (YOLOv8 Segmentation)")
