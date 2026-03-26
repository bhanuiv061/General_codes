# import cv2
# import os
# import numpy as np

# image_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Canon\engineering\image_json\images"
# label_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Canon\engineering\image_json\labels"

# for img_name in os.listdir(image_dir):
#     if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
#         continue

#     base = os.path.splitext(img_name)[0]
#     img_path = os.path.join(image_dir, img_name)
#     label_path = os.path.join(label_dir, base + ".txt")

#     if not os.path.exists(label_path):
#         continue

#     img = cv2.imread(img_path)
#     h, w = img.shape[:2]

#     with open(label_path, "r") as f:
#         lines = f.readlines()

#     for line in lines:
#         data = line.strip().split()
#         class_id = int(data[0])
#         points = data[1:]

#         polygon = []
#         for i in range(0, len(points), 2):
#             x = int(float(points[i]) * w)
#             y = int(float(points[i + 1]) * h)
#             polygon.append([x, y])

#         polygon = np.array(polygon, dtype=np.int32)

#         # draw polygon
#         cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

#         # put class id
#         x0, y0 = polygon[0]
#         cv2.putText(
#             img, f"ID:{class_id}", (x0, y0 - 5),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
#         )

#     cv2.imshow("YOLOv5 Segmentation Viewer", img)
#     key = cv2.waitKey(0)
#     if key == 27:  # ESC to exit
#         break

# cv2.destroyAllWindows()


import os

import cv2
import numpy as np

image_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Canon\engineering\image_json\images"
label_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Canon\engineering\image_json\labels"
output_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Canon\engineering\image_json\visualized"

os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    base = os.path.splitext(img_name)[0]
    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, base + ".txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path) as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        points = data[1:]

        polygon = []
        for i in range(0, len(points), 2):
            x = int(float(points[i]) * w)
            y = int(float(points[i + 1]) * h)
            polygon.append([x, y])

        polygon = np.array(polygon, dtype=np.int32)

        cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

    # save instead of imshow
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, img)
    print(f"✅ Saved: {save_path}")

print("\n🎉 Visualization complete!")
