import os
from collections import defaultdict

"""
========================================================
YOLO Label Class Analyzer (Safe & Robust)
========================================================

- Handles invalid lines like: overlap, comments, text
- Works for detection & segmentation labels
"""

# =========================
# Configuration
# =========================

labels_dir = r"C:\Users\admin\Downloads\crown (2)\crown\crown\images\txt_labels"
# =========================
# Helpers
# =========================


def is_int(val):
    try:
        int(val)
        return True
    except ValueError:
        return False


# =========================
# Analyze labels
# =========================

class_ids = set()
class_count = defaultdict(int)
skipped_lines = 0

for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue

    file_path = os.path.join(labels_dir, file)

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # 🚫 Skip non-YOLO lines
            if not is_int(parts[0]):
                skipped_lines += 1
                continue

            class_id = int(parts[0])
            class_ids.add(class_id)
            class_count[class_id] += 1

# =========================
# Print results
# =========================

print("✅ Unique class IDs found:", sorted(class_ids))
print("✅ Total number of unique classes:", len(class_ids))

print("\n📊 Instances per class:")
for cid in sorted(class_count):
    print(f"Class {cid}: {class_count[cid]} objects")

print(f"\n⚠️ Skipped invalid lines: {skipped_lines}")


# import os
# import shutil

# """
# ========================================================
# YOLO Label Validator & Separator
# ========================================================

# - Moves label files containing invalid lines
# - Invalid = first token is not an integer
# """

# # =========================
# # Configuration
# # =========================

# labels_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Crown\engineering\poc_\data_sets\overlap_dataset\labels"
# invalid_dir = os.path.join(labels_dir, "invalid_labels")

# os.makedirs(invalid_dir, exist_ok=True)

# # =========================
# # Helper
# # =========================

# def is_int(val):
#     try:
#         int(val)
#         return True
#     except ValueError:
#         return False

# # =========================
# # Scan & Separate
# # =========================

# moved_files = 0

# for file in os.listdir(labels_dir):
#     if not file.endswith(".txt"):
#         continue

#     file_path = os.path.join(labels_dir, file)

#     # Skip files already moved
#     if file_path.startswith(invalid_dir):
#         continue

#     has_invalid = False

#     with open(file_path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             parts = line.split()

#             # 🚫 Invalid YOLO line detected
#             if not is_int(parts[0]):
#                 has_invalid = True
#                 break

#     # 🚚 Move file if invalid
#     if has_invalid:
#         shutil.move(file_path, os.path.join(invalid_dir, file))
#         moved_files += 1
#         print(f"❌ Moved invalid label: {file}")

# # =========================
# # Summary
# # =========================

# print("\n=========================")
# print(f"✅ Separation complete")
# print(f"❌ Invalid files moved: {moved_files}")
# print(f"📁 Invalid folder: {invalid_dir}")
