import os

"""
========================================================
YOLO Label Class Mapper (Safe & Robust)
========================================================

- Changes ONLY class 0 -> 12
- Skips invalid / non-YOLO lines (e.g. 'overlap')
- Works with detection & segmentation labels
"""

# =========================
# Configuration
# =========================

labels_dir = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Crown\engineering\poc_\data_sets\overlap_dataset\labels"

OLD_CLASS_ID = 0
NEW_CLASS_ID = 12
CREATE_BACKUP = True

# =========================
# Helper
# =========================


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


# =========================
# Process files
# =========================

for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue

    file_path = os.path.join(labels_dir, file)

    # Backup once
    if CREATE_BACKUP:
        backup_path = file_path + ".bak"
        if not os.path.exists(backup_path):
            with open(file_path) as f:
                with open(backup_path, "w") as bf:
                    bf.write(f.read())

    updated_lines = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # 🚫 Skip non-YOLO lines like "overlap"
            if not is_int(parts[0]):
                print(f"⚠️ Skipping invalid line in {file}: {line}")
                continue

            class_id = int(parts[0])

            # 🔁 Change only target class
            if class_id == OLD_CLASS_ID:
                parts[0] = str(NEW_CLASS_ID)

            updated_lines.append(" ".join(parts))

    # Write cleaned file
    with open(file_path, "w") as f:
        f.write("\n".join(updated_lines) + "\n")

print(f"✅ Class {OLD_CLASS_ID} → {NEW_CLASS_ID} completed safely")
