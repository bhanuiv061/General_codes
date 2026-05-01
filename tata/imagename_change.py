import os

# Paths to your folders
img_folder = r"C:\imagevision projects\tata\24sept_augumented\eng_h_c_24_09\300ml_eng_2_h_c"
label_folder = r"C:\imagevision projects\tata\24sept_augumented\eng_h_c_24_09\labels"

# List all images and labels
images = [f for f in os.listdir(img_folder) if f.lower().endswith((".jpg", ".png"))]
labels = [f for f in os.listdir(label_folder) if f.lower().endswith(".txt")]

# Make a set of label base names
label_bases = {os.path.splitext(l)[0] for l in labels}

for img in images:
    base_name, ext = os.path.splitext(img)

    if base_name in label_bases:
        # Build new names
        new_img = base_name + "_c1" + ext
        new_label = base_name + "_c1.txt"

        # Rename
        os.rename(os.path.join(img_folder, img), os.path.join(img_folder, new_img))
        os.rename(os.path.join(label_folder, base_name + ".txt"), os.path.join(label_folder, new_label))
    else:
        print(f"❌ Label missing for {img}")

# Also check for labels without images
image_bases = {os.path.splitext(i)[0] for i in images}
for lbl in labels:
    base_lbl = os.path.splitext(lbl)[0]
    if base_lbl not in image_bases:
        print(f"❌ Image missing for {lbl}")
