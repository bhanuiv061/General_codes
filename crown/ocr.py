import re

import cv2
import easyocr

img_path = r"C:\Users\admin\Downloads\crown\1.png"
img = cv2.imread(img_path)

# Convert to grayscale (helps OCR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize EasyOCR reader
reader = easyocr.Reader(["en"], gpu=False)

# Run OCR
results = reader.readtext(gray, detail=1, paragraph=False)

# Extract UN number using regex
un_number = None
for bbox, text, conf in results:
    text_clean = text.replace(" ", "").upper()
    match = re.search(r"UN\d{4}", text_clean)
    if conf > 0.8:
        print(f"Detected Text: '{text}' with confidence {conf:.2f}")
        if match:
            un_number = match.group(0)
            break

print("Detected UN Number:", un_number)


# import os
# import cv2
# import easyocr
# import re
# import csv
# from collections import defaultdict

# # ================= CONFIG =================
# IMAGE_FOLDER = r"C:\Users\admin\Downloads\crown"
# OUTPUT_DIR = r"C:\Users\admin\Downloads\crown\crown\cam 1(L30766761)\output"
# ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
# CONFIDENCE_THRES = 0.4

# os.makedirs(ANNOTATED_DIR, exist_ok=True)

# # ================= PREDEFINED LABELS =================
# EXPECTED_LABEL_TEXTS = {
#     "LITHIUMIONBATTERIESFORBIDDENFORTRANSPORTABOARDPASSENGERAIRCRAFT": "LITHIUM ION BATTERIES",
#     "NONSPILLABLEBATTERY": "NONSPILLABLE BATTERY",
#     "FACILITIESMAINTENANCEUSE": "FACILITIES MAINTENANCE USE",
#     "FRAGILE": "FRAGILE",
#     "UP": "UP"
# }

# # ================= OCR INIT =================
# reader = easyocr.Reader(['en'], gpu=False)

# # ================= HELPERS =================
# def normalize_text(text):
#     return re.sub(r'[^A-Z0-9]', '', text.upper())

# def match_predefined_label(normalized_text):
#     for key, label_name in EXPECTED_LABEL_TEXTS.items():
#         if key in normalized_text:
#             return label_name
#     return None

# # ================= MAIN =================
# def run_ocr_on_folder(folder):
#     ocr_counts = defaultdict(int)

#     csv_path = os.path.join(OUTPUT_DIR, "ocr_results.csv")
#     txt_path = os.path.join(OUTPUT_DIR, "label_counts.txt")

#     with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Image", "Detected_Text"])

#         for file in os.listdir(folder):
#             if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
#                 continue

#             img_path = os.path.join(folder, file)
#             img = cv2.imread(img_path)

#             if img is None:
#                 print(f"❌ Failed to load {file}")
#                 continue

#             results = reader.readtext(img)
#             final_text = None

#             for bbox, text, conf in results:
#                 if conf < CONFIDENCE_THRES:
#                     continue

#                 norm_text = normalize_text(text)

#                 # 1️⃣ predefined label
#                 label = match_predefined_label(norm_text)
#                 if label:
#                     final_text = label
#                     break

#                 # 2️⃣ UN number fallback
#                 un_match = re.search(r"UN\d{4}", norm_text)
#                 if un_match:
#                     final_text = un_match.group(0)
#                     break

#                 # 3️⃣ raw OCR fallback
#                 if text.strip():
#                     final_text = text.strip()

#             if final_text:
#                 ocr_counts[final_text] += 1

#                 # draw on image
#                 if results:
#                     (x1, y1) = map(int, results[0][0][0])
#                     cv2.putText(
#                         img,
#                         final_text,
#                         (x1, max(30, y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.9,
#                         (0, 255, 0),
#                         2
#                     )

#                 print(f"✅ {file} → {final_text}")
#             else:
#                 print(f"⚠ {file} → No text detected")

#             writer.writerow([file, final_text if final_text else "NONE"])

#             out_img_path = os.path.join(ANNOTATED_DIR, file)
#             cv2.imwrite(out_img_path, img)

#     # save counts
#     with open(txt_path, "w") as f:
#         for label, cnt in ocr_counts.items():
#             f.write(f"{label}: {cnt}\n")

#     return ocr_counts

# # ================= RUN =================
# if __name__ == "__main__":
#     counts = run_ocr_on_folder(IMAGE_FOLDER)

#     print("\n📊 FINAL COUNTS")
#     for k, v in counts.items():
#         print(f" - {k}: {v}")
