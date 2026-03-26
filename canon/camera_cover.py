# import cv2
# import numpy as np
# import os
# from datetime import datetime

# # ================= CONFIG ================= #
# # INPUT_SOURCE = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\Pictures\Camera Roll\WIN_20260211_14_17_28_Pro.jpg"   # image | video | rtsp | 0
# INPUT_SOURCE = 0
# SAVE_ROOT = "camera_events"
# SAVE_ORIGINAL = os.path.join(SAVE_ROOT, "original")
# SAVE_BW = os.path.join(SAVE_ROOT, "bw")
# SAVE_EDGES = os.path.join(SAVE_ROOT, "edges")

# for p in [SAVE_ORIGINAL, SAVE_BW, SAVE_EDGES]:
#     os.makedirs(p, exist_ok=True)

# BLACK_RATIO_THRESH = 0.40
# TEXTURE_THRESH = 12
# EDGE_DENSITY_THRESH = 0.015
# PERSIST_FRAMES = 25
# # ========================================= #


# def analyze_frame(frame):
#     h, w = frame.shape[:2]
#     total_pixels = h * w

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (7, 7), 0)

#     bw = cv2.adaptiveThreshold(
#         gray, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         21, 5
#     )

#     black_ratio = np.sum(bw == 0) / total_pixels

#     texture = cv2.Laplacian(gray, cv2.CV_64F).var()

#     edges = cv2.Canny(gray, 30, 100)
#     edge_density = np.sum(edges > 0) / total_pixels

#     covered = (
#         black_ratio > BLACK_RATIO_THRESH and
#         texture < TEXTURE_THRESH and
#         edge_density < EDGE_DENSITY_THRESH
#     )

#     return covered, black_ratio, texture, edge_density, bw, edges


# def save_event(frame, bw, edges):
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     cv2.imwrite(f"{SAVE_ORIGINAL}/frame_{ts}.jpg", frame)
#     cv2.imwrite(f"{SAVE_BW}/bw_{ts}.jpg", bw)
#     cv2.imwrite(f"{SAVE_EDGES}/edges_{ts}.jpg", edges)
#     print(f"[SAVED] Event at {ts}")


# # ================= IMAGE MODE ================= #
# if isinstance(INPUT_SOURCE, str) and INPUT_SOURCE.lower().endswith((".jpg", ".png", ".jpeg")):
#     frame = cv2.imread(INPUT_SOURCE)
#     if frame is None:
#         raise ValueError("Image not found")

#     covered, black, texture, edges_d, bw, edges = analyze_frame(frame)

#     status = "CAMERA COVERED" if covered else "CAMERA NORMAL"
#     color = (0, 0, 255) if covered else (0, 255, 0)

#     display = frame.copy()
#     cv2.putText(display, status, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

#     cv2.putText(display, f"Black: {black*100:.1f}%",
#                 (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(display, f"Texture: {texture:.1f}",
#                 (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(display, f"Edges: {edges_d:.4f}",
#                 (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(display, "Press S to SAVE | Q / ESC to EXIT",
#                 (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                 (0, 255, 255), 2)

#     cv2.imshow("Image Analysis", display)
#     cv2.imshow("Black & White", bw)
#     cv2.imshow("Edges", edges)

#     while True:
#         key = cv2.waitKey(0) & 0xFF
#         if key in [27, ord('q')]:
#             break
#         if key == ord('s'):
#             save_event(frame, bw, edges)

#     cv2.destroyAllWindows()
#     exit()


# # ================= VIDEO MODE ================= #
# cap = cv2.VideoCapture(INPUT_SOURCE)
# if not cap.isOpened():
#     raise RuntimeError("Cannot open video source")

# covered_counter = 0
# freeze = False

# print("Running video mode | S = Save | Q / ESC = Exit")

# while True:
#     if not freeze:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         covered, black, texture, edges_d, bw, edges = analyze_frame(frame)

#         if covered:
#             covered_counter += 1
#         else:
#             covered_counter = 0

#         if covered_counter >= PERSIST_FRAMES:
#             freeze = True
#             status = "ALERT: CAMERA COVERED"
#             color = (0, 0, 255)
#         else:
#             status = "Camera Normal"
#             color = (0, 255, 0)

#     display = frame.copy()
#     cv2.putText(display, status, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

#     cv2.putText(display, f"Black: {black*100:.1f}%",
#                 (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(display, f"Texture: {texture:.1f}",
#                 (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(display, f"Edges: {edges_d:.4f}",
#                 (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     if freeze:
#         cv2.putText(display, "Press S to SAVE | Q / ESC to EXIT",
#                     (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                     (0, 255, 255), 2)

#     cv2.imshow("CCTV Obstruction Detection", display)
#     cv2.imshow("Black & White", bw)
#     cv2.imshow("Edges", edges)

#     key = cv2.waitKey(0 if freeze else 1) & 0xFF

#     if key in [27, ord('q')]:
#         break

#     if freeze and key == ord('s'):
#         save_event(frame, bw, edges)
#         freeze = False
#         covered_counter = 0

# cap.release()
# cv2.destroyAllWindows()


# import cv2

# # -------- CONFIG --------
# IMAGE_PATH = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\Pictures\Camera Roll\WIN_20260211_14_17_28_Pro.jpg"  # your image path
# # ------------------------

# img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
# if img is None:
#     raise ValueError("Image not found")

# display = img.copy()

# def mouse_callback(event, x, y, flags, param):
#     global display

#     if event == cv2.EVENT_MOUSEMOVE:
#         b, g, r = img[y, x]

#         avg_intensity = int((int(b) + int(g) + int(r)) / 3)

#         display = img.copy()

#         text1 = f"X:{x} Y:{y}"
#         text2 = f"B:{b} G:{g} R:{r}"
#         text3 = f"Avg Intensity: {avg_intensity}"

#         cv2.putText(display, text1, (20, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         cv2.putText(display, text2, (20, 65),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         cv2.putText(display, text3, (20, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         # Draw cursor point
#         cv2.circle(display, (x, y), 3, (0, 0, 255), -1)

# cv2.namedWindow("Pixel Intensity Inspector", cv2.WINDOW_NORMAL)
# cv2.setMouseCallback("Pixel Intensity Inspector", mouse_callback)

# print("Move mouse over image to inspect pixel intensity")
# print("Press Q or ESC to exit")

# while True:
#     cv2.imshow("Pixel Intensity Inspector", display)
#     key = cv2.waitKey(1) & 0xFF
#     if key == 27 or key == ord('q'):
#         break

# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import os
# from skimage.feature import local_binary_pattern

# # ===================== CONFIG ===================== #
# INPUT_SOURCE = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\Pictures\Camera Roll\WIN_20260211_14_17_28_Pro.jpg"         # 0 = webcam | "video.mp4" | "image.jpg"

# # Intensity ranges (COLOR intensity = (B+G+R)/3)
# SLIGHTLY_DARK_RANGE = (180, 220)
# DARK_RANGE = (221, 255)

# # Thresholds
# MIN_AREA_PERCENT = 25.0          # minimum % of image
# MAX_VARIANCE = 15.0              # low texture
# MIN_LBP_UNIFORM_RATIO = 0.85     # flat surface
# PERSISTENCE_FRAMES = 10          # consecutive frames

# # LBP parameters
# LBP_RADIUS = 1
# LBP_POINTS = 8 * LBP_RADIUS

# # Save directory
# SAVE_DIR = "masked_outputs"
# os.makedirs(SAVE_DIR, exist_ok=True)
# # ================================================= #

# persistent_counter = 0


# def analyze_frame(frame):
#     global persistent_counter

#     h, w = frame.shape[:2]
#     total_pixels = h * w

#     # -------- COLOR INTENSITY (NO GRAYSCALE) --------
#     intensity = np.mean(frame.astype(np.float32), axis=2).astype(np.uint8)

#     # -------- INTENSITY MASKS --------
#     mask_slight = cv2.inRange(
#         intensity, SLIGHTLY_DARK_RANGE[0], SLIGHTLY_DARK_RANGE[1]
#     )

#     mask_dark = cv2.inRange(
#         intensity, DARK_RANGE[0], DARK_RANGE[1]
#     )

#     mask = cv2.bitwise_or(mask_slight, mask_dark)

#     masked_pixels = cv2.countNonZero(mask)
#     area_percent = (masked_pixels / total_pixels) * 100

#     if masked_pixels == 0:
#         persistent_counter = 0
#         return False, mask, area_percent, 0, 0

#     # -------- TEXTURE: VARIANCE --------
#     masked_intensity = intensity[mask > 0]
#     variance = np.var(masked_intensity)

#     # -------- TEXTURE: LBP --------
#     lbp = local_binary_pattern(
#         intensity, LBP_POINTS, LBP_RADIUS, method="uniform"
#     )
#     lbp_masked = lbp[mask > 0]
#     uniform_ratio = np.sum(lbp_masked <= 2) / len(lbp_masked)

#     # -------- DECISION --------
#     covered_now = (
#         area_percent >= MIN_AREA_PERCENT and
#         variance <= MAX_VARIANCE and
#         uniform_ratio >= MIN_LBP_UNIFORM_RATIO
#     )

#     # -------- PERSISTENCE --------
#     if covered_now:
#         persistent_counter += 1
#     else:
#         persistent_counter = 0

#     covered_final = persistent_counter >= PERSISTENCE_FRAMES

#     return covered_final, mask, area_percent, variance, uniform_ratio


# # ===================== IMAGE MODE ===================== #
# if isinstance(INPUT_SOURCE, str) and INPUT_SOURCE.lower().endswith(
#         (".jpg", ".jpeg", ".png")):

#     frame = cv2.imread(INPUT_SOURCE)
#     if frame is None:
#         raise ValueError("Image not found")

#     covered, mask, area, var, lbp_ratio = analyze_frame(frame)

#     overlay = frame.copy()
#     overlay[mask > 0] = (0, 0, 255)

#     result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

#     status = "COVERED" if covered else "NOT COVERED"
#     color = (0, 0, 255) if covered else (0, 255, 0)

#     cv2.putText(result, status, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

#     cv2.putText(result, f"Area: {area:.1f}%",
#                 (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(result, f"Variance: {var:.1f}",
#                 (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(result, f"LBP Uniform: {lbp_ratio:.2f}",
#                 (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.imshow("Final Result", result)
#     cv2.imshow("Masked Region", mask)

#     print("Press Q or ESC to exit")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     exit()


# # ===================== VIDEO MODE ===================== #
# cap = cv2.VideoCapture(INPUT_SOURCE)
# if not cap.isOpened():
#     raise RuntimeError("Cannot open video source")

# print("Running... Press Q or ESC to exit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     covered, mask, area, var, lbp_ratio = analyze_frame(frame)

#     overlay = frame.copy()
#     overlay[mask > 0] = (0, 0, 255)
#     result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

#     status = "ALERT: CAMERA COVERED" if covered else "Camera Normal"
#     color = (0, 0, 255) if covered else (0, 255, 0)

#     cv2.putText(result, status, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

#     cv2.putText(result, f"Area: {area:.1f}%",
#                 (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(result, f"Variance: {var:.1f}",
#                 (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(result, f"LBP Uniform: {lbp_ratio:.2f}",
#                 (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.imshow("CCTV Tamper Detection", result)
#     cv2.imshow("Intensity Mask", mask)

#     if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from skimage.feature import local_binary_pattern

# # ================= CONFIG ================= #
# INPUT_SOURCE = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\Pictures\Camera Roll\WIN_20260211_14_17_28_Pro.jpg"

# # Dark intensity ranges (grayscale)
# DARK_RANGE = (0, 50)
# SLIGHT_DARK_RANGE = (51, 100)

# ALERT_AREA_PERCENT = 20.0

# # Texture thresholds
# MAX_VARIANCE = 20.0
# MIN_LBP_UNIFORM_RATIO = 0.85

# # Uniform frame protection
# UNIFORM_INTENSITY_THRESHOLD = 8

# # Persistence
# PERSISTENCE_FRAMES = 10

# # LBP params
# LBP_RADIUS = 1
# LBP_POINTS = 8 * LBP_RADIUS
# # ========================================= #

# persistent_counter = 0


# def analyze_frame(frame):
#     global persistent_counter

#     h, w = frame.shape[:2]
#     total_pixels = h * w

#     # -------- GRAYSCALE CONVERSION --------
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # -------- UNIFORM FRAME CHECK --------
#     intensity_spread = int(gray.max()) - int(gray.min())
#     frame_is_uniform = intensity_spread <= UNIFORM_INTENSITY_THRESHOLD

#     # -------- DARK AREA MASK --------
#     dark_mask = cv2.inRange(gray, DARK_RANGE[0], DARK_RANGE[1])
#     slight_dark_mask = cv2.inRange(gray, SLIGHT_DARK_RANGE[0], SLIGHT_DARK_RANGE[1])
#     combined_mask = cv2.bitwise_or(dark_mask, slight_dark_mask)

#     dark_pixels = cv2.countNonZero(combined_mask)
#     dark_area_percent = (dark_pixels / total_pixels) * 100

#     if dark_pixels == 0:
#         persistent_counter = 0
#         return False, combined_mask, dark_area_percent, 0, 0, intensity_spread

#     # -------- TEXTURE: VARIANCE --------
#     masked_gray = gray[combined_mask > 0]
#     variance = np.var(masked_gray)

#     # -------- TEXTURE: LBP --------
#     lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
#     lbp_masked = lbp[combined_mask > 0]
#     uniform_ratio = np.sum(lbp_masked <= 2) / len(lbp_masked)

#     # -------- CURRENT FRAME DECISION --------
#     covered_now = (
#         dark_area_percent >= ALERT_AREA_PERCENT and
#         variance <= MAX_VARIANCE and
#         uniform_ratio >= MIN_LBP_UNIFORM_RATIO and
#         not frame_is_uniform
#     )

#     # -------- PERSISTENCE --------
#     if covered_now:
#         persistent_counter += 1
#     else:
#         persistent_counter = 0

#     covered_final = persistent_counter >= PERSISTENCE_FRAMES

#     return covered_final, combined_mask, dark_area_percent, variance, uniform_ratio, intensity_spread


# # ================= IMAGE MODE ================= #
# if isinstance(INPUT_SOURCE, str) and INPUT_SOURCE.lower().endswith((".jpg", ".jpeg", ".png")):

#     frame = cv2.imread(INPUT_SOURCE)
#     if frame is None:
#         raise ValueError("Image not found")

#     covered, mask, area, var, lbp_ratio, spread = analyze_frame(frame)

#     overlay = frame.copy()
#     overlay[mask > 0] = (0, 0, 255)
#     result = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

#     status = "ALERT: CAMERA COVERED" if covered else "Camera Normal"
#     color = (0, 0, 255) if covered else (0, 255, 0)

#     cv2.putText(result, status, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

#     cv2.putText(result, f"Dark Area: {area:.1f}%",
#                 (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

#     cv2.putText(result, f"Variance: {var:.1f}",
#                 (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(result, f"LBP Uniform: {lbp_ratio:.2f}",
#                 (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(result, f"Intensity Spread: {spread}",
#                 (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.imshow("Final Result", result)
#     cv2.imshow("Dark Mask", mask)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     exit()


# # ================= VIDEO MODE ================= #
# cap = cv2.VideoCapture(INPUT_SOURCE)
# if not cap.isOpened():
#     raise RuntimeError("Cannot open video source")

# print("Running... Press Q or ESC to exit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     covered, mask, area, var, lbp_ratio, spread = analyze_frame(frame)

#     overlay = frame.copy()
#     overlay[mask > 0] = (0, 0, 255)
#     result = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

#     status = "ALERT: CAMERA COVERED" if covered else "Camera Normal"
#     color = (0, 0, 255) if covered else (0, 255, 0)

#     cv2.putText(result, status, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

#     cv2.putText(result, f"Dark Area: {area:.1f}%",
#                 (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

#     cv2.putText(result, f"Variance: {var:.1f}",
#                 (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(result, f"LBP Uniform: {lbp_ratio:.2f}",
#                 (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(result, f"Intensity Spread: {spread}",
#                 (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.imshow("CCTV Obstruction Detection", result)
#     cv2.imshow("Dark Mask", mask)

#     if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # ================= CONFIG =================
# SOURCE = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\Pictures\Camera Roll\WIN_20260211_14_17_28_Pro.jpg"   # change if needed
# DARK_THRESHOLD = 20.0           # percent
# DARK_RANGE = (0, 100)           # grayscale dark range
# # =========================================


# def analyze_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     total_pixels = gray.size
#     dark_mask = cv2.inRange(gray, DARK_RANGE[0], DARK_RANGE[1])
#     dark_pixels = cv2.countNonZero(dark_mask)

#     dark_percent = (dark_pixels / total_pixels) * 100

#     # 🚨 ONLY RULE
#     covered = dark_percent >= DARK_THRESHOLD

#     return covered, dark_percent, dark_mask


# def visualize(frame, dark_mask, covered, dark_percent):
#     overlay = frame.copy()
#     overlay[dark_mask > 0] = (0, 0, 255)
#     result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

#     status = "🚨 CAMERA COVERED" if covered else "Camera Normal"
#     color = (0, 0, 255) if covered else (0, 255, 0)

#     cv2.putText(result, status, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

#     cv2.putText(result, f"Dark Area: {dark_percent:.1f}%",
#                 (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     return result


# # ================= RUN =================
# frame = cv2.imread(SOURCE)
# if frame is None:
#     raise RuntimeError("Image not found")

# covered, dark_percent, dark_mask = analyze_frame(frame)
# output = visualize(frame, dark_mask, covered, dark_percent)

# cv2.imshow("Final Result", output)
# cv2.imshow("Dark Mask", dark_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # ================= CONFIG =================
# VIDEO_SOURCE = 0          # 0 = webcam, or "video.mp4", or RTSP URL
# DARK_RANGE = (0, 100)     # dark intensity range
# DARK_THRESHOLD = 20.0     # percent
# # =========================================


# def analyze_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     total_pixels = gray.size
#     dark_mask = cv2.inRange(gray, DARK_RANGE[0], DARK_RANGE[1])
#     dark_pixels = cv2.countNonZero(dark_mask)

#     dark_percent = (dark_pixels / total_pixels) * 100

#     # 🚨 ONLY RULE
#     covered = dark_percent >= DARK_THRESHOLD

#     return covered, dark_percent, dark_mask


# def visualize(frame, dark_mask, covered, dark_percent):
#     overlay = frame.copy()
#     overlay[dark_mask > 0] = (0, 0, 255)
#     result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

#     status = "🚨 CAMERA COVERED" if covered else "Camera Normal"
#     color = (0, 0, 255) if covered else (0, 255, 0)

#     cv2.putText(result, status, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

#     cv2.putText(result, f"Dark Area: {dark_percent:.1f}%",
#                 (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     return result


# # ================= RUN =================
# cap = cv2.VideoCapture(VIDEO_SOURCE)

# if not cap.isOpened():
#     raise RuntimeError("Cannot open video source")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     covered, dark_percent, dark_mask = analyze_frame(frame)
#     output = visualize(frame, dark_mask, covered, dark_percent)

#     cv2.imshow("Camera Tamper Check", output)
#     cv2.imshow("Dark Mask", dark_mask)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # ================= CONFIG =================
# VIDEO_SOURCE =r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\Pictures\Camera Roll\WIN_20260211_14_17_28_Pro.jpg"          # 0 = webcam, or "video.mp4", or RTSP URL
# DARK_RANGE = (0, 100)     # dark intensity range
# DARK_THRESHOLD = 20.0     # percent
# # =========================================


# def analyze_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     total_pixels = gray.size
#     dark_mask = cv2.inRange(gray, DARK_RANGE[0], DARK_RANGE[1])
#     dark_pixels = cv2.countNonZero(dark_mask)

#     dark_percent = (dark_pixels / total_pixels) * 100

#     # 🚨 ONLY RULE
#     covered = dark_percent >= DARK_THRESHOLD

#     return covered, dark_percent, dark_mask


# def visualize(frame, dark_mask, covered, dark_percent):
#     overlay = frame.copy()
#     overlay[dark_mask > 0] = (0, 0, 255)
#     result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

#     status = "🚨 CAMERA COVERED" if covered else "Camera Normal"
#     color = (0, 0, 255) if covered else (0, 255, 0)

#     cv2.putText(result, status, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

#     cv2.putText(result, f"Dark Area: {dark_percent:.1f}%",
#                 (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     return result


# # ================= RUN =================
# cap = cv2.VideoCapture(VIDEO_SOURCE)

# if not cap.isOpened():
#     raise RuntimeError("Cannot open video source")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     covered, dark_percent, dark_mask = analyze_frame(frame)
#     output = visualize(frame, dark_mask, covered, dark_percent)

#     cv2.imshow("Camera Tamper Check", output)
#     cv2.imshow("Dark Mask", dark_mask)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
#         break

# cap.release()
# cv2.destroyAllWindows()


import time

import cv2
import numpy as np

# ================= CONFIG =================
VIDEO_SOURCE = 0  # 0 = webcam | "video.mp4" | RTSP URL
DARK_RANGE = (0, 100)  # Dark intensity range
DARK_THRESHOLD = 20.0  # Percent
FREEZE_TIMEOUT = 1.0  # Seconds (FPS = 0 condition)
# =========================================


def analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------- Screen Off Check --------
    min_val = int(gray.min())
    max_val = int(gray.max())
    screen_off = min_val == max_val

    # -------- Dark Area Check --------
    dark_mask = cv2.inRange(gray, DARK_RANGE[0], DARK_RANGE[1])
    dark_pixels = cv2.countNonZero(dark_mask)
    total_pixels = gray.size
    dark_percent = (dark_pixels / total_pixels) * 100

    return screen_off, dark_percent, dark_mask


def visualize(frame, dark_mask, status, dark_percent, fps):
    vis = frame.copy()

    if status == "CAMERA COVERED":
        overlay = vis.copy()
        overlay[dark_mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    color_map = {
        "VIDEO FREEZE": (0, 0, 255),
        "SCREEN OFF": (0, 165, 255),
        "CAMERA COVERED": (0, 0, 255),
        "NORMAL": (0, 255, 0),
    }

    color = color_map[status]

    cv2.putText(vis, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.putText(vis, f"Dark Area: {dark_percent:.1f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(vis, f"FPS: {fps:.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    return vis


# ================= RUN =================
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video source")

last_frame_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    now = time.time()

    # -------- FPS / Freeze Detection --------
    if ret:
        fps = 1.0 / max(now - last_frame_time, 1e-6)
        last_frame_time = now
    else:
        fps = 0.0

    # -------- Freeze Rule (ONLY FPS = 0) --------
    if (now - last_frame_time) > FREEZE_TIMEOUT:
        status = "VIDEO FREEZE"
        dark_percent = 0.0
        dark_mask = np.zeros((1, 1), dtype=np.uint8)

    else:
        screen_off, dark_percent, dark_mask = analyze_frame(frame)

        if screen_off:
            status = "SCREEN OFF"

        elif dark_percent >= DARK_THRESHOLD:
            status = "CAMERA COVERED"

        else:
            status = "NORMAL"

    output = visualize(frame, dark_mask, status, dark_percent, fps)

    cv2.imshow("Camera Monitor", output)

    # If image → wait for key | if video → ESC to exit
    if VIDEO_SOURCE != 0 and isinstance(VIDEO_SOURCE, str):
        if cv2.waitKey(0) != -1:
            break
    else:
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
