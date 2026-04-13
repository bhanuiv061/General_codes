# import argparse
# import csv
# import os
# import platform
# import sys
# from pathlib import Path
# from datetime import datetime
# import pathlib
# import cv2
# import torch
# import numpy as np
# from sort.sort import Sort
# import time
# start_time = time.time()

# # Fix Windows path issue
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
# from ultralytics.utils.plotting import Annotator
# from models.common import DetectMultiBackend
# from utils.general import (
#     LOGGER,
#     Profile,
#     check_file,
#     check_img_size,
#     check_imshow,
#     check_requirements,
#     increment_path,
#     non_max_suppression,
#     print_args,
#     scale_boxes,
# )
# from utils.torch_utils import select_device, smart_inference_mode


# @smart_inference_mode()
# def run(
#     weights=ROOT / "yolov5s.pt",
#     source=ROOT / "data/images",
#     data=ROOT / "data/coco128.yaml",
#     imgsz=(640, 640),
#     conf_thres=0.25,
#     iou_thres=0.45,
#     max_det=1000,
#     device="",
#     view_img=False,
#     nosave=False,
#     project=ROOT / "runs/detect",
#     name="exp",
#     exist_ok=False,
#     half=False,
#     dnn=False,
#     vid_stride=1,
# ):

#     # ---------------- WATERMARK ----------------
#     def add_diagonal_watermark(frame, text="DEMO WATERMARK", opacity=0.18):
#         overlay = frame.copy()
#         h, w = frame.shape[:2]

#         font_scale = min(w, h) / 900
#         thickness = int(font_scale * 2)
#         color = (255, 255, 255)

#         text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
#         x = (w - text_size[0]) // 2
#         y = (h + text_size[1]) // 2

#         watermark_layer = np.zeros_like(frame, dtype=np.uint8)
#         cv2.putText(watermark_layer, text, (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

#         center = (w // 2, h // 2)
#         matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
#         rotated = cv2.warpAffine(watermark_layer, matrix, (w, h))

#         cv2.addWeighted(rotated, opacity, overlay, 1 - opacity, 0, overlay)
#         return overlay
#     def print_progress(frame_idx, counts):
#         elapsed = time.time() - start_time
#         fps = frame_idx / elapsed if elapsed > 0 else 0
#         total = sum(counts.values())

#         count_str = " | ".join([f"{k}:{v}" for k, v in counts.items()])
#         msg = f"\r🎯 Frame: {frame_idx} | FPS: {fps:.2f} | Total: {total} | {count_str}"
#         print(msg, end="", flush=True)
#     # ---------------- LOGO ----------------
#     def add_logo_top_left(frame, logo_path="logo_white 1.png", width=120):
#         if not os.path.exists(logo_path):
#             return frame

#         logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
#         if logo is None:
#             return frame

#         h_logo, w_logo = logo.shape[:2]
#         aspect = h_logo / w_logo
#         new_h = int(width * aspect)
#         logo = cv2.resize(logo, (width, new_h))

#         x_offset, y_offset = 10, 60

#         if logo.shape[2] == 4:
#             alpha = logo[:, :, 3] / 255.0
#             for c in range(3):
#                 frame[y_offset:y_offset+new_h, x_offset:x_offset+width, c] = (
#                     alpha * logo[:, :, c] +
#                     (1 - alpha) * frame[y_offset:y_offset+new_h, x_offset:x_offset+width, c]
#                 )
#         else:
#             frame[y_offset:y_offset+new_h, x_offset:x_offset+width] = logo[:, :, :3]

#         return frame

#     # ---------------- CLASS SETUP ----------------
#     def init_class_counter(model_names):
#         return {name: 0 for name in model_names.values()}

#     def get_class_color(class_id):
#         np.random.seed(int(class_id) + 42)
#         return tuple(int(x) for x in np.random.randint(0, 255, size=3))

#     def log_counts(frame_num, counts, csv_file):
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(csv_file, mode="a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([timestamp, frame_num] + list(counts.values()))

#     def is_crossing_line(prev_y, curr_y):
#         return prev_y < line_y <= curr_y or prev_y > line_y >= curr_y

#     # ---------------- MODEL ----------------
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = check_img_size(imgsz, s=stride)
#     print(f"📐 Using inference image size: {imgsz}")


#     counts = init_class_counter(names)

#     source = str(source)
#     save_img = not nosave and not source.endswith(".txt")
#     is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
#     webcam = source.isnumeric() and not is_file

#     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt) if webcam else LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

#     tracker = Sort()
#     prev_centroids = {}
#     line_y = 400
#     counted_ids = set()

#     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     video_path = str(save_dir / "output.mp4")
#     video_writer = None


#     csv_file = os.path.join(str(save_dir), f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
#     with open(csv_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["timestamp", "frame"] + list(counts.keys()))
#     # ---------------- LOOP ----------------
#     import time
#     start_time = time.time()

#     try:
#         for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

#             im = torch.from_numpy(im).to(device).float() / 255.0
#             if len(im.shape) == 3:
#                 im = im[None]

#             pred = model(im)
#             pred = non_max_suppression(pred, conf_thres, iou_thres)

#             im0 = im0s[0].copy() if webcam else im0s.copy()
#             annotator = Annotator(im0, line_width=2)

#             # 🎥 Initialize video writer on first frame
#             if video_writer is None:
#                 h, w = im0.shape[:2]
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
#                 video_writer = cv2.VideoWriter(
#                     video_path,
#                     cv2.VideoWriter_fourcc(*'mp4v'),
#                     fps,
#                     (w, h)
#                 )

#             detections = []
#             if len(pred[0]):
#                 pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()
#                 for *xyxy, conf, cls in pred[0]:
#                     x1, y1, x2, y2 = map(int, xyxy)
#                     detections.append([x1, y1, x2, y2, conf.item()])

#             tracks = tracker.update(np.array(detections)) if detections else np.empty((0, 5))
#             current_centroids = {}

#             for x1, y1, x2, y2, track_id in tracks.astype(int):
#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#                 detected_class = "unknown"
#                 detected_cls_id = -1

#                 for *xyxy, conf, cls in pred[0]:
#                     px1, py1, px2, py2 = map(int, xyxy)
#                     if abs(x1 - px1) < 10 and abs(y1 - py1) < 10:
#                         detected_class = names[int(cls)] if int(cls) in names else "unknown"
#                         detected_cls_id = int(cls)
#                         break

#                 current_centroids[track_id] = (cx, cy, detected_class)
#                 color = get_class_color(detected_cls_id)

#                 annotator.box_label([x1, y1, x2, y2], detected_class, color=color)
#                 cv2.circle(im0, (cx, cy), 4, color, -1)

#             # -------- COUNTING --------
#             offset = 15  # tolerance zone around line

#             for obj_id, (cx, cy, cls) in current_centroids.items():
#                 if obj_id in counted_ids:
#                     continue  # already counted

#                 # If object is close enough to the line zone
#                 if abs(cy - line_y) <= offset:
#                     if cls in counts:
#                         counts[cls] += 1
#                         counted_ids.add(obj_id)


#             prev_centroids = current_centroids.copy()

#             cv2.line(im0, (0, line_y), (im0.shape[1], line_y), (0, 255, 255), 2)

#             # -------- DISPLAY COUNTS --------
#             y_offset = 220
#             for cls_name, count in counts.items():
#                 cv2.putText(im0, f"{cls_name}: {count}", (20, y_offset),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
#                 y_offset += 35

#             im0 = add_logo_top_left(im0)
#             im0 = add_diagonal_watermark(im0)

#             log_counts(frame_idx, counts, csv_file)

#             # 🎥 Always write frame
#             if video_writer is not None:
#                 video_writer.write(im0)

#             # -------- LIVE TERMINAL PROGRESS --------
#             elapsed = time.time() - start_time
#             fps_live = frame_idx / elapsed if elapsed > 0 else 0
#             total = sum(counts.values())
#             count_str = " | ".join([f"{k}:{v}" for k, v in counts.items()])
#             print(f"\r🎯 Frame: {frame_idx} | FPS: {fps_live:.2f} | Total: {total} | {count_str}",
#                   end="", flush=True)

#             cv2.imshow("Counting", im0)
#             if cv2.waitKey(1) == ord("q"):
#                 break

#     except KeyboardInterrupt:
#         print("\n🛑 Interrupted by user. Saving video safely...")

#     except Exception as e:
#         print(f"\n❌ Error occurred: {e}")

#     finally:
#         if video_writer is not None:
#             video_writer.release()
#             print(f"\n🎥 Video saved at: {video_path}")

#         cv2.destroyAllWindows()
#         print("✅ Program finished safely.")


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt")
#     parser.add_argument("--source", type=str, default=ROOT / "data/images")
#     #parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
#     parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")

#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--view-img", action="store_true")
#     parser.add_argument("--nosave", action="store_true")
#     parser.add_argument("--project", default=ROOT / "runs/detect")
#     parser.add_argument("--name", default="exp")
#     parser.add_argument("--exist-ok", action="store_true")
#     parser.add_argument(
#     "--imgsz", "--img", "--img-size",
#     nargs="+",
#     type=int,
#     default=[640],
#     help="inference size h,w (single value for square, or h w)"
# )

#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
#     print_args(vars(opt))
#     return opt


# def main(opt):
#     check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
#     run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)


# import argparse
# import os
# import sys
# from pathlib import Path
# import pathlib
# import cv2
# import torch
# import numpy as np
# from sort.sort import Sort
# import time

# # Fix Windows path issue
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# from utils.dataloaders import LoadImages, LoadStreams
# from ultralytics.utils.plotting import Annotator
# from models.common import DetectMultiBackend
# from utils.general import check_img_size, non_max_suppression, scale_boxes, increment_path
# from utils.torch_utils import select_device, smart_inference_mode


# @smart_inference_mode()
# def run(weights, source, imgsz, conf_thres, iou_thres, device, project, name, filter_class=None):

#     def get_class_color(class_id):
#         np.random.seed(int(class_id) + 42)
#         return tuple(int(x) for x in np.random.randint(0, 255, size=3))

#     def add_diagonal_watermark(frame, text="DEMO WATERMARK", opacity=0.18):
#         overlay = frame.copy()
#         h, w = frame.shape[:2]
#         font_scale = min(w, h) / 900
#         thickness = int(font_scale * 2)
#         layer = np.zeros_like(frame, dtype=np.uint8)
#         cv2.putText(layer, text, (w//4, h//2),
#                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
#         M = cv2.getRotationMatrix2D((w//2, h//2), 30, 1.0)
#         rotated = cv2.warpAffine(layer, M, (w, h))
#         return cv2.addWeighted(rotated, opacity, overlay, 1-opacity, 0)

#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names
#     imgsz = check_img_size(imgsz, s=stride)

#     dataset = LoadStreams(source, img_size=imgsz, stride=stride) if source.isnumeric() else LoadImages(source, img_size=imgsz, stride=stride)

#     tracker = Sort()
#     counts = {name: 0 for name in names.values()}
#     # If filtering only one class, keep counter only for that
#     if filter_class:
#         counts = {filter_class: 0}


#     # 🔥 MEMORY SYSTEM
#     track_class_memory = {}
#     track_last_seen = {}
#     track_side_memory = {}
#     counted_ids = set()

#     BUFFER_FRAMES = 300  # ~10 sec
#     line_y = 400
#     offset = 5

#     save_dir = increment_path(Path(project) / name)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     video_path = str(save_dir / "output.mp4")
#     video_writer = None

#     start_time = time.time()

#     try:
#         for frame_idx, data in enumerate(dataset):

#             # 🛑 Skip corrupted frames
#             if data is None:
#                 continue

#             path, im, im0s, vid_cap, s = data
#             if im is None or im0s is None:
#                 continue

#             im = torch.from_numpy(im).to(device).float() / 255.0
#             if len(im.shape) == 3:
#                 im = im[None]

#             pred = model(im)
#             pred = non_max_suppression(pred, conf_thres, iou_thres)

#             im0 = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#             annotator = Annotator(im0, line_width=2)

#             if video_writer is None:
#                 h, w = im0.shape[:2]
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
#                 video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

#             detections, det_boxes, det_classes = [], [], []

#             if len(pred[0]):
#                 pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()
#                 for *xyxy, conf, cls in pred[0]:
#                     class_name = names[int(cls)]

#                     # 🎯 Skip unwanted classes
#                     if filter_class and class_name != filter_class:
#                         continue

#                     x1, y1, x2, y2 = map(int, xyxy)
#                     detections.append([x1, y1, x2, y2, conf.item()])
#                     det_boxes.append([x1, y1, x2, y2])
#                     det_classes.append(int(cls))


#             tracks = tracker.update(np.array(detections)) if detections else np.empty((0, 5))
#             current_centroids = {}

#             for x1, y1, x2, y2, track_id in tracks.astype(int):
#                 cx, cy = (x1+x2)//2, (y1+y2)//2
#                 best_iou, cls_name = 0, "unknown"

#                 for i, box in enumerate(det_boxes):
#                     xx1, yy1 = max(x1, box[0]), max(y1, box[1])
#                     xx2, yy2 = min(x2, box[2]), min(y2, box[3])
#                     inter = max(0, xx2-xx1) * max(0, yy2-yy1)
#                     area1 = (x2-x1)*(y2-y1)
#                     area2 = (box[2]-box[0])*(box[3]-box[1])
#                     iou = inter/(area1+area2-inter+1e-6)
#                     if iou > best_iou:
#                         best_iou = iou
#                         cls_name = names[det_classes[i]]

#                         # Skip if not selected class
#                         if filter_class and cls_name != filter_class:
#                             continue


#                 if best_iou > 0.3:
#                     track_class_memory[track_id] = cls_name
#                 elif track_id in track_class_memory:
#                     cls_name = track_class_memory[track_id]

#                 track_last_seen[track_id] = frame_idx
#                 current_centroids[track_id] = (cx, cy, cls_name)

#                 color = get_class_color(list(names.values()).index(cls_name)) if cls_name in names.values() else (200,200,200)
#                 # annotator.box_label([x1, y1, x2, y2], f"{cls_name} ID:{track_id}", color=color)
#                 annotator.box_label([x1, y1, x2, y2], f"{cls_name}", color=color)

#                 #cv2.circle(im0, (cx, cy), 4, color, -1)

#             # 🔢 ROBUST LINE CROSSING COUNT
#             for obj_id, (cx, cy, cls) in current_centroids.items():

#                 if cls == "unknown":
#                     continue

#                 if cy < line_y - offset:
#                     current_side = "above"
#                 elif cy > line_y + offset:
#                     current_side = "below"
#                 else:
#                     current_side = "on_line"

#                 if obj_id not in track_side_memory:
#                     track_side_memory[obj_id] = current_side
#                     continue

#                 prev_side = track_side_memory[obj_id]

#                 if prev_side != current_side and current_side != "on_line":
#                     if obj_id not in counted_ids:
#                         counts[cls] += 1
#                         counted_ids.add(obj_id)

#                 track_side_memory[obj_id] = current_side

#             # 🧹 Remove stale IDs
#             for tid in list(track_last_seen.keys()):
#                 if frame_idx - track_last_seen[tid] > BUFFER_FRAMES:
#                     track_last_seen.pop(tid, None)
#                     track_class_memory.pop(tid, None)
#                     track_side_memory.pop(tid, None)
#                     counted_ids.discard(tid)

#             cv2.line(im0, (0, line_y), (im0.shape[1], line_y), (0,255,255), 2)

#             y = 220
#             for cls, val in counts.items():
#                 cv2.putText(im0, f"{cls}: {val}", (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
#                 y += 35

#             im0 = add_diagonal_watermark(im0)

#             if im0 is not None and im0.size != 0:
#                 video_writer.write(im0)

#             elapsed = time.time() - start_time
#             fps_live = frame_idx/elapsed if elapsed > 0 else 0
#             total = sum(counts.values())
#             print(f"\r🎯 Frame:{frame_idx} FPS:{fps_live:.1f} Total:{total} " + " | ".join([f"{k}:{v}" for k,v in counts.items()]), end="")

#             cv2.imshow("Counting", im0)
#             if cv2.waitKey(1) == ord("q"):
#                 break

#     finally:
#         if video_writer:
#             video_writer.release()
#         cv2.destroyAllWindows()
#         print("\n✅ Video saved:", video_path)

# def about():
#     """
# YOLOv5 + SORT Object Tracking and Line Crossing Counter
# -------------------------------------------------------

# This script performs real-time object detection, tracking, and counting
# based on objects crossing a virtual horizontal line in a video or live stream.

# It uses:
# - YOLOv5 for object detection
# - SORT tracker for persistent object IDs
# - A memory buffer system to prevent ID switching issues
# - Line-crossing logic to count objects only once

# -------------------------------------------------------
# FEATURES
# -------------------------------------------------------
# • Detects objects using a custom YOLOv5 model
# • Tracks objects with stable IDs using SORT
# • Prevents ID flickering with class memory buffer
# • Counts objects only when they cross a defined line
# • Avoids double counting using counted ID memory
# • Automatically removes old tracks after buffer time
# • Option to filter and count only ONE specific class
# • Saves output video with annotations and watermark
# • Displays live FPS and count statistics

# -------------------------------------------------------
# COUNTING LOGIC
# -------------------------------------------------------
# An object is counted when:
# 1. It has a valid tracked ID
# 2. It moves from one side of the line to the other
# 3. It has not been counted before
# 4. It belongs to the selected class (if filter enabled)

# The system tracks which side of the line each object was previously on.
# If the side changes (above → below or below → above), the counter increases.

# -------------------------------------------------------
# MEMORY SYSTEM (Anti-ID Switching)
# -------------------------------------------------------
# track_class_memory : Remembers last known class for each track ID
# track_last_seen    : Stores last frame index where object appeared
# track_side_memory  : Stores last known position relative to line
# counted_ids        : Prevents double counting same ID

# BUFFER_FRAMES:
# Objects are kept in memory for ~10 seconds (300 frames).
# If an object disappears briefly and reappears, it keeps its old class ID.

# -------------------------------------------------------
# CLASS FILTERING
# -------------------------------------------------------
# Use --filter-class to track and count only one object type.

# Example:
#     --filter-class Car

# This ignores all other detected classes.

# -------------------------------------------------------
# COMMAND LINE ARGUMENTS
# -------------------------------------------------------
# --weights        Path to YOLOv5 model (.pt)
# --source         Video file, folder, stream URL, or webcam index
# --imgsz          Inference image size (default 640)
# --conf-thres     Detection confidence threshold
# --iou-thres      NMS IoU threshold
# --device         CUDA device or CPU
# --project        Output folder
# --name           Run name
# --filter-class   (Optional) Class name to count only

# -------------------------------------------------------
# OUTPUT
# -------------------------------------------------------
# • Annotated video saved to: runs/count/<name>/output.mp4
# • Live window showing detections and counting
# • Terminal shows FPS and total counts in real time

# -------------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------------
# Count ALL classes:
#     python det_tracking_count.py --weights model.pt --source video.mp4

# Count ONLY cars:
#     python det_tracking_count.py --weights model.pt --source video.mp4 --filter-class Car

# -------------------------------------------------------
# """
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", type=str, required=True)
#     parser.add_argument("--source", type=str, required=True)
#     parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/count")
#     parser.add_argument("--name", default="exp")
#     parser.add_argument(
#     "--filter-class",
#     type=str,
#     default=None,
#     help="Name of class to track/count (example: Car)"
# )

#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
#     return opt


# if __name__ == "__main__":
#     opt = parse_opt()
#     # Access and print the docstring properly
#     print(about.__doc__)
#     run(**vars(opt))


# import argparse
# import os
# import sys
# from pathlib import Path
# import pathlib
# import cv2
# import torch
# import numpy as np
# from sort.sort import Sort
# import time

# # Fix Windows path issue
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# from utils.dataloaders import LoadImages, LoadStreams
# from ultralytics.utils.plotting import Annotator
# from models.common import DetectMultiBackend
# from utils.general import check_img_size, non_max_suppression, scale_boxes, increment_path
# from utils.torch_utils import select_device, smart_inference_mode


# @smart_inference_mode()
# def run(weights, source, imgsz, conf_thres, iou_thres, device, project, name, filter_class=None):

#     def get_class_color(class_id):
#         np.random.seed(int(class_id) + 42)
#         return tuple(int(x) for x in np.random.randint(0, 255, size=3))

#     def add_diagonal_watermark(frame, text="Demo_purpose", opacity=0.15):
#         overlay = frame.copy()
#         h, w = frame.shape[:2]
#         font_scale = min(w, h) / 900
#         thickness = int(font_scale * 2)
#         layer = np.zeros_like(frame, dtype=np.uint8)
#         cv2.putText(layer, text, (w//4, h//2),
#                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
#         M = cv2.getRotationMatrix2D((w//2, h//2), 30, 1.0)
#         rotated = cv2.warpAffine(layer, M, (w, h))
#         return cv2.addWeighted(rotated, opacity, overlay, 1-opacity, 0)

#     def add_logo_top_left(frame, logo_path="logo_white 1.png", width=120):
#         if not os.path.exists(logo_path):
#             return frame
#         logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
#         if logo is None:
#             return frame
#         h_logo, w_logo = logo.shape[:2]
#         new_h = int(width * (h_logo / w_logo))
#         logo = cv2.resize(logo, (width, new_h))
#         x_offset, y_offset = 10, 10
#         if logo.shape[2] == 4:
#             alpha = logo[:, :, 3] / 255.0
#             for c in range(3):
#                 frame[y_offset:y_offset+new_h, x_offset:x_offset+width, c] = (
#                     alpha * logo[:, :, c] +
#                     (1 - alpha) * frame[y_offset:y_offset+new_h, x_offset:x_offset+width, c]
#                 )
#         else:
#             frame[y_offset:y_offset+new_h, x_offset:x_offset+width] = logo[:, :, :3]
#         return frame

#     # ---------------- MODEL ----------------
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names
#     imgsz = check_img_size(imgsz, s=stride)

#     dataset = LoadStreams(source, img_size=imgsz, stride=stride) if source.isnumeric() else LoadImages(source, img_size=imgsz, stride=stride)
#     tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#     counts_in = {name: 0 for name in names.values()}
#     counts_out = {name: 0 for name in names.values()}

#     if filter_class:
#         counts_in = {filter_class: 0}
#         counts_out = {filter_class: 0}

#     DISPLAY_CLASSES = list(counts_in.keys())
#     FONT = cv2.FONT_HERSHEY_SIMPLEX

#     track_class_memory, track_last_seen, track_side_memory = {}, {}, {}
#     counted_ids = set()

#     BUFFER_FRAMES = 300
#     line_y = 200
#     offset = 5

#     save_dir = increment_path(Path(project) / name)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     video_path = str(save_dir / "output.mp4")
#     video_writer = None
#     start_time = time.time()

#     try:
#         for frame_idx, data in enumerate(dataset):

#             if data is None:
#                 continue

#             path, im, im0s, vid_cap, s = data
#             if im is None or im0s is None:
#                 continue

#             im = torch.from_numpy(im).to(device).float() / 255.0
#             if len(im.shape) == 3:
#                 im = im[None]

#             pred = model(im)
#             pred = non_max_suppression(pred, conf_thres, iou_thres)

#             im0 = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#             annotator = Annotator(im0, line_width=2)

#             # 🎥 INIT VIDEO WRITER ONCE (NO RESIZE EVER)
#             if video_writer is None:
#                 h, w = im0.shape[:2]
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                 if fps <= 0 or fps > 120:
#                     fps = 30
#                 video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

#             detections, det_boxes, det_classes = [], [], []

#             if len(pred[0]):
#                 pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()
#                 for *xyxy, conf, cls in pred[0]:
#                     class_name = names[int(cls)]
#                     if filter_class and class_name != filter_class:
#                         continue
#                     x1, y1, x2, y2 = map(int, xyxy)
#                     detections.append([x1, y1, x2, y2, conf.item()])
#                     det_boxes.append([x1, y1, x2, y2])
#                     det_classes.append(int(cls))

#             tracks = tracker.update(np.array(detections)) if detections else np.empty((0, 5))
#             current_centroids = {}

#             for x1, y1, x2, y2, track_id in tracks.astype(int):
#                 cx, cy = (x1+x2)//2, (y1+y2)//2
#                 best_iou, cls_name = 0, "unknown"

#                 for i, box in enumerate(det_boxes):
#                     xx1, yy1 = max(x1, box[0]), max(y1, box[1])
#                     xx2, yy2 = min(x2, box[2]), min(y2, box[3])
#                     inter = max(0, xx2-xx1) * max(0, yy2-yy1)
#                     area1 = (x2-x1)*(y2-y1)
#                     area2 = (box[2]-box[0])*(box[3]-box[1])
#                     iou = inter/(area1+area2-inter+1e-6)
#                     if iou > best_iou:
#                         best_iou = iou
#                         cls_name = names[det_classes[i]]

#                 if best_iou > 0.2:
#                     track_class_memory[track_id] = cls_name
#                 elif track_id in track_class_memory:
#                     cls_name = track_class_memory[track_id]

#                 track_last_seen[track_id] = frame_idx
#                 current_centroids[track_id] = (cx, cy, cls_name)

#                 color = get_class_color(list(names.values()).index(cls_name)) if cls_name in names.values() else (200,200,200)
#                 annotator.box_label([x1, y1, x2, y2], f"{cls_name}", color=color)

#             # 🔢 LINE CROSSING COUNT
#             for obj_id, (cx, cy, cls) in current_centroids.items():
#                 if cls == "unknown":
#                     continue
#                 if cy < line_y - offset:
#                     current_side = "above"
#                 elif cy > line_y + offset:
#                     current_side = "below"
#                 else:
#                     current_side = "buffer"

#                 if obj_id not in track_side_memory:
#                     track_side_memory[obj_id] = current_side
#                     continue

#                 prev_side = track_side_memory[obj_id]

#                 if prev_side == "above" and current_side == "below" and obj_id not in counted_ids:
#                     counts_in[cls] += 1
#                     counted_ids.add(obj_id)

#                 elif prev_side == "below" and current_side == "above" and obj_id not in counted_ids:
#                     counts_out[cls] += 1
#                     counted_ids.add(obj_id)

#                 if current_side != "buffer":
#                     track_side_memory[obj_id] = current_side

#             # Cleanup old IDs
#             for tid in list(track_last_seen.keys()):
#                 if frame_idx - track_last_seen[tid] > BUFFER_FRAMES:
#                     track_last_seen.pop(tid, None)
#                     track_class_memory.pop(tid, None)
#                     track_side_memory.pop(tid, None)
#                     counted_ids.discard(tid)

#             cv2.line(im0, (0, line_y), (im0.shape[1], line_y), (0,255,255), 2)

#             # 📊 STABLE TEXT (NO JUMP)
#             start_y = 80
#             for i, cls in enumerate(DISPLAY_CLASSES):
#                 y = start_y + i * 40
#                 cv2.putText(im0, f"{cls} IN : {counts_in.get(cls,0):03d}", (20, y), FONT, 0.7, (0,255,0), 1, cv2.LINE_AA)
#                 cv2.putText(im0, f"{cls} OUT: {counts_out.get(cls,0):03d}", (20, y+18), FONT, 0.7, (0,0,255), 1, cv2.LINE_AA)

#             im0 = add_logo_top_left(im0)
#             im0 = add_diagonal_watermark(im0)

#             # 💾 SAVE EXACT FRAME YOU SEE
#             video_writer.write(im0)

#             cv2.imshow("Counting", im0)
#             if cv2.waitKey(1) == ord("q"):
#                 break

#     finally:
#         if video_writer:
#             video_writer.release()
#         cv2.destroyAllWindows()
#         print("\n✅ Video saved:", video_path)


# def about():
#     """
# YOLOv5 + SORT Object Tracking with IN/OUT Line Crossing Counter
# ===============================================================

# This script performs real-time object detection, multi-object tracking,
# and directional counting (IN / OUT) based on objects crossing a virtual line.

# It combines:
# - YOLOv5 → Object Detection
# - SORT → Persistent Object Tracking with IDs
# - Memory Buffer System → Prevents ID switching errors
# - Line Crossing Logic → Counts direction of movement

# ----------------------------------------------------------------------
# 🎯 PURPOSE
# ----------------------------------------------------------------------
# The system counts objects separately based on the direction they cross
# a predefined horizontal line in the video frame.

# IN  direction  = Object moves from ABOVE the line to BELOW
# OUT direction = Object moves from BELOW the line to ABOVE

# This is useful for:
# • Vehicle entry/exit counting
# • People flow analysis
# • Industrial conveyor monitoring
# • Smart surveillance systems

# ----------------------------------------------------------------------
# 🧠 CORE FEATURES
# ----------------------------------------------------------------------
# ✔ Real-time YOLOv5 object detection
# ✔ SORT tracking with stable object IDs
# ✔ Memory buffer to avoid class flickering
# ✔ Prevents double counting using ID history
# ✔ Direction-based IN / OUT counting
# ✔ Optional single-class filtering
# ✔ Watermarked output video
# ✔ Live FPS and stats in terminal
# ✔ Auto-removal of stale IDs after buffer time

# ----------------------------------------------------------------------
# 📊 COUNTING LOGIC
# ----------------------------------------------------------------------
# Each tracked object has a memory of:

# • Last known class label
# • Last known side of the line (above/below)
# • Last frame seen

# An object is counted ONLY when:
# 1. It has a valid tracking ID
# 2. It crosses from one side of the line to the other
# 3. It has not been counted before
# 4. It matches the selected filter class (if enabled)

# Movement Direction Detection:
# --------------------------------
# ABOVE  → BELOW  = IN count increases
# BELOW  → ABOVE  = OUT count increases

# Objects moving along the line without crossing are ignored.

# ----------------------------------------------------------------------
# 🧠 MEMORY BUFFER SYSTEM
# ----------------------------------------------------------------------
# The system prevents ID flickering and class switching using:

# track_class_memory → Remembers last known class per ID
# track_last_seen    → Stores last frame object was detected
# track_side_memory  → Stores last known side of the line
# counted_ids        → Prevents duplicate counting

# BUFFER_FRAMES = 300 (~10 seconds)

# If an object disappears briefly and reappears, it keeps its original ID
# and class label within this buffer time.

# ----------------------------------------------------------------------
# 🎯 CLASS FILTERING
# ----------------------------------------------------------------------
# You can count only one class and ignore others.

# Example:
#     --filter-class Car

# This means:
# ✔ Only "Car" objects are tracked and counted
# ✖ All other detected classes are ignored

# ----------------------------------------------------------------------
# 📺 OUTPUT
# ----------------------------------------------------------------------
# • Annotated video saved to:
#       runs/count/<name>/output.mp4

# • On-screen display shows:
#       Class IN count
#       Class OUT count

# • Terminal displays live stats:
#       FPS | Total IN | Total OUT | Per-class breakdown

# ----------------------------------------------------------------------
# 🛠 COMMAND LINE ARGUMENTS
# ----------------------------------------------------------------------
# --weights        Path to YOLOv5 model (.pt)
# --source         Video file, folder, stream URL, or webcam index
# --imgsz          Inference image size (default: 640)
# --conf-thres     Detection confidence threshold
# --iou-thres      NMS IoU threshold
# --device         CUDA device or CPU
# --project        Output folder
# --name           Run name
# --filter-class   (Optional) Track/count only this class

# ----------------------------------------------------------------------
# ▶ EXAMPLE USAGE
# ----------------------------------------------------------------------

# Count all classes:
#     python det_tracking_count.py --weights model.pt --source video.mp4

# Count only Cars:
#     python det_tracking_count.py --weights model.pt --source video.mp4 --filter-class Car

# ----------------------------------------------------------------------
# 🏁 RESULT
# ----------------------------------------------------------------------
# Instead of simple totals, you now get directional flow:

# Car IN : 12
# Car OUT: 9

# Perfect for traffic flow, entry/exit analytics, and industrial automation.
#     """

# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", type=str, required=True)
#     parser.add_argument("--source", type=str, required=True)
#     parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/count")
#     parser.add_argument("--name", default="exp")
#     parser.add_argument(
#     "--filter-class",
#     type=str,
#     default=None,
#     help="Name of class to track/count (example: Car)"
# )

#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
#     return opt


# if __name__ == "__main__":
#     opt = parse_opt()
#     # Access and print the docstring properly
#     print(about.__doc__)
#     run(**vars(opt))


import argparse
import os
import pathlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sort.sort import Sort

# Fix Windows path issue
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, increment_path, non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(weights, source, imgsz, conf_thres, iou_thres, device, project, name):

    # 🎯 ONLY THESE CLASSES WILL BE COUNTED
    filter_classes = ["person"]

    counts_in = {c: 0 for c in filter_classes}
    counts_out = {c: 0 for c in filter_classes}
    DISPLAY_CLASSES = filter_classes

    def get_class_color(class_id):
        np.random.seed(int(class_id) + 42)
        return tuple(int(x) for x in np.random.randint(0, 255, size=3))

    def add_diagonal_watermark(frame, text="Demo_purpose", opacity=0.15):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        font_scale = min(w, h) / 900
        thickness = int(font_scale * 2)
        layer = np.zeros_like(frame, dtype=np.uint8)
        cv2.putText(
            layer, text, (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)
        rotated = cv2.warpAffine(layer, M, (w, h))
        return cv2.addWeighted(rotated, opacity, overlay, 1 - opacity, 0)

    def add_logo_top_left(frame, logo_path="logo_white 1.png", width=120):
        if not os.path.exists(logo_path):
            return frame
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            return frame
        h_logo, w_logo = logo.shape[:2]
        new_h = int(width * (h_logo / w_logo))
        logo = cv2.resize(logo, (width, new_h))
        x_offset, y_offset = 10, 10
        if logo.shape[2] == 4:
            alpha = logo[:, :, 3] / 255.0
            for c in range(3):
                frame[y_offset : y_offset + new_h, x_offset : x_offset + width, c] = (
                    alpha * logo[:, :, c]
                    + (1 - alpha) * frame[y_offset : y_offset + new_h, x_offset : x_offset + width, c]
                )
        else:
            frame[y_offset : y_offset + new_h, x_offset : x_offset + width] = logo[:, :, :3]
        return frame

    # ---------------- MODEL ----------------
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    dataset = (
        LoadStreams(source, img_size=imgsz, stride=stride)
        if source.isnumeric()
        else LoadImages(source, img_size=imgsz, stride=stride)
    )
    tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    track_class_memory, track_last_seen, track_side_memory = {}, {}, {}
    counted_ids = set()

    line_y = 200
    offset = 5

    save_dir = increment_path(Path(project) / name)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(save_dir / "output.mp4")
    video_writer = None
    start_time = time.time()

    try:
        for frame_idx, data in enumerate(dataset):
            frame_time = time.time()
            1 / (frame_time - start_time + 1e-6)
            start_time = frame_time

            if data is None:
                continue

            _path, im, im0s, vid_cap, _s = data
            if im is None or im0s is None:
                continue

            im = torch.from_numpy(im).to(device).float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            pred = model(im)
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            im0 = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
            annotator = Annotator(im0, line_width=2)

            if video_writer is None:
                h, w = im0.shape[:2]
                fps = vid_cap.get(cv2.CAP_PROP_FPS) or 30
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

            detections, det_boxes, det_classes = [], [], []

            if len(pred[0]):
                pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()
                for *xyxy, conf, cls in pred[0]:
                    class_name = names[int(cls)].strip().lower()
                    if class_name != "person":
                        continue
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append([x1, y1, x2, y2, conf.item()])
                    det_boxes.append([x1, y1, x2, y2])
                    det_classes.append(int(cls))

            tracks = tracker.update(np.array(detections)) if detections else np.empty((0, 5))
            current_centroids = {}

            for x1, y1, x2, y2, track_id in tracks.astype(int):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                best_iou, cls_name = 0, "unknown"

                for i, box in enumerate(det_boxes):
                    xx1, yy1 = max(x1, box[0]), max(y1, box[1])
                    xx2, yy2 = min(x2, box[2]), min(y2, box[3])
                    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (box[2] - box[0]) * (box[3] - box[1])
                    iou = inter / (area1 + area2 - inter + 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        cls_name = names[det_classes[i]].strip().lower()

                if best_iou > 0.2:
                    track_class_memory[track_id] = cls_name
                elif track_id in track_class_memory:
                    cls_name = track_class_memory[track_id]

                track_last_seen[track_id] = frame_idx
                current_centroids[track_id] = (cx, cy, cls_name)

                # color = get_class_color(det_classes[0]) if det_classes else (200,200,200)
                color = get_class_color(0)  # person class id
                annotator.box_label([x1, y1, x2, y2], f"{cls_name}", color=color)

            # 🔢 COUNTING
            for obj_id, (cx, cy, cls) in current_centroids.items():
                if cls not in filter_classes:
                    continue

                current_side = "above" if cy < line_y - offset else "below" if cy > line_y + offset else "buffer"

                if obj_id not in track_side_memory:
                    track_side_memory[obj_id] = current_side
                    continue

                prev_side = track_side_memory[obj_id]

                if prev_side == "above" and current_side == "below" and obj_id not in counted_ids:
                    counts_in[cls] += 1
                    counted_ids.add(obj_id)

                elif prev_side == "below" and current_side == "above" and obj_id not in counted_ids:
                    counts_out[cls] += 1
                    counted_ids.add(obj_id)

                if current_side != "buffer":
                    track_side_memory[obj_id] = current_side

            # Draw line
            cv2.line(im0, (0, line_y), (im0.shape[1], line_y), (0, 255, 255), 2)

            # Display counts
            y0 = 90
            for i, cls in enumerate(DISPLAY_CLASSES):
                text_in = f"{cls} IN : {counts_in[cls]:03d}"
                text_out = f"{cls} OUT: {counts_out[cls]:03d}"
                cv2.putText(im0, text_in, (20, y0 + i * 50), FONT, 0.8, (0, 100, 0), 2)
                cv2.putText(im0, text_out, (20, y0 + 25 + i * 50), FONT, 0.8, (0, 0, 150), 2)

            im0 = add_logo_top_left(im0)
            im0 = add_diagonal_watermark(im0)
            video_writer.write(im0)

            cv2.imshow("Counting", im0)
            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("\n✅ Video saved:", video_path)


def help():
    """============================================================ YOLOv5 + SORT Person Counting Script.
    ============================================================.

    WHAT THIS SCRIPT DOES
    ---------------------
    • Detects ONLY 'person' class using YOLOv5
    • Tracks detected persons using SORT tracker
    • Draws a horizontal counting line in the frame
    • Counts:
        - IN  : person crossing from above → below the line
        - OUT : person crossing from below → above the line
    • Adds:
        - Bounding boxes
        - Live counts (IN / OUT)
        - Company logo (top-left)
        - Diagonal watermark (demo purpose)
    • Saves the processed video to disk
    • Displays live output window

    ------------------------------------------------------------
    PIPELINE FLOW
    ------------------------------------------------------------
    1️⃣ Load YOLOv5 model (DetectMultiBackend) 2️⃣ Read input:
        - Webcam (source=0)
        - Video file
        - Image folder
    3️⃣ Run object detection (YOLOv5) 4️⃣ Filter detections → ONLY 'person' 5️⃣ Track persons using SORT (assign unique
    IDs) 6️⃣ Compute centroid (cx, cy) for each tracked person 7️⃣ Check which side of the line the person is on:
        - above
        - below
        - buffer zone
    8️⃣ Count crossing events:
        - above → below  → IN
        - below → above  → OUT
        (each ID counted only once)
    9️⃣ Draw overlays:
        - Bounding boxes
        - Counting line
        - IN / OUT counters
        - Logo + watermark
    🔟 Save final annotated video

    ------------------------------------------------------------
    COUNTING LOGIC
    ------------------------------------------------------------
    • Horizontal line position: line_y = 200
    • Buffer offset: ±5 pixels
    • Each tracked ID is counted only once
    • Prevents double counting using:
        - track_side_memory
        - counted_ids

    ------------------------------------------------------------
    OUTPUT
    ------------------------------------------------------------
    • Output folder:
        runs/count/<exp_name>/
    • Output file:
        output.mp4
    • Window name:
        "Counting"

    ------------------------------------------------------------
    COMMAND LINE USAGE
    ------------------------------------------------------------
    python count.py --weights yolov5s.pt --source input.mp4 --imgsz 640 --conf-thres 0.25 --iou-thres 0.45 --device 0
    --project runs/count --name exp

    ------------------------------------------------------------

    Args:
        ---------: --weights Path to YOLOv5 model weights (required) --source Input source (video/image/webcam index)
        --imgsz       Inference image size (default: 640)
        --conf-thres  Confidence threshold (default: 0.25)
        --iou-thres   NMS IoU threshold (default: 0.45) --device CUDA device (0, 1, or 'cpu') --project Output directory
            --name Experiment name

    ------------------------------------------------------------
            EXIT
    ------------------------------------------------------------
            • Press 'q' to stop processing
            • Video is saved safely on exit

            ============================================================
    """


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/count")
    parser.add_argument("--name", default="exp")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    print(help.__doc__)
    run(**vars(opt))
