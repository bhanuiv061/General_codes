# import argparse
# import os
# import sys
# from pathlib import Path
# import pathlib
# import cv2
# import torch
# import numpy as np
# from datetime import datetime
# from sort.sort import Sort
# import time
# from tqdm import tqdm
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
# def run(weights, source, imgsz, conf_thres, iou_thres, device, project, name):

#     # 🎯 ONLY THESE CLASSES WILL BE COUNTED
#     filter_classes = ["person"]

#     counts_in = {c: 0 for c in filter_classes}
#     counts_out = {c: 0 for c in filter_classes}
#     DISPLAY_CLASSES = filter_classes

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
#     # Estimate total frames (works for video & image folder)
#     total_frames = None
#     if isinstance(dataset, LoadImages):
#         total_frames = dataset.nf  # number of images or video frames
#     FONT = cv2.FONT_HERSHEY_SIMPLEX
#     track_class_memory, track_last_seen, track_side_memory = {}, {}, {}
#     counted_ids = set()

#     BUFFER_FRAMES = 300
#     #line_y = 200
#     line_x = 400
#     offset = 5

#     save_dir = increment_path(Path(project) / name)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     video_path = str(save_dir / "output.mp4")
#     video_writer = None
#     start_time = time.time()

#     try:
#         for frame_idx, data in enumerate(
#         tqdm(dataset, total=total_frames, desc="Processing", unit="frame")
#                                                                 ):
#             frame_time = time.time()
#             fps_live = 1 / (frame_time - start_time + 1e-6)
#             start_time = frame_time

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
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS) or 30
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

#             detections, det_boxes, det_classes = [], [], []

#             if len(pred[0]):
#                 pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()
#                 for *xyxy, conf, cls in pred[0]:
#                     class_name = names[int(cls)].strip().lower()
#                     if class_name != "person":
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
#                         cls_name = names[det_classes[i]].strip().lower()

#                 if best_iou > 0.2:
#                     track_class_memory[track_id] = cls_name
#                 elif track_id in track_class_memory:
#                     cls_name = track_class_memory[track_id]

#                 track_last_seen[track_id] = frame_idx
#                 current_centroids[track_id] = (cx, cy, cls_name)

#                 #color = get_class_color(det_classes[0]) if det_classes else (200,200,200)
#                 color = get_class_color(0)  # person class id
#                 annotator.box_label([x1, y1, x2, y2], f"{cls_name}", color=color)

#             # 🔢 COUNTING
#             for obj_id, (cx, cy, cls) in current_centroids.items():
#                 if cls not in filter_classes:
#                     continue

#                 current_side = "above" if cy < line_x - offset else "below" if cy > line_x + offset else "buffer"

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

#             # Draw line
#             cv2.line(im0, (0, line_x), (im0.shape[1], line_x), (0,255,255), 2)

#             # Display counts
#             y0 = 90
#             for i, cls in enumerate(DISPLAY_CLASSES):
#                 text_in = f"{cls} IN : {counts_in[cls]:03d}"
#                 text_out = f"{cls} OUT: {counts_out[cls]:03d}"
#                 cv2.putText(im0, text_in, (20, y0 + i*50), FONT, 0.8, (0,100,0), 2)
#                 cv2.putText(im0, text_out, (20, y0 + 25 + i*50), FONT, 0.8, (0,0,150), 2)

#             im0 = add_logo_top_left(im0)
#             im0 = add_diagonal_watermark(im0)
#             video_writer.write(im0)

#             # cv2.imshow("Counting", im0)
#             # if cv2.waitKey(1) == ord("q"):
#             #     break

#     finally:
#         if video_writer:
#             video_writer.release()
#         #cv2.destroyAllWindows()
#         print("\n✅ Video saved:", video_path)


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
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
#     return opt


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))


#################  working #####################################################


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
# from tqdm import tqdm

# # ================= WINDOWS PATH FIX =================
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


# # ================= ROI SELECTION =================
# def select_roi_with_mouse(frame):
#     """
#     Draw ROI on FULL ORIGINAL FRAME.
#     Left click + drag → draw box
#     ENTER → confirm
#     ESC → cancel
#     """

#     # Hard check for GUI support
#     if not hasattr(cv2, "imshow"):
#         h, w = frame.shape[:2]
#         print("⚠️ OpenCV GUI not available. Using full frame ROI.")
#         return 0, 0, w, h

#     roi = []
#     drawing = False
#     display = frame.copy()

#     def mouse_callback(event, x, y, flags, param):
#         nonlocal roi, drawing, display

#         if event == cv2.EVENT_LBUTTONDOWN:
#             roi = [(x, y)]
#             drawing = True

#         elif event == cv2.EVENT_MOUSEMOVE and drawing:
#             display = frame.copy()
#             cv2.rectangle(display, roi[0], (x, y), (0, 255, 0), 2)

#         elif event == cv2.EVENT_LBUTTONUP:
#             roi.append((x, y))
#             drawing = False
#             display = frame.copy()
#             cv2.rectangle(display, roi[0], roi[1], (0, 255, 0), 2)

#     cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Select ROI", frame.shape[1], frame.shape[0])
#     cv2.setMouseCallback("Select ROI", mouse_callback)

#     while True:
#         cv2.imshow("Select ROI", display)
#         key = cv2.waitKey(1) & 0xFF

#         if key == 13 and len(roi) == 2:  # ENTER
#             break
#         elif key == 27:  # ESC
#             roi = None
#             break

#     cv2.destroyAllWindows()

#     if roi:
#         x1, y1 = roi[0]
#         x2, y2 = roi[1]
#         return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

#     return None


# # ================= MAIN =================
# @smart_inference_mode()
# def run(weights, source, imgsz, conf_thres, iou_thres, device, project, name, mode):

#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names
#     imgsz = check_img_size(imgsz, s=stride)

#     dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#         if source.isnumeric() else LoadImages(source, img_size=imgsz, stride=stride)

#     tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#     # ================= COUNTERS =================
#     counts = {"person": 0}
#     counted_ids = set()
#     id_inside_roi = set()

#     # ================= LINE CONFIG =================
#     line_y = 400
#     offset = 5

#     # ================= ROI =================
#     roi_coords = None
#     if mode == "roi":
#         # Create a TEMP loader only to grab ONE ORIGINAL frame
#         temp_ds = LoadStreams(source, img_size=imgsz, stride=stride) \
#             if source.isnumeric() else LoadImages(source, img_size=imgsz, stride=stride)

#         data = next(iter(temp_ds))
#         path, im, im0s, vid_cap, s = data

#         # IMPORTANT: use im0 / im0s → ORIGINAL FRAME (no resize)
#         frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()

#         h, w = frame.shape[:2]
#         print(f"🖼️ ROI selection on full frame: {w}x{h}")

#         roi_coords = select_roi_with_mouse(frame)

#         if roi_coords is None:
#             print("❌ ROI cancelled by user")
#             return

#         roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
#         print(f"✅ ROI selected: {roi_coords}")

#     # ================= OUTPUT =================
#     save_dir = increment_path(Path(project) / name)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     out_path = save_dir / f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#     writer = None

#     # ================= LOOP =================
#     for data in tqdm(dataset, desc="Processing"):
#         path, im, im0s, cap, _ = data
#         im0 = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()

#         im = torch.from_numpy(im).to(device).float() / 255.0
#         if im.ndim == 3:
#             im = im[None]

#         pred = model(im)
#         pred = non_max_suppression(pred, conf_thres, iou_thres)

#         if writer is None:
#             h, w = im0.shape[:2]
#             fps = cap.get(cv2.CAP_PROP_FPS) if cap else 30
#             writer = cv2.VideoWriter(str(out_path),
#                                      cv2.VideoWriter_fourcc(*"mp4v"),
#                                      fps, (w, h))

#         detections = []

#         if len(pred[0]):
#             pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape)
#             for *xyxy, conf, cls in pred[0]:
#                 if names[int(cls)] != "person":
#                     continue
#                 x1, y1, x2, y2 = map(float, xyxy)
#                 detections.append([x1, y1, x2, y2, float(conf)])

#         tracks = tracker.update(np.array(detections)) if detections else []

#         for x1, y1, x2, y2, tid in tracks:
#             x1, y1, x2, y2, tid = map(int, [x1, y1, x2, y2, tid])
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

#             # ROI MODE
#             if mode == "roi":
#                 inside = roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2
#                 if inside and tid not in id_inside_roi:
#                     counts["person"] += 1
#                     id_inside_roi.add(tid)
#                 if not inside and tid in id_inside_roi:
#                     id_inside_roi.remove(tid)

#             # LINE MODE
#             if mode == "line":
#                 if abs(cy - line_y) < offset and tid not in counted_ids:
#                     counts["person"] += 1
#                     counted_ids.add(tid)

#             cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(im0, f"ID {tid}", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

#         if mode == "roi":
#             cv2.rectangle(im0, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

#         if mode == "line":
#             cv2.line(im0, (0, line_y), (im0.shape[1], line_y), (0, 255, 255), 2)

#         cv2.putText(im0, f"COUNT: {counts['person']}",
#                     (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

#         writer.write(im0)

#     writer.release()
#     print("✅ Saved:", out_path)


# def parse_opt():
#     p = argparse.ArgumentParser()
#     p.add_argument("--weights", required=True)
#     p.add_argument("--source", required=True)
#     p.add_argument("--imgsz", nargs="+", type=int, default=[640])
#     p.add_argument("--conf-thres", type=float, default=0.4)
#     p.add_argument("--iou-thres", type=float, default=0.5)
#     p.add_argument("--device", default="")
#     p.add_argument("--project", default="runs/count")
#     p.add_argument("--name", default="exp")
#     p.add_argument("--mode", choices=["line", "roi"], default="line")
#     opt = p.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
#     return opt


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))


############### python human_counting.py --img 640 --weights yolov5s.pt --project "D:\bhanu\human_detection" --name human_ --conf 0.50 --source "C:\Users\admin\Downloads\human_video_.mp4" --mode roi


# import argparse
# import os
# import sys
# import time
# from pathlib import Path
# from collections import defaultdict

# import cv2
# import torch
# import numpy as np
# from sort.sort import Sort

# # ================= WINDOWS PATH FIX =================
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes, increment_path
# from utils.torch_utils import select_device, smart_inference_mode
# from ultralytics.utils.plotting import Annotator

# # ===================================================
# # ROI SELECTION (FULL FRAME VISIBLE)
# # ===================================================
# def select_roi_with_mouse(frame):
#     h, w = frame.shape[:2]
#     clone = frame.copy()
#     roi = []
#     drawing = False

#     def mouse(event, x, y, flags, param):
#         nonlocal roi, drawing, clone
#         if event == cv2.EVENT_LBUTTONDOWN:
#             roi = [(x, y)]
#             drawing = True
#         elif event == cv2.EVENT_MOUSEMOVE and drawing:
#             clone = frame.copy()
#             cv2.rectangle(clone, roi[0], (x, y), (0, 255, 0), 2)
#         elif event == cv2.EVENT_LBUTTONUP:
#             roi.append((x, y))
#             drawing = False
#             cv2.rectangle(clone, roi[0], roi[1], (0, 255, 0), 2)

#     cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Select ROI", w, h)
#     cv2.setMouseCallback("Select ROI", mouse)

#     while True:
#         cv2.imshow("Select ROI", clone)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 13 and len(roi) == 2:  # ENTER
#             break
#         elif key == 27:  # ESC
#             return None

#     cv2.destroyWindow("Select ROI")
#     x1, y1 = roi[0]
#     x2, y2 = roi[1]
#     return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

# # ===================================================
# # LINE HELPERS
# # ===================================================
# def get_line(frame, orientation):
#     h, w = frame.shape[:2]
#     if orientation == "vertical":
#         return (w // 2, 0), (w // 2, h)
#     return (0, h // 2), (w, h // 2)

# def get_side(cx, cy, line, orientation):
#     (x1, y1), _ = line
#     return "LEFT" if orientation == "vertical" and cx < x1 else \
#            "RIGHT" if orientation == "vertical" else \
#            "TOP" if cy < y1 else "BOTTOM"

# # ===================================================
# # MAIN
# # ===================================================
# @smart_inference_mode()
# def run(
#     weights,
#     source,
#     imgsz,
#     conf_thres,
#     iou_thres,
#     device,
#     project,
#     name,
#     mode,
#     line_orientation,
#     direction,
# ):

#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device)
#     stride = model.stride
#     imgsz = check_img_size(imgsz, s=stride)

#     dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#         if source.isnumeric() else LoadImages(source, img_size=imgsz, stride=stride)

#     tracker = Sort(max_age=45, min_hits=3, iou_threshold=0.3)

#     save_dir = increment_path(Path(project) / name)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     out_path = str(save_dir / f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
#     writer = None

#     # ROI
#     roi_coords = None
#     if mode in ["roi", "roi_line"]:
#         tmp = next(iter(dataset))
#         frame = tmp[2][0] if isinstance(tmp[2], list) else tmp[2]
#         roi_coords = select_roi_with_mouse(frame)
#         if roi_coords is None:
#             print("❌ ROI cancelled")
#             return
#         rx1, ry1, rx2, ry2 = roi_coords

#     # Line
#     line = None

#     # Tracking memory
#     track_side = {}
#     track_frames = defaultdict(int)
#     track_seen_in_roi = defaultdict(bool)
#     counted_ids = set()

#     in_count = 0
#     out_count = 0
#     MIN_FRAMES = 5

#     # ================= LOOP =================
#     for path, im, im0s, vid_cap, s in dataset:
#         im = torch.from_numpy(im).to(device).float() / 255.0
#         if im.ndim == 3:
#             im = im[None]

#         preds = model(im)
#         preds = non_max_suppression(preds, conf_thres, iou_thres, classes=[0])

#         im0 = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#         annotator = Annotator(im0, line_width=2)

#         if writer is None:
#             h, w = im0.shape[:2]
#             fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
#             writer = cv2.VideoWriter(out_path,
#                                      cv2.VideoWriter_fourcc(*"mp4v"),
#                                      fps, (w, h))
#             line = get_line(im0, line_orientation)

#         detections = []
#         if len(preds[0]):
#             preds[0][:, :4] = scale_boxes(im.shape[2:], preds[0][:, :4], im0.shape).round()
#             for *xyxy, conf, cls in preds[0]:
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 detections.append([x1, y1, x2, y2, conf.item()])

#         tracks = tracker.update(np.array(detections)) if detections else []

#         for x1, y1, x2, y2, tid in tracks:
#             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

#             track_frames[tid] += 1
#             curr_side = get_side(cx, cy, line, line_orientation)

#             inside_roi = True
#             if mode in ["roi", "roi_line"]:
#                 inside_roi = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
#                 if inside_roi:
#                     track_seen_in_roi[tid] = True

#             if tid not in track_side:
#                 track_side[tid] = curr_side
#                 continue

#             prev_side = track_side[tid]

#             if track_seen_in_roi[tid] and track_frames[tid] >= MIN_FRAMES and tid not in counted_ids:
#                 if prev_side != curr_side:
#                     if line_orientation == "horizontal":
#                         move = "in" if prev_side == "TOP" and curr_side == "BOTTOM" else \
#                                "out" if prev_side == "BOTTOM" and curr_side == "TOP" else None
#                     else:
#                         move = "in" if prev_side == "LEFT" and curr_side == "RIGHT" else \
#                                "out" if prev_side == "RIGHT" and curr_side == "LEFT" else None

#                     if move and direction in ["both", move]:
#                         in_count += (move == "in")
#                         out_count += (move == "out")
#                         counted_ids.add(tid)

#             track_side[tid] = curr_side

#             if inside_roi:
#                 annotator.box_label([x1, y1, x2, y2], f"ID {int(tid)}", (0, 255, 0))

#         if mode in ["roi", "roi_line"]:
#             cv2.rectangle(im0, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

#         cv2.line(im0, line[0], line[1], (0, 0, 255), 2)

#         cv2.putText(im0, f"IN : {in_count}", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         cv2.putText(im0, f"OUT: {out_count}", (20, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         writer.write(im0)

#     writer.release()
#     print(f"✅ Saved: {out_path}")

# # ===================================================
# # ARGPARSE
# # ===================================================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", type=str, required=True)
#     parser.add_argument("--source", type=str, required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.5)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/count")
#     parser.add_argument("--name", default="exp")
#     parser.add_argument("--mode", choices=["line", "roi", "roi_line"], default="roi_line")
#     parser.add_argument("--line_orientation", choices=["vertical", "horizontal"], default="vertical")
#     parser.add_argument("--direction", choices=["in", "out", "both"], default="both")
#     return parser.parse_args()

# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))


################ sort + yolov5s + roi counting (FULL FRAME VISIBLE) + line counting (SINGLE LINE) + OUTPUT VIDEO WITH COUNTS + ARGPARSE


import argparse
import math
import os
import pathlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sort.sort import Sort
from tqdm import tqdm

# ================= WINDOWS PATH FIX =================
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, increment_path, non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode


# ================= ROI SELECTION =================
def select_roi_with_mouse(frame):
    if not hasattr(cv2, "imshow"):
        h, w = frame.shape[:2]
        print("⚠️ OpenCV GUI not available. Using full frame ROI.")
        return 0, 0, w, h

    roi = []
    drawing = False
    display = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi, drawing, display
        if event == cv2.EVENT_LBUTTONDOWN:
            roi = [(x, y)]
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            display = frame.copy()
            cv2.rectangle(display, roi[0], (x, y), (180, 180, 180), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            roi.append((x, y))
            drawing = False
            display = frame.copy()
            cv2.rectangle(display, roi[0], roi[1], (180, 180, 180), 1)

    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", frame.shape[1], frame.shape[0])
    cv2.setMouseCallback("Select ROI", mouse_callback)

    while True:
        cv2.imshow("Select ROI", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(roi) == 2:
            break
        elif key == 27:
            roi = None
            break

    cv2.destroyAllWindows()

    if roi:
        x1, y1 = roi[0]
        x2, y2 = roi[1]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return None


# ================= ID PERSISTENCE MATCH =================
def match_with_memory(cx, cy, memory, dist_thresh=50):
    best_id = None
    best_dist = float("inf")
    for tid, (px, py, _) in memory.items():
        d = math.hypot(cx - px, cy - py)
        if d < best_dist:
            best_dist = d
            best_id = tid
    if best_dist < dist_thresh:
        return best_id
    return None


# ================= MAIN =================
@smart_inference_mode()
def run(weights, source, imgsz, conf_thres, iou_thres, device, project, name, mode):

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

    # ================= COUNTERS =================
    counts = {"person": 0}
    counted_ids = set()
    id_inside_roi = set()

    # ================= LINE CONFIG =================
    line_y = 400
    offset = 5

    # ================= ROI =================
    roi_coords = None
    if mode == "roi":
        temp_ds = (
            LoadStreams(source, img_size=imgsz, stride=stride)
            if source.isnumeric()
            else LoadImages(source, img_size=imgsz, stride=stride)
        )

        path, im, im0s, _vid_cap, _s = next(iter(temp_ds))
        frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
        roi_coords = select_roi_with_mouse(frame)
        if roi_coords is None:
            print("❌ ROI cancelled")
            return
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords

    # ================= OUTPUT =================
    save_dir = increment_path(Path(project) / name)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer = None

    # ================= TRACK MEMORY =================
    track_memory = {}  # tid -> (cx, cy, last_seen_frame)
    frame_idx = 0
    MAX_MEMORY_FRAMES = 60

    # ================= LOOP =================
    for data in tqdm(dataset, desc="Processing"):
        frame_idx += 1
        _path, im, im0s, cap, _ = data
        im0 = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()

        im = torch.from_numpy(im).to(device).float() / 255.0
        if im.ndim == 3:
            im = im[None]

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        if writer is None:
            h, w = im0.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) if cap else 30
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        detections = []

        if len(pred[0]):
            pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape)
            for *xyxy, conf, cls in pred[0]:
                if names[int(cls)] != "person":
                    continue
                x1, y1, x2, y2 = map(float, xyxy)
                detections.append([x1, y1, x2, y2, float(conf)])

        tracks = tracker.update(np.array(detections)) if detections else []

        for x1, y1, x2, y2, tid in tracks:
            x1, y1, x2, y2, tid = map(int, [x1, y1, x2, y2, tid])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            reused = match_with_memory(cx, cy, track_memory)
            if reused is not None:
                tid = reused

            track_memory[tid] = (cx, cy, frame_idx)

            if mode == "roi":
                inside = roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2
                if inside and tid not in id_inside_roi:
                    counts["person"] += 1
                    id_inside_roi.add(tid)
                if not inside and tid in id_inside_roi:
                    id_inside_roi.remove(tid)

            if mode == "line":
                if abs(cy - line_y) < offset and tid not in counted_ids:
                    counts["person"] += 1
                    counted_ids.add(tid)

            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im0, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ================= CLEAN OLD IDS =================
        expired = [tid for tid, (_, _, last) in track_memory.items() if frame_idx - last > MAX_MEMORY_FRAMES]
        for tid in expired:
            del track_memory[tid]

        if mode == "roi":
            cv2.rectangle(im0, (roi_x1, roi_y1), (roi_x2, roi_y2), (180, 180, 180), 1)

        if mode == "line":
            cv2.line(im0, (0, line_y), (im0.shape[1], line_y), (0, 255, 255), 1)

        # ================= COUNT TEXT WITH BACKGROUND =================
        text = f"COUNT: {counts['person']}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 4
        outline_thickness = 4

        x, y = 30, 50
        padding = 15

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # ✅ 1. Draw background FIRST
        cv2.rectangle(
            im0,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + baseline + padding),
            (0, 0, 0),  # black background
            -1,
        )

        # ✅ 2. Draw outline (white)
        cv2.putText(
            im0,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),  # white outline
            outline_thickness,
            cv2.LINE_AA,
        )

        # ✅ 3. Draw main text (BLACK)
        cv2.putText(
            im0,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),  # BLACK text
            thickness,
            cv2.LINE_AA,
        )

        writer.write(im0)


def parse_opt():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--imgsz", nargs="+", type=int, default=[640])
    p.add_argument("--conf-thres", type=float, default=0.4)
    p.add_argument("--iou-thres", type=float, default=0.5)
    p.add_argument("--device", default="")
    p.add_argument("--project", default="runs/count")
    p.add_argument("--name", default="exp")
    p.add_argument("--mode", choices=["line", "roi"], default="line")
    opt = p.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
