##########3 working ##########################################

# import argparse
# import sys
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import os
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import (
#     check_img_size,
#     non_max_suppression,
#     scale_boxes
# )
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# # ---------------- COUNT CONFIG ----------------
# LINE_X = 500
# OFFSET = 10


# @smart_inference_mode()
# def run(
#     weights,
#     source,
#     imgsz=640,
#     conf_thres=0.25,
#     iou_thres=0.45,
#     device=""
# ):

#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names
#     imgsz = check_img_size(imgsz, s=stride)

#     dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#         if source.isnumeric() else LoadImages(source, img_size=imgsz, stride=stride)

#     tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#     count_in  = {v: 0 for v in names.values()}
#     count_out = {v: 0 for v in names.values()}

#     track_last_side = {}
#     track_class = {}
#     counted_ids = set()

#     for path, im, im0s, vid_cap, s in dataset:

#         im = torch.from_numpy(im).to(device).float() / 255.0
#         if im.ndim == 3:
#             im = im[None]

#         pred = model(im)
#         pred = non_max_suppression(pred, conf_thres, iou_thres)

#         frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#         detections = []

#         if len(pred[0]):
#             pred[0][:, :4] = scale_boxes(
#                 im.shape[2:], pred[0][:, :4], frame.shape
#             ).round()

#             for *xyxy, conf, cls in pred[0]:
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 detections.append([x1, y1, x2, y2, conf.item(), int(cls)])

#         sort_input = (
#             np.array([d[:5] for d in detections])
#             if detections else np.empty((0, 5))
#         )

#         tracks = tracker.update(sort_input)

#         for x1, y1, x2, y2, track_id in tracks.astype(int):
#             cx = (x1 + x2) // 2

#             best_iou, cls_name = 0, "unknown"
#             for d in detections:
#                 xx1 = max(x1, d[0])
#                 yy1 = max(y1, d[1])
#                 xx2 = min(x2, d[2])
#                 yy2 = min(y2, d[3])
#                 inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
#                 area1 = (x2 - x1) * (y2 - y1)
#                 area2 = (d[2] - d[0]) * (d[3] - d[1])
#                 iou = inter / (area1 + area2 - inter + 1e-6)
#                 if iou > best_iou:
#                     best_iou = iou
#                     cls_name = names[d[5]]

#             if best_iou > 0.2:
#                 track_class[track_id] = cls_name
#             elif track_id in track_class:
#                 cls_name = track_class[track_id]

#             current_side = (
#                 "left" if cx < LINE_X - OFFSET else
#                 "right" if cx > LINE_X + OFFSET else
#                 "buffer"
#             )

#             if track_id not in track_last_side:
#                 track_last_side[track_id] = current_side
#                 continue

#             prev_side = track_last_side[track_id]

#             if prev_side == "left" and current_side == "right" and track_id not in counted_ids:
#                 count_in[cls_name] += 1
#                 counted_ids.add(track_id)

#             elif prev_side == "right" and current_side == "left" and track_id not in counted_ids:
#                 count_out[cls_name] += 1
#                 counted_ids.add(track_id)

#             if current_side != "buffer":
#                 track_last_side[track_id] = current_side

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.putText(frame, f"{cls_name} ID:{track_id}",
#                         (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

#         cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0,255,255), 2)

#         y = 30
#         for cls in count_in:
#             cv2.putText(frame,
#                         f"{cls} IN:{count_in[cls]} OUT:{count_out[cls]}",
#                         (10, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
#             y += 22

#         cv2.imshow("YOLOv5 Vertical Line Counting", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cv2.destroyAllWindows()


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", type=str, required=True)
#     parser.add_argument("--source", type=str, required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))


###################   need to check ###########################################


# import argparse
# import sys
# import time
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import os
# import pathlib
# from tqdm import tqdm

# # ---------------- PATH FIX (Windows) ----------------
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# # ---------------- YOLOv5 IMPORTS ----------------
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import (
#     check_img_size,
#     non_max_suppression,
#     scale_boxes
# )
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# # ---------------- COUNT CONFIG ----------------
# LINE_X = 500
# OFFSET = 10
# BUFFER_SECONDS = 10

# # --------------------------------------------------

# @smart_inference_mode()
# def run(
#     weights,
#     source,
#     imgsz=640,
#     conf_thres=0.25,
#     iou_thres=0.45,
#     device="",
#     project="runs/count",
#     name="exp"
# ):
#     # ---------------- VALIDATION ----------------
#     if not os.path.exists(weights):
#         raise FileNotFoundError(f"Weights not found: {weights}")

#     is_webcam = source.isnumeric()
#     if not is_webcam and not os.path.exists(source):
#         raise FileNotFoundError(f"Source not found: {source}")

#     # ---------------- OUTPUT DIR ----------------
#     save_dir = Path(project) / name
#     save_dir.mkdir(parents=True, exist_ok=True)

#     # ---------------- DEVICE & MODEL ----------------
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names
#     imgsz = check_img_size(imgsz, s=stride)

#     # ---------------- DATASET ----------------
#     dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#         if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

#     # ---------------- TRACKER ----------------
#     tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#     # ---------------- COUNTERS ----------------
#     count_in = {v: 0 for v in names.values()}
#     count_out = {v: 0 for v in names.values()}

#     track_last_side = {}
#     track_class = {}
#     last_count_time = {}

#     vid_writer = None
#     pbar = tqdm(dataset, desc="Processing", unit="frame")

#     for path, im, im0s, vid_cap, s in pbar:
#         im = torch.from_numpy(im).to(device).float() / 255.0
#         if im.ndim == 3:
#             im = im[None]

#         pred = model(im)
#         pred = non_max_suppression(pred, conf_thres, iou_thres)

#         frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#         detections = []

#         # ---------------- DETECTIONS ----------------
#         if len(pred[0]):
#             pred[0][:, :4] = scale_boxes(
#                 im.shape[2:], pred[0][:, :4], frame.shape
#             ).round()

#             for *xyxy, conf, cls in pred[0]:
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 detections.append([x1, y1, x2, y2, conf.item(), int(cls)])

#         sort_input = (
#             np.array([d[:5] for d in detections])
#             if detections else np.empty((0, 5))
#         )

#         tracks = tracker.update(sort_input)
#         now = time.time()

#         # ---------------- TRACK PROCESSING ----------------
#         for x1, y1, x2, y2, track_id in tracks.astype(int):
#             cx = (x1 + x2) // 2

#             # -------- CLASS ASSIGNMENT --------
#             best_iou, cls_name = 0, "unknown"
#             for d in detections:
#                 xx1 = max(x1, d[0])
#                 yy1 = max(y1, d[1])
#                 xx2 = min(x2, d[2])
#                 yy2 = min(y2, d[3])
#                 inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
#                 area1 = (x2 - x1) * (y2 - y1)
#                 area2 = (d[2] - d[0]) * (d[3] - d[1])
#                 iou = inter / (area1 + area2 - inter + 1e-6)
#                 if iou > best_iou:
#                     best_iou = iou
#                     cls_name = names[d[5]]

#             if best_iou > 0.2:
#                 track_class[track_id] = cls_name
#             else:
#                 cls_name = track_class.get(track_id, "unknown")

#             # -------- SIDE LOGIC --------
#             if cx < LINE_X - OFFSET:
#                 current_side = "left"
#             elif cx > LINE_X + OFFSET:
#                 current_side = "right"
#             else:
#                 current_side = "buffer"

#             prev_side = track_last_side.get(track_id)
#             last_time = last_count_time.get(track_id, 0)
#             allow_count = (now - last_time) > BUFFER_SECONDS

#             if prev_side and allow_count:
#                 if prev_side == "left" and current_side == "right":
#                     count_in[cls_name] += 1
#                     last_count_time[track_id] = now

#                 elif prev_side == "right" and current_side == "left":
#                     count_out[cls_name] += 1
#                     last_count_time[track_id] = now

#             track_last_side[track_id] = current_side

#             # -------- DRAW --------
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame,
#                         f"{cls_name} ID:{track_id}",
#                         (x1, y1 - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (255, 255, 255),
#                         2)

#         # ---------------- LINE + COUNTS ----------------
#         cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0, 255, 255), 2)

#         y = 30
#         for cls in count_in:
#             cv2.putText(frame,
#                         f"{cls} IN:{count_in[cls]} OUT:{count_out[cls]}",
#                         (10, y),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.55,
#                         (255, 255, 255),
#                         2)
#             y += 22

#         # ---------------- VIDEO SAVE ----------------
#         if vid_writer is None:
#             out_path = save_dir / "result.mp4"
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             vid_writer = cv2.VideoWriter(
#                 str(out_path),
#                 fourcc,
#                 vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25,
#                 (frame.shape[1], frame.shape[0])
#             )

#         vid_writer.write(frame)
#         cv2.imshow("YOLOv5 Vertical Line Counting", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     if vid_writer:
#         vid_writer.release()
#     cv2.destroyAllWindows()

#     print(f"\n✅ Results saved to: {save_dir.resolve()}")


# # ---------------- ARGUMENTS ----------------
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", type=str, required=True)
#     parser.add_argument("--source", type=str, required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/count")
#     parser.add_argument("--name", default="exp")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))


###working code


# import argparse
# import sys
# import time
# import traceback
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import os
# import pathlib
# from tqdm import tqdm
# import signal

# # ---------------- WINDOWS PATH FIX ----------------
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # ---------------- ROOT ----------------
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# # ---------------- YOLOv5 ----------------
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# # ---------------- COUNT CONFIG ----------------
# LINE_X = 650
# OFFSET = 10
# BUFFER_SECONDS = 10

# STOP_REQUESTED = False


# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Exit requested — saving video safely...")


# signal.signal(signal.SIGINT, request_stop)
# signal.signal(signal.SIGTERM, request_stop)


# # ---------------- CLASS COLOR ----------------
# def get_class_color(cls_name):
#     np.random.seed(abs(hash(cls_name)) % (2**32))
#     return tuple(int(c) for c in np.random.randint(40, 255, 3))


# # --------------------------------------------------
# @smart_inference_mode()
# def run(
#     weights,
#     source,
#     imgsz=640,
#     conf_thres=0.25,
#     iou_thres=0.45,
#     device="",
#     project="runs/count",
#     name="exp"
# ):
#     vid_writer = None
#     frame_idx = 0

#     try:
#         # ---------------- VALIDATION ----------------
#         if not os.path.exists(weights):
#             raise FileNotFoundError(f"Weights not found: {weights}")

#         is_webcam = source.isnumeric()
#         if not is_webcam and not source.startswith("rtsp") and not os.path.exists(source):
#             raise FileNotFoundError(f"Source not found: {source}")

#         # ---------------- OUTPUT ----------------
#         save_dir = Path(project) / name
#         save_dir.mkdir(parents=True, exist_ok=True)
#         out_video = save_dir / "result.mp4"

#         # ---------------- DEVICE & MODEL ----------------
#         device = select_device(device)
#         model = DetectMultiBackend(weights, device=device)
#         stride, names = model.stride, model.names
#         imgsz = check_img_size(imgsz, s=stride)
#         model.warmup(imgsz=(1, 3, imgsz, imgsz))

#         # ---------------- DATASET ----------------
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#             if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

#         # ---------------- TOTAL FRAMES ----------------
#         total_frames = None
#         if not is_webcam and hasattr(dataset, "cap") and dataset.cap:
#             tf = int(dataset.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             total_frames = tf if tf > 0 else None

#         tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#         # ---------------- COUNTERS ----------------
#         count_in = {v: 0 for v in names.values()}
#         count_out = {v: 0 for v in names.values()}
#         track_last_side = {}
#         track_class = {}
#         last_count_time = {}

#         pbar = tqdm(dataset, total=total_frames, unit="frame", desc="Inference")

#         for data in pbar:
#             if STOP_REQUESTED:
#                 break

#             try:
#                 frame_idx += 1
#                 path, im, im0s, vid_cap, s = data

#                 if im is None or im0s is None:
#                     continue

#                 im = torch.from_numpy(im).to(device).float() / 255.0
#                 if im.ndim == 3:
#                     im = im[None]

#                 pred = model(im)
#                 pred = non_max_suppression(pred, conf_thres, iou_thres)

#                 frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#                 detections = []

#                 if pred and len(pred[0]):
#                     pred[0][:, :4] = scale_boxes(
#                         im.shape[2:], pred[0][:, :4], frame.shape
#                     ).round()

#                     for *xyxy, conf, cls in pred[0]:
#                         x1, y1, x2, y2 = map(int, xyxy)
#                         detections.append([x1, y1, x2, y2, conf.item(), int(cls)])

#                 sort_input = np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
#                 tracks = tracker.update(sort_input)
#                 now = time.time()

#                 for trk in tracks.astype(int):
#                     x1, y1, x2, y2, track_id = trk
#                     cx = (x1 + x2) // 2

#                     best_iou, cls_name = 0, "unknown"
#                     for d in detections:
#                         xx1, yy1 = max(x1, d[0]), max(y1, d[1])
#                         xx2, yy2 = min(x2, d[2]), min(y2, d[3])
#                         inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
#                         area1 = max(1, (x2 - x1) * (y2 - y1))
#                         area2 = max(1, (d[2] - d[0]) * (d[3] - d[1]))
#                         iou = inter / (area1 + area2 - inter + 1e-6)
#                         if iou > best_iou:
#                             best_iou = iou
#                             cls_name = names.get(d[5], "unknown")

#                     if best_iou > 0.2:
#                         track_class[track_id] = cls_name
#                     else:
#                         cls_name = track_class.get(track_id, "unknown")

#                     if cx < LINE_X - OFFSET:
#                         side = "left"
#                     elif cx > LINE_X + OFFSET:
#                         side = "right"
#                     else:
#                         side = "buffer"

#                     prev = track_last_side.get(track_id)
#                     last_time = last_count_time.get(track_id, 0)

#                     if prev and side != prev and (now - last_time) > BUFFER_SECONDS:
#                         if prev == "left" and side == "right":
#                             count_in[cls_name] += 1
#                             last_count_time[track_id] = now
#                         elif prev == "right" and side == "left":
#                             count_out[cls_name] += 1
#                             last_count_time[track_id] = now

#                     if side != "buffer":
#                         track_last_side[track_id] = side

#                     color = get_class_color(cls_name)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, f"{cls_name} ID:{track_id}",
#                                 (x1, max(20, y1 - 6)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#                 cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0, 255, 255), 2)

#                 y = 30
#                 for cls in count_in:
#                     cv2.putText(frame,
#                                 f"{cls} IN:{count_in[cls]} OUT:{count_out[cls]}",
#                                 (10, y),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.55,
#                                 get_class_color(cls), 2)
#                     y += 22

#                 frame_text = (
#                     f"Frame: {frame_idx}/{total_frames}"
#                     if total_frames else f"Frame: {frame_idx}"
#                 )
#                 cv2.putText(frame, frame_text,
#                             (frame.shape[1] - 260, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                             (0, 255, 255), 2)

#                 if vid_writer is None:
#                     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#                     fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#                     vid_writer = cv2.VideoWriter(
#                         str(out_video), fourcc, fps,
#                         (frame.shape[1], frame.shape[0])
#                     )

#                 vid_writer.write(frame)
#                 cv2.imshow("YOLOv5 Safe Vertical Counter", frame)

#                 if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
#                     request_stop()

#             except Exception as fe:
#                 print("⚠ Frame error:", fe)
#                 traceback.print_exc()
#                 continue

#     finally:
#         if vid_writer:
#             vid_writer.release()
#             print(f"\n✅ Video saved: {out_video.resolve()}")

#         print(f"📊 Total processed frames: {frame_idx}")
#         cv2.destroyAllWindows()


# # ---------------- CLI ----------------
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", type=str, required=True)
#     parser.add_argument("--source", type=str, required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/count")
#     parser.add_argument("--name", default="exp")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))


###########################################working good_based version_ v1


# import argparse
# import sys
# import time
# import traceback
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import os
# import pathlib
# from tqdm import tqdm
# import signal

# # ================= WINDOWS PATH FIX =================
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # ================= ROOT =================
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# # ================= YOLOv5 =================
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# # ================= CONFIG =================
# LINE_X = 800  #650 sahithi
# OFFSET = 10
# BUFFER_SECONDS = 10
# STOP_REQUESTED = False


# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Exit requested — finalizing videos safely...")


# signal.signal(signal.SIGINT, request_stop)
# signal.signal(signal.SIGTERM, request_stop)


# # ================= UTILITIES =================
# def get_class_color(cls_name):
#     np.random.seed(abs(hash(cls_name)) % (2**32))
#     return tuple(int(c) for c in np.random.randint(40, 255, 3))


# def get_next_video_path(save_dir, prefix):
#     save_dir.mkdir(parents=True, exist_ok=True)
#     existing = list(save_dir.glob(f"{prefix}_*.mp4"))
#     if not existing:
#         return save_dir / f"{prefix}_0001.mp4"
#     nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
#     idx = max(nums) + 1 if nums else 1
#     return save_dir / f"{prefix}_{idx:04d}.mp4"


# # ==================================================
# @smart_inference_mode()
# def run(
#     weights,
#     source,
#     imgsz=640,
#     conf_thres=0.25,
#     iou_thres=0.45,
#     device="",
#     project="runs/count",
#     name="exp"
# ):
#     raw_writer = None
#     ann_writer = None
#     frame_idx = 0

#     def draw_text_with_gold_box(
#         img,
#         text,
#         pos,
#         font=cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale=0.75,
#         text_color=(255, 255, 255),
#         bg_color=(0, 0, 0),
#         border_color=(0, 215, 255),  # GOLD (BGR)
#         thickness=2,
#         padding=8,
#         border_thickness=2
#     ):
#         x, y = pos

#         (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#         h += baseline

#         top_left = (x - padding, y - h - padding)
#         bottom_right = (x + w + padding, y + padding)

#         # Background
#         cv2.rectangle(
#             img,
#             top_left,
#             bottom_right,
#             bg_color,
#             -1
#         )

#         # Gold border
#         cv2.rectangle(
#             img,
#             top_left,
#             bottom_right,
#             border_color,
#             border_thickness
#         )

#         # Text
#         cv2.putText(
#             img,
#             text,
#             (x, y),
#             font,
#             font_scale,
#             text_color,
#             thickness,
#             cv2.LINE_AA
#         )

#     try:
#         # -------- VALIDATION --------
#         if not os.path.exists(weights):
#             raise FileNotFoundError(f"Weights not found: {weights}")

#         is_webcam = source.isnumeric()
#         if not is_webcam and not source.startswith("rtsp") and not os.path.exists(source):
#             raise FileNotFoundError(f"Source not found: {source}")

#         save_dir = Path(project) / name
#         raw_video = get_next_video_path(save_dir, "raw")
#         ann_video = get_next_video_path(save_dir, "annotated")

#         # -------- MODEL --------
#         device = select_device(device)
#         model = DetectMultiBackend(weights, device=device)
#         stride, names = model.stride, model.names
#         imgsz = check_img_size(imgsz, s=stride)
#         model.warmup(imgsz=(1, 3, imgsz, imgsz))

#         # -------- DATASET --------
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#             if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

#         total_frames = None
#         if not is_webcam and hasattr(dataset, "cap") and dataset.cap:
#             tf = int(dataset.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             total_frames = tf if tf > 0 else None

#         tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#         count_in = {v: 0 for v in names.values()}
#         count_out = {v: 0 for v in names.values()}
#         track_last_side = {}
#         track_class = {}
#         last_count_time = {}

#         pbar = tqdm(dataset, total=total_frames, unit="frame", desc="Inference")

#         for data in pbar:
#             if STOP_REQUESTED:
#                 break

#             try:
#                 frame_idx += 1
#                 path, im, im0s, vid_cap, s = data
#                 if im is None or im0s is None:
#                     continue

#                 raw_frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#                 frame = raw_frame.copy()

#                 im = torch.from_numpy(im).to(device).float() / 255.0
#                 if im.ndim == 3:
#                     im = im[None]

#                 pred = model(im)
#                 pred = non_max_suppression(pred, conf_thres, iou_thres)

#                 detections = []
#                 if pred and len(pred[0]):
#                     pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], frame.shape).round()
#                     for *xyxy, conf, cls in pred[0]:
#                         x1, y1, x2, y2 = map(int, xyxy)
#                         detections.append([x1, y1, x2, y2, conf.item(), int(cls)])

#                 tracks = tracker.update(
#                     np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
#                 )

#                 now = time.time()

#                 for x1, y1, x2, y2, track_id in tracks.astype(int):
#                     cx = (x1 + x2) // 2
#                     best_iou, cls_name = 0, "unknown"

#                     for d in detections:
#                         xx1, yy1 = max(x1, d[0]), max(y1, d[1])
#                         xx2, yy2 = min(x2, d[2]), min(y2, d[3])
#                         inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
#                         area1 = max(1, (x2 - x1) * (y2 - y1))
#                         area2 = max(1, (d[2] - d[0]) * (d[3] - d[1]))
#                         iou = inter / (area1 + area2 - inter + 1e-6)
#                         if iou > best_iou:
#                             best_iou = iou
#                             cls_name = names.get(d[5], "unknown")

#                     track_class.setdefault(track_id, cls_name)

#                     if cx < LINE_X - OFFSET:
#                         side = "left"
#                     elif cx > LINE_X + OFFSET:
#                         side = "right"
#                     else:
#                         side = "buffer"

#                     prev = track_last_side.get(track_id)
#                     last_time = last_count_time.get(track_id, 0)

#                     if prev and side != prev and (now - last_time) > BUFFER_SECONDS:
#                         if prev == "left" and side == "right":
#                             count_in[cls_name] += 1
#                             last_count_time[track_id] = now
#                         elif prev == "right" and side == "left":
#                             count_out[cls_name] += 1
#                             last_count_time[track_id] = now

#                     if side != "buffer":
#                         track_last_side[track_id] = side

#                     color = get_class_color(cls_name)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, f"{cls_name} ID:{track_id}",
#                                 (x1, max(20, y1 - 6)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#                 cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0, 255, 255), 2)

#                 y = 45
#                 LINE_GAP = 32
#                 CLASS_GAP = 18

#                 for cls in count_in:
#                     draw_text_with_gold_box(
#                         frame,
#                         f"{cls}  IN:{count_in[cls]}  OUT:{count_out[cls]}",
#                         (15, y),
#                         font_scale=0.75,
#                         text_color=get_class_color(cls),   # per-class color
#                         bg_color=(0, 0, 0),                # black box
#                         border_color=(0, 215, 255),         # gold outline
#                         border_thickness=2
#                     )
#                     y += LINE_GAP + CLASS_GAP


#                 if raw_writer is None:
#                     h, w = frame.shape[:2]
#                     fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#                     raw_writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
#                     ann_writer = cv2.VideoWriter(str(ann_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#                 raw_writer.write(raw_frame)
#                 ann_writer.write(frame)

#                 cv2.imshow("YOLOv5 Raw + Annotated", frame)
#                 if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
#                     request_stop()

#             except Exception:
#                 traceback.print_exc()
#                 continue

#     finally:
#         if raw_writer:
#             raw_writer.release()
#         if ann_writer:
#             ann_writer.release()

#         print(f"\n✅ Raw video saved: {raw_video}")
#         print(f"✅ Annotated video saved: {ann_video}")
#         print(f"📊 Total frames processed: {frame_idx}")
#         cv2.destroyAllWindows()


# # ================= CLI =================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", required=True)
#     parser.add_argument("--source", required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/count")
#     parser.add_argument("--name", default="exp")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))


############################################ working with properly


# import argparse
# import sys
# import time
# import traceback
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import os
# import pathlib
# from tqdm import tqdm
# import signal

# # ================= WINDOWS PATH FIX =================
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # ================= ROOT =================
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# # ================= YOLOv5 =================
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# # ================= CONFIG =================
# LINE_X = 800
# BUFFER_PX = 100
# BUFFER_SECONDS = 10
# STOP_REQUESTED = False


# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Exit requested — finalizing videos safely...")


# signal.signal(signal.SIGINT, request_stop)
# signal.signal(signal.SIGTERM, request_stop)

# # ================= UTILITIES =================
# def get_class_color(cls_name):
#     np.random.seed(abs(hash(cls_name)) % (2**32))
#     return tuple(int(c) for c in np.random.randint(40, 255, 3))


# def get_zone(cx):
#     if cx < LINE_X - BUFFER_PX:
#         return "left"
#     elif cx > LINE_X + BUFFER_PX:
#         return "right"
#     else:
#         return "buffer"


# def get_next_video_path(save_dir, prefix):
#     save_dir.mkdir(parents=True, exist_ok=True)
#     existing = list(save_dir.glob(f"{prefix}_*.mp4"))
#     if not existing:
#         return save_dir / f"{prefix}_0001.mp4"
#     nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
#     idx = max(nums) + 1 if nums else 1
#     return save_dir / f"{prefix}_{idx:04d}.mp4"


# def draw_text_with_gold_box(
#     img,
#     text,
#     pos,
#     font=cv2.FONT_HERSHEY_SIMPLEX,
#     font_scale=0.75,
#     text_color=(255, 255, 255),
#     bg_color=(0, 0, 0),
#     border_color=(0, 215, 255),
#     thickness=2,
#     padding=8,
#     border_thickness=2
# ):
#     x, y = pos
#     (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#     h += baseline

#     top_left = (x - padding, y - h - padding)
#     bottom_right = (x + w + padding, y + padding)

#     cv2.rectangle(img, top_left, bottom_right, bg_color, -1)
#     cv2.rectangle(img, top_left, bottom_right, border_color, border_thickness)

#     cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


# # ==================================================
# @smart_inference_mode()
# def run(
#     weights,
#     source,
#     imgsz=640,
#     conf_thres=0.25,
#     iou_thres=0.45,
#     device="",
#     project="runs/count",
#     name="exp"
# ):
#     raw_writer = None
#     ann_writer = None
#     frame_idx = 0

#     try:
#         if not os.path.exists(weights):
#             raise FileNotFoundError(f"Weights not found: {weights}")

#         is_webcam = source.isnumeric()
#         if not is_webcam and not source.startswith("rtsp") and not os.path.exists(source):
#             raise FileNotFoundError(f"Source not found: {source}")

#         save_dir = Path(project) / name
#         raw_video = get_next_video_path(save_dir, "raw")
#         ann_video = get_next_video_path(save_dir, "annotated")

#         device = select_device(device)
#         model = DetectMultiBackend(weights, device=device)
#         stride, names = model.stride, model.names
#         imgsz = check_img_size(imgsz, s=stride)
#         model.warmup(imgsz=(1, 3, imgsz, imgsz))

#         dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#             if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

#         tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#         count_in = {v: 0 for v in names.values()}
#         count_out = {v: 0 for v in names.values()}
#         track_last_side = {}
#         track_class = {}
#         last_count_time = {}

#         for data in tqdm(dataset, unit="frame", desc="Inference"):
#             if STOP_REQUESTED:
#                 break

#             try:
#                 frame_idx += 1
#                 path, im, im0s, vid_cap, s = data

#                 raw_frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#                 frame = raw_frame.copy()

#                 im = torch.from_numpy(im).to(device).float() / 255.0
#                 if im.ndim == 3:
#                     im = im[None]

#                 pred = model(im)
#                 pred = non_max_suppression(pred, conf_thres, iou_thres)

#                 detections = []
#                 if pred and len(pred[0]):
#                     pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], frame.shape).round()
#                     for *xyxy, conf, cls in pred[0]:
#                         x1, y1, x2, y2 = map(int, xyxy)
#                         detections.append([x1, y1, x2, y2, conf.item(), int(cls)])

#                 tracks = tracker.update(
#                     np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
#                 )

#                 now = time.time()

#                 for x1, y1, x2, y2, track_id in tracks.astype(int):
#                     cx = (x1 + x2) // 2
#                     cls_name = track_class.get(track_id, "unknown")

#                     if track_id not in track_class:
#                         best_iou = 0
#                         for d in detections:
#                             xx1, yy1 = max(x1, d[0]), max(y1, d[1])
#                             xx2, yy2 = min(x2, d[2]), min(y2, d[3])
#                             inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
#                             area1 = (x2 - x1) * (y2 - y1)
#                             area2 = (d[2] - d[0]) * (d[3] - d[1])
#                             iou = inter / (area1 + area2 - inter + 1e-6)
#                             if iou > best_iou:
#                                 best_iou = iou
#                                 cls_name = names.get(d[5], "unknown")
#                         track_class[track_id] = cls_name

#                     zone = get_zone(cx)
#                     prev_zone = track_last_side.get(track_id)
#                     last_time = last_count_time.get(track_id, 0)

#                     if prev_zone and zone != prev_zone and (now - last_time) > BUFFER_SECONDS:
#                         if prev_zone == "left" and zone in ["right", "buffer"]:
#                             count_in[cls_name] += 1
#                             last_count_time[track_id] = now
#                             track_last_side[track_id] = "right"

#                         elif prev_zone == "right" and zone in ["left", "buffer"]:
#                             count_out[cls_name] += 1
#                             last_count_time[track_id] = now
#                             track_last_side[track_id] = "left"

#                     if zone in ["left", "right"]:
#                         track_last_side[track_id] = zone

#                     color = get_class_color(cls_name)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, f"{cls_name} ID:{track_id}",
#                                 (x1, max(20, y1 - 6)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#                 # Draw lines
#                 cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0, 255, 255), 2)
#                 cv2.line(frame, (LINE_X - BUFFER_PX, 0), (LINE_X - BUFFER_PX, frame.shape[0]), (255, 215, 0), 1)
#                 cv2.line(frame, (LINE_X + BUFFER_PX, 0), (LINE_X + BUFFER_PX, frame.shape[0]), (255, 215, 0), 1)

#                 y = 40
#                 for cls in count_in:
#                     draw_text_with_gold_box(
#                         frame,
#                         f"{cls}  IN:{count_in[cls]}  OUT:{count_out[cls]}",
#                         (15, y),
#                         text_color=get_class_color(cls)
#                     )
#                     y += 35

#                 if raw_writer is None:
#                     h, w = frame.shape[:2]
#                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                     if fps is None or fps <= 1:
#                         fps = 25

#                     raw_writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
#                     ann_writer = cv2.VideoWriter(str(ann_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#                 raw_writer.write(raw_frame)
#                 ann_writer.write(frame)

#                 cv2.imshow("YOLOv5 Counting", frame)
#                 if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
#                     request_stop()

#             except Exception:
#                 traceback.print_exc()
#                 continue

#     finally:
#         if raw_writer:
#             raw_writer.release()
#         if ann_writer:
#             ann_writer.release()

#         cv2.destroyAllWindows()
#         print(f"\n✅ Raw video saved: {raw_video}")
#         print(f"✅ Annotated video saved: {ann_video}")
#         print(f"📊 Frames processed: {frame_idx}")


# # ================= CLI =================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", required=True)
#     parser.add_argument("--source", required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/count")
#     parser.add_argument("--name", default="exp")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))


###########################################        working  code     ########################################

import argparse
import os
import pathlib
import signal
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ================= WINDOWS PATH FIX =================
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ================= ROOT =================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ================= YOLOv5 =================
from sort.sort import Sort

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode

# ================= CONFIG =================
LINE_X = 800
BUFFER_PX = 100
BUFFER_SECONDS = 10
STOP_REQUESTED = False
# ================= DETECTED CLASSES =================
detected_classes = set()
detected_class_counts = {}


def request_stop(sig=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠ Exit requested — finalizing videos safely...")


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


# ================= UTILITIES =================
def get_class_color(cls_name):
    np.random.seed(abs(hash(cls_name)) % (2**32))
    return tuple(int(c) for c in np.random.randint(40, 255, 3))


def get_zone(cx):
    if cx < LINE_X - BUFFER_PX:
        return "left"
    elif cx > LINE_X + BUFFER_PX:
        return "right"
    else:
        return "buffer"


def get_next_video_path(save_dir, prefix):
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = list(save_dir.glob(f"{prefix}_*.mp4"))
    if not existing:
        return save_dir / f"{prefix}_0001.mp4"
    nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
    idx = max(nums) + 1 if nums else 1
    return save_dir / f"{prefix}_{idx:04d}.mp4"


def draw_text_with_gold_box(
    img,
    text,
    pos,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.75,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    border_color=(0, 215, 255),
    thickness=2,
    padding=8,
    border_thickness=2,
):
    x, y = pos
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h += baseline

    top_left = (x - padding, y - h - padding)
    bottom_right = (x + w + padding, y + padding)

    cv2.rectangle(img, top_left, bottom_right, bg_color, -1)
    cv2.rectangle(img, top_left, bottom_right, border_color, border_thickness)

    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


# ==================================================
@smart_inference_mode()
def run(weights, source, imgsz=640, conf_thres=0.25, iou_thres=0.45, device="", project="runs/count", name="exp"):
    raw_writer = None
    ann_writer = None
    frame_idx = 0

    try:
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")

        is_webcam = source.isnumeric()
        if not is_webcam and not source.startswith("rtsp") and not os.path.exists(source):
            raise FileNotFoundError(f"Source not found: {source}")

        save_dir = Path(project) / name
        raw_video = get_next_video_path(save_dir, "raw")
        ann_video = get_next_video_path(save_dir, "annotated")

        device = select_device(device)
        model = DetectMultiBackend(weights, device=device)
        stride, names = model.stride, model.names
        imgsz = check_img_size(imgsz, s=stride)
        model.warmup(imgsz=(1, 3, imgsz, imgsz))

        dataset = (
            LoadStreams(source, img_size=imgsz, stride=stride)
            if is_webcam
            else LoadImages(source, img_size=imgsz, stride=stride)
        )

        tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

        count_in = {v: 0 for v in names.values()}
        count_out = {v: 0 for v in names.values()}
        track_last_side = {}
        track_class = {}
        last_count_time = {}

        for data in tqdm(dataset, unit="frame", desc="Inference"):
            if STOP_REQUESTED:
                break

            try:
                frame_idx += 1
                _path, im, im0s, vid_cap, _s = data

                raw_frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
                frame = raw_frame.copy()

                im = torch.from_numpy(im).to(device).float() / 255.0
                if im.ndim == 3:
                    im = im[None]

                pred = model(im)
                pred = non_max_suppression(pred, conf_thres, iou_thres)

                detections = []
                if pred and len(pred[0]):
                    pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], frame.shape).round()

                    for *xyxy, conf, cls in pred[0]:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cls_id = int(cls)
                        cls_name = names.get(cls_id, "unknown")

                        detections.append([x1, y1, x2, y2, conf.item(), cls_id])

                        # # ✅ ONLY actual detected classes
                        # detected_classes.add(cls_name)
                        # detected_class_counts[cls_name] = detected_class_counts.get(cls_name, 0) + 1

                tracks = tracker.update(np.array([d[:5] for d in detections]) if detections else np.empty((0, 5)))

                now = time.time()

                for x1, y1, x2, y2, track_id in tracks.astype(int):
                    cx = (x1 + x2) // 2
                    cls_name = track_class.get(track_id, "unknown")

                    if track_id not in track_class:
                        best_iou = 0
                        for d in detections:
                            xx1, yy1 = max(x1, d[0]), max(y1, d[1])
                            xx2, yy2 = min(x2, d[2]), min(y2, d[3])
                            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                            area1 = (x2 - x1) * (y2 - y1)
                            area2 = (d[2] - d[0]) * (d[3] - d[1])
                            iou = inter / (area1 + area2 - inter + 1e-6)
                            if iou > best_iou:
                                best_iou = iou
                                cls_name = names.get(d[5], "unknown")
                        track_class[track_id] = cls_name

                    zone = get_zone(cx)
                    prev_zone = track_last_side.get(track_id)
                    last_time = last_count_time.get(track_id, 0)

                    if prev_zone and zone != prev_zone and (now - last_time) > BUFFER_SECONDS:
                        if prev_zone == "left" and zone in ["right", "buffer"]:
                            count_in[cls_name] += 1

                            # ✅ update ONLY when counted
                            detected_classes.add(cls_name)
                            detected_class_counts[cls_name] = detected_class_counts.get(cls_name, 0) + 1

                            last_count_time[track_id] = now
                            track_last_side[track_id] = "right"

                        elif prev_zone == "right" and zone in ["left", "buffer"]:
                            count_out[cls_name] += 1

                            # ✅ update ONLY when counted
                            detected_classes.add(cls_name)
                            detected_class_counts[cls_name] = detected_class_counts.get(cls_name, 0) + 1

                            last_count_time[track_id] = now
                            track_last_side[track_id] = "left"

                    if zone in ["left", "right"]:
                        track_last_side[track_id] = zone

                    color = get_class_color(cls_name)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(
                        frame,
                        f"{cls_name} ID:{track_id}",
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        1,
                    )

                # Draw lines
                cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0, 255, 255), 2)
                cv2.line(frame, (LINE_X - BUFFER_PX, 0), (LINE_X - BUFFER_PX, frame.shape[0]), (255, 215, 0), 1)
                cv2.line(frame, (LINE_X + BUFFER_PX, 0), (LINE_X + BUFFER_PX, frame.shape[0]), (255, 215, 0), 1)

                y = 30
                for cls in sorted(detected_classes):
                    draw_text_with_gold_box(
                        frame,
                        f"{cls}  IN:{count_in.get(cls, 0)}  OUT:{count_out.get(cls, 0)}",
                        (15, y),
                        font_scale=1,  # ⬅ smaller text
                        thickness=1,  # ⬅ thinner text
                        padding=3,  # ⬅ smaller box
                        border_thickness=1,  # ⬅ thinner border
                        text_color=get_class_color(cls),
                    )
                    y += 26  # ⬅ tighter vertical spacing

                if raw_writer is None:
                    h, w = frame.shape[:2]
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    if fps is None or fps <= 1:
                        fps = 25

                    raw_writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    ann_writer = cv2.VideoWriter(str(ann_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

                raw_writer.write(raw_frame)
                ann_writer.write(frame)

                cv2.imshow("YOLOv5 Counting", frame)
                if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                    request_stop()

            except Exception:
                traceback.print_exc()
                continue

    finally:
        print("\n🧠 Detected Classes Summary (YOLO Detections):")
        if detected_class_counts:
            for cls, cnt in sorted(detected_class_counts.items()):
                print(f" - {cls}: {cnt}")
        else:
            print(" - No classes detected")
        if raw_writer:
            raw_writer.release()
        if ann_writer:
            ann_writer.release()

        cv2.destroyAllWindows()
        print(f"\n✅ Raw video saved: {raw_video}")
        print(f"✅ Annotated video saved: {ann_video}")
        print(f"📊 Frames processed: {frame_idx}")


def help():
    print("""
====================================================================
YOLOv5 + SORT Based Object Counting Script (Line Crossing)
====================================================================

This script performs real-time object detection, tracking, and
bi-directional counting (IN / OUT) using YOLOv5 and SORT tracking.
It supports video files, webcam, and RTSP streams and saves both
raw and annotated output videos safely.

--------------------------------------------------------------------
HIGH-LEVEL WORKFLOW
--------------------------------------------------------------------

1. Load YOLOv5 model and SORT tracker
2. Read frames from image / video / webcam / RTSP
3. Detect objects using YOLOv5
4. Track objects across frames using SORT (Track IDs)
5. Assign each tracked object a class label
6. Detect line-crossing events with buffer logic
7. Count IN and OUT movements per class
8. Draw bounding boxes, IDs, lines, and counters
9. Save raw and annotated videos safely
10. Handle graceful exit (Ctrl+C / SIGTERM)

--------------------------------------------------------------------
KEY COMPONENTS EXPLAINED
--------------------------------------------------------------------

WINDOWS PATH FIX
----------------
Ensures YOLOv5 works correctly on Windows by mapping PosixPath
to WindowsPath.

ROOT & IMPORTS
--------------
• Dynamically sets project root
• Imports YOLOv5 detection utilities
• Imports SORT tracker for object tracking

--------------------------------------------------------------------
CONFIGURATION PARAMETERS
--------------------------------------------------------------------

LINE_X = 800
    • X-coordinate of the virtual counting line

BUFFER_PX = 100
    • Pixel buffer on both sides of the line to avoid false counts

BUFFER_SECONDS = 10
    • Minimum time gap before the same object can be counted again

STOP_REQUESTED
    • Global flag for safe shutdown

--------------------------------------------------------------------
SIGNAL HANDLING
--------------------------------------------------------------------

• Ctrl+C or kill signal triggers a safe shutdown
• Videos are finalized and saved correctly
• Prevents corrupted video files

--------------------------------------------------------------------
DETECTION & TRACKING
--------------------------------------------------------------------

YOLOv5:
-------
• Performs object detection
• Outputs bounding boxes, confidence, and class IDs
• Uses Non-Max Suppression to remove overlaps

SORT Tracker:
-------------
• Assigns unique Track IDs
• Maintains object identity across frames
• Helps avoid double counting

--------------------------------------------------------------------
LINE-CROSSING LOGIC
--------------------------------------------------------------------

Each object is classified into zones based on center X:

• left     : cx < LINE_X - BUFFER_PX
• buffer   : inside buffer zone
• right    : cx > LINE_X + BUFFER_PX

Counting Rules:
---------------
• left  → right / buffer  → IN count
• right → left  / buffer  → OUT count
• BUFFER_SECONDS prevents duplicate counting

--------------------------------------------------------------------
CLASS MANAGEMENT
--------------------------------------------------------------------

• Each Track ID is permanently assigned one class
• Class is matched using highest IoU with detections
• Counts are maintained per class:
    - count_in[class]
    - count_out[class]

--------------------------------------------------------------------
VISUALIZATION
--------------------------------------------------------------------

• Bounding boxes with class + Track ID
• Color-coded per class
• Central counting line and buffer lines
• On-screen counter panel with:
    Class | IN count | OUT count

--------------------------------------------------------------------
VIDEO OUTPUT
--------------------------------------------------------------------

Two videos are saved:

1. Raw Video
   - Original frames (no annotations)

2. Annotated Video
   - Bounding boxes, IDs, lines, and counters

Features:
---------
• Auto-increment filenames (raw_0001.mp4, raw_0002.mp4, ...)
• Safe writer release on exit
• FPS auto-detection fallback

--------------------------------------------------------------------
ERROR HANDLING
--------------------------------------------------------------------

• Per-frame try/except (script does not crash)
• Traceback printed for debugging
• Continues processing remaining frames

--------------------------------------------------------------------
FINAL SUMMARY
--------------------------------------------------------------------

On exit, the script prints:
• Detected classes
• Total counts per class
• Output video paths
• Total frames processed

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------

python count.py \\
    --weights yolov5s.pt \\
    --source video.mp4 \\
    --imgsz 640 \\
    --conf-thres 0.25 \\
    --iou-thres 0.45

--------------------------------------------------------------------
DESIGNED FOR
--------------------------------------------------------------------

✔ Traffic counting
✔ Entry / exit monitoring
✔ Surveillance analytics
✔ Production-safe video processing

====================================================================
""")
    sys.exit(0)


# ================= CLI =================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/count")
    parser.add_argument("--name", default="exp")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    help()
    run(**vars(opt))
