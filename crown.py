# import argparse
# import sys
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import os
# import pathlib
# import signal
# from tqdm import tqdm

# # ================= WINDOWS PATH FIX =================
# _temp = pathlib.PosixPath
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
# IOU_ALERT_THRES = 0.15
# DIST_ALERT_PX = 40

# STOP_REQUESTED = False

# # ================= SIGNAL =================
# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Exit requested")

# signal.signal(signal.SIGINT, request_stop)
# signal.signal(signal.SIGTERM, request_stop)

# # ================= UTILS =================
# def compute_iou(a, b):
#     xA, yA = max(a[0], b[0]), max(a[1], b[1])
#     xB, yB = min(a[2], b[2]), min(a[3], b[3])
#     inter = max(0, xB - xA) * max(0, yB - yA)
#     if inter == 0:
#         return 0.0
#     areaA = (a[2] - a[0]) * (a[3] - a[1])
#     areaB = (b[2] - b[0]) * (b[3] - b[1])
#     return inter / (areaA + areaB - inter + 1e-6)

# def center_dist(a, b):
#     cx1, cy1 = (a[0] + a[2]) // 2, (a[1] + a[3]) // 2
#     cx2, cy2 = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
#     return np.hypot(cx1 - cx2, cy1 - cy2)

# # ==================================================
# @smart_inference_mode()
# def run(weights, source, imgsz=640, conf_thres=0.25, iou_thres=0.45,
#         device="", project="runs/count", name="exp"):

#     save_dir = Path(project) / name
#     save_dir.mkdir(parents=True, exist_ok=True)

#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device)
#     stride = model.stride
#     imgsz = check_img_size(imgsz, s=stride)
#     model.warmup(imgsz=(1, 3, imgsz, imgsz))

#     is_webcam = source.isnumeric() or source.startswith(("rtsp", "http"))
#     is_video_source = Path(source).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]

#     dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#         if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

#     tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)

#     vid_writer = None
#     frame_idx = 0

#     for path, im, im0s, vid_cap, _ in tqdm(dataset, desc="Inference"):

#         if STOP_REQUESTED:
#             break

#         frame_idx += 1
#         frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()

#         im = torch.from_numpy(im).to(device).float() / 255.0
#         if im.ndim == 3:
#             im = im.unsqueeze(0)

#         pred = non_max_suppression(model(im), conf_thres, iou_thres)

#         detections = []
#         if pred and len(pred[0]):
#             pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], frame.shape).round()
#             for *xyxy, conf, cls in pred[0]:
#                 detections.append([*map(int, xyxy), conf.item(), int(cls)])

#         # ================= OVERLAP CHECK =================
#         overlap = False
#         for i in range(len(detections)):
#             for j in range(i + 1, len(detections)):
#                 if detections[i][5] != detections[j][5]:
#                     if compute_iou(detections[i][:4], detections[j][:4]) > IOU_ALERT_THRES \
#                        or center_dist(detections[i][:4], detections[j][:4]) < DIST_ALERT_PX:
#                         overlap = True

#         # ================= TRACKING =================
#         tracks = tracker.update(
#             np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
#         )

#         for x1, y1, x2, y2, tid in tracks.astype(int):
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"ID:{tid}",
#                         (x1, max(15, y1 - 6)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

#         # ================= ALERT OVERLAY =================
#         if overlap:
#             cv2.putText(frame, "WARNING: OBJECTS OVERLAPPING",
#                         (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                         (0, 0, 255), 3, cv2.LINE_AA)

#         # ================= IMAGE SAVE =================
#         if not is_video_source and not is_webcam:
#             out_path = save_dir / f"{Path(path).stem}_{frame_idx:04d}.jpg"
#             cv2.imwrite(str(out_path), frame)
#             print(f"✅ Saved image: {out_path}")
#             continue

#         # ================= VIDEO SAVE (FIXED CODEC) =================
#         if vid_writer is None:
#             h, w = frame.shape[:2]
#             fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#             if fps <= 1:
#                 fps = 25
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # ✅ Fixed: was XVID (.avi)
#             out_path = str(save_dir / "annotated.mp4") # ✅ Fixed: was .avi
#             vid_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
#             print(f"🎥 Video writer initialized → {out_path}")

#         vid_writer.write(frame)

#         cv2.imshow("Overlap Detection", frame)
#         if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
#             request_stop()

#     # ================= CLEANUP =================
#     if vid_writer:
#         vid_writer.release()
#         print("✅ Video saved successfully → annotated.mp4")

#     cv2.destroyAllWindows()
#     print(f"📊 Total frames processed: {frame_idx}")


# # ================= CLI =================
# def parse_opt():
#     p = argparse.ArgumentParser()
#     p.add_argument("--weights", required=True)
#     p.add_argument("--source", required=True)
#     p.add_argument("--imgsz", type=int, default=640)
#     p.add_argument("--conf-thres", type=float, default=0.25)
#     p.add_argument("--iou-thres", type=float, default=0.45)
#     p.add_argument("--device", default="")
#     p.add_argument("--project", default="runs/count")
#     p.add_argument("--name", default="exp")
#     return p.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))


############### working for the crown counting task, adapted from track_count_line.py and overlap_alert.py, with a focus on counting objects crossing a line and maintaining a set of detected classes with counts. It saves both raw and annotated videos, and handles graceful exit on signals. The code is structured for clarity and maintainability, with utility functions for drawing text and determining zones.


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
    thickness=1,
    padding=5,
    border_thickness=1,
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
                        font_scale=0.6,  # ⬅ smaller text
                        thickness=1,
                        padding=2,  # ⬅ tighter box
                        border_thickness=1,
                        text_color=get_class_color(cls),
                    )
                    y += 20  # ⬅ tighter spacing                     # ⬅ tighter vertical spacing

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
    """============================================================ YOLOv5 + SORT LINE-CROSSING COUNTING SYSTEM.
    ============================================================.

    OVERVIEW
    --------
    This script performs real-time object detection, tracking, and directional counting using:

    • YOLOv5 for object detection
    • SORT for object tracking
    • Virtual vertical line for counting (LEFT ↔ RIGHT)
    • Buffered zone to avoid false triggers
    • Safe exit handling with signal support

    It generates TWO videos:
    1) Raw video (no annotations)
    2) Annotated video (boxes, IDs, counts)

    ------------------------------------------------------------
    INPUT SOURCES
    ------------------------------------------------------------
    • Video file
    • Image folder
    • Webcam (use source = "0")
    • RTSP stream

    ------------------------------------------------------------
    COUNTING LOGIC
    ------------------------------------------------------------
    • A vertical counting line is placed at:
        LINE_X = 800

    • Two buffer lines are added:
        LINE_X - BUFFER_PX
        LINE_X + BUFFER_PX

    • Zones:
        - LEFT    : object is fully left of buffer
        - RIGHT   : object is fully right of buffer
        - BUFFER  : object inside buffer region

    ------------------------------------------------------------
    DIRECTION RULES
    ------------------------------------------------------------
    LEFT → RIGHT → counted as IN RIGHT → LEFT → counted as OUT

    ------------------------------------------------------------
    DOUBLE COUNT PREVENTION
    ------------------------------------------------------------
    • Each track ID has:
        - last known zone
        - last count timestamp
    • Same object cannot be counted again for:
        BUFFER_SECONDS = 10 seconds

    ------------------------------------------------------------
    TRACKING
    ------------------------------------------------------------
    • SORT tracker assigns unique IDs
    • Class name is assigned using IoU matching
    • Tracking parameters:
        max_age     = 30
        min_hits    = 2
        iou_thresh  = 0.2

    ------------------------------------------------------------
    DISPLAY FEATURES
    ------------------------------------------------------------
    • Bounding boxes with class name + track ID
    • Vertical counting line + buffer lines
    • Per-class IN / OUT counters
    • Gold bordered text boxes
    • Live OpenCV display window

    ------------------------------------------------------------
    OUTPUT
    ------------------------------------------------------------
    Saved under:
        runs/count/<experiment_name>/

    Files:
        raw_XXXX.mp4        → original frames
        annotated_XXXX.mp4  → counted & labeled frames

    ------------------------------------------------------------
    SAFE EXIT
    ------------------------------------------------------------
    • Press 'q' or 'ESC'
    • Ctrl + C (SIGINT)
    • SIGTERM supported
    • Videos are finalized safely on exit

    ------------------------------------------------------------
    COMMAND LINE USAGE
    ------------------------------------------------------------
    python count.py --weights yolov5s.pt --source input.mp4 --imgsz 640 --conf-thres 0.25 --iou-thres 0.45 --device 0
    --project runs/count --name exp

    ------------------------------------------------------------
    ARGUMENT DETAILS
    ------------------------------------------------------------
    --weights Path to YOLOv5 weights (required) --source Input source (video / webcam / RTSP) --imgsz Inference image
    size (default: 640) --conf-thres Detection confidence threshold --iou-thres NMS IoU threshold --device CUDA device
    ID or 'cpu' --project Output root directory --name Experiment name

    ------------------------------------------------------------
    FINAL NOTES
    ------------------------------------------------------------
    ✔ Robust to dropped frames ✔ Prevents duplicate counting ✔ Handles graceful shutdown ✔ Suitable for traffic / people
    / object flow counting

    ============================================================
    """


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
    print(help.__doc__)
    run(**vars(opt))
