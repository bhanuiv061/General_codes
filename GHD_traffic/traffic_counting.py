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
    filter_classes = ["car", "motorcycle"]

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
    # Create a stable color for each class
    class_colors = {}
    for i, name in names.items():
        np.random.seed(i + 10)
        class_colors[name.lower()] = tuple(int(x) for x in np.random.randint(50, 255, size=3))

    dataset = (
        LoadStreams(source, img_size=imgsz, stride=stride)
        if source.isnumeric()
        else LoadImages(source, img_size=imgsz, stride=stride)
    )
    tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    track_class_memory, track_last_seen, track_side_memory = {}, {}, {}
    counted_ids = set()

    line_y = 250
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
                    names[int(cls)].strip().lower()

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

                color = class_colors.get(cls_name, (200, 200, 200))

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
            # Display counts with light background boxes
            start_y = 90
            line_gap = 70
            padding = 10

            for i, cls in enumerate(DISPLAY_CLASSES):
                y = start_y + i * line_gap

                text_in = f"{cls} IN : {counts_in[cls]:03d}"
                text_out = f"{cls} OUT: {counts_out[cls]:03d}"

                # Measure text size
                (w1, h1), _ = cv2.getTextSize(text_in, FONT, 0.7, 2)
                (w2, h2), _ = cv2.getTextSize(text_out, FONT, 0.7, 2)

                box_width = max(w1, w2) + padding * 2
                h1 + h2 + padding * 3

                base_color = class_colors.get(cls, (180, 180, 180))
                light_color = tuple(min(255, c + 80) for c in base_color)

                top_left = (15, y - h1 - padding)
                bottom_right = (15 + box_width, y + h2 + padding * 2)

                # Draw filled rectangle
                cv2.rectangle(im0, top_left, bottom_right, light_color, -1)

                # Put texts
                cv2.putText(im0, text_in, (15 + padding, y), FONT, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
                cv2.putText(im0, text_out, (15 + padding, y + 25), FONT, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

            # im0 = add_logo_top_left(im0)
            # im0 = add_diagonal_watermark(im0)
            video_writer.write(im0)

            cv2.imshow("Counting", im0)
            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("\n✅ Video saved:", video_path)


def about():
    """YOLOv5 + SORT Vehicle Tracking with Direction-Based Counting.
    =============================================================.

    This script performs real-time object detection, multi-object tracking, and directional vehicle counting based on
    objects crossing a virtual line.

    It combines:
    - YOLOv5 → Object Detection
    - SORT → Persistent Object Tracking with unique IDs
    - Line Crossing Logic → Determines IN / OUT movement
    - Memory Buffer → Prevents double counting & ID switching issues

    ----------------------------------------------------------------------
    🎯 PURPOSE
    ----------------------------------------------------------------------
    The system is designed to monitor vehicle flow by counting ONLY:

    ✔ Cars ✔ Motorcycles

    Vehicles are counted separately based on the direction they cross a predefined horizontal line in the video frame.

    IN direction = Vehicle moves from ABOVE the line to BELOW OUT direction = Vehicle moves from BELOW the line to ABOVE

    This is useful for:
    • Parking entry/exit analytics
    • Traffic flow monitoring
    • Toll gate vehicle counting
    • Smart city surveillance

    ----------------------------------------------------------------------
    🧠 CORE FEATURES
    ----------------------------------------------------------------------
    ✔ Real-time YOLOv5 vehicle detection ✔ SORT tracking with stable object IDs ✔ Memory buffer to avoid class
    flickering ✔ Prevents double counting using ID history ✔ Direction-based IN / OUT counting ✔ Counts ONLY cars and
    motorcycles ✔ Watermarked output video ✔ Live FPS and stats in terminal ✔ Auto-removal of stale IDs after buffer
    time

    ----------------------------------------------------------------------
    📊 COUNTING LOGIC
    ----------------------------------------------------------------------
    Each tracked vehicle maintains memory of:

    • Last known class label
    • Last known side of the counting line (above/below)
    • Last frame seen

    A vehicle is counted ONLY when:
    1. It has a valid tracking ID
    2. It crosses from one side of the line to the other
    3. It has not been counted before
    4. Its class is either "car" or "motorcycle"

    Movement Direction Detection:
    --------------------------------
    ABOVE → BELOW = IN count increases BELOW → ABOVE = OUT count increases

    Vehicles moving along the line without fully crossing are ignored.

    ----------------------------------------------------------------------
    🧠 MEMORY BUFFER SYSTEM
    ----------------------------------------------------------------------
    The system prevents tracking flicker and ID switching using:

    track_class_memory → Remembers last known class per ID track_last_seen → Stores last frame vehicle was detected
    track_side_memory → Stores last known side of the line counted_ids → Prevents duplicate counting

    BUFFER_FRAMES = 300 (~10 seconds)

    If a vehicle disappears briefly and reappears, it keeps its original ID and class label within this buffer time.

    ----------------------------------------------------------------------
    🚦 FILTERED CLASSES
    ----------------------------------------------------------------------
    The system detects all objects but counts ONLY:

    • car
    • motorcycle

    All other detected objects (person, bus, truck, etc.) are tracked but NOT included in IN/OUT counts.

    ----------------------------------------------------------------------
    📺 OUTPUT
    ----------------------------------------------------------------------
    • Annotated video saved to:
    runs/count/<name>/output.mp4

    • On-screen display shows:
    Car IN count Car OUT count Motorcycle IN count Motorcycle OUT count

    • Terminal displays live stats:
    FPS | Total Vehicles | Per-class IN/OUT breakdown

    ----------------------------------------------------------------------
    🛠 COMMAND LINE ARGUMENTS
    ----------------------------------------------------------------------
    --weights Path to YOLOv5 model (.pt) --source Video file, folder, stream URL, or webcam index --imgsz Inference
    image size (default: 640) --conf-thres Detection confidence threshold --iou-thres NMS IoU threshold --device CUDA
    device or CPU --project Output folder --name Run name

    ----------------------------------------------------------------------
    ▶ EXAMPLE USAGE
    ----------------------------------------------------------------------

    python det_tracking_count.py --weights yolov5s.pt --source traffic.mp4

    ----------------------------------------------------------------------
    🏁 RESULT
    ----------------------------------------------------------------------
    Instead of simple detections, you get directional vehicle flow:

    Car IN : 12 Car OUT: 9

    Motorcycle IN : 7 Motorcycle OUT: 4

    Perfect for real-time traffic analytics and vehicle monitoring systems.
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
    print(about.__doc__)
    run(**vars(opt))
