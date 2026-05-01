import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


def help():
    print("""
========================================================
YOLOv5 + SORT Object Tracking, Unique Counting & Line Counter
========================================================

This script performs REAL-TIME or OFFLINE VIDEO analysis using YOLOv5
for object detection and SORT for object tracking.

It detects objects, assigns a unique ID to each object, counts unique
objects per class, and counts ENTRY / EXIT events when objects cross
a horizontal line.

--------------------------------------------------------
WORKING FLOW
--------------------------------------------------------

1. Model Loading
   - Loads a YOLOv5 model using the provided weights.
   - Automatically selects CPU or GPU.

2. Video Input
   - Accepts a video file or live camera stream.
   - Reads frames continuously using OpenCV.

3. Object Detection (YOLOv5)
   - Each frame is resized using letterbox.
   - YOLOv5 detects objects and outputs bounding boxes,
     confidence scores, and class IDs.
   - Non-Max Suppression removes duplicate detections.

4. Object Tracking (SORT)
   - Detected bounding boxes are passed to the SORT tracker.
   - SORT assigns a unique TRACK ID to each object.
   - IDs persist across frames for the same object.

5. Unique Object Counting
   - Each TRACK ID is matched with the best YOLO detection
     using IOU overlap.
   - Every object is counted ONLY ONCE per class.
   - Prevents double counting of the same object.

6. Line Crossing Counter
   - A horizontal line is placed at a fixed Y position.
   - The object center is tracked frame-to-frame.
   - When an object crosses the line:
       • Top → Bottom  → ENTRY
       • Bottom → Top  → EXIT
   - Mode can be:
       entry  → count only entries
       exit   → count only exits
       both   → count both

7. Visualization
   - Draws bounding boxes and track IDs.
   - Displays class-wise unique counts.
   - Displays entry and exit counts.
   - Draws the counting line.

8. Output Saving
   - Saves annotated output video.
   - Saves a text file with:
       • Unique object counts per class
       • Total entry count
       • Total exit count

--------------------------------------------------------
USAGE
--------------------------------------------------------

python track_count_line.py 
    --weights <weights_path>
    --source <video_path | camera_id>
    --line_mode <entry | exit | both>
    --project <output_directory>
    --name <experiment_name>
    --conf_thres <confidence_threshold>
    --iou_thres <iou_threshold>
    --view_img

--------------------------------------------------------
ARGUMENT DETAILS
--------------------------------------------------------

--weights      Path to YOLOv5 model weights (e.g. yolov5s.pt)
--source       Video file path or camera ID (0 for webcam)
--line_mode    Counting mode: entry, exit, or both
--project      Directory to save results
--name         Subfolder name inside project directory
--conf_thres   Minimum confidence threshold for detections
--iou_thres    IOU threshold for NMS
--view_img     Display live result window (press 'q' to quit)

--------------------------------------------------------
OUTPUT FILES
--------------------------------------------------------

• Output Video:
  <project>/<name>/output.mp4

• Counts File:
  <project>/<name>/counts.txt

--------------------------------------------------------
EXAMPLE
--------------------------------------------------------

python track_count_line.py --weights yolov5s.pt --source video.mp4 --line_mode both --view_img

========================================================
""")
    sys.exit(0)


def safe_imshow(win_name, frame):
    try:
        cv2.imshow(win_name, frame)
        return True
    except cv2.error:
        return False


# ---------------- YOLOv5 IMPORT ----------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5-master"
sys.path.append(str(ROOT))
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# ---------------- SORT TRACKER ----------------
sys.path.append(str(FILE.parents[0] / "sort"))
from sort import Sort


# ---------------- UNIQUE OBJECT COUNTER ----------------
class UniqueObjectCounter:
    def __init__(self, class_names):
        self.class_names = class_names
        self.counted_ids = defaultdict(set)

    def update(self, tracks, detections):
        for track in tracks:
            tx1, ty1, tx2, ty2, track_id = track
            track_box = [tx1, ty1, tx2, ty2]

            best_iou = 0
            best_class = None
            for det in detections:
                dx1, dy1, dx2, dy2, _conf, cls = det
                iou = self.iou(track_box, [dx1, dy1, dx2, dy2])
                if iou > best_iou:
                    best_iou = iou
                    best_class = int(cls)

            if best_iou > 0.3:
                self.counted_ids[best_class].add(int(track_id))

    def iou(self, A, B):
        xA, yA = max(A[0], B[0]), max(A[1], B[1])
        xB, yB = min(A[2], B[2]), min(A[3], B[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (A[2] - A[0]) * (A[3] - A[1])
        areaB = (B[2] - B[0]) * (B[3] - B[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def get_counts(self):
        result = {}
        for c, ids in self.counted_ids.items():
            if c in self.class_names:
                name = self.class_names[c]
            else:
                print(f"[WARNING] Unknown class ID: {c}")
                name = f"class_{c}"
            result[name] = len(ids)
        return result


# ---------------- LINE COUNTER ----------------
class LineCounter:
    def __init__(self, line_position, mode="both"):
        self.line_y = line_position
        self.mode = mode
        self.entry_count = 0
        self.exit_count = 0
        self.track_memory = {}

    def update(self, tracks):
        for track in tracks:
            _x1, y1, _x2, y2, track_id = map(int, track)
            cy = int((y1 + y2) / 2)

            if track_id not in self.track_memory:
                self.track_memory[track_id] = cy
                continue

            prev_cy = self.track_memory[track_id]

            if prev_cy < self.line_y and cy >= self.line_y:
                if self.mode in ["entry", "both"]:
                    self.entry_count += 1

            elif prev_cy > self.line_y and cy <= self.line_y:
                if self.mode in ["exit", "both"]:
                    self.exit_count += 1

            self.track_memory[track_id] = cy

    def draw(self, frame):
        _h, w, _ = frame.shape
        cv2.line(frame, (0, self.line_y), (w, self.line_y), (255, 0, 0), 2)
        text = f"Entry: {self.entry_count}  Exit: {self.exit_count}"
        cv2.putText(frame, text, (20, self.line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


# ---------------- MAIN PIPELINE ----------------
def run(weights, source, line_mode, project, name, conf_thres, iou_thres, view_img):

    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device("")
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    unique_counter = UniqueObjectCounter(names)
    line_counter = LineCounter(line_position=300, mode=line_mode)

    cap = cv2.VideoCapture(source)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = str(save_dir / "output.mp4")
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = letterbox(frame, 416, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float() / 255.0
        img = img.unsqueeze(0)

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        detections_for_sort = []
        yolo_dets = []

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    print(f"[DEBUG] Detected class: {int(cls)}")
                    x1, y1, x2, y2 = map(float, xyxy)
                    detections_for_sort.append([x1, y1, x2, y2, float(conf)])
                    yolo_dets.append([x1, y1, x2, y2, float(conf), int(cls)])

        tracks = tracker.update(np.array(detections_for_sort))
        unique_counter.update(tracks, yolo_dets)
        line_counter.update(tracks)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        counts = unique_counter.get_counts()
        y_text = 30
        for cls, cnt in counts.items():
            cv2.putText(frame, f"{cls}: {cnt}", (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_text += 25

        line_counter.draw(frame)
        writer.write(frame)
        if view_img:
            shown = safe_imshow("Tracking + Line Counter", frame)
            if shown and cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    txt_path = save_dir / "counts.txt"
    with open(txt_path, "w") as f:
        f.write("Unique Object Counts Per Class\n")
        for cls, cnt in unique_counter.get_counts().items():
            f.write(f"{cls}: {cnt}\n")
        f.write(f"\nEntry Count: {line_counter.entry_count}\n")
        f.write(f"Exit Count: {line_counter.exit_count}\n")

    print(f"\nResults saved to: {save_dir}")


# ---------------- ARGPARSE ----------------
if __name__ == "__main__":
    if "--help-script" in sys.argv:
        help()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5s.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--line_mode", type=str, default="both", choices=["entry", "exit", "both"])
    parser.add_argument("--project", type=str, default="runs/count")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--view-img", action="store_true", help="show results window")

    args = parser.parse_args()

    source = int(args.source) if args.source.isnumeric() else args.source
    run(args.weights, source, args.line_mode, args.project, args.name, args.conf_thres, args.iou_thres, args.view_img)
