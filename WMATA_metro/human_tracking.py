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
            cv2.rectangle(display, roi[0], (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            roi.append((x, y))
            drawing = False
            display = frame.copy()
            cv2.rectangle(display, roi[0], roi[1], (0, 255, 0), 2)

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
            cv2.rectangle(im0, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        if mode == "line":
            cv2.line(im0, (0, line_y), (im0.shape[1], line_y), (0, 255, 255), 2)

        cv2.putText(im0, f"COUNT: {counts['person']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        writer.write(im0)

    writer.release()
    print("✅ Saved:", out_path)


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
