import argparse
import csv
import os
import pathlib
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from sort.sort import Sort

start_time = time.time()

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
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    check_img_size,
    check_requirements,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    nosave=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    half=False,
    dnn=False,
    vid_stride=1,
):

    # ---------------- WATERMARK ----------------
    def add_diagonal_watermark(frame, text="DEMO WATERMARK", opacity=0.18):
        overlay = frame.copy()
        h, w = frame.shape[:2]

        font_scale = min(w, h) / 900
        thickness = int(font_scale * 2)
        color = (255, 255, 255)

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x = (w - text_size[0]) // 2
        y = (h + text_size[1]) // 2

        watermark_layer = np.zeros_like(frame, dtype=np.uint8)
        cv2.putText(watermark_layer, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
        rotated = cv2.warpAffine(watermark_layer, matrix, (w, h))

        cv2.addWeighted(rotated, opacity, overlay, 1 - opacity, 0, overlay)
        return overlay

    def print_progress(frame_idx, counts):
        elapsed = time.time() - start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        total = sum(counts.values())

        count_str = " | ".join([f"{k}:{v}" for k, v in counts.items()])
        msg = f"\r🎯 Frame: {frame_idx} | FPS: {fps:.2f} | Total: {total} | {count_str}"
        print(msg, end="", flush=True)

    # ---------------- LOGO ----------------
    def add_logo_top_left(frame, logo_path="logo_white 1.png", width=120):
        if not os.path.exists(logo_path):
            return frame

        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            return frame

        h_logo, w_logo = logo.shape[:2]
        aspect = h_logo / w_logo
        new_h = int(width * aspect)
        logo = cv2.resize(logo, (width, new_h))

        x_offset, y_offset = 10, 60

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

    # ---------------- CLASS SETUP ----------------
    def init_class_counter(model_names):
        return {name: 0 for name in model_names.values()}

    def get_class_color(class_id):
        np.random.seed(int(class_id) + 42)
        return tuple(int(x) for x in np.random.randint(0, 255, size=3))

    def log_counts(frame_num, counts, csv_file):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, frame_num, *list(counts.values())])

    def is_crossing_line(prev_y, curr_y):
        return prev_y < line_y <= curr_y or prev_y > line_y >= curr_y

    # ---------------- MODEL ----------------
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    counts = init_class_counter(names)

    source = str(source)
    not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    webcam = source.isnumeric() and not is_file

    dataset = (
        LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        if webcam
        else LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    )

    tracker = Sort()
    prev_centroids = {}
    line_y = 400

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(save_dir / "output.mp4")
    video_writer = None

    csv_file = os.path.join(str(save_dir), f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame", *list(counts.keys())])
    # ---------------- LOOP ----------------
    import time

    start_time = time.time()

    try:
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            im = torch.from_numpy(im).to(device).float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            pred = model(im)
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            im0 = im0s[0].copy() if webcam else im0s.copy()
            annotator = Annotator(im0, line_width=2)

            # 🎥 Initialize video writer on first frame
            if video_writer is None:
                h, w = im0.shape[:2]
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            detections = []
            if len(pred[0]):
                pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()
                for *xyxy, conf, cls in pred[0]:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append([x1, y1, x2, y2, conf.item()])

            tracks = tracker.update(np.array(detections)) if detections else np.empty((0, 5))
            current_centroids = {}

            for x1, y1, x2, y2, track_id in tracks.astype(int):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detected_class = "unknown"
                detected_cls_id = -1

                for *xyxy, conf, cls in pred[0]:
                    px1, py1, _px2, _py2 = map(int, xyxy)
                    if abs(x1 - px1) < 10 and abs(y1 - py1) < 10:
                        detected_class = names[int(cls)] if int(cls) in names else "unknown"
                        detected_cls_id = int(cls)
                        break

                current_centroids[track_id] = (cx, cy, detected_class)
                color = get_class_color(detected_cls_id)

                annotator.box_label([x1, y1, x2, y2], detected_class, color=color)
                cv2.circle(im0, (cx, cy), 4, color, -1)

            # -------- COUNTING --------
            for obj_id, (cx, cy, cls) in current_centroids.items():
                if obj_id in prev_centroids:
                    if is_crossing_line(prev_centroids[obj_id][1], cy):
                        if cls in counts:  # prevent crash
                            counts[cls] += 1

            prev_centroids = current_centroids.copy()

            cv2.line(im0, (0, line_y), (im0.shape[1], line_y), (0, 255, 255), 2)

            # -------- DISPLAY COUNTS --------
            y_offset = 220
            for cls_name, count in counts.items():
                cv2.putText(
                    im0, f"{cls_name}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
                )
                y_offset += 35

            im0 = add_logo_top_left(im0)
            im0 = add_diagonal_watermark(im0)

            log_counts(frame_idx, counts, csv_file)

            # 🎥 Always write frame
            if video_writer is not None:
                video_writer.write(im0)

            # -------- LIVE TERMINAL PROGRESS --------
            elapsed = time.time() - start_time
            fps_live = frame_idx / elapsed if elapsed > 0 else 0
            total = sum(counts.values())
            count_str = " | ".join([f"{k}:{v}" for k, v in counts.items()])
            print(f"\r🎯 Frame: {frame_idx} | FPS: {fps_live:.2f} | Total: {total} | {count_str}", end="", flush=True)

            cv2.imshow("Counting", im0)
            if cv2.waitKey(1) == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving video safely...")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"\n🎥 Video saved at: {video_path}")

        cv2.destroyAllWindows()
        print("✅ Program finished safely.")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str, default=ROOT / "data/images")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--project", default=ROOT / "runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
