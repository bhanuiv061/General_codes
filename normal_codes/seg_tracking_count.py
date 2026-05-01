import argparse
import csv
import os
import pathlib
import sys
import time
from datetime import datetime
from pathlib import Path

start_time = time.time()
import cv2
import numpy as np
import torch
from sort.sort import Sort

# Fix Windows paths
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
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights="yolov5s-seg.pt",
    source="0",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    device="",
    project="runs/seg_count",
    name="exp",
    **kwargs,
):

    # ------------------ HELPERS ------------------

    def bbox_iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def get_class_color(class_id):
        np.random.seed(class_id + 42)
        return tuple(int(x) for x in np.random.randint(0, 255, 3))

    def add_diagonal_watermark(frame, text="DEMO WATERMARK", opacity=0.18):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        font_scale = min(w, h) / 900
        thickness = int(font_scale * 2)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x, y = (w - text_size[0]) // 2, (h + text_size[1]) // 2
        layer = np.zeros_like(frame, dtype=np.uint8)
        cv2.putText(layer, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1)
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

    def is_crossing(prev_y, curr_y):
        return prev_y < line_y <= curr_y or prev_y > line_y >= curr_y

    # ------------------ INIT ------------------

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    counts = {name: 0 for name in names.values()}
    tracker = Sort()
    prev_centroids = {}
    line_y = 400

    dataset = (
        LoadStreams(source, img_size=imgsz, stride=stride)
        if source.isnumeric()
        else LoadImages(source, img_size=imgsz, stride=stride)
    )

    save_dir = increment_path(Path(project) / name)
    save_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    video_path = str(save_dir / "output.mp4")

    csv_file = save_dir / "counts.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "frame", *list(counts.keys())])

    def print_progress(frame_idx, counts):
        elapsed = time.time() - start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        total = sum(counts.values())

        count_str = " | ".join([f"{k}:{v}" for k, v in counts.items()])
        msg = f"\r🎯 Frame: {frame_idx} | FPS: {fps:.2f} | Total: {total} | {count_str}"
        print(msg, end="", flush=True)

    def get_class_name(cls_id):
        return names[cls_id] if cls_id in names else "unknown"

    # ------------------ LOOP ------------------

    try:
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            im = torch.from_numpy(im).to(device).float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            pred, proto = model(im)[:2]
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            im0 = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()

            if video_writer is None:
                h, w = im0.shape[:2]
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            detections = []
            det_classes = []

            if len(pred[0]):
                pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()
                has_masks = proto is not None and pred[0].shape[1] > 6
                if has_masks:
                    masks = process_mask(proto[0], pred[0][:, 6:], pred[0][:, :4], im.shape[2:], upsample=True)

                for i, (*xyxy, conf, cls) in enumerate(pred[0]):
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append([x1, y1, x2, y2, conf.item()])
                    det_classes.append(int(cls))
                    color = get_class_color(int(cls))
                    label = get_class_name(int(cls))

                    if has_masks:
                        mask = masks[i].cpu().numpy() > 0.5
                        im0[mask] = (im0[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

                    cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(im0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            tracks = tracker.update(np.array(detections)) if detections else np.empty((0, 5))
            current_centroids = {}

            for x1, y1, x2, y2, track_id in tracks.astype(int):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                best_iou, detected_class = 0, "unknown"

                for i, (px1, py1, px2, py2, _) in enumerate(detections):
                    iou = bbox_iou([x1, y1, x2, y2], [px1, py1, px2, py2])
                    if iou > best_iou:
                        best_iou = iou
                        detected_class = get_class_name(det_classes[i])

                if best_iou < 0.3:
                    detected_class = "unknown"

                current_centroids[track_id] = (cx, cy, detected_class)

            # ---- COUNTING (correct position) ----
            for obj_id, (cx, cy, cls) in current_centroids.items():
                if obj_id in prev_centroids and is_crossing(prev_centroids[obj_id][1], cy):
                    if cls in counts:
                        counts[cls] += 1

            prev_centroids = current_centroids

            cv2.line(im0, (0, line_y), (im0.shape[1], line_y), (0, 255, 255), 2)

            y = 220
            for cls, val in counts.items():
                cv2.putText(im0, f"{cls}: {val}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                y += 35

            im0 = add_logo_top_left(im0)
            im0 = add_diagonal_watermark(im0)

            video_writer.write(im0)

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%H:%M:%S"), frame_idx, *list(counts.values())])

            print_progress(frame_idx, counts)
            cv2.imshow("Seg Counting", im0)
            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("✅ Video saved:", video_path)


def parse_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-seg.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/predict-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--retina-masks", action="store_true", help="whether to plot masks in native resolution")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking for requirements before launching."""
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
