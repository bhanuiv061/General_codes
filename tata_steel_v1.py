import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import cv2

from sort.sort import Sort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes, increment_path
from utils.torch_utils import select_device, smart_inference_mode

tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)

counted_ids = set()
class_counts = {}
track_class_map = {}
track_last_x = {}

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

@smart_inference_mode()
def run(weights="yolov5s-seg.pt", source="0", imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, device="", project="runs/output", name="exp"):
    global class_counts

    save_dir = increment_path(Path(project) / name, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    if isinstance(names, dict):
        class_list = [names[k] for k in sorted(names.keys())]
    else:
        class_list = names

    print("Classes in model:")
    for i, n in enumerate(class_list):
        print(f"{i}: {n}")

    webcam = source.isnumeric()
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)

    model.warmup(imgsz=(1, 3, *imgsz))

    vid_writer = None

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.float() / 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            if webcam:
                im0 = im0s[i].copy()
            else:
                im0 = im0s.copy()

            h, w = im0.shape[:2]
            line_x = w // 2

            detections = []
            det_info = []

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det[:, :6]:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append([x1, y1, x2, y2, float(conf)])
                    det_info.append((x1, y1, x2, y2, int(cls)))

            detections = np.array(detections) if len(detections) else np.empty((0, 5))

            tracks = tracker.update(detections)

            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)

                best_iou = 0
                best_cls = None

                for (dx1, dy1, dx2, dy2, cls_id) in det_info:
                    iou = compute_iou([x1, y1, x2, y2], [dx1, dy1, dx2, dy2])
                    if iou > best_iou:
                        best_iou = iou
                        best_cls = cls_id

                if best_cls is not None:
                    track_class_map[track_id] = best_cls

                cls_id = track_class_map.get(track_id, None)
                if cls_id is None:
                    continue

                if isinstance(names, dict):
                    cls_name = names.get(cls_id, "unknown")
                else:
                    cls_name = names[cls_id] if cls_id < len(names) else "unknown"

                cx = int((x1 + x2) / 2)
                prev_x = track_last_x.get(track_id, cx)
                track_last_x[track_id] = cx

                if prev_x < line_x and cx >= line_x and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    if cls_name not in class_counts:
                        class_counts[cls_name] = 0
                    class_counts[cls_name] += 1

                label = f"{cls_name} ID {track_id}"

                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(im0, (cx, int((y1 + y2) / 2)), 4, (0, 255, 0), -1)

            cv2.line(im0, (line_x, 0), (line_x, h), (255, 0, 0), 2)

            y_offset = 40
            for cls_name in sorted(class_counts.keys()):
                count = class_counts[cls_name]
                text = f"{cls_name}: {count}"
                cv2.putText(im0, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_offset += 30

            if vid_writer is None:
                save_path = str(save_dir / Path(path).name)
                fps = 30 if vid_cap is None else vid_cap.get(cv2.CAP_PROP_FPS)
                w_out = im0.shape[1]
                h_out = im0.shape[0]
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_out, h_out))

            vid_writer.write(im0)

            cv2.imshow("result", im0)
            if cv2.waitKey(1) == 27:
                return

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5s-seg.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", type=str, default="runs/output")
    parser.add_argument("--name", type=str, default="exp")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)