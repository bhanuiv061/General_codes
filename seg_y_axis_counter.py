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

# # ================= YOLOv5 SEG =================
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes
# from utils.segment.general import process_mask
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# # ================= CONFIG =================
# LINE_Y_RATIO   = 0.5   # default: line at 50% of frame height
# BUFFER_PX      = 60    # pixels above/below line = buffer zone
# BUFFER_SECONDS = 3
# STOP_REQUESTED = False


# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Exit requested — saving videos safely...")


# signal.signal(signal.SIGINT, request_stop)
# signal.signal(signal.SIGTERM, request_stop)

# # ================= UTILITIES =================
# def get_class_color(cls):
#     np.random.seed(abs(hash(cls)) % (2**32))
#     return tuple(int(c) for c in np.random.randint(40, 255, 3))


# def get_zone(cy, line_y):
#     if cy < line_y - BUFFER_PX:
#         return "top"
#     elif cy > line_y + BUFFER_PX:
#         return "bottom"
#     else:
#         return "buffer"


# def get_next_video_path(save_dir, prefix):
#     save_dir.mkdir(parents=True, exist_ok=True)
#     existing = list(save_dir.glob(f"{prefix}_*.mp4"))
#     if not existing:
#         return save_dir / f"{prefix}_0001.mp4"
#     nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
#     return save_dir / f"{prefix}_{max(nums) + 1:04d}.mp4"


# def draw_count_panel(frame, count_in, count_out, all_classes, h, w):
#     """Semi-transparent count panel on top-right of frame."""
#     panel_x     = w - 330
#     panel_y     = 10
#     row_height  = 36
#     vis_cls     = [c for c in all_classes if count_in[c] > 0 or count_out[c] > 0]
#     n_rows      = max(len(vis_cls), 1)
#     panel_h     = 50 + n_rows * row_height

#     # Semi-transparent background
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (panel_x - 10, panel_y),
#                   (w - 5, panel_y + panel_h), (20, 20, 20), -1)
#     cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

#     # Gold border
#     cv2.rectangle(frame, (panel_x - 10, panel_y),
#                   (w - 5, panel_y + panel_h), (0, 215, 255), 2)

#     # Header row
#     cv2.putText(frame, "CLASS           IN   OUT",
#                 (panel_x, panel_y + 28),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

#     # Divider
#     cv2.line(frame,
#              (panel_x - 10, panel_y + 35),
#              (w - 5,        panel_y + 35),
#              (0, 215, 255), 1)

#     if not vis_cls:
#         cv2.putText(frame, "No detections yet",
#                     (panel_x, panel_y + 35 + row_height),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
#         return

#     for i, cls in enumerate(vis_cls):
#         y_pos = panel_y + 38 + (i + 1) * row_height
#         color = get_class_color(cls)
#         text  = f"{cls:<16}{count_in[cls]:>3}  {count_out[cls]:>3}"
#         cv2.putText(frame, text,
#                     (panel_x, y_pos),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# def unpack_model_output(output):
#     if isinstance(output, (list, tuple)) and len(output) >= 2:
#         return output[0], output[1]
#     raise ValueError(f"Unexpected model output: type={type(output)}, len={len(output)}")


# # ==================================================
# @smart_inference_mode()
# def run(
#     weights,
#     source,
#     imgsz=640,
#     conf_thres=0.25,
#     iou_thres=0.45,
#     device="",
#     project="runs/seg-count",
#     name="exp",
#     line_y=None
# ):
#     raw_writer  = None
#     ann_writer  = None
#     frame_idx   = 0
#     raw_video   = None
#     ann_video   = None
#     names       = {}
#     count_in    = {}
#     count_out   = {}
#     all_classes = []
#     LINE_Y      = None   # resolved on first frame from actual video dimensions

#     try:
#         if not os.path.exists(weights):
#             raise FileNotFoundError(f"Weights not found: {weights}")

#         is_webcam = source.isnumeric()
#         save_dir  = Path(project) / name
#         raw_video = get_next_video_path(save_dir, "raw")
#         ann_video = get_next_video_path(save_dir, "annotated")

#         device = select_device(device)
#         model  = DetectMultiBackend(weights, device=device)
#         stride, names = model.stride, model.names
#         imgsz  = check_img_size(imgsz, s=stride)
#         model.warmup(imgsz=(1, 3, imgsz, imgsz))

#         # ── Print class names from .pt ──
#         print(f"\n📦 Classes in model [{len(names)} total]:")
#         for idx, cls_name in names.items():
#             print(f"   [{idx}] {cls_name}")
#         print()

#         all_classes = list(names.values())
#         count_in    = {cls: 0 for cls in all_classes}
#         count_out   = {cls: 0 for cls in all_classes}

#         last_side   = {}
#         last_time   = {}
#         track_class = {}
#         counted_in  = set()
#         counted_out = set()

#         dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#             if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

#         tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#         for data in dataset:
#             if STOP_REQUESTED:
#                 break

#             path, im, im0s, vid_cap, _ = data
#             frame_idx += 1

#             if frame_idx % 100 == 0:
#                 print(f"[INFO] Frames processed: {frame_idx}")

#             raw              = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#             frame            = raw.copy()
#             h_frame, w_frame = frame.shape[:2]

#             # ── Resolve LINE_Y from actual video height (done once) ──
#             if LINE_Y is None:
#                 if line_y is None:
#                     LINE_Y = int(h_frame * LINE_Y_RATIO)
#                 else:
#                     val = float(line_y)
#                     LINE_Y = int(h_frame * val) if 0.0 < val <= 1.0 else int(val)
#                 # Clamp so line + buffer never goes outside frame
#                 LINE_Y = max(BUFFER_PX + 5, min(LINE_Y, h_frame - BUFFER_PX - 5))
#                 print(f"[INFO] Frame size : {w_frame} x {h_frame}")
#                 print(f"[INFO] LINE_Y     : {LINE_Y} px  (buffer ±{BUFFER_PX} px)\n")

#             # ── Preprocess ──
#             im = torch.from_numpy(im).to(device).float() / 255.0
#             if im.ndim == 3:
#                 im = im[None]

#             # ── Inference ──
#             output      = model(im, augment=False, visualize=False)
#             pred, proto = unpack_model_output(output)
#             pred        = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

#             detections = []

#             if pred[0] is not None and len(pred[0]):
#                 pred[0][:, :4] = scale_boxes(
#                     im.shape[2:], pred[0][:, :4], frame.shape
#                 ).round()

#                 masks_tensor = process_mask(
#                     proto[0],
#                     pred[0][:, 6:],
#                     pred[0][:, :4],
#                     (h_frame, w_frame),   # (h, w) — NOT frame.shape
#                     upsample=True
#                 )

#                 for i, (*xyxy, conf, cls) in enumerate(pred[0][:, :6]):
#                     x1, y1, x2, y2 = map(int, xyxy)
#                     detections.append([
#                         x1, y1, x2, y2,
#                         conf.item(),
#                         int(cls),
#                         masks_tensor[i].cpu().numpy().astype(bool)
#                     ])

#             # ── SORT tracking ──
#             tracks = tracker.update(
#                 np.array([d[:5] for d in detections]) if detections
#                 else np.empty((0, 5))
#             )

#             now = time.time()

#             for trk in tracks.astype(int):
#                 x1, y1, x2, y2, tid = trk

#                 best_iou, det = 0, None
#                 for d in detections:
#                     xx1 = max(x1, d[0]); yy1 = max(y1, d[1])
#                     xx2 = min(x2, d[2]); yy2 = min(y2, d[3])
#                     inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
#                     area1 = (x2 - x1) * (y2 - y1)
#                     area2 = (d[2] - d[0]) * (d[3] - d[1])
#                     iou   = inter / (area1 + area2 - inter + 1e-6)
#                     if iou > best_iou:
#                         best_iou, det = iou, d

#                 if det is None:
#                     continue

#                 if tid not in track_class:
#                     track_class[tid] = names[det[5]]
#                 cls_name = track_class[tid]

#                 mask   = det[6]
#                 ys, xs = np.where(mask)
#                 if len(ys) == 0:
#                     continue

#                 # Use Y centroid for horizontal line crossing
#                 cy = int(ys.mean())

#                 zone   = get_zone(cy, LINE_Y)
#                 prev   = last_side.get(tid)
#                 last_t = last_time.get(tid, 0)

#                 # top -> bottom = IN  |  bottom -> top = OUT
#                 if prev and zone != prev and (now - last_t) > BUFFER_SECONDS:
#                     if prev == "top" and zone in ["buffer", "bottom"]:
#                         if tid not in counted_in:
#                             count_in[cls_name] += 1
#                             counted_in.add(tid)
#                         last_time[tid] = now
#                         last_side[tid] = "bottom"

#                     elif prev == "bottom" and zone in ["buffer", "top"]:
#                         if tid not in counted_out:
#                             count_out[cls_name] += 1
#                             counted_out.add(tid)
#                         last_time[tid] = now
#                         last_side[tid] = "top"

#                 if zone in ["top", "bottom"]:
#                     last_side[tid] = zone

#                 # Draw mask overlay
#                 color = get_class_color(cls_name)
#                 frame[mask] = (frame[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

#                 # Label above bounding box
#                 cv2.putText(frame, f"{cls_name} #{tid}",
#                             (x1, max(y1 - 6, 14)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#                 # Centroid dot
#                 cv2.circle(frame, (int(xs.mean()), cy), 5, color, -1)

#             # ══ Draw horizontal counting line (spans full video width) ══
#             # Buffer lines (gold dashed visual)
#             cv2.line(frame,
#                      (0, LINE_Y - BUFFER_PX), (w_frame, LINE_Y - BUFFER_PX),
#                      (255, 215, 0), 1)
#             cv2.line(frame,
#                      (0, LINE_Y + BUFFER_PX), (w_frame, LINE_Y + BUFFER_PX),
#                      (255, 215, 0), 1)
#             # Main counting line (cyan)
#             cv2.line(frame,
#                      (0, LINE_Y), (w_frame, LINE_Y),
#                      (0, 255, 255), 2)

#             # Direction arrows / labels
#             cv2.putText(frame, "▼ IN",
#                         (w_frame // 2 - 30, LINE_Y + BUFFER_PX + 22),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
#             cv2.putText(frame, "▲ OUT",
#                         (w_frame // 2 - 36, LINE_Y - BUFFER_PX - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

#             # ── Count panel (top-right) ──
#             draw_count_panel(frame, count_in, count_out, all_classes, h_frame, w_frame)

#             # ── Video writers ──
#             if raw_writer is None:
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#                 fps = fps if fps > 0 else 25
#                 raw_writer = cv2.VideoWriter(
#                     str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"),
#                     fps, (w_frame, h_frame))
#                 ann_writer = cv2.VideoWriter(
#                     str(ann_video), cv2.VideoWriter_fourcc(*"mp4v"),
#                     fps, (w_frame, h_frame))

#             raw_writer.write(raw)
#             ann_writer.write(frame)

#             cv2.imshow("YOLOv5 Seg Counting", frame)
#             if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
#                 request_stop()

#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         traceback.print_exc()

#     finally:
#         if raw_writer:
#             raw_writer.release()
#         if ann_writer:
#             ann_writer.release()
#         cv2.destroyAllWindows()

#         print(f"\n✅ Raw video        : {raw_video}")
#         print(f"✅ Annotated video   : {ann_video}")
#         print(f"📊 Frames processed  : {frame_idx}")

#         if count_in:
#             print("\n📊 Final Count Summary:")
#             print(f"{'Class':<25} {'IN':>6} {'OUT':>6}")
#             print("-" * 40)
#             for cls in all_classes:
#                 print(f"{cls:<25} {count_in.get(cls, 0):>6} {count_out.get(cls, 0):>6}")


# # ================= CLI =================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights",    required=True)
#     parser.add_argument("--source",     required=True)
#     parser.add_argument("--imgsz",      type=int,   default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres",  type=float, default=0.45)
#     parser.add_argument("--device",     default="")
#     parser.add_argument("--project",    default="runs/seg-count")
#     parser.add_argument("--name",       default="exp")
#     parser.add_argument("--line-y",     default=None,
#                         help="Horizontal line Y position. "
#                              "Float 0.0-1.0 = fraction of height (e.g. 0.5 = centre). "
#                              "Integer > 1 = absolute pixel row (e.g. 360). "
#                              "Default = 0.5 (centre).")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))


































































































import argparse
import sys
import time
import traceback
from pathlib import Path
import cv2
import torch
import numpy as np
import os
import pathlib
import signal

# ================= WINDOWS PATH FIX =================
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ================= ROOT =================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ================= YOLOv5 SEG =================
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode
from sort.sort import Sort

# ================= CONFIG =================
LINE_Y_RATIO   = 0.5   # default: line at 50% of frame height
BUFFER_PX      = 60    # pixels above/below line = buffer zone
STOP_REQUESTED = False


def request_stop(sig=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠ Exit requested — saving videos safely...")


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

# ================= UTILITIES =================
def get_class_color(cls):
    np.random.seed(abs(hash(cls)) % (2**32))
    return tuple(int(c) for c in np.random.randint(40, 255, 3))


def get_next_video_path(save_dir, prefix):
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = list(save_dir.glob(f"{prefix}_*.mp4"))
    if not existing:
        return save_dir / f"{prefix}_0001.mp4"
    nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
    return save_dir / f"{prefix}_{max(nums) + 1:04d}.mp4"


def draw_count_panel(frame, count_in, all_classes, h, w):
    """
    Semi-transparent panel on top-right showing per-class IN count only.
    Only shows classes that have been detected at least once.
    """
    panel_x    = w - 280
    panel_y    = 10
    row_height = 36
    vis_cls    = [c for c in all_classes if count_in[c] > 0]
    n_rows     = max(len(vis_cls), 1)
    panel_h    = 50 + n_rows * row_height

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x - 10, panel_y),
                  (w - 5, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Gold border
    cv2.rectangle(frame,
                  (panel_x - 10, panel_y),
                  (w - 5, panel_y + panel_h),
                  (0, 215, 255), 2)

    # Header
    cv2.putText(frame, "CLASS            COUNT",
                (panel_x, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

    # Divider
    cv2.line(frame,
             (panel_x - 10, panel_y + 35),
             (w - 5,        panel_y + 35),
             (0, 215, 255), 1)

    if not vis_cls:
        cv2.putText(frame, "No crossings yet",
                    (panel_x, panel_y + 35 + row_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return

    for i, cls in enumerate(vis_cls):
        y_pos = panel_y + 38 + (i + 1) * row_height
        color = get_class_color(cls)
        text  = f"{cls:<16}  {count_in[cls]:>4}"
        cv2.putText(frame, text,
                    (panel_x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def unpack_model_output(output):
    if isinstance(output, (list, tuple)) and len(output) >= 2:
        return output[0], output[1]
    raise ValueError(f"Unexpected model output: type={type(output)}, len={len(output)}")


# ==================================================
@smart_inference_mode()
def run(
    weights,
    source,
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    device="",
    project="runs/seg-count",
    name="exp",
    line_y=None
):
    raw_writer  = None
    ann_writer  = None
    frame_idx   = 0
    raw_video   = None
    ann_video   = None
    names       = {}
    count_in    = {}
    all_classes = []
    LINE_Y      = None

    try:
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")

        is_webcam = source.isnumeric()
        save_dir  = Path(project) / name
        raw_video = get_next_video_path(save_dir, "raw")
        ann_video = get_next_video_path(save_dir, "annotated")

        device = select_device(device)
        model  = DetectMultiBackend(weights, device=device)
        stride, names = model.stride, model.names
        imgsz  = check_img_size(imgsz, s=stride)
        model.warmup(imgsz=(1, 3, imgsz, imgsz))

        # ── Print class names from .pt ──
        print(f"\n📦 Classes in model [{len(names)} total]:")
        for idx, cls_name in names.items():
            print(f"   [{idx}] {cls_name}")
        print()

        all_classes = list(names.values())
        count_in    = {cls: 0 for cls in all_classes}

        # ── Per-track state ──
        track_class    = {}   # tid -> cls_name (locked on first detection)
        seen_above     = set()  # tids that have been observed ABOVE the line
        already_counted = set() # tids that have already been counted (no repeat ever)

        dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
            if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

        tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

        for data in dataset:
            if STOP_REQUESTED:
                break

            path, im, im0s, vid_cap, _ = data
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"[INFO] Frames processed: {frame_idx}")

            raw              = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
            frame            = raw.copy()
            h_frame, w_frame = frame.shape[:2]

            # ── Resolve LINE_Y from actual video height (once) ──
            if LINE_Y is None:
                if line_y is None:
                    LINE_Y = int(h_frame * LINE_Y_RATIO)
                else:
                    val    = float(line_y)
                    LINE_Y = int(h_frame * val) if 0.0 < val <= 1.0 else int(val)
                LINE_Y = max(BUFFER_PX + 5, min(LINE_Y, h_frame - BUFFER_PX - 5))
                print(f"[INFO] Frame size : {w_frame} x {h_frame}")
                print(f"[INFO] LINE_Y     : {LINE_Y} px  (buffer ±{BUFFER_PX} px)\n")

            # ── Preprocess ──
            im = torch.from_numpy(im).to(device).float() / 255.0
            if im.ndim == 3:
                im = im[None]

            # ── Inference ──
            output      = model(im, augment=False, visualize=False)
            pred, proto = unpack_model_output(output)
            pred        = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

            detections = []

            if pred[0] is not None and len(pred[0]):
                pred[0][:, :4] = scale_boxes(
                    im.shape[2:], pred[0][:, :4], frame.shape
                ).round()

                masks_tensor = process_mask(
                    proto[0],
                    pred[0][:, 6:],
                    pred[0][:, :4],
                    (h_frame, w_frame),
                    upsample=True
                )

                for i, (*xyxy, conf, cls) in enumerate(pred[0][:, :6]):
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append([
                        x1, y1, x2, y2,
                        conf.item(),
                        int(cls),
                        masks_tensor[i].cpu().numpy().astype(bool)
                    ])

            # ── SORT tracking ──
            tracks = tracker.update(
                np.array([d[:5] for d in detections]) if detections
                else np.empty((0, 5))
            )

            for trk in tracks.astype(int):
                x1, y1, x2, y2, tid = trk

                # Match track to best-IoU detection
                best_iou, det = 0, None
                for d in detections:
                    xx1 = max(x1, d[0]); yy1 = max(y1, d[1])
                    xx2 = min(x2, d[2]); yy2 = min(y2, d[3])
                    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (d[2] - d[0]) * (d[3] - d[1])
                    iou   = inter / (area1 + area2 - inter + 1e-6)
                    if iou > best_iou:
                        best_iou, det = iou, d

                if det is None:
                    continue

                # Lock class to tid on first appearance
                if tid not in track_class:
                    track_class[tid] = names[det[5]]
                cls_name = track_class[tid]

                mask   = det[6]
                ys, xs = np.where(mask)
                if len(ys) == 0:
                    continue

                cy = int(ys.mean())  # Y centroid of mask

                # ── ONE DIRECTION ONLY: top → bottom ──
                # Step 1: mark tid as "seen above line" when it's above the line
                if cy < LINE_Y - BUFFER_PX:
                    if tid not in already_counted:
                        seen_above.add(tid)

                # Step 2: count when it crosses below the line
                #         only if it was seen above AND not yet counted
                elif cy > LINE_Y + BUFFER_PX:
                    if tid in seen_above and tid not in already_counted:
                        count_in[cls_name] += 1
                        already_counted.add(tid)   # permanently block re-count
                        seen_above.discard(tid)
                        print(f"  ✅ Counted: {cls_name} ID:{tid}  |  Total {cls_name}: {count_in[cls_name]}")

                # ── Draw mask overlay ──
                color = get_class_color(cls_name)
                frame[mask] = (frame[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

                # Highlight counted IDs differently
                label_color = (0, 255, 0) if tid in already_counted else color
                cv2.putText(frame,
                            f"{cls_name} #{tid}" + (" ✓" if tid in already_counted else ""),
                            (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

                # Centroid dot
                cv2.circle(frame, (int(xs.mean()), cy), 5, label_color, -1)

            # ══ Draw horizontal counting line (full video width) ══
            # Buffer lines (gold)
            cv2.line(frame,
                     (0, LINE_Y - BUFFER_PX), (w_frame, LINE_Y - BUFFER_PX),
                     (255, 215, 0), 1)
            cv2.line(frame,
                     (0, LINE_Y + BUFFER_PX), (w_frame, LINE_Y + BUFFER_PX),
                     (255, 215, 0), 1)
            # Main line (cyan)
            cv2.line(frame,
                     (0, LINE_Y), (w_frame, LINE_Y),
                     (0, 255, 255), 2)

            # Direction label — centre of line
            mid_x = w_frame // 2
            cv2.putText(frame, "▼  COUNTING LINE  (top → bottom)",
                        (mid_x - 200, LINE_Y - BUFFER_PX - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ── Count panel top-right ──
            draw_count_panel(frame, count_in, all_classes, h_frame, w_frame)

            # ── Video writers ──
            if raw_writer is None:
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
                fps = fps if fps > 0 else 25
                raw_writer = cv2.VideoWriter(
                    str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (w_frame, h_frame))
                ann_writer = cv2.VideoWriter(
                    str(ann_video), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (w_frame, h_frame))

            raw_writer.write(raw)
            ann_writer.write(frame)

            cv2.imshow("YOLOv5 Seg Counting", frame)
            if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                request_stop()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

    finally:
        if raw_writer:
            raw_writer.release()
        if ann_writer:
            ann_writer.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Raw video        : {raw_video}")
        print(f"✅ Annotated video   : {ann_video}")
        print(f"📊 Frames processed  : {frame_idx}")

        if count_in:
            print("\n📊 Final Count Summary (top → bottom only):")
            print(f"{'Class':<25} {'COUNT':>7}")
            print("-" * 35)
            for cls in all_classes:
                print(f"{cls:<25} {count_in.get(cls, 0):>7}")
            print(f"\n{'TOTAL':<25} {sum(count_in.values()):>7}")


# ================= CLI =================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--source",     required=True)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres",  type=float, default=0.45)
    parser.add_argument("--device",     default="")
    parser.add_argument("--project",    default="runs/seg-count")
    parser.add_argument("--name",       default="exp")
    parser.add_argument("--line-y",     default=None,
                        help="Horizontal line Y position. "
                             "Float 0.0-1.0 = fraction of height (e.g. 0.5 = centre). "
                             "Integer > 1 = absolute pixel row (e.g. 360). "
                             "Default = 0.5 (centre).")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))