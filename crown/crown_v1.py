import argparse
import os
import pathlib
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
from tqdm import tqdm

# ================= WINDOWS PATH FIX =================
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ================= ROOT =================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # crown/ folder
YOLO_ROOT = FILE.parents[1]  # yolov5-master/ folder

for p in [str(ROOT), str(YOLO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

ROOT = Path(os.path.relpath(YOLO_ROOT, Path.cwd()))

# ================= YOLOv5 SEG =================
from sort.sort import Sort

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode

# ================= CONFIG =================
STOP_REQUESTED = False

# ================= CLASS SETUP =================
OCR_CLASSES = {"label"}

CLASS_COLORS = {
    "label": (0, 255, 0),
    "caution": (0, 165, 255),
    "crown": (128, 0, 128),
    "up": (255, 255, 0),
    "box": (200, 200, 200),
    "overlap": (0, 0, 255),
}


def get_class_color(cls_name):
    if cls_name in CLASS_COLORS:
        return CLASS_COLORS[cls_name]
    np.random.seed(abs(hash(cls_name)) % (2**32))
    return tuple(int(c) for c in np.random.randint(40, 255, 3))


# ================= OCR CONFIG =================
EXPECTED_LABEL_TEXTS = {
    "UN3481": "UN3481 - Lithium Battery (With Equipment)",
    "UN3480": "UN3480 - Lithium Battery (Standalone)",
    "UN1066": "UN1066 - Nitrogen Compressed",
    "LITHIUMIONBATTERIES": "LITHIUM ION BATTERIES",
    "LITHIUMIONBATTERIESFORBIDDENFORTRANSPORTABOARDPASSENGERAIRCRAFT": "LITHIUM ION BATTERIES",
    "NONSPILLABLEBATTERY": "NONSPILLABLE BATTERY",
    "FRAGILE": "FRAGILE",
    "NITROGENCOMPRESSED": "NITROGEN COMPRESSED",
    "NONFLAMMABLEGAS": "NON-FLAMMABLE GAS",
    "DOTSP10898": "DOT-SP 10898",
    "NITROGENHYDRAULICACCUMULATORS": "NITROGEN HYDRAULIC ACCUMULATORS",
    "FACILITIESMAINTENANCEUSE": "FACILITIES MAINTENANCE USE",
}

UN_DESCRIPTIONS = {
    "UN3481": "Lithium Battery - Packed With Equipment",
    "UN3480": "Lithium Battery - Standalone",
    "UN1066": "Nitrogen, Compressed",
}

ocr_reader = easyocr.Reader(["en"], gpu=True)


# ================= SIGNAL =================
def request_stop(sig=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠ Exit requested — finalizing safely...")


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


# ================= UTILITIES =================
def normalize_text(text):
    return text.upper().replace(" ", "").replace("-", "").replace("_", "")


def draw_text_with_border(
    img,
    text,
    pos,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.52,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    border_color=(0, 215, 255),
    thickness=1,
    padding=2,
    border_thickness=1,
):
    x, y = pos
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h += baseline
    cv2.rectangle(img, (x - padding, y - h - padding), (x + w + padding, y + padding), bg_color, -1)
    cv2.rectangle(img, (x - padding, y - h - padding), (x + w + padding, y + padding), border_color, border_thickness)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)


def draw_mask_overlay(frame, mask_bin, color, alpha=0.35):
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[mask_bin > 0] = color
    cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0, frame)
    contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 1)


def ocr_label(crop):
    """Try OCR at 4 rotations. Only called for 'label' class."""
    for angle in [0, 90, 180, 270]:
        img = crop.copy()
        if angle == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        results = ocr_reader.readtext(gray, detail=1, paragraph=False)
        texts = [t for _, t, c in results if c > 0.3]

        if texts:
            full = " ".join(texts)
            norm = normalize_text(full)
            un_match = re.search(r"UN\d{4}", norm)
            un_code = un_match.group(0) if un_match else None
            un_desc = UN_DESCRIPTIONS.get(un_code, "") if un_code else ""

            matched_key = None
            for key in EXPECTED_LABEL_TEXTS:
                if key in norm:
                    matched_key = key
                    break

            if un_code or matched_key:
                return full, norm, un_code, un_desc, matched_key

    return None, None, None, None, None


# ==================================================
@smart_inference_mode()
def run(weights, source, imgsz=640, conf_thres=0.25, iou_thres=0.45, device="", project="runs/ocr", name="exp"):
    # ── label OCR counts ──
    track_ocr_text = {}
    ocr_text_counts = {}

    # ── per-class object counts (once per unique track_id) ──
    counted_track_ids = {}
    class_counts = {}

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    # Normalize names keys to int in case YOLOv5 returns string keys e.g. {'0': 'label'}
    names = {int(k): v for k, v in names.items()}
    imgsz = check_img_size(imgsz, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))
    print(f"Model classes: {names}")

    dataset = (
        LoadStreams(source, img_size=imgsz, stride=stride)
        if source.isnumeric()
        else LoadImages(source, img_size=imgsz, stride=stride)
    )

    # ── initialize once before loop ──
    tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)
    out_writer = None
    out_path = ""
    start_time = time.time()
    frame_num = 0
    total_frames = 0

    # ════════════════════════════════════════════════
    # ── PRE-READ TOTAL FRAME COUNT ──
    # Probe the video file before the loop so tqdm
    # can show a proper [current/total] progress bar.
    # ════════════════════════════════════════════════
    if not source.isnumeric():
        _probe = cv2.VideoCapture(source)
        if _probe.isOpened():
            total_frames = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
            _probe.release()
        print(f"📹 Total frames in video: {total_frames}")

    # ── TQDM PROGRESS BAR ──
    # Shows:  45%|████████          | 450/1000 frames [00:12<00:14, 34.2 frame/s]
    pbar = tqdm(
        dataset,
        total=total_frames if total_frames > 0 else None,
        unit="frame",
        dynamic_ncols=True,
        color="green",
        bar_format=("{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}, {rate_fmt}]"),
    )

    for data in pbar:
        if STOP_REQUESTED:
            break

        _path, im, im0s, _vid_cap, _s = data

        frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
        frame_h, frame_w = frame.shape[:2]

        im_tensor = torch.from_numpy(im).to(device).float() / 255.0
        if im_tensor.ndim == 3:
            im_tensor = im_tensor[None]

        # ===== INFERENCE =====
        pred_raw = model(im_tensor)

        if isinstance(pred_raw, (list, tuple)) and len(pred_raw) == 2:
            pred_out, proto = pred_raw
        else:
            pred_out, proto = pred_raw, None

        pred = non_max_suppression(pred_out, conf_thres, iou_thres, nm=32 if proto is not None else 0)

        detections = []

        if len(pred[0]):
            det = pred[0]

            masks_binary = None
            if proto is not None and det.shape[1] > 6:
                mask_coeffs = det[:, 6:]
                boxes_scaled = scale_boxes(im_tensor.shape[2:], det[:, :4].clone(), frame.shape)
                masks_raw = process_mask(proto[0], mask_coeffs, det[:, :4], im_tensor.shape[2:], upsample=True)
                masks_binary = []
                for m in masks_raw:
                    m_np = m.cpu().numpy()
                    m_resized = cv2.resize(m_np, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                    masks_binary.append((m_resized > 0.5).astype(np.uint8))
                det[:, :4] = boxes_scaled.round()
            else:
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], frame.shape).round()

            for i in range(len(det)):
                x1, y1, x2, y2 = map(int, det[i, :4])
                conf = float(det[i, 4])
                cls = int(det[i, 5])  # always column 5, safe for seg models
                detections.append([x1, y1, x2, y2, conf, cls, masks_binary[i] if masks_binary else None])

        # ===== SORT TRACKING =====
        tracks = tracker.update(
            np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections]) if detections else np.empty((0, 5))
        )

        # ===== AUTO OVERLAP DETECTION =====
        OVERLAP_IOU_THRESHOLD = 0.05
        overlapping_ids = set()
        track_list = list(tracks.astype(int))
        for i in range(len(track_list)):
            for j in range(i + 1, len(track_list)):
                x1a, y1a, x2a, y2a, id_a = track_list[i]
                x1b, y1b, x2b, y2b, id_b = track_list[j]
                ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
                ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter > 0:
                    area_a = (x2a - x1a) * (y2a - y1a)
                    area_b = (x2b - x1b) * (y2b - y1b)
                    iou = inter / (area_a + area_b - inter + 1e-6)
                    if iou >= OVERLAP_IOU_THRESHOLD:
                        overlapping_ids.add(id_a)
                        overlapping_ids.add(id_b)

        for x1, y1, x2, y2, track_id in tracks.astype(int):
            best_iou, best_det = 0, None
            cls_name = "unknown"

            for d in detections:
                dx1, dy1, dx2, dy2 = d[:4]
                ix1, iy1 = max(x1, dx1), max(y1, dy1)
                ix2, iy2 = min(x2, dx2), min(y2, dy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (dx2 - dx1) * (dy2 - dy1)
                iou = inter / (area1 + area2 - inter + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_det = d
                    cls_name = names.get(int(d[5]), f"class_{d[5]}")

            color = get_class_color(cls_name)
            is_overlapping = track_id in overlapping_ids

            # ===== MASK OVERLAY =====
            if best_det is not None and best_det[6] is not None:
                draw_mask_overlay(frame, best_det[6], color, alpha=0.35)

            # ===== BOUNDING BOX =====
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ===== OVERLAP WARNING =====
            if is_overlapping:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "! OVERLAP", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ══════════════════════════════════════════════
            # COUNTING LOGIC
            # ══════════════════════════════════════════════
            if cls_name in OCR_CLASSES:
                # label class — count only on predefined OCR match
                if track_id not in track_ocr_text and best_det is not None:
                    mask = best_det[6]
                    if mask is not None:
                        ys, xs = np.where(mask > 0)
                        if len(xs) and len(ys):
                            mx1, my1 = int(xs.min()), int(ys.min())
                            mx2, my2 = int(xs.max()), int(ys.max())
                            crop = frame[my1:my2, mx1:mx2]
                        else:
                            crop = frame[best_det[1] : best_det[3], best_det[0] : best_det[2]]
                    else:
                        crop = frame[best_det[1] : best_det[3], best_det[0] : best_det[2]]

                    if crop.size:
                        full, norm, un_code, un_desc, matched_key = ocr_label(crop)

                        if un_code or matched_key:
                            display_text = un_code if un_code else EXPECTED_LABEL_TEXTS.get(matched_key, matched_key)
                            track_ocr_text[track_id] = {
                                "full": full,
                                "norm": norm,
                                "un_code": un_code,
                                "un_desc": un_desc,
                                "matched_key": matched_key,
                                "display": display_text,
                            }
                            count_key = display_text
                            ocr_text_counts[count_key] = ocr_text_counts.get(count_key, 0) + 1

                            if track_id not in counted_track_ids:
                                counted_track_ids[track_id] = cls_name
                                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                                if is_overlapping:
                                    class_counts["overlap"] = class_counts.get("overlap", 0) + 1

            else:
                # all other classes — count on first detection
                if track_id not in counted_track_ids:
                    counted_track_ids[track_id] = cls_name
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    if is_overlapping:
                        class_counts["overlap"] = class_counts.get("overlap", 0) + 1

            # ===== FRAME LABEL TEXT =====
            label = f"{cls_name} ID:{track_id}"
            if track_id in track_ocr_text:
                ocr = track_ocr_text[track_id]
                label += f" | {ocr['display']}"
                if ocr["un_desc"]:
                    cv2.putText(frame, ocr["un_desc"], (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # =============================================
        # LEFT PANEL
        # =============================================
        y_offset = 20

        draw_text_with_border(frame, "=== CLASS COUNTS ===", (15, y_offset), text_color=(0, 215, 255))
        y_offset += 22

        for cls_key, cnt in sorted(class_counts.items()):
            cls_color = get_class_color(cls_key)
            draw_text_with_border(
                frame, f"{cls_key:<12}: {cnt}", (15, y_offset), text_color=cls_color, border_color=cls_color
            )
            y_offset += 20

        y_offset += 8

        draw_text_with_border(frame, "=== LABEL OCR ===", (15, y_offset), text_color=(0, 215, 255))
        y_offset += 22

        if ocr_text_counts:
            for detected_text, cnt in sorted(ocr_text_counts.items()):
                draw_text_with_border(frame, f"{detected_text} : {cnt}", (15, y_offset), text_color=(0, 255, 255))
                y_offset += 20
        else:
            draw_text_with_border(frame, "no labels read yet", (15, y_offset), text_color=(80, 80, 80))

        # ===== VIDEO WRITER =====
        if out_writer is None:
            h, w = frame.shape[:2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(project) / name
            out_dir.mkdir(parents=True, exist_ok=True)
            source_stem = Path(source).stem  # e.g. "sample_1"
            out_path = str(out_dir / f"{source_stem}_{timestamp}.mp4")
            out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            print(f"\n💾 Saving output to: {out_path}")

        out_writer.write(frame)

        # ===== UPDATE PROGRESS BAR SUFFIX =====
        frame_num += 1
        elapsed = time.time() - start_time
        fps_live = frame_num / elapsed if elapsed > 0 else 0
        pbar.set_postfix(
            {
                "FPS": f"{fps_live:.1f}",
                "Tracks": len(tracks),
                "OCR'd": len(track_ocr_text),
            }
        )

    # ← loop ends here
    pbar.close()

    if out_writer:
        out_writer.release()
        print(f"\n✅ Output saved to: {out_path}")

    # ===== FINAL TERMINAL SUMMARY =====
    print("\n" + "=" * 55)
    print("  FINAL CLASS COUNT SUMMARY  (unique objects)")
    print("=" * 55)
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls:<16}: {cnt}")

    print("\n  LABEL OCR BREAKDOWN  (predefined text, unique per track)")
    print("-" * 55)
    if ocr_text_counts:
        for count_key, cnt in sorted(ocr_text_counts.items()):
            un_desc = UN_DESCRIPTIONS.get(count_key, "")
            display = EXPECTED_LABEL_TEXTS.get(count_key, count_key)
            if un_desc:
                print(f"  {count_key} ({un_desc}): {cnt}")
            else:
                print(f"  {display}: {cnt}")
    else:
        print("  no predefined label text detected")
    print("=" * 55)
    print("✅ Finished processing video")


# ================= CLI =================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/ocr")
    parser.add_argument("--name", default="exp")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
