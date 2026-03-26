import argparse
import os
import pathlib
import re
import signal
import sys
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
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ================= YOLOv5 SEG =================
from sort.sort import Sort

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode

# ================= CONFIG =================
LINE_X = 800
BUFFER_PX = 100
BUFFER_SECONDS = 10
STOP_REQUESTED = False

# ================= CLASS SETUP =================
# Only 'label' class gets OCR — all others just get detection + tracking
OCR_CLASSES = {"label"}

CLASS_COLORS = {
    "label": (0, 255, 0),  # Green
    "caution": (0, 165, 255),  # Orange
    "crown": (128, 0, 128),  # Purple
    "up": (255, 255, 0),  # Cyan
    "box": (200, 200, 200),  # Grey
    "overlap": (0, 0, 255),  # Red
}


def get_class_color(cls_name):
    return CLASS_COLORS.get(cls_name, (255, 255, 255))  # white fallback


# ================= OCR CONFIG =================
# Normalized text (uppercased, no spaces/hyphens) → display name
EXPECTED_LABEL_TEXTS = {
    # UN Battery labels
    "UN3481": "UN3481 - Lithium Battery (With Equipment)",
    "UN3480": "UN3480 - Lithium Battery (Standalone)",
    "UN1066": "UN1066 - Nitrogen Compressed",
    # Lithium battery text variants
    "LITHIUMIONBATTERIES": "LITHIUM ION BATTERIES",
    "LITHIUMIONBATTERIESFORBIDDENFORTRANSPORTABOARDPASSENGERAIRCRAFT": "LITHIUM ION BATTERIES",
    # Battery / spill labels
    "NONSPILLABLEBATTERY": "NONSPILLABLE BATTERY",
    # Fragile
    "FRAGILE": "FRAGILE",
    # Nitrogen / gas labels
    "NITROGENCOMPRESSED": "NITROGEN COMPRESSED",
    "NONFLAMMABLEGAS": "NON-FLAMMABLE GAS",
    "DOTSP10898": "DOT-SP 10898",
    "NITROGENHYDRAULICACCUMULATORS": "NITROGEN HYDRAULIC ACCUMULATORS",
    # Facilities
    "FACILITIESMAINTENANCEUSE": "FACILITIES MAINTENANCE USE",
}

UN_DESCRIPTIONS = {
    "UN3481": "Lithium Battery - Packed With Equipment",
    "UN3480": "Lithium Battery - Standalone",
    "UN1066": "Nitrogen, Compressed",
}

ocr_reader = easyocr.Reader(["en"], gpu=False)


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


def draw_text_with_gold_box(
    img,
    text,
    pos,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
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
    """Blend a binary mask onto the frame with a given color and transparency."""
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[mask_bin > 0] = color
    cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0, frame)
    contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 1)


def ocr_label(crop):
    """Try OCR at 4 rotations. Return (full_text, normalized, un_code, un_desc, matched_key). Only fires for 'label'
    class crops.
    """
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

        results = ocr_reader.readtext(gray, detail=1, paragraph=True)
        texts = [t for _, t, c in results if c > 0.3]

        if texts:
            full = " ".join(texts)
            norm = normalize_text(full)

            # Extract UN code (e.g. UN3480, UN3481, UN1066)
            un_match = re.search(r"UN\d{4}", norm)
            un_code = un_match.group(0) if un_match else None
            un_desc = UN_DESCRIPTIONS.get(un_code, "") if un_code else ""

            # Match against known label texts
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
    # track_id → OCR result dict (only populated for 'label' class)
    track_ocr_text = {}

    # count_key → int  (UN code if found, else matched_key)
    # This is the SAME logic as original: incremented once per unique track_id
    ocr_text_counts = {}

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    dataset = (
        LoadStreams(source, img_size=imgsz, stride=stride)
        if source.isnumeric()
        else LoadImages(source, img_size=imgsz, stride=stride)
    )

    tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

    for data in tqdm(dataset, unit="frame"):
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

        # Seg models return (pred, proto); detection models return pred directly
        if isinstance(pred_raw, (list, tuple)) and len(pred_raw) == 2:
            pred_out, proto = pred_raw
        else:
            pred_out = pred_raw
            proto = None

        pred = non_max_suppression(pred_out, conf_thres, iou_thres, nm=32 if proto is not None else 0)

        detections = []  # [x1, y1, x2, y2, conf, cls_idx, mask_or_None]

        if len(pred[0]):
            det = pred[0]

            # ===== PROCESS MASKS (seg only) =====
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

            for i, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append([x1, y1, x2, y2, conf.item(), int(cls), masks_binary[i] if masks_binary else None])

        # ===== SORT TRACKING =====
        tracks = tracker.update(
            np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections]) if detections else np.empty((0, 5))
        )

        for x1, y1, x2, y2, track_id in tracks.astype(int):
            best_iou, best_det = 0, None
            cls_name = "unknown"

            # Match track box back to best detection via IoU
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
                    cls_name = names[d[5]]

            color = get_class_color(cls_name)

            # ===== DRAW MASK OVERLAY (all classes) =====
            if best_det is not None and best_det[6] is not None:
                draw_mask_overlay(frame, best_det[6], color, alpha=0.35)

            # ===== BOUNDING BOX (all classes) =====
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ===== OVERLAP CLASS — extra red warning =====
            if cls_name == "overlap":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "! OVERLAP", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ===== OCR — ONLY for 'label' class, ONLY once per track_id =====
            if cls_name in OCR_CLASSES and track_id not in track_ocr_text and best_det is not None:
                mask = best_det[6]
                if mask is not None:
                    # Use tight mask bounding box for cleaner OCR crop
                    ys, xs = np.where(mask > 0)
                    if len(xs) and len(ys):
                        mx1, my1 = int(xs.min()), int(ys.min())
                        mx2, my2 = int(xs.max()), int(ys.max())
                        crop = frame[my1:my2, mx1:mx2]
                    else:
                        crop = frame[best_det[1] : best_det[3], best_det[0] : best_det[2]]
                else:
                    # Fallback: bbox crop
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
                        # ── SAME COUNT LOGIC AS ORIGINAL ──
                        # Count key: UN code if found, else the matched normalized key
                        count_key = un_code if un_code else matched_key
                        ocr_text_counts[count_key] = ocr_text_counts.get(count_key, 0) + 1

            # ===== LABEL TEXT ON FRAME =====
            label = f"{cls_name} ID:{track_id}"
            if track_id in track_ocr_text:
                ocr = track_ocr_text[track_id]
                label += f" | {ocr['display']}"
                # Show UN description on a second line below the box
                if ocr["un_desc"]:
                    cv2.putText(frame, ocr["un_desc"], (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ===== LEFT PANEL — OCR COUNTS (label class only) =====
        y_offset = 30
        draw_text_with_gold_box(frame, "=== LABEL COUNTS ===", (15, y_offset), text_color=(0, 215, 255))
        y_offset += 25

        for count_key, cnt in sorted(ocr_text_counts.items()):
            # Show friendly name if available, else raw key
            display_name = EXPECTED_LABEL_TEXTS.get(count_key, count_key)
            draw_text_with_gold_box(frame, f"{display_name} : {cnt}", (15, y_offset), text_color=(0, 255, 255))
            y_offset += 22

        cv2.imshow("YOLOv5-Seg | Label OCR", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break

    cv2.destroyAllWindows()

    # ===== FINAL SUMMARY IN TERMINAL =====
    print("\n" + "=" * 50)
    print("  FINAL LABEL COUNT SUMMARY")
    print("=" * 50)
    for count_key, cnt in sorted(ocr_text_counts.items()):
        display_name = EXPECTED_LABEL_TEXTS.get(count_key, count_key)
        un_desc = UN_DESCRIPTIONS.get(count_key, "")
        if un_desc:
            print(f"  {count_key} ({un_desc}): {cnt}")
        else:
            print(f"  {display_name}: {cnt}")
    print("=" * 50)
    print("✅ Finished processing video")


# ================= CLI =================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to yolov5-seg .pt file")
    parser.add_argument("--source", required=True, help="Video file path or webcam index")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="", help="cuda device or cpu")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
