import argparse
import csv
from datetime import datetime
from pathlib import Path

# YOLOv5 imports
from models.common import DetectMultiBackend
from utils.torch_utils import select_device


# --------------------------------------------------
# Write verification results to CSV (inside folder)
# --------------------------------------------------
def write_log_csv(weights_path, model, stride, recommended_imgsz, device):
    weights_path = Path(weights_path)

    # Create folder: verification_logs (next to weights)
    log_dir = weights_path.parent / "verification_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # CSV file name based on weights file name
    log_path = log_dir / f"{weights_path.stem}_verification_log.csv"

    file_exists = log_path.exists()

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header if first time
        if not file_exists:
            writer.writerow(
                ["timestamp", "weights_path", "device", "num_classes", "class_names", "stride", "recommended_imgsz"]
            )

        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(weights_path),
                str(device),
                len(model.names),
                ", ".join(model.names.values()) if isinstance(model.names, dict) else model.names,
                stride,
                recommended_imgsz,
            ]
        )

    print(f"\n📝 Log saved to: {log_path}")


# --------------------------------------------------
# Verify weights
# --------------------------------------------------
def verify_weights(weights):
    weights = Path(weights)

    if not weights.exists():
        print("❌ Weights file not found:", weights)
        return

    # Auto device selection
    device = select_device("")

    print(f"\n🔍 Verifying weights: {weights}")
    print(f"🖥️ Device: {device}")

    # Load model
    model = DetectMultiBackend(weights, device=device)

    print("\n✅ Model loaded successfully")
    print("• Backend (.pt):", model.pt)
    print("• Number of classes:", len(model.names))
    print("• Class names:", model.names)

    # Handle stride safely
    stride = model.stride
    if isinstance(stride, (list, tuple)):
        stride = max(stride)
    elif hasattr(stride, "max"):
        stride = int(stride.max())
    else:
        stride = int(stride)

    print("• Stride:", stride)

    recommended_imgsz = stride * 32

    print("\n📌 Image Size Information")
    print(f"👉 Recommended inference imgsz: {recommended_imgsz}")

    # Model parameter count
    total_params = sum(p.numel() for p in model.model.parameters())
    print("\n📊 Model Info")
    print(f"• Total Parameters: {total_params:,}")
    print(f"• FP16 Supported: {device.type != 'cpu'}")

    # Write CSV inside folder
    write_log_csv(weights, model, stride, recommended_imgsz, device)

    print("\n🎯 Weight verification completed successfully")


# --------------------------------------------------
# Argparse
# --------------------------------------------------
def parse_opt():
    parser = argparse.ArgumentParser(description="YOLOv5 .pt weight verifier with folder logging")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLOv5 .pt file")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    verify_weights(opt.weights)
