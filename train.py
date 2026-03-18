import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed.\n  Run:  pip install ultralytics")
    raise SystemExit(1)


def train(
    weights: str  = "yolo_weights/yolov8n.pt",
    yaml:    str  = "data.yaml",
    epochs:  int  = 80,
    imgsz:   int  = 640,
    batch:   int  = 8,
    name:    str  = "pancard_detector",
    device:  str  = "",          # "" = auto (GPU if available, else CPU)
):
    weights_path = Path(weights)
    yaml_path    = Path(yaml)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Download yolov8n.pt from:\n"
            "  https://github.com/ultralytics/assets/releases/latest\n"
            "and place it in the yolo_weights/ folder."
        )

    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    print(f"""
╔══════════════════════════════════════════╗
║         PAN Card — YOLO Training         ║
╠══════════════════════════════════════════╣
║  Weights : {weights:<31}║
║  Data    : {yaml:<31}║
║  Epochs  : {str(epochs):<31}║
║  Img size: {str(imgsz):<31}║
║  Batch   : {str(batch):<31}║
╚══════════════════════════════════════════╝
""")

    # ── Load model ──────────────────────────────────────────
    model = YOLO(str(weights_path))

    # ── Train ───────────────────────────────────────────────
    results = model.train(
        data      = str(yaml_path),
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        name      = name,
        device    = device if device else None,
        patience  = 0,          # early-stop if no improvement for 20 epochs
        save      = True,
        plots     = True,
        augment   = True,        # YOLOv8 built-in augmentation (mosaic, mixup …)
        hsv_h     = 0.015,       # hue augmentation
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 15.0,        # rotation ± 15°
        translate = 0.1,
        scale     = 0.5,
        shear     = 5.0,
        flipud    = 0.1,
        fliplr    = 0.5,
        mosaic    = 1.0,
    )

    # ── Report ──────────────────────────────────────────────
    best = Path(results.save_dir) / "weights" / "best.pt"
    last = Path(results.save_dir) / "weights" / "last.pt"
    print(f"""
[✓] Training complete!
    Best weights : {best}
    Last weights : {last}
    Results dir  : {results.save_dir}

Next step — run detect.py with your trained weights:
    python detect.py --weights {best} --input test_image.jpg
""")
    return str(best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO PAN Card Detector")
    parser.add_argument("--weights", default="yolo_weights/yolov8n.pt",
                        help="Pre-trained YOLO weights path")
    parser.add_argument("--yaml",    default="data.yaml",
                        help="Dataset YAML path")
    parser.add_argument("--epochs",  type=int, default=80)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--name",    default="pancard_detector",
                        help="Run name (saved under runs/detect/<name>)")
    parser.add_argument("--device",  default="",
                        help="Device: '' auto | '0' GPU 0 | 'cpu'")
    args = parser.parse_args()

    train(args.weights, args.yaml, args.epochs,
          args.imgsz, args.batch, args.name, args.device)
