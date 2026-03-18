import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed.\n  Run:  pip install ultralytics")
    raise SystemExit(1)


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────
YELLOW = (0, 220, 220)
RED    = (30, 30, 220)
GREEN  = (50, 200, 50)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
FONT   = cv2.FONT_HERSHEY_DUPLEX


def draw_pan_box(image: np.ndarray, box, conf: float) -> np.ndarray:
    """Draw a yellow bounding box + confidence label."""
    out = image.copy()
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, 3)

    label  = f"PAN Card  {conf:.0%}"
    scale  = 0.7
    thick  = 2
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, thick)
    pad = 6
    # Background pill behind label
    cv2.rectangle(out, (x1, y1 - th - pad * 2),
                  (x1 + tw + pad * 2, y1), YELLOW, -1)
    cv2.putText(out, label, (x1 + pad, y1 - pad),
                FONT, scale, BLACK, thick, cv2.LINE_AA)
    return out


def draw_not_pan(image: np.ndarray) -> np.ndarray:
    """Red overlay with centred 'Not a PAN Card' message."""
    out = image.copy()
    h, w = out.shape[:2]

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.40, out, 0.60, 0, out)

    msg   = "Not a PAN Card"
    scale = min(w, h) / 380
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(msg, FONT, scale, thick)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(out, msg, (x + 2, y + 2), FONT, scale, BLACK,  thick + 2, cv2.LINE_AA)
    cv2.putText(out, msg, (x,     y),     FONT, scale, WHITE,  thick,     cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def run_detection(model: YOLO, image_path: str,
                  output_dir: Path, conf_threshold: float = 0.40) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Cannot read image: {image_path}")
        return {}

    # Resize large images for speed (preserves aspect ratio)
    h, w = image.shape[:2]
    if max(h, w) > 1200:
        scale  = 1200 / max(h, w)
        image  = cv2.resize(image, (int(w * scale), int(h * scale)))

    results = model.predict(image, conf=conf_threshold, verbose=False)
    result  = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        name   = model.names[cls_id]
        xyxy   = box.xyxy[0].cpu().numpy()
        detections.append({"class": name, "conf": conf, "box": xyxy})

    pan_detections = [d for d in detections if d["class"] == "pancard"]

    # ── Draw output ──────────────────────────────────────────
    if pan_detections:
        out_img = image.copy()
        for det in pan_detections:
            out_img = draw_pan_box(out_img, det["box"], det["conf"])
        detected = True
        top_conf = max(d["conf"] for d in pan_detections)
    else:
        out_img  = draw_not_pan(image)
        detected = False
        top_conf = 0.0

    out_name = output_dir / (Path(image_path).stem + "_result.jpg")
    cv2.imwrite(str(out_name), out_img)

    return {
        "image": image_path,
        "pan_detected": detected,
        "confidence": top_conf,
        "num_detections": len(pan_detections),
        "output": str(out_name),
    }


def pretty_print(r: dict):
    status = "✓  PAN Card Detected" if r["pan_detected"] else "✗  Not a PAN Card"
    print(f"""
  Image      : {Path(r['image']).name}
  Status     : {status}
  Confidence : {r['confidence']:.0%}
  Output     : {r['output']}""")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PAN Card YOLO Detector")
    parser.add_argument("--weights", "-w", required=True,
                        help="Path to trained best.pt weights file")
    parser.add_argument("--input",   "-i", required=True,
                        help="Input image path or folder")
    parser.add_argument("--output",  "-o", default="output",
                        help="Output folder (default: ./output)")
    parser.add_argument("--conf",    "-c", type=float, default=0.40,
                        help="Confidence threshold 0.0-1.0 (default: 0.40)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    input_path = Path(args.input)
    if input_path.is_dir():
        image_files = (list(input_path.glob("*.jpg")) +
                       list(input_path.glob("*.jpeg")) +
                       list(input_path.glob("*.png")))
        if not image_files:
            print(f"[ERROR] No images found in {input_path}")
            sys.exit(1)
        print(f"[INFO] Processing {len(image_files)} images …\n")
        pan_count = 0
        for img_path in image_files:
            r = run_detection(model, str(img_path), output_dir, args.conf)
            pretty_print(r)
            if r["pan_detected"]:
                pan_count += 1
        print(f"\n[SUMMARY]  {pan_count}/{len(image_files)} images contain a PAN card.")
    else:
        r = run_detection(model, str(input_path), output_dir, args.conf)
        pretty_print(r)
        print()
        sys.exit(0 if r["pan_detected"] else 1)


if __name__ == "__main__":
    main()
