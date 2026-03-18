import os
import shutil
import random
import argparse
from pathlib import Path


STRUCTURE = """
project_yolo/
├── dataset/
│   ├── images/
│   │   ├── all/        ← all augmented images go here
│   │   ├── train/      ← auto-filled by this script
│   │   └── test/       ← auto-filled by this script
│   └── labels/
│       ├── all/        ← all label .txt files go here
│       ├── train/      ← auto-filled by this script (YOLO reads this)
│       └── test/       ← auto-filled by this script (YOLO reads this)
├── yolo_weights/
│   └── yolov8n.pt
├── runs/               ← training results saved here
├── data.yaml
├── augment.py
├── fix_all.py
├── fix_labels.py
├── setup_folders.py
├── train.py
├── detect.py
└── app.py
"""


def create_structure():
    dirs = [
        "dataset/images/all",
        "dataset/images/train",
        "dataset/images/test",
        "dataset/labels/all",
        "dataset/labels/train",
        "dataset/labels/test",
        "yolo_weights",
        "runs",
        "output",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("[✓] Folder structure created:")
    print(STRUCTURE)


def fix_label_classes(label_path: Path) -> bool:
    """Fix any class ID that is not 0 — returns True if file was changed."""
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    changed   = False
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] != "0":
            parts[0] = "0"
            changed  = True
        if parts:
            new_lines.append(" ".join(parts) + "\n")

    if changed:
        with open(label_path, "w") as f:
            f.writelines(new_lines)
    return changed


def clear_folder(folder: Path):
    """Delete all files inside a folder (keep the folder itself)."""
    if folder.exists():
        for f in folder.iterdir():
            if f.is_file():
                f.unlink()


def split_dataset(images_dir: str, labels_dir: str, split: float = 0.8):
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)

    if not img_dir.exists():
        print(f"[ERROR] Images folder not found: {img_dir}")
        return
    if not lbl_dir.exists():
        print(f"[ERROR] Labels folder not found: {lbl_dir}")
        return

    # ── Step 1: Clear old split folders ───────────────────────────────────────
    print("Clearing old train/test folders...")
    for folder in ["dataset/images/train", "dataset/images/test",
                   "dataset/labels/train", "dataset/labels/test"]:
        clear_folder(Path(folder))
    print("[✓] Cleared.\n")

    # ── Step 2: Separate PAN images from negatives ─────────────────────────────
    all_images = sorted(
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.jpeg")) +
        list(img_dir.glob("*.png"))
    )

    if not all_images:
        print(f"[ERROR] No images found in {images_dir}")
        return

    pan_images = []   # have a non-empty label file
    neg_images = []   # no label or empty label

    fixed_count = 0
    for img in all_images:
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            # Auto-fix class IDs while we're here
            if fix_label_classes(lbl):
                fixed_count += 1
            pan_images.append(img)
        else:
            neg_images.append(img)

    print(f"  PAN card images (labelled) : {len(pan_images)}")
    print(f"  Negative images (unlabelled): {len(neg_images)}")
    if fixed_count:
        print(f"  [✓] Auto-fixed class IDs in {fixed_count} label files")
    print()

    if len(pan_images) == 0:
        print("[ERROR] No labelled PAN card images found!")
        print("  Make sure your label .txt files are in:", labels_dir)
        print("  And each file has content like:  0 0.5 0.5 0.8 0.6")
        return

    # ── Step 3: Split PAN images 80/20 ────────────────────────────────────────
    random.shuffle(pan_images)
    n_train    = max(1, int(len(pan_images) * split))
    train_pans = pan_images[:n_train]
    test_pans  = pan_images[n_train:]

    # Ensure at least 1 image in test
    if len(test_pans) == 0:
        test_pans  = train_pans[-1:]
        train_pans = train_pans[:-1]

    train_img = Path("dataset/images/train")
    test_img  = Path("dataset/images/test")
    train_lbl = Path("dataset/labels/train")
    test_lbl  = Path("dataset/labels/test")

    # ── Step 4: Copy PAN images + labels to train and test ────────────────────
    for img in train_pans:
        shutil.copy(img, train_img / img.name)
        lbl = lbl_dir / (img.stem + ".txt")
        shutil.copy(lbl, train_lbl / lbl.name)

    for img in test_pans:
        shutil.copy(img, test_img / img.name)
        lbl = lbl_dir / (img.stem + ".txt")
        shutil.copy(lbl, test_lbl / lbl.name)

    # ── Step 5: Copy ALL negatives to train only ──────────────────────────────
    for img in neg_images:
        shutil.copy(img, train_img / img.name)

    # ── Step 6: Delete old YOLO cache files if present ────────────────────────
    for cache in ["dataset/labels/train.cache", "dataset/labels/test.cache"]:
        if Path(cache).exists():
            Path(cache).unlink()
            print(f"  [✓] Deleted old cache: {cache}")

    # ── Summary ────────────────────────────────────────────────────────────────
    train_img_count = len(list(train_img.iterdir()))
    test_img_count  = len(list(test_img.iterdir()))
    train_lbl_count = len(list(train_lbl.iterdir()))
    test_lbl_count  = len(list(test_lbl.iterdir()))

    print(f"""
╔══════════════════════════════════════════╗
║           Dataset Split Summary          ║
╠══════════════════════════════════════════╣
║  Train images  : {str(train_img_count):<24}║
║  Test images   : {str(test_img_count):<24}║
║  Train labels  : {str(train_lbl_count):<24}║
║  Test labels   : {str(test_lbl_count):<24}║
╠══════════════════════════════════════════╣
║  PAN in train  : {str(len(train_pans)):<24}║
║  PAN in test   : {str(len(test_pans)):<24}║
║  Negatives     : {str(len(neg_images)):<24}║
╚══════════════════════════════════════════╝

Next step → python train.py --epochs 100 --batch 8 --imgsz 640
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup project_yolo folder structure")
    parser.add_argument("--images", default="dataset/images/all/",
                        help="Source images folder (default: dataset/images/all/)")
    parser.add_argument("--labels", default="dataset/labels/all/",
                        help="Source labels folder (default: dataset/labels/all/)")
    parser.add_argument("--split",  type=float, default=0.8,
                        help="Train ratio (default: 0.8)")
    args = parser.parse_args()

    create_structure()
    split_dataset(args.images, args.labels, args.split)