import cv2
import numpy as np
import os
import argparse
import random
from pathlib import Path


def resize_aug(img):
    """Resize to a random scale between 60%-130% then pad/crop back."""
    h, w = img.shape[:2]
    scale = random.uniform(0.6, 1.3)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    # Pad or crop to original size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y_off = max(0, (h - new_h) // 2)
    x_off = max(0, (w - new_w) // 2)
    y_end = min(h, y_off + new_h)
    x_end = min(w, x_off + new_w)
    ry_end = min(new_h, y_end - y_off)
    rx_end = min(new_w, x_end - x_off)
    canvas[y_off:y_end, x_off:x_end] = resized[:ry_end, :rx_end]
    return canvas


def blur_aug(img):
    """Random Gaussian or motion blur."""
    choice = random.choice(["gaussian", "motion"])
    if choice == "gaussian":
        k = random.choice([3, 5, 7, 9])
        return cv2.GaussianBlur(img, (k, k), 0)
    else:
        # Motion blur
        k = random.randint(5, 20)
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = np.ones(k) / k
        angle  = random.uniform(0, 180)
        M = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        return cv2.filter2D(img, -1, kernel)


def rotate_aug(img):
    """Rotate by -45° to +45° with white background fill."""
    h, w = img.shape[:2]
    angle = random.uniform(-45, 45)
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # Expand canvas so corners don't get clipped
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    return cv2.resize(rotated, (w, h))


def brightness_aug(img):
    """Random brightness / contrast shift."""
    alpha = random.uniform(0.6, 1.4)   # contrast
    beta  = random.randint(-40, 40)    # brightness
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def noise_aug(img):
    """Add Gaussian noise."""
    sigma = random.uniform(5, 25)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def perspective_aug(img):
    """Random slight perspective/skew warp."""
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.12)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, margin),        random.randint(0, margin)],
        [w - random.randint(0, margin),    random.randint(0, margin)],
        [w - random.randint(0, margin),    h - random.randint(0, margin)],
        [random.randint(0, margin),        h - random.randint(0, margin)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))


def flip_aug(img):
    """Horizontal flip."""
    return cv2.flip(img, 1)


AUG_FUNCTIONS = [resize_aug, blur_aug, rotate_aug,
                 brightness_aug, noise_aug, perspective_aug, flip_aug]


def augment_image(img, num_augments=3):
    """Apply a random chain of augmentations."""
    chosen = random.sample(AUG_FUNCTIONS, k=min(num_augments, len(AUG_FUNCTIONS)))
    result = img.copy()
    for fn in chosen:
        result = fn(result)
    return result


def run(input_dir: str, output_dir: str, count: int):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("*.jpg")) + \
                  list(input_path.glob("*.jpeg")) + \
                  list(input_path.glob("*.png"))

    if not image_files:
        print(f"[ERROR] No images found in {input_dir}")
        return

    print(f"[INFO] Found {len(image_files)} source image(s) → generating {count} augmented images …")

    generated = 0
    idx = 0
    while generated < count:
        src_img = cv2.imread(str(image_files[idx % len(image_files)]))
        if src_img is None:
            idx += 1
            continue

        aug = augment_image(src_img, num_augments=random.randint(2, 4))
        out_name = output_path / f"pan_{generated:04d}.jpg"
        cv2.imwrite(str(out_name), aug)
        generated += 1
        idx += 1

        if generated % 10 == 0:
            print(f"  … {generated}/{count}")

    print(f"[DONE] {generated} images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAN Card Image Augmentation")
    parser.add_argument("--input",  "-i", required=True, help="Folder with raw PAN card images")
    parser.add_argument("--output", "-o", required=True, help="Output folder for augmented images")
    parser.add_argument("--count",  "-n", type=int, default=80,
                        help="Total augmented images to generate (default: 80)")
    args = parser.parse_args()
    run(args.input, args.output, args.count)
