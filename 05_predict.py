"""
Script 05 - Inference / Prediction
=====================================
Run the trained model on new images to count shoes.
Displays original image alongside the model's prediction (segmentation masks,
bounding boxes, confidence scores).

Default source: data/predict/len3.jpg
Sample images available in data/predict/:
    - len3.jpg   (default)
    - len43.jpg
    - len48.jpg

Usage:
    python 05_predict.py                                          # runs on len3.jpg
    python 05_predict.py --source data/predict/len43.jpg
    python 05_predict.py --source data/predict/                   # all 3 images
    python 05_predict.py --source data/predict/len3.jpg --weights outputs/runs/yolov8s-seg/weights/best.pt

Outputs (saved to outputs/predictions/):
    - <name>_comparison.png  : original vs predicted side by side
    - predictions_summary.csv : file name + predicted count
"""

import os
import sys
import glob
import argparse
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PRED_DIR   = os.path.join(OUTPUT_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)


def auto_find_weights():
    pattern    = os.path.join(OUTPUT_DIR, "runs", "*", "weights", "best.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def add_count_badge(img_bgr, count):
    """Overlay a 'Shoes: N' badge on the top-left corner of a BGR image."""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    text       = f"Shoes: {count}"
    font       = cv2.FONT_HERSHEY_DUPLEX
    scale      = max(0.6, min(w, h) / 400)
    thickness  = max(1, int(scale * 2))
    (tw, th), _= cv2.getTextSize(text, font, scale, thickness)
    pad = 8
    cv2.rectangle(img, (5, 5), (5 + tw + pad * 2, 5 + th + pad * 2), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (5 + tw + pad * 2, 5 + th + pad * 2), (0, 200, 0), 2)
    cv2.putText(img, text, (5 + pad, 5 + pad + th), font, scale, (0, 255, 0), thickness)
    return img


def predict_and_visualise(model, img_path, conf, iou):
    """
    Run inference on one image.
    Returns (count, comparison_save_path).
    The saved image is a side-by-side of original | prediction with masks drawn.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  WARNING: could not read {img_path}")
        return 0, None

    # ── Run model ─────────────────────────────────────────────────────────────
    results = model.predict(source=img_path, conf=conf, iou=iou, verbose=False)
    r       = results[0]

    # ── Annotated image via YOLO's built-in renderer ──────────────────────────
    # r.plot() draws segmentation masks, bounding boxes and confidence labels
    annotated_bgr = r.plot(
        conf   = True,   # show confidence scores
        labels = True,   # show class labels
        boxes  = True,   # show bounding boxes
        masks  = True,   # show segmentation masks (filled, semi-transparent)
        line_width = 2,
    )

    count         = len(r.boxes) if r.boxes is not None else 0
    annotated_bgr = add_count_badge(annotated_bgr, count)

    # ── Build side-by-side matplotlib figure ──────────────────────────────────
    orig_rgb = cv2.cvtColor(img_bgr,      cv2.COLOR_BGR2RGB)
    pred_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"{os.path.basename(img_path)}  —  Predicted shoes: {count}",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(pred_rgb)
    axes[1].set_title("Model prediction  (masks + boxes + confidence)", fontsize=11)
    axes[1].axis("off")

    # Add a legend for each detected shoe
    colors = plt.cm.Set1(np.linspace(0, 0.8, max(count, 1)))
    if count > 0:
        patches = [
            mpatches.Patch(color=colors[i], label=f"Shoe {i+1}")
            for i in range(count)
        ]
        axes[1].legend(handles=patches, loc="lower right", fontsize=9,
                       framealpha=0.8)

    plt.tight_layout()

    # ── Save figure ───────────────────────────────────────────────────────────
    base     = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(PRED_DIR, f"{base}_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

    return count, out_path


def main(args):
    print("=" * 60)
    print("  PREDICTION")
    print("=" * 60)

    # ── Weights ───────────────────────────────────────────────────────────────
    weights = args.weights or auto_find_weights()
    if not weights or not os.path.exists(weights):
        print("ERROR: No weights found. Train a model first with 02_train.py")
        print("       Or pass --weights path/to/best.pt")
        sys.exit(1)
    print(f"Weights : {weights}")
    print(f"Conf    : {args.conf}")
    print(f"IOU     : {args.iou}")

    model = YOLO(weights)

    # ── Collect image paths ───────────────────────────────────────────────────
    source = args.source
    if os.path.isdir(source):
        image_paths = sorted(
            glob.glob(os.path.join(source, "*.jpg")) +
            glob.glob(os.path.join(source, "*.png")) +
            glob.glob(os.path.join(source, "*.jpeg"))
        )
    elif os.path.isfile(source):
        image_paths = [source]
    else:
        print(f"ERROR: source path not found: {source}")
        sys.exit(1)

    if not image_paths:
        print(f"No images found at: {source}")
        sys.exit(1)

    print(f"\nProcessing {len(image_paths)} image(s)...\n")

    # ── Run inference ─────────────────────────────────────────────────────────
    rows = []
    for i, img_path in enumerate(image_paths, 1):
        count, out_path = predict_and_visualise(model, img_path, args.conf, args.iou)
        rows.append({"image": os.path.basename(img_path), "predicted_count": count})
        print(f"  [{i}/{len(image_paths)}] {os.path.basename(img_path):40s}  shoes={count}")
        if out_path:
            print(f"           -> {out_path}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(PRED_DIR, "predictions_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "predicted_count"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["predicted_count"] for r in rows)
    print(f"\nTotal shoes detected : {total}")
    print(f"Images processed     : {len(rows)}")
    print(f"Comparison images    : {PRED_DIR}")
    print(f"Summary CSV          : {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "predict", "len3.jpg"
)


def parse_args():
    p = argparse.ArgumentParser(description="Predict shoe count in images")
    p.add_argument("--source",  default=DEFAULT_SOURCE,
                   help="Image file or folder path (default: data/predict/len3.jpg)")
    p.add_argument("--weights", default=None,
                   help="Path to best.pt (auto-detected if omitted)")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="NMS IoU threshold (default: 0.45)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
