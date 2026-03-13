"""
Script 01 - Dataset Exploration
================================
Run this first to understand the dataset before training.

Usage:
    python 01_explore_data.py

Outputs (saved to outputs/):
    - data_stats.json       : dataset statistics
    - sample_annotations.png: grid of sample images with bounding boxes drawn
"""

import os
import json
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = {
    "train": os.path.join(DATA_DIR, "train"),
    "valid": os.path.join(DATA_DIR, "valid"),
    "test":  os.path.join(DATA_DIR, "test"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def polygon_to_bbox(coords):
    """Convert a flat YOLO polygon [x1,y1,x2,y2,...] to (cx,cy,w,h) bbox."""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    return cx, cy, w, h


def read_labels(label_path):
    """Return list of (class_id, polygon_coords) from a YOLO label file."""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            coords   = [float(v) for v in parts[1:]]
            annotations.append((class_id, coords))
    return annotations


def count_per_image(split_dir):
    """Return a list of annotation counts, one per image in the split."""
    label_dir  = os.path.join(split_dir, "labels")
    image_dir  = os.path.join(split_dir, "images")
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.png"))
    counts = []
    for img_path in image_files:
        base       = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        annotations = read_labels(label_path)
        counts.append(len(annotations))
    return counts


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DATASET EXPLORATION")
    print("=" * 60)

    stats = {}

    # ── Per-split statistics ──────────────────────────────────────────────────
    for split_name, split_dir in SPLITS.items():
        image_dir = os.path.join(split_dir, "images")
        n_images  = len(
            glob.glob(os.path.join(image_dir, "*.jpg")) +
            glob.glob(os.path.join(image_dir, "*.png"))
        )
        counts = count_per_image(split_dir)

        split_stats = {
            "n_images":       n_images,
            "total_objects":  int(sum(counts)),
            "avg_per_image":  round(float(np.mean(counts)), 3),
            "max_per_image":  int(max(counts)) if counts else 0,
            "min_per_image":  int(min(counts)) if counts else 0,
            "std_per_image":  round(float(np.std(counts)), 3),
            "count_histogram": dict(Counter(counts)),
        }
        stats[split_name] = split_stats

        print(f"\n[{split_name.upper()}]")
        print(f"  Images          : {n_images}")
        print(f"  Total shoes     : {split_stats['total_objects']}")
        print(f"  Avg per image   : {split_stats['avg_per_image']}")
        print(f"  Max per image   : {split_stats['max_per_image']}")
        print(f"  Distribution    : {dict(sorted(Counter(counts).items()))}")

    # ── Save stats ────────────────────────────────────────────────────────────
    stats_path = os.path.join(OUTPUT_DIR, "data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")

    # ── Sample visualisation ──────────────────────────────────────────────────
    print("\nGenerating sample annotation visualisation...")

    # Pick images that have at least 1 annotation for a better visual
    image_dir  = os.path.join(SPLITS["train"], "images")
    label_dir  = os.path.join(SPLITS["train"], "labels")
    all_images = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                 glob.glob(os.path.join(image_dir, "*.png"))

    annotated = []
    for img_path in all_images:
        base       = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        if read_labels(label_path):
            annotated.append(img_path)

    random.seed(42)
    sample = random.sample(annotated, min(12, len(annotated)))

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Sample Training Images with Shoe Annotations", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, img_path in zip(axes, sample):
        img  = Image.open(img_path).convert("RGB")
        W, H = img.size

        base       = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        annotations = read_labels(label_path)

        ax.imshow(img)
        ax.set_title(f"Shoes: {len(annotations)}", fontsize=9)
        ax.axis("off")

        # Draw bounding boxes derived from polygons
        for _, coords in annotations:
            cx, cy, w, h = polygon_to_bbox(coords)
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            bw = w * W
            bh = h * H
            rect = patches.Rectangle(
                (x1, y1), bw, bh,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

    # Hide unused axes
    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.tight_layout()
    vis_path = os.path.join(OUTPUT_DIR, "sample_annotations.png")
    plt.savefig(vis_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Visualisation saved to: {vis_path}")

    # ── Count distribution plot ───────────────────────────────────────────────
    import seaborn as sns
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Shoe Count Distribution per Split", fontsize=13, fontweight="bold")

    colors = {"train": "steelblue", "valid": "darkorange", "test": "green"}
    for ax, (split_name, split_dir) in zip(axes, SPLITS.items()):
        counts = count_per_image(split_dir)
        hist   = Counter(counts)
        keys   = sorted(hist.keys())
        vals   = [hist[k] for k in keys]
        ax.bar([str(k) for k in keys], vals, color=colors[split_name], edgecolor="black")
        ax.set_title(split_name.capitalize(), fontsize=11)
        ax.set_xlabel("Shoes per image")
        ax.set_ylabel("Number of images")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    dist_path = os.path.join(OUTPUT_DIR, "count_distribution.png")
    plt.savefig(dist_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Distribution chart saved to: {dist_path}")

    print("\nDone! Run 02_train.py next.")


if __name__ == "__main__":
    main()
