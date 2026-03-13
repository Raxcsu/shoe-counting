"""
Script 02 - Model Training
===========================
Trains a YOLO segmentation model on the shoe dataset.

Usage:
    python 02_train.py                          # train with default settings
    python 02_train.py --model yolov8n-seg.pt   # YOLOv8 nano  (fast, ~3.4M params)
    python 02_train.py --model yolov8s-seg.pt   # YOLOv8 small (better, ~11.8M params)
    python 02_train.py --model yolo11n-seg.pt   # YOLO11 nano  (newest)
    python 02_train.py --epochs 100 --batch 16

Outputs (saved to outputs/runs/<model_name>/):
    - weights/best.pt               : best model weights
    - weights/last.pt               : last epoch weights
    - results.csv                   : per-epoch training curves
    - training_summary_<model>.json : timing + final metrics summary
"""

import os
import sys
import json
import time
import argparse
import yaml
import torch
from datetime import datetime, timedelta
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Fix data.yaml ─────────────────────────────────────────────────────────────
def create_fixed_data_yaml():
    """
    The original data.yaml uses relative paths that don't match the actual
    folder structure. This creates a corrected version with absolute paths.
    """
    fixed = {
        "train": os.path.join(DATA_DIR, "train", "images"),
        "val":   os.path.join(DATA_DIR, "valid", "images"),
        "test":  os.path.join(DATA_DIR, "test",  "images"),
        "nc":    1,
        "names": ["shoe"],
    }
    # Use forward slashes even on Windows (ultralytics handles it fine)
    fixed["train"] = fixed["train"].replace("\\", "/")
    fixed["val"]   = fixed["val"].replace("\\", "/")
    fixed["test"]  = fixed["test"].replace("\\", "/")

    yaml_path = os.path.join(OUTPUT_DIR, "data_fixed.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(fixed, f, default_flow_style=False)
    print(f"Fixed data.yaml written to: {yaml_path}")
    return yaml_path


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    print("=" * 60)
    print("  TRAINING")
    print("=" * 60)

    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Device: CPU (training will be slow — consider a GPU)")
    print(f"Model : {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch : {args.batch}")
    print(f"Image : {args.imgsz}x{args.imgsz}")

    # Model name without extension (used for folder/file naming)
    model_tag = os.path.splitext(args.model)[0]

    # Fixed data yaml
    data_yaml = create_fixed_data_yaml()

    # Load pretrained model
    model = YOLO(args.model)

    # Training run directory
    run_dir = os.path.join(OUTPUT_DIR, "runs", model_tag)
    os.makedirs(run_dir, exist_ok=True)

    # ── Start training ────────────────────────────────────────────────────────
    print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')} ...")
    t_start = time.time()

    results = model.train(
        data        = data_yaml,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = device,
        project     = os.path.join(OUTPUT_DIR, "runs"),
        name        = model_tag,
        exist_ok    = True,           # overwrite if re-running
        patience    = args.patience,  # early stopping
        optimizer   = "AdamW",
        lr0         = 0.001,
        lrf         = 0.01,
        weight_decay= 0.0005,
        warmup_epochs = 3,
        cos_lr      = True,           # cosine LR schedule
        mosaic      = 1.0,            # mosaic augmentation
        degrees     = 15.0,           # rotation augment (matches dataset)
        plots       = True,           # save training plots
        save        = True,
        save_period = -1,             # only save best + last
        verbose     = True,
    )

    t_end     = time.time()
    elapsed   = t_end - t_start
    elapsed_h = str(timedelta(seconds=int(elapsed)))

    print(f"\nTraining finished at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total training time: {elapsed_h} ({elapsed:.1f} s)")

    # ── Extract final metrics ─────────────────────────────────────────────────
    # results.results_dict contains the final epoch's metrics
    metrics_dict = {}
    try:
        metrics_dict = {k: float(v) for k, v in results.results_dict.items()}
    except Exception:
        pass

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "model":             args.model,
        "model_tag":         model_tag,
        "epochs_requested":  args.epochs,
        "epochs_trained":    int(getattr(results, "epoch", args.epochs)),
        "imgsz":             args.imgsz,
        "batch":             args.batch,
        "device":            device,
        "training_time_s":   round(elapsed, 1),
        "training_time_hms": elapsed_h,
        "trained_at":        datetime.now().isoformat(),
        "best_weights":      os.path.join(run_dir, "weights", "best.pt"),
        "last_weights":      os.path.join(run_dir, "weights", "last.pt"),
        "final_metrics":     metrics_dict,
    }

    summary_path = os.path.join(OUTPUT_DIR, f"training_summary_{model_tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to : {summary_path}")
    print(f"Best weights at  : {summary['best_weights']}")

    # ── Print key metrics ─────────────────────────────────────────────────────
    print("\n── Final validation metrics ──────────────────────────")
    for k, v in metrics_dict.items():
        if any(x in k for x in ["map", "precision", "recall", "loss"]):
            print(f"  {k:40s}: {v:.4f}")

    print(f"\nDone! Run 03_evaluate.py next.")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO shoe counting model")
    p.add_argument("--model",    default="yolov8n-seg.pt",
                   help="Pretrained model to start from (downloaded automatically)")
    p.add_argument("--epochs",   type=int, default=100,
                   help="Max training epochs (default: 100)")
    p.add_argument("--batch",    type=int, default=16,
                   help="Batch size (reduce if OOM, e.g. 8)")
    p.add_argument("--imgsz",    type=int, default=432,
                   help="Input image size (default: 432, matches dataset)")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience in epochs (default: 20)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
