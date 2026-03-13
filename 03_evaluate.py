"""
Script 03 - Model Evaluation
==============================
Evaluates a trained model on the test split.
Produces both detection metrics and counting-specific metrics.

Usage:
    python 03_evaluate.py                                   # auto-finds best model
    python 03_evaluate.py --weights outputs/runs/yolov8n-seg/weights/best.pt
    python 03_evaluate.py --weights outputs/runs/yolov8s-seg/weights/best.pt --conf 0.4

Outputs (saved to outputs/):
    - metrics_<model>.json        : all metrics in machine-readable format
    - metrics_<model>.csv         : per-image results (true count vs predicted count)
    - metrics_<model>_report.txt  : human-readable summary report
    - metrics_<model>_plots.png   : visualisation of counting errors
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from datetime import datetime
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_LABEL_DIR = os.path.join(DATA_DIR, "test", "labels")


# ── Helpers ───────────────────────────────────────────────────────────────────
def read_true_count(label_path):
    """Count the number of annotated objects in a YOLO label file."""
    if not os.path.exists(label_path):
        return 0
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    return len(lines)


def auto_find_weights():
    """Find the most recently trained best.pt under outputs/runs/."""
    pattern = os.path.join(OUTPUT_DIR, "runs", "*", "weights", "best.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # Pick most recently modified
    return max(candidates, key=os.path.getmtime)


def model_tag_from_path(weights_path):
    """Extract model tag from weights path, e.g. 'yolov8n-seg'."""
    parts = weights_path.replace("\\", "/").split("/")
    # structure: .../runs/<model_tag>/weights/best.pt
    try:
        idx = parts.index("runs")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "model"


# ── YOLO built-in validation ──────────────────────────────────────────────────
def run_yolo_val(model, data_yaml, conf, iou):
    """Run YOLO's built-in validation to get mAP and other detection metrics."""
    print("\nRunning YOLO validation (mAP, Precision, Recall)...")
    val_results = model.val(
        data   = data_yaml,
        split  = "test",
        conf   = conf,
        iou    = iou,
        verbose= False,
    )
    metrics = {}
    try:
        rd = val_results.results_dict
        metrics = {k: float(v) for k, v in rd.items()}
    except Exception:
        pass
    # Also pull from seg/box attributes directly
    try:
        metrics["precision"]   = float(val_results.seg.p.mean())
        metrics["recall"]      = float(val_results.seg.r.mean())
        metrics["map50"]       = float(val_results.seg.map50)
        metrics["map50_95"]    = float(val_results.seg.map)
        metrics["f1"]          = 2 * metrics["precision"] * metrics["recall"] / \
                                  max(metrics["precision"] + metrics["recall"], 1e-9)
    except Exception:
        try:
            metrics["precision"]   = float(val_results.box.p.mean())
            metrics["recall"]      = float(val_results.box.r.mean())
            metrics["map50"]       = float(val_results.box.map50)
            metrics["map50_95"]    = float(val_results.box.map)
            metrics["f1"]          = 2 * metrics["precision"] * metrics["recall"] / \
                                      max(metrics["precision"] + metrics["recall"], 1e-9)
        except Exception:
            pass
    return metrics


# ── Per-image counting evaluation ─────────────────────────────────────────────
def run_counting_eval(model, conf, iou):
    """
    Run inference on every test image and compare predicted count to ground truth.
    Returns a DataFrame with one row per image.
    """
    print("\nRunning per-image counting evaluation...")
    image_paths = sorted(
        glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpg")) +
        glob.glob(os.path.join(TEST_IMAGE_DIR, "*.png"))
    )

    rows = []
    for img_path in image_paths:
        base       = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(TEST_LABEL_DIR, base + ".txt")
        true_count = read_true_count(label_path)

        # Run inference (verbose=False to suppress per-image prints)
        preds = model.predict(
            source  = img_path,
            conf    = conf,
            iou     = iou,
            verbose = False,
        )
        pred_count = len(preds[0].boxes) if preds and preds[0].boxes is not None else 0

        rows.append({
            "image":      os.path.basename(img_path),
            "true_count": true_count,
            "pred_count": pred_count,
            "error":      pred_count - true_count,
            "abs_error":  abs(pred_count - true_count),
        })

    df = pd.DataFrame(rows)
    return df


# ── Counting metrics ──────────────────────────────────────────────────────────
def compute_counting_metrics(df):
    errors     = df["error"].values
    abs_errors = df["abs_error"].values
    true_counts = df["true_count"].values
    pred_counts = df["pred_count"].values

    mae     = float(np.mean(abs_errors))
    rmse    = float(np.sqrt(np.mean(errors ** 2)))
    exact   = float(np.mean(abs_errors == 0))                # exact match rate
    within1 = float(np.mean(abs_errors <= 1))                # within ±1

    # Mean Absolute Percentage Error (skip images with 0 true shoes)
    nonzero = true_counts > 0
    mape = float(np.mean(abs_errors[nonzero] / true_counts[nonzero])) if nonzero.any() else 0.0

    # Over/under counting bias
    mean_error = float(np.mean(errors))   # positive → over-counting

    return {
        "mae":                    round(mae, 4),
        "rmse":                   round(rmse, 4),
        "mape":                   round(mape, 4),
        "exact_match_rate":       round(exact, 4),
        "within_1_count_rate":    round(within1, 4),
        "mean_error_bias":        round(mean_error, 4),
        "total_true_shoes":       int(true_counts.sum()),
        "total_pred_shoes":       int(pred_counts.sum()),
        "n_images":               len(df),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────
def make_plots(df, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Shoe Counting Evaluation – Test Set", fontsize=14, fontweight="bold")

    true  = df["true_count"].values
    pred  = df["pred_count"].values
    errs  = df["error"].values
    abs_e = df["abs_error"].values

    # 1. Scatter: true vs predicted
    ax = axes[0, 0]
    ax.scatter(true, pred, alpha=0.6, edgecolors="k", linewidths=0.5)
    lim = max(true.max(), pred.max()) + 0.5
    ax.plot([0, lim], [0, lim], "r--", label="Perfect prediction")
    ax.set_xlabel("True count")
    ax.set_ylabel("Predicted count")
    ax.set_title("True vs Predicted Count")
    ax.legend()
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)

    # 2. Error histogram
    ax = axes[0, 1]
    bins = range(int(errs.min()) - 1, int(errs.max()) + 2)
    ax.hist(errs, bins=bins, color="steelblue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Prediction error (pred − true)")
    ax.set_ylabel("Number of images")
    ax.set_title("Counting Error Distribution")

    # 3. Absolute error bar
    ax = axes[0, 2]
    from collections import Counter
    abs_hist = Counter(abs_e.astype(int))
    keys = sorted(abs_hist.keys())
    vals = [abs_hist[k] for k in keys]
    ax.bar([str(k) for k in keys], vals, color="darkorange", edgecolor="black")
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Number of images")
    ax.set_title("Absolute Error Distribution")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)

    # 4. Error by true count (box-like grouped bar)
    ax = axes[1, 0]
    for tc in sorted(df["true_count"].unique()):
        subset = df[df["true_count"] == tc]["error"].values
        ax.scatter([tc] * len(subset), subset, alpha=0.5,
                   label=f"true={tc}", s=30)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("True count")
    ax.set_ylabel("Error")
    ax.set_title("Error by True Count")
    ax.legend(fontsize=8)

    # 5. Cumulative error rate
    ax = axes[1, 1]
    thresholds = range(0, int(abs_e.max()) + 2)
    cum_rates  = [np.mean(abs_e <= t) for t in thresholds]
    ax.plot(list(thresholds), cum_rates, marker="o", color="green")
    ax.set_xlabel("Max allowed error")
    ax.set_ylabel("Fraction of images")
    ax.set_title("Cumulative Accuracy vs Tolerance")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.4)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    cm = compute_counting_metrics(df)
    text = (
        f"Counting Metrics (test set)\n"
        f"{'─'*30}\n"
        f"MAE              : {cm['mae']:.4f}\n"
        f"RMSE             : {cm['rmse']:.4f}\n"
        f"MAPE             : {cm['mape']:.2%}\n"
        f"Exact match rate : {cm['exact_match_rate']:.2%}\n"
        f"Within ±1 rate   : {cm['within_1_count_rate']:.2%}\n"
        f"Mean bias        : {cm['mean_error_bias']:+.4f}\n"
        f"{'─'*30}\n"
        f"Total images     : {cm['n_images']}\n"
        f"True shoes total : {cm['total_true_shoes']}\n"
        f"Pred shoes total : {cm['total_pred_shoes']}\n"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to: {out_path}")


# ── Report ─────────────────────────────────────────────────────────────────────
def write_report(model_tag, det_metrics, count_metrics, weights_path, out_path):
    lines = [
        "=" * 60,
        f"  EVALUATION REPORT – {model_tag}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"Weights : {weights_path}",
        f"Test set: {TEST_IMAGE_DIR}",
        "",
        "── DETECTION / SEGMENTATION METRICS ─────────────────────",
        f"  mAP@50         : {det_metrics.get('map50',  float('nan')):.4f}",
        f"  mAP@50-95      : {det_metrics.get('map50_95', float('nan')):.4f}",
        f"  Precision      : {det_metrics.get('precision', float('nan')):.4f}",
        f"  Recall         : {det_metrics.get('recall',    float('nan')):.4f}",
        f"  F1 Score       : {det_metrics.get('f1',        float('nan')):.4f}",
        "",
        "── COUNTING METRICS ──────────────────────────────────────",
        f"  MAE                    : {count_metrics['mae']:.4f}",
        f"  RMSE                   : {count_metrics['rmse']:.4f}",
        f"  MAPE                   : {count_metrics['mape']:.2%}",
        f"  Exact match rate       : {count_metrics['exact_match_rate']:.2%}",
        f"  Within ±1 count rate   : {count_metrics['within_1_count_rate']:.2%}",
        f"  Mean error (bias)      : {count_metrics['mean_error_bias']:+.4f}",
        "",
        "── TOTALS ────────────────────────────────────────────────",
        f"  Test images            : {count_metrics['n_images']}",
        f"  Ground truth shoes     : {count_metrics['total_true_shoes']}",
        f"  Predicted shoes        : {count_metrics['total_pred_shoes']}",
        "",
        "── METRIC GUIDE ──────────────────────────────────────────",
        "  mAP@50         Higher is better (detection quality).",
        "  MAE            Lower is better (avg counting error).",
        "  RMSE           Lower is better (penalises large errors).",
        "  Exact match    Higher is better (% images with correct count).",
        "  Within ±1      Higher is better (% images off by at most 1).",
        "  Bias           Near 0 is better (+= over-counting, -= under).",
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to : {out_path}")
    # Also print to terminal
    print("\n" + "\n".join(lines[3:]))


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate(args):
    print("=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    # ── Find weights ──────────────────────────────────────────────────────────
    weights = args.weights or auto_find_weights()
    if not weights or not os.path.exists(weights):
        print("ERROR: No weights found. Train a model first with 02_train.py")
        print(f"       Or pass --weights path/to/best.pt")
        sys.exit(1)
    print(f"Weights : {weights}")

    model_tag = model_tag_from_path(weights)
    model     = YOLO(weights)

    # ── Data yaml ─────────────────────────────────────────────────────────────
    data_yaml = os.path.join(OUTPUT_DIR, "data_fixed.yaml")
    if not os.path.exists(data_yaml):
        # Create it if train was skipped / yaml missing
        import yaml as _yaml
        fixed = {
            "train": os.path.join(DATA_DIR, "train", "images").replace("\\", "/"),
            "val":   os.path.join(DATA_DIR, "valid", "images").replace("\\", "/"),
            "test":  os.path.join(DATA_DIR, "test",  "images").replace("\\", "/"),
            "nc":    1,
            "names": ["shoe"],
        }
        with open(data_yaml, "w") as f:
            _yaml.dump(fixed, f, default_flow_style=False)

    # ── YOLO built-in validation metrics ──────────────────────────────────────
    det_metrics = run_yolo_val(model, data_yaml, conf=args.conf, iou=args.iou)

    # ── Per-image counting evaluation ─────────────────────────────────────────
    df = run_counting_eval(model, conf=args.conf, iou=args.iou)

    # ── Counting metrics ──────────────────────────────────────────────────────
    count_metrics = compute_counting_metrics(df)

    # ── Save results ──────────────────────────────────────────────────────────
    all_metrics = {
        "model_tag":       model_tag,
        "weights":         weights,
        "conf_threshold":  args.conf,
        "iou_threshold":   args.iou,
        "evaluated_at":    datetime.now().isoformat(),
        "detection":       det_metrics,
        "counting":        count_metrics,
    }

    json_path = os.path.join(OUTPUT_DIR, f"metrics_{model_tag}.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics JSON : {json_path}")

    csv_path = os.path.join(OUTPUT_DIR, f"metrics_{model_tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Per-image CSV: {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_path = os.path.join(OUTPUT_DIR, f"metrics_{model_tag}_plots.png")
    make_plots(df, plot_path)

    # ── Text report ───────────────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, f"metrics_{model_tag}_report.txt")
    write_report(model_tag, det_metrics, count_metrics, weights, report_path)

    print(f"\nDone! To compare multiple models run 04_compare_models.py")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained shoe counting model")
    p.add_argument("--weights", default=None,
                   help="Path to best.pt (auto-detected if omitted)")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="NMS IoU threshold (default: 0.45)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
