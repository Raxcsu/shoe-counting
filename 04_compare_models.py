"""
Script 04 - Compare Models
===========================
Reads all metrics_*.json files from outputs/ and generates a side-by-side
comparison table and chart. Run after evaluating multiple models.

Usage:
    python 04_compare_models.py

To generate data to compare, train and evaluate two (or more) models:
    python 02_train.py --model yolov8n-seg.pt
    python 03_evaluate.py
    python 02_train.py --model yolov8s-seg.pt
    python 03_evaluate.py
    python 04_compare_models.py

Outputs (saved to outputs/):
    - comparison_table.csv  : all models side by side
    - comparison_plots.png  : bar charts for key metrics
    - comparison_report.txt : human-readable ranking
"""

import os
import sys
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_metrics():
    pattern = os.path.join(OUTPUT_DIR, "metrics_*.json")
    files   = sorted(glob.glob(pattern))
    # Exclude files that end with _report.txt (safety)
    files   = [f for f in files if f.endswith(".json")]

    if not files:
        print("No metrics_*.json files found in outputs/.")
        print("Run 03_evaluate.py first (for each model you want to compare).")
        sys.exit(1)

    records = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        det  = data.get("detection", {})
        cnt  = data.get("counting", {})
        row  = {
            "model":              data.get("model_tag", os.path.basename(fp)),
            "mAP@50":             det.get("map50",          float("nan")),
            "mAP@50-95":          det.get("map50_95",       float("nan")),
            "Precision":          det.get("precision",      float("nan")),
            "Recall":             det.get("recall",         float("nan")),
            "F1":                 det.get("f1",             float("nan")),
            "MAE":                cnt.get("mae",            float("nan")),
            "RMSE":               cnt.get("rmse",           float("nan")),
            "MAPE":               cnt.get("mape",           float("nan")),
            "Exact_match_%":      cnt.get("exact_match_rate",    float("nan")),
            "Within1_%":          cnt.get("within_1_count_rate", float("nan")),
            "Bias":               cnt.get("mean_error_bias",     float("nan")),
        }
        records.append(row)

    return pd.DataFrame(records)


def rank_models(df):
    """
    Assign a composite score per model.
    Higher-is-better metrics are normalised to [0,1] positively,
    lower-is-better ones negatively.
    """
    # Columns where higher = better
    higher_better = ["mAP@50", "mAP@50-95", "Precision", "Recall", "F1",
                     "Exact_match_%", "Within1_%"]
    # Columns where lower = better
    lower_better  = ["MAE", "RMSE", "MAPE"]

    score = pd.Series(0.0, index=df.index)
    for col in higher_better:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                score += (df[col] - col_min) / (col_max - col_min)
    for col in lower_better:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                score += 1 - (df[col] - col_min) / (col_max - col_min)

    df = df.copy()
    df["composite_score"] = score.round(4)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # rank starts at 1
    return df


def make_comparison_plots(df, out_path):
    metrics = [
        ("mAP@50",       True,  "Detection: mAP@50"),
        ("Precision",    True,  "Detection: Precision"),
        ("Recall",       True,  "Detection: Recall"),
        ("MAE",          False, "Counting: MAE (↓ better)"),
        ("Exact_match_%",True,  "Counting: Exact Match %"),
        ("Within1_%",    True,  "Counting: Within ±1 %"),
    ]

    n = len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    models = df["model"].tolist()
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(models)))

    for ax, (col, higher_better, title) in zip(axes, metrics):
        vals = df[col].values
        bars = ax.bar(models, vals, color=colors, edgecolor="black")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(col)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)

        # Annotate bars
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        # Highlight best
        if not any(np.isnan(vals)):
            best_idx = np.argmax(vals) if higher_better else np.argmin(vals)
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Comparison plots saved to: {out_path}")


def write_comparison_report(df, out_path):
    lines = [
        "=" * 70,
        "  MODEL COMPARISON REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "Rank | Model             | mAP50 | MAE   | Exact% | Within1% | Score",
        "─" * 70,
    ]
    for rank, row in df.iterrows():
        lines.append(
            f"  {rank:<3} | {row['model']:<17} | "
            f"{row['mAP@50']:5.3f} | "
            f"{row['MAE']:5.3f} | "
            f"{row['Exact_match_%']:6.2%} | "
            f"{row['Within1_%']:8.2%} | "
            f"{row['composite_score']:.4f}"
        )
    lines += [
        "",
        "─" * 70,
        "Composite score: higher = better overall.",
        "Gold border in plots = best value for that metric.",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Comparison report saved to: {out_path}")
    print("\n" + "\n".join(lines))


def main():
    print("=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    df = load_metrics()
    print(f"\nFound {len(df)} model(s): {df['model'].tolist()}")

    df_ranked = rank_models(df)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "comparison_table.csv")
    df_ranked.to_csv(csv_path, index_label="rank")
    print(f"\nComparison table saved to: {csv_path}")

    # Plots
    plot_path = os.path.join(OUTPUT_DIR, "comparison_plots.png")
    make_comparison_plots(df_ranked, plot_path)

    # Report
    report_path = os.path.join(OUTPUT_DIR, "comparison_report.txt")
    write_comparison_report(df_ranked, report_path)


if __name__ == "__main__":
    main()
