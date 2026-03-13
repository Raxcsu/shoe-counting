# Shoe Counting — YOLO Object Detection

An ML pipeline to **detect and count shoes in images** using YOLOv8 segmentation.
The model is trained on the LEN shoe_detection dataset (1150 images, MIT license).

---

## Project structure

```
shoe_counting/
├── 01_explore_data.py      # Dataset stats and sample visualisation
├── 02_train.py             # Train the YOLO model
├── 03_evaluate.py          # Evaluate with detection + counting metrics
├── 04_compare_models.py    # Compare multiple trained models side by side
├── 05_predict.py           # Run inference on new images
├── requirements.txt
└── outputs/                # Auto-created — results, weights, plots
    ├── data_fixed.yaml
    ├── data_stats.json
    ├── runs/<model>/weights/best.pt
    ├── training_summary_<model>.json
    ├── metrics_<model>.json / .csv
    └── predictions/
```

> **Not in the repo:** `data/train`, `data/valid`, `data/test` (large dataset splits), `venv/`, model weights `*.pt`
> `data/predict/` (sample images) **is** included in the repo.

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/shoe_counting.git
cd shoe_counting
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux / macOS)
source venv/bin/activate
```

### 2. Install dependencies

**CPU only:**
```bash
pip install -r requirements.txt
```

**With GPU (CUDA) — recommended:**
```bash
pip install -r requirements.txt

# Then replace torch with the CUDA build. Check your version with: nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```

Verify CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# True  ← GPU ready
```

### 3. Add the dataset

Download the dataset from [Roboflow](https://universe.roboflow.com/research-juuyr/shoe_detection-wbsnz/dataset/2) in **YOLOv11 format** and place it so the structure looks like:

```
shoe_counting/
└── data/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

---

## Usage

### Step 1 — Explore the dataset
```bash
python 01_explore_data.py
```
Outputs `outputs/data_stats.json`, `outputs/sample_annotations.png`, `outputs/count_distribution.png`.

---

### Step 2 — Train the model
```bash
python 02_train.py
```

Options:
```
--model     Pretrained base model (default: yolov8n-seg.pt)
--epochs    Max training epochs   (default: 100)
--batch     Batch size            (default: 16, reduce to 8 if out of VRAM)
--imgsz     Input image size      (default: 432)
--patience  Early stopping        (default: 20 epochs without improvement)
```

Examples:
```bash
python 02_train.py --model yolov8n-seg.pt --epochs 100 --batch 16   # fast
python 02_train.py --model yolov8s-seg.pt --epochs 100 --batch 8    # more accurate
python 02_train.py --model yolo11n-seg.pt --epochs 150 --batch 16   # newest arch
```

Weights are saved to `outputs/runs/<model>/weights/best.pt`.
Training time is recorded in `outputs/training_summary_<model>.json`.

---

### Step 3 — Evaluate
```bash
python 03_evaluate.py
```

Options:
```
--weights   Path to best.pt (auto-detected if omitted)
--conf      Confidence threshold (default: 0.25)
--iou       NMS IoU threshold   (default: 0.45)
```

Produces:
- `outputs/metrics_<model>.json` — all metrics
- `outputs/metrics_<model>.csv` — per-image true vs predicted count
- `outputs/metrics_<model>_plots.png` — visualisation charts
- `outputs/metrics_<model>_report.txt` — human-readable summary

---

### Step 4 — Compare models *(optional)*
Train and evaluate a second model, then:
```bash
python 04_compare_models.py
```
Produces `outputs/comparison_table.csv` and `outputs/comparison_plots.png`.

---

### Step 5 — Predict on new images

Sample images are included in `data/predict/`: `len3.jpg`, `len43.jpg`, `len48.jpg`.

```bash
# Default — runs on data/predict/len3.jpg
python 05_predict.py

# Specific sample image
python 05_predict.py --source data/predict/len43.jpg

# All sample images at once
python 05_predict.py --source data/predict/

# Custom image
python 05_predict.py --source path/to/your/image.jpg
```

Options:
```
--source    Image file or folder (default: data/predict/len3.jpg)
--weights   Path to best.pt (auto-detected if omitted)
--conf      Confidence threshold (default: 0.25)
--iou       NMS IoU threshold   (default: 0.45)
```

Each image is saved to `outputs/predictions/<name>_comparison.png` as a side-by-side of **original | prediction** with segmentation masks, bounding boxes, and confidence scores drawn by the model.

---

## Metrics explained

| Metric | Type | Better when |
|---|---|---|
| mAP@50 | Detection | Higher |
| mAP@50-95 | Detection | Higher |
| Precision | Detection | Higher |
| Recall | Detection | Higher |
| F1 | Detection | Higher |
| **MAE** | Counting | **Lower** — avg error per image |
| **RMSE** | Counting | **Lower** — penalises big errors |
| **Exact match %** | Counting | **Higher** — % images with perfect count |
| Within ±1 % | Counting | Higher |
| Bias | Counting | Near 0 — +positive = over-counting |

---

## Model options

| Model | Params | Notes |
|---|---|---|
| `yolov8n-seg.pt` | ~3.4 M | Nano — fastest, good baseline |
| `yolov8s-seg.pt` | ~11.8 M | Small — good balance |
| `yolov8m-seg.pt` | ~27 M | Medium — most accurate |
| `yolo11n-seg.pt` | ~2.9 M | Newest architecture |

---

## Requirements

- Python 3.9+
- See `requirements.txt`
- GPU strongly recommended (CUDA 11.8+ or 12.x)
