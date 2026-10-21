# YOLOv1 from Scratch — PyTorch (PASCAL VOC)

A faithful, from-scratch PyTorch implementation of **YOLOv1** ("You Only Look Once:
Unified, Real-Time Object Detection", Redmon et al., 2016) trained on **PASCAL VOC
2007 + 2012** and evaluated with the VOC 2007 **11-point mAP@0.5** protocol.

Every hyperparameter lives in a single YAML file (`configs/voc.yaml`) — nothing is
hard-coded in the modules. The whole pipeline is importable as `src.*`, so it works
identically from a notebook (Google Colab), a shell, or a test runner.

This README is written to get you from a **blank Colab notebook to a trained model,
evaluation report, and interactive demo** with copy-paste cells.

---

## Table of contents

- [What's implemented](#whats-implemented)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Google Colab: step-by-step](#google-colab-step-by-step)
  - [0. GPU check](#0-gpu-check)
  - [1. Get the code](#1-get-the-code)
  - [2. Install dependencies](#2-install-dependencies)
  - [3. (Recommended) mount Google Drive](#3-recommended-mount-google-drive)
  - [4. Smoke test the pipeline](#4-smoke-test-the-pipeline)
  - [5. Overfit sanity gate](#5-overfit-sanity-gate)
  - [6. Train](#6-train)
  - [7. Evaluate mAP@0.5](#7-evaluate-map05)
  - [8. Single-image inference](#8-single-image-inference)
  - [9. Interactive Gradio demo](#9-interactive-gradio-demo)
  - [10. Download artifacts](#10-download-artifacts)
- [Running locally (non-Colab)](#running-locally-non-colab)
- [Configuration reference](#configuration-reference)
- [How it works (pipeline internals)](#how-it-works-pipeline-internals)
- [Tests](#tests)
- [Colab tips, limits & troubleshooting](#colab-tips-limits--troubleshooting)

---

## What's implemented

- **Two model variants** (`src/model.py`)
  - `full` — the paper's 24-conv GoogLeNet-inspired trunk with 1×1 reduction layers.
  - `fast` — a 9-conv Fast-YOLO-style trunk for quick CPU/Colab iteration.
  - Optional `pretrained: true` reuses a torchvision GoogLeNet conv trunk.
- **Paper-faithful loss** (`src/loss.py`) — sum-squared error with
  `λ_coord=5`, `λ_noobj=0.5`, `sqrt(w)`, `sqrt(h)`, and coordinate loss applied
  **only to the responsible box** (the higher-IoU predictor per cell).
- **Data pipeline** (`src/dataset.py`) — wraps `torchvision.datasets.VOCDetection`,
  supports direct resize (paper default) or letterbox, applies the paper's
  augmentation (random scale/translate ≤ 20%, HSV exposure/saturation up to 1.5×),
  and encodes labels into the `7×7×30` target tensor.
- **LR schedule** (`src/config.py`) — the paper's piecewise ramp+decay
  (warmup → `1e-2` → `1e-3` → `1e-4`), plus a `reduced` schedule for fast runs.
- **mAP@0.5 evaluation** (`src/metrics.py`) — VOC2007 11-point interpolation,
  hard-example/difficult handling, JSON report + per-class AP bar chart.
- **Inference** (`src/predict.py`) — decode `7×7×30` → boxes, per-class NMS, draw.
- **Typed config** (`src/config.py`) — dataclasses populated from YAML with defaults.
- **Tests** (`tests/`) — target encoder, geometry/NMS/decode round-trips, loss math.

> **Note:** `requirements.txt` lists `gradio` and `opencv-python` for a demo, but no
> demo app ships in the repo yet. [Step 9](#9-interactive-gradio-demo) gives you a
> ready-to-run Gradio cell you can paste into Colab.

---

## Repository layout

```
cv_project_yolo/
├── configs/
│   └── voc.yaml            # ALL hyperparameters (single source of truth)
├── src/
│   ├── __init__.py
│   ├── config.py           # dataclasses + YAML loader + LR schedule + VOC classes
│   ├── dataset.py          # VOCDetection wrapper, augment, letterbox, dataloaders
│   ├── model.py            # YOLOv1 full/fast/pretrained trunks + FC head
│   ├── loss.py             # YOLOv1 sum-squared-error loss
│   ├── metrics.py          # VOC2007 mAP@0.5 evaluation + report
│   ├── utils.py            # box geometry, NMS, target encode, prediction decode
│   ├── train.py            # training entry point (CLI)
│   └── predict.py          # single-image inference (CLI)
├── tests/
│   ├── test_utils.py
│   ├── test_loss.py
│   └── test_target_encoder.py
├── conftest.py             # makes the project root importable under pytest
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+
- A CUDA GPU is strongly recommended for the `full` variant (Colab's free T4 is fine).
- Core: `torch>=2.0`, `torchvision>=0.15`, `numpy`, `Pillow`, `PyYAML`
- Training/eval: `tqdm`, `tensorboard`, `matplotlib`
- Demo: `gradio`, `opencv-python`
- Tests: `pytest`

Colab already ships a CUDA build of PyTorch, so **do not blindly reinstall `torch`** —
it is intentionally left out of the install command below to avoid clobbering Colab's
GPU build.

---

## Google Colab: step-by-step

Runtime → **Change runtime type → Hardware accelerator → GPU (T4)** before you start.

### 0. GPU check

```python
!nvidia-smi
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
```

If `cuda available` is `False`, fix the runtime type first — training the `full`
model on CPU is impractically slow.

### 1. Get the code

**Option A — clone from Git (recommended once you push this repo):**

```python
# Replace with your repository URL
REPO_URL = "https://github.com/<you>/cv_project_yolo.git"
!git clone {REPO_URL} /content/cv_project_yolo
%cd /content/cv_project_yolo
```

**Option B — upload a zip** (for a local-only copy):

```python
# In the Colab file browser (folder icon on the left), upload cv_project_yolo.zip, then:
!unzip -q /content/cv_project_yolo.zip -d /content/
%cd /content/cv_project_yolo
```

**Option C — work from Google Drive** (survives runtime resets; see step 3):

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/cv_project_yolo   # after uploading the project there
```

Confirm you are in the right place and the package imports:

```python
!ls
import sys; sys.path.insert(0, "/content/cv_project_yolo")  # harmless if already cwd
from src.config import load_config, VOC_CLASSES
print(len(VOC_CLASSES), "classes")
```

### 2. Install dependencies

Colab has `torch`/`torchvision` already. Install the rest:

```python
# From the repo root (skip torch/torchvision so we keep Colab's CUDA build):
!pip -q install numpy Pillow PyYAML tqdm tensorboard matplotlib gradio opencv-python pytest
```

Verify:

```python
import yaml, torchvision, gradio, cv2
print("deps OK", torchvision.__version__, gradio.__version__)
```

### 3. (Recommended) mount Google Drive

Colab wipes `/content` when the runtime disconnects. Persist checkpoints, logs, and
data by pointing the config at Drive — this lets you **resume** after a disconnect.

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/yolov1/checkpoints', exist_ok=True)
os.makedirs('/content/drive/MyDrive/yolov1/runs', exist_ok=True)
os.makedirs('/content/drive/MyDrive/yolov1/data', exist_ok=True)
```

Then either edit `configs/voc.yaml` at runtime, or override from Python when you
load the config (both shown in [Configuration reference](#configuration-reference)).

> Keeping the VOC dataset on Drive saves re-downloading ~2 GB every session. If Drive
> I/O feels slow, keep data on `/content` and only put checkpoints on Drive.

### 4. Smoke test the pipeline

Download VOC, build the dataloaders, and push one batch through the model — do this
before any long run. The first call downloads VOC2007 + VOC2012 (~2 GB) into
`data/VOCdevkit/`.

```python
import torch
from src.config import load_config
from src.dataset import build_dataloaders
from src.model import build_model
from src.loss import build_loss

cfg = load_config("configs/voc.yaml")
# Fast smoke settings: small batch, fewer workers, no augmentation weirdness
cfg.train.batch_size = 4
cfg.data.num_workers = 2
cfg.model.variant = "fast"        # cheap; switch back to "full" for real training
cfg.model.hidden_dim = 1024

train_loader, val_loader = build_dataloaders(cfg, download=True)
images, targets = next(iter(train_loader))
print("image batch:", images.shape)      # (4, 3, 448, 448)
print("target batch:", targets.shape)    # (4, 7, 7, 30)

device = torch.device("cuda")
model = build_model(cfg).to(device)
criterion = build_loss(cfg)
with torch.no_grad():
    out = model(images.to(device))
print("model output:", out.shape)        # (4, 7, 7, 30)
print("loss:", criterion(out, targets.to(device))["loss"].item())
```

If shapes match and a loss number prints, the pipeline is wired correctly.

### 5. Overfit sanity gate

A correct implementation must be able to drive the loss toward ~0 on a handful of
samples. Run this before spending GPU time on the full schedule:

```python
!python -m src.train --config configs/voc.yaml \
  --overfit-batch 8 --variant fast --epochs 150 --batch-size 8 --no-amp
```

Watch the printed `train=` loss — it should fall steadily toward ~0 (roughly `< 0.05`).
If it plateaus high, something is wrong in the target encoding or loss; run the tests
first (`pytest -q`). This uses the `fast` trunk, so it's quick.

### 6. Train

**Fast run (Colab-friendly, gets a working model quickly):**

```python
!python -m src.train --config configs/voc.yaml \
  --variant fast --epochs 30 --batch-size 16 --schedule reduced
```

**Full paper variant** (24-conv trunk, paper's 135-epoch schedule). This is the real
thing and needs a GPU for hours; use Drive checkpoints and expect to resume across
sessions:

```python
!python -m src.train --config configs/voc.yaml \
  --variant full --epochs 135 --batch-size 16 --schedule paper
```

**Resume after a disconnect:**

```python
!python -m src.train --config configs/voc.yaml \
  --resume /content/drive/MyDrive/yolov1/checkpoints/last.pt
```

Checkpoints `last.pt` and `best.pt` are written to `train.checkpoint_dir`; TensorBoard
scalars go to `train.log_dir`.

### 7. Evaluate mAP@0.5

Runs the VOC2007 **test** set with 11-point interpolation and writes
`assets/figures/voc_eval.json` + `per_class_ap.png`.

```python
!python -m src.metrics --config configs/voc.yaml \
  --checkpoint checkpoints/best.pt --output-dir assets/figures
```

Quick debug on a subset first (fast feedback):

```python
!python -m src.metrics --config configs/voc.yaml \
  --checkpoint checkpoints/best.pt --limit 200
```

Reference point: the original paper reports ~63.4% mAP on VOC2007 for the full model
with 7×7 grid. A from-scratch Colab run (fewer epochs / smaller effective batch) will
land lower — treat that as expected, not a bug.

### 8. Single-image inference

The CLI takes one image and writes an annotated copy:

```python
!mkdir -p /content/samples
# Grab a VOC test image (the dataset already downloaded in step 4):
!cp data/VOCdevkit/VOC2007/JPEGImages/000001.jpg /content/samples/ 2>/dev/null || true

!python -m src.predict --config configs/voc.yaml \
  --checkpoint checkpoints/best.pt \
  --image /content/samples/000001.jpg \
  --output /content/samples/000001_pred.jpg \
  --conf 0.25 --nms 0.5
```

Display the result in the notebook:

```python
from PIL import Image
from IPython.display import display
display(Image.open("/content/samples/000001_pred.jpg"))
```

Or call the API directly (no CLI, no temp files):

```python
import torch
from PIL import Image
from src.config import load_config
from src.predict import load_model, detect, draw_detections

cfg = load_config("configs/voc.yaml")
device = torch.device("cuda")
model = load_model(cfg, "checkpoints/best.pt", device)

img = Image.open("/content/samples/000001.jpg").convert("RGB")
dets = detect(model, img, cfg, device, conf_threshold=0.25, nms_iou=0.5)
for d in dets:
    print(d["name"], round(d["score"], 3), [round(v) for v in d["box"]])
display(draw_detections(img, dets))
```

### 9. Interactive Gradio demo

No demo file ships in the repo, but this cell gives you a working web demo. In Colab
it prints a public `*.gradio.live` link (valid ~72h) — open it from your browser.

```python
import torch, gradio as gr
from PIL import Image
from src.config import load_config
from src.predict import load_model, detect, draw_detections

cfg = load_config("configs/voc.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(cfg, "checkpoints/best.pt", device)

def predict(image, conf, nms):
    if image is None:
        return None, ""
    img = image.convert("RGB")
    dets = detect(model, img, cfg, device, conf_threshold=conf, nms_iou=nms)
    lines = [f"{d['name']:<12} {d['score']:.3f}  {[round(v) for v in d['box']]}" for d in dets]
    return draw_detections(img, dets), "\n".join(lines) or "no detections"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Input image"),
        gr.Slider(0.05, 0.9, value=0.25, label="Confidence threshold"),
        gr.Slider(0.1, 0.9, value=0.5, label="NMS IoU"),
    ],
    outputs=[gr.Image(type="pil", label="Detections"), gr.Textbox(label="Boxes")],
    title="YOLOv1 (from scratch) — PASCAL VOC",
    description="20 VOC classes. Upload an image and tune the thresholds.",
)
demo.launch(share=True, debug=False)
```

Stop the demo by interrupting the cell (■) so it releases the port.

### 10. Download artifacts

```python
# Copy the model + reports to Drive
!mkdir -p /content/drive/MyDrive/yolov1
!cp checkpoints/best.pt /content/drive/MyDrive/yolov1/ 2>/dev/null || true
!cp -r assets/figures /content/drive/MyDrive/yolov1/ 2>/dev/null || true

# Or download a zip directly to your laptop
!zip -qr yolov1_artifacts.zip checkpoints assets/figures
from google.colab import files
files.download("yolov1_artifacts.zip")
```

---

## Running locally (non-Colab)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install a CUDA or CPU build of torch first, per https://pytorch.org/get-started
pip install -r requirements.txt

pytest -q
python -m src.train   --config configs/voc.yaml --variant fast --epochs 30
python -m src.metrics --config configs/voc.yaml --checkpoint checkpoints/best.pt
python -m src.predict --config configs/voc.yaml --checkpoint checkpoints/best.pt --image some.jpg
```

---

## Configuration reference

Everything is driven by `configs/voc.yaml`. Unknown keys are ignored and missing keys
fall back to the dataclass defaults in `src/config.py`.

| Section | Key | Default | Meaning |
|---|---|---|---|
| `data` | `root` | `data` | Where `VOCdevkit/` is created |
| | `year` / `image_set` | `2007` / `trainval` | Primary training split |
| | `extra_year` / `extra_image_set` | `2012` / `trainval` | Extra split; set `null` for VOC2007 only |
| | `image_size` | `448` | Network input resolution |
| | `letterbox` | `false` | `false` = paper's direct resize; `true` = keep aspect + pad |
| | `num_workers` | `4` | DataLoader workers (use `2` on Colab) |
| `model` | `S` / `B` / `C` | `7` / `2` / `20` | Grid, boxes per cell, classes |
| | `variant` | `full` | `full` (24 conv) or `fast` (9 conv) |
| | `pretrained` | `false` | Reuse torchvision GoogLeNet conv trunk |
| | `dropout` | `0.5` | Dropout after the first FC |
| | `hidden_dim` | `4096` | FC hidden size (lower to save memory) |
| | `batchnorm` | `true` | `false` reproduces the paper's dropout-only stack |
| `loss` | `lambda_coord` | `5.0` | Coordinate-loss weight |
| | `lambda_noobj` | `0.5` | No-object confidence weight |
| `train` | `epochs` | `135` | Total epochs |
| | `schedule` | `paper` | `paper` (135) or `reduced` (30) LR schedule |
| | `batch_size` | `16` | Lower for CPU/small GPUs |
| | `lr` / `momentum` / `weight_decay` | `1e-2` / `0.9` / `5e-4` | SGD settings |
| | `amp` | `true` | Mixed precision (CUDA only) |
| | `grad_clip` | `10.0` | Gradient-norm clip |
| | `eval_every` | `5` | Validation cadence |
| | `checkpoint_dir` / `log_dir` | `checkpoints` / `runs` | Output locations |
| | `resume` | `null` | Checkpoint path to resume from |
| | `seed` | `42` | RNG seed |
| | `device` | `auto` | `auto` / `cpu` / `cuda` |
| `augment` | `enabled` | `true` | Toggle augmentation |
| | `scale` / `translate` | `0.2` | ± fraction |
| | `hsv` | `0.5` | ± exposure/saturation (→ up to 1.5×) |
| | `hflip` | `false` | Horizontal flip (off: left/right class semantics) |
| `inference` | `conf_threshold` | `0.25` | Detection score cutoff |
| | `nms_iou` | `0.5` | NMS IoU threshold |

**Edit the YAML in Colab:**

```python
# Point data/checkpoints at Drive, tweak batch size, etc.
from pathlib import Path
p = Path("configs/voc.yaml")
p.write_text(p.read_text()
    .replace("root: data", "root: /content/drive/MyDrive/yolov1/data")
    .replace("checkpoint_dir: checkpoints", "checkpoint_dir: /content/drive/MyDrive/yolov1/checkpoints")
    .replace("log_dir: runs", "log_dir: /content/drive/MyDrive/yolov1/runs")
)
```

**Or override in Python** (handy inside notebooks — no file edit):

```python
cfg = load_config("configs/voc.yaml")
cfg.data.root = "/content/drive/MyDrive/yolov1/data"
cfg.train.batch_size = 16
cfg.model.hidden_dim = 2048   # if you hit CUDA OOM on the full variant
```

CLI flags (`--variant`, `--epochs`, `--batch-size`, `--lr`, `--schedule`,
`--pretrained`, `--no-amp`, `--resume`, `--device`, `--seed`) override the YAML for
`src.train`. `src.predict` / `src.metrics` accept `--conf`, `--nms`, `--limit`.

---

## How it works (pipeline internals)

**Target encoding** (`src.utils.encode_targets`). Each object is assigned to the grid
cell containing its centre. The cell stores `(x, y)` **relative to the cell** (`[0,1]`)
and `(w, h)` as a **fraction of the whole image**, in box slot 0, plus a one-hot class
vector. If two objects share a cell (rare at S=7), the larger box wins.

**Responsibility** (`src.loss.YOLOv1Loss`). For every object cell, the predictor with
the **highest IoU** against the ground truth is "responsible" and receives the
coordinate loss and a confidence target of 1; the other predictor's confidence is
pushed to 0. Confidence loss is applied to all boxes, weighted by `λ_noobj` in empty
cells.

**Prediction decoding** (`src.utils.decode_predictions`). The raw `7×7×30` output is
converted to image-space `xyxy`: cell-relative offsets plus the cell index give the
centre; `w`, `h` are used directly. `score = confidence × class_probability`, then
**per-class greedy NMS** removes overlaps.

**LR schedule** (`src.config.build_lr_schedule`). Epoch 0 ramps linearly from
`0.1·lr` to `lr`; then the base LR is held, then dropped ×10 at each schedule
boundary (paper: 75/30/30 epochs → `1e-2`, `1e-3`, `1e-4`).

---

## Tests

```bash
pytest -q
```

Covers: xywh/xyxy round-trips, IoU, NMS, encode→decode round-trip, confidence
thresholding (`tests/test_utils.py`); perfect-prediction zero loss, weighting of
`λ_coord`/`λ_noobj`, and "coord loss only on the responsible box" (`tests/test_loss.py`);
centre-cell assignment, cell-relative offsets, empty cells, and the
larger-box-wins conflict rule (`tests/test_target_encoder.py`).

Run them right after cloning — a green suite means geometry and loss math are sane
before you burn GPU time.

---

## Colab tips, limits & troubleshooting

- **Runtime type.** Always GPU (T4). Verify with `torch.cuda.is_available()`.
- **Session limits.** Free Colab disconnects after idle periods and caps runtimes;
  the `full`/135-epoch run **will not finish in one session**. Use Drive checkpoints and
  `--resume`. Background-mounted Drive + frequent checkpoints are your safety net.
- **Re-downloading data every session is wasteful.** Keep VOC on Drive (config
  `data.root`) or accept a ~2 GB re-download.
- **CUDA OOM.** Reduce `train.batch_size`, set `model.hidden_dim` to `2048` or `1024`,
  use `--variant fast`, or disable `amp`. The bigger FC (`4096`) is the usual culprit
  at 448×448.
- **`num_workers` on Colab.** Use `2`; very high values can hang the runtime.
- **Do not reinstall torch.** It's preinstalled with CUDA; a plain `pip install torch`
  can downgrade to a CPU wheel and silently kill GPU training.
- **`file_list` / download errors.** The first `download=True` fetches VOC2007 and
  VOC2012; a flaky network can corrupt a partial download — delete
  `data/VOCdevkit/` and retry.
- **Low mAP vs the paper.** Expected unless you run the full 135-epoch schedule,
  `full` variant, augmentation on, and the paper's batch size. The `fast`/`reduced`
  path is for iteration, not for matching published numbers.
- **Gradio link.** `demo.launch(share=True)` prints a temporary public URL; stop the
  cell to free it. Uploaded images stay within the session.
- **TensorBoard in Colab.** `%load_ext tensorboard` then
  `%tensorboard --logdir runs` (or the Drive log dir).

---

## Attribution

Implementation follows the YOLOv1 paper (Redmon et al., 2016). Dataset:
PASCAL VOC 2007/2012, loaded via `torchvision.datasets.VOCDetection`. This project is
an educational from-scratch reproduction and is not affiliated with the original authors.
