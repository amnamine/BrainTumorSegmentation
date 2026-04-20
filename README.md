# Brain Tumor Segmentation — Deployment & Models

This repository bundles **inference-ready artifacts** and **desktop or web interfaces** for **brain tumor segmentation** on MRI-style images. It supports two complementary approaches trained on the same broad problem domain (binary tumor vs background):

| Track | Framework | Role in this repo |
|--------|-----------|-------------------|
| **U-Net** | TensorFlow / Keras | Pixel-wise mask (`unet_brain_tumor.keras`), Streamlit + Tkinter apps, optional TFLite export |
| **YOLO26-seg** | Ultralytics (PyTorch) | Instance-style segmentation with boxes + masks (`brainseg26n-seg.pt`), Tkinter GUI, training notebook & curves |

The goal of the project is **practical deployment**: load an image, run the model, visualize the tumor region — while keeping the training story and metrics traceable for reporting and comparison.

---

## Table of contents

1. [Project overview](#project-overview)  
2. [Repository layout](#repository-layout)  
3. [Dataset & training context (YOLO path)](#dataset--training-context-yolo-path)  
4. [Models & file formats](#models--file-formats)  
5. [YOLO26-seg vs U-Net — metrics comparison](#yolo26-seg-vs-u-net--metrics-comparison)  
6. [Running the applications](#running-the-applications)  
7. [TFLite conversion (U-Net)](#tflite-conversion-u-net)  
8. [Dependencies](#dependencies)  
9. [Limitations & reproducibility notes](#limitations--reproducibility-notes)  
10. [Credits](#credits)

---

## Project overview

### Problem

**Segmentation of brain tumors** from imaging: given a 2D slice (PNG), predict where the tumor is. Both pipelines treat this as a **single foreground class** (tumor) versus background.

### Why two models?

- **U-Net** outputs a **dense probability map** per pixel (classic medical segmentation). The shipped Keras model uses batch normalization, dropout in the bottleneck, and a **1-channel sigmoid** output at **256×256** resolution (~31M parameters).
- **YOLO26-seg (nano)** frames the task in the **Ultralytics detection/segmentation** paradigm: bounding boxes plus **mask prototypes**. Training logs use **COCO-style** box and mask metrics (`mAP@0.5`, `mAP@0.5:0.95`, precision, recall). The companion notebook converts binary masks to **YOLO polygon labels** and trains `yolo26n-seg.pt` on the same Kaggle dataset family.

Together they illustrate **two deployment stacks** (TensorFlow CPU-first apps vs PyTorch YOLO GUI) rather than a single locked architecture.

---

## Repository layout

| Path | Description |
|------|-------------|
| `streamlit_app.py` | Streamlit web UI for **U-Net** (`unet_brain_tumor.keras`). CPU-only via `CUDA_VISIBLE_DEVICES=-1`. |
| `tkinter_app.py` | Tkinter desktop UI for **U-Net**; same preprocessing (resize 256, `/255`, threshold `0.5`). |
| `yolo_gui.py` | Tkinter desktop UI for **YOLO**; loads `brainseg26n-seg.pt`, `conf=0.25`, uses `results[0].plot()`. |
| `unet_brain_tumor.keras` | Trained U-Net checkpoint (Keras 3 / Functional API). |
| `unet_brain_tumor_fp16.tflite` | Float16 TFLite export of the U-Net (see `convert_tflite.py`). |
| `convert_tflite.py` | Script: Keras → TFLite with default optimizations + FP16. |
| `brainseg26n-seg.pt` | Trained YOLO26 nano segmentation weights (rename/location may match `yolo_gui.py` expectation in project root). |
| `yolo26/` | Training artifacts: `results.csv`, PR/F1 curves, confusion matrices, `YOLO26segBT.ipynb`, optional copy of `.pt`. |

---

## Dataset & training context (YOLO path)

The notebook `yolo26/YOLO26segBT.ipynb` documents the following:

- **Dataset**: [nikhilroxtomar/brain-tumor-segmentation](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation) (via `kagglehub`), with `images/` and `masks/` PNGs.
- **Split**: **85% train / 15% validation**, `random_state=42` (`sklearn.model_selection.train_test_split`).
- **Label conversion**: binary masks → **normalized polygon** contours (OpenCV), class `0` = `Tumor`, small contours filtered (`area < 10` pixels).
- **Training**: Ultralytics `YOLO('yolo26n-seg.pt')`, `epochs=50`, `imgsz=256`, `batch=256`, `device=0` (notebook targets a single GPU, e.g. Colab T4). Run name / project: `yolo26_run` under `brain_tumor_segmentation` in the original notebook paths.
- **Exported metrics**: logged per epoch in `yolo26/results.csv` and summarized in plots (`results.png`, mask/box PR and F1 curves, confusion matrices).

The **U-Net** checkpoint is assumed to be trained on **compatible** brain MRI PNG data at **256×256** with comparable normalization; the exact split and metric log are **not** stored in this deployment folder.

---

## Models & file formats

### U-Net (`unet_brain_tumor.keras`)

- **Input**: `(None, 256, 256, 3)` RGB, float in \([0,1]\) after division by 255 in the apps.  
- **Output**: `(None, 256, 256, 1)` — interpreted as tumor probability; apps **binarize at 0.5** and display as grayscale mask.  
- **Parameters**: **31,055,297** total (~118.5 MB as reported by Keras).  
- **Inference**: `compile=False` when loading so custom losses/metrics from training are not required.

### YOLO26-seg (`brainseg26n-seg.pt`)

- **Ultralytics** segmentation model (nano variant of the YOLO26-seg family).  
- **GUI**: default confidence **0.25**; overlaid visualization from `plot()`.

---

## YOLO26-seg vs U-Net — metrics comparison

This section is written so you can **paste numbers into a report** and still understand **what is actually comparable**.

### 1. What the repository contains

| Model | Quantitative metrics in this repo? | Source |
|-------|--------------------------------------|--------|
| **YOLO26-seg** | **Yes** — full training curves and per-epoch validation metrics | `yolo26/results.csv` (+ PNG curves in `yolo26/`) |
| **U-Net** | **No** — only the trained weights and inference code | `unet_brain_tumor.keras` — no bundled `history`, CSV, or test-set Dice/IoU |

So: **YOLO numbers below are measured on the validation split defined in the notebook**; **U-Net headline metrics (Dice, IoU, Hausdorff, etc.) are not committed here**. To add them, export your training/evaluation log from the original training project and place it beside this README (or extend this section).

### 2. Different metric families (important for fair comparison)

- **YOLO (Ultralytics)** reports **object-level** metrics on the detection/segmentation task: **Precision**, **Recall**, **mAP@0.5**, **mAP@0.5:0.95** for **(B)** bounding boxes and **(M)** masks. These are **not** the same as pixel-wise Dice or IoU on the full image, though they are all “segmentation quality” indicators.
- **U-Net (typical in medical imaging)** is often scored with **pixel-wise** metrics: **Dice coefficient**, **Jaccard / IoU**, sometimes **specificity/sensitivity** per slice. Those require **per-pixel** comparison to ground-truth masks on a fixed test set.

**Direct numeric comparison** (e.g. “YOLO mAP vs U-Net Dice”) only becomes meaningful if you **evaluate both models on the same held-out images** with **two reporting tables**: keep YOLO’s mAP as-is, and **additionally** compute Dice/IoU (or mutual conversion scripts) for both outputs on that set. This repository does not include that joint benchmark script.

### 3. YOLO26-seg — numbers from `yolo26/results.csv`

All values below are read from the shipped CSV (50 epochs). **(B)** = box, **(M)** = mask.

#### Final epoch (epoch 50) — validation

| Metric | Box (B) | Mask (M) |
|--------|---------|----------|
| **Precision** | 0.8989 | 0.8745 |
| **Recall** | 0.8370 | 0.8304 |
| **mAP@0.5** | 0.9133 | 0.8951 |
| **mAP@0.5:0.95** | 0.5992 | 0.5158 |

#### Best validation across training (peak per column)

| Metric | Best value | Epoch (approx.) |
|--------|------------|-----------------|
| Precision (B) | 0.9189 | 7 |
| Recall (B) | 0.9522 | 5 |
| mAP@0.5 (B) | 0.9142 | 48 |
| **mAP@0.5:0.95 (B)** | **0.5992** | **50** |
| Precision (M) | 0.8948 | 6 |
| Recall (M) | 0.8761 | 3 |
| mAP@0.5 (M) | 0.9060 | 42 |
| **mAP@0.5:0.95 (M)** | **0.5235** | **47** |

**Interpretation (short):** mask **mAP@0.5** in the high **0.89–0.91** range indicates strong overlap at the IoU=0.5 threshold; **mAP@0.5:0.95** (~0.52–0.60 for masks/boxes) is stricter and reflects localization quality across multiple IoU thresholds. See Ultralytics documentation for exact definitions used in your package version.

### 4. U-Net — what to report when you have it

When you recover or compute metrics for `unet_brain_tumor.keras`, a clear table for a thesis or paper might look like:

| Metric (example) | Value | Split / notes |
|------------------|-------|----------------|
| Dice | *(fill in)* | e.g. test, same random seed as YOLO if possible |
| IoU / Jaccard | *(fill in)* | |
| Pixel accuracy | *(optional)* | often less informative for imbalanced tumors |

**Suggested alignment with YOLO:** use the **same Kaggle dataset**, the **same 85/15 split seed (42)**, resize/crop rules **identical to training** for U-Net, then report **Dice/IoU for U-Net** alongside **mAP for YOLO**, with a footnote that the **primary** metrics differ by convention.

### 5. Side-by-side summary (as shipped)

| Aspect | YOLO26-seg (this repo) | U-Net (this repo) |
|--------|-------------------------|-------------------|
| **Reported validation metrics** | Precision/Recall/mAP (box + mask), 50 epochs | *Not stored* |
| **Typical academic metrics** | COCO-style mAP | Dice, IoU (to be added by you) |
| **Output** | Boxes + rasterized instance mask overlay | Full-resolution (256) dense mask |
| **Params (reference)** | Nano YOLO26-seg (see Ultralytics doc for exact count) | ~31.1M |

---

## Running the applications

From the repository root (with the corresponding model files present).

### U-Net — Streamlit

```bash
streamlit run streamlit_app.py
```

Requires `unet_brain_tumor.keras` in the working directory.

### U-Net — Tkinter

```bash
python tkinter_app.py
```

### YOLO — Tkinter

```bash
python yolo_gui.py
```

Expects `brainseg26n-seg.pt` in the working directory (adjust path in `yolo_gui.py` if your weights live under `yolo26/`).

---

## TFLite conversion (U-Net)

```bash
python convert_tflite.py
```

Produces `unet_brain_tumor_fp16.tflite` using `tf.lite.TFLiteConverter` with default optimizations and **float16** weights. Use this for on-device or embedded targets that support FP16 TFLite delegates.

---

## Dependencies

There is no `requirements.txt` in this bundle; inferred dependencies:

- **U-Net apps / conversion**: `tensorflow`, `streamlit`, `numpy`, `Pillow`  
- **Tkinter U-Net**: standard library `tkinter` (platform-dependent), `Pillow`, `tensorflow`, `numpy`  
- **YOLO GUI**: `ultralytics`, `opencv-python`, `Pillow`, `torch` (installed as a dependency of Ultralytics)

Install versions compatible with your GPU/CPU setup and your Keras checkpoint format.

---

## Limitations & reproducibility notes

- **Metric parity**: YOLO and U-Net were **not** jointly evaluated in this folder; the comparison section documents **what exists** and how to extend it.  
- **JPEG vs PNG**: `tkinter_app.py` reads files with `decode_png`; for JPEG inputs you may need `decode_image` or unified decoding — Streamlit uses PIL and is format-agnostic.  
- **Paths**: Keep `.keras`, `.pt`, and optional `.tflite` next to the scripts or update paths in code.  
- **Training notebook environment**: Cell metadata references Colab/Kaggle; paths like `/content/...` are not valid on Windows without adaptation.

---

## Credits

- **Dataset (YOLO notebook)**: [nikhilroxtomar/brain-tumor-segmentation](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation) on Kaggle.  
- **YOLO training & tooling**: [Ultralytics](https://github.com/ultralytics/ultralytics).  
- **U-Net**: standard encoder–decoder architecture as implemented in the shipped Keras functional model.

---

*Last README update aligned with repository contents and `yolo26/results.csv` (50 epochs). Add U-Net metric rows when your evaluation artifacts are available.*
