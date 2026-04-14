# 🫁 Chest X-Ray Pneumonia Classifier

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![MONAI](https://img.shields.io/badge/MONAI-1.x-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **⚠️ NOT FOR CLINICAL USE.** This is a learning project built as part of a self-directed AI for Biomedical Engineering study programme. It has not been validated for clinical deployment and must not be used for patient diagnosis or triage.

A binary chest X-ray classifier that distinguishes **Normal** from **Pneumonia** using a DenseNet-121 backbone trained with two-phase transfer learning. Built with PyTorch and MONAI on the Kaggle Chest X-Ray (Pneumonia) dataset. Includes ROC curve evaluation, Grad-CAM visualisation, and a full project report.

## 🔍 Grad-CAM Visualisation

![Grad-CAM Analysis](report/figures/gradcam_visualisation.png)

> Model attention on lung consolidation patterns for pneumonia detection. Correctly classified cases show warm activation (red/yellow) in lower lung zones. False negatives show attention outside lung fields—revealing model limitations.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Results](#results)
- [Grad-CAM Visualisation](#grad-cam-visualisation)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Project Report](#project-report)
- [References](#references)

---

## What It Does

1. **Loads and preprocesses** chest X-rays from the Kaggle Pneumonia dataset — resizing to 224×224, normalising to [0, 1], converting to single-channel greyscale, and applying training augmentations (random flips, minor rotations).
2. **Fine-tunes DenseNet-121** in two phases: first training only the classification head (backbone frozen), then unfreezing the full network for end-to-end fine-tuning at a lower learning rate.
3. **Evaluates** with AUC-ROC, sensitivity, specificity, and confusion matrix — not just accuracy, which is misleading on this imbalanced dataset.
4. **Generates Grad-CAM heatmaps** overlaid on test images to show which regions of the X-ray the model attends to, and identifies failure modes where attention falls outside the lung fields.

---

## Results

| Metric | Value |
|---|---|
| Test AUC | **0.8887** |
| Sensitivity (threshold = 0.01) | 0.5103 |
| Specificity (threshold = 0.01) | 0.9615 |

The model exceeds AUC 0.85, the technical success criterion defined in the project report. The gap between AUC (0.89) and sensitivity (0.51) reflects the difficulty of threshold calibration on a heavily imbalanced dataset with an unrepresentative 16-image validation split. See [Limitations](#limitations) and the [project report](report/PROJECT_REPORT.md) for a full discussion.

**ROC curve and confusion matrix** are generated in `notebooks/04_evaluation.ipynb`.

---

## Grad-CAM Visualisation

Grad-CAM heatmaps are generated for 8 test images (4 Normal, 4 Pneumonia) using the final activation layer of DenseNet-121 (`class_layers.relu`).

![Grad-CAM Visualisation](report/figures/gradcam_visualisation.png)

*Correctly classified pneumonia cases show warm activation (red/yellow) in the lower and perihilar lung zones, consistent with consolidation patterns characteristic of pneumonia. False negative cases show near-zero pneumonia probability with activation concentrated at the inferior image border — outside the lung fields entirely.*

![Grad-CAM Error Analysis](report/figures/gradcam_errors.png)

---

## Dataset

**Kaggle Chest X-Ray Images (Pneumonia)**  
Source: Paul Mooney / Kaggle — Guangzhou Women and Children's Medical Center  
Download: [kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

The dataset is ~3:1 imbalanced (pneumonia:normal in training). All images originate from a single paediatric hospital; see [Limitations](#limitations).

---

## How to Run

### Option A — Kaggle Notebooks (recommended, free GPU)

This project was developed and tested on Kaggle Notebooks with a T4x2 GPU. The simplest way to reproduce results is to open each notebook directly on Kaggle and attach the chest X-ray dataset.

### Option B — Local Setup

**Requirements:** Python 3.11, conda, ~10 GB disk space for the dataset.

```bash
# 1. Clone the repository
git clone https://github.com/arapkirui513-hub/chest-xray-classifier.git
cd chest-xray-classifier

# 2. Create and activate the conda environment
conda create -n ai-biomed-py311 python=3.11 -y
conda activate ai-biomed-py311

# 3. Install dependencies
pip install torch torchvision monai[all] matplotlib scikit-learn grad-cam kaggle

# 4. Download the dataset (requires a Kaggle API token at ~/.kaggle/kaggle.json)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/

# 5. Run the notebooks in order
jupyter notebook notebooks/
```

**Notebook order:**

| Notebook | Purpose |
|---|---|
| `week3_day4_day5.ipynb` | Data pipeline, two-phase DenseNet-121 training, loss curves, error analysis |
| `week4_day1_evaluation.ipynb` | ROC curve, AUC, sensitivity, specificity, confusion matrix |
```

Each notebook runs from top to bottom without modification if the dataset is placed at `data/chest_xray/`.

**Expected runtime:** ~25–35 minutes for full training on a T4 GPU. CPU-only training is possible but will take several hours.

---

## Repository Structure

```
chest-xray-classifier/
├── README.md
├── .gitignore
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_pipeline.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_gradcam.ipynb
├── figures/
│   ├── roc_curve.png
│   └── confusion_matrix.png
├── outputs/
│   └── densenet121_best.pth        ← trained model weights (not tracked by git)
└── report/
    ├── PROJECT_REPORT.md
    └── figures/
        ├── gradcam_visualisation.png
        └── gradcam_errors.png
```

> Model weights (`outputs/*.pth`) are excluded from git via `.gitignore` due to file size. Retrain using `notebooks/03_training.ipynb` to reproduce them.

---

## Limitations

This project is a learning exercise, not a production system. Key limitations:

- **Sensitivity of 0.51 is clinically unacceptable.** A real screening tool requires sensitivity > 0.90. Approximately one in two pneumonia cases would be missed at the operating threshold.
- **Single-source dataset.** All images originate from one paediatric hospital in China. The model has not been validated on adult populations, different equipment, or other institutions.
- **Spurious boundary features.** Grad-CAM analysis shows some false negative cases have activation concentrated at the inferior image border rather than within lung tissue — evidence of artefactual correlations in this dataset.
- **PNEUMONIA label conflates bacterial and viral subtypes**, which have distinct radiological presentations. The model cannot distinguish between them.
- **Not MHRA-registered.** Clinical deployment in the UK would require registration as a Class IIa medical device, prospective clinical validation, and clinician oversight on every prediction. None of these conditions are met.

The most impactful next step would be applying lung segmentation as a preprocessing step to force the model to attend only to lung parenchyma, directly addressing the boundary activation problem identified in Grad-CAM analysis.

---

## Project Report

A full 2-page project report is available at [`report/PROJECT_REPORT.md`](report/PROJECT_REPORT.md).

It covers: problem statement, dataset analysis, preprocessing and model architecture, training setup, evaluation results, Grad-CAM findings, limitations, and what would be required for clinical deployment under UK MHRA regulation.

---

## References

- Rajpurkar, P. et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. [arXiv:1711.05225](https://arxiv.org/abs/1711.05225)
- Wang, X. et al. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. CVPR 2017.
- Selvaraju, R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017.
- MONAI Consortium (2020). MONAI: Medical Open Network for AI. [monai.io](https://monai.io)

---

*Built as part of a self-directed Pre-Stanmore AI for Biomedical Engineering study programme — Week 4, Day 4.*
