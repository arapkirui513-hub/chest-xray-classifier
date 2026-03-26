\# Chest X-Ray Classifier



Binary classification of chest X-rays (Normal vs. Pneumonia) using MONAI

and PyTorch, trained on the Kaggle Chest X-Ray dataset.



Built as part of my pre-Stanmore AI for Biomedical Engineering self-study.



\---



\## Results



| Metric | Value |

|--------|-------|

| Test AUC | 0.9086 |

| Sensitivity (threshold 0.35) | 0.992 |

| Specificity (threshold 0.35) | 0.393 |

| Missed pneumonia cases | 3 / 390 |



!\[ROC Curve](figures/roc\_curve.png)

!\[Threshold Curve](figures/threshold\_curve.png)



\---



\## Dataset



\*\*Kaggle Chest X-Ray Images (Pneumonia)\*\*

5,216 training images · 624 test images · Binary: Normal / Pneumonia

Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia



\---



\## Model



\- Architecture: DenseNet-121 (MONAI model zoo)

\- Training: Two-phase transfer learning

&#x20; - Phase 1 (5 epochs, frozen backbone): val AUC 0.9347

&#x20; - Phase 2 (5 epochs, unfrozen, lr=1e-4): val AUC 0.9968

\- Loss: BCEWithLogitsLoss

\- Threshold: 0.35 (chosen to maximise sensitivity for screening)



\---



\## How to Run

```bash

git clone https://github.com/arapkirui513-hub/chest-xray-classifier

cd chest-xray-classifier

pip install torch monai scikit-learn matplotlib

```



Open `notebooks/week3\\\_day4\\\_day5.ipynb` and update the dataset path to

your local copy, then run top to bottom.



\---



\## Error Analysis



The model struggles with:

1\. Subtle ground-glass opacities and diffuse interstitial patterns

2\. Prominent hilar vasculature mistaken for infiltrates

3\. Under-inflated lungs causing basal densification



See `notebooks/week3\\\_day4\\\_day5.ipynb` for full analysis.



!\[False Negatives](figures/false\_negatives.png)

!\[False Positives](figures/false\_positives.png)



\---



\## Limitations



\- Validation AUC (0.9968) vs test AUC (0.9086) gap suggests overfitting

&#x20; to the validation distribution

\- Low specificity (0.393) means high false alarm rate — not suitable

&#x20; for standalone clinical use

\- Dataset contains only two classes; real deployment requires multi-label

&#x20; classification across 14+ pathologies

\- No Grad-CAM visualisation yet (coming Week 4 Day 2)



\---



\## NOT FOR CLINICAL USE



This is a learning project. It has not been validated for clinical deployment.

