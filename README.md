![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.11.14-blue.svg)
![Computer Vision](https://img.shields.io/badge/Area-Computer%20Vision-success)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Result at a Glance

| Metric | Performance / Outcome |
| :--- | :--- |
| **Accuracy** | **99.6%** on industrial casting dataset |
| **Efficiency** | **16x smaller** than ResNet18 (4MB vs 64MB) |
| **Reliability** | Verified with **Grad-CAM**|
| **Deployment** | Optimized for **Real-time Edge devices**, no GPU required |

---

## Problem
 
Manual quality control in casting is slow, inconsistent, and hard to scale on high-speed production lines.
 
**Goal:** Classify parts as "OK" or "Defective" with high accuracy and low resource consumption.
 
---

## Dataset
 
- **Source:** [Casting Product Dataset — Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/data)
- **Content:** 1,300 images of submersible pump impellers
- **Classes:** Defective / OK
- **Challenge:** Small, imbalanced dataset → solved with extensive data augmentation (flips, rotations, brightness, noise, CoarseDropout)

---

## Approach
 
### Models Compared
- **ResNet18** — pretrained baseline (64.5 MB, 100% accuracy)
- **Custom CNN** — lightweight architecture with residual blocks (4 MB, 99.6% accuracy)
### Why Custom CNN?
ResNet18 is too large for edge deployment. The goal was to build something significantly smaller without sacrificing real performance.
 
### Interpretability — Grad-CAM
Accuracy alone isn't enough. Every model iteration was validated with Grad-CAM heatmaps to confirm the model focuses on actual defects — not background artifacts.
 
---

## Development Journey (5 Iterations)
 
The custom CNN went through 5 architectural iterations guided by Grad-CAM feedback:
 
| Version | Issue | Fix |
| :--- | :--- | :--- |
| v1 | Scattered attention | Architecture too shallow |
| v2 | Overfitting (100% acc, wrong focus) | Added depth without regularization |
| v3 | Same — center-heavy bias | Heavy augmentation didn't fix architecture |
| v4 | Underfitting (95% acc) | Too aggressive downscaling |
| **v5 (Final)** | ✅ Sharp, defect-focused attention | Balanced depth + stride strategy + full regularization |
 
**Final Architecture:**
- Stem: 3×3 Conv (stride 2) → BN → ReLU (3→16 channels)
- Stage 0: 2 blocks, stride 1 (16→32) — preserves early spatial detail
- Stage 1: 2 blocks, stride 2 (32→64)
- Stage 2: 2 blocks, stride 2 (64→128)
- Regularization: Dropout + Weight Decay + `pos_weight` in loss

---

## Results
 
| Metric | ResNet18 | Custom CNN |
| :--- | :--- | :--- |
| Test Accuracy | 100% | **99.6%** |
| Parameters | 11.17M | **0.69M** |
| Model Size | 64.5 MB | **4 MB** |
| Grad-CAM | ✅ Defect-focused | ✅ Defect-focused |
| Edge Deployment | ❌ Heavy | ✅ Ready |

<p align="center">
   <img src="media/heatmaps/final_simple_model/correct/model_simple_error_140_FN.jpg" width="700">
</p>

---

## Key Takeaways
 
- **Grad-CAM is essential** — high accuracy without interpretability verification is meaningless
- **Architecture balance beats size** — more depth led to overfitting; the right stride strategy mattered more
- **Iterative, hypothesis-driven development** outperforms random hyperparameter tuning

> Full source code, training scripts, and Grad-CAM visualizations are in this repository. Custom CNN weights included. ResNet18 weights: [Google Drive](https://drive.google.com/file/d/12v_llowvkEUnrz8_rUohmBaaeAPk7aOG/view?usp=drive_link).
