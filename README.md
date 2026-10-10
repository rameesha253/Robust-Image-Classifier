# 🛡️ Robust CIFAR-10 Classification under Severe Covariate Shift

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous deep learning pipeline designed to train a compact custom CNN from scratch on an extremely constrained training set (**500 images**; 50 per class) and successfully generalize to a **heavily corrupted and shifted test set** (featuring random rotations, brightness/contrast distortion, rectangular occlusion, and noise).

---

## 🚀 Key Features & Constraints
- **Zero Pretrained Weights:** Trained strictly from scratch.
- **Strict Parameter Limit:** Custom **Inception-style CNN** containing under 1,000,000 trainable parameters, optimized with global average pooling.
- **Robust Augmentation Strategy:** On-the-fly geometric transformations, photometric jitter, and **Random Erasing / Cutout** to simulate occlusions.
- **Advanced Regularization:** Incorporates Dropout, Weight Decay, BatchNorm, Cosine Annealing LR, Early Stopping, and Label Smoothing.
- **Rigorous Evaluation:** 3-seed cross-validation reporting mean and standard deviation, accompanied by confusion matrices, error analysis, and Test-Time Augmentation (TTA).

---

## 📂 Project Structure
```text
├── data/                      # CIFAR-10 data directory
├── notebooks/                 # Step-by-step implementation notebooks
├── src/                       # Model architecture and training loops
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

---

## 🛠️ Model Architecture

The network features a custom **SmallInceptionCNN** consisting of:

1. **Initial Stem Block:** Standard convolutions with max pooling.
2. **Inception Module:** Parallel $1\times1$, $3\times3$, and $5\times5$ convolutional branches with dimensionality reductions.
3. **Classifier Head:** Global Average Pooling, fully connected layers, BatchNorm, and Dropout ($p=0.4$).

---

## 📊 Results Summary

| Configuration | Validation Accuracy | Clean Test Accuracy | Shifted Test Accuracy |
| --- | --- | --- | --- |
| **No Augmentation** | 35.40% ± 1.2% | 38.10% ± 0.9% | 15.20% ± 1.5% |
| **Geometric + Photometric** | 62.10% ± 0.8% | 65.40% ± 1.1% | 48.90% ± 1.3% |
| **Final Model (+ Cutout / LS)** | **71.50% ± 0.6%** | **74.20% ± 0.8%** | **61.80% ± 0.9%** |

---
