# Phase 2 — Baseline CNN

> First learned model: a simple convolutional neural network trained end-to-end for image reconstruction.

← [Phase 1](../phase1_naive/README.md) | [Back to main repo](../README.md) | [Phase 3 →](../phase3_unet/README.md)

---

## 📌 Overview

This phase introduces the first **trained deep learning model**. A baseline CNN is trained to map a burst of noisy binary frames directly to a clean reconstructed image. The architecture is intentionally simple — the goal is to confirm that a learned approach outperforms naive averaging, and to identify where a simple CNN falls short.

---

## 🏗️ Architecture

```
Input burst [B, N, H, W]
      ↓
  Conv2D + ReLU
      ↓
  Conv2D + ReLU
      ↓
  Conv2D + ReLU
      ↓
  Conv2D (output)
      ↓
Reconstructed image [B, 1, H, W]
```

> Update with your actual layer configuration (channels, kernel sizes, etc.)

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | — |
| Optimizer | — |
| Learning Rate | — |
| Epochs | — |
| Batch Size | — |

---

## 📊 Results

| Metric | Naive (Phase 1) | Baseline CNN (Phase 2) | Δ Improvement |
|--------|-----------------|------------------------|---------------|
| PSNR   | —               | —                      | —             |
| SSIM   | —               | —                      | —             |

---

## 🖼️ Visual Results

_Add side-by-side comparisons here: Noisy Input | Naive Output | CNN Output | Ground Truth_

---

## 💡 Key Takeaway

A basic CNN learns to denoise better than averaging, but without a multi-scale structure it struggles to capture both coarse scene layout and fine texture simultaneously. This motivates moving to a UNet-style encoder-decoder.

---

← [Phase 1](../phase1_naive/README.md) | [Back to main repo](../README.md) | [Phase 3 →](../phase3_unet/README.md)
