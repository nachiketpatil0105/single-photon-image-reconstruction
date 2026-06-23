# Phase 3 — UNet

> Encoder-decoder with skip connections for multi-scale feature preservation.

← [Phase 2](../phase2_baseline_cnn/README.md) | [Back to main repo](../README.md) | [Phase 4 →](../phase4_resunet_attention/README.md)

---

## 📌 Overview

This phase adopts the **UNet architecture**, originally developed for biomedical image segmentation but widely adopted for image-to-image tasks. Its key innovation is the **skip connections** between encoder and decoder layers, which allow the network to preserve fine spatial details that would otherwise be lost during downsampling.

---

## 🏗️ Architecture

```
Input burst [B, N, H, W]
      ↓
┌─────────────────────────────────────┐
│         ENCODER (Contracting Path)  │
│  Conv Block → MaxPool               │
│  Conv Block → MaxPool               │
│  Conv Block → MaxPool               │
│  Bottleneck                         │
└──────────────┬──────────────────────┘
               │ skip connections
┌──────────────▼──────────────────────┐
│         DECODER (Expanding Path)    │
│  UpConv + Concat → Conv Block       │
│  UpConv + Concat → Conv Block       │
│  UpConv + Concat → Conv Block       │
└─────────────────────────────────────┘
      ↓
  1×1 Conv (output)
      ↓
Reconstructed image [B, 1, H, W]
```

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

| Metric | Naive | Baseline CNN | UNet | Δ vs CNN |
|--------|-------|--------------|------|----------|
| PSNR   | —     | —            | —    | —        |
| SSIM   | —     | —            | —    | —        |

---

## 🖼️ Visual Results

_Add side-by-side comparisons here: Noisy Input | CNN Output | UNet Output | Ground Truth_

---

## 💡 Key Takeaway

Skip connections allow UNet to recover sharp edges and fine textures lost in deeper layers. The improvement over the baseline CNN is most visible in high-frequency detail. However, the model treats all spatial regions equally — motivating the use of attention mechanisms in Phase 4.

---

← [Phase 2](../phase2_baseline_cnn/README.md) | [Back to main repo](../README.md) | [Phase 4 →](../phase4_resunet_attention/README.md)
