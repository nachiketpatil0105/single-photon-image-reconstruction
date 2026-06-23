# Phase 4 — ResUNet with Attention Gates

> Best model: residual connections + attention gates for focused, high-fidelity reconstruction.

← [Phase 3](../phase3_unet/README.md) | [Back to main repo](../README.md)

---

## 📌 Overview

This final phase combines two powerful improvements over standard UNet:

1. **Residual Connections** — Within each conv block, a skip connection adds the block's input to its output. This eases gradient flow during training and lets the network learn refinements rather than full transformations.

2. **Attention Gates** — Applied at each skip connection in the decoder, attention gates learn to suppress irrelevant or noisy activations and amplify features in regions that matter for reconstruction. This is particularly useful for single-photon data, where the signal of interest is spatially sparse.

---

## 🏗️ Architecture

```
Input burst [B, N, H, W]
      ↓
┌─────────────────────────────────────────────┐
│         ENCODER (Residual Conv Blocks)      │
│  ResBlock → MaxPool                         │
│  ResBlock → MaxPool                         │
│  ResBlock → MaxPool                         │
│  Bottleneck ResBlock                        │
└───────────────────┬─────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │   Attention Gate    │  ← filters skip connection features
         └──────────┬──────────┘
┌───────────────────▼─────────────────────────┐
│         DECODER (UpConv + Attention Concat) │
│  UpConv + AttGate(skip) + Concat → ResBlock │
│  UpConv + AttGate(skip) + Concat → ResBlock │
│  UpConv + AttGate(skip) + Concat → ResBlock │
└─────────────────────────────────────────────┘
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

## 📊 Results — Full Comparison

| Metric | Naive | Baseline CNN | UNet | ResUNet + Attention | Best Δ |
|--------|-------|--------------|------|----------------------|--------|
| PSNR   | —     | —            | —    | —                    | —      |
| SSIM   | —     | —            | —    | —                    | —      |

---

## 🖼️ Visual Results

_Add side-by-side comparisons here: Noisy Input | UNet Output | ResUNet+Attn Output | Ground Truth_

---

## 💡 Key Takeaway

Attention gates allow the decoder to focus on informative spatial regions rather than processing the entire feature map uniformly. Combined with residual learning, this yields the sharpest and most faithful reconstructions across the phase progression — with the most significant gains in low-photon, high-noise regions.

---

← [Phase 3](../phase3_unet/README.md) | [Back to main repo](../README.md)
