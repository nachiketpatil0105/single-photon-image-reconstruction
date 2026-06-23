# Phase 1 — Naive Averaging Baseline

> A no-learning baseline: average a burst of noisy binary frames to reduce shot noise.

← [Back to main repo](../README.md)

---

## 📌 Overview

Before training any model, we establish a naive baseline using **temporal averaging**. Given a burst of binary single-photon frames, each pixel value is averaged across all frames in the burst. The intuition: photon shot noise is random, so averaging many frames should approximate the true underlying scene intensity.

This phase has **no trainable parameters** — it exists solely to set a lower-bound reference for all subsequent deep learning phases.

---

## 🔧 Method

```
Input: Burst of N binary frames  [B, N, H, W]
       ↓
   Average across N frames
       ↓
Output: Reconstructed image      [B, 1, H, W]
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| PSNR   | —     |
| SSIM   | —     |

> Add your naive baseline scores here.

---

## 🖼️ Visual Results

_Add reconstructed sample images here (e.g., `results/sample_output.png`)_

---

## 💡 Key Takeaway

Naive averaging smooths out noise but loses sharpness and fine detail — especially at low photon counts. This motivates the need for learned reconstruction.

---

← [Back to main repo](../README.md) | [Phase 2 →](../phase2_baseline_cnn/README.md)
