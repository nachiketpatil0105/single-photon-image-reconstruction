# Single Photon Image Reconstruction

Reconstructing clean RGB images from noisy binary single-photon camera data — progressing
from a naive summation baseline to a ResUNet with Guided Attention Gates.

[![Dataset](https://img.shields.io/badge/Dataset-Single%20Photon%20Challenge-blue?style=flat-square)](https://singlephotonchallenge.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=flat-square)](https://www.python.org/)

---

## The Problem

Single-photon cameras detect individual photons — each frame is binary (0 or 1 per pixel)
and dominated by shot noise. Given a burst of such frames, the task is to reconstruct the
clean scene.

---

## Phase 2 vs Phase 3 — same scene, same model scale

![Phase 2 vs Phase 3](assets/phase2_vs_phase3.png)

---

## Results

| Phase | Model | Avg PSNR | Avg SSIM | Params |
|:-----:|-------|:--------:|:--------:|-------:|
| 1 | Naive Summation | 21.55 dB | 0.4547 | — |
| 2 | Baseline CNN | 28.90 dB | 0.8701 | ~1.7M |
| 3 | UNet | 32.62 dB | 0.8819 | ~31M |
| **4** | **ResUNet + Attention** | **34.00 dB** | **0.8833** | ~237M |

**Total: +12.45 dB PSNR over the naive baseline.**

---

## Structure

```
single-photon-reconstruction/
├── phase1_naive/
│   ├── naive_reconstruction.py
│   └── results/
├── phase2_baseline_cnn/
│   ├── cnn_reconstruction.py
│   └── results/
├── phase3_unet/
│   ├── unet_reconstruction.py
│   └── results/
├── phase4_resunet_attention/
│   ├── resunet_reconstruction.py
│   └── results/
├── assets/
└── requirements.txt
```

---

## Phases

**[Phase 1 — Naive Summation](./phase1_naive/README.md)**
No learning. Sum binary frames, normalize. Establishes the performance floor.

**[Phase 2 — Baseline CNN](./phase2_baseline_cnn/README.md)**
Flat 8-layer CNN. 128 frames stacked as 384 input channels. First trained model.

**[Phase 3 — UNet](./phase3_unet/README.md)**
Encoder-decoder with skip connections. Upgraded training: Charbonnier + MS-SSIM + VGG
loss, cosine annealing, mixed precision, augmentation, train/val/test split.

**[Phase 4 — ResUNet + Attention](./phase4_resunet_attention/README.md)**
Residual blocks, 5-level encoder, double bottleneck at 2048ch, guided attention gates on
every skip, deep supervision, edge loss. 1850 training samples on a dedicated GPU server.

---

## Setup

```bash
git clone https://github.com/<your-username>/single-photon-reconstruction.git
cd single-photon-reconstruction
pip install -r requirements.txt
```

See each phase folder for dataset path configuration and running instructions.
Phases 1–3 run on a standard GPU. Phase 4 requires ~40GB VRAM.

---

## Dataset

[The Single Photon Challenge](https://singlephotonchallenge.com/) — synthetic indoor
scenes. Input: binary SPC bursts as bit-packed `.npy` files `(1024, H, W, 100, 3)`.
Target: ground truth RGB images.

```bibtex
@software{visionsim,
    author  = {Jungerman, Sacha and Gupta, Shantanu and Sadekar, Kaustubh
               and Leblang, Max and Gupta, Mohit},
    title   = {{visionsim}},
    url     = {https://github.com/WISION-Lab/visionsim},
    year    = {2025}
}
```
