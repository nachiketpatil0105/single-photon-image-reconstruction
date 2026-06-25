# 📷 Single Photon Image Reconstruction

> Reconstructing high-quality images from noisy, binary single-photon camera data —
> progressing from a naive summation baseline to a ResUNet with Guided Attention Gates.

[![Dataset](https://img.shields.io/badge/Dataset-Single%20Photon%20Challenge-blue?style=flat-square)](https://singlephotonchallenge.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=flat-square)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square)](https://pytorch.org/)
[![Author](https://img.shields.io/badge/Author-nachiketpatil0105-black?style=flat-square&logo=github)](https://github.com/nachiketpatil0105)

---

## Problem Statement

Single-photon cameras detect individual photons, making them extremely light-sensitive —
but each captured frame is binary and dominated by photon shot noise. Given a burst of
such noisy binary frames, the goal is to recover a clean, high-fidelity RGB image.

This project develops a series of increasingly capable deep learning models to solve this
reconstruction problem. Each phase is built on the lessons and limitations of the previous
one, resulting in a clear progression from a parameter-free baseline to a 237M parameter
architecture with residual blocks, a 5-level encoder, and guided attention gates on every
skip connection.

---

## Results at a Glance

| Phase | Model | Avg PSNR | Avg SSIM | Params |
|:-----:|-------|:--------:|:--------:|-------:|
| 1 | Naive Summation | 21.55 dB | 0.4547 | 0 |
| 2 | Baseline CNN | 28.90 dB | 0.8701 | ~1.72M |
| 3 | UNet | 32.62 dB | 0.8819 | ~31.2M |
| **4** | **ResUNet + Attention** | **34.00 dB** | **0.8833** | **~236.9M** |

**Total improvement: +12.45 dB PSNR and +0.43 SSIM over the naive baseline.**

Each phase's README contains full per-sample results, training curves, architecture
diagrams, and analysis of what drove the improvement.

---

## Repository Structure

```
single-photon-reconstruction/
│
├── phase1_naive/
│   ├── naive_reconstruction.py
│   ├── results/
│   └── README.md
│
├── phase2_baseline_cnn/
│   ├── cnn_reconstruction.py
│   ├── run_and_collect_results.ipynb
│   ├── results/
│   └── README.md
│
├── phase3_unet/
│   ├── unet_reconstruction.py
│   ├── run_and_collect_results.ipynb
│   ├── results/
│   └── README.md
│
├── phase4_resunet_attention/
│   ├── resunet_reconstruction.py
│   ├── results/
│   └── README.md
│
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [The Single Photon Challenge](https://singlephotonchallenge.com/)

A large synthetic dataset of indoor scenes consisting of:
- Input: bursts of binary single-photon camera frames, stored as bit-packed `.npy` files
  with shape `(1024, H, W, 100, 3)` — 1024 frames × 100 packed temporal bins × 3 channels
- Target: corresponding ground truth high-fidelity RGB images

The dataset was produced by researchers from University of Wisconsin–Madison, Portland
State University, Purdue University, Carnegie Mellon University, and the US Naval
Research Lab.

---

## Phase Summaries

### Phase 1 — Naive Summation

No learning, no parameters. Binary photon frames are summed across a burst and normalized
to recover approximate scene intensity. We sweep batch sizes from 16 to 1024 to find where
diminishing returns kick in. Best PSNR: 21.55 dB. This is the performance floor every
subsequent model must beat.

→ [View Phase 1](./phase1_naive/README.md)

---

### Phase 2 — Baseline CNN

First trained model. All 128 SPC frames are stacked into 384 input channels (128 × 3 RGB)
and passed through a flat 8-layer CNN with no downsampling. Trained with a combined
L1 + SSIM loss for 50 epochs. Average PSNR reached 28.90 dB — a +7.35 dB gain over
naive summation — confirming that learning to combine frames outperforms any fixed rule.

→ [View Phase 2](./phase2_baseline_cnn/README.md)

---

### Phase 3 — UNet

Introduces the encoder-decoder architecture with skip connections. The encoder compresses
spatial resolution across 3 levels to build global scene understanding; the decoder
reconstructs at full resolution, receiving encoder features at each level via skip
connections. Also introduces a substantially upgraded training setup: Charbonnier + MS-SSIM
+ VGG perceptual loss, cosine annealing, mixed precision, data augmentation, and a
proper train/val/test split. Average PSNR: 32.62 dB (+3.72 dB over Phase 2).

→ [View Phase 3](./phase3_unet/README.md)

---

### Phase 4 — ResUNet with Guided Attention Gates

The final model. Four improvements over the Phase 3 UNet:

1. **Residual ConvBlocks** — each block learns a correction over its input
   (GroupNorm + SiLU + residual projection), making training more stable at depth.
2. **5-level encoder + double bottleneck** — one extra downsampling level gives the
   bottleneck a 32×32 receptive field; two stacked ConvBlocks at 2048 channels maximise
   reasoning capacity there.
3. **Guided Attention Gates** (Oktay et al., 2018) — each skip connection is filtered by
   an attention map computed jointly from the encoder feature (where) and the decoder
   context (what), suppressing irrelevant spatial positions before concatenation.
4. **Deep supervision** — an auxiliary output at dec2 is annealed from weight 0.4 to 0
   over training, keeping intermediate features semantically meaningful without
   over-constraining the final output.

Trained on 1850 samples (40× more than Phase 3) on a dedicated GPU server.
Average PSNR: 34.00 dB. Best single-sample PSNR: 37.43 dB.

→ [View Phase 4](./phase4_resunet_attention/README.md)

---

## Setup

```bash
git clone https://github.com/nachiketpatil0105/single-photon-reconstruction.git
cd single-photon-reconstruction
pip install -r requirements.txt
```

Phases 1–3 can be run on Kaggle (free GPU tier). Phase 4 requires a high-VRAM GPU server
due to the 237M parameter model and batch size 2 at 512×512 resolution.

---

## Citation

If you use the Single Photon Challenge dataset, please cite:

```bibtex
@software{visionsim,
    author = {Jungerman, Sacha and Gupta, Shantanu and Sadekar, Kaustubh
              and Leblang, Max and Gupta, Mohit},
    license = {MIT},
    month = may,
    title = {{visionsim}},
    url = {https://github.com/WISION-Lab/visionsim},
    year = {2025}
}
```

---

## Author

**Nachiket Patil**
GitHub: [@nachiketpatil0105](https://github.com/nachiketpatil0105)
