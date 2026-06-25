# Single Photon Image Reconstruction

Reconstructing clean RGB images from noisy binary single-photon camera data. This project
explores four progressively stronger deep learning architectures, each trained to map a
burst of photon frames to a high-quality image.

[![Dataset](https://img.shields.io/badge/Dataset-Single%20Photon%20Challenge-blue?style=flat-square)](https://singlephotonchallenge.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=flat-square)](https://www.python.org/)

---

## The Problem

Single-photon cameras detect individual photons — each frame is binary (0 or 1 per pixel)
and dominated by shot noise. Given a burst of such frames, the task is to reconstruct the
clean scene behind the noise.

---

## Results

| Phase | Model | Avg PSNR | Avg SSIM | Params |
|:-----:|-------|:--------:|:--------:|-------:|
| 1 | Naive Summation | 21.55 dB | 0.4547 | — |
| 2 | Baseline CNN | 28.90 dB | 0.8701 | ~1.7M |
| 3 | UNet | 32.62 dB | 0.8819 | ~31M |
| **4** | **ResUNet + Attention** | **34.00 dB** | **0.8833** | ~237M |

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
└── requirements.txt
```

---

## Phases

**[Phase 1 — Naive Summation](./phase1_naive/README.md)**
No learning. Sum binary frames across the burst. Establishes the performance floor.

**[Phase 2 — Baseline CNN](./phase2_baseline_cnn/README.md)**
Flat 8-layer CNN. All 128 frames fed as 384 input channels. First trained model.

**[Phase 3 — UNet](./phase3_unet/README.md)**
Encoder-decoder with skip connections, multi-scale feature reuse, and a significantly
upgraded training pipeline (Charbonnier + MS-SSIM + VGG loss, augmentation, cosine LR).

**[Phase 4 — ResUNet + Attention](./phase4_resunet_attention/README.md)**
Residual blocks, 5-level encoder, double bottleneck, guided attention gates on every skip,
deep supervision, and edge loss. Trained on 1850 samples on a dedicated GPU server.

---

## Setup

```bash
git clone https://github.com/<your-username>/single-photon-reconstruction.git
cd single-photon-reconstruction
pip install -r requirements.txt
```

See each phase folder for dataset path configuration and running instructions.

---

## Dataset

[The Single Photon Challenge](https://singlephotonchallenge.com/) — synthetic indoor
scenes with binary SPC bursts as input and ground truth RGB images as target.

```bibtex
@software{visionsim,
    author  = {Jungerman, Sacha and Gupta, Shantanu and Sadekar, Kaustubh
               and Leblang, Max and Gupta, Mohit},
    title   = {{visionsim}},
    url     = {https://github.com/WISION-Lab/visionsim},
    year    = {2025}
}
```
