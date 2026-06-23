# 📷 Single Photon Image Reconstruction

> Reconstructing high-quality images from noisy, binary single-photon camera data — progressing from a naive baseline to a ResUNet with Attention Gates.

[![Dataset](https://img.shields.io/badge/Dataset-Single%20Photon%20Challenge-blue?style=flat-square)](https://singlephotonchallenge.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=flat-square)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square)](https://pytorch.org/)
[![Author](https://img.shields.io/badge/Author-nachiketpatil0105-black?style=flat-square&logo=github)](https://github.com/nachiketpatil0105)

---

## 🧠 Problem Statement

Single-photon cameras can detect individual photons, making them incredibly sensitive — but each captured frame is binary and dominated by photon shot noise. Given a burst of such noisy binary frames, the goal is to recover a **clean, high-fidelity image**.

This project explores a series of deep learning architectures to solve this reconstruction problem, with each phase building on lessons from the previous one.

---

## 🚀 Project Phases at a Glance

| Phase | Model | Description | PSNR | SSIM |
|-------|-------|-------------|------|------|
| 1 | **Naive** | Simple averaging, no learning | — | — |
| 2 | **Baseline CNN** | Basic convolutional network | — | — |
| 3 | **UNet** | Encoder-decoder with skip connections | — | — |
| 4 | **ResUNet + Attention** | Residual blocks + attention gates | — | — |

> 📌 Fill in your PSNR/SSIM values above — each phase folder has its own results section.

---

## 📁 Repository Structure

```
single-photon-reconstruction/
│
├── phase1_naive/               # Naive averaging baseline
│   ├── naive.py
│   ├── results/
│   └── README.md
│
├── phase2_baseline_cnn/        # Basic CNN
│   ├── model.py
│   ├── train.py
│   ├── results/
│   └── README.md
│
├── phase3_unet/                # UNet architecture
│   ├── model.py
│   ├── train.py
│   ├── results/
│   └── README.md
│
├── phase4_resunet_attention/   # ResUNet with Attention Gates
│   ├── model.py
│   ├── train.py
│   ├── results/
│   └── README.md
│
└── README.md                   # You are here
```

---

## 📊 Dataset

**Source:** [The Single Photon Challenge](https://singlephotonchallenge.com/)

A large synthetic dataset consisting of:
- Bursts of binary single-photon camera images (input)
- Corresponding ground truth high-fidelity images (target)

The dataset was created by researchers from University of Wisconsin–Madison, Portland State University, Purdue University, Carnegie Mellon University, and the US Naval Research Lab.

---

## 🔬 Phase Summaries

### Phase 1 — Naive Averaging
No machine learning involved. Frames in a burst are averaged together to reduce shot noise. This establishes a lower-bound baseline for comparison.
→ [View Phase 1](./phase1_naive/README.md)

### Phase 2 — Baseline CNN
A straightforward convolutional neural network trained end-to-end on burst input → clean image output. Establishes a learned baseline.
→ [View Phase 2](./phase2_baseline_cnn/README.md)

### Phase 3 — UNet
Adopts the classic UNet encoder-decoder architecture with skip connections, enabling finer spatial reconstruction by preserving low-level features.
→ [View Phase 3](./phase3_unet/README.md)

### Phase 4 — ResUNet with Attention Gates
Combines residual connections with attention gates to suppress irrelevant features and focus reconstruction on salient image regions. Best performing model.
→ [View Phase 4](./phase4_resunet_attention/README.md)

---

## ⚙️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/nachiketpatil0105/single-photon-reconstruction.git
cd single-photon-reconstruction

# Install dependencies
pip install -r requirements.txt

# Run a specific phase
cd phase4_resunet_attention
python train.py
```

---

## 📈 Results Progression

Each phase improved on the previous. See individual phase READMEs for:
- Architecture diagrams / model description
- Training configuration (loss, optimizer, epochs)
- Quantitative results (PSNR, SSIM)
- Visual comparison of reconstructed images

---

## 📜 Citation

If you use the Single Photon Challenge dataset, please cite:

```bibtex
@software{visionsim,
    author = {Jungerman, Sacha and Gupta, Shantanu and Sadekar, Kaustubh and Leblang, Max and Gupta, Mohit},
    license = {MIT},
    month = may,
    title = {{visionsim}},
    url = {https://github.com/WISION-Lab/visionsim},
    year = {2025}
}
```

---

## 👤 Author

**Nachiket Patil**
GitHub: [@nachiketpatil0105](https://github.com/nachiketpatil0105)
