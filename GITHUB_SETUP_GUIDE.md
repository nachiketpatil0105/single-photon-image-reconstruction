# Getting Started

## Clone and install

```bash
git clone https://github.com/<your-username>/single-photon-reconstruction.git
cd single-photon-reconstruction
pip install -r requirements.txt
```

## Running each phase

**Phase 1 — Naive Summation**
```bash
cd phase1_naive
# Edit SCENES paths at the top of the script
python naive_reconstruction.py
# Outputs → phase1_naive/results/
```

**Phase 2 — Baseline CNN**
```bash
cd phase2_baseline_cnn
# Edit DATA_ROOT at the top of the script
python cnn_reconstruction.py
# Outputs → phase2_baseline_cnn/results/
```

**Phase 3 — UNet**
```bash
cd phase3_unet
# Edit DATA_ROOT at the top of the script
python unet_reconstruction.py
# Outputs → phase3_unet/results/ and phase3_unet/checkpoints/
```

**Phase 4 — ResUNet + Attention**
```bash
cd phase4_resunet_attention
# Edit BASE_DIR, DATA_DIR_IMG, DATA_DIR_NPY at the top of the script
# Requires a high-VRAM GPU server (~40GB)
python resunet_reconstruction.py
# Outputs → output/ and checkpoints/
```

## Uploading to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/single-photon-reconstruction.git
git push -u origin main
```
