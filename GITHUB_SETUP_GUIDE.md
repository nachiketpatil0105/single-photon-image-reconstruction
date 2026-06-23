# GitHub Setup Guide

A step-by-step checklist to publish this documentation to GitHub.

---

## Step 1 — Create the GitHub Repository

1. Go to https://github.com/new
2. Repository name: `single-photon-reconstruction`
3. Set to **Public** (so recruiters can view it)
4. Do NOT initialize with a README (we have our own)
5. Click **Create repository**

---

## Step 2 — Upload This Folder Structure

Option A — Via GitHub web UI (drag & drop):
- Open the repo → click **Add file** → **Upload files**
- Drag the entire folder contents in

Option B — Via Git CLI:
```bash
cd single-photon-reconstruction
git init
git add .
git commit -m "Initial documentation commit"
git branch -M main
git remote add origin https://github.com/nachiketpatil0105/single-photon-reconstruction.git
git push -u origin main
```

---

## Step 3 — Add Your Code Files

For each phase folder, add:
- `model.py` — your model architecture
- `train.py` — training script
- `results/` — output images (PNG/JPG)

---

## Step 4 — Fill in the Blanks

Search for `—` across all README files and replace with your actual values:
- PSNR and SSIM scores for each phase
- Training config (loss, optimizer, learning rate, epochs, batch size)
- Result images in each `results/` folder

---

## Step 5 — Add a Results Banner (Optional but Impactful)

In the main `README.md`, consider adding a side-by-side image grid showing:
- Raw noisy input
- Naive output
- CNN output
- UNet output
- ResUNet+Attention output
- Ground truth

This single visual will be the first thing a recruiter or engineer sees.

---

## Step 6 — Link on Your Resume

Add this to your resume's Projects section:

> **Single Photon Image Reconstruction** | [github.com/nachiketpatil0105/single-photon-reconstruction](https://github.com/nachiketpatil0105/single-photon-reconstruction)  
> Developed a series of deep learning models (Naive → CNN → UNet → ResUNet+Attention) for reconstructing high-fidelity images from binary single-photon camera data. Achieved [X dB PSNR / X SSIM] on the Single Photon Challenge benchmark.

---

## Folder Checklist

- [ ] `README.md` (main) — fill in PSNR/SSIM table
- [ ] `phase1_naive/README.md` — add results + images
- [ ] `phase1_naive/naive.py` — upload your code
- [ ] `phase2_baseline_cnn/README.md` — add results + images
- [ ] `phase2_baseline_cnn/model.py` + `train.py` — upload code
- [ ] `phase3_unet/README.md` — add results + images
- [ ] `phase3_unet/model.py` + `train.py` — upload code
- [ ] `phase4_resunet_attention/README.md` — add results + images
- [ ] `phase4_resunet_attention/model.py` + `train.py` — upload code
- [ ] `requirements.txt` — verify packages match your environment
