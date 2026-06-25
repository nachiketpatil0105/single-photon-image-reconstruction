import os
import csv
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

os.makedirs("results",     exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# Dataset path — update if your environment differs
DATA_ROOT = "/kaggle/input/datasets/nachiketpatil0105/spc-data/train/"

# Training config
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 150
GRAD_CLIP_NORM = 1.0
PATCH_SIZE     = 512    # random crop size during training
AUG_PROB       = 0.5

# Loss weights
W_CHARB  = 0.25
W_MSSSIM = 0.50
W_VGG    = 0.25

LOG_CSV = "results/training_log.csv"


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def unpack_spc(npy_path):
    """
    Load the last 128 frames from an SPC .npy file and return a
    (384, 800, 800) float32 array ready for the model.

    SPC files are bit-packed with shape (1024, H, W, 100, 3).
    We take the last 128 frames, unpack the bits to recover the
    temporal bins, then flatten frame × channel into 384 input channels.
    A small epsilon in the denominator avoids division by zero on
    completely dark frames.
    """
    data = np.load(npy_path, mmap_mode="r")[-128:]  # (128, 800, 100, 3)
    data = np.unpackbits(data, axis=2)               # (128, 800, 800, 3)
    data = data.reshape(128, 800, 800, 3)
    data = np.transpose(data, (1, 2, 0, 3))          # (800, 800, 128, 3)
    data = data.reshape(800, 800, 384)               # (800, 800, 384)
    data = data / (data.max() + 1e-8)
    return data.transpose(2, 0, 1).astype(np.float32)  # (384, 800, 800)


class SPCDataset(Dataset):
    """
    PyTorch Dataset for SPC reconstruction.

    Training samples are augmented with random flips, rotations, and a
    random 512×512 crop to increase variety without collecting more data.
    Validation and test samples are served at full 800×800 resolution
    with no augmentation so that metrics are comparable across runs.
    """

    def __init__(self, npy_paths, png_paths, augment=False):
        self.npy_paths = npy_paths
        self.png_paths = png_paths
        self.augment   = augment

    def __len__(self):
        return len(self.npy_paths)

    def _augment(self, x, img):
        # Horizontal flip
        if random.random() < AUG_PROB:
            x   = np.flip(x,   axis=2).copy()
            img = np.flip(img, axis=2).copy()
        # Vertical flip
        if random.random() < AUG_PROB:
            x   = np.flip(x,   axis=1).copy()
            img = np.flip(img, axis=1).copy()
        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            x   = np.rot90(x,   k, axes=(1, 2)).copy()
            img = np.rot90(img, k, axes=(1, 2)).copy()
        return x, img

    def _crop(self, x, img):
        _, H, W = x.shape
        top  = random.randint(0, H - PATCH_SIZE)
        left = random.randint(0, W - PATCH_SIZE)
        return (x[:,   top:top+PATCH_SIZE, left:left+PATCH_SIZE],
                img[:, top:top+PATCH_SIZE, left:left+PATCH_SIZE])

    def __getitem__(self, idx):
        x   = unpack_spc(self.npy_paths[idx])
        img = np.array(Image.open(self.png_paths[idx])).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (3, 800, 800)

        if self.augment:
            x, img = self._augment(x, img)
            x, img = self._crop(x, img)

        return torch.from_numpy(x).float(), torch.from_numpy(img).float()


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss: sqrt((pred - target)^2 + eps^2).

    A smooth alternative to L1 — differentiable everywhere including at
    zero, which makes gradients more stable near convergence. Less
    sensitive to outlier pixels than plain L2 (MSE).
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using intermediate VGG16 feature maps.

    Rather than comparing pixels directly, this loss compares how the
    prediction and ground truth look to a pretrained feature extractor.
    We use three layers (relu1_2, relu2_2, relu3_3) weighted so that
    deeper, more structural features contribute more to the loss.
    VGG weights are frozen — this is not trained.
    """

    LAYER_WEIGHTS = {"relu1_2": 0.2, "relu2_2": 0.3, "relu3_3": 0.5}

    def __init__(self):
        super().__init__()
        vgg      = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        self.slice1 = nn.Sequential(*features[:5]).eval()    # → relu1_2
        self.slice2 = nn.Sequential(*features[5:10]).eval()  # → relu2_2
        self.slice3 = nn.Sequential(*features[10:17]).eval() # → relu3_3

        for p in self.parameters():
            p.requires_grad = False

        self.l1 = nn.L1Loss()
        self.w  = self.LAYER_WEIGHTS

    def forward(self, pred, target):
        f1_p = self.slice1(pred);    f1_t = self.slice1(target)
        f2_p = self.slice2(f1_p);   f2_t = self.slice2(f1_t)
        f3_p = self.slice3(f2_p);   f3_t = self.slice3(f2_t)

        return (self.w["relu1_2"] * self.l1(f1_p, f1_t) +
                self.w["relu2_2"] * self.l1(f2_p, f2_t) +
                self.w["relu3_3"] * self.l1(f3_p, f3_t))


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class UNetBasic(nn.Module):
    """
    UNet encoder-decoder for single-photon image reconstruction.

    The encoder progressively halves spatial resolution while doubling
    channels, building a compact representation of the scene. The decoder
    mirrors this in reverse, but at each level it receives the corresponding
    encoder feature map via a skip connection (torch.cat). This lets the
    decoder combine high-level semantic context from the bottleneck with
    low-level spatial detail from the encoder — the key advantage over
    the flat CNN in Phase 2.

    All upsampling is done with learned ConvTranspose2d rather than
    bilinear interpolation, giving the model more control over how it
    fills in spatial detail during reconstruction.

    Channel progression:
        Encoder : 384 → 128 → 256 → 512 → 1024 (bottleneck)
        Decoder : 1024 → 512 → 256 → 128 → 3 (output)
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = self._block(384, 128)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self._block(128, 256)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self._block(256, 512)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(512, 1024)

        # Decoder
        self.up1  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = self._block(512 + 512, 512)

        self.up2  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self._block(256 + 256, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(128 + 128, 128)

        # Output — 1×1 conv to collapse to 3 channels (RGB)
        self.out = nn.Conv2d(128, 3, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x);  p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u1 = self.up1(b);  d1 = self.dec1(torch.cat([u1, e3], dim=1))
        u2 = self.up2(d1); d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u3 = self.up3(d2); d3 = self.dec3(torch.cat([u3, e1], dim=1))

        return self.out(d3)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, charb, msssim_fn,
                    vgg_fn, ssim_fn, device):
    model.train()
    total_loss = total_ssim = total_msssim = total_psnr = 0.0

    for x, target in loader:
        x, target = x.to(device), target.to(device)
        optimizer.zero_grad()

        with autocast("cuda"):
            pred         = model(x)
            pred_c       = torch.clamp(pred, 0.0, 1.0)
            charb_l      = charb(pred, target)
            msssim_score = msssim_fn(pred_c, target)
            vgg_l        = vgg_fn(pred_c, target)
            loss = W_CHARB * charb_l + W_MSSSIM * (1 - msssim_score) + W_VGG * vgg_l

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            total_ssim   += ssim_fn(pred_c, target).item()
            total_psnr   += compute_psnr(pred_c, target)
            total_msssim += msssim_score.item()
        total_loss += loss.item()

    n = len(loader)
    return total_loss/n, total_ssim/n, total_msssim/n, total_psnr/n


def validate(model, loader, charb, msssim_fn, vgg_fn, ssim_fn, device):
    model.eval()
    total_loss = total_ssim = total_msssim = total_psnr = 0.0

    with torch.no_grad():
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            with autocast("cuda"):
                pred         = model(x)
                pred_c       = torch.clamp(pred, 0.0, 1.0)
                charb_l      = charb(pred, target)
                msssim_score = msssim_fn(pred_c, target)
                vgg_l        = vgg_fn(pred_c, target)
                loss = W_CHARB * charb_l + W_MSSSIM * (1 - msssim_score) + W_VGG * vgg_l

            total_loss   += loss.item()
            total_ssim   += ssim_fn(pred_c, target).item()
            total_msssim += msssim_score.item()
            total_psnr   += compute_psnr(pred_c, target)

    n = max(len(loader), 1)
    return total_loss/n, total_ssim/n, total_msssim/n, total_psnr/n


# ------------------------------------------------------------------
# Evaluation & plots
# ------------------------------------------------------------------

def evaluate_test(model, test_loader, test_paths, device):
    """
    Run inference on the test set using the best saved checkpoint.
    Computes PSNR and SSIM with scikit-image (standard for reporting),
    saves a 3-panel comparison figure per sample, and returns all results.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for i, (x, target) in enumerate(test_loader):
            x, target = x.to(device), target.to(device)
            with autocast("cuda"):
                pred = torch.clamp(model(x), 0.0, 1.0)

            pred_np   = pred.squeeze().cpu().float().numpy().transpose(1, 2, 0)
            target_np = target.squeeze().cpu().float().numpy().transpose(1, 2, 0)
            input_np  = x.squeeze().cpu().float().numpy()[:3].transpose(1, 2, 0)

            p = psnr_sk(target_np, pred_np, data_range=1.0)
            s = ssim_sk(target_np, pred_np, channel_axis=2, data_range=1.0)

            scene = Path(test_paths[i]).parent.name
            results.append({"sample": i, "scene": scene, "PSNR": float(p), "SSIM": float(s)})
            print(f"  Sample {i} ({scene})  PSNR={p:.2f} dB  SSIM={s:.4f}")

            save_comparison(input_np, pred_np, target_np, i, scene, p, s)

    return results


def save_comparison(input_img, pred_img, target_img, idx, scene_name, p, s):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"UNet Reconstruction — {scene_name}", fontsize=13, fontweight="bold")

    for ax, (img, title, sub) in zip(axes, [
        (input_img,  "SPC Input (first 3 ch)", ""),
        (pred_img,   "UNet Prediction",         f"PSNR={p:.2f} dB  SSIM={s:.4f}"),
        (target_img, "Ground Truth",            ""),
    ]):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(sub, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    path = f"results/comparison_sample{idx}_{scene_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


def save_training_curves(log_csv):
    epochs, t_loss, v_loss = [], [], []
    t_ssim, v_ssim         = [], []
    t_psnr, v_psnr         = [], []

    with open(log_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            t_loss.append(float(row["train_loss"]));  v_loss.append(float(row["val_loss"]))
            t_ssim.append(float(row["train_ssim"]));  v_ssim.append(float(row["val_ssim"]))
            t_psnr.append(float(row["train_psnr"]));  v_psnr.append(float(row["val_psnr"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("UNet — Training & Validation Curves", fontsize=14, fontweight="bold")

    for ax, (t, v, title) in zip(axes, [
        (t_loss, v_loss, "Loss"),
        (t_ssim, v_ssim, "SSIM"),
        (t_psnr, v_psnr, "PSNR (dB)"),
    ]):
        ax.plot(epochs, t, label="Train")
        ax.plot(epochs, v, label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "results/training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def print_summary(results):
    print(f"\n{'Sample':<8} {'Scene':<20} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("─" * 50)
    for r in results:
        print(f"{r['sample']:<8} {r['scene']:<20} {r['PSNR']:>10.2f} {r['SSIM']:>8.4f}")
    avg_p = sum(r["PSNR"] for r in results) / len(results)
    avg_s = sum(r["SSIM"] for r in results) / len(results)
    print("─" * 50)
    print(f"{'Average':<8} {'':<20} {avg_p:>10.2f} {avg_s:>8.4f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Collect paths and split 80/10/10
    path      = Path(DATA_ROOT)
    npy_files = sorted(path.rglob("*.npy"))
    png_files = sorted(path.rglob("*.png"))
    n         = len(npy_files)
    n_test    = max(1, int(0.10 * n))
    n_val     = max(1, int(0.10 * n))
    n_train   = n - n_val - n_test

    train_X, train_y = npy_files[:n_train],             png_files[:n_train]
    val_X,   val_y   = npy_files[n_train:n_train+n_val], png_files[n_train:n_train+n_val]
    test_X,  test_y  = npy_files[n_train+n_val:],        png_files[n_train+n_val:]

    print(f"Train: {len(train_X)}  |  Val: {len(val_X)}  |  Test: {len(test_X)}\n")

    train_loader = DataLoader(SPCDataset(train_X, train_y, augment=True),
                              batch_size=1, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(SPCDataset(val_X,   val_y),
                              batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(SPCDataset(test_X,  test_y),
                              batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Model + optimiser + scheduler
    model     = UNetBasic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler    = GradScaler("cuda")

    # Losses and metrics
    charb_fn  = CharbonnierLoss(eps=1e-3)
    vgg_fn    = VGGPerceptualLoss().to(device)
    ssim_fn   = SSIM(data_range=1.0).to(device)
    msssim_fn = MSSSIM(data_range=1.0, kernel_size=11).to(device)

    # CSV header
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr",
            "train_loss", "train_ssim", "train_msssim", "train_psnr",
            "val_loss",   "val_ssim",   "val_msssim",   "val_psnr",
        ])

    best_val_ssim = -1.0

    print("Training...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        t_loss, t_ssim, t_msssim, t_psnr = train_one_epoch(
            model, train_loader, optimizer, scaler,
            charb_fn, msssim_fn, vgg_fn, ssim_fn, device
        )
        v_loss, v_ssim, v_msssim, v_psnr = validate(
            model, val_loader, charb_fn, msssim_fn, vgg_fn, ssim_fn, device
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  LR: {lr:.2e}  "
              f"| Train  Loss: {t_loss:.4f}  SSIM: {t_ssim:.4f}  PSNR: {t_psnr:.2f} dB  "
              f"| Val    Loss: {v_loss:.4f}  SSIM: {v_ssim:.4f}  PSNR: {v_psnr:.2f} dB")

        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.2e}",
                f"{t_loss:.4f}", f"{t_ssim:.4f}", f"{t_msssim:.4f}", f"{t_psnr:.2f}",
                f"{v_loss:.4f}", f"{v_ssim:.4f}", f"{v_msssim:.4f}", f"{v_psnr:.2f}",
            ])

        if v_ssim > best_val_ssim:
            best_val_ssim = v_ssim
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_ssim":             v_ssim,
            }, "checkpoints/best_model.pth")
            print(f"  ✓ New best val SSIM: {best_val_ssim:.4f} — saved")

    print(f"\nTraining complete. Best val SSIM: {best_val_ssim:.4f}")

    # Plot training curves
    save_training_curves(LOG_CSV)

    # Load best model and evaluate on test set
    print("\nLoading best model for test evaluation...")
    ckpt = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded from epoch {ckpt['epoch']}  (val SSIM {ckpt['val_ssim']:.4f})\n")

    print("Evaluating on test set...")
    results = evaluate_test(model, test_loader, test_y, device)
    print_summary(results)

    avg_psnr = sum(r["PSNR"] for r in results) / len(results)
    avg_ssim = sum(r["SSIM"] for r in results) / len(results)

    with open("results/metrics.json", "w") as f:
        json.dump({"test_results": results, "avg_psnr": avg_psnr, "avg_ssim": avg_ssim}, f, indent=2)
    print("\nMetrics saved → results/metrics.json")
