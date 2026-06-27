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

os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Update to match your dataset location
DATA_ROOT = "/kaggle/input/datasets/nachiketpatil0105/spc-data/train/"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 150
GRAD_CLIP_NORM = 1.0
PATCH_SIZE = 512
AUG_PROB = 0.5

# Loss weights
W_CHARB = 0.25
W_MSSSIM = 0.5
W_VGG = 0.25

LOG_CSV = "results/training_log.csv"


def unpack_last_frame(npy_path):
    ld = np.load(npy_path, mmap_mode="r")[-128:]  # (128,800,100,3)
    ld = np.unpackbits(ld, axis=2)                 # (128,800,800,3)
    ld = ld.reshape(128, 800, 800, 3)
    ld = np.transpose(ld, (1, 2, 0, 3))            # (800,800,128,3)
    ld = ld.reshape(800, 800, 384)                  # (800,800,384)
    ld = ld / (ld.max() + 1e-8)
    return ld.astype(np.float32)


class SPCDataset(Dataset):
    """augment=True for training (flip + rotate + crop), False for val/test (full 800×800)."""

    def __init__(self, npy_paths, png_paths, augment=False, patch_size=None):
        self.npy_paths = npy_paths
        self.png_paths = png_paths
        self.augment = augment
        self.patch_size = patch_size

    def __len__(self):
        return len(self.npy_paths)

    def _apply_augmentation(self, x_np, img_np):
        if random.random() < AUG_PROB:
            x_np = np.flip(x_np, axis=2).copy()
            img_np = np.flip(img_np, axis=2).copy()
        if random.random() < AUG_PROB:
            x_np = np.flip(x_np, axis=1).copy()
            img_np = np.flip(img_np, axis=1).copy()
        k = random.randint(0, 3)
        if k > 0:
            x_np = np.rot90(x_np, k, axes=(1, 2)).copy()
            img_np = np.rot90(img_np, k, axes=(1, 2)).copy()
        return x_np, img_np

    def _random_crop(self, x_np, img_np, size):
        _, H, W = x_np.shape
        top = random.randint(0, H - size)
        left = random.randint(0, W - size)
        return x_np[:, top:top+size, left:left+size], img_np[:, top:top+size, left:left+size]

    def __getitem__(self, idx):
        x_np = unpack_last_frame(self.npy_paths[idx])
        x_np = x_np.transpose(2, 0, 1)                               # (384,800,800)
        img_np = np.array(Image.open(self.png_paths[idx])).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)                           # (3,800,800)
        if self.augment:
            x_np, img_np = self._apply_augmentation(x_np, img_np)
            if self.patch_size is not None:
                x_np, img_np = self._random_crop(x_np, img_np, self.patch_size)
        return torch.from_numpy(x_np).float(), torch.from_numpy(img_np).float()


class VGGPerceptualLoss(nn.Module):
    """Multi-layer perceptual loss using VGG16 at relu1_2, relu2_2, relu3_3."""

    LAYER_WEIGHTS = {"relu1_2": 0.2, "relu2_2": 0.3, "relu3_3": 0.5}

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())
        self.slice1 = nn.Sequential(*features[:5]).eval()
        self.slice2 = nn.Sequential(*features[5:10]).eval()
        self.slice3 = nn.Sequential(*features[10:17]).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.l1 = nn.L1Loss()
        self.w = self.LAYER_WEIGHTS

    def forward(self, pred, target):
        f1_p = self.slice1(pred); f1_t = self.slice1(target)
        f2_p = self.slice2(f1_p); f2_t = self.slice2(f1_t)
        f3_p = self.slice3(f2_p); f3_t = self.slice3(f2_t)
        return (self.w["relu1_2"] * self.l1(f1_p, f1_t) +
                self.w["relu2_2"] * self.l1(f2_p, f2_t) +
                self.w["relu3_3"] * self.l1(f3_p, f3_t))


class CharbonnierLoss(nn.Module):
    """Smooth L1 replacement: sqrt((x-y)^2 + eps^2). Differentiable everywhere."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


class UNetBasic(nn.Module):
    """UNet encoder-decoder with skip connections for SPC reconstruction."""

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
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = self._block(512+512, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = self._block(256+256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._block(128+128, 128)
        self.out = nn.Conv2d(128, 3, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        u1 = self.up1(b); d1 = self.dec1(torch.cat([u1, e3], dim=1))
        u2 = self.up2(d1); d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u3 = self.up3(d2); d3 = self.dec3(torch.cat([u3, e1], dim=1))
        return self.out(d3)


def compute_psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def save_comparison(input_img, pred_img, target_img, idx, scene_name, p, s):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"UNet Reconstruction — {scene_name}", fontsize=13, fontweight="bold")
    for ax, (img, title, sub) in zip(axes, [
        (input_img[:, :, 0:3], "Input (SPC)", ""),
        (pred_img, "Model Output", f"PSNR={p:.2f} dB  SSIM={s:.4f}"),
        (target_img, "Ground Truth", ""),
    ]):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(sub, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    path = f"results/comparison_sample{idx}_{scene_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


def save_training_curves(log_csv):
    epochs_log = []
    train_losses, val_losses = [], []
    train_ssims, val_ssims = [], []
    train_psnrs, val_psnrs = [], []
    with open(log_csv) as f:
        for row in csv.DictReader(f):
            epochs_log.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            train_ssims.append(float(row["train_ssim"]))
            val_ssims.append(float(row["val_ssim"]))
            train_psnrs.append(float(row["train_psnr"]))
            val_psnrs.append(float(row["val_psnr"]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training & Validation Curves", fontsize=14, fontweight="bold")
    for ax, (t, v, title) in zip(axes, [
        (train_losses, val_losses, "Loss"),
        (train_ssims, val_ssims, "SSIM"),
        (train_psnrs, val_psnrs, "PSNR (dB)"),
    ]):
        ax.plot(epochs_log, t, label="Train")
        ax.plot(epochs_log, v, label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True)
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    path = Path(DATA_ROOT)
    npy_files = sorted(list(path.rglob("*.npy")))
    png_files = sorted(list(path.rglob("*.png")))

    n = len(npy_files)
    n_test = max(1, int(0.10 * n))
    n_val = max(1, int(0.10 * n))
    n_train = n - n_val - n_test

    train_X = npy_files[:n_train]
    train_y = png_files[:n_train]
    val_X = npy_files[n_train:n_train+n_val]
    val_y = png_files[n_train:n_train+n_val]
    test_X = npy_files[n_train+n_val:]
    test_y = png_files[n_train+n_val:]

    print(f"Train: {len(train_X)}  |  Val: {len(val_X)}  |  Test: {len(test_X)}\n")

    train_loader = DataLoader(
        SPCDataset(train_X, train_y, augment=True, patch_size=PATCH_SIZE),
        batch_size=1, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        SPCDataset(val_X, val_y, augment=False),
        batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        SPCDataset(test_X, test_y, augment=False),
        batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    model = UNetBasic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler("cuda")

    charb_loss = CharbonnierLoss(eps=1e-3)
    vgg_loss_fn = VGGPerceptualLoss().to(device)
    ssim_metric = SSIM(data_range=1.0).to(device)
    msssim_metric = MSSSIM(data_range=1.0, kernel_size=11).to(device)

    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr",
            "train_loss", "train_ssim", "train_msssim", "train_psnr",
            "val_loss", "val_ssim", "val_msssim", "val_psnr",
        ])

    best_val_ssim = -1.0

    print("Training...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        t_loss = t_ssim = t_msssim = t_psnr = 0.0
        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                prediction = model(x)
                pred_clamped = torch.clamp(prediction, 0.0, 1.0)
                charb = charb_loss(prediction, target)
                msssim_score = msssim_metric(pred_clamped, target)
                msssim_l = 1.0 - msssim_score
                vgg = vgg_loss_fn(pred_clamped, target)
                loss = W_CHARB * charb + W_MSSSIM * msssim_l + W_VGG * vgg
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                ssim_score = ssim_metric(pred_clamped, target)
                psnr_score = compute_psnr(pred_clamped, target)
            t_loss += loss.item()
            t_ssim += ssim_score.item()
            t_msssim += msssim_score.item()
            t_psnr += psnr_score

        n_tr = len(train_loader)
        t_loss /= n_tr; t_ssim /= n_tr
        t_msssim /= n_tr; t_psnr /= n_tr

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Validation
        model.eval()
        v_loss = v_ssim = v_msssim = v_psnr = 0.0
        with torch.no_grad():
            for x_v, target_v in val_loader:
                x_v = x_v.to(device)
                target_v = target_v.to(device)
                with autocast("cuda"):
                    pred_v = model(x_v)
                    pred_v_c = torch.clamp(pred_v, 0.0, 1.0)
                    charb_v = charb_loss(pred_v, target_v)
                    msssim_v = msssim_metric(pred_v_c, target_v)
                    msssim_lv = 1.0 - msssim_v
                    vgg_v = vgg_loss_fn(pred_v_c, target_v)
                    loss_v = W_CHARB * charb_v + W_MSSSIM * msssim_lv + W_VGG * vgg_v
                ssim_v = ssim_metric(pred_v_c, target_v)
                psnr_v = compute_psnr(pred_v_c, target_v)
                v_loss += loss_v.item()
                v_ssim += ssim_v.item()
                v_msssim += msssim_v.item()
                v_psnr += psnr_v

        n_vl = max(len(val_loader), 1)
        v_loss /= n_vl; v_ssim /= n_vl
        v_msssim /= n_vl; v_psnr /= n_vl

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  LR: {current_lr:.2e}  "
            f"| Train  Loss: {t_loss:.4f}  SSIM: {t_ssim:.4f}  MS-SSIM: {t_msssim:.4f}  PSNR: {t_psnr:.2f} dB  "
            f"| Val    Loss: {v_loss:.4f}  SSIM: {v_ssim:.4f}  MS-SSIM: {v_msssim:.4f}  PSNR: {v_psnr:.2f} dB"
        )

        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{current_lr:.2e}",
                f"{t_loss:.4f}", f"{t_ssim:.4f}", f"{t_msssim:.4f}", f"{t_psnr:.2f}",
                f"{v_loss:.4f}", f"{v_ssim:.4f}", f"{v_msssim:.4f}", f"{v_psnr:.2f}",
            ])

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": t_loss,
            "val_ssim": v_ssim,
        }
        if v_ssim > best_val_ssim:
            best_val_ssim = v_ssim
            torch.save(ckpt, "checkpoints/best_model.pth")
            print(f"  ✓ New best val SSIM: {best_val_ssim:.4f} — saved best_model.pth")

        # Save test images every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                for i, (x_t, target_t) in enumerate(test_loader):
                    x_t = x_t.to(device)
                    target_t = target_t.to(device)
                    with autocast("cuda"):
                        pred_t = model(x_t)
                    pred_t = torch.clamp(pred_t, 0.0, 1.0)
                    pred_np = pred_t.squeeze().cpu().float().numpy().transpose(1, 2, 0)
                    target_np = target_t.squeeze().cpu().float().numpy().transpose(1, 2, 0)
                    input_np = x_t.squeeze().cpu().float().numpy().transpose(1, 2, 0)
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(input_np[:, :, 0:3]); axes[0].set_title("Input (SPC)"); axes[0].axis("off")
                    axes[1].imshow(pred_np); axes[1].set_title("Model Output"); axes[1].axis("off")
                    axes[2].imshow(target_np); axes[2].set_title("Ground Truth"); axes[2].axis("off")
                    plt.suptitle(f"Epoch {epoch} — Test sample {i}")
                    plt.tight_layout()
                    plt.savefig(f"results/epoch_{epoch:03d}_sample_{i}.png", dpi=150)
                    plt.close()
            print(f"  → Saved test images for epoch {epoch}")

    print(f"\nTraining complete. Best val SSIM: {best_val_ssim:.4f}")
    save_training_curves(LOG_CSV)

    # Final evaluation on best checkpoint
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded from epoch {checkpoint['epoch']}  (val SSIM {checkpoint['val_ssim']:.4f})\n")

    model.eval()
    results = []
    total_ssim = total_msssim = total_psnr = 0.0
    with torch.no_grad():
        for i, (x_t, target_t) in enumerate(test_loader):
            x_t = x_t.to(device)
            target_t = target_t.to(device)
            with autocast("cuda"):
                pred_t = model(x_t)
            pred_t = torch.clamp(pred_t, 0.0, 1.0)

            ssim_s = ssim_metric(pred_t, target_t).item()
            msssim_s = msssim_metric(pred_t, target_t).item()
            psnr_s = compute_psnr(pred_t, target_t)

            total_ssim += ssim_s
            total_msssim += msssim_s
            total_psnr += psnr_s

            pred_np = pred_t.squeeze().cpu().float().numpy().transpose(1, 2, 0)
            target_np = target_t.squeeze().cpu().float().numpy().transpose(1, 2, 0)
            input_np = x_t.squeeze().cpu().float().numpy().transpose(1, 2, 0)

            p = psnr_sk(target_np, pred_np, data_range=1.0)
            s = ssim_sk(target_np, pred_np, channel_axis=2, data_range=1.0)
            scene = Path(test_y[i]).parent.name
            results.append({"sample": i, "scene": scene, "PSNR": float(p), "SSIM": float(s)})
            print(f"Test sample {i} ({scene})  SSIM: {ssim_s:.4f}  MS-SSIM: {msssim_s:.4f}  PSNR: {psnr_s:.2f} dB")

            save_comparison(input_np, pred_np, target_np, i, scene, p, s)

    n = len(test_loader)
    print(f"\nAverage SSIM   : {total_ssim / n:.4f}")
    print(f"Average MS-SSIM: {total_msssim / n:.4f}")
    print(f"Average PSNR   : {total_psnr / n:.2f} dB")

    print_summary(results)

    with open("results/metrics.json", "w") as f:
        json.dump({"test_results": results,
                   "avg_psnr": total_psnr / n,
                   "avg_ssim": total_ssim / n}, f, indent=2)
    print("\nMetrics saved → results/metrics.json")