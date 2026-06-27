import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

os.makedirs("results", exist_ok=True)

# Update to match your dataset location
DATA_ROOT = "/kaggle/input/datasets/nachiketpatil0105/spc-data/train/"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NUM_FRAMES = 128


def unpack_last_frame(npy_path):
    # Load last 128 frames and reshape into (384, 800, 800) channel-first tensor
    ld = np.load(npy_path, mmap_mode="r")[-NUM_FRAMES:]  # (128,800,100,3)
    ld = np.unpackbits(ld, axis=2)                        # (128,800,800,3)
    ld = ld.reshape(NUM_FRAMES, 800, 800, 3)
    ld = np.transpose(ld, (1, 2, 0, 3))                  # (800,800,128,3)
    ld = ld.reshape(800, 800, NUM_FRAMES * 3)             # (800,800,384)
    ld = ld / ld.max()
    return ld.astype(np.float32)


class SimCNN(nn.Module):
    def __init__(self):
        super(SimCNN, self).__init__()
        # 384 input channels (128 frames × 3 RGB), contracts to 3 output channels
        self.net = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def save_comparison(input_img, pred_img, target_img, idx, scene_name, p, s):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"CNN Reconstruction — {scene_name}", fontsize=13, fontweight="bold")
    panels = [
        (input_img[:, :, 0:3], "Input (SPC)", ""),
        (pred_img, "Model Output", f"PSNR={p:.2f} dB  SSIM={s:.4f}"),
        (target_img, "Ground Truth", ""),
    ]
    for ax, (img, title, sub) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(sub, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    path = f"results/comparison_sample{idx}_{scene_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


def save_loss_curve(epoch_losses):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (0.5 × L1 + 0.5 × (1 - SSIM))")
    ax.set_title("Training Loss Curve — Baseline CNN", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "results/loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def print_summary(results):
    print(f"\n{'Sample':<8} {'Scene':<14} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("─" * 45)
    for r in results:
        print(f"{r['sample']:<8} {r['scene']:<14} {r['PSNR']:>10.2f} {r['SSIM']:>8.4f}")
    avg_psnr = sum(r["PSNR"] for r in results) / len(results)
    avg_ssim = sum(r["SSIM"] for r in results) / len(results)
    print("─" * 45)
    print(f"{'Average':<8} {'':<14} {avg_psnr:>10.2f} {avg_ssim:>8.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    path = Path(DATA_ROOT)
    npy_files = sorted(list(path.rglob("*.npy")))
    png_files = sorted(list(path.rglob("*.png")))

    train_X = npy_files[:45]
    train_y = png_files[:45]
    test_X = npy_files[-5:]
    test_y = png_files[-5:]

    print(f"Train: {len(train_X)} samples  |  Test: {len(test_X)} samples\n")

    model = SimCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ssim_fn = SSIM(data_range=1.0).to(device)
    l1_fn = nn.L1Loss()

    # Training
    model.train()
    epoch_losses = []
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for npy_path, png_path in zip(train_X, train_y):
            x = unpack_last_frame(npy_path)
            x = x.transpose(2, 0, 1)                      # (384,800,800)
            x = torch.from_numpy(x).float().unsqueeze(0).to(device)

            img = np.array(Image.open(png_path)) / 255.0
            img = img.transpose(2, 0, 1)
            target = torch.from_numpy(img).float().unsqueeze(0).to(device)

            prediction = model(x)
            ssim_loss = 1 - ssim_fn(prediction, target)
            l1 = l1_fn(prediction, target)
            loss = 0.5 * ssim_loss + 0.5 * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_X)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Loss: {epoch_loss:.4f}")

    save_loss_curve(epoch_losses)
    torch.save(model.state_dict(), "results/cnn_model.pth")
    print("\nModel saved → results/cnn_model.pth")

    # Evaluation
    model.eval()
    results = []
    with torch.no_grad():
        for i, (npy_path, png_path) in enumerate(zip(test_X, test_y)):
            x = unpack_last_frame(npy_path)
            x = x.transpose(2, 0, 1)
            x = torch.from_numpy(x).float().unsqueeze(0).to(device)

            img = np.array(Image.open(png_path)) / 255.0
            img = img.transpose(2, 0, 1)
            target = torch.from_numpy(img).float().unsqueeze(0).to(device)

            prediction = model(x)

            pred_np = np.clip(prediction.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)
            target_np = target.squeeze().cpu().numpy().transpose(1, 2, 0)
            input_np = x.squeeze().cpu().numpy().transpose(1, 2, 0)

            p = psnr_metric(target_np, pred_np, data_range=1.0)
            s = ssim_metric(target_np, pred_np, channel_axis=2, data_range=1.0)

            scene = Path(png_path).parent.name
            results.append({"sample": i, "scene": scene, "PSNR": float(p), "SSIM": float(s)})
            print(f"Test sample {i} ({scene})  PSNR={p:.2f} dB  SSIM={s:.4f}")

            save_comparison(input_np, pred_np, target_np, i, scene, p, s)

    print_summary(results)

    with open("results/metrics.json", "w") as f:
        json.dump({
            "test_results": results,
            "epoch_losses": epoch_losses,
            "avg_psnr": sum(r["PSNR"] for r in results) / len(results),
            "avg_ssim": sum(r["SSIM"] for r in results) / len(results),
        }, f, indent=2)
    print("\nMetrics saved → results/metrics.json")