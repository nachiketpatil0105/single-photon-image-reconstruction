import numpy as np
import csv
import random
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from torch.utils.tensorboard import SummaryWriter

# Update these paths to match your environment
BASE_DIR = Path("/home/user/spc")
BASE_DIR_2 = Path("/home/user/spc2")
DATA_DIR_IMG = BASE_DIR / "Data_PT"
DATA_DIR_NPY = BASE_DIR / "Preprocessed_PT"
OUTPUT_DIR = BASE_DIR_2 / "output"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "tensorboard"
LOG_CSV = OUTPUT_DIR / "training_log.csv"
LOG_FILE = OUTPUT_DIR / "training.log"

for d in [CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(DEVICE)
torch.backends.cudnn.benchmark = True
log.info(f"Using device: {DEVICE}")

SEED = 42
PATCH_SIZE = 512
AUG_PROB = 0.2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 70
BATCH_SIZE = 2
NUM_WORKERS = 12
GRAD_CLIP_NORM = 1.0

# Loss weights
W_CHARB = 0.5
W_MSSSIM = 0.3
W_VGG = 0.1
W_EDGE = 0.1
# Deep supervision weight — starts at 0.4, annealed to 0 by end of training
W_DEEP_SUP = 0.4

COSINE_T0 = 5
ETA_MIN = 5e-5

random.seed(SEED)

# Collect paired .pt file paths from train_* folders
train_npy = []
train_png = []
for npy_folder, img_folder in zip(
    sorted(DATA_DIR_NPY.glob("train_*")),
    sorted(DATA_DIR_IMG.glob("train_*")),
):
    npys = sorted(npy_folder.rglob("*.pt"))
    pngs = sorted(img_folder.rglob("*.pt"))
    if len(npys) == 0:
        continue
    train_npy.extend(npys)
    train_png.extend(pngs)

log.info(f"Total training samples: {len(train_npy)}")


class SPCDataset(Dataset):
    """Loads preprocessed .pt tensors. Slices last 96 channels at load time."""

    def __init__(self, npy_paths, png_paths):
        self.npy_paths = npy_paths
        self.png_paths = png_paths

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        x = torch.load(self.npy_paths[idx], weights_only=True)
        target = torch.load(self.png_paths[idx], weights_only=True)
        x = x[-96:, :, :]
        return x, target


def gpu_augment(x, y):
    """Random flips and 90° rotations applied on GPU after device transfer."""
    if torch.rand(1).item() < AUG_PROB:
        x = torch.flip(x, dims=[3]); y = torch.flip(y, dims=[3])
    if torch.rand(1).item() < AUG_PROB:
        x = torch.flip(x, dims=[2]); y = torch.flip(y, dims=[2])
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        x = torch.rot90(x, k, dims=[2, 3])
        y = torch.rot90(y, k, dims=[2, 3])
    return x, y


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using frozen VGG16 features at relu1_2, relu2_2, relu3_3."""

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


class EdgeLoss(nn.Module):
    """Penalises errors in horizontal and vertical image gradients."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        def gradient(x):
            dx = x[:, :, :, :-1] - x[:, :, :, 1:]
            dy = x[:, :, :-1, :] - x[:, :, 1:, :]
            return dx, dy
        pred_dx, pred_dy = gradient(pred)
        tgt_dx, tgt_dy = gradient(target)
        return self.l1(pred_dx, tgt_dx) + self.l1(pred_dy, tgt_dy)


class ConvBlock(nn.Module):
    """Two conv layers with GroupNorm + SiLU and a residual projection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.act = nn.SiLU(inplace=True)
        self.res = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.act(self.block(x) + self.res(x))


class GuidedAttentionGate(nn.Module):
    """Attention gate using both encoder skip and decoder feature (Oktay et al., 2018)."""

    def __init__(self, skip_ch, gate_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or max(skip_ch // 2, 8)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=True), nn.Sigmoid())
        self.norm = nn.GroupNorm(1, inter_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, skip, gate):
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        combined = self.act(self.norm(self.W_skip(skip) + self.W_gate(gate)))
        alpha = self.psi(combined)  # (B,1,H,W)
        return skip * alpha


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample → attend skip → concatenate → ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, skip_ch, 2, stride=2)
        self.attn = GuidedAttentionGate(skip_ch=skip_ch, gate_ch=skip_ch)
        self.conv = ConvBlock(skip_ch * 2, out_ch)

    def forward(self, x, skip):
        x_up = self.up(x)
        skip_gated = self.attn(skip, gate=x_up)
        if x_up.shape[2:] != skip_gated.shape[2:]:
            x_up = F.interpolate(x_up, size=skip_gated.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x_up, skip_gated], dim=1))


class SimCNN(nn.Module):
    """ResUNet with Guided Attention Gates for single-photon image reconstruction."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Conv2d(96, 128, kernel_size=1)
        # Encoder
        self.enc1 = ConvBlock(128, 128)    # 512×512
        self.enc2 = Down(128, 256)         # 256×256
        self.enc3 = Down(256, 512)         # 128×128
        self.enc4 = Down(512, 1024)        #  64×64
        self.enc5 = Down(1024, 1024)       #  32×32
        # Double bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(1024, 2048),
            ConvBlock(2048, 2048),
        )
        # Decoder
        self.dec0 = Up(2048, 1024, 1024)
        self.dec1 = Up(1024, 1024, 512)
        self.dec2 = Up(512, 512, 256)
        self.dec3 = Up(256, 256, 128)
        self.dec4 = Up(128, 128, 128)
        self.out = nn.Conv2d(128, 3, kernel_size=1)
        # Auxiliary deep supervision head at dec2 (active during training only)
        self.aux_head = nn.Sequential(nn.Conv2d(256, 3, kernel_size=1))

    def forward(self, x):
        x = self.input_proj(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)
        d0 = self.dec0(b, e5)
        d1 = self.dec1(d0, e4)
        d2 = self.dec2(d1, e3)
        d3 = self.dec3(d2, e2)
        d4 = self.dec4(d3, e1)
        main_out = self.out(d4)
        if self.training:
            aux_out = self.aux_head(d2)
            aux_out = F.interpolate(aux_out, size=main_out.shape[2:], mode="bilinear", align_corners=False)
            return main_out, aux_out
        return main_out


def compute_psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * np.log10((data_range ** 2) / mse)


if __name__ == "__main__":
    train_dataset = SPCDataset(train_npy, train_png)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    log.info(f"Train loader: {len(train_loader)} batches")

    model = SimCNN().to(DEVICE)
    model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=1, eta_min=ETA_MIN
    )
    scaler = GradScaler("cuda")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params — Total: {total_params:,}  Trainable: {trainable_params:,}")

    # Resume from checkpoint if one exists
    start_epoch = 1
    best_train_ssim = -1.0
    ckpt_path = CKPT_DIR / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_train_ssim = ckpt["best_train_ssim"]
        log.info(f"Resumed from epoch {ckpt['epoch']}  best SSIM: {best_train_ssim:.4f}")
    else:
        log.info("No checkpoint found — training from scratch.")

    ssim_metric = SSIM(data_range=1.0).to(DEVICE)
    msssim_metric = MSSSIM(data_range=1.0, kernel_size=11).to(DEVICE)
    charb_loss = CharbonnierLoss(eps=1e-3)
    vgg_loss_fn = VGGPerceptualLoss().to(DEVICE)
    edge_loss_fn = EdgeLoss().to(DEVICE)
    writer = SummaryWriter(log_dir=str(LOG_DIR))

    writer.add_text("model/info", f"Total: {total_params:,} | Trainable: {trainable_params:,}")
    writer.add_text("config/hyperparams", (
        f"LR={LEARNING_RATE} | Epochs={NUM_EPOCHS} | Patch={PATCH_SIZE} | "
        f"Batch={BATCH_SIZE} | CosineT0={COSINE_T0} | EtaMin={ETA_MIN} | "
        f"W_CHARB={W_CHARB} | W_MSSSIM={W_MSSSIM} | W_VGG={W_VGG} | "
        f"W_EDGE={W_EDGE} | W_DEEP_SUP={W_DEEP_SUP}"
    ))

    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr", "deep_sup_w",
            "train_loss", "train_charb", "train_msssim_loss", "train_vgg",
            "train_edge", "train_aux",
            "train_ssim", "train_msssim", "train_psnr",
        ])

    log.info("All components ready. Starting training.")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Deep supervision weight: linear anneal from W_DEEP_SUP → 0
        deep_sup_w = W_DEEP_SUP * max(0.0, 1.0 - epoch / NUM_EPOCHS)

        model.train()
        t_loss = t_charb = t_msssim_l = t_vgg = t_edge = t_aux = 0.0
        t_ssim = t_msssim = t_psnr = 0.0

        for x, target in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            x, target = gpu_augment(x, target)
            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction, aux_pred = model(x)
                pred_clamped = torch.clamp(prediction, 0.0, 1.0)
                aux_pred_clamped = torch.clamp(aux_pred, 0.0, 1.0)
                charb = charb_loss(prediction, target)
                msssim_score = msssim_metric(pred_clamped, target)
                msssim_l = 1.0 - msssim_score
                vgg = vgg_loss_fn(pred_clamped, target)
                edge = edge_loss_fn(pred_clamped, target)
                aux_loss = charb_loss(aux_pred, target)
                loss = (
                    W_CHARB * charb +
                    W_MSSSIM * msssim_l +
                    W_VGG * vgg +
                    W_EDGE * edge +
                    deep_sup_w * aux_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                ssim_score = ssim_metric(pred_clamped.float(), target.float())
                psnr_score = compute_psnr(pred_clamped.float(), target.float())

            t_loss += loss.item()
            t_charb += charb.item()
            t_msssim_l += msssim_l.item()
            t_vgg += vgg.item()
            t_edge += edge.item()
            t_aux += aux_loss.item()
            t_ssim += ssim_score.item()
            t_msssim += msssim_score.item()
            t_psnr += psnr_score

        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()[0]
        n = len(train_loader)
        t_loss /= n; t_charb /= n; t_msssim_l /= n; t_vgg /= n
        t_edge /= n; t_aux /= n; t_ssim /= n; t_msssim /= n; t_psnr /= n

        total_grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        writer.add_scalars("Loss/total", {"train": t_loss}, epoch)
        writer.add_scalars("Loss/charbonnier", {"train": t_charb}, epoch)
        writer.add_scalars("Loss/msssim", {"train": t_msssim_l}, epoch)
        writer.add_scalars("Loss/vgg", {"train": t_vgg}, epoch)
        writer.add_scalars("Loss/edge", {"train": t_edge}, epoch)
        writer.add_scalars("Loss/deep_sup_aux", {"train": t_aux}, epoch)
        writer.add_scalars("Metric/SSIM", {"train": t_ssim}, epoch)
        writer.add_scalars("Metric/MS-SSIM", {"train": t_msssim}, epoch)
        writer.add_scalars("Metric/PSNR_dB", {"train": t_psnr}, epoch)
        writer.add_scalar("Train/lr", current_lr, epoch)
        writer.add_scalar("Train/grad_norm", total_grad_norm, epoch)
        writer.add_scalar("Train/deep_sup_weight", deep_sup_w, epoch)

        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"weights/{name}", param.data.cpu(), epoch)
                if param.grad is not None:
                    writer.add_histogram(f"grads/{name}", param.grad.cpu(), epoch)

        log.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  LR: {current_lr:.2e}  "
            f"DS_w: {deep_sup_w:.3f}  GradNorm: {total_grad_norm:.4f}  "
            f"| Loss: {t_loss:.4f}  Aux: {t_aux:.4f}  "
            f"SSIM: {t_ssim:.4f}  MS-SSIM: {t_msssim:.4f}  PSNR: {t_psnr:.2f} dB"
        )

        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{current_lr:.2e}", f"{deep_sup_w:.4f}",
                f"{t_loss:.4f}", f"{t_charb:.4f}", f"{t_msssim_l:.4f}",
                f"{t_vgg:.4f}", f"{t_edge:.4f}", f"{t_aux:.4f}",
                f"{t_ssim:.4f}", f"{t_msssim:.4f}", f"{t_psnr:.2f}",
            ])

        if t_ssim > best_train_ssim:
            best_train_ssim = t_ssim
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_train_ssim": best_train_ssim,
                "train_loss": t_loss,
            }, CKPT_DIR / "best_model.pth")
            log.info(f"  ✓ New best train SSIM: {best_train_ssim:.4f} — saved best_model.pth")
            writer.add_scalar("Best/train_ssim", best_train_ssim, epoch)

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_train_ssim": best_train_ssim,
            }, CKPT_DIR / f"ckpt_epoch{epoch:03d}.pth")

    writer.add_hparams(
        hparam_dict={
            "lr": LEARNING_RATE, "epochs": NUM_EPOCHS, "patch_size": PATCH_SIZE,
            "batch_size": BATCH_SIZE, "cosine_t0": COSINE_T0, "eta_min": ETA_MIN,
            "w_charb": W_CHARB, "w_msssim": W_MSSSIM, "w_vgg": W_VGG,
            "w_edge": W_EDGE, "w_deep_sup": W_DEEP_SUP,
        },
        metric_dict={"hparam/best_train_ssim": best_train_ssim},
    )
    writer.close()
    log.info(f"Training complete. Best train SSIM: {best_train_ssim:.4f}")