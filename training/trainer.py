# training/trainer.py
"""
Training infrastructure.
val_batches in SpeedConfig controls how many val batches
are evaluated per epoch — set to None for full val.
"""

import copy
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from losses.losses import InpaintingLossManager


# ── Scheduler ─────────────────────────────────────────────────────────────────

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_peak, lr_min):
        self.optimizer    = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.lr_peak = lr_peak
        self.lr_min  = lr_min

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = self.lr_peak * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            decay    = max(1, self.total_epochs - self.warmup_epochs)
            progress = (epoch - self.warmup_epochs) / decay
            lr = self.lr_min + 0.5 * (self.lr_peak - self.lr_min) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── AMP helpers ────────────────────────────────────────────────────────────────

def _autocast_ctx(use_amp: bool):
    if use_amp and torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.amp.autocast(device_type="cpu", dtype=torch.float32, enabled=False)


def _make_scaler(use_amp: bool):
    if use_amp and torch.cuda.is_available():
        return torch.amp.GradScaler("cuda")
    return torch.amp.GradScaler("cpu", enabled=False)


# ── Train one epoch ────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, loss_manager, optimizer,
    scaler, device, grad_clip, use_amp,
) -> Dict[str, float]:
    model.train()
    running  = defaultdict(float)
    n_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks, boundaries in pbar:
        images     = images.to(device, non_blocking=True)
        masks      = masks.to(device, non_blocking=True)
        boundaries = boundaries.to(device, non_blocking=True)
        masked_input = images * (1.0 - masks)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_ctx(use_amp):
            pred_dict = model(masked_input, masks, boundaries)
            losses    = loss_manager.compute(pred_dict, images, masks, boundaries)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for k, v in losses.items():
            running[k] += v.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})

    return {k: v / max(n_batches, 1) for k, v in running.items()}


# ── Validate ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model, loader, loss_manager, device,
    use_amp=True, max_batches=None,
) -> Dict[str, float]:
    model.eval()
    running  = defaultdict(float)
    n_batches = 0

    pbar = tqdm(loader, desc="  Val", leave=False)
    for batch_idx, (images, masks, boundaries) in enumerate(pbar):
        # ── Respect val_batches limit ──────────────────────────────────────
        if max_batches is not None and batch_idx >= max_batches:
            break

        images     = images.to(device, non_blocking=True)
        masks      = masks.to(device, non_blocking=True)
        boundaries = boundaries.to(device, non_blocking=True)
        masked_input = images * (1.0 - masks)

        with _autocast_ctx(use_amp):
            pred_dict = model(masked_input, masks, boundaries)
            losses    = loss_manager.compute(pred_dict, images, masks, boundaries)

        for k, v in losses.items():
            running[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


# ── Full experiment runner ─────────────────────────────────────────────────────

def run_experiment(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    loss_config: str = "full",
    num_epochs: int = 2,
    freeze_encoder_epochs: int = 1,
    patience: int = 2,
    device: Optional[torch.device] = None,
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  Loss config:    {loss_config}")
    print(f"  Max epochs:     {num_epochs}")
    print(f"  Freeze enc:     first {freeze_encoder_epochs} epochs")
    print(f"  Patience:       {patience}")
    print(f"  Device:         {device}")
    print(f"  Val batches:    {cfg.speed.val_batches if cfg.speed.val_batches else 'all'}")
    print(f"{'='*60}")

    model       = model.to(device)
    loss_manager = InpaintingLossManager(cfg.loss, device)
    optimizer   = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr_peak,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler  = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=num_epochs,
        lr_peak=cfg.train.lr_peak,
        lr_min=cfg.train.lr_min,
    )
    scaler     = _make_scaler(cfg.train.use_amp)
    early_stop = EarlyStopping(patience=patience)

    best_val_loss = float("inf")
    best_state    = None
    history       = {"train_loss": [], "val_loss": [], "lr": []}
    ckpt_path     = Path(cfg.paths.checkpoints_dir) / f"{name}.pt"

    _ENC_KEYS = ("enc_conv1", "enc_bn1", "enc_layer")

    def set_encoder_frozen(frozen: bool):
        for pname, p in model.named_parameters():
            if any(k in pname for k in _ENC_KEYS):
                p.requires_grad = not frozen

    set_encoder_frozen(True)
    t0 = time.time()

    for epoch in range(num_epochs):
        if epoch == freeze_encoder_epochs:
            set_encoder_frozen(False)
            print(f"  [Epoch {epoch}] Encoder unfrozen.")

        lr = scheduler.step(epoch)

        orig_config      = cfg.loss.loss_config
        cfg.loss.loss_config = loss_config

        train_losses = train_one_epoch(
            model, train_loader, loss_manager, optimizer,
            scaler, device, cfg.train.grad_clip, cfg.train.use_amp,
        )
        val_losses = validate(
            model, val_loader, loss_manager, device,
            cfg.train.use_amp, cfg.speed.val_batches,
        )

        cfg.loss.loss_config = orig_config

        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_losses["total"])
        history["lr"].append(lr)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch+1:3d}/{num_epochs} | "
            f"train={train_losses['total']:.4f}  "
            f"val={val_losses['total']:.4f} | "
            f"lr={lr:.2e} | {elapsed:.0f}s"
        )

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_state    = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)

        if early_stop.step(val_losses["total"]):
            print(f"  Early stopping at epoch {epoch+1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    total_elapsed = time.time() - t0
    print(f"  Done in {total_elapsed/60:.1f} min. Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {ckpt_path}")

    return {
        "name": name,
        "history": history,
        "best_val_loss": best_val_loss,
        "elapsed_min": total_elapsed / 60,
        "ckpt_path": str(ckpt_path),
    }


# ── Load or train ──────────────────────────────────────────────────────────────

def load_or_train(
    name: str,
    model_factory,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    loss_config: str = "base",
    num_epochs: int = 2,
    patience: int = 2,
    freeze_encoder_epochs: int = 1,
) -> Dict:
    ckpt_path = Path(cfg.paths.checkpoints_dir) / f"{name}.pt"

    if ckpt_path.exists():
        print(f"  ✓ Loading checkpoint: {ckpt_path}")
        model = model_factory()
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model = model.to(device)

        loss_manager = InpaintingLossManager(cfg.loss, device)
        orig = cfg.loss.loss_config
        cfg.loss.loss_config = loss_config
        val_losses = validate(
            model, val_loader, loss_manager, device,
            cfg.train.use_amp, cfg.speed.val_batches,
        )
        cfg.loss.loss_config = orig

        return {
            "name": name,
            "history": {"train_loss": [], "val_loss": [], "lr": []},
            "best_val_loss": val_losses["total"],
            "elapsed_min": 0.0,
            "ckpt_path": str(ckpt_path),
            "model": model,
        }

    model  = model_factory()
    result = run_experiment(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        loss_config=loss_config,
        num_epochs=num_epochs,
        freeze_encoder_epochs=freeze_encoder_epochs,
        patience=patience,
        device=device,
    )
    result["model"] = model
    return result
