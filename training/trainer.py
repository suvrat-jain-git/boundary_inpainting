# training/trainer.py
"""
Training infrastructure.

A100 Optimizations enabled by default:
  - bfloat16 autocast (wider dynamic range, no GradScaler needed on A100)
  - TF32 matmul + cuDNN (set at process start, ~1.5× free speedup)
  - channels_last memory format (NHWC, ~15% faster on A100 CNNs)
  - cudnn.benchmark=True (fixed input shape → faster cuDNN kernels)
  - EMA (exponential moving average, decay=0.999)
  - Validate every val_every_n_epochs epochs (reduces eval overhead)

Print format (per epoch, NO tqdm):
  Epoch 012/050 | train total=0.1234  hole=0.0456  bound=0.0321  spec=0.0234 \
               | val total=0.1456  val_bound=0.0123 | lr=2.99e-04 | 127s
"""

import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import Config
from losses.losses import InpaintingLossManager


# ── A100 global flags (called once at process start) ──────────────────────────

def enable_a100_flags():
    """Enable TF32, cuDNN benchmark, and other A100-specific speedups."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
        cudnn.benchmark  = True


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMAWeights:
    """
    Exponential Moving Average of model parameters.

    Usage:
        ema = EMAWeights(model, decay=0.999)
        # after each optimizer step:
        ema.update(model)
        # to evaluate with EMA weights:
        with ema.average_parameters(model):
            metrics = evaluate(model, ...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def apply_to(self, model: nn.Module):
        """Load EMA weights into model (for final eval)."""
        model.load_state_dict(self.shadow, strict=False)

    def restore_from(self, model: nn.Module, original_state: dict):
        """Restore original weights after EMA eval."""
        model.load_state_dict(original_state)

    class _ContextApply:
        def __init__(self, ema, model):
            self.ema   = ema
            self.model = model
            self._orig = None

        def __enter__(self):
            self._orig = copy.deepcopy(self.model.state_dict())
            self.ema.apply_to(self.model)
            return self.model

        def __exit__(self, *args):
            self.model.load_state_dict(self._orig)

    def average_parameters(self, model: nn.Module):
        """Context manager: temporarily swap in EMA weights."""
        return self._ContextApply(self, model)


# ── Scheduler ─────────────────────────────────────────────────────────────────

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_peak, lr_min):
        self.optimizer     = optimizer
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
    def __init__(self, patience: int = 8, min_delta: float = 1e-5):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_loss   = float("inf")
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


# ── AMP context helper ─────────────────────────────────────────────────────────

def _autocast_ctx(use_bf16: bool = True):
    """
    Return the appropriate autocast context for the current device.
    A100: prefers bfloat16 (wider range, no NaN risk, no GradScaler needed).
    Fallback for non-CUDA: disabled.
    """
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return torch.amp.autocast(device_type="cpu", dtype=torch.float32, enabled=False)


# ── Train one epoch ────────────────────────────────────────────────────────────

def train_one_epoch(
    model:                nn.Module,
    loader:               DataLoader,
    loss_manager:         InpaintingLossManager,
    optimizer:            torch.optim.Optimizer,
    device:               torch.device,
    grad_clip:            float,
    use_bf16:             bool,
    ema:                  Optional[EMAWeights] = None,
    perc_every_n_steps:   int = 1,
) -> Dict[str, float]:
    model.train()
    running   = defaultdict(float)
    n_batches = 0

    for images, masks, boundaries in loader:
        images     = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks      = masks.to(device, non_blocking=True)
        boundaries = boundaries.to(device, non_blocking=True)
        masked_input = images * (1.0 - masks)

        optimizer.zero_grad(set_to_none=True)

        # Skip perceptual loss on most steps to save VGG forward time.
        # Accumulated running avg stays accurate; gradients are still dense.
        skip_perc = (perc_every_n_steps > 1) and (n_batches % perc_every_n_steps != 0)
        loss_manager.skip_perceptual = skip_perc

        with _autocast_ctx(use_bf16):
            pred_dict = model(masked_input, masks, boundaries)
            losses    = loss_manager.compute(pred_dict, images, masks, boundaries)

        # bf16 on A100 does not need GradScaler — just backward directly
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        for k, v in losses.items():
            val = v.item()
            if not math.isnan(val):
                running[k] += val
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


# ── Validate ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:        nn.Module,
    loader:       DataLoader,
    loss_manager: InpaintingLossManager,
    device:       torch.device,
    use_bf16:     bool = True,
    max_batches:  Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    running   = defaultdict(float)
    n_batches = 0

    for batch_idx, (images, masks, boundaries) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images     = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks      = masks.to(device, non_blocking=True)
        boundaries = boundaries.to(device, non_blocking=True)
        masked_input = images * (1.0 - masks)

        with _autocast_ctx(use_bf16):
            pred_dict = model(masked_input, masks, boundaries)
            losses    = loss_manager.compute(pred_dict, images, masks, boundaries)

        for k, v in losses.items():
            running[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


# ── Per-epoch print ────────────────────────────────────────────────────────────

def _fmt_losses(d: Dict[str, float], prefix: str = "") -> str:
    """Format a loss dict into a compact string for epoch printing."""
    parts = []
    for key in ("total", "hole_l1", "valid_l1", "perceptual", "boundary", "spectral", "tv"):
        if key in d:
            short = key.replace("_l1", "").replace("erceptual", "erc")
            parts.append(f"{prefix}{short}={d[key]:.4f}")
    return "  ".join(parts)


# ── Full experiment runner ─────────────────────────────────────────────────────

def run_experiment(
    name:                  str,
    model:                 nn.Module,
    train_loader:          DataLoader,
    val_loader:            DataLoader,
    cfg:                   Config,
    loss_config:           str = "full",
    num_epochs:            int = 2,
    freeze_encoder_epochs: int = 1,
    patience:              int = 8,
    device:                Optional[torch.device] = None,
    val_every_n_epochs:    int = 1,
    perc_every_n_steps:    int = 1,
    initial_weights_path:  Optional[Path] = None,
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A100 flags
    enable_a100_flags()

    use_bf16 = getattr(cfg.train, "use_bf16", True) and torch.cuda.is_available()

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  Loss config  : {loss_config}")
    print(f"  Max epochs   : {num_epochs}")
    print(f"  Val every    : {val_every_n_epochs} epoch(s)")
    print(f"  Freeze enc   : first {freeze_encoder_epochs} epoch(s)")
    print(f"  Patience     : {patience}")
    print(f"  Device       : {device}")
    print(f"  Precision    : {'bfloat16' if use_bf16 else 'float16/float32'}")
    print(f"  EMA          : {cfg.train.use_ema}")
    print(f"  Val batches  : {cfg.speed.val_batches if cfg.speed.val_batches else 'all'}")
    print(f"  Perc every   : {perc_every_n_steps} step(s)")
    print(f"{'='*60}")

    # channels_last for CNN speed on A100
    if torch.cuda.is_available():
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    loss_manager = InpaintingLossManager(cfg.loss, device)

    # Separate param groups: higher LR for ASBC learnable weights
    asbc_params  = list(loss_manager.adaptive_spectral.parameters())
    other_params = [p for p in model.parameters()]
    param_groups = [
        {"params": other_params,  "lr": cfg.train.lr_peak},
        {"params": asbc_params,   "lr": cfg.train.lr_peak * 10,  "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)

    scheduler  = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=num_epochs,
        lr_peak=cfg.train.lr_peak,
        lr_min=cfg.train.lr_min,
    )
    early_stop = EarlyStopping(patience=patience)
    ema        = EMAWeights(model, decay=cfg.train.ema_decay) if cfg.train.use_ema else None

    best_val_loss    = float("inf")
    best_state       = None
    curves           = {"train": [], "val": [], "lr": []}   # for training_curves.json
    start_epoch      = 0
    ckpt_path        = Path(cfg.paths.checkpoints_dir) / f"{name}.pt"
    ema_ckpt_path    = Path(cfg.paths.checkpoints_dir) / f"{name}_ema.pt"
    resume_ckpt_path = Path(cfg.paths.checkpoints_dir) / f"{name}_resume.pt"

    _ENC_KEYS = ("enc_conv1", "enc_bn1", "enc_layer", "enc_convnext", "enc_proj")

    def set_encoder_frozen(frozen: bool):
        for pname, p in model.named_parameters():
            if any(k in pname for k in _ENC_KEYS):
                p.requires_grad = not frozen

    # ── Resume from previous interrupted run if resume checkpoint exists ──────
    if resume_ckpt_path.exists():
        print(f"  Resuming interrupted run from: {resume_ckpt_path}")
        try:
            resume = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(resume["model_state_dict"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            if ema is not None and resume.get("ema_shadow") is not None:
                ema.shadow = resume["ema_shadow"]
            best_val_loss        = resume["best_val_loss"]
            best_state           = resume.get("best_state_dict")
            curves               = resume["curves"]
            start_epoch          = resume["epoch"] + 1   # next epoch to run
            early_stop.counter   = resume["early_stop_counter"]
            early_stop.best_loss = resume["early_stop_best_loss"]
            print(f"  Resumed at epoch {start_epoch}/{num_epochs}. Best val so far: {best_val_loss:.4f}")
        except Exception as e:
            print(f"  Warning: could not load resume checkpoint ({e}). Starting from scratch.")
            start_epoch = 0

    # ── Warm-start from pre-trained weights (extension / fine-tune mode) ──────
    # Used when extending training from a completed run (no _resume.pt, but
    # caller supplies the path to the previous best / EMA checkpoint).
    if start_epoch == 0 and initial_weights_path is not None:
        _ip = Path(initial_weights_path)
        if _ip.exists():
            print(f"  Warm-start: loading initial weights from {_ip.name}")
            _ws = torch.load(_ip, map_location=device, weights_only=True)
            model.load_state_dict(_ws, strict=False)
            if ema is not None:
                ema.shadow = {k: v.clone().detach()
                              for k, v in model.state_dict().items()}
            print(f"  Warm-start ready.  lr_peak={cfg.train.lr_peak:.2e}  "
                  f"warmup={cfg.train.warmup_epochs} epoch(s)")
        else:
            print(f"  Warm-start skipped: {_ip} not found.")

    # Correct encoder freeze state for current start_epoch
    set_encoder_frozen(start_epoch < freeze_encoder_epochs)
    t0 = time.time()

    # Set the active loss config once for the whole experiment
    cfg.loss.loss_config = loss_config

    for epoch in range(start_epoch, num_epochs):
        if epoch == freeze_encoder_epochs:
            set_encoder_frozen(False)
            print(f"  [Epoch {epoch+1}] Encoder unfrozen.")

        lr = scheduler.step(epoch)
        t_ep = time.time()

        train_losses = train_one_epoch(
            model, train_loader, loss_manager, optimizer,
            device, cfg.train.grad_clip, use_bf16, ema,
            perc_every_n_steps=perc_every_n_steps,
        )

        # Validate every val_every_n_epochs epochs (or on the last epoch)
        should_val = (
            (epoch + 1) % val_every_n_epochs == 0
            or epoch == num_epochs - 1
        )

        if should_val:
            eval_model = model
            if ema is not None:
                # Evaluate with EMA weights for a more accurate signal
                _orig_state = copy.deepcopy(model.state_dict())
                ema.apply_to(model)
                val_losses = validate(
                    model, val_loader, loss_manager, device,
                    use_bf16, cfg.speed.val_batches,
                )
                model.load_state_dict(_orig_state)
            else:
                val_losses = validate(
                    model, val_loader, loss_manager, device,
                    use_bf16, cfg.speed.val_batches,
                )
        else:
            val_losses = {"total": float("nan")}

        curves["train"].append(train_losses)
        curves["val"].append(val_losses)
        curves["lr"].append(lr)

        elapsed = time.time() - t0

        # ── Per-epoch print (no tqdm) ──────────────────────────────────────────
        tl = train_losses
        vl = val_losses
        print(
            f"  Epoch {epoch+1:3d}/{num_epochs} | "
            f"train total={tl.get('total',0):.4f}  "
            f"hole={tl.get('hole_l1',0):.4f}  "
            f"bound={tl.get('boundary',0):.4f}  "
            f"spec={tl.get('spectral',0):.4f}  "
            f"tv={tl.get('tv',0):.4f} | "
            f"val total={vl.get('total',0):.4f}  "
            f"val_bound={vl.get('boundary',0):.4f} | "
            f"lr={lr:.2e} | {elapsed:.0f}s"
        )

        # Log ASBC band weights if using adaptive spectral loss
        if loss_config in ("adaptive_spectral", "full_adaptive"):
            w = loss_manager.adaptive_spectral.get_band_weights()
            w_str = "  ".join(f"b{i}={w[i]:.3f}" for i in range(len(w)))
            print(f"  ASBC band weights: {w_str}")

        if should_val and not math.isnan(vl["total"]):
            if vl["total"] < best_val_loss:
                best_val_loss = vl["total"]
                best_state    = copy.deepcopy(model.state_dict())
                torch.save(best_state, ckpt_path)
                if ema is not None:
                    torch.save(ema.shadow, ema_ckpt_path)

        # ── Save resume checkpoint every epoch for disconnect recovery ─────────
        torch.save({
            "epoch":                 epoch,
            "model_state_dict":      model.state_dict(),
            "optimizer_state_dict":  optimizer.state_dict(),
            "ema_shadow":            ema.shadow if ema is not None else None,
            "best_val_loss":         best_val_loss,
            "best_state_dict":       best_state,
            "curves":                curves,
            "early_stop_counter":    early_stop.counter,
            "early_stop_best_loss":  early_stop.best_loss,
        }, resume_ckpt_path)

        if should_val and not math.isnan(vl["total"]):
            if early_stop.step(vl["total"]):
                print(f"  Early stopping at epoch {epoch+1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Clean up resume checkpoint — training completed successfully
    if resume_ckpt_path.exists():
        resume_ckpt_path.unlink()

    # Save training curves JSON
    curves_path = Path(cfg.paths.results_dir) / f"{name}_curves.json"
    try:
        with open(curves_path, "w") as f:
            json.dump({
                "name":  name,
                "train": [{k: float(v) for k, v in ep.items()} for ep in curves["train"]],
                "val":   [{k: float(v) for k, v in ep.items()} for ep in curves["val"]],
                "lr":    curves["lr"],
            }, f, indent=2)
    except Exception as e:
        print(f"  Warning: could not save training curves: {e}")

    total_elapsed = time.time() - t0
    print(f"  Done in {total_elapsed/60:.1f} min. Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint : {ckpt_path}")
    if ema is not None:
        print(f"  EMA ckpt   : {ema_ckpt_path}")

    return {
        "name":          name,
        "history":       {"train_loss": [e.get("total", 0) for e in curves["train"]],
                          "val_loss":   [e.get("total", 0) for e in curves["val"]],
                          "lr":         curves["lr"]},
        "best_val_loss": best_val_loss,
        "elapsed_min":   total_elapsed / 60,
        "ckpt_path":     str(ckpt_path),
        "ema_ckpt_path": str(ema_ckpt_path) if ema is not None else None,
    }


# ── Load or train ──────────────────────────────────────────────────────────────

def load_or_train(
    name:                  str,
    model_factory,
    train_loader:          DataLoader,
    val_loader:            DataLoader,
    cfg:                   Config,
    device:                torch.device,
    loss_config:           str = "base",
    num_epochs:            int = 2,
    patience:              int = 8,
    freeze_encoder_epochs: int = 1,
    val_every_n_epochs:    int = 1,
    perc_every_n_steps:    int = 1,
) -> Dict:
    ckpt_path     = Path(cfg.paths.checkpoints_dir) / f"{name}.pt"
    ema_ckpt_path = Path(cfg.paths.checkpoints_dir) / f"{name}_ema.pt"
    curves_path   = Path(cfg.paths.results_dir)     / f"{name}_curves.json"

    # Only treat as fully done if BOTH _ema.pt AND _curves.json exist.
    # _curves.json is written only at the very end of run_experiment().
    # If missing, run_experiment() will auto-resume from _resume.pt if present.
    if ema_ckpt_path.exists() and curves_path.exists():
        print(f"  Loading checkpoint: {ckpt_path}")
        model = model_factory()
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model = model.to(device)

        use_bf16 = getattr(cfg.train, "use_bf16", True) and torch.cuda.is_available()
        loss_manager = InpaintingLossManager(cfg.loss, device)
        orig = cfg.loss.loss_config
        cfg.loss.loss_config = loss_config
        val_losses = validate(
            model, val_loader, loss_manager, device,
            use_bf16, cfg.speed.val_batches,
        )
        cfg.loss.loss_config = orig

        return {
            "name":          name,
            "history":       {"train_loss": [], "val_loss": [], "lr": []},
            "best_val_loss": val_losses["total"],
            "elapsed_min":   0.0,
            "ckpt_path":     str(ckpt_path),
            "model":         model,
        }

    if not (ema_ckpt_path.exists() and curves_path.exists()):
        if ckpt_path.exists():
            resume_ckpt_path = Path(cfg.paths.checkpoints_dir) / f"{name}_resume.pt"
            if resume_ckpt_path.exists():
                # Extract epoch number from resume checkpoint to show progress
                try:
                    r = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
                    done = r.get("epoch", "?") + 1
                    print(f"  Resuming {name} from epoch {done}/{num_epochs} (resume checkpoint found).")
                except Exception:
                    print(f"  Restarting {name} (resume checkpoint unreadable).")
            else:
                print(f"  Restarting {name} from scratch (no resume checkpoint).")

    # Auto-detect warm-start: _ema.pt exists, _curves.json missing, no _resume.pt.
    # This means a completed run whose epochs are being extended.
    _resume_exists = (Path(cfg.paths.checkpoints_dir) / f"{name}_resume.pt").exists()
    _auto_warm = (
        ema_ckpt_path.exists()
        and not curves_path.exists()
        and not _resume_exists
    )
    if _auto_warm:
        print(f"  Extension mode detected for '{name}': "
              f"warm-starting from {ema_ckpt_path.name}")

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
        val_every_n_epochs=val_every_n_epochs,
        perc_every_n_steps=perc_every_n_steps,
        initial_weights_path=ema_ckpt_path if _auto_warm else None,
    )
    result["model"] = model
    return result
