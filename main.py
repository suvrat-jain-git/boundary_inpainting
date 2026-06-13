# main.py
"""
Main pipeline entry point.

Usage:
    python main.py                                   # full pipeline
    python main.py --phase 4                         # only phase 4
    python main.py --eval_only                       # evaluate checkpoints only
    python main.py --places_dir ./datasets/places365 # custom dataset path
    python main.py --profile production              # use production config

Speed control — edit configs/config.py → SpeedConfig, or use --profile:
    test_fraction   : fraction of dataset used (0.001 = tiny, 1.0 = full)
    epochs_per_phase: epochs for phases 1-3 selection
    epochs_full     : epochs for ALL phase-4 ablation configs (fair equal training)
    val_batches     : max val batches per epoch (None = all)

FAIRNESS NOTE: ALL phase-4 ablation configs (L0-L4, L3c) use epochs_full —
the same number of epochs. This is required for a valid ablation table.
(Prior code used epochs_per_phase for L0-L3, which was shorter than L4's
epochs_full, making the comparison unfair.)
"""

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import Config, ModelConfig, get_config, get_production_config, get_smoke_config
from data.dataset import setup_data
from evaluation.metrics import ModelEvaluator, run_sensitivity_ablation, save_results, compute_fid_from_loader
from losses.losses import InpaintingLossManager
from models.architecture import BoundaryAwareInpainter, build_model
from training.trainer import load_or_train, run_experiment, validate, enable_a100_flags
from utils.visualize import (
    plot_ablation_table, plot_baseline_comparison, plot_mask_coverage,
    plot_mask_examples, plot_qualitative, plot_sample_grid,
    plot_sensitivity, plot_training_curves,
)


# ── Reproducibility ────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: we do NOT set cudnn.deterministic=True here because that disables
    # cudnn.benchmark — we want benchmark=True for A100 speed.
    # Reproducibility is ensured by fixed seeds and cached masks.


# ── Pipeline State (resume safety) ────────────────────────────────────────────

class PipelineState:
    """
    Tracks completed experiment runs via pipeline_state.json.
    Each entry: run_key → {"status": "done", "best_val_loss": float, "metrics": {...}}

    On re-run, completed runs are skipped — results are loaded from the JSON.
    This lets you safely interrupt and resume the Colab session.
    """

    def __init__(self, results_dir: str):
        self.path  = Path(results_dir) / "pipeline_state.json"
        self._data: Dict = {}
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
                print(f"  Pipeline state loaded ({len(self._data)} completed runs).")
            except Exception:
                self._data = {}

    def is_done(self, key: str) -> bool:
        return self._data.get(key, {}).get("status") == "done"

    def mark_done(self, key: str, payload: dict):
        self._data[key] = {"status": "done", **payload}
        self._save()

    def get(self, key: str) -> dict:
        return self._data.get(key, {})

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)


# ── Model factory helper ───────────────────────────────────────────────────────

def make_model_cfg(backbone, use_attn_skip, use_gated_conv, multiscale=True):
    return ModelConfig(
        backbone=backbone,
        use_attention_skip=use_attn_skip,
        use_gated_conv=use_gated_conv,
        multiscale_output=multiscale,
        use_transformer_bottleneck=True,
        use_bilinear_upsample=True,
        use_boundary_conditioning=True,
    )


# ── Phase 1: Backbone Selection ────────────────────────────────────────────────

def run_phase1(data, cfg, device):
    print("\n" + "="*60)
    print("PHASE 1: Backbone Selection (ResNet-18 vs ResNet-34)")
    print("="*60)

    results = {}
    for backbone in ["resnet18", "resnet34"]:
        def factory(bb=backbone):
            c = ModelConfig(
                backbone=bb,
                use_attention_skip=False,
                use_gated_conv=False,
                multiscale_output=False,
                use_transformer_bottleneck=False,
                use_bilinear_upsample=True,
                use_boundary_conditioning=False,
            )
            return BoundaryAwareInpainter(c)

        result = load_or_train(
            name=f"phase1_{backbone}",
            model_factory=factory,
            train_loader=data["fast_train_loader"],
            val_loader=data["fast_val_loader"],
            cfg=cfg,
            device=device,
            loss_config="base",
            num_epochs=cfg.speed.epochs_per_phase,
            patience=cfg.train.patience,
            freeze_encoder_epochs=cfg.train.freeze_encoder_epochs,
            val_every_n_epochs=cfg.speed.val_every_n_epochs,
            perc_every_n_steps=cfg.speed.perc_every_n_steps,
        )
        results[backbone] = result
        if "model" in result:
            del result["model"]
        torch.cuda.empty_cache()
        gc.collect()

    winner = min(results, key=lambda k: results[k]["best_val_loss"])
    print(f"\n  Phase 1 Winner: {winner} "
          f"(val_loss={results[winner]['best_val_loss']:.4f})")
    return results, winner


# ── Phase 2: Skip Connection Study ────────────────────────────────────────────

def run_phase2(data, cfg, device, backbone):
    print("\n" + "="*60)
    print("PHASE 2: Skip Connection Study")
    print("="*60)

    results = {}
    for skip_type, use_attn in [("standard", False), ("attention_mask", True)]:
        def factory(ua=use_attn):
            c = ModelConfig(
                backbone=backbone,
                use_attention_skip=ua,
                use_gated_conv=False,
                multiscale_output=False,
                use_transformer_bottleneck=False,
                use_bilinear_upsample=True,
                use_boundary_conditioning=False,
            )
            return BoundaryAwareInpainter(c)

        result = load_or_train(
            name=f"phase2_{skip_type}",
            model_factory=factory,
            train_loader=data["fast_train_loader"],
            val_loader=data["fast_val_loader"],
            cfg=cfg,
            device=device,
            loss_config="base",
            num_epochs=cfg.speed.epochs_per_phase,
            patience=cfg.train.patience,
            freeze_encoder_epochs=cfg.train.freeze_encoder_epochs,
            val_every_n_epochs=cfg.speed.val_every_n_epochs,
            perc_every_n_steps=cfg.speed.perc_every_n_steps,
        )
        results[skip_type] = result
        if "model" in result:
            del result["model"]
        torch.cuda.empty_cache()
        gc.collect()

    winner     = min(results, key=lambda k: results[k]["best_val_loss"])
    use_attn   = (winner == "attention_mask")
    print(f"\n  Phase 2 Winner: {winner} "
          f"(val_loss={results[winner]['best_val_loss']:.4f})")
    return results, winner, use_attn


# ── Phase 3: Gated Convolution Study ──────────────────────────────────────────

def run_phase3(data, cfg, device, backbone, use_attn_skip):
    print("\n" + "="*60)
    print("PHASE 3: Gated Convolution Study")
    print("="*60)

    results = {}
    for conv_type, use_gated in [("standard_conv", False), ("gated_conv", True)]:
        def factory(ug=use_gated):
            c = ModelConfig(
                backbone=backbone,
                use_attention_skip=use_attn_skip,
                use_gated_conv=ug,
                multiscale_output=False,
                use_transformer_bottleneck=False,
                use_bilinear_upsample=True,
                use_boundary_conditioning=False,
            )
            return BoundaryAwareInpainter(c)

        result = load_or_train(
            name=f"phase3_{conv_type}",
            model_factory=factory,
            train_loader=data["fast_train_loader"],
            val_loader=data["fast_val_loader"],
            cfg=cfg,
            device=device,
            loss_config="base",
            num_epochs=cfg.speed.epochs_per_phase,
            patience=cfg.train.patience,
            freeze_encoder_epochs=cfg.train.freeze_encoder_epochs,
            val_every_n_epochs=cfg.speed.val_every_n_epochs,
            perc_every_n_steps=cfg.speed.perc_every_n_steps,
        )
        results[conv_type] = result
        if "model" in result:
            del result["model"]
        torch.cuda.empty_cache()
        gc.collect()

    winner     = min(results, key=lambda k: results[k]["best_val_loss"])
    use_gated  = (winner == "gated_conv")
    print(f"\n  Phase 3 Winner: {winner} "
          f"(val_loss={results[winner]['best_val_loss']:.4f})")
    return results, winner, use_gated


# ── Phase 4: Loss Ablation ─────────────────────────────────────────────────────

# FAIRNESS FIX: ALL configs use epochs_full — same number of training epochs.
# Previous code used epochs_per_phase for L0-L3, making the table unfair.
PHASE4_CONFIGS = {
    "L0_base":              {"loss_config": "base",               "multiscale": False},
    "L1_boundary_uniform":  {"loss_config": "boundary_uniform",   "multiscale": False},
    "L2_boundary_grad":     {"loss_config": "boundary_grad",      "multiscale": False},
    "L3_spectral_only":     {"loss_config": "spectral_only",      "multiscale": False},
    "L3c_adaptive_spectral":{"loss_config": "adaptive_spectral",  "multiscale": False},  # ASBC
    "L4_full_method":       {"loss_config": "full",               "multiscale": True},
}


def run_phase4(data, cfg, device, backbone, use_attn_skip, use_gated_conv,
               state: Optional[PipelineState] = None, seed: int = 42):
    print("\n" + "="*60)
    print(f"PHASE 4: Loss Ablation Study  (seed={seed})")
    print("  NOTE: ALL configs trained for epochs_full={} (fair equal training)"
          .format(cfg.speed.epochs_full))
    print("="*60)

    results   = {}
    histories = {}
    seed_everything(seed)

    for exp_name, exp_cfg in PHASE4_CONFIGS.items():
        run_key = f"phase4_{exp_name}_seed{seed}"

        if state is not None and state.is_done(run_key):
            print(f"  Skipping {run_key} (already done)")
            results[exp_name] = state.get(run_key)
            continue

        def factory(ms=exp_cfg["multiscale"]):
            c = make_model_cfg(backbone, use_attn_skip, use_gated_conv, multiscale=ms)
            return BoundaryAwareInpainter(c)

        ckpt_name = f"phase4_{exp_name}_seed{seed}" if seed != 42 else f"phase4_{exp_name}"
        result = load_or_train(
            name=ckpt_name,
            model_factory=factory,
            train_loader=data["fast_train_loader"],
            val_loader=data["fast_val_loader"],
            cfg=cfg,
            device=device,
            loss_config=exp_cfg["loss_config"],
            num_epochs=cfg.speed.epochs_full,          # ← ALL configs use epochs_full
            patience=cfg.train.patience,
            freeze_encoder_epochs=cfg.train.freeze_encoder_epochs,
            val_every_n_epochs=cfg.speed.val_every_n_epochs,
            perc_every_n_steps=cfg.speed.perc_every_n_steps,
        )
        results[exp_name]   = result
        histories[exp_name] = result.get("history", {})

        if state is not None:
            state.mark_done(run_key, {
                "best_val_loss": result["best_val_loss"],
                "elapsed_min":   result.get("elapsed_min", 0),
                "ckpt_path":     result.get("ckpt_path", ""),
            })

        if "model" in result:
            del result["model"]
        torch.cuda.empty_cache()
        gc.collect()

    return results, histories


# ── Multi-seed runs for L0 and L4 ─────────────────────────────────────────────

def run_multiseed_l0l4(data, cfg, device, backbone, use_attn_skip, use_gated_conv,
                       seeds=(42, 1, 2), state: Optional[PipelineState] = None):
    """
    Train L0 (base) and L4 (full) under multiple seeds for statistical reporting.
    Seed 42 results are already computed in phase 4 — skip those.
    Returns: {exp_name: [result_seed42, result_seed1, result_seed2, ...]}
    """
    print("\n" + "="*60)
    print(f"MULTI-SEED RUNS  (seeds={seeds}, configs=L0+L4)")
    print("="*60)

    multi_results = {"L0_base": [], "L4_full_method": []}

    for exp_name, exp_cfg in [
        ("L0_base",       {"loss_config": "base", "multiscale": False}),
        ("L4_full_method",{"loss_config": "full", "multiscale": True}),
    ]:
        for seed in seeds:
            if seed == 42:
                # Seed 42 was already run in phase 4 — load it
                ckpt_name = f"phase4_{exp_name}"
            else:
                ckpt_name = f"multiseed_{exp_name}_seed{seed}"

            run_key = f"multiseed_{exp_name}_seed{seed}"
            if seed == 42:
                run_key = f"phase4_{exp_name}_seed42"

            if state is not None and state.is_done(run_key):
                print(f"  Skipping {run_key} (already done)")
                multi_results[exp_name].append(state.get(run_key))
                continue

            seed_everything(seed)

            def factory(ms=exp_cfg["multiscale"]):
                c = make_model_cfg(backbone, use_attn_skip, use_gated_conv, multiscale=ms)
                return BoundaryAwareInpainter(c)

            result = load_or_train(
                name=ckpt_name,
                model_factory=factory,
                train_loader=data["fast_train_loader"],
                val_loader=data["fast_val_loader"],
                cfg=cfg,
                device=device,
                loss_config=exp_cfg["loss_config"],
                num_epochs=cfg.speed.epochs_full,
                patience=cfg.train.patience,
                freeze_encoder_epochs=cfg.train.freeze_encoder_epochs,
                val_every_n_epochs=cfg.speed.val_every_n_epochs,
                perc_every_n_steps=cfg.speed.perc_every_n_steps,
            )
            multi_results[exp_name].append(result)

            if state is not None:
                state.mark_done(run_key, {
                    "best_val_loss": result["best_val_loss"],
                    "seed": seed,
                })

            if "model" in result:
                del result["model"]
            torch.cuda.empty_cache()
            gc.collect()

    return multi_results


# ── ConvNeXt Runs ─────────────────────────────────────────────────────────────

def run_convnext_runs(data, cfg, device, use_attn_skip, use_gated_conv,
                      state: Optional[PipelineState] = None, seed: int = 42):
    """
    Train L0 and L4 with ConvNeXt-Tiny backbone (same hyperparams).
    Demonstrates architecture-agnostic claim for the paper.
    """
    print("\n" + "="*60)
    print(f"CONVNEXT RUNS  (backbone=convnext_tiny, seed={seed})")
    print("="*60)

    results = {}
    seed_everything(seed)

    for exp_name, exp_cfg in [
        ("C0_convnext_base", {"loss_config": "base", "multiscale": False}),
        ("C4_convnext_full", {"loss_config": "full", "multiscale": True}),
    ]:
        run_key = f"{exp_name}_seed{seed}"

        if state is not None and state.is_done(run_key):
            print(f"  Skipping {run_key} (already done)")
            results[exp_name] = state.get(run_key)
            continue

        def factory(ms=exp_cfg["multiscale"]):
            c = make_model_cfg("convnext_tiny", use_attn_skip, use_gated_conv, multiscale=ms)
            return BoundaryAwareInpainter(c)

        result = load_or_train(
            name=exp_name,
            model_factory=factory,
            train_loader=data["fast_train_loader"],
            val_loader=data["fast_val_loader"],
            cfg=cfg,
            device=device,
            loss_config=exp_cfg["loss_config"],
            num_epochs=cfg.speed.epochs_full,
            patience=cfg.train.patience,
            freeze_encoder_epochs=cfg.train.freeze_encoder_epochs,
            val_every_n_epochs=cfg.speed.val_every_n_epochs,
            perc_every_n_steps=cfg.speed.perc_every_n_steps,
        )
        results[exp_name] = result

        if state is not None:
            state.mark_done(run_key, {
                "best_val_loss": result["best_val_loss"],
                "elapsed_min":   result.get("elapsed_min", 0),
            })

        if "model" in result:
            del result["model"]
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _resolve_ckpt(path: Path) -> Path:
    """Return path if it exists, else try the _seed42 variant, else return original."""
    if path.exists():
        return path
    seed42 = path.with_name(path.stem + "_seed42" + path.suffix)
    if seed42.exists():
        return seed42
    return path   # will fail exists() check downstream → skipped


def _load_model_for_eval(exp_name, ckpt_path, backbone, use_attn_skip, use_gated_conv,
                         multiscale, device):
    """Load a checkpoint into a model and return it."""
    ckpt_path = _resolve_ckpt(Path(ckpt_path))
    if not Path(ckpt_path).exists():
        return None
    model_cfg = make_model_cfg(backbone, use_attn_skip, use_gated_conv, multiscale=multiscale)
    model     = BoundaryAwareInpainter(model_cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model


def run_evaluation(data, cfg, device, backbone, use_attn_skip, use_gated_conv):
    """
    Evaluate all phase-4 checkpoints on:
      1. Places365 test set (main table)
      2. CelebA-HQ (cross-domain)
      3. DTD textures (cross-domain)
      4. FID for L4 and ConvNeXt (if clean-fid available)

    Cross-domain evaluation covers ALL phase-4 configs (not just L0/L4) so
    the paper can claim: "our full method generalises across all domains."
    """
    print("\n" + "="*60)
    print("EVALUATION: Test Set Metrics (Places365 + Cross-Domain)")
    print("="*60)

    evaluator   = ModelEvaluator(device, cfg.data.mask_bins)
    use_bf16    = getattr(cfg.train, "use_bf16", True) and torch.cuda.is_available()
    ckpt_dir    = Path(cfg.paths.checkpoints_dir)
    results_dir = cfg.paths.results_dir

    phase4_eval = {}
    celeba_eval = {}
    dtd_eval    = {}
    fid_scores  = {}

    # ── Places365 evaluation (all phase-4 + ConvNeXt runs) ────────────────────
    for exp_name, exp_cfg in PHASE4_CONFIGS.items():
        ckpt_path = ckpt_dir / f"phase4_{exp_name}.pt"
        model     = _load_model_for_eval(
            exp_name, ckpt_path, backbone, use_attn_skip, use_gated_conv,
            exp_cfg["multiscale"], device,
        )
        if model is None:
            print(f"  No checkpoint for {exp_name} — skipping")
            continue

        # Use EMA weights for all configs where available
        ema_ckpt = _resolve_ckpt(ckpt_dir / f"phase4_{exp_name}_ema.pt")
        if ema_ckpt.exists():
            model.load_state_dict(
                torch.load(ema_ckpt, map_location=device, weights_only=True),
                strict=False,
            )
            print(f"  {exp_name}: using EMA weights")

        print(f"\n  Evaluating {exp_name}...")
        g, bins = evaluator.evaluate(
            model, data["fast_test_loader"], use_amp=use_bf16,
        )
        phase4_eval[exp_name] = {"global": g, "bins": bins}
        print(
            f"    PSNR={g.get('psnr',0):.2f}  SSIM={g.get('ssim',0):.4f}  "
            f"Hole-MAE={g.get('hole_mae',0):.4f}  "
            f"Boundary-MAE={g.get('boundary_mae',0):.4f}  "
            f"Spectral={g.get('spectral_coherence',0):.4f}"
        )
        if "lpips" in g:
            print(f"    LPIPS={g['lpips']:.4f}")

        # FID for L4 and ASBC
        if exp_name in ("L4_full_method", "L3c_adaptive_spectral") and cfg.eval.compute_fid:
            print(f"  Computing FID for {exp_name}...")
            fid = compute_fid_from_loader(
                model, data["fast_test_loader"], device,
                results_dir, f"phase4_{exp_name}",
                use_bf16=use_bf16, max_batches=None,
            )
            fid_scores[exp_name] = fid
            phase4_eval[exp_name]["global"]["fid"] = fid

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ConvNeXt checkpoints
    for exp_name in ["C0_convnext_base", "C4_convnext_full"]:
        ckpt_path = ckpt_dir / f"{exp_name}.pt"
        model     = _load_model_for_eval(
            exp_name, ckpt_path, "convnext_tiny", use_attn_skip, use_gated_conv,
            multiscale=(exp_name == "C4_convnext_full"), device=device,
        )
        if model is None:
            continue

        print(f"\n  Evaluating {exp_name} (ConvNeXt-T)...")
        g, bins = evaluator.evaluate(
            model, data["fast_test_loader"], use_amp=use_bf16,
        )
        phase4_eval[exp_name] = {"global": g, "bins": bins}
        print(
            f"    PSNR={g.get('psnr',0):.2f}  Boundary-MAE={g.get('boundary_mae',0):.4f}  "
            f"Spectral={g.get('spectral_coherence',0):.4f}"
        )

        if exp_name == "C4_convnext_full" and cfg.eval.compute_fid:
            fid = compute_fid_from_loader(
                model, data["fast_test_loader"], device,
                results_dir, exp_name, use_bf16=use_bf16,
            )
            fid_scores[exp_name]  = fid
            phase4_eval[exp_name]["global"]["fid"] = fid

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── Cross-domain: CelebA-HQ (ALL phase-4 configs) ─────────────────────────
    if data["celeba_loader"] is not None:
        print("\n  Cross-Domain Evaluation: CelebA-HQ")
        for exp_name, exp_cfg in PHASE4_CONFIGS.items():
            ckpt_path = ckpt_dir / f"phase4_{exp_name}.pt"
            model     = _load_model_for_eval(
                exp_name, ckpt_path, backbone, use_attn_skip, use_gated_conv,
                exp_cfg["multiscale"], device,
            )
            if model is None:
                continue

            g, _ = evaluator.evaluate(model, data["celeba_loader"], use_amp=use_bf16)
            celeba_eval[exp_name] = g
            print(
                f"    {exp_name}: PSNR={g.get('psnr',0):.2f}  "
                f"Boundary-MAE={g.get('boundary_mae',0):.4f}  "
                f"Spectral={g.get('spectral_coherence',0):.4f}"
            )
            del model
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print("\n  CelebA-HQ not available — skipping cross-domain eval.")

    # ── Cross-domain: DTD textures ─────────────────────────────────────────────
    if data.get("dtd_loader") is not None:
        print("\n  Cross-Domain Evaluation: DTD Textures")
        for exp_name in ["L0_base", "L3c_adaptive_spectral", "L4_full_method"]:
            ckpt_path = ckpt_dir / f"phase4_{exp_name}.pt"
            exp_cfg   = PHASE4_CONFIGS.get(exp_name, {})
            model     = _load_model_for_eval(
                exp_name, ckpt_path, backbone, use_attn_skip, use_gated_conv,
                exp_cfg.get("multiscale", False), device,
            )
            if model is None:
                continue

            g, _ = evaluator.evaluate(model, data["dtd_loader"], use_amp=use_bf16)
            dtd_eval[exp_name] = g
            print(
                f"    {exp_name}: PSNR={g.get('psnr',0):.2f}  "
                f"Boundary-MAE={g.get('boundary_mae',0):.4f}"
            )
            del model
            torch.cuda.empty_cache()
            gc.collect()

    return phase4_eval, celeba_eval, dtd_eval, fid_scores


# ── Aggregate multi-seed statistics ───────────────────────────────────────────

def aggregate_multiseed(multi_results, cfg, device, data, backbone, use_attn_skip, use_gated_conv):
    """
    For L0 and L4 multi-seed runs, compute test metrics for each seed
    and return mean ± std per metric.
    """
    import math
    evaluator = ModelEvaluator(device, cfg.data.mask_bins)
    use_bf16  = getattr(cfg.train, "use_bf16", True) and torch.cuda.is_available()
    ckpt_dir  = Path(cfg.paths.checkpoints_dir)
    seeds     = [42, 1, 2]

    stats = {}
    for exp_name in ["L0_base", "L4_full_method"]:
        all_metrics = {}
        ms_flag = (exp_name == "L4_full_method")

        for seed in seeds:
            ckpt_name = f"phase4_{exp_name}" if seed == 42 else f"multiseed_{exp_name}_seed{seed}"
            ckpt_path = ckpt_dir / f"{ckpt_name}.pt"
            model     = _load_model_for_eval(
                exp_name, ckpt_path, backbone, use_attn_skip, use_gated_conv,
                multiscale=ms_flag, device=device,
            )
            if model is None:
                print(f"  Missing ckpt for {exp_name} seed {seed}")
                continue

            g, _ = evaluator.evaluate(model, data["fast_test_loader"], use_amp=use_bf16)
            for k, v in g.items():
                all_metrics.setdefault(k, []).append(v)
            del model
            torch.cuda.empty_cache()

        if all_metrics:
            stats[exp_name] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in all_metrics.items()
            }
            print(f"  {exp_name} ({len(list(all_metrics.values())[0])} seeds):")
            for k in ("psnr", "ssim", "boundary_mae", "spectral_coherence"):
                if k in stats[exp_name]:
                    m, s = stats[exp_name][k]["mean"], stats[exp_name][k]["std"]
                    print(f"    {k}: {m:.4f} ± {s:.4f}")

    return stats


# ── Sensitivity Analysis ───────────────────────────────────────────────────────

def run_sensitivity(data, cfg, device, backbone, use_attn_skip, use_gated_conv):
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)

    ckpt_path = Path(cfg.paths.checkpoints_dir) / "phase4_L4_full_method.pt"
    if not ckpt_path.exists():
        print("  L4 checkpoint not found — skipping sensitivity analysis")
        return {}

    model_cfg = make_model_cfg(backbone, use_attn_skip, use_gated_conv, multiscale=True)
    model     = BoundaryAwareInpainter(model_cfg).to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )

    sensitivity = run_sensitivity_ablation(
        model=model,
        loader=data["fast_test_loader"],
        device=device,
        mask_bins=cfg.data.mask_bins,
        k_values=cfg.eval.k_values,
        patch_sizes=cfg.eval.patch_sizes,
        dilation_values=cfg.eval.dilation_values,
        use_amp=getattr(cfg.train, "use_bf16", True),
    )

    del model
    torch.cuda.empty_cache()
    return sensitivity


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Boundary-Aware Inpainting Pipeline")
    parser.add_argument("--skip_phases123", action="store_true",
                        help="Skip phases 1-3 (use resnet34+attn_skip+gated_conv). "
                             "Saves ~3h. Recommended when architecture is already decided.")
    parser.add_argument("--skip_multiseed", action="store_true",
                        help="Skip multi-seed runs. Saves ~4 extra 35-epoch runs.")
    parser.add_argument("--skip_convnext",  action="store_true",
                        help="Skip ConvNeXt backbone runs. Saves ~2 extra 35-epoch runs.")
    parser.add_argument("--phase",      type=int, default=0,
                        help="Run specific phase only (1-4). 0 = full pipeline.")
    parser.add_argument("--eval_only",  action="store_true")
    parser.add_argument("--sensitivity",action="store_true")
    parser.add_argument("--profile",    type=str, default="default",
                        choices=["default", "production", "smoke"],
                        help="Config profile: default / production / smoke")
    parser.add_argument("--places_dir", type=str, default=None)
    parser.add_argument("--celeba_dir", type=str, default=None)
    parser.add_argument("--dtd_dir",    type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers",type=int, default=None)
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    if args.profile == "production":
        cfg = get_production_config()
    elif args.profile == "smoke":
        cfg = get_smoke_config()
    else:
        cfg = get_config()

    if args.places_dir:  cfg.data.places365_dir = args.places_dir
    if args.celeba_dir:  cfg.data.celeba_dir    = args.celeba_dir
    if args.dtd_dir:     cfg.data.dtd_dir       = args.dtd_dir
    if args.batch_size:  cfg.data.batch_size    = args.batch_size
    if args.num_workers is not None:
                         cfg.data.num_workers   = args.num_workers

    cfg.make_dirs()
    enable_a100_flags()

    state = PipelineState(cfg.paths.results_dir)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = getattr(cfg.train, "use_bf16", True) and device.type == "cuda"
    print(f"\nDevice   : {device}")
    if device.type == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
        print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"Precision: {'bfloat16' if use_bf16 else 'float32'}")
    print(f"Profile  : {args.profile}")
    print(f"\nSpeed settings:")
    print(f"  test_fraction    = {cfg.speed.test_fraction}")
    print(f"  epochs_per_phase = {cfg.speed.epochs_per_phase}")
    print(f"  epochs_full      = {cfg.speed.epochs_full}  (ALL phase-4 configs)")
    print(f"  val_every_n      = {cfg.speed.val_every_n_epochs}")
    print(f"  val_batches      = {cfg.speed.val_batches}")

    seed_everything(cfg.train.seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nSetting up data...")
    data        = setup_data(cfg)
    results_dir = Path(cfg.paths.results_dir)

    # ── EDA ──────────────────────────────────────────────────────────────────
    print("\nGenerating EDA plots...")
    try:
        plot_sample_grid(data["places_splits"]["train"], results_dir)
        plot_mask_examples(results_dir, cfg.data.img_size)
        plot_mask_coverage(results_dir, img_size=cfg.data.img_size)
    except Exception as e:
        print(f"  EDA warning: {e}")

    # ── Sensitivity only ──────────────────────────────────────────────────────
    if args.sensitivity:
        sensitivity = run_sensitivity(data, cfg, device, "resnet34", True, True)
        save_results(sensitivity, cfg.paths.results_dir, "sensitivity")
        plot_sensitivity(sensitivity, results_dir)
        return

    # ── Phases 1-3: Architecture selection ───────────────────────────────────
    _skip123 = args.skip_phases123 or args.eval_only or (args.phase not in (0, 1, 2, 3))
    if args.phase in (0, 1) and not _skip123:
        p1_results, p1_winner = run_phase1(data, cfg, device)
    else:
        p1_winner  = "resnet34"   # validated winner from Phase 1 logs
        p1_results = {}
        if args.skip_phases123:
            print("  Phases 1-3 skipped. Using: resnet34 + attn_skip + gated_conv")

    if args.phase in (0, 2) and not _skip123:
        p2_results, p2_winner, use_attn_skip = run_phase2(data, cfg, device, p1_winner)
    else:
        p2_winner    = "attention_mask"
        use_attn_skip = True
        p2_results   = {}

    if args.phase in (0, 3) and not _skip123:
        p3_results, p3_winner, use_gated_conv = run_phase3(
            data, cfg, device, p1_winner, use_attn_skip
        )
    else:
        p3_winner    = "gated_conv"
        use_gated_conv = True
        p3_results   = {}

    print(f"\n  Architecture locked:")
    print(f"    Backbone  : {p1_winner}")
    print(f"    Skip type : {p2_winner}")
    print(f"    Conv type : {p3_winner}")
    print(f"    + Transformer bottleneck + Bilinear upsample + Boundary conditioning")

    # ── Phase 4: Loss ablation (fair equal-epoch) ─────────────────────────────
    if args.phase in (0, 4) and not args.eval_only:
        p4_results, p4_histories = run_phase4(
            data, cfg, device, p1_winner, use_attn_skip, use_gated_conv,
            state=state, seed=cfg.train.seed,
        )
        plot_training_curves(p4_histories, results_dir, "Phase 4 Training Curves")
    else:
        p4_results   = {}
        p4_histories = {}

    # ── Multi-seed runs (L0 + L4, seeds 42/1/2) ───────────────────────────────
    if args.phase == 0 and not args.eval_only and not args.skip_multiseed:
        multi_results = run_multiseed_l0l4(
            data, cfg, device, p1_winner, use_attn_skip, use_gated_conv,
            seeds=(42, 1, 2), state=state,
        )
    else:
        multi_results = {}

    # ── ConvNeXt-Tiny runs ────────────────────────────────────────────────────
    if args.phase == 0 and not args.eval_only and not args.skip_convnext:
        convnext_results = run_convnext_runs(
            data, cfg, device, use_attn_skip, use_gated_conv,
            state=state, seed=cfg.train.seed,
        )
    else:
        convnext_results = {}

    # ── Evaluation ────────────────────────────────────────────────────────────
    phase4_eval, celeba_eval, dtd_eval, fid_scores = run_evaluation(
        data, cfg, device, p1_winner, use_attn_skip, use_gated_conv,
    )

    # ── Multi-seed statistics ─────────────────────────────────────────────────
    if multi_results:
        print("\n" + "="*60)
        print("MULTI-SEED STATISTICS  (mean ± std, 3 seeds)")
        print("="*60)
        seed_stats = aggregate_multiseed(
            multi_results, cfg, device, data, p1_winner, use_attn_skip, use_gated_conv,
        )
    else:
        seed_stats = {}

    # ── Sensitivity ───────────────────────────────────────────────────────────
    sensitivity = run_sensitivity(
        data, cfg, device, p1_winner, use_attn_skip, use_gated_conv,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating result plots...")
    plot_ablation_table(phase4_eval, results_dir)
    plot_sensitivity(sensitivity, results_dir)

    try:
        l0_path = Path(cfg.paths.checkpoints_dir) / "phase4_L0_base.pt"
        l4_path = Path(cfg.paths.checkpoints_dir) / "phase4_L4_full_method.pt"
        if l0_path.exists() and l4_path.exists():
            mc0 = make_model_cfg(p1_winner, use_attn_skip, use_gated_conv, False)
            mc4 = make_model_cfg(p1_winner, use_attn_skip, use_gated_conv, True)
            m0  = BoundaryAwareInpainter(mc0).to(device)
            m4  = BoundaryAwareInpainter(mc4).to(device)
            m0.load_state_dict(torch.load(l0_path, map_location=device, weights_only=True))
            m4.load_state_dict(torch.load(l4_path, map_location=device, weights_only=True))
            plot_qualitative(m0, m4, data["fast_test_loader"], device, results_dir)
            del m0, m4
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Qualitative plot warning: {e}")

    # ── Save all results ──────────────────────────────────────────────────────
    all_results = {
        "phase1_winner":   p1_winner,
        "phase2_winner":   p2_winner,
        "phase3_winner":   p3_winner,
        "phase4_eval":     phase4_eval,
        "celeba_eval":     celeba_eval,
        "dtd_eval":        dtd_eval,
        "fid_scores":      fid_scores,
        "seed_stats":      seed_stats,
        "sensitivity":     sensitivity,
        "speed_settings": {
            "profile":         args.profile,
            "test_fraction":   cfg.speed.test_fraction,
            "epochs_per_phase":cfg.speed.epochs_per_phase,
            "epochs_full":     cfg.speed.epochs_full,
            "val_batches":     cfg.speed.val_batches,
        },
    }
    save_results(all_results, cfg.paths.results_dir, "all_results")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    if "L4_full_method" in phase4_eval:
        g = phase4_eval["L4_full_method"]["global"]
        print("\nFinal model (L4) test metrics (Places365):")
        for k in ("psnr", "ssim", "lpips", "hole_mae", "boundary_mae",
                  "spectral_coherence", "fid"):
            if k in g:
                print(f"  {k}: {g[k]:.4f}")
    if "L4_full_method" in seed_stats:
        s = seed_stats["L4_full_method"]
        print("\nL4 multi-seed (3 seeds, mean ± std):")
        for k in ("psnr", "ssim", "boundary_mae"):
            if k in s:
                print(f"  {k}: {s[k]['mean']:.4f} ± {s[k]['std']:.4f}")
    print(f"\nResults     → {cfg.paths.results_dir}/")
    print(f"Checkpoints → {cfg.paths.checkpoints_dir}/")


if __name__ == "__main__":
    main()


