# run.py
"""
Resume-safe pipeline driver.

This is the canonical entry point for Colab / SLURM / local A100 runs.

    python run.py                         # full production run
    python run.py --profile smoke         # quick 5-epoch smoke test
    python run.py --profile production    # explicit production mode
    python run.py --skip_download         # skip dataset download check

All output is mirrored to results/run_pipeline.log.
Each experiment is tracked in results/pipeline_state.json —
interrupted runs can be safely resumed by re-running this script.

Expected A100 (85 GB) runtime (production):
  Phase 1-3 (selection): ~1 h
  Phase 4 ablation (6 configs × 50 epochs): ~12 h
  Multi-seed L0/L4 (2 extra seeds × 2 configs): ~5 h
  ConvNeXt C0+C4: ~4 h
  Evaluation + FID: ~2 h
  Total: ~24 h  (well under the 50 h DICTA hard limit)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Suppress tqdm progress bars from PyTorch model downloads and any other
# library that respects TQDM_DISABLE (set before any torch import).
os.environ.setdefault("TQDM_DISABLE", "1")

# ── Logging (stdout + file) ───────────────────────────────────────────────────

def _setup_logging(results_dir: str):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(results_dir) / "run_pipeline.log"
    fmt      = "%(asctime)s | %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
        force=True,
    )
    print(f"Logging to: {log_path}")


def log(msg: str):
    logging.info(msg)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(cfg, skip_download: bool = False,
                  skip_phases123: bool = True,
                  skip_multiseed: bool = False,
                  skip_convnext:  bool = False,
                  eval_only:      bool = False):
    import gc
    import random

    import numpy as np
    import torch

    from configs.config import Config
    from data.dataset import setup_data
    from evaluation.metrics import (
        ModelEvaluator, compute_fid_from_loader,
        run_sensitivity_ablation, save_results,
    )
    from main import (
        PHASE4_CONFIGS, PipelineState,
        aggregate_multiseed, make_model_cfg,
        run_convnext_runs, run_evaluation,
        run_multiseed_l0l4, run_phase1, run_phase2,
        run_phase3, run_phase4, run_sensitivity,
        seed_everything,
    )
    from models.architecture import BoundaryAwareInpainter
    from training.trainer import enable_a100_flags
    from utils.visualize import (
        plot_ablation_table, plot_sensitivity, plot_training_curves,
    )

    enable_a100_flags()
    cfg.make_dirs()

    state   = PipelineState(cfg.paths.results_dir)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = getattr(cfg.train, "use_bf16", True) and device.type == "cuda"
    results_dir = Path(cfg.paths.results_dir)

    log(f"Device:    {device}")
    if device.type == "cuda":
        log(f"GPU:       {torch.cuda.get_device_name(0)}")
        log(f"VRAM:      {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log(f"Precision: {'bfloat16' if use_bf16 else 'float32'}")
    log(f"epochs_per_phase = {cfg.speed.epochs_per_phase}")
    log(f"epochs_full      = {cfg.speed.epochs_full}  (all phase-4 configs, fair equal)")
    log(f"test_fraction    = {cfg.speed.test_fraction}")

    # ── Optional dataset download ──────────────────────────────────────────────
    if not skip_download:
        try:
            from data.download import download_all
            download_all(base_dir="./datasets")
        except Exception as e:
            log(f"Dataset download warning: {e}")

    # ── Data setup ────────────────────────────────────────────────────────────
    log("Setting up data...")
    data = setup_data(cfg)

    seed_everything(cfg.train.seed)

    # ── Phase 1-3: Architecture selection ─────────────────────────────────────
    if not skip_phases123:
        log("\n[Phase 1] Backbone selection")
        _, p1_winner = run_phase1(data, cfg, device)
        log(f"  -> winner: {p1_winner}")

        log("\n[Phase 2] Skip connection study")
        _, p2_winner, use_attn_skip = run_phase2(data, cfg, device, p1_winner)
        log(f"  -> winner: {p2_winner}  use_attn_skip={use_attn_skip}")

        log("\n[Phase 3] Gated conv study")
        _, p3_winner, use_gated_conv = run_phase3(
            data, cfg, device, p1_winner, use_attn_skip,
        )
        log(f"  -> winner: {p3_winner}  use_gated_conv={use_gated_conv}")
    else:
        p1_winner    = "resnet34"
        p2_winner    = "attention_mask"
        p3_winner    = "gated_conv"
        use_attn_skip  = True
        use_gated_conv = True
        log("\n[Phases 1-3 skipped] Using: resnet34 + attn_skip + gated_conv")

    # ── Phase 4: Loss ablation (all configs, equal epochs) ────────────────────
    if not eval_only:
        log("\n[Phase 4] Loss ablation (6 configs × epochs_full={})".format(cfg.speed.epochs_full))
        p4_results, p4_histories = run_phase4(
            data, cfg, device, p1_winner, use_attn_skip, use_gated_conv,
            state=state, seed=cfg.train.seed,
        )
        try:
            plot_training_curves(p4_histories, results_dir, "Phase 4 Training Curves")
        except Exception as e:
            log(f"  Training curves plot warning: {e}")
    else:
        p4_results, p4_histories = {}, {}
        log("\n[eval_only] Skipping Phase 4 training — loading existing checkpoints.")

    # ── Multi-seed runs (L0 + L4, seeds 42/1/2) ───────────────────────────────
    if not skip_multiseed and not eval_only:
        log("\n[Multi-seed] L0 + L4 with seeds [42, 1, 2]")
        multi_results = run_multiseed_l0l4(
            data, cfg, device, p1_winner, use_attn_skip, use_gated_conv,
            seeds=(42, 1, 2), state=state,
        )
    else:
        multi_results = {}

    # ── ConvNeXt runs ─────────────────────────────────────────────────────────
    if not skip_convnext and not eval_only:
        log("\n[ConvNeXt] C0_base + C4_full with convnext_tiny backbone")
        convnext_results = run_convnext_runs(
            data, cfg, device, use_attn_skip, use_gated_conv,
            state=state, seed=cfg.train.seed,
        )
    else:
        convnext_results = {}

    # ── Evaluation ────────────────────────────────────────────────────────────
    log("\n[Evaluation] Test set metrics + cross-domain + FID")
    phase4_eval, celeba_eval, dtd_eval, fid_scores = run_evaluation(
        data, cfg, device, p1_winner, use_attn_skip, use_gated_conv,
    )

    # ── Multi-seed statistics ──────────────────────────────────────────────────
    log("\n[Multi-seed stats] mean ± std over 3 seeds")
    seed_stats = aggregate_multiseed(
        multi_results, cfg, device, data, p1_winner, use_attn_skip, use_gated_conv,
    )

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    log("\n[Sensitivity] k / patch_size / dilation ablation")
    sensitivity = run_sensitivity(
        data, cfg, device, p1_winner, use_attn_skip, use_gated_conv,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    log("\nGenerating plots...")
    try:
        plot_ablation_table(phase4_eval, results_dir)
        plot_sensitivity(sensitivity, results_dir)
    except Exception as e:
        log(f"  Plot warning: {e}")

    # ── Save ──────────────────────────────────────────────────────────────────
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
            "test_fraction":   cfg.speed.test_fraction,
            "epochs_per_phase":cfg.speed.epochs_per_phase,
            "epochs_full":     cfg.speed.epochs_full,
        },
    }
    save_results(all_results, cfg.paths.results_dir, "all_results")

    # ── Summary print ─────────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("PIPELINE COMPLETE")
    log("="*60)
    if "L4_full_method" in phase4_eval:
        g = phase4_eval["L4_full_method"]["global"]
        log("\nL4 full method (Places365):")
        for k in ("psnr", "ssim", "lpips", "hole_mae", "boundary_mae",
                  "spectral_coherence", "fid"):
            if k in g:
                log(f"  {k}: {g[k]:.4f}")
    if "L4_full_method" in seed_stats:
        s = seed_stats["L4_full_method"]
        log("\nL4 multi-seed (3 seeds, mean ± std):")
        for k in ("psnr", "ssim", "boundary_mae"):
            if k in s:
                log(f"  {k}: {s[k]['mean']:.4f} ± {s[k]['std']:.4f}")
    log(f"\nResults     -> {cfg.paths.results_dir}/")
    log(f"Checkpoints -> {cfg.paths.checkpoints_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile",       type=str, default="production",
                        choices=["default", "production", "smoke"])
    parser.add_argument("--skip_download",  action="store_true",
                        help="Skip dataset download check")
    parser.add_argument("--skip_phases123", action="store_true", default=True,
                        help="Skip phases 1-3 (default: True — use resnet34+attn+gated)")
    parser.add_argument("--run_phases123",  action="store_true",
                        help="Explicitly run phases 1-3 (overrides skip_phases123)")
    parser.add_argument("--skip_multiseed", action="store_true",
                        help="Skip multi-seed L0/L4 runs")
    parser.add_argument("--skip_convnext",  action="store_true",
                        help="Skip ConvNeXt backbone runs")
    parser.add_argument("--eval_only",      action="store_true",
                        help="Skip all training, run evaluation+plots only")
    parser.add_argument("--places_dir",    type=str, default=None)
    parser.add_argument("--celeba_dir",    type=str, default=None)
    parser.add_argument("--dtd_dir",       type=str, default=None)
    args = parser.parse_args()

    # Import here to avoid circular import issues
    from configs.config import get_config, get_production_config, get_smoke_config

    if args.profile == "production":
        cfg = get_production_config()
    elif args.profile == "smoke":
        cfg = get_smoke_config()
    else:
        cfg = get_config()

    if args.places_dir: cfg.data.places365_dir = args.places_dir
    if args.celeba_dir: cfg.data.celeba_dir    = args.celeba_dir
    if args.dtd_dir:    cfg.data.dtd_dir       = args.dtd_dir

    cfg.make_dirs()
    _setup_logging(cfg.paths.results_dir)
    t0 = time.time()

    try:
        run_pipeline(
            cfg,
            skip_download=args.skip_download,
            skip_phases123=args.skip_phases123 and not args.run_phases123,
            skip_multiseed=args.skip_multiseed,
            skip_convnext=args.skip_convnext,
            eval_only=args.eval_only,
        )
    except KeyboardInterrupt:
        log("\nInterrupted — pipeline state saved. Re-run to resume.")
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

    elapsed = (time.time() - t0) / 3600
    log(f"\nTotal wall time: {elapsed:.2f} h")
