# main.py
"""
Main pipeline entry point.

Usage:
    python main.py                                   # full pipeline
    python main.py --phase 4                         # only phase 4
    python main.py --eval_only                       # evaluate checkpoints only
    python main.py --places_dir ./datasets/places365 # custom dataset path

Speed control — edit configs/config.py → SpeedConfig:
    test_fraction  : fraction of dataset used (0.001 = tiny test, 1.0 = full)
    epochs_per_phase: epochs for phases 1-3 and L0-L3
    epochs_full    : epochs for L4 final model
    val_batches    : max val batches per epoch (None = all)

Windows note: always run as  python main.py  (not as a module)
"""

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import Config, ModelConfig, get_config
from data.dataset import setup_data
from evaluation.metrics import ModelEvaluator, run_sensitivity_ablation, save_results
from losses.losses import InpaintingLossManager
from models.architecture import BoundaryAwareInpainter, build_model
from training.trainer import load_or_train, run_experiment, validate
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        )
        results[backbone] = result
        if "model" in result:
            del result["model"]
        torch.cuda.empty_cache()
        gc.collect()

    winner = min(results, key=lambda k: results[k]["best_val_loss"])
    print(f"\n✓ Phase 1 Winner: {winner} "
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
        )
        results[skip_type] = result
        if "model" in result:
            del result["model"]
        torch.cuda.empty_cache()
        gc.collect()

    winner     = min(results, key=lambda k: results[k]["best_val_loss"])
    use_attn   = (winner == "attention_mask")
    print(f"\n✓ Phase 2 Winner: {winner} "
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
        )
        results[conv_type] = result
        if "model" in result:
            del result["model"]
        torch.cuda.empty_cache()
        gc.collect()

    winner     = min(results, key=lambda k: results[k]["best_val_loss"])
    use_gated  = (winner == "gated_conv")
    print(f"\n✓ Phase 3 Winner: {winner} "
          f"(val_loss={results[winner]['best_val_loss']:.4f})")
    return results, winner, use_gated


# ── Phase 4: Loss Ablation ─────────────────────────────────────────────────────

def run_phase4(data, cfg, device, backbone, use_attn_skip, use_gated_conv):
    print("\n" + "="*60)
    print("PHASE 4: Loss Ablation Study")
    print("="*60)

    phase4_configs = {
        "L0_base":             {"loss_config": "base",             "multiscale": False, "epochs": cfg.speed.epochs_per_phase},
        "L1_boundary_uniform": {"loss_config": "boundary_uniform", "multiscale": False, "epochs": cfg.speed.epochs_per_phase},
        "L2_boundary_grad":    {"loss_config": "boundary_grad",    "multiscale": False, "epochs": cfg.speed.epochs_per_phase},
        "L3_spectral_only":    {"loss_config": "spectral_only",    "multiscale": False, "epochs": cfg.speed.epochs_per_phase},
        "L4_full_method":      {"loss_config": "full",             "multiscale": True,  "epochs": cfg.speed.epochs_full},
    }

    results   = {}
    histories = {}

    for exp_name, exp_cfg in phase4_configs.items():
        def factory(ms=exp_cfg["multiscale"]):
            c = make_model_cfg(backbone, use_attn_skip, use_gated_conv, multiscale=ms)
            return BoundaryAwareInpainter(c)

        result = load_or_train(
            name=f"phase4_{exp_name}",
            model_factory=factory,
            train_loader=data["fast_train_loader"],   # ← uses test_fraction subset
            val_loader=data["fast_val_loader"],        # ← uses test_fraction subset
            cfg=cfg,
            device=device,
            loss_config=exp_cfg["loss_config"],
            num_epochs=exp_cfg["epochs"],
            patience=cfg.train.patience,
            freeze_encoder_epochs=cfg.train.freeze_encoder_epochs,
        )
        results[exp_name]   = result
        histories[exp_name] = result.get("history", {})

        if "model" in result:
            del result["model"]
        torch.cuda.empty_cache()
        gc.collect()

    return results, histories, phase4_configs


# ── Evaluation ─────────────────────────────────────────────────────────────────

def run_evaluation(data, cfg, device, phase4_configs, backbone, use_attn_skip, use_gated_conv):
    print("\n" + "="*60)
    print("EVALUATION: Test Set Metrics")
    print("="*60)

    evaluator    = ModelEvaluator(device, cfg.data.mask_bins)
    phase4_eval  = {}

    for exp_name, exp_cfg in phase4_configs.items():
        ckpt_path = Path(cfg.paths.checkpoints_dir) / f"phase4_{exp_name}.pt"
        if not ckpt_path.exists():
            print(f"  ⚠ Checkpoint not found: {exp_name} — skipping")
            continue

        model_cfg = make_model_cfg(
            backbone, use_attn_skip, use_gated_conv,
            multiscale=exp_cfg["multiscale"],
        )
        model = BoundaryAwareInpainter(model_cfg).to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )

        print(f"\n  Evaluating {exp_name}...")
        # Use fast_test_loader to keep eval quick during testing
        g, bins = evaluator.evaluate(
            model, data["fast_test_loader"],
            use_amp=cfg.train.use_amp,
        )
        phase4_eval[exp_name] = {"global": g, "bins": bins}

        print(f"    PSNR={g.get('psnr', 0):.2f}  "
              f"SSIM={g.get('ssim', 0):.4f}  "
              f"Hole-MAE={g.get('hole_mae', 0):.4f}  "
              f"Boundary-MAE={g.get('boundary_mae', 0):.4f}  "
              f"Spectral={g.get('spectral_coherence', 0):.4f}")
        if "lpips" in g:
            print(f"    LPIPS={g['lpips']:.4f}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # CelebA-HQ (with boundary metrics — addresses reviewer Q10)
    celeba_eval = {}
    if data["celeba_loader"] is not None:
        print("\n  Cross-Domain Evaluation: CelebA-HQ")
        for exp_name in ["L0_base", "L4_full_method"]:
            ckpt_path = Path(cfg.paths.checkpoints_dir) / f"phase4_{exp_name}.pt"
            if not ckpt_path.exists():
                continue

            exp_cfg   = phase4_configs.get(exp_name, {})
            model_cfg = make_model_cfg(
                backbone, use_attn_skip, use_gated_conv,
                multiscale=exp_cfg.get("multiscale", False),
            )
            model = BoundaryAwareInpainter(model_cfg).to(device)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )

            g, _ = evaluator.evaluate(model, data["celeba_loader"], use_amp=cfg.train.use_amp)
            celeba_eval[f"{exp_name}_celeba"] = g
            print(f"    {exp_name}: PSNR={g.get('psnr',0):.2f}  "
                  f"Boundary-MAE={g.get('boundary_mae',0):.4f}  "
                  f"Spectral={g.get('spectral_coherence',0):.4f}")

            del model
            torch.cuda.empty_cache()

    return phase4_eval, celeba_eval


# ── Sensitivity Analysis ───────────────────────────────────────────────────────

def run_sensitivity(data, cfg, device, backbone, use_attn_skip, use_gated_conv):
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)

    ckpt_path = Path(cfg.paths.checkpoints_dir) / "phase4_L4_full_method.pt"
    if not ckpt_path.exists():
        print("  ⚠ L4 checkpoint not found — skipping sensitivity analysis")
        return {}

    model_cfg = make_model_cfg(backbone, use_attn_skip, use_gated_conv, multiscale=True)
    model = BoundaryAwareInpainter(model_cfg).to(device)
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
        use_amp=cfg.train.use_amp,
    )

    del model
    torch.cuda.empty_cache()
    return sensitivity


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Boundary-Aware Inpainting Pipeline")
    parser.add_argument("--phase",      type=int, default=0,
                        help="Run specific phase only (1-4). 0 = full pipeline.")
    parser.add_argument("--eval_only",  action="store_true",
                        help="Skip training, only evaluate checkpoints.")
    parser.add_argument("--sensitivity",action="store_true",
                        help="Run sensitivity ablation only.")
    parser.add_argument("--places_dir", type=str, default=None)
    parser.add_argument("--celeba_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers",type=int, default=None)
    parser.add_argument("--no_amp",     action="store_true")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = get_config()

    if args.places_dir:   cfg.data.places365_dir = args.places_dir
    if args.celeba_dir:   cfg.data.celeba_dir    = args.celeba_dir
    if args.batch_size:   cfg.data.batch_size    = args.batch_size
    if args.num_workers is not None: cfg.data.num_workers = args.num_workers
    if args.no_amp:       cfg.train.use_amp      = False

    cfg.make_dirs()

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"AMP:    {cfg.train.use_amp and device.type == 'cuda'}")
    print(f"\nSpeed settings:")
    print(f"  test_fraction   = {cfg.speed.test_fraction}")
    print(f"  epochs_per_phase= {cfg.speed.epochs_per_phase}")
    print(f"  epochs_full     = {cfg.speed.epochs_full}")
    print(f"  val_batches     = {cfg.speed.val_batches}")

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

    # ── Phases ────────────────────────────────────────────────────────────────
    if args.phase in (0, 1) and not args.eval_only:
        p1_results, p1_winner = run_phase1(data, cfg, device)
    else:
        p1_winner  = "resnet34"
        p1_results = {}

    if args.phase in (0, 2) and not args.eval_only:
        p2_results, p2_winner, use_attn_skip = run_phase2(data, cfg, device, p1_winner)
    else:
        p2_winner    = "attention_mask"
        use_attn_skip = True
        p2_results   = {}

    if args.phase in (0, 3) and not args.eval_only:
        p3_results, p3_winner, use_gated_conv = run_phase3(
            data, cfg, device, p1_winner, use_attn_skip
        )
    else:
        p3_winner    = "gated_conv"
        use_gated_conv = True
        p3_results   = {}

    print(f"\nArchitecture locked:")
    print(f"  Backbone  : {p1_winner}")
    print(f"  Skip type : {p2_winner}")
    print(f"  Conv type : {p3_winner}")
    print(f"  + Transformer bottleneck + Bilinear upsample + Boundary conditioning")

    if args.phase in (0, 4) and not args.eval_only:
        p4_results, p4_histories, phase4_configs = run_phase4(
            data, cfg, device, p1_winner, use_attn_skip, use_gated_conv
        )
        plot_training_curves(p4_histories, results_dir, "Phase 4 Training Curves")
    else:
        phase4_configs = {
            "L0_base":             {"loss_config": "base",             "multiscale": False, "epochs": cfg.speed.epochs_per_phase},
            "L1_boundary_uniform": {"loss_config": "boundary_uniform", "multiscale": False, "epochs": cfg.speed.epochs_per_phase},
            "L2_boundary_grad":    {"loss_config": "boundary_grad",    "multiscale": False, "epochs": cfg.speed.epochs_per_phase},
            "L3_spectral_only":    {"loss_config": "spectral_only",    "multiscale": False, "epochs": cfg.speed.epochs_per_phase},
            "L4_full_method":      {"loss_config": "full",             "multiscale": True,  "epochs": cfg.speed.epochs_full},
        }
        p4_results   = {}
        p4_histories = {}

    # ── Evaluation ────────────────────────────────────────────────────────────
    phase4_eval, celeba_eval = run_evaluation(
        data, cfg, device, phase4_configs,
        p1_winner, use_attn_skip, use_gated_conv,
    )

    # ── Sensitivity ───────────────────────────────────────────────────────────
    sensitivity = run_sensitivity(
        data, cfg, device, p1_winner, use_attn_skip, use_gated_conv
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

    # ── Save results ──────────────────────────────────────────────────────────
    all_results = {
        "phase1_winner":  p1_winner,
        "phase2_winner":  p2_winner,
        "phase3_winner":  p3_winner,
        "phase4_eval":    phase4_eval,
        "celeba_eval":    celeba_eval,
        "sensitivity":    sensitivity,
        "speed_settings": {
            "test_fraction":    cfg.speed.test_fraction,
            "epochs_per_phase": cfg.speed.epochs_per_phase,
            "epochs_full":      cfg.speed.epochs_full,
            "val_batches":      cfg.speed.val_batches,
        },
    }
    save_results(all_results, cfg.paths.results_dir, "all_results")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    if "L4_full_method" in phase4_eval:
        g = phase4_eval["L4_full_method"]["global"]
        print("\nFinal model (L4) test metrics:")
        for k, v in g.items():
            print(f"  {k}: {v:.4f}")
    print(f"\nResults     → {cfg.paths.results_dir}/")
    print(f"Checkpoints → {cfg.paths.checkpoints_dir}/")


if __name__ == "__main__":
    main()
