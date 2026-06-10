# Boundary-Aware Inpainting — DICTA Submission Plan

## 1. What We Are Building

A **boundary-aware image inpainting pipeline** that explicitly supervises the
transition seam between inpainted and known regions using:

- Gradient-weighted boundary loss (novel supervision signal)
- Spectral Boundary Coherence Loss (localized FFT on boundary patches only;
  distinct from global Focal Frequency Loss)
- **[NEW] Adaptive Spectral Boundary Coherence Loss (ASBC)** — learnable
  per-frequency-band weights over the spectral loss (primary paper novelty)
- Transformer bottleneck for global context (addresses LaMa's receptive-field advantage)
- Mask-injected skip gates & boundary-conditioning modules throughout the decoder

Primary dataset: **Places365 val_256** (~2.4 GB, 36,500 images)  
Cross-domain evaluation: **CelebA-HQ 256** (faces), **DTD** (textures) — zero-shot only

Compute budget: **single A100 80 GB VRAM**, hard cap **50 h**.  
Estimated core runtime with all optimizations: **~16 h**.

---

## 2. Novel Contribution: ASBC

> "Adaptive Spectral Boundary Coherence Loss"

**Claim:** Different frequency bands contribute unequally to visible seam artifacts.
Low frequencies govern colour continuity; high frequencies govern texture sharpness.
A fixed uniform spectral loss treats all bands equally; ASBC learns their optimal
weighting end-to-end.

**Mechanism:**

- `n_bands` learnable log-weight parameters (softmax-normalized) over radial FFT bins
- Applied to the same boundary patches as the existing spectral loss
- Anti-collapse entropy regularizer (−λ H(w)) prevents trivial weight collapse
- Vectorized patch FFT (single `torch.fft.fft2` call over all B×K patches)

**Ablation rows:**

- L3b: spectral with fixed uniform weights (existing)
- L3c: spectral with learned ASBC weights (new)

---

## 3. Code Changes Overview

### 3.1 requirements.txt

Add: `clean-fid>=0.1.35`, `huggingface_hub>=0.26`, `gdown>=5.0`, `datasets>=2.18`

### 3.2 configs/config.py

| Change                                                     | Reason                       |
| ---------------------------------------------------------- | ---------------------------- |
| Add `adaptive_spectral: bool`, `spectral_n_bands: int = 4` | ASBC config                  |
| Add `spectral_reg_weight: float = 0.01`                    | entropy reg coefficient      |
| Add `use_ema: bool = True`, `ema_decay: float = 0.999`     | EMA weights                  |
| Add `use_bf16: bool = True`                                | A100 bf16 > fp16             |
| Add `val_every_n_epochs: int = 2`                          | reduce eval overhead         |
| `patience: int = 2 → 8`                                    | prevent false early-stopping |
| Add production/smoke config factories                      | runtime profiles             |
| Add `dtd_dir: str = "./datasets/dtd"`                      | 3rd cross-domain set         |
| Add `celeba_dir` to defaults                               | was missing                  |
| `test_fraction: 0.001 → 1.0` in production profile         | use full dataset             |

### 3.3 losses/losses.py

1. **Vectorize `SpectralBoundaryCoherenceLoss`**: eliminate Python loop over K patches;
   stack all patches to (N, C, ps, ps) → single `torch.fft.fft2` → reshape.
2. **Add `AdaptiveSpectralBoundaryCoherenceLoss`** (ASBC): learnable frequency-band
   weights with entropy regularization.
3. Wire ASBC into `InpaintingLossManager` under loss config `"adaptive_spectral"`.

### 3.4 models/architecture.py

Add **ConvNeXt-Tiny** backbone option (timm):

- `timm.create_model("convnext_tiny", pretrained=True, features_only=True)`
- Patch stem Conv2d: 3 → 4 channels (copy channel 0 weight for new channel)
- 1×1 projection adapters map ConvNeXt feature dims (96/192/384/768) to
  standard decoder dims (64/128/256/512) — keeps decoder architecture identical
- Gate behind `cfg.backbone == "convnext_tiny"`

### 3.5 training/trainer.py

| Change                                                                      | Reason                                |
| --------------------------------------------------------------------------- | ------------------------------------- |
| Remove all `tqdm` imports / usage                                           | user request; plain epoch prints only |
| `torch.bfloat16` autocast (not fp16)                                        | A100 native; no GradScaler needed     |
| `torch.backends.cuda.matmul.allow_tf32 = True`                              | free ~1.5× speedup                    |
| `torch.backends.cudnn.allow_tf32 = True`                                    | free speedup                          |
| `cudnn.benchmark = True`                                                    | fixed input shape → faster kernels    |
| `model.to(memory_format=torch.channels_last)`                               | A100 NHWC path                        |
| **EMA** weight tracking (decay=0.999)                                       | better final metrics                  |
| Validate every `val_every_n_epochs` epochs                                  | reduce eval overhead                  |
| Save `training_curves.json` per run                                         | figure generation                     |
| Plain per-epoch print with all loss terms + val metrics + LR + elapsed time | visibility                            |
| Log ASBC band weights to epoch print when adaptive_spectral=True            | debugging                             |

#### Per-epoch print format

```
Epoch 012/050 | train total=0.1234  hole=0.0456  valid=0.0123  bound=0.0321  spec=0.0234  tv=0.0089 | val total=0.1456  psnr=27.34  ssim=0.8821  hole_mae=0.0234  bound_mae=0.0123 | lr=2.99e-04 | 127s
```

### 3.6 evaluation/metrics.py

1. **Add FID** via `cleanfid`: save N real/predicted images to temp dirs, call
   `clean_fid.compute_fid(real_dir, fake_dir)`.
2. Remove `tqdm` from evaluation loops (replace with plain batch-count print).
3. Add `compute_fid_from_loader()` helper.

### 3.7 main.py

| Change                                                               | Reason                      |
| -------------------------------------------------------------------- | --------------------------- |
| **Fix fairness bug**: ALL L0–L4 use `epochs_full`                    | ablation table was invalid  |
| Add L3c (ASBC) ablation row                                          | paper novelty               |
| Multi-seed (seeds 42,1,2) for L0 and L4 only                         | statistical credibility     |
| ConvNeXt-T runs: L0+L4, seed 42                                      | architecture-agnostic claim |
| Cross-domain eval for ALL phase-4 configs (not just L0/L4)           | stronger claim              |
| DTD cross-domain eval (same as CelebA-HQ)                            | 3rd domain                  |
| `pipeline_state.json` read/write; skip completed (name, seed) tuples | resume safety               |
| Wire FID calls for L4 and ConvNeXt runs                              | modern eval standard        |
| Use EMA model for final test metrics                                 | better numbers              |

### 3.8 data/download.py (NEW)

Idempotent downloaders with `{dir}/.download_complete` sentinel:

- `download_places365(root, drive_cache=None)` — MIT HTTP download, extract tar
- `download_celeba_hq(root, drive_cache=None)` — HuggingFace Hub snapshot
- `download_dtd(root, drive_cache=None)` — torchvision DTD autodownload
- `download_all(base_dir, drive_cache=None)` — calls all three

### 3.9 run.py (NEW)

Resume-safe experiment driver:

- Reads/writes `results/pipeline_state.json` — skip `(name, seed)` already done
- Full experiment matrix: phase1-3 selection → phase4 L0–L4+L3c → multi-seed L0/L4
  → ConvNeXt L0/L4 → sensitivity ablation
- Mirrors stdout to `results/run_pipeline.log`
- Plain prints: `[run N/M] name=... seed=... status=...`

### 3.10 colab_setup.py (REWRITE)

Project-specific Drive↔SSD sync for _boundary inpainting_ project:

- Drive root: `/content/drive/MyDrive/boundary_inpainting/`
- Phase map: datasets, checkpoints, results
- `init()`: mount → restore code → restore checkpoints+results from Drive
- `save(tag)`: compress and upload checkpoints+results to Drive
- `auto_save(interval_min=15)`: background thread auto-save
- `require_data()`: guard that raises if Places365 not present
- `incremental_save_results()`: save only results/\* to Drive

### 3.11 colab_pipeline.ipynb (REWRITE)

Three cells for the boundary inpainting project:

1. **[Download-only / CPU cell]** Install deps → mount Drive → `download_all()` →
   GPU check → if no GPU: "switch to A100" exit
2. **[Main run cell]** `init()` restore → `subprocess.Popen(["python","-u","run.py"])`
   with streaming stdout → `save("final")` → print summary table
3. **[Optional smoke cell]** 5-epoch sanity check, assert losses decrease, ASBC
   weights diverge from uniform
4. **[Optional figures cell]** aggregate JSONs, render ablation table + curves

---

## 4. Experiment Matrix

| ID  | Name                        | Backbone   | Loss       | Seeds    | Cross-domain | Notes                  |
| --- | --------------------------- | ---------- | ---------- | -------- | ------------ | ---------------------- |
| P1  | Phase 1: backbone selection | R18 vs R34 | base       | 42       | –            | selection only         |
| P2  | Phase 2: skip gate          | winner P1  | base       | 42       | –            | selection only         |
| P3  | Phase 3: gated conv         | winner P2  | base       | 42       | –            | selection only         |
| L0  | base                        | R34        | base       | 42, 1, 2 | CelebA+DTD   | ablation anchor        |
| L1  | boundary_uniform            | R34        | bound_uni  | 42       | CelebA+DTD   | ablation               |
| L2  | boundary_grad               | R34        | bound_grad | 42       | CelebA+DTD   | ablation               |
| L3  | spectral_only               | R34        | spectral   | 42       | CelebA+DTD   | ablation               |
| L3c | **adaptive_spectral**       | R34        | ASBC       | 42       | CelebA+DTD   | **PAPER NOVELTY**      |
| L4  | full_method                 | R34        | full       | 42, 1, 2 | CelebA+DTD   | full system            |
| C0  | convnext_base               | ConvNeXt-T | base       | 42       | CelebA+DTD   | arch-agnostic          |
| C4  | convnext_full               | ConvNeXt-T | full       | 42       | CelebA+DTD   | arch-agnostic          |
| SEN | sensitivity                 | R34        | full       | 42       | –            | eval-only from L4 ckpt |

**Total training runs**: 3 (P1-3) + 7 phase4 + 3 multi-seed + 2 ConvNeXt = **15 train runs**

---

## 5. Runtime Estimate (A100 80 GB)

| Phase                   | Runs | Epochs each | Hours     |
| ----------------------- | ---- | ----------- | --------- |
| Phase 1-3 selection     | 6    | 5           | ~1.5 h    |
| Phase 4 L0-L4+L3c       | 7    | 50          | ~10 h     |
| Multi-seed L0/L4        | 4    | 50          | ~5 h      |
| ConvNeXt L0+L4          | 2    | 50          | ~2 h      |
| Sensitivity (eval only) | 1    | eval        | ~0.3 h    |
| **Total**               |      |             | **~19 h** |

Well within 50 h hard limit. Buffer ~31 h for reruns / extended evaluation.

---

## 6. Speed Optimizations (A100 Specific)

All applied together in `trainer.py`:

| Optimization                                                  | Expected Gain                  |
| ------------------------------------------------------------- | ------------------------------ |
| `torch.bfloat16` autocast (A100 native)                       | ~1.5–2× vs fp16 on A100        |
| TF32 matmul + cudnn                                           | ~1.5× free (enable 2 flags)    |
| `channels_last` memory format (NHWC)                          | ~15–20% CNN speedup            |
| `cudnn.benchmark = True`                                      | ~5–10% for fixed input shape   |
| Vectorized patch FFT (batched rfft2)                          | ~3–5× speedup on spectral loss |
| Validate every 2 epochs (not every 1)                         | ~1.4× overall loop speedup     |
| `num_workers=4`, `pin_memory=True`, `persistent_workers=True` | dataloader                     |

**Ceiling notes:**

- Multi-GPU not needed; single A100 is sufficient and simplifies pipeline
- Gradient checkpointing not needed; model fits easily in 80 GB
- Flash Attention optional (transformer bottleneck is small, 8×8=64 tokens)

---

## 7. Metrics Reported

| Metric                       | Domain              | Notes          |
| ---------------------------- | ------------------- | -------------- |
| PSNR                         | Global              | standard       |
| SSIM                         | Global              | standard       |
| LPIPS (AlexNet)              | Global              | perceptual     |
| Hole-MAE                     | Hole region         | our focus      |
| **Boundary-MAE**             | ±r px band          | **our metric** |
| **Spectral Coherence Score** | boundary patches    | **our metric** |
| FID                          | Global              | added          |
| Per-difficulty bins          | easy/mid/hard/xhard | mask coverage  |

Mean ± std across 3 seeds reported for L0 and L4 (anchor + full method).

---

## 8. What Is Done / Not Done

### Done (before this session)

- [x] Full model architecture (ResNet encoder, transformer bottleneck, decoder, skip gates)
- [x] Loss stack (hole/valid/perceptual/boundary_uniform/boundary_grad/spectral/TV)
- [x] SpectralBoundaryCoherenceLoss (localized FFT — novel vs FFL)
- [x] Training loop with AMP, AdamW, CosineWarmup, early stopping
- [x] Dataset pipeline (Places365, CelebA-HQ, masks, boundary band)
- [x] Metrics: PSNR, SSIM, LPIPS, Hole-MAE, Boundary-MAE, Spectral Coherence
- [x] Multi-scale auxiliary heads at /8 and /4
- [x] BoundaryConditioningModule (FiLM-style decoder injection)
- [x] Phase 1-3 pipeline structure (backbone/skip/conv selection)

### Not Done (this session implements)

- [ ] **ASBC loss** (primary paper novelty)
- [ ] **Vectorized FFT** (speed)
- [ ] **ConvNeXt-Tiny backbone** (architecture-agnostic claim)
- [ ] **bf16 + TF32 + channels_last** (A100 optimizations)
- [ ] **EMA** (better final numbers)
- [ ] **FID metric** (modern eval standard)
- [ ] **Fair equal-epoch ablation** (correctness bug fix)
- [ ] **Multi-seed reporting** (statistical credibility)
- [ ] **DTD cross-domain eval** (third domain)
- [ ] **Pipeline resume safety** (pipeline_state.json)
- [ ] **Automatic dataset download** (data/download.py)
- [ ] **Colab run script** (run.py + colab files)
- [ ] **Per-epoch plain prints** (no tqdm)

---

## 9. Key Literature Distinctions

| Our Component                 | Related Work                      | Key Distinction                                         |
| ----------------------------- | --------------------------------- | ------------------------------------------------------- |
| SpectralBoundaryCoherenceLoss | Focal Frequency Loss (Jiang 2021) | LOCAL (boundary patches only) vs GLOBAL                 |
| ASBC                          | FFL, CoModGAN                     | Adaptive per-band weights; no analog in inpainting      |
| BoundaryConditioningModule    | SPADE (Park 2019)                 | Boundary band as conditioning signal (not semantic map) |
| MaskInjectedSkipGate          | Attention U-Net (Oktay 2018)      | Mask channel as explicit third gating signal            |
| TransformerBottleneck         | LaMa (Suvorov 2022), Restormer    | Bottleneck-only; preserves boundary-focused decoder     |
| GradientWeightedBoundaryLoss  | EdgeConnect (Nazeri 2019)         | Weighting signal (not a separate prediction stage)      |

---

## 10. Checklist Before Paper Submission

- [ ] All metrics reproduced across 3 seeds for L0 and L4
- [ ] ASBC weight histograms saved (show non-uniform learned distribution)
- [ ] Ablation table filled (L0 → L1 → L2 → L3 → L3c → L4)
- [ ] Cross-domain table filled (Places → CelebA-HQ → DTD)
- [ ] Qualitative figures: 4+ image pairs with boundary overlay
- [ ] Parameter count table (R34 vs ConvNeXt-T)
- [ ] Sensitivity table (K, patch_size, dilation r)
- [ ] Training curves figure (val loss per config)
- [ ] FID included in comparison table
- [ ] All runs used same dataset seed 42 for fair comparison (multi-seed is separate)
