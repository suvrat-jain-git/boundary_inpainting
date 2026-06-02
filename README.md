# Boundary-Aware Semantic Image Inpainting with Spectral Coherence Supervision

## Project Structure

```
inpainting/
├── main.py                    # Entry point — run this
├── requirements.txt
├── configs/
│   └── config.py              # All hyperparameters (documented with justification)
├── data/
│   └── dataset.py             # Mask generation, dataset, dataloaders
├── models/
│   └── architecture.py        # Full model: encoder + transformer bottleneck + decoder
├── losses/
│   └── losses.py              # All loss functions (spectral, boundary, perceptual, TV)
├── training/
│   └── trainer.py             # Training loops, scheduler, early stopping
├── evaluation/
│   └── metrics.py             # PSNR, SSIM, LPIPS, Boundary-MAE, Spectral Coherence
├── utils/
│   └── visualize.py           # All plots and tables
├── data/                      # Put your datasets here
│   ├── places365/
│   └── celeba_hq/
├── checkpoints/               # Auto-created
├── results/                   # Auto-created
└── logs/                      # Auto-created
```

---

## Setup (Windows 11)

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install PyTorch (CUDA 12.1 — check https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install other dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

### Places365
Download from http://places2.csail.mit.edu/
Place images under `./data/places365/` (any subfolder structure is fine — the code uses `rglob`)

### CelebA-HQ (optional — for cross-domain eval)
Download from https://github.com/tkarras/progressive_growing_of_gans
Place images under `./data/celeba_hq/`

---

## Running the Pipeline

```bash
# Full pipeline (all 4 phases + evaluation + sensitivity)
python main.py --places_dir ./data/places365

# With custom batch size (reduce if OOM)
python main.py --places_dir ./data/places365 --batch_size 4

# Windows: if DataLoader errors, use num_workers=0
python main.py --places_dir ./data/places365 --num_workers 0

# Only Phase 4 (uses existing phase 1-3 checkpoints)
python main.py --phase 4

# Evaluate only (skip training, load checkpoints)
python main.py --eval_only

# Sensitivity ablation only (needs L4 checkpoint)
python main.py --sensitivity

# CPU only (no GPU)
python main.py --no_amp
```

---

## Architecture Overview

```
Input (RGB + Mask)
       ↓
  ResNet-34 Encoder          ← pretrained, 4-channel modified
       ↓
  TransformerBottleneck      ← NEW: MHA self-attention for global context
       ↓
  Decoder (×4 stages)
    ├── BilinearUpsample      ← NEW: no checkerboard artifacts
    ├── MaskInjectedSkipGate  ← attention-gated skip connections
    ├── GatedConv2d           ← selective feature processing
    └── BoundaryConditioning  ← NEW: boundary band fed into each stage
       ↓
  Output (RGB, Sigmoid)
```

### Why each new component:
- **TransformerBottleneck**: CNN receptive field is limited at bottleneck; self-attention gives global context needed for large holes (same problem LaMa solves with Fourier convolutions, but applied surgically at bottleneck only)
- **BilinearUpsample**: ConvTranspose2d produces checkerboard artifacts (Odena et al. 2016) — especially damaging near inpainting boundaries
- **BoundaryConditioning**: Ties the architecture to our theoretical contribution — if the loss supervises the boundary, the architecture should also reason about it explicitly

---

## Loss Functions

| Config | Terms Active |
|--------|-------------|
| `base` | Hole-L1 + Valid-L1 + Perceptual |
| `boundary_uniform` | base + uniform boundary L1 |
| `boundary_grad` | base + gradient-weighted boundary L1 |
| `spectral_only` | base + spectral coherence |
| `full` | all + TV + multi-scale boundary |

### Spectral Loss vs FFL (Focal Frequency Loss):
- FFL applies spectral loss **globally** across the whole image
- Our loss samples patches **only within the boundary band**
- This localises spectral supervision exactly where seam artifacts occur

---

## Hyperparameter Justification

All hyperparameters are chosen via sensitivity ablation (run `--sensitivity`):

| Parameter | Value | Justification |
|-----------|-------|---------------|
| K (spectral patches) | 8 | Grid search over [4,8,16,32]; diminishing returns after 8 |
| Patch size | 16 | Captures local texture; 32 loses boundary specificity |
| Dilation r | 5 | Covers transition zone; r>7 dilutes signal |
| Boundary weight | 3.0 | Ablated; 3× hole weight balances local vs global |
| Spectral weight | 1.0 | Equal weight; ablated over [0.1, 0.5, 1.0, 2.0] |

---

## Evaluation Metrics

| Metric | Description | Higher/Lower |
|--------|-------------|-------------|
| PSNR | Global pixel quality (dB) | ↑ better |
| SSIM | Structural similarity | ↑ better |
| LPIPS | Perceptual distance (AlexNet) | ↓ better |
| Hole-MAE | L1 error in inpainted region | ↓ better |
| Boundary-MAE | L1 error in boundary band | ↓ better |
| Spectral Coherence | FFT cross-correlation at boundary | ↑ better |

---

## Baseline Comparison

To compare against LaMa or EdgeConnect:
1. Download pretrained weights from their repos
2. Set paths in `configs/config.py` → `EvalConfig.lama_checkpoint`
3. The `BaselineComparator` class in `evaluation/metrics.py` handles loading and evaluation

---

## Windows-Specific Notes

- Always run inside `if __name__ == '__main__':` — `main.py` handles this
- If `RuntimeError: DataLoader worker ...`: set `--num_workers 0`
- For CUDA OOM: reduce `--batch_size 4` or `--batch_size 2`
- AMP (mixed precision) is auto-enabled on CUDA, disabled on CPU
- Paths use `pathlib.Path` which is Windows-compatible
