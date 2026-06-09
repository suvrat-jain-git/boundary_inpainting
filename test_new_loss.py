# test_new_loss.py
# Run this to verify AdaptiveSpectralBoundaryLoss works correctly
# Uses fake random data — no GPU or real dataset needed

import torch
from losses.losses import AdaptiveSpectralBoundaryLoss

print("=" * 50)
print("Testing AdaptiveSpectralBoundaryLoss")
print("=" * 50)

# ── Create fake data (same shape as real training data) ──
B, C, H, W = 2, 3, 256, 256   # batch=2, RGB, 256x256

# Fake predicted image (random values between 0 and 1)
pred = torch.rand(B, C, H, W, requires_grad=True)

# Fake ground truth image
target = torch.rand(B, C, H, W)

# Fake mask (1 = hole, 0 = valid)
mask = torch.zeros(B, 1, H, W)
mask[:, :, 80:160, 80:160] = 1.0   # square hole in the middle

# Fake boundary band (ring around the hole)
boundary = torch.zeros(B, 1, H, W)
boundary[:, :, 75:165, 75:165] = 1.0
boundary[:, :, 80:160, 80:160] = 0.0   # hollow ring

print(f"pred shape:     {pred.shape}")
print(f"target shape:   {target.shape}")
print(f"mask shape:     {mask.shape}")
print(f"boundary shape: {boundary.shape}")
print()

# ── Test 1: Forward pass runs without error ──────────────
print("Test 1: Forward pass...")
loss_fn = AdaptiveSpectralBoundaryLoss(
    patch_size=16,
    n_samples=8,
    n_bands=4,
    entropy_weight=0.1,
)
loss = loss_fn(pred, target, mask, boundary)
print(f"  Loss value: {loss.item():.6f}")
print(f"  Loss is a number (not NaN): {not torch.isnan(loss)}")
print(f"  Loss is finite: {torch.isfinite(loss)}")
print("  PASSED" if (not torch.isnan(loss) and torch.isfinite(loss)) else "  FAILED")
print()

# ── Test 2: Backward pass works (gradients flow) ─────────
print("Test 2: Backward pass (gradients)...")
loss.backward()
print(f"  pred.grad is not None: {pred.grad is not None}")
print(f"  band_logits.grad: {loss_fn.band_logits.grad}")
print("  PASSED" if pred.grad is not None else "  FAILED")
print()

# ── Test 3: Band weights sum to 1 ────────────────────────
print("Test 3: Band weights sum to 1...")
weights = torch.softmax(loss_fn.band_logits, dim=0)
print(f"  Band weights: {weights.detach().numpy().round(3)}")
print(f"  Sum: {weights.sum().item():.6f}")
print("  PASSED" if abs(weights.sum().item() - 1.0) < 1e-5 else "  FAILED")
print()

# ── Test 4: Weights change after a fake gradient step ────
print("Test 4: Weights actually learn...")
optimizer = torch.optim.Adam([loss_fn.band_logits], lr=0.01)
initial_weights = torch.softmax(loss_fn.band_logits, dim=0).detach().clone()

for step in range(5):
    optimizer.zero_grad()
    pred2 = torch.rand(B, C, H, W, requires_grad=True)
    l = loss_fn(pred2, target, mask, boundary)
    l.backward()
    optimizer.step()

final_weights = torch.softmax(loss_fn.band_logits, dim=0).detach()
weights_changed = not torch.allclose(initial_weights, final_weights)
print(f"  Initial weights: {initial_weights.numpy().round(3)}")
print(f"  Final weights:   {final_weights.numpy().round(3)}")
print(f"  Weights changed: {weights_changed}")
print("  PASSED" if weights_changed else "  FAILED")
print()

# ── Test 5: Compare fixed vs adaptive loss values ────────
print("Test 5: Fixed spectral vs adaptive spectral...")
from losses.losses import SpectralBoundaryCoherenceLoss
fixed_loss_fn = SpectralBoundaryCoherenceLoss(patch_size=16, n_samples=8)
pred3 = torch.rand(B, C, H, W)
fixed_val = fixed_loss_fn(pred3, target, mask, boundary).item()
adaptive_val = loss_fn(pred3, target, mask, boundary).item()
print(f"  Fixed spectral loss:    {fixed_val:.6f}")
print(f"  Adaptive spectral loss: {adaptive_val:.6f}")
print("  Both ran successfully: PASSED")
print()

print("=" * 50)
print("All tests complete.")
print("=" * 50)