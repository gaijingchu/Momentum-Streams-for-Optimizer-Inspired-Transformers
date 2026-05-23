"""Dump learned per-layer scalars from TMMFormer checkpoint."""
import torch
import torch.nn.functional as F

ckpt = torch.load("checkpoints_tmm/best.pt", map_location="cpu", weights_only=False)
sd = ckpt["model"]

print(f"step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
print()
print(f"{'Layer':<6}{'mu_a':>8}{'beta_a':>9}{'gamma_a':>10}{'nu_a':>9}  | {'mu_m':>8}{'beta_m':>9}{'gamma_m':>10}{'nu_m':>9}")
print("-" * 86)

sig = lambda x: torch.sigmoid(x).item()
sp = lambda x: F.softplus(x).item()

for i in range(12):
    g = lambda k: sd[f"layers.{i}.{k}"]
    mu_a = sig(g("mu_attn_raw"))
    be_a = sig(g("beta_attn_raw"))
    ga_a = sp(g("gamma_attn_raw"))
    nu_a = sp(g("nu_attn_raw"))
    mu_m = sig(g("mu_mlp_raw"))
    be_m = sig(g("beta_mlp_raw"))
    ga_m = sp(g("gamma_mlp_raw"))
    nu_m = sp(g("nu_mlp_raw"))
    print(f"L{i:<5}{mu_a:8.3f}{be_a:9.3f}{ga_a:10.3f}{nu_a:9.3f}  | {mu_m:8.3f}{be_m:9.3f}{ga_m:10.3f}{nu_m:9.3f}")

print()
print("Summary of nu (TMM iterate-update coefficient):")
nus = []
for i in range(12):
    nus.append(sp(sd[f"layers.{i}.nu_attn_raw"]))
    nus.append(sp(sd[f"layers.{i}.nu_mlp_raw"]))
import statistics
print(f"  count={len(nus)}, min={min(nus):.3f}, max={max(nus):.3f}, mean={statistics.mean(nus):.3f}, std={statistics.stdev(nus):.3f}")
print(f"  (initialization: nu = softplus(0.5413) = {sp(torch.tensor(0.5413)):.3f})")
print(f"  (YuriiFormer ≡ TMM with nu = 1)")
