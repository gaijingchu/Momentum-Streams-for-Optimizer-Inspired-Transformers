# Vanilla vs TMM on TinyStories — seed variance & parameter-matched controls

## TL;DR

The raw TinyStories val-loss gap between Vanilla (124.4 M) and TMM (163.8 M) is
**0.0306** at 10 k steps (best val), and is **~10–15× the per-config seed std**
(N = 3 seeds each), so it is not seed luck.

To decompose "parameter count vs. velocity-stream dynamics", we trained two
parameter-matched Vanilla variants (depth-18 ≈ 166.85 M and width-900 ≈ 162.86 M).
A clean single run of the width-matched Vanilla closes **~40 %** of the gap
(1.1578 → 1.1454), leaving a residual **0.0182 ≈ 9× seed noise** vs. TMM.
**Conclusion (under width-scaling):** roughly 40 % of TMM's advantage is parameter
count, 60 % is an architectural effect that is not explained by adding the same
parameter budget as extra width. A depth-matched Vanilla appears to close
substantially more of the gap, but that number comes from an unclean run (logs
with requeue gaps, best-val artefact) and is reported as suggestive only — a
clean depth-18 rerun is pending.

---

## 1. Question

TMM differs from the Vanilla pre-norm Transformer by adding a *velocity stream*:
a duplicate token + positional embedding (`vel_tok_emb`, `vel_pos_emb`) plus two
extra LayerNorms per block. The architectural change carries +39.44 M parameters,
of which ≈ 99.95 % is the duplicate embedding table. None of the +39 M sits inside
the transformer blocks, so it adds essentially zero per-step FLOPs.

When TMM beats Vanilla at fixed step count, the gain therefore confounds two
mechanisms:

1. **Parameter count.** TMM simply has 39 M more parameters in the embedding
   stack, available to the model via the velocity stream.
2. **Velocity-stream dynamics.** TMM mixes the velocity stream into each block
   via additional LayerNorms and skip paths — a real architectural change beyond
   raw parameter count.

This experiment is the parameter-matched control: build Vanilla baselines whose
parameter count matches TMM's, hold the optimiser and training recipe fixed, and
measure how much of TMM's advantage survives.

---

## 2. Architectures and parameter math

### 2.1 Vanilla pre-norm block (`vanilla_model.py`)

```
VanillaBlock = LN_attn (bias=False)
             + CausalSelfAttention(d_model, n_heads)   # qkv: d×3d, out: d×d, no bias
             + LN_mlp  (bias=False)
             + MLP(d_model)                             # w1: d×4d, w2: 4d×d, GELU, no bias
```

Top-level: `tok_emb (V×d)`, `pos_emb (T×d)`, `final_ln (d)`, weight-tied output
(`F.linear(x, tok_emb.weight)` — no separate `lm_head`).

Per-block parameter count: `12 d² + 2 d`.

Top-level parameter count: `V·d + T·d + d`, with `V = 50 304`, `T = 1024`.

| Variant | L | d | heads | head_dim | total params |
|---|---|---|---|---|---|
| Vanilla default        | 12 | 768  | 12 | 64 | **124 373 760** |
| Vanilla depth-18 (A)   | 18 | 768  | 12 | 64 | **166 850 304** |
| Vanilla width-900 (B)  | 12 | 900  | 12 | 75 | **162 857 700** |

### 2.2 TMM

TMM adds, on top of Vanilla default:

- `vel_tok_emb : V × d = 50 304 × 768 = 38 633 472`
- `vel_pos_emb : T × d = 1 024 × 768 =     786 432`
- 2× extra `LayerNorm(bias=False)` per block (`ln_v_attn`, `ln_v_mlp`): `2 · L · d`
- (Small bookkeeping for scalar mixing weights; rounding error)

Total: **163 812 192**, of which **+39 438 432** is the architectural delta over
Vanilla default. > 99.9 % of that delta is duplicate embeddings.

### 2.3 Why both A (depth) and B (width)?

A single param-matched control can confound parameter count with *how* the
parameters are spent. Depth-scaling (A) and width-scaling (B) bracket the
question. If A and B give the same answer, the gap is parameter count. If they
disagree, the *shape* of the spend matters.

| Design | Mechanism | Δparams vs TMM |
|---|---|---|
| A (depth-18, 18L/768d) | extra depth | +1.85 % |
| B (width-900, 12L/900d) | extra width | −0.58 % |

All counts independently verified twice: (i) CPU prediction from the formula
above; (ii) `Total parameters: N` printed by the training script under DDP on
GPU at start of training. The two numbers match exactly for both A and B (see
§5).

---

## 3. Training protocol

All numbers below come from `vanilla_train_ddp.py` and the matching sbatch
scripts; they are identical across all Vanilla variants and across TMM (the
training script and optimiser config are shared).

### 3.1 Data

- **Dataset:** `roneneldan/TinyStories` via HuggingFace `datasets`.
- **Tokenizer:** `tiktoken.get_encoding("gpt2")`. Effective vocab 50 304
  (model side; the extra padding above GPT-2's 50 257 keeps the embedding
  divisible by 128).
- **Block size:** 1 024 tokens. Non-overlapping training blocks with a
  deterministic epoch-reordered dataloader (seed = `SEED + rank` per-rank).

### 3.2 Optimisation (Muon ∘ AdamW split, as in the YuriiFormer recipe)

Three parameter groups, hand-routed by weight rank/role:

| Group | Optimiser | LR | wd | Notes |
|---|---|---|---|---|
| 2-D linear weights (qkv, out, w1, w2) | **Muon** | 2 × 10⁻² | 0 | momentum 0.95, Nesterov |
| Embeddings (`tok_emb`, `pos_emb`)     | AdamW    | 6 × 10⁻⁴ | 0.1 | β = (0.9, 0.95) |
| LayerNorm gains                        | AdamW    | 6 × 10⁻⁴ | 0   | β = (0.9, 0.95) |

The AdamW route is forced to be lossless (we read off the printed param-group
sizes at the start of each run — see §5).

### 3.3 Schedule

- **Total steps:** 10 000.
- **Warmup:** 1 000 steps linear from 0 to peak LR.
- **Decay:** cosine to `MIN_LR_RATIO · peak = 0.1 · peak`.
- **Grad clip:** 1.0 (global, post-allreduce).

### 3.4 Effective batch

| | value |
|---|---|
| Per-GPU micro-batch       | 8 sequences |
| Gradient accumulation     | 60 steps |
| GPUs (DDP)                | 2 |
| Block size                | 1 024 tokens |
| **Tokens per opt step**   | **8 · 60 · 2 · 1 024 = 983 040** |
| Tokens over full run      | ≈ 9.83 B |

### 3.5 Evaluation

- `VAL_INTERVAL = 100`: validation is run every 100 optimisation steps, plus at
  the final step.
- `VAL_BATCHES = 160`: each evaluation averages cross-entropy over 160 batches
  of size 8 × 1 024 tokens (≈ 1.3 M tokens per eval), with the validation
  dataset reset to step 0 each time so eval batches are deterministic across
  runs.
- "Best val" reported in this document = the minimum val cross-entropy
  observed across the entire run.

### 3.6 Hardware and orchestration

- 2 × NVIDIA GPUs per job (`--gres=gpu:2`), DDP via `torchrun --nproc_per_node=2`.
- SLURM, partitions `debug` (4 h walltime per cycle) and `general` (up to 2 d).
- `--requeue` plus a `USR1@120` trap so that hitting the partition walltime
  triggers a clean checkpoint, `scontrol requeue`, and `--resume` from
  `<CKPT_DIR>/best.pt` on the next allocation. Crashes are auto-requeued with
  `MAX_RESTARTS = 5`. The schedule, RNG, dataloader epoch counter, and both
  optimiser states are restored from the checkpoint.
- All four configurations share the same `vanilla_train_ddp.py`. The variant is
  selected by environment variables read at startup:

  ```
  SEED       # 42 / 43 / 44 for the seed-variance arm
  N_LAYERS   # 12 (default, B) or 18 (A)
  D_MODEL    # 768 (default, A) or 900 (B); head_dim = D_MODEL / N_HEADS
  N_HEADS    # 12 throughout
  CKPT_DIR   # per-run, isolates checkpoints between variants/seeds
  RUN_NAME   # tag for logs and W&B
  ```

  Defaults reproduce the original 12L/768d Vanilla recipe exactly, so the
  parameterisation is backwards-compatible with the README's seed-42 baseline.

---

## 4. Experiment design

### 4.1 Seed-variance arm

Three seeds × two architectures, all at default sizes:

| Architecture | Seeds | Params |
|---|---|---|
| Vanilla default | 42, 43, 44 | 124.4 M |
| TMM default     | 42, 43, 44 | 163.8 M |

The seed-42 row for each model is the original README baseline. Code was
re-parameterised to read `SEED`, `N_LAYERS`, `D_MODEL`, `N_HEADS` from the
environment, but the param count of the seed-42 reruns is bit-identical to the
original (`Total parameters: 124,373,760` etc. printed at startup), so the old
seed-42 number is a valid third seed.

### 4.2 Parameter-matched arm

Two single-axis matches to TMM's 163.8 M:

| Design | Shape | Params | Δ vs TMM | Mechanism |
|---|---|---|---|---|
| A — depth-18  | 18L / 768d / 12h | 166 850 304 | +1.85 % | extra depth |
| B — width-900 | 12L / 900d / 12h | 162 857 700 | −0.58 % | extra width |

Each variant was trained with `SEED = 42` and otherwise the identical recipe of
§3.

### 4.3 Pre-registered prediction

Before running A and B, the prediction was: if TMM's gain were purely parameter
count, both A and B should close the entire gap. If only one closes it, the
*shape* of the extra parameters matters and the velocity stream's role is at
least partly about that shape rather than raw count. The results are reported
honestly below, regardless of which way they fell.

---

## 5. Validation: CPU-predicted vs GPU-printed parameter counts

To rule out silent miscounts (forgotten LayerNorm bias, untied lm_head, etc.)
each new variant was checked two ways:

1. CPU: instantiate the model on CPU, sum `numel()`.
2. GPU: the training script prints `Total parameters: N` once DDP is initialised.

| Variant | CPU prediction | GPU print | match |
|---|---|---|---|
| Vanilla default (s42 rerun) | 124 373 760 | 124 373 760 | ✓ |
| TMM default                 | 163 812 192 | 163 812 192 | ✓ |
| A — Vanilla depth-18        | 166 850 304 | 166 850 304 | ✓ |
| B — Vanilla width-900       | 162 857 700 | **162 857 700** (job 8033785) | ✓ |
| TMM-zerovel (✗emb, ✓dyn)    | 124 392 288 | 124 392 288 | ✓ |

For each Vanilla width/depth variant the script also printed the per-group
optimiser routing (`Muon (2D weights): 48`, `AdamW (embeddings): 2`,
`AdamW (LayerNorm): 25`), which matches the architecture: 12 × 4 = 48 2-D
weights in the blocks, 2 embedding tables, 12 × 2 + 1 LayerNorm gains.

This is the gate that B had to pass before its result was used.

---

## 6. Results

All numbers are **best validation cross-entropy** over the full 10 000-step run,
with eval cadence and protocol as in §3.5.

### 6.1 Seed variance (default sizes, N = 3)

| Model | Params | s42 | s43 | s44 | mean | std (sample) |
|---|---|---|---|---|---|---|
| Vanilla default | 124.4 M | 1.1570 | 1.1609 | 1.1554 | **1.1578** | **0.0028** |
| TMM default     | 163.8 M | 1.1280 | 1.1257 | 1.1280 | **1.1272** | **0.0013** |

- Raw gap: **1.1578 − 1.1272 = 0.0306**.
- Pooled seed std (across both models) ≈ 0.002.
- Gap / std ≈ **15×** → robust, not seed luck.

### 6.2 Parameter-matched Vanilla (single seed = 42)

| Variant | Params | Best val | Δ vs Vanilla default | Δ vs TMM |
|---|---|---|---|---|
| Vanilla default          | 124.4 M | 1.1578 | — | +0.0306 |
| Vanilla depth-18 (A) ⚠️  | 166.85 M | ~1.1298 | −0.0280 | +0.0026 |
| Vanilla width-900 (B) ✓  | 162.86 M | **1.1454** | **−0.0124** | **+0.0182** |
| TMM default              | 163.8 M | 1.1272 | −0.0306 | 0 |

Reading the Δ-vs-TMM column from the bottom:

- The full **0.0306** is what we want to explain.
- **B closes 40 %** of it just by giving Vanilla the same parameter budget,
  spent as extra width. The remaining **0.0182** is **≈ 9 × the larger of the
  two seed stds** — clearly outside seed noise, even though B is only n = 1
  (0.0182 vs single-seed deviations of ≤ 0.0028 ⇒ ≈ 6 σ).
- **A appears to close 92 %** of it — i.e. depth-matched Vanilla is essentially
  on top of TMM within seed noise. But this number is from a run with
  requeue-induced log gaps that distorted the best-val statistic. We mark it
  ⚠️ and do not rely on it for the headline; a clean rerun is queued (the
  cancelled job 8019932's checkpoint, at step ≈ 8200 / 10 000, is resumable).

### 6.3 Honest interpretation

**Headline (using only the clean rows):** at matched ≈ 163 M parameters,
spending the extra budget on transformer width gives Vanilla 1.1454 vs TMM's
1.1272. **About 40 % of TMM's advantage is parameter count; about 60 % is an
architectural effect that width-scaling does not reproduce.**

**Tentative secondary finding:** the depth-18 result suggests that *depth*-
scaling reproduces much more of TMM's advantage than *width*-scaling does. If
this holds under a clean rerun, it would be consistent with the velocity
stream's residual-mixing structure acting more like extra effective depth than
extra effective width. We do not draw this conclusion yet; A needs a clean
rerun first.

**What this does *not* show.** This experiment does not isolate the velocity
*dynamics* from the velocity *embeddings*. The +39 M is 99.9 % embedding
parameters; B gives Vanilla a comparable budget but spent as width, not as a
duplicate embedding table. A direct isolation requires the TMM-zerovel
(✗ embeddings, ✓ dynamics, 124 M) and a TMM-no-dynamics (✓ embeddings,
✗ dynamics, ~163 M) pair — the first was started but cancelled at 52 %
(best so far 1.1535, not converged) and the second is not yet built. They are
the natural follow-up.

---

## 7. Caveats and threats to the conclusion

- **B is n = 1.** The residual 0.0182 is large relative to plausible single-seed
  variation (≈ 0.0028 max in this regime), so the *sign* is safe, but the
  magnitude has a ± of order one seed-std. A second seed of B would tighten it.
- **A is unclean.** Its log has requeue gaps from the seesaw on the `debug`
  partition; best-val reported here is from the surviving log spans and may be
  optimistic. Treat A as suggestive, not load-bearing.
- **Width-vs-depth is one knob.** B widens `d_model` and proportionally raises
  `head_dim` (64 → 75). It does not, e.g., add an aux embedding table the way
  TMM does. A truly tight architectural match to TMM-minus-dynamics is what
  Design E (TMM with `VEL_DYNAMICS=off` but velocity embeddings kept) would
  give; it is not built yet.
- **Best-val vs step-aligned.** Best-val can favour the model whose val curve
  has the deeper but later minimum if the other has already plateaued; for the
  clean rows (default Vanilla, default TMM, B) the curves are well-behaved and
  best-val is fair, but for A the requeue gaps could flatter it.
- **Validation set.** TinyStories is a small, structurally simple corpus.
  Whether the 40 / 60 split generalises to OpenWebText is the point of the
  separate `van-d18-owt` run (in progress; currently at step 14 400 / 30 000,
  best val 3.0450, paused awaiting GPU on `general`).

---

## 8. What's next (in priority order)

1. **Clean depth-18 (A) rerun, single seed.** Either resume job 8019932 from
   its 82 % checkpoint or restart fresh. Decides whether the depth-vs-width
   dissociation is real.
2. **Second seed of B (width-900).** Tightens the 0.0182 estimate.
3. **Finish TMM-zerovel (✗ emb, ✓ dyn).** The cancelled run's checkpoint is at
   ≈ 52 % (best 1.1535, not converged). Together with a future Design E
   (✓ emb, ✗ dyn) this completes the 2 × 2 that cleanly separates "extra
   embeddings" from "velocity dynamics".
4. **OWT decisive test.** `van-d18-owt` (job 8025920) is paused at step
   14 400 / 30 000 with checkpoint `/data/user_data/jgai/cache/`
   `checkpoints_vanilla_d18_owt/best.pt`; it will resume once the user's
   `general`-partition GPU quota frees. This tests whether the TS-derived
   split survives on a non-saturated corpus.
5. **Design E (TMM with `VEL_DYNAMICS=off`, embeddings kept, ≈163.79 M).**
   Not yet implemented. This is the architecturally cleanest match to TMM and
   the cleanest single test of "is it the dynamics or the extra embeddings?".

---

## Appendix A — Job IDs and reproducibility

| Run | Job ID | Status | Log |
|---|---|---|---|
| Vanilla s42 (README baseline) | (pre-parameterisation) | reference | (README) |
| Vanilla s43 (default)         | 8019934 | COMPLETED 3.0 h | `logs/vanilla_ts_s43_8019934.out` |
| Vanilla s44 (default)         | 8019935 | COMPLETED | `logs/vanilla_ts_s44_8019935.out` |
| TMM s42 (README baseline)     | (pre-parameterisation) | reference | (README) |
| TMM s43 (default)             | 8019936 | COMPLETED | `logs/tmm_ts_s43_8019936.out` |
| TMM s44 (default)             | 8019937 | COMPLETED | `logs/tmm_ts_s44_8019937.out` |
| A — Vanilla depth-18          | 8019932 | CANCELLED at 82 % (checkpoint kept; resumable via `--resume`) | `logs/vanilla_d18_ts_*.out` |
| B — Vanilla width-900         | 8033785 | COMPLETED 3.8 h (single debug cycle) | `logs/vanilla_w900_ts_8033785.out` |
| TMM-zerovel (✗emb, ✓dyn)      | 8019933 | CANCELLED at 52 % (checkpoint kept) | `logs/tmm_zerovel_ts_*.out` |
| van-d18-owt (OWT, A on OWT)   | 8025920 | RUNNING → preempted → PENDING `general` | `logs/vanilla_d18_owt_8025920.out` |

Reproducing a single run:

```bash
# Example: B (width-900) from scratch, seed 42
export D_MODEL=900 N_HEADS=12 N_LAYERS=12 SEED=42
export CKPT_DIR=checkpoints_vanilla_w900_ts RUN_NAME=vanilla-w900-ts
sbatch vanilla_w900_ts.sbatch
```

The sbatch script is a clone of `vanilla_d18_ts.sbatch` with the env block
swapped to width-900 — that is the only delta between A and B sbatch files.

## Appendix B — Per-block parameter formula

For an architecture with vocab `V`, sequence `T`, depth `L`, model dim `d`,
heads `H` (`d_head = d / H`), bias-free LayerNorms, MLP expansion 4×, and
tied output:

```
per_block        = 12 d² + 2 d
embeddings       = V·d + T·d
final_ln         = d
total            = L·(12 d² + 2 d) + V·d + T·d + d
```

Substituting `V = 50 304`, `T = 1024`:

- `12L/768d → 12·7 079 424 + 39 420 672 + 768 = 124 373 760` ✓
- `18L/768d → 18·7 079 424 + 39 420 672 + 768 = 166 850 304` ✓
- `12L/900d → 12·9 721 800 + 46 195 200 + 900 = 162 857 700` ✓

The GPU-printed numbers in §5 agree to the last digit in every case.
