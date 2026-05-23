# Optimizer and Learning-Rate Ablations on OWT (Vanilla vs TMM)

This document covers two ablations on OpenWebText pretraining that test
whether the "TMM > Vanilla" architectural ordering survives changes to the
optimizer recipe:

1. **Optimizer-on-2D-weights ablation.** Hold architecture, schedule, batch, and
   the embedding / LayerNorm optimizer fixed; replace the Muon optimizer that
   acts on the 2D matrix weights (qkv, out, w1, w2) with AdamW. Run for both
   architectures → a full 2 × 2 of architecture × optimizer.
2. **Learning-rate sweep (partial).** Hold optimizer recipe fixed at the
   Muon-hybrid setup; perturb the Muon LR away from its default and check that
   the architectural ordering is not a knife-edge of the chosen LR.

---

## TL;DR

The optimizer-on-2D-weights ablation is **complete (2 × 2)**:

| | Muon hybrid (2D = Muon @ 4e-3, rest = AdamW @ 6e-4) | pure AdamW (everything @ 6e-4) |
|---|---|---|
| **Vanilla 124 M** | **3.0078** | 3.0103 |
| **TMM 163.8 M**   | **2.9342** | **2.9696** |

- **TMM beats Vanilla under both optimizers** (gap 0.0736 under Muon hybrid,
  0.0407 under pure AdamW). The architectural ordering is robust to the choice
  of 2D-weight optimizer.
- The two optimizers are **essentially indistinguishable on Vanilla** at
  30 k steps (Δ = 0.0025, ≈ one val-eval apart) but **distinguishable on TMM**
  (Δ = 0.0354, ~ 14× larger). Equivalently, **TMM benefits from the Muon
  hybrid much more than Vanilla does at the end of training.**

The LR sweep is **partial (2 of 8 designed runs)** — only the half-LR arm on
the Muon side ran:

| Arch | Muon LR | best val | Δ vs baseline |
|---|---|---|---|
| Vanilla | 4e-3 (baseline) | 3.0078 | — |
| Vanilla | **2e-3** (half) | **3.0288** | +0.0210 |
| TMM     | 4e-3 (baseline) | 2.9342 | — |
| TMM     | **2e-3** (half) | **2.9634** | +0.0292 |

The TMM > Vanilla ordering is preserved at half LR (gap 0.0654 vs baseline
0.0736; ≈ 12 % compression). The other 6 designed sweep cells (×2 LR up, pure-
AdamW LR ± 1.7×) were submitted then cancelled, so the LR robustness statement
is *only along the "Muon-LR halved" direction*.

---

## 1. Why these ablations exist

The headline architectural finding — at default OWT recipe TMM (163.8 M) beats
Vanilla (124.4 M) on val loss at 30 k steps by ~ 0.074 nats — depends on a
specific optimizer and LR recipe. Two adversarial readings need to be ruled
out:

- **"TMM only wins because Muon happens to suit TMM's architecture."** This is
  the optimizer-on-2D-weights ablation. If the gap survives switching from
  Muon-hybrid to a single pure-AdamW optimizer, the architectural ordering is
  not Muon-specific.
- **"TMM only wins at one knife-edge LR."** This is the LR sweep. If the gap
  survives perturbing the Muon LR, the architectural ordering is not LR-
  specific. (Only the down-perturbation is in hand — see §4.)

Neither ablation re-tunes hyperparameters per architecture; both test
**robustness of the qualitative ordering**, not "best achievable" on each side.

---

## 2. Shared training protocol

All runs in this document share the recipe below. Variant-specific deltas are
called out in §3 and §4.

### 2.1 Data and tokenisation

- **Dataset:** OpenWebText. Pre-tokenised arrays cached under
  `$CACHE/owt_data/{train,val}_tokens.npy`. Validation set is the same fixed
  shard for every run, so val numbers are directly comparable across runs.
- **Tokenizer:** `tiktoken.get_encoding("gpt2")`. Effective model vocab 50 304
  (GPT-2's 50 257 padded up to be divisible by 128).
- **Block size:** 1 024 tokens.
- **Sampler:** deterministic epoch-reordered, seeded `SEED + rank` per rank.

### 2.2 Schedule

- **Total steps:** 30 000.
- **Warmup:** 3 000 steps linear from 0 to peak LR.
- **Decay:** cosine to `MIN_LR_RATIO · peak = 0.1 · peak`.
- **Grad clip:** 1.0 (global, post-allreduce).
- **Seed:** 42 throughout (N = 1).

### 2.3 Effective batch

| | Value |
|---|---|
| Per-GPU micro-batch        | 8 sequences |
| Gradient accumulation     | 30 per GPU |
| GPUs (DDP)                | 2 |
| Block size                | 1 024 |
| **Sequences / step**      | 8 · 30 · 2 = **480** |
| **Tokens / step**         | 480 · 1 024 = **491 520** |
| Tokens over full run      | ≈ 14.7 B |

### 2.4 Evaluation

- `VAL_INTERVAL = 100`: validation runs every 100 optimisation steps and at
  the final step.
- `VAL_BATCHES = 160`: each eval averages cross-entropy over
  160 × 8 × 1 024 ≈ 1.3 M held-out OWT tokens, with the validation dataset
  reset before each pass so the eval batches are deterministic across runs.
- **Best val** = the minimum val cross-entropy observed across the run.

### 2.5 Hardware and orchestration

- 2 × NVIDIA L40S per job (`--gres=gpu:2`), DDP via
  `torchrun --nproc_per_node=2`.
- SLURM with `--requeue` and a `USR1@120` trap so a partition-walltime hit
  triggers a clean checkpoint, requeue, and `--resume` from
  `<CKPT_DIR>/best.pt` on the next allocation.
- One per-step wall: ~ 2.7–3.3 s/step on L40S × 2 (variant-dependent).
- Total wall time per 30 k run: 22–28 h.

### 2.6 Optimiser routing (default Muon-hybrid)

All four params groups are routed by parameter role at startup
(`configure_optimizers` in `*_train_owt.py`):

| Group | Optimiser | LR | wd | Notes |
|---|---|---|---|---|
| 2-D linear weights (`qkv_proj`, `out_proj`, `w1`, `w2`)              | **Muon** | 4 × 10⁻³ | 0   | momentum 0.95, Nesterov |
| Embeddings (`tok_emb`, `pos_emb`, plus `vel_*` for TMM)              | AdamW    | 6 × 10⁻⁴ | 0.1 | β = (0.9, 0.95) |
| LayerNorm gains, 1-D weights                                          | AdamW    | 6 × 10⁻⁴ | 0   | β = (0.9, 0.95) |
| TMM only: learned per-layer scalars `_raw`                            | AdamW    | 3 × 10⁻³ | 0   | β = (0.9, 0.95) |

Group sizes printed at startup for sanity check:
`Muon (2D weights): 48 params  | AdamW (embeddings): 2 (Vanilla) / 4 (TMM)
| AdamW (LayerNorm): 25 | (TMM) scalars: variable`.

### 2.7 Why these LRs (honest)

The LRs above are **inherited defaults from the prior YuriiFormer/TMM
codebase**; they were not chosen by sweeping in this experiment. The
rationale, as I understand it:

- **`ADAMW_LR = 6e-4`** is the canonical nanoGPT / GPT-2-small AdamW peak LR
  (e.g. Karpathy's nanoGPT). Used throughout the codebase at the same value
  in every config; not retuned.
- **`MUON_LR = 4e-3`** on OWT is ≈ 7× `ADAMW_LR`. Muon's Newton-Schulz step
  produces an update whose spectral norm — not its `‖g‖₂` — is the natural
  scale, so its working LR range is typically 5–30× AdamW's at small-LM scale
  (per the Muon paper's recommendations and the public Muon repos).
- **`MUON_LR = 0.02` on TinyStories** (5× larger) reflects the smaller dataset,
  shorter total schedule, and somewhat smaller effective-batch token count;
  again, this was the existing default, not retuned here.

The LR sweep in §4 is a *post-hoc robustness check* on the OWT Muon LR, not
the procedure that selected it.

---

## 3. Ablation 1: optimizer on 2D weights

### 3.1 Design

Lock everything in §2; vary one thing — what optimiser updates the 48 2-D
matrix weights (12 layers × 4 weights/layer). Two cells × two architectures
= four runs.

| Cell | 2-D weights | rest (emb, LN, scalars) | files |
|---|---|---|---|
| Muon hybrid | **Muon** @ 4e-3 | AdamW @ 6e-4 (wd 0.1 on embeddings, 0 on LN) | `vanilla_train_owt.py`, `tmm_train_owt.py` |
| pure AdamW  | **AdamW** @ 6e-4 (wd 0.1) | same AdamW @ 6e-4 (one optimiser for everything) | `vanilla_adamw_train_owt.py`, `tmm_adamw_train_owt.py` |

Schedule, batch, data, seed, grad clip, eval cadence — identical across cells.

### 3.2 Final numbers (best val loss at 30 k steps)

| Architecture | Muon hybrid | pure AdamW | Δ (AdamW − Muon-hybrid) |
|---|---|---|---|
| Vanilla 124.4 M | **3.0078** | 3.0103 | +0.0025 |
| TMM 163.8 M     | **2.9342** | 2.9696 | +0.0354 |
| Δ across arch (Vanilla − TMM) | +0.0736 | +0.0407 | — |

### 3.3 Step-aligned val loss (so you can see where the gaps appear)

Vanilla cells (full numbers from README):

| step | Vanilla + pure AdamW | Vanilla + Muon hybrid | Δ |
|---:|---:|---:|---:|
| 1 000  | 4.844 | **4.638** | +0.206 |
| 2 000  | 3.923 | **3.734** | +0.189 |
| 3 000  | 3.651 | **3.528** | +0.122 |
| 5 000  | 3.397 | **3.334** | +0.063 |
| 7 000  | 3.294 | **3.251** | +0.043 |
| 10 000 | 3.213 | **3.178** | +0.035 |
| 15 000 | 3.132 | **3.107** | +0.025 |
| 20 000 | 3.074 | **3.055** | +0.018 |
| 25 000 | 3.030 | **3.022** | +0.009 |
| 29 900 | 3.011 | **3.008** | +0.002 |

TMM cells (Muon-hybrid baseline from `tmm_owt_{6989891,7007288}.out`; the
pure-AdamW run's local `.out` was truncated past step 7 000 — see §6 —
so 10 k–25 k for TMM + pure AdamW are not extracted here; final 30 k value
is from wandb summary):

| step | TMM + pure AdamW | TMM + Muon hybrid | Δ |
|---:|---:|---:|---:|
| 1 000  | 5.048 | **4.921** | +0.127 |
| 2 000  | 4.130 | **3.747** | +0.383 |
| 3 000  | 3.717 | **3.479** | +0.238 |
| 5 000  | 3.401 | **3.263** | +0.138 |
| 7 000  | 3.280 | **3.175** | +0.105 |
| 10 000 | — (truncated) | **3.103** | — |
| 15 000 | — (truncated) | **3.031** | — |
| 20 000 | — (truncated) | **2.982** | — |
| 25 000 | — (truncated) | **2.948** | — |
| 29 900 | **2.970** (summary) | **2.934** | +0.036 |

### 3.4 Interpretation

- **The Vanilla curves close almost completely by 30 k.** Δ shrinks
  monotonically from 0.206 nats at step 1 000 to 0.002 nats at step 29 900.
  On Vanilla, Muon-hybrid is essentially a **sample-efficiency win in early
  training, not a converged-loss win**. If your budget is ≤ 5 k steps, you
  save ~ 0.06 nats; at 30 k it is one val-eval of noise.
- **The TMM curves stay apart.** Δ falls from 0.383 at step 2 000 to 0.036
  at step 30 000 — an order-of-magnitude wider gap than Vanilla's at the
  same point. **TMM extracts more out of the Muon hybrid than Vanilla does**,
  and the gain persists past convergence.
- **TMM beats Vanilla under either optimiser.** Reading the bottom row of
  §3.2: under Muon hybrid TMM is ahead by 0.074 nats; under pure AdamW TMM
  is ahead by 0.041. Switching both architectures to pure AdamW roughly
  *halves* the architectural advantage but does not erase it. So part of
  TMM's measured advantage at default recipe is real architecture, and part
  is **a Muon × TMM interaction** that the Muon hybrid amplifies.

A speculative read of why TMM × Muon interact: TMM's velocity stream adds
a duplicate embedding table and additional residual paths through extra
LayerNorms. Muon's Newton-Schulz orthogonalisation acts on the 2-D weight
matrices only; if the velocity stream changes the conditioning of those
matrices in a way that AdamW's per-coord adaptive scale handles less well
than Muon's spectral scale, that would produce exactly the pattern observed
(bigger gap on TMM than on Vanilla when you switch the 2-D optimiser).
This is *not* established here — it is a hypothesis that would need a
direct measurement of 2-D-weight spectra under each pretrain.

---

## 4. Ablation 2: learning-rate sweep (partial)

### 4.1 What was designed

A 2 × 2 × 2 = 8-cell grid around the baseline, all at default 30 k OWT
recipe:

| arch | optimiser cell | LR perturbation |
|---|---|---|
| Vanilla | Muon hybrid | `MUON_LR ∈ {2e-3, 8e-3}` (baseline 4e-3) |
| Vanilla | pure AdamW  | `LR ∈ {3e-4, 1e-3}` (baseline 6e-4) |
| TMM     | Muon hybrid | `MUON_LR ∈ {2e-3, 8e-3}` |
| TMM     | pure AdamW  | `LR ∈ {3e-4, 1e-3}` |

Together with the baselines this would have been a 3-point LR grid per
arch × optimiser, so the sweep would have tested both LR direction and
both optimiser choice.

The submit commands (one per cell):

```bash
sbatch -J vanilla-muon-lr2e-3   MUON_LR=2e-3 vanilla_train_owt.sbatch
sbatch -J vanilla-muon-lr8e-3   MUON_LR=8e-3 vanilla_train_owt.sbatch
sbatch -J vanilla-adamw-lr3e-4  LR=3e-4      vanilla_adamw_train_owt_general.sbatch
sbatch -J vanilla-adamw-lr1e-3  LR=1e-3      vanilla_adamw_train_owt_general.sbatch
sbatch -J tmm-muon-lr2e-3       MUON_LR=2e-3 tmm_train_owt.sbatch
sbatch -J tmm-muon-lr8e-3       MUON_LR=8e-3 tmm_train_owt.sbatch
sbatch -J tmm-adamw-lr3e-4      LR=3e-4      tmm_adamw_train_owt_general.sbatch
sbatch -J tmm-adamw-lr1e-3      LR=1e-3      tmm_adamw_train_owt_general.sbatch
```

### 4.2 What actually ran

Only **2 of the 8** cells, both on the Muon-hybrid leg and both at the
**half-LR** (down) perturbation:

| Job | Cell | Result | Wall time |
|---|---|---|---|
| 7933587 | Vanilla + Muon hybrid + `MUON_LR=2e-3` | best val **3.0288** | 27.7 h |
| 7933591 | TMM     + Muon hybrid + `MUON_LR=2e-3` | best val **2.9634** | 22.9 h |

The other 6 cells (×2 LR up to 8e-3, plus all 4 pure-AdamW LR cells) were
submitted then cancelled before they consumed material compute. The LR
robustness statement below is therefore *only along the "Muon LR halved"
direction*.

### 4.3 Step-aligned val loss

Re-using the baseline numbers from §3.3:

| step | Vanilla baseline (4e-3) | Vanilla half-LR (2e-3) | TMM baseline (4e-3) | TMM half-LR (2e-3) |
|---:|---:|---:|---:|---:|
| 1 000  | 4.638 | 4.978 | 4.921 | 5.169 |
| 2 000  | 3.734 | 3.876 | 3.747 | 3.937 |
| 3 000  | 3.528 | 3.605 | 3.479 | 3.578 |
| 5 000  | 3.334 | 3.376 | 3.263 | 3.320 |
| 7 000  | 3.251 | 3.283 | 3.175 | 3.221 |
| 10 000 | 3.178 | 3.206 | 3.103 | 3.139 |
| 15 000 | 3.107 | 3.130 | 3.031 | 3.062 |
| 20 000 | 3.055 | 3.078 | 2.982 | 3.011 |
| 25 000 | 3.022 | 3.044 | 2.948 | 2.978 |
| 29 900 | 3.008 | 3.029 | 2.934 | 2.964 |

### 4.4 Interpretation

- **Halving the Muon LR uniformly worsens both architectures.** At step
  29 900, Vanilla degrades by +0.0210 and TMM degrades by +0.0292. Neither
  arch was running too-hot at the baseline (no sign of instability at
  4e-3 that 2e-3 would fix), so halving is unambiguously a worse recipe for
  both — consistent with the baseline LR being on the high-LR side of, or
  near, the optimum.
- **The TMM > Vanilla ordering is preserved at half LR.** Gap goes from
  0.0736 (baseline) to 0.0654 (half LR) — a ≈ 12 % compression but the
  sign and order of magnitude survive.
- **The two arches degrade at slightly different rates.** TMM loses 0.029
  nats per halving, Vanilla loses 0.021. So if anything the baseline LR
  is more important for TMM than for Vanilla — consistent with §3's pattern
  that TMM is the architecture that is *more* sensitive to the optimiser
  side of the recipe. This is suggestive only; a single perturbation
  point (and only in one direction) cannot establish a curvature claim.

### 4.5 What the partial sweep cannot rule out

- **LR-up.** With `MUON_LR = 8e-3` not run, we cannot exclude the possibility
  that Vanilla benefits from a higher LR more than TMM does and could close
  the gap. This is the highest-priority missing cell.
- **Pure-AdamW LR.** With the 4 pure-AdamW LR cells not run, the §3 pure-
  AdamW comparison sits at a single LR (6e-4) for each architecture, and the
  observation that TMM benefits more from Muon than Vanilla does could in
  principle be partly an artefact of pure-AdamW being LR-undertuned for one
  architecture.
- **Single seed.** N = 1 throughout; the gap-compression number (0.074 → 0.065)
  has an unmeasured per-seed uncertainty.

---

## 5. Combined reading of the two ablations

Taking §3 and §4 together, three statements are well-supported:

1. **TMM > Vanilla survives both perturbations tested.** It survives swapping
   the 2-D optimiser from Muon hybrid to pure AdamW (gap shrinks from
   0.074 to 0.041 but stays in TMM's favour), and it survives halving the
   Muon LR (gap shrinks from 0.074 to 0.065 but stays in TMM's favour).
2. **TMM is the more optimiser-sensitive architecture.** The same Muon →
   AdamW swap costs Vanilla 0.003 nats but costs TMM 0.035 nats. The same
   LR halving costs Vanilla 0.021 nats but costs TMM 0.029 nats. Wherever
   the optimiser is more aggressive (Muon vs AdamW, higher LR vs lower),
   TMM gains relative to Vanilla.
3. **The 30 k-step Vanilla-side optimizer comparison is essentially a tie.**
   Vanilla + Muon hybrid 3.0078 vs Vanilla + pure AdamW 3.0103 is within one
   val-eval of noise. The README's statement that "the Muon+AdamW hybrid is
   consistently ahead throughout training" is correct in early training but
   the converged gap is ≈ 0 for Vanilla.

Statements that are **not** yet supported:

- That TMM > Vanilla survives at **higher** Muon LR (8e-3 cell missing).
- That the §3 Vanilla-side AdamW gap stays at 0.0025 across LR (no pure-
  AdamW LR sweep).
- A mechanism for the TMM × Muon interaction. The hypothesis in §3.4 is
  consistent with the data but not tested.

---

## 6. The TMM + pure-AdamW logging incident

For posterity, since this caused a misread of the data:

- `tmm_adamw_owt_7912083.out` is **978 lines and stops mid-training at
  step 7 880** (≈ 6 h elapsed, val 3.247). The file's last line is a single
  `=== Done: Thu May 14 11:12:06 EDT 2026 ===` that follows the step-7 880
  line with no intervening output for the remaining ~ 17 h.
- `sacct -j 7912083` returns `State=COMPLETED, ExitCode=0:0,
  Elapsed=22:58:00`. The SLURM wrapper exited cleanly.
- The local wandb mirror at `wandb/run-20260513_121437-mcf3amf0/` is the
  authoritative record. `files/wandb-summary.json` shows
  `_step=29999, final/best_val_loss=2.96957, final/total_time_hours=22.96`.
  Metadata: `job_id=7912083, job_name=tmm-adamw-owt,
  model="tmm-adamw (pure AdamW, no Muon)", host=babel-n5-24`.
- The local wandb `files/output.log` is also truncated past step ~ 7 000.
  The wandb binary `run-*.wandb` and `logs/debug-internal.log` carry the
  full history but are not text-greppable in standard tools.

Read: the training process completed all 30 000 steps and wrote the final
checkpoint; the SLURM stdout pipeline (and the wandb file-mirror, which
follows the same pipe on this codepath) silently stopped flushing the
per-step prints partway through. The numbers from wandb summary are the
ones to trust.

**Lesson for future ablations on this cluster:** for any > ~ 10 h run,
treat the SLURM `.out` file as suggestive only and verify completion against
`wandb-summary.json` or the saved checkpoint state-dict (which has the
final `step` field).

---

## Appendix A — Job IDs and log paths

| Run | Job ID | sacct State | log | best val | wall time |
|---|---|---|---|---|---|
| Vanilla + Muon hybrid (baseline)             | 6989892 | COMPLETED | `logs/vanilla_owt_6989892.out`         | 3.0078 | (see README) |
| Vanilla + pure AdamW                          | 7890344 | COMPLETED | `logs/vanilla_adamw_owt_7890344.out`   | 3.0103 | (see README) |
| TMM + Muon hybrid (baseline, 2 segments)      | 6989891 → 7007288 | COMPLETED (resume) | `logs/tmm_owt_6989891.out` + `logs/tmm_owt_7007288.out` | 2.9342 | total ~ 30 h |
| TMM + pure AdamW                              | 7912083 | COMPLETED | `logs/tmm_adamw_owt_7912083.out` ⚠ truncated; **use** `wandb/run-20260513_121437-mcf3amf0/files/wandb-summary.json` | **2.9696** | 22.96 h |
| Vanilla + Muon hybrid + `MUON_LR=2e-3` (LR sweep) | 7933587 | COMPLETED | `logs/vanilla_owt_7933587.out` | 3.0288 | 27.7 h |
| TMM + Muon hybrid + `MUON_LR=2e-3` (LR sweep)     | 7933591 | COMPLETED | `logs/tmm_owt_7933591.out`     | 2.9634 | 22.9 h |

## Appendix B — Reproducing a single cell

```bash
# Baseline TMM, default LRs
sbatch tmm_train_owt.sbatch                    # Muon hybrid, MUON_LR=4e-3
sbatch tmm_adamw_train_owt_general.sbatch      # pure AdamW, LR=6e-4

# LR-perturbed (e.g. half-LR TMM):
sbatch -J tmm-muon-lr2e-3 \
       --export=ALL,MUON_LR=2e-3,LR_SUFFIX=_lr2e-3 \
       tmm_train_owt.sbatch
```

`LR_SUFFIX` propagates into `CKPT_DIR` and `RUN_NAME` so that LR-perturbed
runs do not overwrite the baseline checkpoint.

## Appendix C — Optimiser configuration source-of-truth

The four-group routing and per-group hyperparameters are defined in
`configure_optimizers()` of each `*_train_owt.py` file. The Muon-hybrid
files (`vanilla_train_owt.py`, `tmm_train_owt.py`) build a `Muon` optimiser
for the 2-D weights and a separate `AdamW` for embeddings + LN + scalars.
The pure-AdamW files (`vanilla_adamw_train_owt.py`,
`tmm_adamw_train_owt.py`) build one `AdamW` covering all groups, with
group-specific `weight_decay` (0.1 on embeddings and 2-D weights, 0 on LN).
`MUON_LR` and `ADAMW_LR` are read from the environment at startup, so
LR-sweep cells reuse the same script with only `--export=ALL,MUON_LR=...`
or `LR=...` overrides.
