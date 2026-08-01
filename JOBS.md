# Hugging Face `cpu-upgrade` job registry

All research compute for this reproduction runs on Hugging Face `cpu-upgrade`. No GPU
is used anywhere; the local machine only inspects, edits and orchestrates.

Measured allocation on every job below: cgroup quota **8 vCPU**, AMD EPYC 7R13,
`os.cpu_count()` reports 64 (hence the thread pinning in `src/threads.py`).
Estimated core requirement before each run: 8 (torch CPU convolutions saturate the
quota; larger flavors are not offered for this tier).

Each job runs the fixed command `uv run --frozen repro/run_all.py` against a pinned
git ref; the ref alone determines the experiment, via `repro/config.py`.

| Job id | Node / ref | Stage | Timeout | Purpose |
| --- | --- | --- | --- | --- |
| `6a6d74eea00abefd4b28abbf` | (standalone) | calibration | 45m | CPU quota, throughput, released-array inventory |
| `6a6d76d1a00abefd4b28abde` | `main@78fc470` | smoke | 40m | Hessian-machinery validation, 4 activations |
| `6a6d7b1ba00abefd4b28ac69` | `exp/released@8376189` | released | 60m | first re-analysis of the authors' released arrays |
| `6a6d7bf06b79c09949c1e004` | `exp/released-v2@af44aec` | released | 60m | + ablation key handling, Assumption 6/7 audit |
| `6a6d7b1e6b79c09949c1dfe5` | `exp/pretrained@3e779d9` | pretrained | 6h | official SSL checkpoints, Theorem 3.1 / Prop 3.3 |
| `6a6d7cbba00abefd4b28acb3` | `exp/collapse-swish-collapsed@7ae0a85` | collapse | 14h | Swish head, pseudo-collapsed init |
| `6a6d7cbe6b79c09949c1e02c` | `exp/collapse-relu-collapsed@c0e6048` | collapse | 14h | ReLU head, pseudo-collapsed init |
| `6a6d7cc16b79c09949c1e02f` | `exp/collapse-gelu-collapsed@e4a770e` | collapse | 14h | GELU head, pseudo-collapsed init |
| `6a6d7cc46b79c09949c1e031` | `exp/collapse-linear-collapsed@baeb36d` | collapse | 14h | linear head, pseudo-collapsed init (negative control) |
| `6a6d7cc7a00abefd4b28acb7` | `exp/collapse-swish-normal@d9681bc` | collapse | 14h | Swish head, standard init |
| `6a6d7cca6b79c09949c1e034` | `exp/orbits@b2e5cee` | orbits | 26h | SimCLR pretraining + orbit geometry |

Runtimes and outcomes are recorded in `.openresearch/artifacts/` as each job lands.

## Budget and harvest policy

**Hard cap: no job may be launched with an estimated runtime over 1 hour, and any of
this campaign's jobs that overruns an hour is cancelled.** Set 2026-08-01, after the
first round of long runs was terminated mid-flight by a platform credit cutoff having
banked 0-1 epochs each. Work that does not fit an hour is split until it does.

Two consequences that are easy to get wrong:

- **The data fetch counts against the hour.** Staging CIFAR-10 from `cs.toronto.edu`
  inside a job measured **2106 s (35 min)** at ~80-110 kB/s — more than half the budget
  before a single gradient step. Any job needing CIFAR-10 must fetch from a fast mirror
  under the same MD5 gate (`c58f30108f718f92721af3b95e74349a`) or it cannot fit.
- **Only this campaign's jobs are ever cancelled.** The `DineshAI` account is shared:
  on 2026-08-01, 28 unrelated jobs from another agent's mixture-of-experts sweep were
  running under it. Confirm ownership from a job's own log before cancelling it.

Re-scoped plan under the cap (each row is one job, all `cpu-upgrade`):

| Node | Est. runtime | How it was made to fit |
| --- | --- | --- |
| pretrained ×4 (one per checkpoint) | ~25-40 min | was one ~3h job over 4 checkpoints; splitting per checkpoint also removes the risk that one slow model starves the rest |
| geometry certificate (Claims 2, 3) | ~10 min | symbolic; no checkpoint download, no training. Was buried at the end of the pretrained job, which is why it never ran — it is now its own node and cannot be starved |
| losses ×1 (Claim 5 part A) | ~15 min | eight objectives, analytic Hessians only |
| collapse ×5, epoch-chunked | ~45 min each | `max_steps_per_epoch` bounds the tracked steps, and a chunk resumes nothing — each job reports its own states, which is sound because lambda_min and M are **per-state** quantities, not trajectory-dependent |
| orbits | ~50 min | fewer epochs, evaluated at the epochs reached; the exact 21.85x is already banked from the released arrays, so this stays corroboration |

Every harvested run states, on its claim page, exactly how many of the paper's epochs
were completed and why the run was stopped. A shortened run is reported as a shortened
run; it is never described as the paper's full budget.
