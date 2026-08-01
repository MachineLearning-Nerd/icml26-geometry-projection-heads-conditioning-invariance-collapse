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

`cpu-upgrade` bills at roughly $0.03 per hour, so even if every job above ran to its
timeout the total is on the order of **$3**. The binding constraint is wall-clock, not
money, so the long runs are **harvested early rather than run to timeout**:

| Node | Timeout | Harvest point | Why that is enough |
| --- | --- | --- | --- |
| collapse ×5 | 14h | ~10 epochs (~6h) | the sign structure of lambda_min and the size of the interaction term M are per-state quantities, reported every epoch; more epochs add trajectory length, not a different answer |
| orbits | 26h | whatever epoch is reached at the ~10-12h mark | orbit geometry is evaluated every 5 epochs, and the paper's exact 21.85x is already verified against the released arrays — this run is independent corroboration |
| pretrained | 3h | run to completion | it is short |

Every harvested run states, on its claim page, exactly how many of the paper's epochs
were completed and why the run was stopped. A shortened run is reported as a shortened
run; it is never described as the paper's full budget.
