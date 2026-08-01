# icml26-repro-y4uR1LFClc-the-geometry-of-projection-heads-conditioning-invariance-and-collapse

ICML 2026 agent reproduction workspace for `y4uR1LFClc`.

Paper: **The Geometry of Projection Heads: Conditioning, Invariance, and Collapse** ·
arXiv [2605.17180](https://arxiv.org/abs/2605.17180) ·
logbook Space [`DineshAI/y4uR1LFClc`](https://huggingface.co/spaces/DineshAI/y4uR1LFClc)

## Fixed reproduction command

Every experiment-tree node runs exactly this command; variants live in committed
code under `repro/`, never in the command and never in environment variables.

```bash
uv run repro/run_all.py
```

## Compute policy

All research compute — every experiment, verifier, benchmark, and data-generation
step — runs on Hugging Face **`cpu-upgrade`** jobs. No GPU is used anywhere. The
local machine is used only for repository inspection, editing, and orchestration.

Measured `cpu-upgrade` allocation (job `6a6d74eea00abefd4b28abbf`, 2026-08-01):
cgroup quota **8 vCPU** on an AMD EPYC 7R13 while `os.cpu_count()` reports 64 — so
`src/threads.py` pins every BLAS/OpenMP pool to the cgroup quota before numpy or
torch is imported. Without that pinning these jobs run an order of magnitude slower.

## Layout

```
src/threads.py             cgroup-aware thread pinning (import before numpy/torch)
repro/                     claim verifiers and experiment drivers
jobs/                      cpu-upgrade job entrypoints (PEP-723 uv scripts)
joblog.py                  orchestration helper: read a job's status and logs
.openresearch/artifacts/   durable per-claim evidence
```

## Paper source of record

`https://ar5iv.labs.arxiv.org/html/2605.17180`, retrieved 2026-08-01,
SHA-256 `c344481c6fa2c59b6439f41d2053c737d92e11da1e4a7890941c776188ade7a4`.

The authors' public code and released raw arrays are at
`https://github.com/farischaudhry/projection-head-geometry`.
