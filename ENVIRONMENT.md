# Environment and reproduction boundary

## Fixed entrypoint

Every experiment branch is identified by its committed `repro/config.py` and runs:

```sh
uv run --frozen repro/run_all.py
```

The main branch is a documentation and verifier entrypoint; the stage-specific
contracts live on the descriptive experiment/release branches listed in
`branch-audit.md`.

## Runtime

- CPU-only campaign; every runner asserts that CUDA is unavailable.
- Hugging Face `cpu-upgrade`: measured 8-vCPU cgroup quota on AMD EPYC 7R13.
- Local geometry checks: 8-core Apple-silicon CPU.
- Pinned environment is preserved in `repro/pyproject.toml.txt` and `repro/uv.lock.txt`.
- `src/threads.py` pins BLAS/OpenMP pools before NumPy or Torch import.

## Reproduction boundaries

- Hessian-tracking runs completed only 1 of the paper's 100 epochs for the completed
  independent configurations.
- The independent SimCLR orbit run reached 0 of 50 epochs; the exact 21.85x result is
  from the authors' released arrays.
- Claim 3 uses a genuine low-dimensional head-output slice, not the full 8192-wide
  ambient tensor.
- DINO's very wide heads are reduced to the top 512 real principal directions for the
  loss-geometry tests; orbit analysis uses the released native-width heads.
- No downstream accuracy, probe result, current judge score, or author endorsement is
  claimed.
