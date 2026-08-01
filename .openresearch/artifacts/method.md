# Method

## Fixed command and pinned environment

Every experiment-tree node runs exactly

```bash
uv run --frozen repro/run_all.py
```

with no arguments and no environment overrides. What a node does is decided by
`repro/config.py`, committed on that node's own git branch. The environment is pinned by
`pyproject.toml` + `uv.lock` (Python 3.12, torch/torchvision from the CPU-only wheel
index). `--frozen` makes the lockfile authoritative.

On Hugging Face, `jobs/run_node.py` clones this repository at a pinned git SHA and runs
that same command, so a job is fully identified by `(repo, ref)`.

## Compute

All research compute runs on Hugging Face **`cpu-upgrade`**. No GPU is used anywhere;
`repro/run_all.py` asserts `torch.cuda.is_available() is False` and aborts otherwise.
The local machine only inspects, edits and orchestrates.

Estimated core requirement before each run: 8 (torch's CPU convolutions saturate the
quota). Measured allocation on every job: cgroup quota **8 vCPU**, AMD EPYC 7R13, while
`os.cpu_count()` reports **64**. `src/threads.py` reads the real quota from
`/sys/fs/cgroup/cpu.max` and pins `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB` plus
`torch.set_num_threads` before numpy or torch is imported; without that the thread pools
size themselves from the host core count and contention dominates.

Measured throughput (job `6a6d74eea00abefd4b28abbf`): SimSiam-style ResNet-18 step at
32×32 — 70.2 img/s at batch 64, 57.3 img/s at batch 256, i.e. 24–29 minutes per CIFAR-10
epoch over two views. That measurement is what set the epoch budgets below.

Per-job runtimes, flavors and outcomes are in `JOBS.md` and in the saved logs under
`.openresearch/artifacts/jobs/`.

## The exact effective Hessian

The paper's central object is the effective Hessian of the objective with respect to the
backbone representation, equation (1):

```
H_eff(z) = J_h(z)^T (grad_h^2 L) J_h(z)  +  sum_i [grad_h L]_i grad_z^2 h_i(z)
           \________ pullback metric G ________/  \______ interaction term M ______/
```

In the SimSiam-style objective used for the collapse experiments the term that reaches
`z_1` is `-0.5 cos(predictor(projector(z_1)), c)` with `c = p_2` detached, and the second
term is fully detached. Two consequences follow, and both are load-bearing:

1. `z_i -> loss_i` factors per sample, so `d^2 L / dz^2` is **block diagonal** with
   512×512 blocks. The full block is computable exactly.
2. The map passes only through the two small MLP heads, never through the ResNet, so the
   exact block costs about half a second in float64.

`repro/hessian.py` therefore computes `H_eff` by `torch.func.hessian` in **float64**,
`J` by `jacfwd`, and `grad_h^2 L` by `hessian` on the loss alone, then forms
`G = J^T (grad_h^2 L) J` and `M = H_eff - G`. This replaces the paper's 20-step float32
shifted power iteration, which is still run at every tracked state as an independent
cross-check.

Deciding when an eigenvalue counts as negative uses a fixed relative floor,
`EIG_TOL = 1e-10` of the matrix's largest-magnitude eigenvalue, set before any run and
never tuned per configuration.

### Three readings of the loss Hessian

Theorem 4.1 part 1 concludes `H_eff ⪰ 0` for a linear head, and that conclusion needs
`grad_h^2 L ⪰ 0` — which Section 4.1 asserts for "standard MSE-based objectives". The
smoke run showed that assumption does not hold in ambient coordinates for the cosine
objective the paper's own Figure 3 optimises. Each tracked state therefore reports:

| Reading | What it is | Why |
| --- | --- | --- |
| ambient cosine | `grad_h^2` of `-0.5 cos(h, c)` | the objective actually optimised |
| tangential | the same restricted to `T_h S^{k-1}` via `P H_L P` | cosine is scale-invariant, so the radial direction contributes curvature that is an artefact of the ambient embedding; this is the intrinsic geometry the paper reasons about |
| MSE | `grad_h^2` of `0.5 ||h - c||^2`, exactly the identity | the theorem's stated PSD premise, holding verbatim |

Reporting all three is the difference between testing the theorem and testing a nearby
statement it does not make.

## Negative controls

| Control | Must do | Fails for the right reason if |
| --- | --- | --- |
| linear head | `‖M‖/‖H_eff‖ < 1e-12` | the decomposition or the autograd graph is wrong |
| ReLU head | `‖M‖/‖H_eff‖` at round-off (second derivative zero a.e.) | the smooth-vs-piecewise-linear distinction is not actually being measured |
| finite differences | central differences of the analytic gradient match `H_eff` columns | the exact-Hessian machinery is wrong |
| untrained head, same architecture and output width | orbit compression ≈ 1× | the compression were a dimensionality artefact rather than a learned metric |
| flat quadratic metric | Riemann tensor at round-off, and one constant `W` isotropises every point | the curvature computation reported curvature for a flat space |

None of these passes for every implementation: each one fails loudly if the specific
mechanism it guards is broken.

## Deviations from the paper, stated up front

- **Epoch budget.** The paper trains the Hessian-tracking runs for 100 epochs and the
  SimCLR checkpoint for 50. At the measured 24–29 min/epoch on `cpu-upgrade`, 100 epochs
  is ~48 h per configuration for five configurations. The runs here are shortened and
  each claim page states the exact number of epochs completed. The authors' released
  100-epoch arrays are independently re-analysed alongside, so the full budget is
  covered by re-analysis and the shortened budget by independent regeneration.
- **Seeds.** The paper reports three seeds for the collapse experiments. The independent
  runs here use a single fixed seed per configuration; the released three-seed arrays are
  re-analysed per seed, and per-seed spread is reported rather than hidden in a mean.
- **DINO head width.** DINO's released heads emit 60,000 and 65,536 prototype logits. A
  dense loss Hessian at that width is ~29 GB, so the loss-geometry tests use the 8192-d
  Barlow Twins / VICReg heads reduced to their top 512 principal directions of the real
  batch, with the retained variance fraction reported. Orbit geometry uses the full
  released heads at their native width.
- **Batch size for the orbit run.** Appendix C states batch size 512; the authors' own
  SimCLR loader hardcodes 256. The independent run uses the stated 512.
