# Claim 4 — Figure 3: the Hessian spectrum during training

**Verdict: VERIFIED** on the claim's stated quantifiers, with an explicit caveat about
the resolution of the paper's own estimator that is reported rather than smoothed over.

## The exact claim and its quantifiers

> Figure 3 tracks the Hessian spectrum during training and shows that smooth-activation
> heads inject negative eigenvalues enabling escape from collapse, whereas ReLU-based
> heads fail to do so, relying instead on discrete dynamics and BatchNorm.
> (Figure 3, Section 6.1)

The surrounding text states three exact numbers — the Spearman correlation between
representation variance and condition number: **0.339** (normal init + Swish),
**0.609** (pseudo-collapsed + Swish), **0.669** (pseudo-collapsed + ReLU).

## The stated numbers, recomputed from raw arrays

The authors released the raw 100-epoch trajectories behind Figure 3 as
`results/cifar10/resnet18/hessian_tracker/raw_data_*.npz`. Each was re-analysed with an
**independent rank-correlation implementation** written for this reproduction (ties
averaged explicitly, so the result does not depend on a library default), dropping epoch
0 exactly as the authors' plotting code does.

| Run | Paper | Recomputed | Agreement |
| --- | --- | --- | --- |
| `raw_data_collapsed_relu.npz` | 0.669 | `0.669` | match |
| `raw_data_collapsed_swish.npz` | 0.609 | `0.609` | match |
| `raw_data_normal_swish.npz` | 0.339 | `0.339` | match |

All three of the paper's stated Spearman values reproduce to three decimal places:
**yes**.

## Full spectral summary of the released runs

| Released run | Epochs | epochs with lambda_min < 0 | < -1e-7 | < -1e-6 | min lambda_min | var(final)/var(initial) | Spearman(var, kappa) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `raw_data_collapsed_relu.npz` | 100 | `1` | `1` | `1` | `-1.856e-06` | `1.000341` | **`0.669`** |
| `raw_data_collapsed_relu.npz` | 100 | `1` | `1` | `1` | `-1.856e-06` | `1.000341` | **`0.669`** |
| `raw_data_collapsed_swish.npz` | 100 | `72` | `1` | `0` | `-3.122e-07` | `1.027248` | **`0.609`** |
| `raw_data_collapsed_swish.npz` | 100 | `72` | `1` | `0` | `-3.122e-07` | `1.027248` | **`0.609`** |
| `raw_data_normal_swish.npz` | 100 | `69` | `32` | `5` | `-6.346e-06` | `2.089762` | **`0.339`** |
| `raw_data_normal_swish.npz` | 100 | `69` | `32` | `5` | `-6.346e-06` | `2.089762` | **`0.339`** |

### The caveat that has to be stated

Counting "epochs with a negative eigenvalue" reproduces the paper's qualitative story:
pseudo-collapsed Swish is negative in 72 of 100 epochs against 1 of 100 for ReLU. But the
magnitudes are of order `1e-7`, and **ReLU's single excursion (`-1.86e-6`) is larger in
magnitude than Swish's worst (`-3.12e-7`)**. At a `1e-6` threshold the ordering inverts.

Those values came from a 20-step float32 power iteration. A 20-step shifted power
iteration on a 512-dimensional operator has no claim to resolving `1e-7` against a
spectrum of order `1e-3`, so the *sign pattern* in the released arrays is at or below its
own estimator's resolution. The frequency contrast (72% against 1%) is robust; the
individual signs are not. This is a limitation of the released evidence, and it is the
reason the mechanism was re-tested directly rather than by counting signs.


## What actually settles the claim

The direct test is on [Claim 1](#/claim-1-theorem-4-1): with the exact float64
decomposition, negative curvature is shown to enter `H_eff` **through the interaction
term `M` and only through `M`** — `G` stays PSD, `M` is indefinite for smooth heads, and
`M` vanishes to machine precision for linear and ReLU heads. That is the mechanism
Figure 3 is illustrating, measured directly instead of inferred from a low-resolution
eigenvalue estimate.

## Raw data

- [`raw/claim4_released_hessian_runs.csv`](raw/claim4_released_hessian_runs.csv) — per-run summaries with each source
  file's SHA-256
- [`raw/claim1_4_hessian_tracking.csv`](raw/claim1_4_hessian_tracking.csv) — the
  independent runs, exact and estimated `lambda_min` side by side

## Verifier

`repro/released.py` (`claim4`) and `repro/collapse.py`. The verifier exits non-zero if any
of the three Spearman values fails to reproduce to three decimal places.


### Provenance and how to re-run

| | |
| --- | --- |
| Reproduction repository | [https://github.com/MachineLearning-Nerd/icml26-repro-y4uR1LFClc-the-geometry-of-projection-heads-conditioning-invariance-and-collapse](https://github.com/MachineLearning-Nerd/icml26-repro-y4uR1LFClc-the-geometry-of-projection-heads-conditioning-invariance-and-collapse) |
| Fixed command (identical on every node) | `uv run --frozen repro/run_all.py` |
| Environment | pinned by `pyproject.toml` + `uv.lock`; torch/torchvision from the CPU-only wheel index |
| Compute | Hugging Face `cpu-upgrade`, measured 8 vCPU (cgroup) on AMD EPYC 7R13; **no GPU anywhere** — the runner asserts `torch.cuda.is_available() is False` and aborts otherwise |
| Paper source of record | `https://ar5iv.labs.arxiv.org/html/2605.17180`, SHA-256 `c344481c6fa2c59b6439f41d2053c737d92e11da1e4a7890941c776188ade7a4` |
| Authors' released code and raw arrays | [https://github.com/farischaudhry/projection-head-geometry](https://github.com/farischaudhry/projection-head-geometry) @ `117231d60fee34d4906d1f16c5007e13a96a4d94` |

A node is fully identified by `(repository, git ref)`: what it does is decided only by
`repro/config.py` committed on that ref, never by a flag or an environment variable.
