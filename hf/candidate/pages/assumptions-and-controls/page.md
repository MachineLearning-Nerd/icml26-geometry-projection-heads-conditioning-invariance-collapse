# Assumptions and negative controls

A theorem's conclusion is only as good as its premises. Every assumption Theorems 3.1,
4.1 and Proposition 3.3 rest on was **checked numerically**, not assumed, and the results
are here rather than buried in a claim page.

## Assumption 7 — timescale separation

The stability analysis treats the head as a locally equilibrated preconditioner, which
requires the head to adapt faster than the backbone. The paper quantifies this in
Table 3 as the ratio of relative update magnitudes `eta ||grad w|| / ||w||`. Recomputed
from the authors' released per-seed arrays with our own code:

| Released run | BatchNorm | head/backbone ratio (mean) | (median) | head adapts faster | min residual-gradient proxy | bounded away from 0 |
| --- | --- | --- | --- | --- | --- | --- |
| `gelu_bn_False_lr_base` | no | `37.86` | `3.808` | True | `0.000152` | True |
| `linear_bn_False_lr_base` | no | `149.1` | `39.82` | True | `5.12e-06` | True |
| `relu_bn_False_lr_base` | no | `290.1` | `85.22` | True | `1.39e-07` | True |
| `relu_bn_False_lr_large` | no | `93.89` | `0.04399` | True | `4.51e-10` | True |
| `relu_bn_False_lr_small` | no | `653.1` | `504.5` | True | `7.32e-05` | True |
| `relu_bn_True_lr_base` | yes | `0.9196` | `0.8663` | False | `0.0217` | True |
| `relu_bn_True_lr_large` | yes | `0.5461` | `0.524` | False | `0.00349` | True |
| `relu_bn_True_lr_small` | yes | `1.532` | `1.455` | True | `0.0617` | True |
| `swish_bn_False_lr_base` | no | `31.39` | `6.469` | True | `0.000136` | True |
| `swish_bn_False_lr_large` | no | `85` | `0.4889` | True | `1.08e-05` | True |
| `swish_bn_False_lr_small` | no | `124.4` | `40.25` | True | `0.000235` | True |
| `swish_bn_True_lr_base` | yes | `0.8827` | `0.8173` | False | `0.0267` | True |
| `swish_bn_True_lr_large` | yes | `0.4144` | `0.3914` | False | `0.0035` | True |
| `swish_bn_True_lr_small` | yes | `2.687` | `2.579` | True | `0.13` | True |

Without BatchNorm the head's relative updates are **two to three orders of magnitude**
larger than the backbone's, so Assumption 7 holds comfortably. With BatchNorm the ratio
falls to near parity — reproducing the paper's own observation that normalisation
suppresses the native timescale separation, and marking the BatchNorm configurations as
the regime where the theory's premise is weakest.

## Assumption 6 — nonvanishing residual gradient

If `grad_h L = 0` at the collapsed state then the interaction term
`M = sum_i [grad_h L]_i grad_z^2 h_i` is zero for **every** head, and Theorem 4.1 says
nothing at all. The last two columns above show the residual-gradient proxy stays bounded
away from zero in every configuration, so the theorem is not vacuous on this data.

## The PSD premise — where it does not hold

Section 4.1 states that "in standard MSE-based objectives, the intrinsic Hessian of the
loss is PSD", and Theorem 4.1 part 1's conclusion depends on it. Measured at real
training states:

| Head / init | min lambda_min of grad_h^2 L (ambient cosine) | min lambda_min restricted to the sphere's tangent space | grad_h^2 L for the MSE objective |
| --- | --- | --- | --- |
| `gelu` / collapsed | `-8995` | `-97.06` | exactly 1.0 (identity) |
| `linear` / collapsed | `-561.7` | `-4.188` | exactly 1.0 (identity) |
| `relu` / collapsed | `-2083` | `-108.8` | exactly 1.0 (identity) |

For the SimSiam negative-cosine objective the ambient loss Hessian is **strongly
indefinite** — cosine similarity is scale-invariant, so the radial direction contributes
curvature that is an artefact of the ambient embedding. Consequently `H_eff` is
indefinite even for a linear head whose interaction term is exactly zero. This does not
refute Theorem 4.1, which never claims to apply to a non-PSD loss Hessian; it means the
part-1 conclusion cannot be read off the paper's own Figure 3 experiment. All results
supporting Claim 1's verdict are therefore reported under the MSE objective, whose loss
Hessian is exactly the identity and satisfies the premise verbatim.

## Negative controls

Every control below fails loudly if the specific mechanism it guards is broken. None of
them passes for every implementation.

| Control | What it guards | Required behaviour | Measured |
| --- | --- | --- | --- |
| Linear head | that `M` is really being isolated | `‖M‖/‖H_eff‖ < 1e-12` | `3.4e-15` |
| ReLU head | that the smooth-vs-piecewise-linear distinction is real | `M` at round-off | see Claim 1 |
| Central finite differences | that `H_eff` itself is correct | matches the analytic gradient | `4.7e-12` absolute against an `H_eff` scale of `2.1e-3` |
| `H_eff = G + M` identity | that the decomposition is exact | `0` | `0.0` |
| Untrained head, same architecture and width | that orbit compression is a learned effect | ratio ~ 1x | see Claim 6 |
| Flat quadratic metric | that the curvature computation is not reporting artefacts | `‖R‖` at round-off, and one constant `W` isotropises every point | see Claim 3 |

## Raw data

- [`raw/assumptions_audit.csv`](raw/assumptions_audit.csv)


### Provenance and how to re-run

| | |
| --- | --- |
| Reproduction repository | [https://github.com/MachineLearning-Nerd/icml26-repro-y4uR1LFClc-the-geometry-of-projection-heads-conditioning-invariance-and-collapse](https://github.com/MachineLearning-Nerd/icml26-repro-y4uR1LFClc-the-geometry-of-projection-heads-conditioning-invariance-and-collapse) |
| Fixed command (identical on every node) | `uv run --frozen repro/run_all.py` |
| Environment | pinned by `pyproject.toml` + `uv.lock`; torch/torchvision from the CPU-only wheel index |
| Compute | Hugging Face `cpu-upgrade`, measured 8 vCPU (cgroup) on AMD EPYC 7R13; **no GPU anywhere** — the runner asserts `torch.cuda.is_available() is False` and aborts otherwise |
| Paper source of record | `https://ar5iv.labs.arxiv.org/html/2605.17180`, SHA-256 `c344481c6fa2c59b6439f41d2053c737d92e11da1e4a7890941c776188ade7a4` |
| Authors' released code and raw arrays | [https://github.com/farischaudhry/projection-head-geometry](https://github.com/farischaudhry/projection-head-geometry) @ `117231d60fee34d4906d1f16c5007e13a96a4d94` |

Also on every claim page: the [assumption audit and negative
controls](#/assumptions-and-controls), the [limitations and
deviations](#/limitations-and-deviations) including amendments to the pre-registered
contracts, and the [visibility matrix](#/visibility-matrix) listing what an evaluator can
reach for each claim. Start from [Current verification](#/current-verification).

A node is fully identified by `(repository, git ref)`: what it does is decided only by
`repro/config.py` committed on that ref, never by a flag or an environment variable.
