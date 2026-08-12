# Claim 1 — Theorem 4.1: generic instability of collapse

**Verdict: VERIFIED** — decided against the pre-registered contract reproduced at the
bottom of this page, on the exact float64 effective Hessian of real trained networks.

## The exact claim and its quantifiers

> **Theorem 4.1 (Generic Instability of Collapse).** Let `z*` be a collapsed state.
> Under Assumptions 1, 6 and 7, assume the residual gradient `rho = grad_h L` is nonzero
> at `z*`.
> 1. **Linear heads preserve collapse:** if `h(z) = Wz` is linear, the interaction term
>    vanishes (`grad^2 h = 0`). The effective Hessian is PSD (`H_eff >= 0`). Thus the
>    collapsed state is a non-repelling critical region and gradient descent has no
>    infinitesimal escape direction.
> 2. **Nonlinear heads destabilize collapse:** if `h_phi` is a nonlinear MLP with smooth
>    activations, then under Assumption 5, `z*` is *generically* a strict instability
>    region.

Two quantifiers matter and are easy to lose. Part 2 says **generically**, not always.
And part 1's *conclusion* (`H_eff >= 0`) is conditional on `grad_h^2 L >= 0`, which
Section 4.1 asserts holds "in standard MSE-based objectives". Section 4.2 adds that
because `grad^2 ReLU = 0` almost everywhere, ReLU heads behave like linear ones **under
continuous gradient flow, without BatchNorm** — a ReLU run with BatchNorm or a large step
size is explicitly outside the claim.

## What was measured, and why it is the theorem's own quantity

The paper's equation (1) splits the effective Hessian into two terms:

```
H_eff(z) = J_h(z)^T (grad_h^2 L) J_h(z)  +  sum_i [grad_h L]_i grad_z^2 h_i(z)
           \________ pullback metric G ________/  \______ interaction term M ______/
```

Theorem 4.1 turns entirely on `M`. So rather than infer the mechanism from an eigenvalue
sign, `G` and `M` are computed **separately and exactly**, in float64, at real training
states of real ResNet-18/CIFAR-10 SimSiam-style runs. This is tractable because in that
objective the term reaching `z_1` is `-0.5 cos(predictor(projector(z_1)), c)` with the
second term fully detached, so `d^2L/dz^2` is block diagonal with 512x512 blocks whose
graph passes only through the two small MLP heads.

Results below are under the theorem's **own** premise — a standard MSE objective, whose
loss Hessian is exactly the identity — so part 1's conclusion is entitled to hold. The
ambient-cosine and sphere-tangential readings are in the CSV and are discussed under
"Assumption audit".

Each row is one exact 512x512 effective Hessian in float64, at the first tracked state
of that configuration. `n_neg` counts eigenvalues below `-1e-10 x` the matrix's own
spectral scale; the scale is printed so the floor is checkable by hand.

| Head / init | ‖M‖/‖H_eff‖ | n_neg(G) | n_neg(H_eff) | lambda_min(H_eff) | spectral scale | H_L PSD |
| --- | --- | --- | --- | --- | --- | --- |
| `gelu` / collapsed | `0.121` | 0 | **72** | `-6.322e-08` | `1.961e-06` | yes |
| `linear` / collapsed | `2.82e-15` | 0 | **0** | `2.238e-11` | `3.132e-05` | yes |
| `relu` / collapsed | `2.64e-15` | 0 | **0** | `-4.715e-21` | `1.077e-05` | yes |
| `swish` / collapsed | `0.0765` | 0 | **57** | `-3.382e-08` | `1.956e-06` | yes |
| `swish` / normal | `0.535` | 0 | **148** | `-6.909e-03` | `2.579e-02` | yes |

### Reading the table

- **Linear head:** `‖M‖/‖H_eff‖` at `2.82e-15` — the interaction term vanishes to
  machine precision, exactly as part 1 states, and `H_eff = G` stays PSD with
  `n_neg = 0`.
- **ReLU head:** `‖M‖/‖H_eff‖` at `2.64e-15`. The second derivative of ReLU is zero
  almost everywhere, so the theorem predicts it behaves like the linear head, and it does
  — `n_neg = 0`, despite ReLU being a nonlinearity.
- **Smooth heads (GELU, Swish):** `‖M‖/‖H_eff‖` at least `0.0765`, and at least
  `57` negative eigenvalues appear where the linear and ReLU heads have none.
- **`n_neg(G) = 0` in every row.** The pullback metric is PSD throughout, so every
  negative eigenvalue in `H_eff` is contributed by `M`. That is Theorem 4.1's stated
  mechanism, measured rather than inferred from a sign.

The linear and ReLU rows are not decoration: they are the negative controls. A pipeline
that reported negative curvature for them would be reporting numerical noise, and the
claim would have been withdrawn.

### Assumption 6 is not assumed here, it is forced

Theorem 4.1 part 2 requires the residual gradient `rho = grad_h L` to be nonzero at the
collapsed state; if `rho = 0` the statement is vacuous. This is checked rather than
asserted, and it does not need a separate measurement:

> `M = sum_i rho_i grad_z^2 h_i`, so `rho = 0` forces `M = 0` **exactly**.
> Measuring `‖M‖/‖H_eff‖ >= 0.0765` for the smooth heads therefore *proves*
> `rho != 0` at those states.

Part 1 needs no such guard: for a linear head `grad_z^2 h = 0`, so `M = 0` is an
identity in `rho` and holds whatever the gradient is. This is why the linear and ReLU
rows cannot pass vacuously — their content is `n_neg(H_eff) = 0`, not `M = 0`.

**Why counts and not a `lambda_min < 0` fraction.** For a piecewise-linear head `M`
vanishes and `lambda_min` is a round-off residue landing on either side of zero at
random: ReLU's was `-1.1e-16` against a spectral scale of `2.0e-6`. An earlier revision
of the tracker aggregated a bare `lambda_min < 0` and so reported ReLU as negative in
100% of states — noise presented as the paper's mechanism. Fixed in commit `bb64527`;
every figure on this page uses the relative floor.

## Assumption audit

Theorem 4.1's premises were checked rather than assumed.

**`grad_h^2 L >= 0` (the PSD premise).** It does *not* hold for the objective the paper's
own Figure 3 optimises. At real training states the ambient Hessian of SimSiam's negative
cosine has strongly negative eigenvalues, so `H_eff` is indefinite **even for a linear
head whose interaction term is exactly zero**. This is not a refutation of Theorem 4.1 —
the theorem does not claim to apply to a non-PSD loss Hessian — but it does mean that the
part-1 conclusion cannot be read off the paper's own experiment. The `H_L_lambda_min_min`
column of the CSV records this at every state.

**Assumption 6 (nonvanishing residual gradient).** If `grad_h L = 0` then `M = 0` for
*every* head and the theorem is vacuous. Recomputed from the authors' released
per-seed arrays, the residual-gradient proxy stays bounded away from zero in every
no-BatchNorm configuration.

**Assumption 7 (timescale separation).** The head must adapt faster than the backbone.
Recomputed from the released arrays, the head/backbone relative-update ratio without
BatchNorm is of order 10^2, and near parity with BatchNorm — reproducing the pattern of
the paper's Table 3. See [Assumptions and controls](#/assumptions-and-controls).

## Downstream behaviour on the authors' released runs

Representation variance is a *consequence* of the theorem, not the theorem. It is
reported here because it is what Figure 4 plots, and because it does not always agree
with the theorem's prediction — which is stated rather than hidden.

| Setting | Head | Epochs | mean var(final)/var(initial) | mean max/initial | per-seed final/initial |
| --- | --- | --- | --- | --- | --- |
| cifar10/resnet18 | `gelu` | 50 | `6.449` | `15.59` | 5.57, 7.7, 6.08 |
| cifar10/resnet18 | `linear` | 50 | `16.76` | `22.96` | 44.1, 0.208, 5.92 |
| cifar10/resnet18 | `relu` | 50 | `0.1063` | `1` | 0.102, 0.0969, 0.12 |
| cifar10/resnet18 | `swish` | 50 | `4.49` | `9.475` | 3.01, 6.31, 4.14 |
| cifar10/vit_tiny | `gelu` | 50 | `0.07159` | `1` | 0.0443, 0.0937, 0.0768 |
| cifar10/vit_tiny | `linear` | 50 | `0.444` | `1` | 0.899, 0.0346, 0.398 |
| cifar10/vit_tiny | `relu` | 50 | `0.1702` | `1` | 0.178, 0.195, 0.138 |
| cifar10/vit_tiny | `swish` | 50 | `0.1044` | `1` | 0.0736, 0.2, 0.0392 |
| cifar100/resnet18 | `gelu` | 20 | `7.557` | `13.8` | 6.01, 9.44, 7.22 |
| cifar100/resnet18 | `linear` | 20 | `1.07` | `1.269` | 1.81, 0.711, 0.693 |
| cifar100/resnet18 | `relu` | 20 | `0.1281` | `1` | 0.111, 0.123, 0.15 |
| cifar100/resnet18 | `swish` | 20 | `3.581` | `5.143` | 0.423, 4.54, 5.78 |

The linear head on CIFAR-10/ResNet-18 grows rather than staying flat, and its per-seed
values are extremely dispersed. On ViT-Tiny the smooth heads end *below* their initial
variance. Neither observation contradicts Theorem 4.1 as stated: the theorem concerns the
existence of an infinitesimal escape direction at an exactly collapsed `z*`, whereas
these runs start from a *pseudo*-collapsed initialisation (projector weights scaled by
0.1) and run finite-step SGD for many epochs. Treating a variance trajectory as a direct
test of the theorem would be substituting a proxy for the claim; the exact `M` and
`lambda_min` measurements above are the direct test.

## Raw data

- [`raw/claim1_4_hessian_tracking.csv`](raw/claim1_4_hessian_tracking.csv) — every tracked epoch of every
  configuration, all three loss-Hessian readings, plus the paper's own power-iteration
  estimate at the same states
- [`raw/claim1_released_variance.json`](raw/claim1_released_variance.json) — the released per-seed variance trajectories re-analysed above

## Verifier and controls

`repro/hessian.py` and `repro/collapse.py`. `collapse.verify_machinery()` runs before
every tracking run and exits non-zero unless: central finite differences of the analytic
gradient reproduce `H_eff`; `M` vanishes to `< 1e-12` relative for a linear head; and `M`
is materially non-zero for a smooth head. Measured on these runs: finite-difference
agreement `4.7e-12` absolute against an `H_eff` scale of `2.1e-3`, linear-head
`‖M‖/‖H_eff‖ = 3.4e-15`, Swish-head `‖M‖/‖H_eff‖ = 0.49`, and the identity
`H_eff = G + M` exact to `0.0`.


### Provenance and how to re-run

| | |
| --- | --- |
| Reproduction repository | [https://github.com/MachineLearning-Nerd/icml26-geometry-projection-heads-conditioning-invariance-collapse](https://github.com/MachineLearning-Nerd/icml26-geometry-projection-heads-conditioning-invariance-collapse) |
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
