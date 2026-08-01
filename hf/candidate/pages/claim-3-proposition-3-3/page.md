# Claim 3 — Proposition 3.3: the curvature barrier for linear heads

**Verdict: VERIFIED.** The non-existence step is discharged as a proof, not a search,
and both of its hypotheses are then established numerically on real SSL loss geometries
with a flat-metric control that behaves the opposite way.

## What this verdict rests on, and what was excluded

Eight curvature computations were attempted (four objectives x two real pretrained
checkpoints). **2 qualify**; the rest are excluded, and excluding them is not a
detail to bury:

| Outcome | Count | Why it is excluded |
| --- | --- | --- |
| Positive-definite `g_L`, curvature computed | **2** | counted |
| Indefinite `g_L` (`lambda_min(g_L) < 0`) | 4 | Assumption 1 requires `g_L = grad^2 L` to *be* a Riemannian metric. Where the real loss Hessian is indefinite it is not one, so the proposition's hypothesis does not hold there and the point is evidence of nothing — in either direction |
| Fourth-derivative failure | 2 | the curvature tensor needs four derivatives of the loss; these objectives still route through a backward formula that masks a degenerate case in place, which is not differentiable at that order |

The indefinite cases are a **finding, not just an attrition**: at real head outputs of
real pretrained SSL models, the loss Hessians of InfoNCE and SimSiam are indefinite —
`infonce`/`barlowtwins_resnet50` at `-8.689e-04`, `simsiam`/`barlowtwins_resnet50` at `-1.676e-03`, `infonce`/`vicreg_resnet50` at `-1.199e-05`, `simsiam`/`vicreg_resnet50` at `-1.398e-04`.
That is the same assumption violation this reproduction reports for the SimSiam cosine
objective under [Claim 1](#/claim-1-theorem-4-1), found independently here by a different
computation. It means Proposition 3.3 simply does not speak to those objectives, which is
a statement about the proposition's scope rather than about its truth.

So the verdict below rests on a **narrow base: 2 loss geometries, from
1 objective(s)**, plus the flat-metric control and the
proof-level non-existence argument. That is enough to make the proposition non-vacuous
and to exhibit the barrier, and it is not enough to claim the barrier was surveyed across
SSL objectives. Both halves of that sentence are meant.

## The exact claim and its quantifiers

> **Proposition 3.3 (Curvature Barrier for Linear Heads).** Assume Assumption 1. Let
> `g_L(u) := grad_u^2 L(u)` be the Riemannian metric induced by the loss. If the
> intrinsic geometry `(H, g_L)` has nonvanishing Riemann curvature `R_L != 0`, there
> exists **no** global constant linear map `W` in `R^{k x d}` such that the induced
> effective Hessian `H_eff(z) = W^T g_L(Wz) W` is **everywhere** nondegenerate and
> isotropic on the optimization manifold `Z`.

This is a **non-existence statement quantified over all constant `W`**. No finite search
over `W` can establish it. Fitting one `W` at one point and observing that it fails
elsewhere is corroboration, not proof — so the argument is given first, and the
measurements then establish its hypotheses.

## Step 1 — the non-existence argument, reconstructed independently

Suppose such a `W` exists. "Everywhere nondegenerate and isotropic" means
`W^T g_L(Wz) W = I_d` for all `z` in `Z`. Then for any two points `z_1, z_2`,

```
W^T ( g_L(W z_1) - g_L(W z_2) ) W = 0 .
```

Isotropy forces `rank(W^T g_L W) = d`, hence `rank(W) = d`, so `W` has a left inverse on
its column space. Therefore `g_L(W z_1)` and `g_L(W z_2)` agree as bilinear forms on
`range(W)` for **every** pair of points: `g_L` is *constant* on the image
`{W z : z in Z}` in the coordinates supplied by `W`. A metric with constant components
in some coordinate system has identically vanishing Christoffel symbols, hence an
identically vanishing Riemann tensor on that set — contradicting `R_L != 0`. []

The proposition therefore reduces to two machine-checkable facts about a real SSL loss
geometry: that `R_L` is genuinely nonzero, and that `g_L` genuinely varies between real
points so the constancy a global `W` would force is actually false.

## Step 2 — the Riemann tensor of a real SSL loss metric

`g_L = grad^2 L` was computed for real SSL objectives on an `m`-dimensional slice of a
real head output space. That slice is itself a genuine head output space — a head with
`k = m` mapping into it realises exactly this loss geometry — so the proposition applies
verbatim.

The full Riemann tensor was computed **two independent ways**: from autograd Christoffel
symbols with no closed form assumed, and from the classical Hessian-metric identity
`R_ijkl = 1/4 (C_ikm g^{mn} C_jln - C_ilm g^{mn} C_jkn)` with cubic form
`C_ijk = d_i d_j d_k L`, which uses only third derivatives. Agreement between two
routes with different derivative orders is what rules out a differentiation bug.

| Objective | slice dim | ||R|| (Christoffel route) | ||R|| (cubic-form route) | relative difference | max |sectional curvature| | ||dg|| |
| --- | --- | --- | --- | --- | --- | --- |
| `vicreg` | 6 | `1.26826e-05` | `1.26826e-05` | `3.69e-10` | `3.55466e-06` | `0.018276` |
| `vicreg` | 6 | `2.53399e-07` | `2.53399e-07` | `1.83e-08` | `7.70677e-08` | `0.00137109` |

Largest curvature found: **`1.26826e-05`**. Worst disagreement between the two routes:
**`1.83e-08`**.

### The negative control

The same code was run on a genuinely quadratic loss, whose metric is constant by
construction and whose curvature must therefore be exactly zero:

`||R|| = 0`, `||dg|| = 0` — at round-off, as required.

This is a control that fails for the intended reason: had the curvature machinery been
reporting artefacts of the discretisation or of the autograd graph, the flat metric would
have shown curvature too.

## Step 3 — the obstruction, measured on real points

A single `W` was fitted to isotropise `g_L` at one real point and then applied unchanged
at other real points, alongside the same procedure on the flat control.

| Objective | r | isotropy error at each point (fitted at point 0) | ||g_L(point) - g_L(point 0)||_F |
| --- | --- | --- | --- |
| `infonce` | 6 | 2.74e-14, 1.89, 0.533, 1.18, 1.03 | 0, 0.00378, 0.00361, 0.00492, 0.00121 |
| `vicreg` | 64 | 1.68e-14, 3.25, 1.91, 3.22, 3.45 | 0, 0.56, 0.341, 0.554, 0.595 |
| `quadratic_flat_control` | 64 | 1.74e-14, 1.74e-14, 1.74e-14, 1.74e-14, 1.74e-14 | 0, 0, 0, 0, 0 |
| `infonce` | 63 | 6.51e-14, 0.335, 1.5, 0.204, 2.27 | 0, 0.00021, 0.000148, 0.000179, 0.000224 |
| `vicreg` | 64 | 1.68e-14, 3.03, 1.78, 3, 3.22 | 0, 0.559, 0.328, 0.553, 0.593 |
| `quadratic_flat_control` | 64 | 1.74e-14, 1.74e-14, 1.74e-14, 1.74e-14, 1.74e-14 | 0, 0, 0, 0, 0 |

The real objectives leave isotropy error bounded away from zero at every point other than
the one the map was fitted at, and their metrics differ materially between points — the
constancy that a global `W` would force is false. The flat control is isotropised at
every point by the single map, exactly as the argument predicts.

## Raw data

- [`raw/claim3_curvature.csv`](raw/claim3_curvature.csv) — curvature by both routes, metric spectra,
  sectional curvatures

## Verifier

`repro/geometry.py` (`riemann_tensor`, `riemann_from_cubic_form`,
`constant_W_obstruction`) driven by `repro/pretrained.py`. It exits non-zero if the two
curvature routes disagree, if the flat control reports nonzero curvature, or if the
fitted `W` isotropises every point (which would contradict the obstruction).


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
