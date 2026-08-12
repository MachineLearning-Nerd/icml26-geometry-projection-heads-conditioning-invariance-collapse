# Claim 2 — Theorem 3.1: local subspace whitening

**Verdict: VERIFIED.** The universal quantifier is discharged symbolically, and the
construction is then instantiated on the real loss Hessians of eight real SSL objectives
at real head outputs of an official pretrained SSL checkpoint.

## The exact claim and its quantifiers

> **Theorem 3.1 (Local Subspace Whitening).** Let `z*` be a fixed point. Under
> Assumptions 1 and 3, let `r = rank(grad_h^2 L | h(z*))` be the intrinsic rank of the
> loss. **For any subspace `S` of `T_{z*}Z` of dimension `r`**, there exists a linear
> projection head `W` in `R^{k x d}` (with `k >= r`) such that the effective Hessian
> restricted to `S` is isometric to the identity on `S`:
> `v^T H_eff(z*) v = ||v||_2^2` for all `v` in `S`.

This is a **universally quantified existence statement over a continuum** of subspaces.
Finitely many random draws are corroboration only — which is why the quantifier is
discharged symbolically first. The judged claim wording adds "achieving isotropy **only**
on the active subspace determined by the loss rank", so the behaviour off `S` is part of
what must be shown.

## Step 1 — discharging the quantifier symbolically

Write the loss Hessian's eigendecomposition `H_L = U diag(lam) U^T` and keep its `r`
positive eigenpairs `(U_r, Lam_r)`. Given an orthonormal basis `B` of `S`, set

```
W = U_r Lam_r^(-1/2) B^T        (k x d)
```

Then `W B = U_r Lam_r^(-1/2)` and

```
B^T W^T H_L W B = Lam_r^(-1/2) (U_r^T H_L U_r) Lam_r^(-1/2)
                = Lam_r^(-1/2) Lam_r Lam_r^(-1/2)
                = I_r
```

which is an **identity in `(U_r, Lam_r, B)`**, not a numerical coincidence: it holds for
every PSD `H_L` of rank `r` and every `S`.

`repro/geometry.py` (`theorem_31_symbolic_certificate`) discharges this as an
**ideal-membership proof**, which is what makes it a discharge of the quantifier rather
than a sample of subspaces. `U` and `B` are *fully free* symbolic matrices — no
parameterisation is imposed, so nothing restricts which subspace `S = range(B)` or which
eigenbasis `U_r` is covered. The only hypotheses are the orthonormality relations

```
U^T U - I_r = 0        B^T B - I_r = 0
```

and the certificate shows every entry of `B^T W^T H_L W B - I_r` reduces to zero modulo
the ideal those relations generate, via a Gröbner basis in the entries of `U` and `B`.
Membership in that ideal means the identity holds for **every** `U` and `B` satisfying
orthonormality — that is, for every `r`-dimensional `S` and every rank-`r` PSD `H_L` of
these dimensions.

| `r` | `k` | `d` | orthonormality relations | free entries in `U`,`B` | Gröbner basis | isotropy in ideal | `W` annihilates `S`-perp | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 2 | 2 | 5 | 2 | **True** | **True** | `0.04` |
| 2 | 4 | 3 | 6 | 14 | 14 | **True** | **True** | `4.06` |
| 2 | 5 | 4 | 6 | 18 | 14 | **True** | **True** | `16.09` |
Every size reports **proven** (3 of
3). A size that exceeds its wall-clock budget is recorded as timed out and
can never be counted as a pass.

The eigenvalues enter as `lam_i = m_i^2`, so `Lam_r^(-1/2) = diag(1/m_i)` stays rational
and no radicals appear, and the `m_i` sit in the coefficient field rather than among the
generators. Both choices are what make the computation terminate: an earlier revision
expanded a product of symbolic Householder reflections carrying `sqrt()` entries and did
not finish in over ten minutes on the same machine. A Householder parameterisation would
also have been *weaker* — a reflection is one specific family of orthogonal matrices, so
`range(B)` would not have covered every subspace, and the universal quantifier would have
remained undischarged.

The `S`-perp column is the "only on the active subspace" half: `W` annihilates `S`-perp
identically, so `H_eff = W^T H_L W` has rank exactly `r` with range `S`.

## Step 2 — instantiating it on real loss Hessians

The construction was then applied to the **real** Hessians `grad_h^2 L` of all eight SSL
objectives named in Section 6, evaluated at real head outputs of an official pretrained
SSL checkpoint on real CIFAR-10 images, for **25 random `r`-dimensional subspaces**
each.

| Objective | Family | k | numerical rank of grad_h^2 L | r used | worst isotropy error | worst off-subspace leakage | worst rank error | loss Hessian PSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dino` | contrastive | 64 | 55 | 55 | `8.04e-11` | `1.4e-15` | 0 | True |
| `dino` | contrastive | 64 | 1 | 1 | `1.78e-15` | `1.77e-16` | 0 | True |
| `infonce` | contrastive | 64 | 6 | 6 | `2.93e-14` | `6.81e-16` | 0 | False |
| `infonce` | contrastive | 64 | 63 | 63 | `6.57e-14` | `1.17e-16` | 0 | False |
| `moco` | contrastive | 64 | 4 | 4 | `4.68e-15` | `5.96e-16` | 0 | False |
| `moco` | contrastive | 64 | 63 | 63 | `3.68e-14` | `1.06e-16` | 0 | False |
| `simclr` | contrastive | 64 | 6 | 6 | `2.93e-14` | `6.81e-16` | 0 | False |
| `simclr` | contrastive | 64 | 63 | 63 | `6.57e-14` | `1.17e-16` | 0 | False |
| `barlow_twins` | non-contrastive | 64 | 45 | 45 | `7.98e-15` | `6.07e-16` | 0 | False |
| `barlow_twins` | non-contrastive | 64 | 1 | 1 | `1.55e-15` | `1.68e-16` | 0 | False |
| `byol` | non-contrastive | 64 | 63 | 63 | `9.63e-11` | `2.7e-15` | 0 | False |
| `byol` | non-contrastive | 64 | 1 | 1 | `1.44e-15` | `2.94e-16` | 0 | False |
| `simsiam` | non-contrastive | 64 | 1 | 1 | `3e-15` | `3.79e-16` | 0 | False |
| `simsiam` | non-contrastive | 64 | 1 | 1 | `1.33e-15` | `3.83e-16` | 0 | False |
| `vicreg` | non-contrastive | 64 | 64 | 64 | `1.73e-14` | `8.59e-30` | 0 | True |
| `vicreg` | non-contrastive | 64 | 64 | 64 | `1.75e-14` | `8.59e-30` | 0 | True |

Worst isotropy error across every objective and every subspace draw: **`9.63e-11`**.
Worst off-subspace leakage: **`2.7e-15`**.

The `infonce` and `simclr` rows are identical by construction: NT-Xent and InfoNCE are
the same objective at the same temperature, so `repro/losses.py` implements one in terms
of the other. That is one measurement under two of the paper's names, not two
independent instantiations.

### Assumption audit

`r <= d` is required for an `r`-dimensional subspace of the tangent space to exist at
all; the `numerical rank` and `r used` columns show whether the rank had to be truncated
for that reason. The rank decision threshold and the surrounding eigenvalue gap are
recorded per objective in the CSV, so the rank is auditable rather than asserted. Where
a loss Hessian is not PSD the positive part is used and the `loss Hessian PSD` column
records it.

## Raw data

- [`raw/claim2_theorem31_real_loss_hessians.csv`](raw/claim2_theorem31_real_loss_hessians.csv) — one row per objective, with
  spectra, rank decisions and worst-case errors

## Verifier

`repro/geometry.py` (`check_theorem_31`, `theorem_31_symbolic_certificate`) driven by
`repro/pretrained.py`. It exits non-zero if the symbolic identity fails to reduce to
zero, or if any instantiation exceeds `1e-8` isotropy error or off-subspace leakage.


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
