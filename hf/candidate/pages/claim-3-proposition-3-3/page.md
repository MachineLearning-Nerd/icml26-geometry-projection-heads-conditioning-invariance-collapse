# Claim 3 — Proposition 3.3: the curvature barrier for linear heads

**Verdict: BLOCKED.** The concrete missing capability is named below. This claim is not
reported as verified on partial evidence, and it is not reported as falsified either.

## The exact claim and its quantifiers

> **Proposition 3.3 (Curvature Barrier for Linear Heads).** Assume Assumption 1. Let
> `g_L(u) := grad_u^2 L(u)` be the Riemannian metric induced by the loss. If the
> intrinsic geometry `(H, g_L)` has nonvanishing Riemann curvature `R_L != 0`, there
> exists **no** global constant linear map `W` such that the induced effective Hessian
> `H_eff(z) = W^T g_L(Wz) W` is **everywhere** nondegenerate and isotropic on `Z`.

This is a **non-existence** statement, universally quantified over all constant `W`. No
finite search over `W` can establish it. Fitting one `W` at one point and observing that
it fails elsewhere is corroboration, not proof — so this claim is only dischargeable by
a proof-level argument, and this reproduction treats anything less as BLOCKED.

## What is required, and what is missing

The planned discharge was a reconstructed proof, not a search, in three steps, all
implemented and published as source on this Space (`repro/geometry.py`):

1. **Establish the hypothesis, not just the conclusion.** `R_L != 0` must be measured
   for the loss geometry in question, otherwise the proposition is vacuous — a flat
   metric genuinely *is* globally whitenable. `riemann_tensor` computes `R` from
   autograd Christoffel symbols and `riemann_from_cubic_form` recomputes it
   independently via `R_ijkl = (1/4)(C_ikm g^mn C_jln - C_ilm g^mn C_jkn)`; the two
   routes must agree, so a bug in either is caught.
2. **The proof step** (`constant_W_obstruction`). If `W` were everywhere isotropic then
   isotropy forces `rank(W) = d`, so `g_L` is constant on `range(W)`, so the Christoffel
   symbols vanish there, so `R_L` vanishes on that range — contradicting step 1. The
   non-existence is thereby derived, never searched.
3. **Negative control.** A flat metric must produce `R = 0` *and* admit a global `W`.
   A control that failed for both curved and flat metrics would prove nothing.

**The concrete missing capability: compute.** All three run inside the pretrained-head
job, which the platform terminated for lack of account credit (HTTP 402) before reaching
the geometry stage. No `PROP33` record exists. Re-running the published command on a
funded account is the whole of what is needed.

## What exists at toy scale, and why it is not enough

A clean-room numpy check on a constructed curved metric `g_L(z) = I + z z^T` shows a
condition number of about `12.62` at one point against `1.0` at another, while a flat
metric stays whitenable. It is preserved verbatim on the
[Historical rejected baseline](#/verification-run) page. It illustrates the barrier on a
4-dimensional constructed example; it does not measure `R_L` on a real loss geometry and
it does not discharge a non-existence statement over all `W`. That is corroboration, not
credit.


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
