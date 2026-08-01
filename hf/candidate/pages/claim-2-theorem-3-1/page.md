# Claim 2 — Theorem 3.1: implicit local subspace whitening

**Verdict: BLOCKED.** The concrete missing capability is named below. This claim is not
reported as verified on partial evidence, and it is not reported as falsified either.

## The exact claim and its quantifiers

> **Theorem 3.1 (Local Subspace Whitening).** Let `z*` be a fixed point. Under
> Assumptions 1 and 3, let `r = rank(grad_h^2 L|h(z*))` be the intrinsic rank of the
> loss. **For any subspace `S` of `T_z*Z` of dimension `r`**, there exists a linear
> projection head `W` (with `k >= r`) such that the effective Hessian restricted to `S`
> is isometric to the identity on `S`: `v^T H_eff(z*) v = ‖v‖^2` for all `v` in `S`.

Two quantifiers decide what counts as evidence. The statement is **universally**
quantified over a continuum of `r`-dimensional subspaces `S`, with an existential `W`
inside it. Finitely many random draws of `S` are corroboration only — the universal
quantifier requires a symbolic discharge. The judged claim wording also says isotropy is
achieved "**only** on the active subspace", so the off-`S` behaviour is part of what
must be shown, not an aside.

## What is required, and what is missing

The planned discharge had two independent parts, both implemented and both published as
source on this Space:

1. **Symbolic certificate** (`repro/geometry.py`, `theorem_31_symbolic_certificate`) —
   builds `W = U_r Lambda_r^(-1/2) B^T` over free symbols using two Householder
   reflections, so orthonormality is exact by construction rather than solved for, then
   shows `B^T W^T H_L W B - I_r` cancels to the zero matrix elementwise. That discharges
   "for any `S`" as an identity in the symbols rather than a sample of subspaces.
2. **Real loss landscapes** (`repro/pretrained.py`, `loss_geometry`) — the same
   construction applied to the loss Hessians of official pretrained SSL projection
   heads, to show the identity is not an artefact of a hand-built matrix.

**The concrete missing capability: compute.** Both parts run inside the pretrained-head
job, which the platform terminated for lack of account credit (HTTP 402) after it had
emitted 9 of 16 orbit records and before it reached the loss-geometry stage. No `THM31`
or `THM31_SYMBOLIC` record exists, so there is nothing to report. Re-running the
published command on a funded account is the whole of what is needed; no new method,
data or derivation is required.

## What exists at toy scale, and why it is not enough

A clean-room numpy check of the whitening identity on a hand-constructed 4x4 rank-2
diagonal Hessian reaches a whitening error of about `1e-16`. It is preserved verbatim on
the [Historical rejected baseline](#/verification-run) page. It is **corroboration at
toy scale only**: one constructed matrix, one subspace, no real loss landscape, and no
discharge of the universal quantifier. Under this reproduction's own rules that is not
full credit, which is why this page reads BLOCKED rather than VERIFIED.

## Assumptions that would need auditing

`r <= d`, otherwise no `r`-dimensional subspace of `T_z*Z` exists at all; and
`grad_h^2 L` PSD, otherwise `H_L^(+1/2)` is not real. Both are checked by the published
verifier and neither has been measured on a real head here.


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
