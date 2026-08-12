# Repro - The Geometry of Projection Heads: Conditioning, Invariance, and Collapse

**Start at [Current verification](#/current-verification).** It is the canonical
entrypoint: it states what supersedes what, gives the fixed command and pinned
environment, and links every claim page, raw data file and control.

arXiv [2605.17180](https://arxiv.org/abs/2605.17180) ·
OpenReview [y4uR1LFClc](https://openreview.net/forum?id=y4uR1LFClc) ·
reproduction repository [https://github.com/MachineLearning-Nerd/icml26-geometry-projection-heads-conditioning-invariance-collapse](https://github.com/MachineLearning-Nerd/icml26-geometry-projection-heads-conditioning-invariance-collapse)

All research compute ran on CPU — Hugging Face `cpu-upgrade` (8 vCPU, measured) for the
training and pretrained-checkpoint runs, and an 8-core local CPU for the symbolic
certificate and the real-loss-landscape geometry. **No GPU was used anywhere.** Each
result states which platform produced it; see
[Current verification → Compute](#/current-verification).

## Pages

| Page |
| --- |
| [Current verification (start here)](#/current-verification) |
| [Claim 1 - Theorem 4.1 (VERIFIED)](#/claim-1-theorem-4-1) |
| [Claim 2 - Theorem 3.1 whitening (VERIFIED)](#/claim-2-theorem-3-1) |
| [Claim 3 - Proposition 3.3 curvature barrier (VERIFIED)](#/claim-3-proposition-3-3) |
| [Claim 4 - Figure 3 Hessian spectrum (VERIFIED)](#/claim-4-figure-3) |
| [Claim 5 - Section 6 generality (VERIFIED)](#/claim-5-section-6-generality) |
| [Claim 6 - Figure 5 orbit compression (VERIFIED)](#/claim-6-orbit-compression) |
| [Assumptions and negative controls](#/assumptions-and-controls) |
| [Limitations and deviations](#/limitations-and-deviations) |
| [Visibility matrix](#/visibility-matrix) |
| [Overview](#/overview) |
| [Claims](#/claims) |
| [Evidence](#/evidence) |
| [Historical rejected baseline (toy 4x4 verification run)](#/verification-run) |
| [Conclusion](#/conclusion) |
