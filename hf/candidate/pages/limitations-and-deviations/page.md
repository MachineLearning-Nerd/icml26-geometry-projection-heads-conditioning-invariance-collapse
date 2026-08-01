# Limitations and deviations

Stated plainly, including the ones that weaken the result.

## Shortened training budgets

The paper trains the Hessian-tracking runs for 100 epochs and the SimCLR checkpoint for
50. Measured throughput on `cpu-upgrade` was 57-70 images/s for a ResNet-18 step at
32x32, and the augmentation pipeline competes with training for the same 8 vCPU, giving
roughly 45 minutes per CIFAR-10 epoch over two views. Running five configurations for 100
epochs is several hundred CPU-hours.

- Independent training runs did not report before release.

The shortened runs are reported as shortened runs everywhere they appear. They are never
described as the paper's full budget. The full 100-epoch behaviour is covered by
re-analysis of the authors' released arrays, and the shortened budget by independent
regeneration; neither substitutes for the other, and both are shown.

## Seeds

The paper reports three seeds for the collapse experiments. The independent runs here use
one fixed seed per configuration. The released three-seed arrays are re-analysed **per
seed**, and per-seed values are printed rather than hidden inside a mean — which matters,
because for the linear head on CIFAR-10/ResNet-18 the three seeds disagree by two orders
of magnitude.

## The PSD premise does not hold for the paper's own objective

This is the most important caveat in the reproduction. For SimSiam's negative-cosine
loss the ambient `grad_h^2 L` is strongly indefinite, so Theorem 4.1 part 1's conclusion
(`H_eff >= 0` for a linear head) does not apply to the objective the paper's own Figure 3
optimises. Claim 1's verdict is therefore decided under the theorem's stated premise — a
standard MSE objective — and the ambient and sphere-tangential readings are reported
alongside so the difference is visible rather than hidden. See
[Assumptions and controls](#/assumptions-and-controls).

## Representation variance is a proxy, and is treated as one

Figures 3 and 4 plot representation variance, but Theorem 4.1 is a statement about the
spectrum of the effective Hessian at a collapsed state. On the authors' released runs the
variance trajectories do **not** always agree with the theorem's downstream reading: the
linear head on CIFAR-10/ResNet-18 grows rather than staying flat, and on ViT-Tiny the
smooth heads end below their initial variance. Those observations are reported on
[Claim 1](#/claim-1-theorem-4-1). They are not treated as falsifying the theorem, because
the theorem concerns an infinitesimal escape direction at an *exactly* collapsed `z*`
whereas these runs use a pseudo-collapsed initialisation and finite-step SGD — and
because a proxy disagreeing with a theorem is not the same as the theorem being false.

## Head width for the loss-geometry tests

DINO's released heads emit 60,000 (ResNet-50) and 65,536 (ViT-S) prototype logits, where
a dense loss Hessian is roughly 29 GB and out of reach on this hardware. The
loss-geometry tests therefore use the 8192-d Barlow Twins / VICReg heads reduced to their
top 512 principal directions of the real image batch, with the retained variance fraction
reported. That reduced space is still a real head output space. Orbit geometry uses the
released heads at their **full native width**, DINO included.

## Curvature is computed on a slice

The Riemann tensor for Proposition 3.3 is computed on a low-dimensional slice of the head
output space, not on the full ambient space — the full tensor at width 8192 is not
computable here. The slice is a genuine head output space (a head with `k = m` mapping
into it realises exactly that geometry), so the proposition applies verbatim to it, but
this is a scoped instance rather than a statement about the ambient geometry.

## What is not claimed

- No claim is made about downstream accuracy, probe results, or Table 1's information
  rows; those are outside the six claims under evaluation.
- The independently trained SimCLR run is **corroboration** that orbit compression is a
  real property of a trained head. It is not, and is not presented as, a reproduction of
  the specific value 21.85x, which is a property of one particular checkpoint.
- No score is claimed. Only the live judge can change the score.


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
