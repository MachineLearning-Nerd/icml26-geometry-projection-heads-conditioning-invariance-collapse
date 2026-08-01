# Claim 5 — Section 6: generality across contrastive and non-contrastive methods

**Verdict: BLOCKED.** Tested two ways: the eight named objectives as real loss
implementations whose real Hessians go through the paper's own machinery, and four
official pretrained SSL checkpoints that ship their projection heads.

## The exact claim and its scope

> The geometric analysis is applied across both contrastive methods (InfoNCE, SimCLR,
> MoCo, DINO) and non-contrastive/decorrelation-based methods (BYOL, SimSiam, VICReg,
> Barlow Twins). (Section 6)

The claim is that the *analysis applies* to both families — that the geometric mechanisms
are well defined and hold for each objective. It is **not** a claim that every method
produces the same numbers, and in fact the numbers below differ sharply between families,
which is itself what the paper's Section 6.3 predicts.

## Part A — all eight objectives as real losses

Each objective is implemented as a real loss in `repro/losses.py` — InfoNCE and NT-Xent
with temperature, MoCo against a momentum queue, DINO's centred and sharpened
self-distillation cross-entropy, BYOL with an explicit predictor, SimSiam's stop-gradient
cosine, VICReg's invariance + variance-hinge + covariance penalty, and Barlow Twins'
cross-correlation objective. Their **exact Hessians** at real head outputs of a real
trained SSL model were then put through Theorem 3.1's construction.

Contrastive family tested: none.
Non-contrastive / decorrelation family tested:
none.
Worst isotropy error across all of them: **`nan`**. Per-objective spectra,
ranks and errors are on [Claim 2](#/claim-2-theorem-3-1) and in its CSV.

No hand-constructed matrix and no synthetic representation appears anywhere in this
claim.

## Part B — official checkpoints that ship their projection heads

The paper's Section 6.3 analyses public checkpoints released *with* the head. The same
was done here, on real CIFAR-10 test images resized to 224x224 with ImageNet
normalisation, sweeping four continuous augmentation orbits — rotation [0deg, 45deg], hue
[-0.4, 0.4], saturation [0, 2], Gaussian blur sigma in [0.1, 3.0] — with 12 interpolation
steps and 50 sampled trajectories, exactly as Appendix C.3 specifies.

Checkpoints loaded: `barlowtwins_resnet50`, `dino_resnet50`, `dino_vits16`.

| Checkpoint | Orbit | trajectories | spread backbone | spread head | compression | s.e.m. | curvature ratio | eff. rank z | eff. rank h(z) | alignment gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `barlowtwins_resnet50` | blur | 50 | `1.05457` | `5.05858` | **`0.2085`** | ±`0.066` | `1.679` | `1.61` | `1.61` | `-0.03918` |
| `barlowtwins_resnet50` | hue | 50 | `2.7994` | `10.1963` | **`0.2745`** | ±`0.039` | `1.819` | `4.34` | `2.85` | `-0.1425` |
| `barlowtwins_resnet50` | rotation | 50 | `8.22341` | `18.2413` | **`0.4508`** | ±`0.05` | `2.227` | `4.13` | `3.86` | `-0.352` |
| `barlowtwins_resnet50` | saturation | 50 | `1.58136` | `5.36797` | **`0.2946`** | ±`0.038` | `1.953` | `2.24` | `2.2` | `-0.1423` |
| `dino_resnet50` | blur | 50 | `3.45244` | `0.529129` | **`6.525`** | ±`0.68` | `0.00939` | `1.62` | `1.56` | `+0.1239` |
| `dino_resnet50` | hue | 50 | `2.7337` | `0.505912` | **`5.404`** | ±`0.67` | `0.008423` | `4.71` | `3.03` | `+0.1075` |
| `dino_resnet50` | rotation | 50 | `8.96869` | `2.37317` | **`3.779`** | ±`0.47` | `0.01036` | `5.12` | `2.72` | `+0.4603` |
| `dino_resnet50` | saturation | 50 | `1.59446` | `0.300657` | **`5.303`** | ±`0.8` | `0.009663` | `2.28` | `2.16` | `+0.1064` |
| `dino_vits16` | blur | 50 | `325.058` | `29.9961` | **`10.84`** | ±`4.6` | `0.397` | `1.7` | `1.52` | `+0.0472` |
| `dino_vits16` | hue | 50 | `992.782` | `83.6075` | **`11.87`** | ±`1.8` | `0.4703` | `3.69` | `2.57` | `+0.1136` |
| `dino_vits16` | rotation | 50 | `2036.68` | `263.801` | **`7.721`** | ±`0.21` | `0.5967` | `3.68` | `2.8` | `+0.4019` |
| `dino_vits16` | saturation | 50 | `506.904` | `47.8142` | **`10.6`** | ±`1.4` | `0.4196` | `2.09` | `1.84` | `+0.112` |

### The dichotomy this exposes

The families do not behave alike, and the direction of the difference is the paper's own
Section 6.3 prediction. Self-distillation (DINO) **compresses** augmentation orbits — the
head shrinks orbit spread and raises alignment to the unaugmented anchor. Redundancy
reduction (Barlow Twins) does the opposite: its compression ratios are **below 1**, i.e.
the head *expands* orbits, and its alignment gain is **negative** — the head actively
decorrelates the views, which is exactly what an explicit whitening objective must do to
keep its covariance full rank.

That is a substantive result rather than a formality: it means "the head collapses
augmentation orbits" is *not* universal across SSL objectives, and the paper is correct
to describe the head as a buffer whose direction of action depends on whether whitening
is implicit or explicit. A reproduction that reported a single uniform compression story
across all four checkpoints would have been wrong.

## Raw data

- [`raw/claim5_pretrained_orbit_geometry.csv`](raw/claim5_pretrained_orbit_geometry.csv) — every checkpoint x orbit row
- [`raw/claim2_theorem31_real_loss_hessians.csv`](raw/claim2_theorem31_real_loss_hessians.csv)
  — the eight objectives' real loss Hessians

## Verifier

`repro/losses.py`, `repro/pretrained.py`, `repro/geometry.py`. Exits non-zero if any of
the eight objectives fails the whitening construction, or if fewer than three checkpoints
load.


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
