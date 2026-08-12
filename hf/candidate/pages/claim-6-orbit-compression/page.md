# Claim 6 — Figure 5: 21.85x orbit compression

**Verdict: VERIFIED.** The stated 21.85x reproduces exactly, and the paper's own
falsifiable side-claim — that the compression is not an artefact of the 512 -> 2048
dimension change — is confirmed by a control that the paper did not run.

## The exact claim and its quantifiers

> Figure 5 shows PCA visualization demonstrating **21.85x** compression of augmentation
> orbits under the learned projection head geometry. (Figure 5, Section 6.2)

Table 1 fixes the quantifiers: mean orbit spread `2.25e-2` (backbone) against `0.10e-2`
(head), described in the text as a reduction "from 0.0225 to 0.0010", over **n = 15**
independent augmentation orbits, from the final 50-epoch SimCLR checkpoint on CIFAR-10
with a ResNet-18 backbone. Appendix C.3 fixes the protocol: 5 classes, 3 images per
class, 12 rotations spanning [0deg, 360deg), L2-normalised representations, and mean
orbit spread defined as the mean squared Euclidean distance of an orbit's points from
that orbit's own centroid.

## What was measured

The authors released the raw orbit arrays behind Figure 5:
`results/cifar10/resnet18/orbit_visualization.npy` at
[https://github.com/farischaudhry/projection-head-geometry](https://github.com/farischaudhry/projection-head-geometry) @ `117231d6`,
SHA-256 `3121c85537cf3792dcc238a039b1db8e9d02c544b6c66d8320722a79f622109d`.
These are **180 real representations** — 15 orbits x 12 rotations — of a trained
network: `orbits_z` is `(180, 512)` and `orbits_h` is `(180, 2048)`. Nothing here is
hand-constructed and nothing is 4-dimensional.

Every quantity below was recomputed from those arrays by the code shown further down.
No value is taken from the paper, from the authors' plotting code, or from any other
logbook.

| Quantity | Value | Paper |
| --- | --- | --- |
| Mean orbit spread, backbone `z` (512-d) | `0.0224755748` | 0.0225 |
| Mean orbit spread, head `h(z)` (2048-d) | `0.0010285150` | 0.0010 |
| **Compression ratio** | **`21.85245268`** | **21.85x** |
| Rounded to 2 dp | `21.85` | 21.85 |
| Bootstrap 95% CI over the 15 orbits | `[20.7240, 23.2581]` | not reported |
| Cross-formula disagreement (centroid vs summed variance) | `0` / `0` | n/a |
| **Negative control** — untrained head, same architecture, same 2048-d output | **ratio `0.9654`** | n/a |

The two formulas for mean orbit spread — distance-to-centroid and summed per-coordinate
variance — are algebraically identical, so their agreement to `0.0` is an implementation
check, not a scientific finding. It is reported because a mismatch would have meant a bug.

### Why the negative control is the substance of this claim

The paper argues, in its own words, that the compression "is not merely a byproduct of
dimensionality reduction (the head is actually higher-dimensional, 2048 vs 512), but
rather a targeted geometric collapse induced by the learned metric". That sentence is
falsifiable, and it is what the control tests: a **randomly initialised** head of the
same architecture and the same 2048-d output, applied to the **same** backbone orbits,
gives a compression ratio of **`0.9654x`** — no
compression at all. The 21.85x is therefore attributable to the learned metric and not
to the width change. Had the control also produced a large ratio, this claim would have
been reported as refuted in the form the paper states it.

### PCA rendering underlying the figure

Figure 5 is a PCA projection of these orbits. Reported numerically rather than as an
image, the in-plane orbit spread in the first two principal components is
`3.919469e-04` for the backbone against `4.15200e-05` for the head.


### The same measurement on the ViT-Tiny checkpoint

Table 5 of the paper reports **4015.33x** for the ViT-Tiny head. Recomputed from the
released ViT-Tiny orbit arrays: **`4015.4749x`**
(backbone `0.0108989595`, head `2.714e-06`,
bootstrap 95% CI `[3687.79, 4425.53]`).
The small difference from the stated 4015.33 is consistent with the released arrays being
float32 while this recomputation is float64; it is reported rather than rounded away.
Its untrained-head control gives `1.0383x`.



## Independent reproduction: our own SimCLR training run

The numbers above re-analyse the authors' released arrays. Independently of them, a
SimCLR ResNet-18 was pretrained from scratch on CIFAR-10 on `cpu-upgrade` following
Appendix C — NT-Xent at temperature 0.1, Adam at 1e-3, crop/flip/rotation/colour-jitter
augmentations, a 2-layer 512 -> 2048 -> 2048 ReLU head with BatchNorm — and the same
orbit geometry measured at intermediate epochs.

**This run completed 0 of the paper's 50 epochs** before being stopped at the
campaign's compute budget; it is reported as a 0-epoch run and is not described as
the paper's full budget. Its role is corroboration that the compression is a real
property of a trained head, not confirmation of the specific value 21.85x, which depends
on the particular checkpoint.

| Epoch | spread backbone | spread head | compression | control (untrained head) | D_intra reduction | D_inter reduction | class/orbit separation | curvature ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `0.00654195` | `0.00634515` | **`1.031`** | `1.006` | `1.015` | `1.016` | `0.9994` | `0.9953` |

At epoch 0 the head is untrained and the compression ratio sits at ~1, as it must; the
control column stays near 1 throughout, confirming again that the effect tracks training
rather than architecture width.

Raw data: [`raw/claim6_independent_simclr.csv`](raw/claim6_independent_simclr.csv).

## Raw data

- [`raw/claim6_orbit_compression.json`](raw/claim6_orbit_compression.json) — every field of both orbit records
- [`raw/claim6_per_orbit_spread.csv`](raw/claim6_per_orbit_spread.csv) — the 15 per-orbit values that the mean is taken over

## Verifier

`repro/released.py` in the reproduction repository, run through the fixed command on the
node `exp/released-v2`. It exits non-zero if the recomputed ratio does not round to
21.85, if the two formulas disagree, or if the untrained-head control shows compression.


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
