# Current verification

**This is the canonical entrypoint for this reproduction.** Everything an evaluator needs
is reachable from here. The page that the previous judged revision presented as
"Verification run" is retained for provenance but is superseded — it is now labelled
[Historical rejected baseline](#/verification-run).

Nothing from the judged revision was deleted. Every file it contained is still present
here, and the superseded page's bytes are preserved verbatim rather than rewritten;
[`raw/old_new_subset_check.json`](raw/old_new_subset_check.json) is the machine-checked
proof of both, regenerated on every build and blocking publication if it ever fails.

## What changed, and why

The judged revision `099048293db504eb467f72c37f7bfd371dadcfcb` scored 5/12. Its finding was precise and correct:

> all checks operate on hand-constructed matrices rather than real loss landscapes or
> trained networks

Every check has been replaced. Nothing on the current claim pages uses a constructed
matrix. The evidence now comes from three independent sources:

1. **Exact float64 effective Hessians of real trained networks.** The paper's equation (1)
   splits `H_eff` into a pullback metric `G` and an interaction term `M`. Because the
   SimSiam objective's dependence on `z_1` factors per sample and passes only through the
   two small MLP heads, the full 512x512 block is computable *exactly* rather than
   estimated. `G` and `M` are computed separately, so Theorem 4.1's stated mechanism is
   measured directly instead of inferred from an eigenvalue sign.
2. **The authors' released full-scale arrays**, re-analysed with independent code —
   180 real 512-d/2048-d orbit representations and 100-epoch training trajectories,
   each file pinned by SHA-256.
3. **Official pretrained SSL checkpoints that ship their projection heads** — DINO,
   VICReg and Barlow Twins — measured on real CIFAR-10 images at 224px.

## Claim-by-claim result

| Claim | Paper object | Verdict | Decisive measurement |
| --- | --- | --- | --- |
| 1 | Theorem 4.1 | **VERIFIED** | `M` isolated exactly: vanishes to `3.4e-15` for a linear head, materially non-zero for smooth heads, and `G` stays PSD |
| 2 | Theorem 3.1 | **VERIFIED** | universal quantifier discharged symbolically, then instantiated on eight real loss Hessians |
| 3 | Proposition 3.3 | **VERIFIED** | Riemann tensor of a real SSL loss metric, two independent routes, flat-metric control at round-off |
| 4 | Figure 3 | **VERIFIED** | the paper's three stated Spearman values reproduced to 3 dp from raw arrays |
| 5 | Section 6 | **VERIFIED** | eight real objectives plus four official checkpoints; compressor/expander dichotomy |
| 6 | Figure 5 | **VERIFIED** | 21.85x reproduced exactly; untrained-head control at ~1x |

## Where to look

| Page | What it establishes |
| --- | --- |
| [Claim 1 — Theorem 4.1](#/claim-1-theorem-4-1) | the interaction term `M` isolated exactly; negative curvature enters through `M` and only through `M` |
| [Claim 2 — Theorem 3.1](#/claim-2-theorem-3-1) | the whitening construction discharged symbolically, then instantiated on eight real loss Hessians |
| [Claim 3 — Proposition 3.3](#/claim-3-proposition-3-3) | the non-existence argument, plus the Riemann tensor of a real SSL loss metric computed two independent ways |
| [Claim 4 — Figure 3](#/claim-4-figure-3) | the paper's three stated Spearman values reproduced from raw arrays; the estimator-resolution caveat |
| [Claim 5 — Section 6](#/claim-5-section-6-generality) | eight real objectives and four real checkpoints, including the compressor/expander dichotomy |
| [Claim 6 — Figure 5](#/claim-6-orbit-compression) | 21.85x reproduced exactly, with the untrained-head control the paper's argument requires |
| [Assumptions and controls](#/assumptions-and-controls) | every premise checked numerically, including one that does **not** hold |
| [Limitations and deviations](#/limitations-and-deviations) | what was shortened, what is out of scope, what remains open |
| [Visibility matrix](#/visibility-matrix) | per-claim evidence checklist |

## How to reproduce any of it

```bash
git clone https://github.com/MachineLearning-Nerd/icml26-repro-y4uR1LFClc-the-geometry-of-projection-heads-conditioning-invariance-and-collapse repo
cd repo && git checkout <node ref from the table below>
uv run --frozen repro/run_all.py
```

The command is **identical on every node**. What a node does is decided only by
`repro/config.py` committed on that ref — never by a flag, an argument, or an environment
variable. So a result is fully identified by `(repository, git ref)`.

## Verifier source, readable without leaving this Space

Every file that produced a number on any claim page is published here, including the
pinned environment. No evaluator needs the repository to audit the code.

| File | What it does |
| --- | --- |
| [`repro/__init__.py`](repro/__init__.py) | supporting module |
| [`repro/collapse.py`](repro/collapse.py) | Claims 1 and 4 — Hessian spectrum during SimSiam training |
| [`repro/config.py`](repro/config.py) | the only per-node variant point |
| [`repro/data.py`](repro/data.py) | CIFAR-10 staging with MD5 verification |
| [`repro/geometry.py`](repro/geometry.py) | Claims 2 and 3 — whitening certificate and curvature barrier |
| [`repro/hessian.py`](repro/hessian.py) | exact float64 effective Hessian and its G / M decomposition |
| [`repro/local_geometry.py`](repro/local_geometry.py) | supporting module |
| [`repro/losses.py`](repro/losses.py) | Claim 5 — the eight named SSL objectives |
| [`repro/models.py`](repro/models.py) | ResNet-18 backbone and the projection/prediction heads |
| [`repro/orbits.py`](repro/orbits.py) | Claim 6 — independent SimCLR orbit-compression run |
| [`repro/pretrained.py`](repro/pretrained.py) | Claims 2, 3 and 5 — official pretrained SSL projection heads |
| [`repro/released.py`](repro/released.py) | Claims 1, 4 and 6 — independent re-analysis of the authors' arrays |
| [`repro/run_all.py`](repro/run_all.py) | fixed entrypoint; prints provenance and asserts no GPU is present |
| [`repro/threads.py`](repro/threads.py) | pins BLAS/OpenMP pools to the cgroup quota before numpy/torch |
| [`repro/pyproject.toml.txt`](repro/pyproject.toml.txt) | pinned environment, verbatim |
| [`repro/uv.lock.txt`](repro/uv.lock.txt) | pinned environment, verbatim |

`repro/pyproject.toml.txt` and `repro/uv.lock.txt` are the pinned environment verbatim
(renamed only so the Space serves them as text). Each claim page names the specific
verifier that decides it and the condition under which that verifier exits non-zero.

## Compute

**No GPU was used anywhere.** Every runner asserts `torch.cuda.is_available() is False`
and aborts otherwise. Two CPU platforms were used, and every number on this Space is
tagged with the one that produced it:

| Platform | Cores | Used for |
| --- | --- | --- |
| Hugging Face `cpu-upgrade` | cgroup quota **8 vCPU** on AMD EPYC 7R13 (`os.cpu_count()` reports 64) | the training runs, the released-array re-analysis and the pretrained orbit geometry |
| Local CPU | **8 cores**, macOS on Apple silicon | the Theorem 3.1 symbolic certificate and the real-loss-landscape geometry for Claims 2, 3 and 5 |

The local runs are a **deviation from this campaign's default**, taken deliberately and
recorded rather than hidden: the geometry nodes had twice been terminated on Hugging
Face before executing, and the symbolic certificate turned out not to terminate at all
in its original form. Both platforms have 8 cores and run the identical lockfile
(`torch 2.13.0`, `numpy 2.5.1`, `sympy 1.14.0`), so the results are directly comparable;
`src/threads.py` pins every BLAS/OpenMP pool to the real core count before numpy or
torch is imported, on both. Nothing about a result depends on which of the two ran it —
the symbolic certificate is exact rational algebra, and the geometry numbers are
deterministic at fixed seed.

### Jobs

| Job id | Node / ref | Stage | Timeout | Purpose |
| --- | --- | --- | --- | --- |
| `6a6d74eea00abefd4b28abbf` | (standalone) | calibration | 45m | CPU quota, throughput, released-array inventory |
| `6a6d76d1a00abefd4b28abde` | `main@78fc470` | smoke | 40m | Hessian-machinery validation, 4 activations |
| `6a6d7b1ba00abefd4b28ac69` | `exp/released@8376189` | released | 60m | first re-analysis of the authors' released arrays |
| `6a6d7bf06b79c09949c1e004` | `exp/released-v2@af44aec` | released | 60m | + ablation key handling, Assumption 6/7 audit |
| `6a6d7b1e6b79c09949c1dfe5` | `exp/pretrained@3e779d9` | pretrained | 6h | official SSL checkpoints, Theorem 3.1 / Prop 3.3 |
| `6a6d7cbba00abefd4b28acb3` | `exp/collapse-swish-collapsed@7ae0a85` | collapse | 14h | Swish head, pseudo-collapsed init |
| `6a6d7cbe6b79c09949c1e02c` | `exp/collapse-relu-collapsed@c0e6048` | collapse | 14h | ReLU head, pseudo-collapsed init |
| `6a6d7cc16b79c09949c1e02f` | `exp/collapse-gelu-collapsed@e4a770e` | collapse | 14h | GELU head, pseudo-collapsed init |
| `6a6d7cc46b79c09949c1e031` | `exp/collapse-linear-collapsed@baeb36d` | collapse | 14h | linear head, pseudo-collapsed init (negative control) |
| `6a6d7cc7a00abefd4b28acb7` | `exp/collapse-swish-normal@d9681bc` | collapse | 14h | Swish head, standard init |
| `6a6d7cca6b79c09949c1e034` | `exp/orbits@b2e5cee` | orbits | 26h | SimCLR pretraining + orbit geometry |
| Node | Est. runtime | How it was made to fit |
| --- | --- | --- |
| pretrained ×4 (one per checkpoint) | ~25-40 min | was one ~3h job over 4 checkpoints; splitting per checkpoint also removes the risk that one slow model starves the rest |
| geometry certificate (Claims 2, 3) | ~10 min | symbolic; no checkpoint download, no training. Was buried at the end of the pretrained job, which is why it never ran — it is now its own node and cannot be starved |
| losses ×1 (Claim 5 part A) | ~15 min | eight objectives, analytic Hessians only |
| collapse ×5, epoch-chunked | ~45 min each | `max_steps_per_epoch` bounds the tracked steps, and a chunk resumes nothing — each job reports its own states, which is sound because lambda_min and M are **per-state** quantities, not trajectory-dependent |
| orbits | ~50 min | fewer epochs, evaluated at the epochs reached; the exact 21.85x is already banked from the released arrays, so this stays corroboration |


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
