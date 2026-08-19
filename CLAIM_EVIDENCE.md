# Claim evidence ledger

This ledger records what each paper claim is, which repository code or artifact
produces the result, and the boundary on the conclusion. The detailed numerical
tables remain in `hf/candidate/pages/claim-*/page.md`; this file is the compact
repository entrypoint.

## Paper

- **Title:** The Geometry of Projection Heads: Conditioning, Invariance, and Collapse
- **Author:** Faris Chaudhry
- **arXiv:** [2605.17180](https://arxiv.org/abs/2605.17180)
- **OpenReview:** [y4uR1LFClc](https://openreview.net/forum?id=y4uR1LFClc)
- **Source hash:** `c344481c6fa2c59b6439f41d2053c737d92e11da1e4a7890941c776188ade7a4`

## Claim-to-production paths

| Claim | Paper object | Producer and evidence | Scoped result | Boundary |
| --- | --- | --- | --- | --- |
| C1 | Theorem 4.1, collapse instability | `repro/hessian.py`, `repro/collapse.py`, `repro/released.py`; `hf/candidate/pages/claim-1-theorem-4-1/page.md`; `raw/claim1_4_hessian_tracking.csv` | `verified_scoped` | Exact float64 `G/M` decomposition on real states. The PSD conclusion is evaluated under the theorem's MSE premise; the paper's SimSiam cosine Hessian is separately reported as indefinite. Shortened independent runs and released arrays are not a full 100-epoch rerun. |
| C2 | Theorem 3.1, local subspace whitening | `repro/geometry.py`; symbolic certificate plus `raw/claim2_theorem31_real_loss_hessians.csv`; claim page | `verified_scoped` | The universal construction is discharged symbolically, then checked on eight real objectives and 25 random subspaces per instantiation. Rank truncation and non-PSD loss Hessians are recorded rather than hidden. |
| C3 | Proposition 3.3, curvature barrier | `repro/geometry.py`, `repro/pretrained.py`; `raw/claim3_curvature.csv`; claim page | `verified_scoped` | Proof-level non-existence argument plus two qualifying positive-definite real geometries. Four indefinite cases and two fourth-derivative failures are excluded because the proposition's Riemannian premise does not hold there. |
| C4 | Figure 3, Hessian spectrum | `repro/released.py`, `repro/collapse.py`; `raw/claim4_released_hessian_runs.csv`; claim page | `verified_scoped` | Released 100-epoch arrays reproduce Spearman 0.339/0.609/0.669. The 20-step float32 estimator's resolution limit is reported, and exact float64 shortened-state checks provide mechanism evidence. |
| C5 | Section 6, method generality | `repro/losses.py`, `repro/pretrained.py`, `repro/geometry.py`; `raw/claim5_pretrained_orbit_geometry.csv`; claim page | `verified_scoped` | Eight named objectives and four official checkpoints are covered; InfoNCE and SimCLR are one distinct implementation by construction. This is an enumerated scope, not a survey of all SSL methods. |
| C6 | Figure 5, orbit compression | `repro/released.py`, `repro/orbits.py`; `raw/claim6_orbit_compression.json`, `raw/claim6_per_orbit_spread.csv`; claim page | `verified_scoped` | Authors' released arrays reproduce 21.85245268 (21.85x), with formula agreement and a 0.9654x untrained-head control. Independent training reached only the 0-epoch corroboration point and is not a second full-budget reproduction. |

## How claims are produced

1. `source_audit.md` and `claim_contracts.json` freeze the paper statements, quantifiers, assumptions, thresholds, and amendments.
2. `repro/run_all.py` is the fixed entrypoint; `repro/config.py` is the only branch-specific stage selector.
3. Claims 1 and 4 use exact Hessian code plus released trajectories; Claims 2 and 3 use symbolic/real geometry code; Claim 5 uses real loss implementations and official checkpoints; Claim 6 uses released orbit arrays plus an explicitly limited independent run.
4. The candidate logbook pages publish the raw summaries, negative controls, source hashes, and limitations for each claim.
5. `verify_final.py` checks the repository-level state, branch inventory, canonical identity, required files, and manifest. It does not rerun the multi-hour scientific jobs.

## Publication boundary

All six scoped contracts are marked verified, but the collection is not presented as
a full independent reproduction: several training runs were shortened, Claim 3 is
based on two qualifying geometries, Claim 4 relies on released trajectories for its
100-epoch values, and Claim 6's exact number comes from released arrays. Therefore
`publication_allowed=false`, no current judge score is claimed, and no author
endorsement is implied.
