# Status

- Paper: *The Geometry of Projection Heads: Conditioning, Invariance, and Collapse*
- Author: Faris Chaudhry
- Repository: `MachineLearning-Nerd/icml26-geometry-projection-heads-conditioning-invariance-collapse`
- Overall verdict: `SCOPED_CLAIMS_1_TO_6_VERIFIED_WITH_EXPLICIT_BUDGET_AND_ASSUMPTION_BOUNDARIES`
- Publication boundary: `NO_FULL_BUDGET_INDEPENDENT_TRAINING_RELEASED_ARRAY_AND_CHECKPOINT_DEPENDENCIES`; `publication_allowed=false`, `score_claim=false`, `official_author_endorsement=false`.
- Historical judged baseline: 5/12 at revision `099048293db504eb467f72c37f7bfd371dadcfcb`; no current score is claimed.
- Branches: 12 descriptive branches (`main`, 9 `experiment/*`, and 2 `release/*`); the stale `master` branch and old `exp/*` names are absent.
- Attribution target: `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>` for every reachable author and committer record.

## Claim outcomes

| Claim | Outcome | Short evidence boundary |
| --- | --- | --- |
| C1 | `verified_scoped` | Exact float64 `G/M` decomposition on real states under the theorem's MSE premise. |
| C2 | `verified_scoped` | Symbolic all-subspaces certificate plus eight real objectives. |
| C3 | `verified_scoped` | Proof plus two qualifying real geometries; excluded cases are explicit. |
| C4 | `verified_scoped` | Released 100-epoch arrays match 0.339/0.609/0.669; estimator resolution is reported. |
| C5 | `verified_scoped` | Eight named objectives and four official checkpoints. |
| C6 | `verified_scoped` | Released arrays give 21.85245268 (21.85x); untrained control is 0.9654x. |

The complete claim-to-production mapping is in [CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md),
and the machine-readable record is in [reproduction_verdicts.json](reproduction_verdicts.json).
