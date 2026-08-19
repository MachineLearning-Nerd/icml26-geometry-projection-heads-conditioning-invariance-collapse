# Scoped reproduction report

## Overall status

`SCOPED_CLAIMS_1_TO_6_VERIFIED_WITH_EXPLICIT_BUDGET_AND_ASSUMPTION_BOUNDARIES`

All six registered claim contracts are satisfied by the current candidate evidence,
with the scope and limitations stated in the claim pages and `CLAIM_EVIDENCE.md`.
This is a scoped audit, not a claim that every original training budget was rerun
independently.

| Claim | Status | Evidence boundary |
| --- | --- | --- |
| C1 | `verified_scoped` | Exact real-state float64 Hessian decomposition; MSE PSD premise and shortened-run limits are explicit. |
| C2 | `verified_scoped` | Symbolic universal certificate plus real Hessian instantiations. |
| C3 | `verified_scoped` | Proof plus two qualifying real positive-definite geometries; excluded cases remain excluded. |
| C4 | `verified_scoped` | Released 100-epoch arrays match all three Spearman values; estimator resolution is a caveat. |
| C5 | `verified_scoped` | Eight named objectives and four official checkpoints, with InfoNCE/SimCLR implementation overlap disclosed. |
| C6 | `verified_scoped` | Released-array 21.85x and negative control; independent training is 0-epoch corroboration. |

## Publication policy

`NO_FULL_BUDGET_INDEPENDENT_TRAINING_RELEASED_ARRAY_AND_CHECKPOINT_DEPENDENCIES`;
`publication_allowed=false`, `score_claim=false`, and
`official_author_endorsement=false`.

The historical judged baseline was 5/12 at revision
`099048293db504eb467f72c37f7bfd371dadcfcb`. It is preserved as provenance only and is
not presented as a current score.

## What is ready for review

The README, claim ledger, candidate pages, raw summaries, source hash, branch map,
citation, author thanks, state, and aggregate manifest are published. Run
`python3 verify_final.py` from a clone to check the live 12-branch contract and the
canonical MachineLearning-Nerd history.
