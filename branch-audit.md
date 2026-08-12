# Branch audit

This file is the branch contract for the cleaned repository. The old refs are
included so a result can be traced back to the original experiment node. Branch
names describe purpose; the committed repro/config.py on the ref remains the
authority for exact behavior.

## Original refs and clean refs

| Original ref | Original tip | Clean ref | Role and evidence |
| --- | --- | --- | --- |
| main | 613725d | main | Integrated documentation, current candidate logbook, and the latest verifier source. Claims 1–6 are summarized here. |
| master | 613725d | deleted | Stale duplicate of main; no unique commits or evidence. |
| exp/collapse-gelu-collapsed | e4a770e | experiment/collapse-gelu-collapsed | GELU projection head with collapsed initialization; exact effective-Hessian mechanism and negative-spectrum evidence for Claims 1 and 4. |
| exp/collapse-linear-collapsed | baeb36d | experiment/collapse-linear-collapsed | Linear projection head with collapsed initialization; negative control where the interaction term M vanishes. Claims 1 and 4. |
| exp/collapse-relu-collapsed | c0e6048 | experiment/collapse-relu-collapsed | ReLU projection head with collapsed initialization; piecewise-linear negative control and Figure 3 spectrum. Claims 1 and 4. |
| exp/collapse-swish-collapsed | 7ae0a85 | experiment/collapse-swish-collapsed | Swish projection head with collapsed initialization; smooth-head interaction and negative-eigenvalue analysis. Claims 1 and 4. |
| exp/collapse-swish-normal | d9681bc | experiment/collapse-swish-normal | Swish projection head with standard initialization; non-collapsed comparison for the Hessian-spectrum analysis. Claims 1 and 4. |
| exp/orbits | b2e5cee | experiment/orbits | SimCLR pretraining and augmentation-orbit geometry; independent corroboration for Claim 6. |
| exp/pretrained | 3e779d9 | experiment/pretrained | Official SSL checkpoint geometry and initial loss-geometry path. Claims 2, 3, and 5. |
| exp/pretrained-v2 | d5d6b32 | experiment/pretrained-v2 | Follow-up official-checkpoint geometry with expanded measurements. Claims 2, 3, and 5. |
| exp/pretrained-v3 | e4e9cb3 | experiment/pretrained-v3 | Corrected VICReg loading and bounded Hessian width; current official-checkpoint geometry path. Claims 2, 3, and 5. |
| exp/released | 8376189 | release/released-array-audit | Independent re-analysis of the authors' released full-scale arrays. Claims 1, 4, and 6. |
| exp/released-v2 | af44aec | release/released-array-audit-v2 | Released ablation-key convention and Assumptions 6–7 audit. Claims 1, 4, and 6. |

## Final branch shape

After cleanup the public repository contains 12 branches:

    main
    experiment/collapse-gelu-collapsed
    experiment/collapse-linear-collapsed
    experiment/collapse-relu-collapsed
    experiment/collapse-swish-collapsed
    experiment/collapse-swish-normal
    experiment/orbits
    experiment/pretrained
    experiment/pretrained-v2
    experiment/pretrained-v3
    release/released-array-audit
    release/released-array-audit-v2

There are no exp/ branches and no duplicate master branch after the migration.

## Evidence rules

1. main is the documentation landing ref, not a substitute for an experiment ref.
2. Each experiment or release ref is independently inspectable and retains its
   original code and artifacts.
3. A claim page may combine multiple refs, but it must name the source artifact and
   the producing verifier.
4. A numerical result is not promoted from partial to full-budget evidence merely
   because a later node exists. The limitations page records shortened runs,
   platform changes, and excluded numerical cases.
5. All reachable commits use the approved MachineLearning-Nerd author and committer
   identity. Branch tips and live GitHub refs are verified after publication.

## Claim routing

| Claim | Primary refs | Supporting artifacts |
| --- | --- | --- |
| 1 — Theorem 4.1 | collapse activation refs and release refs | repro/hessian.py, repro/collapse.py, released arrays, PSD-premise audit |
| 2 — Theorem 3.1 | main, experiment/pretrained-v3 | symbolic certificate, real loss Hessians, geometry.py |
| 3 — Proposition 3.3 | main, experiment/pretrained-v3 | two-route curvature calculation, positive-definite cases, flat control |
| 4 — Figure 3 | collapse activation refs and release refs | released spectra, Spearman reproduction, estimator-resolution audit |
| 5 — Section 6 | experiment/pretrained, v2, v3 | eight named objectives, seven distinct implementations, four official checkpoints |
| 6 — Figure 5 | experiment/orbits and release refs | orbit_visualization.npy, spread formulas, bootstrap interval, untrained control |

## Migration note

The original repository name included an internal reproduction identifier and the
experiment refs used abbreviated exp/ prefixes. The clean name and branch names are
descriptive without changing the evidence or code. The original repository name,
paper identifier, old branch tips, and historical judged snapshot remain documented
for provenance.
