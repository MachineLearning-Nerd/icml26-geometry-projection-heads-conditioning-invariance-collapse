# Conclusion

## Current scoped result

The current candidate evidence marks all six registered claim contracts as
**VERIFIED**, with the qualifications recorded on the individual claim pages. This is
the result of the current evidence campaign; it is not a claim that every original
training budget was rerun.

| Claim | Current evidence | Important qualification |
| --- | --- | --- |
| 1 — Theorem 4.1 | Exact float64 effective-Hessian decomposition on real trained states | The PSD conclusion is evaluated under the theorem's MSE premise; the paper's cosine objective is indefinite |
| 2 — Theorem 3.1 | Exact symbolic universal whitening certificate plus eight real loss Hessians | InfoNCE and SimCLR are the same implementation and count as one distinct objective |
| 3 — Proposition 3.3 | Proof-level obstruction and two qualifying real VICReg geometries with flat controls | Four indefinite cases and two fourth-derivative failures are excluded from the numerical curvature table |
| 4 — Figure 3 | Released arrays reproduce Spearman 0.669, 0.609 and 0.339 | The paper's short float32 power iteration under-resolves some eigenvalue magnitudes |
| 5 — Section 6 | Eight named methods, seven distinct losses, and four official checkpoints | The result establishes the documented scope, not every possible SSL objective or checkpoint |
| 6 — Figure 5 | Released arrays reproduce 21.85x and the untrained-head control is near 1x | The independent training run reached only the 0-epoch corroboration point |

## Reproduction status

The historical judged Space revision 099048293db504eb467f72c37f7bfd371dadcfcb
scored 5/12 because its checks used hand-constructed matrices. That snapshot remains
preserved at [Historical rejected baseline](#/verification-run). The current pages
replace those checks with exact Hessians, real trained networks, the authors'
released arrays, official checkpoints, and explicit assumption audits.

The most important remaining boundary is training budget: the independent collapse
runs are shortened, while the paper-scale 100-epoch behaviour is checked by
independent re-analysis of the authors' released arrays. Those are complementary
pieces of evidence, not interchangeable claims of a full clean-room rerun.

For the evidence registry, start at [Current verification](#/current-verification),
then follow Claims 1–6 and [Limitations and deviations](#/limitations-and-deviations).
