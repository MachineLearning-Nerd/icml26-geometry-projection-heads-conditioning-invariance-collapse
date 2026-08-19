# Source audit

- **Paper:** The Geometry of Projection Heads: Conditioning, Invariance, and Collapse
- **Author:** Faris Chaudhry
- **arXiv:** https://arxiv.org/abs/2605.17180
- **OpenReview:** https://openreview.net/forum?id=y4uR1LFClc
- **Source of record:** https://ar5iv.labs.arxiv.org/html/2605.17180
- **Source SHA-256:** `c344481c6fa2c59b6439f41d2053c737d92e11da1e4a7890941c776188ade7a4`
- **Authors' code and arrays:** https://github.com/farischaudhry/projection-head-geometry at `117231d60fee34d4906d1f16c5007e13a96a4d94`

The six evaluated objects are Theorem 4.1 (collapse instability), Theorem 3.1
(local subspace whitening), Proposition 3.3 (curvature barrier), Figure 3 (Hessian
spectrum), Section 6 (generality across SSL methods), and Figure 5/Table 1 (orbit
compression). Exact statements, assumptions, quantifiers, and registered predicates
are preserved in `.openresearch/artifacts/source_audit.md` and
`.openresearch/artifacts/claim_contracts.json`.

The most important source boundaries are:

- Theorem 4.1's PSD conclusion is conditional on a PSD loss Hessian. The paper's
  SimSiam negative-cosine objective violates that ambient premise, so this is reported
  as an assumption boundary rather than as a theorem refutation.
- Proposition 3.3 is evaluated only where the loss Hessian is a positive-definite
  Riemannian metric; four indefinite and two fourth-derivative-failure cases are
  excluded explicitly.
- Figure 3's released lambda-min values were produced by a 20-step float32 estimator;
  its resolution caveat is part of the result.
- Figure 5's 21.85x value is recomputed from the authors' released arrays. The local
  independent SimCLR run reached 0 epochs and is corroboration only.
