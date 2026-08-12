# Evidence registry

This page is a compact index of the current evidence. The claim pages contain the
tables, raw-file links, assumptions, controls, and verifier paths that decide each
verdict.

| Claim | Evidence source | Production path | Current result |
| --- | --- | --- | --- |
| 1 — Theorem 4.1 | Real trained networks, exact float64 Hessians, and released trajectories | repro/collapse.py, repro/hessian.py, repro/released.py compute H_eff = G + M, spectra, and controls | **VERIFIED**; M/H_eff is about 3.4e-15 for linear and material for smooth heads |
| 2 — Theorem 3.1 | Symbolic algebra and eight real loss Hessians | repro/geometry.py proves the universal construction and checks isotropy/off-subspace leakage | **VERIFIED**; worst reported isotropy error 9.63e-11 |
| 3 — Proposition 3.3 | Proof-level non-existence argument plus real loss metrics | repro/geometry.py computes two curvature routes and a flat control | **VERIFIED** within the qualifying positive-definite cases |
| 4 — Figure 3 | Authors' released full-scale arrays plus exact float64 spot checks | repro/released.py and repro/collapse.py reproduce correlations and audit estimator resolution | **VERIFIED**; Spearman 0.669, 0.609, 0.339 |
| 5 — Section 6 | Eight named objective labels, seven distinct implementations, and four official checkpoints | repro/losses.py supplies losses; repro/pretrained.py measures real checkpoint orbit geometry | **VERIFIED** for the enumerated scope |
| 6 — Figure 5 | Authors' released orbit_visualization.npy and an untrained-head control | repro/released.py recomputes spread ratios and bootstrap intervals | **VERIFIED**; 21.85245268 rounds to 21.85x |

## Evidence principles

- A claim is decided by the current candidate evidence, not by the preserved toy
  baseline.
- Every numerical page names the input artifact, verifier, and relevant git ref.
- Negative controls are retained: linear/ReLU heads, flat metrics, untrained heads,
  estimator-resolution checks, and the PSD-premise audit.
- The paper's original full-budget training is not silently substituted by a shortened
  run. See [Limitations and deviations](#/limitations-and-deviations).

## Raw and source files

The main raw artifacts are linked from the individual claim pages. The reproducible
source lives in repro/; the fixed entrypoint is uv run --frozen repro/run_all.py. The
authors' released code is pinned in the provenance tables and the source audit.

Start with [Current verification](#/current-verification) for the canonical route
through the evidence.
