# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_972dc30abd08", "created_at": "2026-07-30T17:02:22+00:00", "title": "Verification output (last 40 lines)"}
-->
## Verification output (last 40 lines)

```
      "status": "PASS",
      "by_loss": {
        "MSE": {
          "whitening_err": 1.1102230246251565e-16,
          "r": 2,
          "linear_min_lam": 0.0,
          "tanh_min_lam": -0.7672323100919166,
          "smooth_destabilizes": true
        },
        "BarlowTwins": {
          "whitening_err": 1.1102230246251565e-16,
          "r": 3,
          "linear_min_lam": 0.0,
          "tanh_min_lam": -0.7672323100919166,
          "smooth_destabilizes": true
        },
        "VICReg": {
          "whitening_err": 1.1102230246251565e-16,
          "r": 3,
          "linear_min_lam": 0.0,
          "tanh_min_lam": -0.3836161550459583,
          "smooth_destabilizes": true
        },
        "InfoNCE": {
          "whitening_err": 0.0,
          "r": 1,
          "linear_min_lam": 0.0,
          "tanh_min_lam": -0.7672323100919166,
          "smooth_destabilizes": true
        }
      },
      "note": "Section 6: the whitening (Thm 3.1) and collapse-destabilization (Thm 4.1) hold across SSL loss families (MSE, Barlow-Twins-, VICReg-, InfoNCE-style)."
    }
  },
  "n_claims_passed": 5,
  "n_claims_total": 5,
  "all_passed": true
}

SUMMARY: claims 5/5 passed, all_passed=True
```
