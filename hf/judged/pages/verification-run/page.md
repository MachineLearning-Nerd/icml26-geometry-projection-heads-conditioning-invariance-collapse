# Verification run


---
<!-- trackio-cell
{"type": "code", "id": "cell_87a91f887bc1", "created_at": "2026-07-30T17:02:23+00:00", "title": "verify all claims", "command": [".venv/bin/python", "repro/src/verify.py"], "exit_code": 0, "duration_s": 0.131}
-->
````bash
$ .venv/bin/python repro/src/verify.py
````

exit 0 · 0.1s


````python title=verify.py
"""
verify.py - verify the 5 anchored claims for "The Geometry of Projection Heads"
(y4uR1LFClc, arXiv 2605.17180).

  [0]/C0  Theorem 4.1 (Generic Instability of Collapse): smooth heads inject negative eigenvalues into
          the effective Hessian at collapse (escape direction); linear/ReLU heads keep H_eff PSD.
  [1]/C1  Theorem 3.1 (Local Subspace Whitening): a linear head makes H_eff isometric to I on the active
          r-dim subspace (machine precision).
  [2]/C2  Proposition 3.3 (Curvature Barrier): nonzero curvature => no global linear whitening
          (control: a FLAT metric IS globally whitenable).
  [3]/C3  Figure 3 (Hessian spectrum): smooth heads show negative eigenvalues; linear/ReLU do not.
  [4]/C4  Section 6 (generality): whitening + collapse-destabilization hold across SSL loss families
          (MSE, Barlow-Twins-style, VICReg-style, InfoNCE-style).
  ([5] Fig 5 PCA compression on real images is the only dataset benchmark - outside this set.)

All via linear algebra on the effective Hessian (eq 1) of constructed SSL-loss Hessians & projection heads.
verdict.json summarizes all checks.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import core as C

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)
verdict = {"paper": "y4uR1LFClc", "checks": {}}

k = 4
H_L = np.diag([3.0, 2.0, 0.0, 0.0])           # PSD loss Hessian, rank r=2 < d=4 (nullspace exists)
rho = np.ones(k)                                # nonzero residual gradient at collapse
W = np.eye(k)
z = np.ones(k) * 0.8

# =====================================================================================
# C0 / [0] Theorem 4.1 - Generic Instability of Collapse
# =====================================================================================
dest = C.thm41_destabilize(H_L, rho, W, z)
c0_ok = (dest["linear"]["lam_min_always_nonneg"] and dest["relu"]["lam_min_always_nonneg"]
         and dest["tanh"]["reaches_negative"] and dest["swish"]["reaches_negative"])
verdict["checks"]["C0_thm41_collapse_instability"] = {
    "status": "PASS" if c0_ok else "FAIL",
    "by_head": {h: dict(min_lam_min=v["min_lam_min"], sign=v["sign"]) for h, v in dest.items()},
    "note": "Theorem 4.1: linear/ReLU heads keep H_eff PSD (collapse non-repelling); smooth (tanh/Swish) "
            "heads reach a negative min eigenvalue (escape direction). z swept over the negative-curvature regime.",
}

# =====================================================================================
# C1 / [1] Theorem 3.1 - Local Subspace Whitening
# =====================================================================================
wh = C.thm31_whitening(H_L)
c1_ok = wh["rel_err"] < 1e-6
verdict["checks"]["C1_thm31_subspace_whitening"] = {
    "status": "PASS" if c1_ok else "FAIL",
    "rank_r": wh["r"], "rel_err": wh["rel_err"],
    "note": "Theorem 3.1: linear head W=H_L^{+1/2} makes the effective Hessian isometric to the identity on "
            "the active r-dim subspace S (S^T H_eff S = I_r), to machine precision.",
}

# =====================================================================================
# C2 / [2] Proposition 3.3 - Curvature Barrier (+ flat-metric negative control)
# =====================================================================================
ng = C.prop33_nogo()
# negative control: a FLAT (constant) metric is globally whitenable (cond stays ~1 everywhere)
d = 4
Wflat = np.diag([1.0 / np.sqrt(2.0)] * d)        # whitening for g=2I
cond_flat = float(np.linalg.cond(Wflat.T @ (2.0 * np.eye(d)) @ Wflat))
c2_ok = ng["isotropic_at_z1"] and ng["anisotropic_at_z2"] and (cond_flat < 1.05)
verdict["checks"]["C2_prop33_curvature_barrier"] = {
    "status": "PASS" if c2_ok else "FAIL",
    "curved_cond_at_z1": ng["cond_at_z1"], "curved_cond_at_z2": ng["cond_at_z2"],
    "flat_metric_control_cond": cond_flat,
    "note": "Proposition 3.3: a curved metric g_L(z)=I+zz^T cannot be globally whitened by one linear W "
            "(isotropic at z1 but anisotropic at z2); CONTROL: a flat metric IS globally whitenable (cond~1).",
}

# =====================================================================================
# C3 / [3] Figure 3 - Hessian spectrum (smooth injects negatives; linear/ReLU do not)
# =====================================================================================
spec = C.hessian_spectrum(H_L, rho, W, np.ones(k) * 2.6)
c3_ok = (spec["linear"]["n_negative"] == 0 and spec["relu"]["n_negative"] == 0
         and spec["tanh"]["n_negative"] > 0 and spec["swish"]["n_negative"] > 0)
verdict["checks"]["C3_fig3_hessian_spectrum"] = {
    "status": "PASS" if c3_ok else "FAIL",
    "n_negative_eigenvalues": {h: v["n_negative"] for h, v in spec.items()},
    "note": "Figure 3: the effective-Hessian spectrum has negative eigenvalues under smooth (tanh/Swish) "
            "heads but not under linear/ReLU heads (evaluated in the negative-curvature regime).",
}

# =====================================================================================
# C4 / [4] Section 6 - generality across SSL loss families
# =====================================================================================
gen = C.generality(k, z=np.ones(k) * 2.6)
c4_ok = all(g["whitening_err"] < 1e-6 and g["smooth_destabilizes"] for g in gen.values())
verdict["checks"]["C4_sec6_generality"] = {
    "status": "PASS" if c4_ok else "FAIL",
    "by_loss": {n: dict(whitening_err=g["whitening_err"], r=g["r"],
                        linear_min_lam=g["linear_min_lam"], tanh_min_lam=g["tanh_min_lam"],
                        smooth_destabilizes=g["smooth_destabilizes"]) for n, g in gen.items()},
    "note": "Section 6: the whitening (Thm 3.1) and collapse-destabilization (Thm 4.1) hold across SSL loss "
            "families (MSE, Barlow-Twins-, VICReg-, InfoNCE-style).",
}

# -------------------------------------------------------------------------------------
verdict["n_claims_passed"] = sum(1 for v in verdict["checks"].values() if v["status"] == "PASS")
verdict["n_claims_total"] = 5
verdict["all_passed"] = all(v["status"] == "PASS" for v in verdict["checks"].values())
with open(os.path.join(OUT, "verdict.json"), "w") as fh:
    json.dump(verdict, fh, indent=2)
print(json.dumps(verdict, indent=2))
print("\nSUMMARY: claims {n}/{t} passed, all_passed={a}".format(
    n=verdict["n_claims_passed"], t=verdict["n_claims_total"], a=verdict["all_passed"]))

````


````output
{
  "paper": "y4uR1LFClc",
  "checks": {
    "C0_thm41_collapse_instability": {
      "status": "PASS",
      "by_head": {
        "linear": {
          "min_lam_min": 0.0,
          "sign": "nonneg-always"
        },
        "relu": {
          "min_lam_min": 0.0,
          "sign": "nonneg-always"
        },
        "tanh": {
          "min_lam_min": -0.7672323100919166,
          "sign": "negative-reaching"
        },
        "swish": {
          "min_lam_min": -0.03689012655290577,
          "sign": "negative-reaching"
        }
      },
      "note": "Theorem 4.1: linear/ReLU heads keep H_eff PSD (collapse non-repelling); smooth (tanh/Swish) heads reach a negative min eigenvalue (escape direction). z swept over the negative-curvature regime."
    },
    "C1_thm31_subspace_whitening": {
      "status": "PASS",
      "rank_r": 2,
      "rel_err": 1.1102230246251565e-16,
      "note": "Theorem 3.1: linear head W=H_L^{+1/2} makes the effective Hessian isometric to the identity on the active r-dim subspace S (S^T H_eff S = I_r), to machine precision."
    },
    "C2_prop33_curvature_barrier": {
      "status": "PASS",
      "curved_cond_at_z1": 1.000000000000001,
      "curved_cond_at_z2": 12.62466627680713,
      "flat_metric_control_cond": 1.0,
      "note": "Proposition 3.3: a curved metric g_L(z)=I+zz^T cannot be globally whitened by one linear W (isotropic at z1 but anisotropic at z2); CONTROL: a flat metric IS globally whitenable (cond~1)."
    },
    "C3_fig3_hessian_spectrum": {
      "status": "PASS",
      "n_negative_eigenvalues": {
        "linear": 0,
        "relu": 0,
        "tanh": 4,
        "swish": 2
      },
      "note": "Figure 3: the effective-Hessian spectrum has negative eigenvalues under smooth (tanh/Swish) heads but not under linear/ReLU heads (evaluated in the negative-curvature regime)."
    },
    "C4_sec6_generality": {
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

````
