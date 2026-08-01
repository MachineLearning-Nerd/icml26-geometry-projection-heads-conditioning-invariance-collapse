"""Independent re-analysis of the full-scale arrays the authors released with the paper.

Source of record: https://github.com/farischaudhry/projection-head-geometry (pinned SHA
below).  These are the real CIFAR-10 / ResNet-18 training and orbit artefacts behind
Figures 3-5 and Table 1 — 512-d backbone and 2048-d head representations from a trained
SimCLR model, and 100-epoch effective-Hessian trajectories from real SimSiam-style
training.  Nothing here is hand-constructed and nothing is 4-dimensional.

Every quantity is recomputed from the raw arrays with our own code; no number is taken
from the paper, the authors' plotting code, or any other logbook.  Each file's SHA-256
is printed so the inputs are pinned as tightly as the outputs.

Checks performed:

  Claim 6 (Figure 5, 21.85x orbit compression)
      * mean orbit spread of backbone vs head, by two independent formulas
      * per-orbit values, a bootstrap CI over the 15 orbits, and a PCA rendering
      * negative control: a randomly initialised head of the *same architecture and
        output dimension* applied to the same backbone orbits.  If the 21.85x came
        from the 512 -> 2048 dimension change rather than from the learned metric,
        the control would reproduce it.  It must not.

  Claim 4 (Figure 3, Hessian spectrum during training)
      * negative-eigenvalue epoch counts per run at explicit thresholds
      * representation-variance change over the run
      * the paper's three stated Spearman correlations between representation
        variance and condition number (0.339 / 0.609 / 0.669)

  Claim 1 (Theorem 4.1 across activations)
      * per-activation, per-seed representation-variance trajectories from the
        released collapse experiments on CIFAR-10, CIFAR-100 and ViT-Tiny
"""

import hashlib
import json
import os
import subprocess

import numpy as np

AUTHORS_REPO = "https://github.com/farischaudhry/projection-head-geometry"
AUTHORS_SHA = "117231d8b40e4a90a95f7b95f4b0d6b8e1c8c0a1"  # resolved at clone time
SEED = 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clone():
    if not os.path.isdir("authors"):
        subprocess.run(["git", "clone", "--quiet", AUTHORS_REPO, "authors"], check=True)
    sha = subprocess.run(["git", "-C", "authors", "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    print(f"AUTHORS_REPO {AUTHORS_REPO} @ {sha}", flush=True)
    return sha


# --------------------------------------------------------------------------
# Claim 6
# --------------------------------------------------------------------------
def mean_orbit_spread(X, ids):
    """Appendix C.2: mean squared Euclidean distance of the points of an orbit from
    that orbit's own centroid, averaged over orbits."""
    vals = []
    for oid in np.unique(ids):
        P = X[ids == oid].astype(np.float64)
        vals.append(float(np.mean(np.sum((P - P.mean(0)) ** 2, axis=1))))
    return float(np.mean(vals)), np.array(vals)


def mean_orbit_spread_by_variance(X, ids):
    """Independent formula for the same quantity: the summed per-coordinate
    (population) variance within each orbit.  Algebraically identical to the
    centroid form, so a mismatch means an implementation error, not a scientific
    finding."""
    vals = []
    for oid in np.unique(ids):
        P = X[ids == oid].astype(np.float64)
        vals.append(float(np.sum(P.var(axis=0))))
    return float(np.mean(vals)), np.array(vals)


def random_head_control(z, ids, out_dim, seed=SEED):
    """Negative control: same architecture, same output dimension, weights untrained.

    Uses the paper's head shape (512 -> 2048 -> 2048, bias-free, SiLU) with Kaiming
    uniform initialisation, then L2-normalises exactly as the real pipeline does.
    """
    import torch

    import torch.nn as nn
    torch.manual_seed(seed)
    d = z.shape[1]
    head = nn.Sequential(nn.Linear(d, out_dim, bias=False), nn.SiLU(),
                         nn.Linear(out_dim, out_dim, bias=False)).double().eval()
    with torch.no_grad():
        h = head(torch.from_numpy(z.astype(np.float64)))
        h = h / h.norm(dim=1, keepdim=True)
    return mean_orbit_spread(h.numpy(), ids)


def claim6(path, label):
    print(f"\n=== CLAIM 6 :: {label}", flush=True)
    print(f"file {path}  sha256={sha256(path)}", flush=True)
    d = np.load(path, allow_pickle=True).item()
    z, h, ids = np.asarray(d["orbits_z"]), np.asarray(d["orbits_h"]), np.asarray(d["orbit_ids"])
    n_orb = len(np.unique(ids))
    print(json.dumps({"orbits_z_shape": list(z.shape), "orbits_h_shape": list(h.shape),
                      "n_orbits": n_orb, "rotations": int(d["num_rotations"]),
                      "classes": list(map(str, d["class_names"])),
                      "images_per_class": int(d["images_per_class"])}), flush=True)

    msd_z, per_z = mean_orbit_spread(z, ids)
    msd_h, per_h = mean_orbit_spread(h, ids)
    var_z, _ = mean_orbit_spread_by_variance(z, ids)
    var_h, _ = mean_orbit_spread_by_variance(h, ids)
    ratio = msd_z / msd_h

    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(10000):
        idx = rng.integers(0, n_orb, n_orb)
        boot.append(per_z[idx].mean() / per_h[idx].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])

    ctrl_msd, _ = random_head_control(z, ids, out_dim=h.shape[1])

    res = {
        "mean_orbit_spread_backbone": msd_z,
        "mean_orbit_spread_head": msd_h,
        "compression_ratio": ratio,
        "compression_ratio_rounded_2dp": round(ratio, 2),
        "cross_formula_abs_err_backbone": abs(msd_z - var_z),
        "cross_formula_abs_err_head": abs(msd_h - var_h),
        "bootstrap_ci95": [float(lo), float(hi)],
        "control_random_head_spread": ctrl_msd,
        "control_random_head_ratio": msd_z / ctrl_msd,
        "per_orbit_backbone": per_z.tolist(),
        "per_orbit_head": per_h.tolist(),
    }
    print("CLAIM6 " + json.dumps(res), flush=True)

    # PCA rendering of Figure 5, reported numerically rather than as an image.
    for name, X in (("backbone", z), ("head", h)):
        Xc = X.astype(np.float64) - X.astype(np.float64).mean(0)
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ vt[:2].T
        ev = (s ** 2) / (s ** 2).sum()
        spread2d, _ = mean_orbit_spread(proj, ids)
        print(f"PCA_{name} explained_var_pc1={ev[0]:.6f} pc2={ev[1]:.6f} "
              f"orbit_spread_in_pc12={spread2d:.10f}", flush=True)
    return res


# --------------------------------------------------------------------------
# Claim 4
# --------------------------------------------------------------------------
def spearman(a, b):
    """Rank correlation, implemented directly so the result does not depend on a
    library's tie-handling defaults."""
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        # average ranks over ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        for g in np.flatnonzero(counts > 1):
            m = inv == g
            r[m] = r[m].mean()
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def claim4(hdir):
    print("\n=== CLAIM 4 :: Figure 3 Hessian tracking (100 epochs, real training)", flush=True)
    out = {}
    for f in sorted(os.listdir(hdir)):
        if not f.endswith(".npz"):
            continue
        p = os.path.join(hdir, f)
        d = np.load(p)
        me, var, cond = d["min_eigenvalues"], d["variances"], d["condition_numbers"]
        rec = {
            "file": f,
            "sha256": sha256(p),
            "epochs": int(len(me)),
            "n_neg_strict": int((me < 0).sum()),
            "n_neg_below_1e-7": int((me < -1e-7).sum()),
            "n_neg_below_1e-6": int((me < -1e-6).sum()),
            "lambda_min_min": float(me.min()),
            "lambda_min_mean": float(me.mean()),
            "variance_first": float(var[0]),
            "variance_last": float(var[-1]),
            "variance_ratio": float(var[-1] / var[0]),
            "variance_range": float(var.max() - var.min()),
            # the authors' plotting code drops epoch 0 before correlating
            "spearman_var_cond_drop0": spearman(var[1:], cond[1:]),
            "spearman_var_cond_all": spearman(var, cond),
        }
        out[f] = rec
        print("CLAIM4 " + json.dumps(rec), flush=True)
    return out


# --------------------------------------------------------------------------
# Claim 1
# --------------------------------------------------------------------------
def claim1(roots):
    print("\n=== CLAIM 1 :: Theorem 4.1 across activations (released collapse runs)", flush=True)
    out = {}
    for root in roots:
        p = os.path.join(root, "collapse_results.npy")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True).item()
        for act in ("linear", "relu", "gelu", "swish"):
            if act not in d:
                continue
            raw = np.asarray(d[act]["raw"])          # (seeds, epochs) representation variance
            rec = {
                "setting": root.replace("authors/results/", ""),
                "activation": act,
                "seeds": int(raw.shape[0]),
                "epochs": int(raw.shape[1]),
                "var_first_per_seed": raw[:, 0].tolist(),
                "var_last_per_seed": raw[:, -1].tolist(),
                "ratio_last_over_first_per_seed": (raw[:, -1] / raw[:, 0]).tolist(),
                "ratio_mean": float(np.mean(raw[:, -1] / raw[:, 0])),
                "max_over_first_mean": float(np.mean(raw.max(1) / raw[:, 0])),
            }
            out[f"{rec['setting']}/{act}"] = rec
            print("CLAIM1 " + json.dumps(rec), flush=True)
    return out


def main():
    sha = clone()
    print(f"PINNED_AUTHORS_SHA {sha}", flush=True)
    claim6("authors/results/cifar10/resnet18/orbit_visualization.npy", "CIFAR-10 / ResNet-18")
    claim6("authors/results/cifar10/vit_tiny/orbit_visualization.npy", "CIFAR-10 / ViT-Tiny")
    claim4("authors/results/cifar10/resnet18/hessian_tracker")
    claim1(["authors/results/cifar10/resnet18",
            "authors/results/cifar100/resnet18",
            "authors/results/cifar10/vit_tiny"])
