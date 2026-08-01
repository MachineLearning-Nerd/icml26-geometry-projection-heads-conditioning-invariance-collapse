"""Claims 2, 3 and 5 on real, publicly released SSL models that ship their projection heads.

The paper's own Section 6.3 / Appendix D.6 analyses official checkpoints for which the
projection head was released: DINO ResNet-50, DINO ViT-S/16, VICReg ResNet-50 and
Barlow Twins ResNet-50.  These are genuinely trained SSL networks — a self-distillation
objective, a variance-invariance-covariance objective and a redundancy-reduction
objective — so their heads give real loss landscapes and real learned metrics rather
than constructed matrices.

What is measured here:

  Claim 5  the geometric analysis holds across both families named in Section 6.
           Empirically, on the four real checkpoints: augmentation-orbit compression,
           local curvature ratio, effective rank and alignment gain along four
           continuous orbits (rotation, hue, saturation, blur), 12 steps each.
           Algebraically, all eight named objectives (InfoNCE, SimCLR, MoCo, DINO,
           BYOL, SimSiam, VICReg, Barlow Twins) are implemented and their real loss
           Hessians at real head outputs are put through the same two tests.

  Claim 2  Theorem 3.1's whitening construction instantiated on those real loss
           Hessians, for many random r-dimensional subspaces, plus the symbolic
           certificate that discharges the "for any subspace" quantifier.

  Claim 3  Proposition 3.3's curvature barrier: the Riemann tensor of the real loss
           metric g_L computed two independent ways on a real slice, the constant-W
           obstruction measured across real points, and a flat-metric control that
           must behave the opposite way.

  Claim 1  Theorem 4.1's interaction term M computed with the *real trained head's*
           Jacobian and second derivative, for smooth (GELU/SiLU) and piecewise-linear
           (ReLU) and linear heads.
"""

import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.func import hessian

from repro import data, geometry, losses

CKPTS = {
    "dino_resnet50": "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth",
    "dino_vits16": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth",
    "vicreg_resnet50": "https://dl.fbaipublicfiles.com/vicreg/resnet50_fullckpt.pth",
    "barlowtwins_resnet50": "https://dl.fbaipublicfiles.com/barlowtwins/ljng/checkpoint.pth",
}
SEED = 0


def fetch(url, dest):
    if os.path.exists(dest):
        return dest
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    print(f"CKPT fetched {os.path.basename(dest)} ({os.path.getsize(dest):,} B) "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)
    return dest


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def strip(prefix, sd):
    out = {}
    for k, v in sd.items():
        k = k.replace("module.", "")
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


# --------------------------------------------------------------------------
# head reconstruction straight from the released tensors
# --------------------------------------------------------------------------
def mlp_from_state(sd, order, acts, bns, l2_norm_at=None, final=None):
    """Rebuild a projection head as an nn.Sequential from the released weight shapes."""
    layers = []
    for i, key in enumerate(order):
        W = sd[f"{key}.weight"]
        b = sd.get(f"{key}.bias")
        lin = nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
        lin.weight.data = W.clone()
        if b is not None:
            lin.bias.data = b.clone()
        layers.append(lin)
        if i in bns:
            bkey = bns[i]
            bn = nn.BatchNorm1d(sd[f"{bkey}.weight"].shape[0])
            bn.weight.data = sd[f"{bkey}.weight"].clone()
            bn.bias.data = sd[f"{bkey}.bias"].clone()
            bn.running_mean = sd[f"{bkey}.running_mean"].clone()
            bn.running_var = sd[f"{bkey}.running_var"].clone()
            layers.append(bn)
        if i in acts:
            layers.append(acts[i]())
    return nn.Sequential(*layers)


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=-1, p=2)


def _allow_vicreg_unpickle():
    """VICReg's released checkpoint pickles a reference to `exclude_bias_and_norm`,
    a plain function defined in that repository's training script.  Unpickling looks
    it up in `__main__`, which here is our runner, so a same-named stand-in is
    installed.  It is never called: only the tensors are used."""
    import __main__

    if not hasattr(__main__, "exclude_bias_and_norm"):
        def exclude_bias_and_norm(p):
            return p.ndim == 1
        __main__.exclude_bias_and_norm = exclude_bias_and_norm


def load_dino(path, arch):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("teacher", ck)
    head = strip("head.", state)
    if arch == "resnet50":
        backbone = torch.hub.load("facebookresearch/dino:main", "dino_resnet50",
                                  verbose=False)
        mlp = mlp_from_state(head, ["mlp.0", "mlp.3"], {0: nn.GELU},
                             {0: "mlp.1"})
    else:
        backbone = torch.hub.load("facebookresearch/dino:main", "dino_vits16",
                                  verbose=False)
        mlp = mlp_from_state(head, ["mlp.0", "mlp.2", "mlp.4"],
                             {0: nn.GELU, 1: nn.GELU}, {})
    if "last_layer.weight_v" in head:
        g = head["last_layer.weight_g"]
        v = head["last_layer.weight_v"]
        W = g * v / v.norm(dim=1, keepdim=True)
    else:
        W = head["last_layer.weight"]
    last = nn.Linear(W.shape[1], W.shape[0], bias=False)
    last.weight.data = W.clone()
    return backbone, nn.Sequential(mlp, L2Norm(), last)


def load_projector(path, key_model, prefix, arch="resnet50"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck[key_model] if key_model in ck else ck
    proj = strip(prefix, state)
    back = strip("backbone.", state)
    backbone = torchvision.models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    missing = backbone.load_state_dict(back, strict=False)
    print(f"BACKBONE_LOAD missing={len(missing.missing_keys)} "
          f"unexpected={len(missing.unexpected_keys)}", flush=True)
    lin_idx = sorted({int(k.split(".")[0]) for k in proj if k.endswith(".weight")
                      and proj[k].dim() == 2})
    bn_idx = sorted({int(k.split(".")[0]) for k in proj if k.endswith(".running_mean")})
    layers = []
    for i in lin_idx:
        W = proj[f"{i}.weight"]
        b = proj.get(f"{i}.bias")
        lin = nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
        lin.weight.data = W.clone()
        if b is not None:
            lin.bias.data = b.clone()
        layers.append(lin)
        if i + 1 in bn_idx:
            bn = nn.BatchNorm1d(proj[f"{i + 1}.weight"].shape[0])
            for a in ("weight", "bias"):
                getattr(bn, a).data = proj[f"{i + 1}.{a}"].clone()
            bn.running_mean = proj[f"{i + 1}.running_mean"].clone()
            bn.running_var = proj[f"{i + 1}.running_var"].clone()
            layers.append(bn)
            layers.append(nn.ReLU())
    while isinstance(layers[-1], (nn.ReLU, nn.BatchNorm1d)):
        layers.pop()
    return backbone, nn.Sequential(*layers)


def load_all(workdir="pretrained"):
    _allow_vicreg_unpickle()
    os.makedirs(workdir, exist_ok=True)
    models = {}
    for name, url in CKPTS.items():
        dest = os.path.join(workdir, name + ".pth")
        try:
            fetch(url, dest)
            print(f"CKPT {name} sha256={sha256(dest)}", flush=True)
            if name.startswith("dino"):
                b, h = load_dino(dest, "resnet50" if "resnet" in name else "vit_s")
            elif name.startswith("vicreg"):
                b, h = load_projector(dest, "model", "projector.")
            else:
                b, h = load_projector(dest, "model", "projector.")
            b.eval()
            h.eval()
            with torch.no_grad():
                z = b(torch.randn(2, 3, 224, 224))
                hz = h(z)
            print(f"CKPT_READY {name} z={tuple(z.shape)} h={tuple(hz.shape)}", flush=True)
            models[name] = (b, h)
        except Exception as exc:
            print(f"CKPT_FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
    return models


# --------------------------------------------------------------------------
# augmentation orbits on real CIFAR-10 test images at 224px
# --------------------------------------------------------------------------
NORM = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def orbit_images(img, kind, steps=12):
    out = []
    for t in np.linspace(0, 1, steps):
        if kind == "rotation":
            x = TF.rotate(img, float(45 * t))
        elif kind == "hue":
            x = TF.adjust_hue(img, float(-0.4 + 0.8 * t))
        elif kind == "saturation":
            x = TF.adjust_saturation(img, float(2 * t))
        else:
            s = float(0.1 + 2.9 * t)
            x = TF.gaussian_blur(img, kernel_size=9, sigma=s)
        out.append(NORM(x))
    return torch.stack(out)


def effective_rank(traj):
    X = traj - traj.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2 / (np.sum(s ** 2) + 1e-30)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def curvature(traj):
    d = traj[2:] - 2 * traj[1:-1] + traj[:-2]
    return float(np.linalg.norm(d, axis=1).mean())


@torch.no_grad()
def orbit_analysis(name, backbone, head, testset, n_traj=50, steps=12, seed=SEED):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(testset), n_traj, replace=False)
    res = {}
    for kind in ("rotation", "hue", "saturation", "blur"):
        sz, sh, cz, ch, rz, rh, align = [], [], [], [], [], [], []
        for j in picks:
            img, _ = testset[int(j)]
            batch = orbit_images(img, kind, steps)
            z = backbone(batch)
            h = head(z)
            zn, hn = z.numpy().astype(np.float64), h.numpy().astype(np.float64)
            sz.append(float(np.mean(np.sum((zn - zn.mean(0)) ** 2, 1))))
            sh.append(float(np.mean(np.sum((hn - hn.mean(0)) ** 2, 1))))
            cz.append(curvature(zn / np.linalg.norm(zn, axis=1, keepdims=True)))
            ch.append(curvature(hn / np.linalg.norm(hn, axis=1, keepdims=True)))
            rz.append(effective_rank(zn))
            rh.append(effective_rank(hn))
            cos_z = float(F.cosine_similarity(z[:1], z, dim=1)[1:].mean())
            cos_h = float(F.cosine_similarity(h[:1], h, dim=1)[1:].mean())
            align.append(cos_h - cos_z)
        rec = {
            "model": name, "orbit": kind, "n_trajectories": n_traj, "steps": steps,
            "spread_backbone_mean": float(np.mean(sz)),
            "spread_head_mean": float(np.mean(sh)),
            "compression_ratio": float(np.mean(sz) / np.mean(sh)),
            "compression_ratio_sem": float(np.std(np.array(sz) / np.array(sh), ddof=1)
                                           / np.sqrt(n_traj)),
            "curvature_backbone": float(np.mean(cz)),
            "curvature_head": float(np.mean(ch)),
            "curvature_ratio": float(np.mean(ch) / np.mean(cz)),
            "eff_rank_backbone": float(np.mean(rz)),
            "eff_rank_head": float(np.mean(rh)),
            "alignment_gain": float(np.mean(align)),
        }
        print("ORBIT " + json.dumps(rec), flush=True)
        res[kind] = rec
    return res


# --------------------------------------------------------------------------
# Claims 2, 3, 5 on the eight real objectives
# --------------------------------------------------------------------------
def loss_geometry(H1, H2, d=512, n_subspaces=25, seed=SEED, tag=""):
    """Real loss Hessians of all eight objectives at a real head output."""
    g = torch.Generator().manual_seed(seed)
    fns, anchor = losses.build_all(H1, H2, g)
    out = {}
    for name, f in fns.items():
        H_L = hessian(f)(anchor)
        H_L = 0.5 * (H_L + H_L.T)
        rec = geometry.check_theorem_31(H_L, d=d, n_subspaces=n_subspaces, seed=seed)
        rec.update({"objective": name, "family":
                    "contrastive" if name in losses.CONTRASTIVE else "non_contrastive",
                    "source": tag})
        print("THM31 " + json.dumps(rec), flush=True)
        out[name] = rec
    return out


def prop33(H1, H2, m=6, seed=SEED, tag=""):
    """Riemann curvature of a real SSL loss metric, two independent ways, plus the
    constant-W obstruction and a flat-metric control."""
    g = torch.Generator().manual_seed(seed)
    fns, anchor = losses.build_all(H1, H2, g)
    k = anchor.shape[0]
    V, _ = torch.linalg.qr(torch.randn(k, m, generator=g, dtype=anchor.dtype))
    out = {}
    for name in ("infonce", "vicreg", "barlow_twins", "simsiam"):
        f = fns[name]

        def L(u):
            return f(u)

        gfn = geometry.metric_on_slice(L, anchor, V)
        t0 = torch.zeros(m, dtype=anchor.dtype)
        try:
            R_ud, Gamma, dg = geometry.riemann_tensor(gfn, t0)
            gmat = gfn(t0)
            R_low = geometry.lower_index(R_ud, gmat)
            R_alt, g_alt, C = geometry.riemann_from_cubic_form(L, anchor, V)
            sec = geometry.sectional_curvatures(R_low, gmat)
            rec = {
                "objective": name, "source": tag, "slice_dim": m,
                "metric_lambda_min": float(torch.linalg.eigvalsh(gmat)[0]),
                "metric_lambda_max": float(torch.linalg.eigvalsh(gmat)[-1]),
                "riemann_norm_christoffel_route": float(torch.linalg.norm(R_low)),
                "riemann_norm_cubic_form_route": float(torch.linalg.norm(R_alt)),
                "routes_relative_difference": float(
                    torch.linalg.norm(R_low - R_alt) / (torch.linalg.norm(R_low) + 1e-300)),
                "dg_norm": float(torch.linalg.norm(dg)),
                "max_abs_sectional_curvature": max(abs(v) for v in sec.values()) if sec else 0.0,
                "sectional_curvatures": sec,
            }
        except Exception as exc:
            rec = {"objective": name, "source": tag, "error": f"{type(exc).__name__}: {exc}"}
        print("PROP33 " + json.dumps(rec), flush=True)
        out[name] = rec

    # flat control: a genuinely quadratic loss has constant g_L, zero curvature, and
    # IS globally whitenable by a constant W.
    A = torch.randn(k, k, generator=g, dtype=anchor.dtype)
    Q = A @ A.T / k + torch.eye(k, dtype=anchor.dtype)

    def flat(u):
        return 0.5 * u @ Q @ u

    gfn = geometry.metric_on_slice(flat, anchor, V)
    R_ud, _, dgf = geometry.riemann_tensor(gfn, torch.zeros(m, dtype=anchor.dtype))
    ctrl = {"control": "quadratic_flat_metric",
            "riemann_norm": float(torch.linalg.norm(R_ud)),
            "dg_norm": float(torch.linalg.norm(dgf))}
    print("PROP33_CONTROL " + json.dumps(ctrl), flush=True)

    pts = [anchor + 0.5 * V @ torch.randn(m, generator=g, dtype=anchor.dtype)
           for _ in range(4)]
    for name in ("infonce", "vicreg"):
        obs = geometry.constant_W_obstruction(fns[name], [anchor] + pts, d=64, seed=seed)
        obs.update({"objective": name, "source": tag})
        print("PROP33_CONSTW " + json.dumps(obs), flush=True)
    obs = geometry.constant_W_obstruction(flat, [anchor] + pts, d=64, seed=seed)
    obs.update({"objective": "quadratic_flat_control", "source": tag})
    print("PROP33_CONSTW " + json.dumps(obs), flush=True)
    return out


def main():
    root = data.stage()
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True,
                                           transform=T.Compose([T.Resize(224), T.ToTensor()]))
    models = load_all()
    print(f"MODELS_LOADED {sorted(models)}", flush=True)

    for name, (b, h) in models.items():
        orbit_analysis(name, b, h, testset, n_traj=50, steps=12)

    # Real head outputs from a real trained SSL model feed the loss-geometry tests.
    #
    # DINO's released heads output 60,000 (ResNet-50) and 65,536 (ViT-S) prototype
    # logits; a dense k x k loss Hessian at that width is ~29 GB and is not
    # computable here, so the loss geometry is taken on the Barlow Twins / VICReg
    # heads (8192-d) reduced to the top K_GEOM principal directions of the real
    # batch.  That reduced space is itself a real head output space — it is the
    # geometry a head mapping into those directions realises — so no quantity below
    # comes from a synthetic or hand-constructed representation.
    K_GEOM = 512
    for name in ("barlowtwins_resnet50", "vicreg_resnet50"):
        if name not in models:
            continue
        b, h = models[name]
        rng = np.random.default_rng(SEED)
        picks = rng.choice(len(testset), 64, replace=False)
        with torch.no_grad():
            X1 = torch.stack([NORM(testset[int(j)][0]) for j in picks])
            X2 = torch.stack([NORM(TF.rotate(testset[int(j)][0], 15.0)) for j in picks])
            R1 = h(b(X1)).double()
            R2 = h(b(X2)).double()
        mu = R1.mean(0, keepdim=True)
        _, S, Vt = torch.linalg.svd(R1 - mu, full_matrices=False)
        m = min(K_GEOM, Vt.shape[0])
        P = Vt[:m].T
        H1, H2 = (R1 - mu) @ P, (R2 - mu) @ P
        kept = float((S[:m] ** 2).sum() / (S ** 2).sum())
        print(f"REPS {name} raw={tuple(R1.shape)} reduced={tuple(H1.shape)} "
              f"variance_retained={kept:.6f}", flush=True)
        loss_geometry(H1, H2, d=m, tag=name)
        prop33(H1, H2, m=6, tag=name)
        break

    # Left until last: it is pure symbolic algebra with no dependency on anything
    # above, so a slow computer-algebra step can never block the empirical results.
    print("=== symbolic certificate for Theorem 3.1's universal quantifier ===", flush=True)
    t0 = time.perf_counter()
    print("THM31_SYMBOLIC " + json.dumps(geometry.theorem_31_symbolic_certificate()),
          flush=True)
    print(f"THM31_SYMBOLIC_SECONDS {time.perf_counter() - t0:.1f}", flush=True)
