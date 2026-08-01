# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=2.0",
#   "torch>=2.5",
#   "torchvision>=0.20",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# ///
"""Stage 0 — cpu-upgrade calibration + audit of the authors' released artifacts.

Two jobs in one run, both cheap:

1. Report the container's real CPU quota and measure ResNet-18/CIFAR forward+backward
   throughput, so the cost of the full-scale training rounds can be forecast instead
   of guessed.
2. Load every raw array released with the paper (github.com/farischaudhry/projection-head-geometry,
   pinned SHA) and print its exact structure, plus a first pass at the Figure 5 orbit
   compression ratio and the Figure 3 lambda_min trajectories.

Everything is printed to stdout: job filesystems are discarded when the job ends.
"""

import os

# --- thread pinning: must precede numpy/torch imports -----------------------
def _cgroup_quota() -> int:
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(float(quota) / float(period)))
    except OSError:
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // p)
    except OSError:
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


NUM_THREADS = _cgroup_quota()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(NUM_THREADS)

import json
import platform
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(1)

AUTHORS_REPO = "https://github.com/farischaudhry/projection-head-geometry"
AUTHORS_SHA = "117231d"


def banner(t):
    print("\n" + "=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def environment():
    banner("STAGE0.1  ENVIRONMENT")
    info = {
        "cgroup_cpu_quota": NUM_THREADS,
        "os_cpu_count": os.cpu_count(),
        "sched_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        mem = int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1])
        info["mem_total_gb"] = round(mem / 1024 / 1024, 1)
    except Exception:
        pass
    try:
        model = {l.split(":")[0].strip(): l.split(":", 1)[1].strip()
                 for l in open("/proc/cpuinfo") if ":" in l}.get("model name")
        info["cpu_model"] = model
    except Exception:
        pass
    print(json.dumps(info, indent=2), flush=True)
    return info


# --- the paper's architecture, reimplemented from the paper's description ----
class ResNetBackbone(nn.Module):
    """ResNet-18 adapted to 32x32 CIFAR inputs (3x3 stem, no maxpool), 512-d output."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=2048,
                 activation="relu", use_bn=False):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "swish": nn.SiLU,
               "linear": nn.Identity}[activation]
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def throughput():
    """Measure fwd+bwd throughput so full-scale training cost can be forecast."""
    banner("STAGE0.2  THROUGHPUT CALIBRATION (ResNet-18 @ 32x32, SimSiam-style step)")
    torch.manual_seed(0)
    backbone = ResNetBackbone()
    projector = ProjectionHead(512, 2048, 2048, "swish", use_bn=False)
    predictor = ProjectionHead(2048, 512, 2048, "swish", use_bn=False)
    params = list(backbone.parameters()) + list(projector.parameters()) + list(predictor.parameters())
    opt = torch.optim.SGD(params, lr=0.05, momentum=0.9)
    n_params = sum(p.numel() for p in params)
    print(f"trainable parameters: {n_params:,}", flush=True)

    out = {}
    for bs in (64, 256):
        x1 = torch.randn(bs, 3, 32, 32)
        x2 = torch.randn(bs, 3, 32, 32)
        # one warm-up step (oneDNN primitive caching)
        for it in range(4):
            if it == 1:
                t0 = time.perf_counter()
            z1, z2 = backbone(x1), backbone(x2)
            p1, p2 = projector(z1), projector(z2)
            h1, h2 = predictor(p1), predictor(p2)
            loss = -0.5 * (F.cosine_similarity(h1, p2.detach()).mean()
                           + F.cosine_similarity(h2, p1.detach()).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        dt = (time.perf_counter() - t0) / 3
        imgs_per_s = 2 * bs / dt
        out[bs] = {"sec_per_step": round(dt, 4), "images_per_s": round(imgs_per_s, 1)}
        print(f"batch {bs:4d}: {dt:7.3f} s/step  ->  {imgs_per_s:8.1f} img/s "
              f"({50000 * 2 / imgs_per_s / 60:.1f} min per CIFAR-10 epoch, 2 views)", flush=True)

    # Hessian-vector-product cost: 20 power iterations on d(loss)/dz
    bs = 256
    x1 = torch.randn(bs, 3, 32, 32)
    z1 = backbone(x1).detach().requires_grad_(True)
    p1 = projector(z1)
    h1 = predictor(p1)
    loss = -F.cosine_similarity(h1, p1.detach()).mean()
    t0 = time.perf_counter()
    g = torch.autograd.grad(loss, z1, create_graph=True)[0]
    v = torch.randn_like(z1)
    for _ in range(20):
        Hv = torch.autograd.grad(g, z1, grad_outputs=v, retain_graph=True)[0]
        v = Hv / (Hv.norm() + 1e-8)
    dt = time.perf_counter() - t0
    out["hvp_20iter_sec"] = round(dt, 3)
    print(f"20 HVP power iterations (batch 256): {dt:.3f} s", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


def released_artifacts():
    banner("STAGE0.3  AUTHORS' RELEASED RAW ARTIFACTS")
    subprocess.run(["git", "clone", "--quiet", AUTHORS_REPO, "authors"], check=True)
    sha = subprocess.run(["git", "-C", "authors", "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    print(f"authors repo HEAD = {sha}", flush=True)

    for root, _dirs, files in os.walk("authors/results"):
        for f in sorted(files):
            if not f.endswith((".npy", ".npz")):
                continue
            path = os.path.join(root, f)
            print(f"\n--- {path}  ({os.path.getsize(path):,} bytes)", flush=True)
            obj = np.load(path, allow_pickle=True)
            if path.endswith(".npz"):
                for k in obj.files:
                    a = obj[k]
                    print(f"    [{k}] shape={a.shape} dtype={a.dtype}", flush=True)
            else:
                a = obj
                if a.dtype == object and a.shape == ():
                    d = a.item()
                    print(f"    dict with keys: {list(d)[:40]}", flush=True)
                    describe(d, indent=6)
                else:
                    print(f"    array shape={a.shape} dtype={a.dtype}", flush=True)


def describe(d, indent=4, depth=0):
    pad = " " * indent
    if depth > 2:
        return
    for k, v in list(d.items())[:40]:
        if isinstance(v, dict):
            print(f"{pad}{k}: dict({list(v)[:12]})", flush=True)
            describe(v, indent + 2, depth + 1)
        elif isinstance(v, np.ndarray):
            print(f"{pad}{k}: ndarray shape={v.shape} dtype={v.dtype}", flush=True)
        elif isinstance(v, (list, tuple)):
            print(f"{pad}{k}: {type(v).__name__}(len={len(v)}) head={str(v[:4])[:120]}", flush=True)
        else:
            print(f"{pad}{k}: {type(v).__name__} = {str(v)[:120]}", flush=True)


def first_pass_claims():
    banner("STAGE0.4  FIRST PASS — Figure 5 orbit compression, Figure 3 lambda_min")
    orb_path = "authors/results/cifar10/resnet18/orbit_visualization.npy"
    if os.path.exists(orb_path):
        d = np.load(orb_path, allow_pickle=True).item()
        z, h = np.asarray(d["orbits_z"]), np.asarray(d["orbits_h"])
        ids = np.asarray(d["orbit_ids"])
        print(f"orbits_z {z.shape}  orbits_h {h.shape}  n_orbits={len(np.unique(ids))}", flush=True)

        def mean_orbit_spread(X):
            vals = []
            for oid in np.unique(ids):
                P = X[ids == oid].astype(np.float64)
                c = P.mean(axis=0)
                vals.append(float(np.mean(np.sum((P - c) ** 2, axis=1))))
            return float(np.mean(vals)), vals

        msd_z, per_z = mean_orbit_spread(z)
        msd_h, per_h = mean_orbit_spread(h)
        print(f"mean orbit spread  backbone = {msd_z:.10f}", flush=True)
        print(f"mean orbit spread  head     = {msd_h:.10f}", flush=True)
        print(f"COMPRESSION RATIO           = {msd_z / msd_h:.8f}x   (paper: 21.85x)", flush=True)
        print(f"per-orbit backbone: {[round(v, 6) for v in per_z]}", flush=True)
        print(f"per-orbit head:     {[round(v, 8) for v in per_h]}", flush=True)

    hdir = "authors/results/cifar10/resnet18/hessian_tracker"
    for f in sorted(os.listdir(hdir)) if os.path.isdir(hdir) else []:
        if not f.endswith(".npz"):
            continue
        d = np.load(os.path.join(hdir, f))
        me = d["min_eigenvalues"]
        var = d["variances"]
        print(f"\n{f}: epochs={len(me)}  neg_eig_epochs={int((me < 0).sum())}  "
              f"min={me.min():.6g}  var[0]={var[0]:.6g} var[-1]={var[-1]:.6g} "
              f"ratio={var[-1] / var[0]:.6f}", flush=True)


if __name__ == "__main__":
    t_start = time.perf_counter()
    environment()
    throughput()
    released_artifacts()
    first_pass_claims()
    banner(f"STAGE0 COMPLETE in {time.perf_counter() - t_start:.1f} s")
