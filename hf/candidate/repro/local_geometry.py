"""Claims 2, 3 and 5 (part A) on real SSL loss landscapes, as a standalone node.

Identical computation to the corresponding section of `repro/pretrained.py`, lifted out
so it does not sit behind four checkpoint downloads and a full orbit sweep.  That
ordering is why it never executed: both earlier runs were terminated before reaching it.

Only the checkpoints whose heads are narrow enough for a dense loss Hessian are loaded
(Barlow Twins and VICReg, 8192-d).  DINO's released heads emit 60,000 and 65,536
prototype logits, where a dense k x k Hessian is ~29 GB.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import threads  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
import torchvision.transforms as T  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402

threads.pin_torch(torch)

from repro import data, geometry, pretrained  # noqa: E402

SEED = 0
K_GEOM = 512
HEADS = ("barlowtwins_resnet50", "vicreg_resnet50")


def load_narrow_heads(workdir="pretrained"):
    pretrained._allow_vicreg_unpickle()
    os.makedirs(workdir, exist_ok=True)
    out = {}
    for name in HEADS:
        dest = os.path.join(workdir, name + ".pth")
        try:
            pretrained.fetch(pretrained.CKPTS[name], dest)
            print(f"CKPT {name} sha256={pretrained.sha256(dest)}", flush=True)
            b, h = pretrained.load_projector(dest, "model", "projector.")
            b.eval()
            h.eval()
            with torch.no_grad():
                z = b(torch.randn(2, 3, 224, 224))
                hz = h(z)
            print(f"CKPT_READY {name} z={tuple(z.shape)} h={tuple(hz.shape)}", flush=True)
            out[name] = (b, h)
        except Exception as exc:
            print(f"CKPT_FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
    return out


def real_head_outputs(b, h, testset, n=64, seed=SEED):
    """A real pair of head-output batches: clean views and a rotated positive view."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(testset), n, replace=False)
    with torch.no_grad():
        X1 = torch.stack([pretrained.NORM(testset[int(j)][0]) for j in picks])
        X2 = torch.stack([pretrained.NORM(TF.rotate(testset[int(j)][0], 15.0))
                          for j in picks])
        R1 = h(b(X1)).double()
        R2 = h(b(X2)).double()
    mu = R1.mean(0, keepdim=True)
    _, S, Vt = torch.linalg.svd(R1 - mu, full_matrices=False)
    m = min(K_GEOM, Vt.shape[0])
    P = Vt[:m].T
    kept = float((S[:m] ** 2).sum() / (S ** 2).sum())
    return (R1 - mu) @ P, (R2 - mu) @ P, m, kept, tuple(R1.shape)


def main():
    t0 = time.perf_counter()
    print("PROVENANCE " + json.dumps({
        "stage": "local_geometry", "cpu_threads": threads.NUM_THREADS,
        "os_cpu_count": os.cpu_count(), "torch": torch.__version__,
        "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
        "compute": "local CPU (documented deviation from the cpu-upgrade default)",
    }), flush=True)
    assert not torch.cuda.is_available(), "GPU detected; this campaign is CPU-only"

    # Symbolic first now that it is cheap: it depends on nothing below, and it is the
    # part that discharges Theorem 3.1's universal quantifier.
    print("=== Theorem 3.1: symbolic discharge of the universal quantifier ===", flush=True)
    print("THM31_SYMBOLIC " + json.dumps(geometry.theorem_31_symbolic_certificate()),
          flush=True)
    print(f"T+ {time.perf_counter() - t0:.1f}s", flush=True)

    root = data.stage()
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True,
        transform=T.Compose([T.Resize(224), T.ToTensor()]))
    print(f"DATA cifar10 test n={len(testset)}  T+ {time.perf_counter() - t0:.1f}s",
          flush=True)

    models = load_narrow_heads()
    print(f"MODELS_LOADED {sorted(models)}  T+ {time.perf_counter() - t0:.1f}s", flush=True)

    for name, (b, h) in models.items():
        H1, H2, m, kept, raw = real_head_outputs(b, h, testset)
        print(f"REPS {name} raw={raw} reduced={tuple(H1.shape)} "
              f"variance_retained={kept:.6f}", flush=True)
        pretrained.loss_geometry(H1, H2, d=m, tag=name)     # -> THM31 (8 objectives)
        print(f"T+ {time.perf_counter() - t0:.1f}s after loss_geometry {name}", flush=True)
        pretrained.prop33(H1, H2, m=6, tag=name)            # -> PROP33 + control
        print(f"T+ {time.perf_counter() - t0:.1f}s after prop33 {name}", flush=True)

    print(f"LOCAL_GEOMETRY COMPLETE in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
