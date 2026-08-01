"""Claim 6 — independent full-scale reproduction of the Figure 5 / Table 1 geometry.

Pretrains SimCLR on CIFAR-10 with a ResNet-18 backbone exactly as Appendix C describes
(NT-Xent, temperature 0.1, Adam lr 1e-3, batch 256, crop/flip/rotation/colour-jitter
augmentations, 2-layer 512 -> 2048 -> 2048 ReLU head with BatchNorm), then measures the
augmentation-orbit geometry of the trained network: 5 classes x 3 images x 12 rotations
spanning [0, 360), L2-normalised representations, backbone z (512-d) against head h(z)
(2048-d).

Orbit geometry is evaluated at several intermediate epochs, not only at the end, so a
run that is cut short by a job timeout still yields usable evidence and so the
trajectory of the compression ratio is visible rather than a single endpoint.

Negative control: the identical measurement on a randomly initialised head of the same
architecture and the same 2048-d output.  The paper argues the compression is "not
merely a byproduct of dimensionality reduction"; the control is what makes that
argument falsifiable here.
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from repro import data
from repro.models import ProjectionHead, ResNetBackbone

ROT_SWEEP_DEG = list(range(0, 50, 5))  # Appendix C.3: 0 to 45 degrees in 5-degree steps


class TwoCropTransform:
    def __init__(self, base):
        self.base = base

    def __call__(self, x):
        return [self.base(x), self.base(x)]


def simclr_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.ToTensor(),
    ])


def nt_xent(z1, z2, temperature):
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return F.cross_entropy(logits, labels)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def mean_orbit_spread(X, ids):
    vals = [float(np.mean(np.sum((X[ids == o] - X[ids == o].mean(0)) ** 2, axis=1)))
            for o in np.unique(ids)]
    return float(np.mean(vals)), np.array(vals)


def intra_orbit_distance(X, ids):
    vals = []
    for o in np.unique(ids):
        P = X[ids == o]
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        iu = np.triu_indices(len(P), k=1)
        vals.append(float(D[iu].mean()))
    return float(np.mean(vals)), np.array(vals)


def inter_class_distance(X, labels):
    cents = np.stack([X[labels == c].mean(0) for c in np.unique(labels)])
    D = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)
    iu = np.triu_indices(len(cents), k=1)
    return float(D[iu].mean())


def local_curvature(traj):
    """Appendix C.2: mean norm of the second central difference along the trajectory."""
    d = traj[2:] - 2 * traj[1:-1] + traj[:-2]
    return float(np.linalg.norm(d, axis=1).mean())


@torch.no_grad()
def collect_orbits(backbone, head, testset, num_classes, images_per_class,
                   num_rotations, seed):
    backbone.eval()
    head.eval()
    rng = np.random.default_rng(seed)
    targets = np.asarray(testset.targets)
    selected = np.linspace(0, 9, num_classes, dtype=int)
    angles = np.linspace(0, 360, num_rotations, endpoint=False)

    Z, H, L, I = [], [], [], []
    oid = 0
    for c in selected:
        idx = np.flatnonzero(targets == c)
        pick = rng.choice(idx, images_per_class, replace=False)
        for j in pick:
            img, _ = testset[int(j)]
            t = img.unsqueeze(0)
            for a in angles:
                rot = transforms.functional.rotate(t, float(a))
                z = F.normalize(backbone(rot), dim=1)
                h = F.normalize(head(z), dim=1)
                Z.append(z.numpy().ravel())
                H.append(h.numpy().ravel())
                L.append(int(c))
                I.append(oid)
            oid += 1
    return (np.asarray(Z, np.float64), np.asarray(H, np.float64),
            np.asarray(L), np.asarray(I))


@torch.no_grad()
def curvature_trajectories(backbone, head, testset, n_images, seed):
    backbone.eval()
    head.eval()
    rng = np.random.default_rng(seed + 999)
    picks = rng.choice(len(testset), n_images, replace=False)
    kz, kh = [], []
    for j in picks:
        img, _ = testset[int(j)]
        tz, th = [], []
        for a in ROT_SWEEP_DEG:
            rot = transforms.functional.rotate(img.unsqueeze(0), float(a))
            z = F.normalize(backbone(rot), dim=1)
            h = F.normalize(head(z), dim=1)
            tz.append(z.numpy().ravel())
            th.append(h.numpy().ravel())
        kz.append(local_curvature(np.asarray(tz)))
        kh.append(local_curvature(np.asarray(th)))
    return float(np.mean(kz)), float(np.std(kz)), float(np.mean(kh)), float(np.std(kh))


def table1(Z, H, L, I, tag):
    msd_z, per_z = mean_orbit_spread(Z, I)
    msd_h, per_h = mean_orbit_spread(H, I)
    di_z, _ = intra_orbit_distance(Z, I)
    di_h, _ = intra_orbit_distance(H, I)
    dc_z = inter_class_distance(Z, L)
    dc_h = inter_class_distance(H, L)
    rec = {
        "tag": tag,
        "mean_orbit_spread_backbone": msd_z,
        "mean_orbit_spread_head": msd_h,
        "orbit_compression_ratio": msd_z / msd_h,
        "D_intra_backbone": di_z,
        "D_intra_head": di_h,
        "D_intra_reduction": di_z / di_h,
        "D_inter_backbone": dc_z,
        "D_inter_head": dc_h,
        "D_inter_reduction": dc_z / dc_h,
        "class_orbit_ratio_backbone": dc_z / di_z,
        "class_orbit_ratio_head": dc_h / di_h,
        "class_orbit_separation": (dc_h / di_h) / (dc_z / di_z),
        "per_orbit_backbone": per_z.tolist(),
        "per_orbit_head": per_h.tolist(),
    }
    print("TABLE1 " + json.dumps(rec), flush=True)
    return rec


def evaluate(backbone, head, testset, epoch, cfg):
    Z, H, L, I = collect_orbits(backbone, head, testset, cfg["num_classes"],
                                cfg["images_per_class"], cfg["num_rotations"], cfg["seed"])
    rec = table1(Z, H, L, I, f"epoch{epoch}_trained_head")

    torch.manual_seed(cfg["seed"] + 12345)
    control = ProjectionHead(512, 2048, 2048, "relu", use_bn=True).eval()
    with torch.no_grad():
        Hc = F.normalize(control(torch.from_numpy(Z).float()), dim=1).numpy().astype(np.float64)
    ctrl_msd, _ = mean_orbit_spread(Hc, I)
    ctrl = {"tag": f"epoch{epoch}_control_random_head",
            "mean_orbit_spread_control_head": ctrl_msd,
            "control_compression_ratio": rec["mean_orbit_spread_backbone"] / ctrl_msd}
    print("CONTROL " + json.dumps(ctrl), flush=True)

    cz_m, cz_s, ch_m, ch_s = curvature_trajectories(backbone, head, testset, 15, cfg["seed"])
    curv = {"tag": f"epoch{epoch}", "curvature_backbone_mean": cz_m,
            "curvature_backbone_std": cz_s, "curvature_head_mean": ch_m,
            "curvature_head_std": ch_s, "curvature_ratio": ch_m / cz_m}
    print("CURVATURE " + json.dumps(curv), flush=True)
    return rec, ctrl, curv


def main(epochs=50, batch_size=256, lr=1e-3, temperature=0.1, seed=0,
         num_classes=5, images_per_class=3, num_rotations=12, eval_every=5):
    cfg = dict(epochs=epochs, batch_size=batch_size, lr=lr, temperature=temperature,
               seed=seed, num_classes=num_classes, images_per_class=images_per_class,
               num_rotations=num_rotations, optimiser="adam",
               head="512-2048-2048 relu + BN", backbone="resnet18_cifar",
               augmentations="RRC(32), HFlip, Rot(45), ColorJitter(.4,.4,.4,.1) p=.8")
    print("CONFIG " + json.dumps(cfg), flush=True)
    t0 = time.perf_counter()

    root = data.stage()
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                         transform=TwoCropTransform(simclr_transform()))
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True,
                                        transform=transforms.ToTensor())
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=2, generator=g)

    backbone = ResNetBackbone()
    head = ProjectionHead(512, 2048, 2048, "relu", use_bn=True)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)

    print("--- geometry at initialisation (untrained: no learned metric yet) ---", flush=True)
    evaluate(backbone, head, test, 0, cfg)

    for epoch in range(1, epochs + 1):
        backbone.train()
        head.train()
        losses = []
        for (x1, x2), _ in loader:
            z1 = F.normalize(head(backbone(x1)), dim=1)
            z2 = F.normalize(head(backbone(x2)), dim=1)
            loss = nt_xent(z1, z2, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        print(f"EPOCH {json.dumps({'epoch': epoch, 'nt_xent': float(np.mean(losses)), 'elapsed_s': round(time.perf_counter() - t0, 1)})}", flush=True)
        if epoch % eval_every == 0 or epoch == epochs:
            evaluate(backbone, head, test, epoch, cfg)
    print(f"ORBITS DONE in {time.perf_counter() - t0:.1f}s", flush=True)
