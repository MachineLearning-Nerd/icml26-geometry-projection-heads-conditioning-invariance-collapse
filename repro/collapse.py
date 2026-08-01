"""Claims 1 and 4 — the effective-Hessian spectrum during real SSL training.

Reproduces the paper's Figure 3 / Section 6.1 experiment: SimSiam-style training of a
ResNet-18 on CIFAR-10 from a pseudo-collapsed (and, as a contrast, a standard)
initialisation with BatchNorm removed, tracking the effective Hessian of the objective
with respect to the backbone representation z.

Two things differ from the paper's own tracker, both strengthening the test:

  * lambda_min is computed from the *exact* 512x512 effective Hessian in float64
    (see repro/hessian.py) rather than from a 20-step shifted power iteration.  The
    paper's estimator is still run every time, as an independent cross-check.
  * H_eff is split into the paper's own two terms G and M, so Theorem 4.1's stated
    mechanism ("the interaction term vanishes for linear heads") is checked directly
    instead of being inferred from an eigenvalue sign.

The linear and ReLU heads are the built-in negative controls: their second derivative
is zero (identically, and almost-everywhere respectively), so M must vanish and
lambda_min(H_eff) must not go below the round-off floor.  A pipeline that reported
negative curvature for those heads would be reporting numerical noise.
"""

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from repro import data
from repro import hessian as HZ
from repro.models import ProjectionHead, ResNetBackbone

# Round-off floor for calling an eigenvalue negative, expressed relative to the
# largest-magnitude eigenvalue of the same matrix.  float64 eigvalsh on a 512x512
# matrix is accurate to a few units in the last place, so 1e-10 is ~6 orders of
# magnitude of headroom.  Fixed before any run; never tuned per configuration.
EIG_TOL = 1e-10


def _scale(summ, name):
    """Largest-magnitude eigenvalue of one matrix — the round-off floor's reference."""
    return max(abs(summ[f"{name}_lambda_min"]), abs(summ[f"{name}_lambda_max"]))


def _frac_neg(lambda_min, scale, tol=EIG_TOL):
    """Fraction of samples whose lambda_min is negative *beyond float64 round-off*.

    A bare `lambda_min < 0` test is meaningless here: for a piecewise-linear head the
    interaction term vanishes and lambda_min is a round-off residue that lands on
    either side of zero at random.  ReLU produced lambda_min = -1.1e-16 against a
    spectral scale of 2.0e-6, i.e. 1e-10 of the scale; calling that negative would
    report noise as the paper's mechanism.  Same relative rule as `summarise`.
    """
    lambda_min = np.asarray(lambda_min, dtype=float)
    scale = np.maximum(np.asarray(scale, dtype=float), 1e-30)
    return float(np.mean(lambda_min < -tol * scale))


class TwoCropTransform:
    def __init__(self, base):
        self.base = base

    def __call__(self, x):
        return [self.base(x), self.base(x)]


def loaders(batch_size, seed, root="./data"):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    data.stage(root)
    ds = torchvision.datasets.CIFAR10(root=root, train=True,
                                      transform=TwoCropTransform(transform), download=True)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                      drop_last=True, generator=g)


def build(activation, init, seed, use_bn=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone = ResNetBackbone()
    projector = ProjectionHead(512, 2048, 2048, activation, use_bn=use_bn)
    predictor = ProjectionHead(2048, 512, 2048, activation, use_bn=use_bn)
    if init == "collapsed":
        with torch.no_grad():
            for m in projector.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.data *= 0.1
    return backbone, projector, predictor


def verify_machinery(activation="swish", seed=0):
    """Independent checks that the exact-Hessian machinery itself is correct.

    1. Finite differences of the analytic gradient must reproduce H_eff.
    2. For a linear head the interaction term M must be exactly zero.
    3. G must equal J^T H_L J by construction, and H_eff = G + M by definition.
    """
    print("\n--- machinery verification ---", flush=True)
    out = {}
    for act in ("linear", activation):
        _, projector, predictor = build(act, "normal", seed)
        z = torch.randn(512, generator=torch.Generator().manual_seed(seed)) * 0.1
        c = torch.randn(2048, generator=torch.Generator().manual_seed(seed + 1))
        parts = HZ.decompose(projector, predictor, z, c)
        H = parts["H_eff"]

        import copy as _copy
        pj = _copy.deepcopy(projector).double().eval()
        pd = _copy.deepcopy(predictor).double().eval()
        params_j = {k: v.detach() for k, v in pj.named_parameters()}
        params_d = {k: v.detach() for k, v in pd.named_parameters()}
        head = HZ._head_map(pj, pd, params_j, params_d)
        loss = HZ.negcos_loss(c.double())

        def grad_at(zz):
            zz = zz.clone().requires_grad_(True)
            return torch.autograd.grad(loss(head(zz)), zz)[0]

        eps = 1e-5
        idx = [0, 7, 100, 511]
        fd_err = 0.0
        z64 = z.double()
        for i in idx:
            e = torch.zeros(512, dtype=torch.float64)
            e[i] = eps
            col = (grad_at(z64 + e) - grad_at(z64 - e)) / (2 * eps)
            fd_err = max(fd_err, float((col - H[:, i]).abs().max()))
        rec = {
            "fd_max_abs_err": fd_err,
            "H_scale": float(H.abs().max()),
            "M_fro": float(torch.linalg.norm(parts["M"])),
            "H_eff_fro": float(torch.linalg.norm(parts["H_eff"])),
            "identity_G_plus_M_err": float(
                torch.linalg.norm(parts["H_eff"] - parts["G"] - parts["M"])),
            "asymmetry": float(torch.linalg.norm(H - H.T)),
        }
        rec["M_over_Heff"] = rec["M_fro"] / (rec["H_eff_fro"] + 1e-300)
        out[act] = rec
        print(f"{act:8s} {json.dumps(rec)}", flush=True)

    ok = (out["linear"]["M_over_Heff"] < 1e-12
          and out[activation]["M_over_Heff"] > 1e-6
          and all(v["fd_max_abs_err"] < 1e-6 * max(v["H_scale"], 1e-12) + 1e-9
                  for v in out.values()))
    print(f"machinery_ok = {ok}", flush=True)
    return out, ok


def run(activation, init, epochs, batch_size=256, seed=0, lr=0.05,
        hess_batches=5, hess_samples=8, max_steps_per_epoch=None, use_bn=False):
    t0 = time.perf_counter()
    cfg = dict(activation=activation, init=init, epochs=epochs, batch_size=batch_size,
               seed=seed, lr=lr, momentum=0.9, hess_batches=hess_batches,
               hess_samples=hess_samples, use_bn=use_bn, eig_tol=EIG_TOL,
               objective="simsiam_negative_cosine", dataset="cifar10",
               backbone="resnet18_cifar", head="512-2048-2048 / 2048-512-2048, bias-free")
    print("CONFIG " + json.dumps(cfg), flush=True)

    backbone, projector, predictor = build(activation, init, seed, use_bn)
    loader = loaders(batch_size, seed)
    params = (list(backbone.parameters()) + list(projector.parameters())
              + list(predictor.parameters()))
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)

    for epoch in range(epochs):
        ep_exact, ep_power, ep_M, ep_G, ep_HL = [], [], [], [], []
        ep_exact_tan, ep_exact_mse, ep_Mrel, ep_Mrel_mse, ep_HLtan = [], [], [], [], []
        ep_var, ep_cond = [], []
        sc_exact, sc_tan, sc_mse = [], [], []
        for i, ((x1, x2), _) in enumerate(loader):
            if max_steps_per_epoch is not None and i >= max_steps_per_epoch:
                break
            z1 = backbone(x1)
            z2 = backbone(x2)
            z1 = z1.requires_grad_(True) if not z1.requires_grad else z1
            p1, p2 = projector(z1), projector(z2)
            h1, h2 = predictor(p1), predictor(p2)
            loss = HZ.simsiam_loss(h1, h2, p1, p2)

            if i < hess_batches:
                ep_power.append(HZ.power_iteration_lambda_min(loss, z1))
                zs = z1.detach()[:hess_samples]
                cs = p2.detach()[:hess_samples]
                for s in range(zs.shape[0]):
                    # (a) the objective actually optimised, in ambient coordinates
                    cos_parts = HZ.decompose(projector, predictor, zs[s], cs[s],
                                             HZ.negcos_loss)
                    cos_summ = HZ.summarise(cos_parts, EIG_TOL)
                    # (c) the paper's stated PSD case: a standard MSE objective
                    mse_parts = HZ.decompose(projector, predictor, zs[s], cs[s],
                                             HZ.mse_loss)
                    mse_summ = HZ.summarise(mse_parts, EIG_TOL)
                    ep_exact.append(cos_summ["H_eff_lambda_min"])
                    ep_exact_tan.append(cos_summ["H_eff_tan_lambda_min"])
                    ep_exact_mse.append(mse_summ["H_eff_lambda_min"])
                    sc_exact.append(_scale(cos_summ, "H_eff"))
                    sc_tan.append(_scale(cos_summ, "H_eff_tan"))
                    sc_mse.append(_scale(mse_summ, "H_eff"))
                    ep_G.append(cos_summ["G_lambda_min"])
                    ep_M.append(cos_summ["M_lambda_min"])
                    ep_Mrel.append(cos_summ["M_over_Heff_fro"])
                    ep_Mrel_mse.append(mse_summ["M_over_Heff_fro"])
                    ep_HL.append(cos_summ["H_L_lambda_min"])
                    ep_HLtan.append(cos_summ["H_L_tan_lambda_min"])
                    if epoch == 0 and i == 0 and s == 0:
                        print("SAMPLE0_COS " + json.dumps(cos_summ), flush=True)
                        print("SAMPLE0_MSE " + json.dumps(mse_summ), flush=True)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                zc = F.normalize(torch.cat([z1.detach(), z2.detach()], 0), dim=1)
                ep_var.append(float(zc.std(0).mean()))
                zk = zc - zc.mean(0, keepdim=True)
                cov = (zk.T @ zk) / (zc.shape[0] - 1)
                ev = torch.linalg.eigvalsh(cov)
                ep_cond.append(float(ev[-1] / torch.clamp(ev[0], min=1e-7)))

        rec = {
            "epoch": epoch,
            "activation": activation,
            "init": init,
            # (a) ambient cosine objective — what Figure 3 actually optimises
            "exact_lambda_min_mean": float(np.mean(ep_exact)),
            "exact_lambda_min_min": float(np.min(ep_exact)),
            "exact_frac_neg": _frac_neg(ep_exact, sc_exact),
            "exact_frac_neg_raw_sign": float(np.mean(np.array(ep_exact) < 0.0)),
            "exact_scale_mean": float(np.mean(sc_exact)),
            # (b) same objective restricted to the sphere's tangent space
            "tan_lambda_min_mean": float(np.mean(ep_exact_tan)),
            "tan_lambda_min_min": float(np.min(ep_exact_tan)),
            "tan_frac_neg": _frac_neg(ep_exact_tan, sc_tan),
            "tan_frac_neg_raw_sign": float(np.mean(np.array(ep_exact_tan) < 0.0)),
            "tan_scale_mean": float(np.mean(sc_tan)),
            # (c) the paper's stated PSD premise: standard MSE objective
            "mse_lambda_min_mean": float(np.mean(ep_exact_mse)),
            "mse_lambda_min_min": float(np.min(ep_exact_mse)),
            "mse_frac_neg": _frac_neg(ep_exact_mse, sc_mse),
            "mse_frac_neg_raw_sign": float(np.mean(np.array(ep_exact_mse) < 0.0)),
            "mse_scale_mean": float(np.mean(sc_mse)),
            "M_over_Heff_cos_mean": float(np.mean(ep_Mrel)),
            "M_over_Heff_mse_mean": float(np.mean(ep_Mrel_mse)),
            "power_lambda_min_mean": float(np.mean(ep_power)),
            "G_lambda_min_min": float(np.min(ep_G)),
            "M_lambda_min_min": float(np.min(ep_M)),
            "H_L_lambda_min_min": float(np.min(ep_HL)),
            "H_L_tan_lambda_min_min": float(np.min(ep_HLtan)),
            "variance": float(np.mean(ep_var)),
            "cond_number": float(np.mean(ep_cond)),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        print("EPOCH " + json.dumps(rec), flush=True)
        # Raw per-sample spectra, so any reader can recompute every fraction above at
        # a tolerance of their own choosing instead of trusting EIG_TOL.
        print("SAMPLES " + json.dumps({
            "epoch": epoch, "activation": activation, "init": init,
            "exact_lambda_min": ep_exact, "exact_scale": sc_exact,
            "tan_lambda_min": ep_exact_tan, "tan_scale": sc_tan,
            "mse_lambda_min": ep_exact_mse, "mse_scale": sc_mse,
        }), flush=True)
    print(f"DONE {activation}/{init} in {time.perf_counter() - t0:.1f}s", flush=True)
