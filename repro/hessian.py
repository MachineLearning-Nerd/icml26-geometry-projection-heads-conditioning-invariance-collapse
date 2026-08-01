r"""Exact effective Hessian of the SSL objective with respect to the backbone
representation, and its decomposition into the two terms of the paper's eq. (1).

The paper writes, at a representation z,

    H_eff(z) = J_h(z)^T (grad_h^2 L) J_h(z)   +   sum_i [grad_h L]_i grad_z^2 h_i(z)
               \_______ pullback metric G _______/   \______ interaction term M ______/

Theorem 4.1 turns entirely on M: for a linear head grad_z^2 h = 0 so M vanishes
identically and H_eff = G is PSD, while for a smooth nonlinear head M is generically
indefinite and drags lambda_min(H_eff) below zero.

In the SimSiam-style objective used for the collapse experiments the term that
reaches z_1 is  -0.5 * cos(predictor(projector(z_1)), c)  with c = p_2 detached, and
the other term is fully detached.  The map z_i -> loss_i therefore factors per sample
and passes only through the two small MLP heads, never through the ResNet.  That makes
the full 512x512 block *exactly* computable in float64 rather than estimated by power
iteration, and makes G and M separately computable — which is what these functions do.

Everything here is deterministic: no sampling, no iteration counts, no tolerances that
could be tuned to produce a desired sign.
"""

import copy

import torch
import torch.nn.functional as F
from torch.func import functional_call, hessian, jacfwd


def _head_map(projector, predictor, params_proj, params_pred):
    """z (d,) -> h (k,), applied to a single sample, in the module's dtype."""

    def f(z):
        p = functional_call(projector, params_proj, (z.unsqueeze(0),)).squeeze(0)
        return functional_call(predictor, params_pred, (p.unsqueeze(0),)).squeeze(0)

    return f


def negcos_loss(c):
    """SimSiam per-sample loss  l(h) = -0.5 * cos(h, c)  with c a fixed target."""
    cn = c / (c.norm() + 1e-12)

    def f(h):
        return -0.5 * torch.dot(h / (h.norm() + 1e-12), cn)

    return f


def mse_loss_normalised(c):
    """PSD-by-construction alternative: l(h) = 0.25 * || h/|h| - c/|c| ||^2.

    Identical to `negcos_loss` up to an additive constant on the unit sphere, but its
    Hessian in the *ambient* h coordinates differs.  Used as the assumption control for
    the paper's "the intrinsic Hessian of the loss is PSD" premise.
    """
    cn = c / (c.norm() + 1e-12)

    def f(h):
        return 0.25 * torch.sum((h / (h.norm() + 1e-12) - cn) ** 2)

    return f


def decompose(projector, predictor, z, c, loss_factory=negcos_loss):
    """Exact H_eff, G and M at one representation z, in float64.

    Returns a dict of tensors; all shapes (d, d) except H_L which is (k, k).
    """
    # Work on detached float64 copies: the live modules stay float32 and keep the
    # exact Parameter objects the optimiser's momentum buffers are keyed on.
    projector = copy.deepcopy(projector).double().eval()
    predictor = copy.deepcopy(predictor).double().eval()
    params_proj = {k: v.detach() for k, v in projector.named_parameters()}
    params_pred = {k: v.detach() for k, v in predictor.named_parameters()}
    z = z.detach().double()
    c = c.detach().double()

    head = _head_map(projector, predictor, params_proj, params_pred)
    loss = loss_factory(c)

    def total(zz):
        return loss(head(zz))

    H_eff = hessian(total)(z)                     # (d, d)
    J = jacfwd(head)(z)                           # (k, d)
    h = head(z)
    H_L = hessian(loss)(h)                        # (k, k)
    G = J.T @ H_L @ J                             # pullback metric
    M = H_eff - G                                 # interaction term
    return {"H_eff": H_eff, "G": G, "M": M, "H_L": H_L, "J": J}


def spectrum(A):
    """Symmetrised eigenvalues, ascending.  Symmetrisation removes only the
    antisymmetric part produced by floating-point asymmetry (~1e-16 here)."""
    A = 0.5 * (A + A.T)
    return torch.linalg.eigvalsh(A)


def summarise(parts, tol):
    """Scalar summary of one representation's effective Hessian.

    `tol` is the numerical-noise floor: an eigenvalue is only called negative when it
    is below -tol.  It is set from a measured round-off scale, never tuned per run.
    """
    out = {}
    for name in ("H_eff", "G", "M"):
        ev = spectrum(parts[name])
        scale = float(ev.abs().max())
        out[f"{name}_lambda_min"] = float(ev[0])
        out[f"{name}_lambda_max"] = float(ev[-1])
        out[f"{name}_n_neg"] = int((ev < -tol * max(scale, 1e-30)).sum())
        out[f"{name}_fro"] = float(torch.linalg.norm(parts[name]))
    ev = spectrum(parts["H_L"])
    out["H_L_lambda_min"] = float(ev[0])
    out["H_L_lambda_max"] = float(ev[-1])
    out["M_over_Heff_fro"] = out["M_fro"] / (out["H_eff_fro"] + 1e-300)
    return out


def power_iteration_lambda_min(loss_scalar, z, num_iterations=20):
    """The paper's own estimator, kept as an independent cross-check of the exact
    spectra above (Appendix C, 'Hessian tracking'): shifted power iteration on
    Hessian-vector products, 20 iterations."""
    grad_z = torch.autograd.grad(loss_scalar, z, create_graph=True)[0]
    v = torch.randn_like(z)
    v = v / (v.norm() + 1e-8)
    for _ in range(num_iterations):
        Hv = torch.autograd.grad(grad_z, z, grad_outputs=v, retain_graph=True)[0]
        v = Hv / (Hv.norm() + 1e-8)
    lam_max = torch.sum(v * torch.autograd.grad(grad_z, z, grad_outputs=v, retain_graph=True)[0])
    vm = torch.randn_like(z)
    vm = vm / (vm.norm() + 1e-8)
    for _ in range(num_iterations):
        Hv = torch.autograd.grad(grad_z, z, grad_outputs=vm, retain_graph=True)[0]
        shifted = Hv - lam_max * vm
        vm = shifted / (shifted.norm() + 1e-8)
    Hv = torch.autograd.grad(grad_z, z, grad_outputs=vm, retain_graph=True)[0]
    return float(torch.sum(vm * Hv))


def simsiam_loss(h1, h2, p1, p2):
    """The exact objective used by the paper's Hessian tracker."""
    return 0.5 * (-(F.cosine_similarity(h1, p2.detach()).mean()
                    + F.cosine_similarity(h2, p1.detach()).mean()))
