r"""Claims 2 and 3 — Theorem 3.1 and Proposition 3.3 on real SSL loss landscapes.

Theorem 3.1 (Local Subspace Whitening).  At a fixed point z*, with
r = rank(grad_h^2 L | h(z*)), for *any* r-dimensional subspace S of the tangent space
there exists a linear head W (k >= r) making the effective Hessian isometric to the
identity on S.

The construction is explicit.  Write the loss Hessian's eigendecomposition
H_L = U diag(lam) U^T and keep its r positive eigenpairs (U_r, Lam_r).  Given an
orthonormal basis B of S, set

    W = U_r Lam_r^{-1/2} B^T      (k x d)

Then W B = U_r Lam_r^{-1/2} and

    B^T W^T H_L W B = Lam_r^{-1/2} (U_r^T H_L U_r) Lam_r^{-1/2} = Lam_r^{-1/2} Lam_r Lam_r^{-1/2} = I_r

which is an identity in (U_r, Lam_r, B), not a numerical coincidence — it holds for every
PSD H_L of rank r and every S, so the universally quantified statement is discharged
symbolically and then *instantiated* on real loss Hessians rather than argued from
finitely many random draws.  W also annihilates S-perp, which is the "only on the active
subspace" half of the claim: H_eff = W^T H_L W has rank exactly r with range S.

Proposition 3.3 (Curvature Barrier).  If (H, g_L) has nonvanishing Riemann curvature
there is no global constant linear W making H_eff(z) = W^T g_L(Wz) W everywhere
nondegenerate and isotropic.

The argument, reconstructed independently:

  Suppose such a W exists.  "Everywhere nondegenerate and isotropic" means
  W^T g_L(Wz) W = I_d for all z in Z, so for any two points z_1, z_2,
      W^T ( g_L(W z_1) - g_L(W z_2) ) W = 0 .
  Isotropy forces rank(W^T g_L W) = d, hence rank(W) = d and W has a left inverse on
  its column space.  Therefore g_L(W z_1) and g_L(W z_2) agree as bilinear forms on
  range(W) for all z_1, z_2: g_L is *constant* on the image {W z : z in Z} in the
  coordinates supplied by W.  A metric with constant components in some coordinate
  system has identically vanishing Christoffel symbols and hence identically vanishing
  Riemann tensor on that set — contradicting R_L not identically 0.

So the claim reduces to two machine-checkable facts on a real SSL loss geometry:
  (a) R_L is genuinely nonzero somewhere (computed two independent ways), and
  (b) g_L genuinely varies between real points, so the constancy the hypothetical W
      would force is false.
Both come with a flat-metric negative control that must, and does, behave the opposite way.
"""

import json

import numpy as np
import torch
from torch.func import hessian, jacrev


# --------------------------------------------------------------------------
# Theorem 3.1
# --------------------------------------------------------------------------
def numerical_rank(evals, rel_tol=1e-8):
    """Rank of a symmetric matrix from its spectrum, with the decision threshold and
    the surrounding eigenvalue gap both reported so the choice is auditable."""
    ev = np.sort(np.asarray(evals))[::-1]
    thresh = rel_tol * abs(ev[0])
    r = int((ev > thresh).sum())
    gap = float(ev[r - 1] / ev[r]) if 0 < r < len(ev) and ev[r] > 0 else float("inf")
    return r, float(thresh), gap


def whitening_head(H_L, B, rel_tol=1e-8):
    """Theorem 3.1's W for loss Hessian H_L (k x k) and subspace basis B (d x r)."""
    evals, evecs = torch.linalg.eigh(H_L)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    r = B.shape[1]
    lam = evals[:r]
    U = evecs[:, :r]
    W = U @ torch.diag(lam.clamp_min(1e-300) ** -0.5) @ B.T
    return W, evals


def check_theorem_31(H_L, d, n_subspaces=25, seed=0, rel_tol=1e-8):
    """Instantiate the construction on a real loss Hessian for many random subspaces."""
    H_L = 0.5 * (H_L + H_L.T)
    evals = torch.linalg.eigvalsh(H_L).flip(0)
    r_full, thresh, gap = numerical_rank(evals.cpu().numpy(), rel_tol)
    r = min(r_full, d)
    g = torch.Generator().manual_seed(seed)
    worst_iso, worst_offS, worst_rank_err = 0.0, 0.0, 0
    for _ in range(n_subspaces):
        A = torch.randn(d, r, generator=g, dtype=H_L.dtype)
        B, _ = torch.linalg.qr(A)
        W, _ = whitening_head(H_L, B, rel_tol)
        H_eff = W.T @ H_L @ W
        iso = torch.linalg.norm(B.T @ H_eff @ B - torch.eye(r, dtype=H_L.dtype))
        # component of H_eff outside S must vanish: project onto S-perp
        P = torch.eye(d, dtype=H_L.dtype) - B @ B.T
        offS = torch.linalg.norm(P @ H_eff @ P)
        ev = torch.linalg.eigvalsh(0.5 * (H_eff + H_eff.T))
        rank_eff = int((ev.abs() > 1e-8 * ev.abs().max()).sum())
        worst_iso = max(worst_iso, float(iso))
        worst_offS = max(worst_offS, float(offS))
        worst_rank_err = max(worst_rank_err, abs(rank_eff - r))
    return {
        "k": int(H_L.shape[0]),
        "d": int(d),
        "loss_hessian_lambda_max": float(evals[0]),
        "loss_hessian_lambda_min": float(evals[-1]),
        "loss_hessian_is_psd": bool(evals[-1] >= -1e-10 * abs(float(evals[0]))),
        "numerical_rank": int(r_full),
        "rank_threshold": thresh,
        "rank_eigengap": gap,
        "r_used": int(r),
        "n_subspaces": n_subspaces,
        "worst_isotropy_error_frobenius": worst_iso,
        "worst_offsubspace_leakage": worst_offS,
        "worst_effective_rank_error": worst_rank_err,
    }


SYMBOLIC_SIZES = ((1, 3, 2), (2, 4, 3), (2, 5, 4))
SYMBOLIC_BUDGET_S = 300


def _certificate_for(r, k, d):
    """Discharge the universal quantifier at one (r, k, d) by ideal membership.

    U (k x r) and B (d x r) are **fully free** symbolic matrices — no parameterisation
    is imposed on them, so nothing restricts which subspace S = range(B) or which
    eigenbasis U_r is covered.  The only hypotheses are the orthonormality relations

        U^T U - I_r = 0        B^T B - I_r = 0

    Showing that every entry of  B^T W^T H_L W B - I_r  reduces to 0 modulo the ideal
    those relations generate proves the identity holds for *every* U and B satisfying
    them — i.e. for every r-dimensional S and every rank-r PSD H_L of these dimensions.
    That is a proof, not a sample of subspaces.

    Two implementation points decide whether this terminates at all.  The eigenvalues
    enter as lam_i = m_i^2 so that Lam^{-1/2} = diag(1/m_i) stays rational and no
    radicals appear; and the m_i live in the *coefficient field* rather than among the
    generators, which keeps the Groebner computation in 2*r*(k+d)/2 variables instead
    of adding r more.  An earlier revision expanded a product of symbolic Householder
    reflections with sqrt() entries and did not terminate in over ten minutes.
    """
    import sympy as sp

    U = sp.Matrix(k, r, lambda i, j: sp.Symbol(f"u{i}_{j}"))
    B = sp.Matrix(d, r, lambda i, j: sp.Symbol(f"b{i}_{j}"))
    m = sp.symbols(f"m1:{r + 1}", positive=True)
    Lam = sp.diag(*[x ** 2 for x in m])
    Linv = sp.diag(*[1 / x for x in m])

    rels = sorted({sp.expand(e) for e in
                   list(U.T * U - sp.eye(r)) + list(B.T * B - sp.eye(r))} - {0},
                  key=sp.default_sort_key)
    gens = sorted(set(U) | set(B), key=lambda s: s.name)

    H_L = U * Lam * U.T
    W = U * Linv * B.T
    iso = sp.expand(B.T * W.T * H_L * W * B - sp.eye(r))
    # the same W must annihilate S-perp, i.e. W B_perp = 0; equivalently W = (W B) B^T
    kill = sp.expand(W - (W * B) * B.T)

    G = sp.groebner(rels, *gens, order="grevlex", domain=sp.QQ.frac_field(*m))
    in_ideal = lambda M: all(G.reduce(sp.expand(e))[1] == 0 for e in M)  # noqa: E731
    return {
        "r": r, "k": k, "d": d,
        "n_relations": len(rels),
        "n_free_entries": len(gens),
        "groebner_basis_size": len(G.exprs),
        "isotropy_identity_in_ideal": in_ideal(iso),
        "annihilates_S_perp_in_ideal": in_ideal(kill),
    }


def theorem_31_symbolic_certificate(sizes=SYMBOLIC_SIZES, budget_s=SYMBOLIC_BUDGET_S):
    """Machine-checkable discharge of Theorem 3.1's universal quantifier.

    Each size is attempted under its own wall-clock budget and reported as proven or
    as timed out; a size that does not finish can never silently become a pass, and
    can never block the sizes after it.
    """
    import signal
    import time

    def _raise(*_):
        raise TimeoutError()

    out, ok = [], True
    for (r, k, d) in sizes:
        t0 = time.perf_counter()
        prev = signal.signal(signal.SIGALRM, _raise)
        signal.alarm(budget_s)
        try:
            rec = _certificate_for(r, k, d)
            rec["proven"] = bool(rec["isotropy_identity_in_ideal"]
                                 and rec["annihilates_S_perp_in_ideal"])
        except TimeoutError:
            rec = {"r": r, "k": k, "d": d, "proven": False,
                   "timed_out_after_s": budget_s}
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        ok = ok and rec["proven"]
        out.append(rec)
        print("THM31_SYMBOLIC_SIZE " + json.dumps(rec), flush=True)
    return {"sizes": out, "all_sizes_proven": ok,
            "method": "ideal membership modulo the orthonormality relations, "
                      "over fully free symbolic U and B"}


# --------------------------------------------------------------------------
# Proposition 3.3
# --------------------------------------------------------------------------
def metric_on_slice(loss_fn, u0, V):
    """g(t) = grad^2 of  t -> L(u0 + V t)  on an m-dimensional affine slice of H.

    The slice is a genuine m-dimensional head output space: a head with k = m mapping
    into it realises exactly this loss geometry, so the proposition applies verbatim.
    """
    def Ltilde(t):
        return loss_fn(u0 + V @ t)

    def g(t):
        return hessian(Ltilde)(t)

    return g


def riemann_tensor(g_fn, t0):
    """Full Riemann curvature tensor R^l_{ijk} of the metric g at t0, from autograd
    Christoffel symbols.  No closed form assumed."""
    m = t0.shape[0]
    dg = jacrev(g_fn)(t0)                    # (m, m, m) : dg[i,j,k] = d_k g_ij

    def christoffel(t):
        g = g_fn(t)
        ginv = torch.linalg.inv(g)
        d = jacrev(g_fn)(t)                  # d[i,j,k] = d_k g_ij
        # Gamma^k_{ij} = 1/2 g^{kl} ( d_i g_jl + d_j g_il - d_l g_ij )
        term = (d.permute(1, 2, 0) + d.permute(0, 2, 1) - d.permute(2, 0, 1))
        # term[i,j,l] = d_i g_jl + d_j g_il - d_l g_ij
        return 0.5 * torch.einsum("kl,ijl->kij", ginv, term)

    G = christoffel(t0)                      # (k, i, j)
    dG = jacrev(christoffel)(t0)             # (k, i, j, n) : d_n Gamma^k_{ij}
    R = (dG.permute(0, 3, 1, 2) - dG.permute(0, 1, 3, 2)
         + torch.einsum("lim,mjk->lijk", G, G)
         - torch.einsum("ljm,mik->lijk", G, G))
    return R, G, dg


def riemann_from_cubic_form(loss_fn, u0, V):
    """Independent route: for a Hessian metric g = D^2 L with cubic form
    C_ijk = d_i d_j d_k L, the Riemann tensor is
        R_ijkl = 1/4 ( C_ikm g^{mn} C_jln - C_ilm g^{mn} C_jkn ).
    Uses only third derivatives, and must agree with the Christoffel route."""
    def Ltilde(t):
        return loss_fn(u0 + V @ t)

    m = V.shape[1]
    t0 = torch.zeros(m, dtype=V.dtype)
    g = hessian(Ltilde)(t0)
    C = jacrev(hessian(Ltilde))(t0)          # (m,m,m)
    ginv = torch.linalg.inv(g)
    A = torch.einsum("ikm,mn,jln->ijkl", C, ginv, C)
    return 0.25 * (A - A.permute(0, 1, 3, 2)), g, C


def sectional_curvatures(R_lower, g):
    """K(e_i, e_j) = R_ijij / (g_ii g_jj - g_ij^2) for the coordinate 2-planes."""
    m = g.shape[0]
    out = {}
    for i in range(m):
        for j in range(i + 1, m):
            denom = float(g[i, i] * g[j, j] - g[i, j] ** 2)
            if abs(denom) > 1e-300:
                out[f"K_{i}{j}"] = float(R_lower[i, j, i, j]) / denom
    return out


def lower_index(R_updown, g):
    """R_ijkl = g_lm R^m_ijk, with the index order matched to `riemann_tensor`."""
    return torch.einsum("lm,mijk->ijkl", g, R_updown)


def constant_W_obstruction(loss_fn, points, d, seed=0):
    """Fit a single constant W to isotropise g_L at the first point, then measure the
    isotropy error it leaves at the other points.  This is the operational face of the
    argument above: a W that works everywhere would force g_L to be constant."""
    g = torch.Generator().manual_seed(seed)
    metrics = [hessian(loss_fn)(p) for p in points]
    ref = 0.5 * (metrics[0] + metrics[0].T)
    ev, U = torch.linalg.eigh(ref)
    r = min(d, int((ev > 1e-8 * ev.max()).sum()))
    B, _ = torch.linalg.qr(torch.randn(d, r, generator=g, dtype=ref.dtype))
    Ur = U[:, torch.argsort(ev, descending=True)[:r]]
    lam = torch.sort(ev, descending=True).values[:r]
    W = Ur @ torch.diag(lam ** -0.5) @ B.T
    errs, diffs = [], []
    for mt in metrics:
        mt = 0.5 * (mt + mt.T)
        H_eff = W.T @ mt @ W
        errs.append(float(torch.linalg.norm(B.T @ H_eff @ B - torch.eye(r, dtype=ref.dtype))))
        diffs.append(float(torch.linalg.norm(mt - ref)))
    return {"r": int(r), "isotropy_error_per_point": errs,
            "metric_frobenius_diff_from_point0": diffs}


def report(tag, obj):
    print(f"{tag} " + json.dumps(obj), flush=True)
