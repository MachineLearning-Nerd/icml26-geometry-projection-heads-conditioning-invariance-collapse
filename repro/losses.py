"""The eight SSL objectives named in the paper's Section 6 generality claim, as real
implementations rather than stand-in matrices.

Contrastive / self-distillation family:  InfoNCE, SimCLR (NT-Xent), MoCo, DINO
Non-contrastive / decorrelation family:  BYOL, SimSiam, VICReg, Barlow Twins

Each factory returns a scalar function of a *single* anchor's head output h, with the
rest of the batch, the queue, the teacher outputs and the centre held fixed at their
observed values.  That is exactly what the paper's `grad_h^2 L | h(z*)` denotes: the
Hessian of the objective at one point of the head's output space.  Holding the context
fixed is the definition of a Hessian at a point, not a simplification of the loss.

Every function is written to be twice differentiable by autograd so the loss Hessian is
exact rather than finite-differenced.
"""

import torch
import torch.nn.functional as F


def _n(x, dim=-1):
    # sqrt(sum of squares) rather than Tensor.norm: norm's backward masks the
    # zero-norm case in place, which does not survive the third and fourth
    # derivatives that Proposition 3.3's curvature tensor takes through this.
    # The head outputs here are never the zero vector, so the two agree to
    # machine precision.
    return x / (torch.sqrt((x * x).sum(dim, keepdim=True) + 1e-30) + 1e-12)


def infonce(pos, negs, temperature=0.1):
    """-log softmax over cosine similarities against one positive and N negatives."""
    p, N = _n(pos).detach(), _n(negs, dim=1).detach()

    def f(h):
        hh = _n(h)
        logits = torch.cat([(hh * p).sum().unsqueeze(0), N @ hh]) / temperature
        # Stable log-sum-exp written out: torch.logsumexp masks -inf entries in
        # place, which breaks the higher-order autograd Proposition 3.3 needs.
        shift = logits.max().detach()
        return -logits[0] + shift + torch.log(torch.exp(logits - shift).sum())

    return f


def simclr(pos, negs, temperature=0.1):
    """NT-Xent as implemented in the paper's own code: cross-entropy over the
    similarity row of the anchor against the other view of the batch."""
    return infonce(pos, negs, temperature)


def moco(pos, queue, temperature=0.2):
    """InfoNCE against a momentum queue of detached keys (MoCo's defining difference
    from SimCLR is where the negatives come from, and its default temperature)."""
    return infonce(pos, queue, temperature)


def dino(teacher_logits, centre, t_temp=0.04, s_temp=0.1):
    """Self-distillation cross-entropy between a centred, sharpened teacher
    distribution and the student's distribution over the same prototypes."""
    pt = F.softmax((teacher_logits.detach() - centre.detach()) / t_temp, dim=0)

    def f(h):
        return -(pt * F.log_softmax(h / s_temp, dim=0)).sum()

    return f


def byol(target, predictor_weight):
    """2 - 2 cos(q(h), target) with an explicit linear predictor q."""
    t = _n(target).detach()
    Wq = predictor_weight.detach()

    def f(h):
        return 2 - 2 * (_n(Wq @ h) * t).sum()

    return f


def simsiam(target):
    """-cos(h, stopgrad(target)); the paper's collapse experiments use this form."""
    t = _n(target).detach()

    def f(h):
        return -(_n(h) * t).sum()

    return f


def vicreg(pos, batch, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4):
    """Invariance + variance hinge + covariance off-diagonal penalty.

    The anchor participates in all three terms; the remaining batch rows are fixed.
    """
    p = pos.detach()
    rest = batch.detach()
    n = rest.shape[0] + 1
    d = rest.shape[1]

    def f(h):
        X = torch.cat([h.unsqueeze(0), rest], 0)
        inv = ((h - p) ** 2).mean()
        Xc = X - X.mean(0, keepdim=True)
        std = torch.sqrt(Xc.var(0) + eps)
        var = F.relu(1.0 - std).mean()
        cov = (Xc.T @ Xc) / (n - 1)
        off = (cov ** 2).sum() - (torch.diagonal(cov) ** 2).sum()
        return sim_coeff * inv + std_coeff * var + cov_coeff * off / d

    return f


def barlow_twins(view_b, batch_a, lambd=5e-3, eps=1e-5):
    """Cross-correlation of batch-standardised views driven towards the identity."""
    B = view_b.detach()
    A_rest = batch_a.detach()
    n = A_rest.shape[0] + 1

    def f(h):
        A = torch.cat([h.unsqueeze(0), A_rest], 0)
        An = (A - A.mean(0, keepdim=True)) / (A.std(0, keepdim=True) + eps)
        Bn = (B - B.mean(0, keepdim=True)) / (B.std(0, keepdim=True) + eps)
        C = (An.T @ Bn) / n
        on = ((torch.diagonal(C) - 1) ** 2).sum()
        off = (C ** 2).sum() - (torch.diagonal(C) ** 2).sum()
        return on + lambd * off

    return f


CONTRASTIVE = ("infonce", "simclr", "moco", "dino")
NON_CONTRASTIVE = ("byol", "simsiam", "vicreg", "barlow_twins")
ALL = CONTRASTIVE + NON_CONTRASTIVE


def build_all(H1, H2, generator):
    """Instantiate all eight per-anchor objectives from a real pair of head-output
    batches H1, H2 (rows are samples).  Anchor is row 0 of H1, positive is row 0 of H2.
    """
    k = H1.shape[1]
    anchor, pos = H1[0], H2[0]
    negs = H2[1:]
    queue = H2[1:].roll(1, 0)
    centre = H2.mean(0)
    Wq = torch.randn(k, k, generator=generator, dtype=H1.dtype) / (k ** 0.5)
    return {
        "infonce": infonce(pos, negs),
        "simclr": simclr(pos, negs),
        "moco": moco(pos, queue),
        "dino": dino(H2[0], centre),
        "byol": byol(pos, Wq),
        "simsiam": simsiam(pos),
        "vicreg": vicreg(pos, H1[1:]),
        "barlow_twins": barlow_twins(H2, H1[1:]),
    }, anchor
