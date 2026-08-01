# Source audit — exact claim statements, assumptions and quantifiers

**Paper source of record.** `https://ar5iv.labs.arxiv.org/html/2605.17180`, retrieved
2026-08-01 with an explicit browser User-Agent
(`Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36`),
SHA-256 `c344481c6fa2c59b6439f41d2053c737d92e11da1e4a7890941c776188ade7a4`
(650,579 bytes). arXiv abstract page: `https://arxiv.org/abs/2605.17180`.
OpenReview forum: `https://openreview.net/forum?id=y4uR1LFClc`.

**Authors' released code and raw arrays.**
`https://github.com/farischaudhry/projection-head-geometry`, pinned at
`117231d60fee34d4906d1f16c5007e13a96a4d94`. Per-file SHA-256 values are printed by
every job that reads them and are recorded in `jobs/*.jsonl`.

**Central object.** Equation (1), the effective Hessian of the objective with respect to
the backbone representation z:

```
H_eff(z*) = J_h(z*)^T (grad_h^2 L) J_h(z*)  +  sum_{i=1..k} [grad_h L]_i grad_z^2 h_i(z*)
            \________ pullback metric G(z*) ________/  \______ interaction term M(z*) ______/
```

---

## Claim 1 — Theorem 4.1 (Generic Instability of Collapse), Section 4

> Let z* be a collapsed state. Under Assumptions 1, 6 and 7, assume the residual
> gradient vector ρ = ∇_h L is nonzero at z*.
> 1. **Linear heads preserve collapse:** If h(z) = Wz is linear, the interaction term
>    vanishes (∇²h = 0). The effective Hessian is PSD (H_eff ⪰ 0). Thus the collapsed
>    state is a non-repelling critical region and gradient descent has no infinitesimal
>    escape direction.
> 2. **Nonlinear heads destabilize collapse:** If h_φ is a nonlinear MLP with smooth
>    activations, then under Assumption 5, the collapsed state z* is generically a
>    strict instability region.

Section 4.2 adds the ReLU corollary: because ∇²ReLU(x) = 0 almost everywhere, M(z*)
vanishes for ReLU heads too, "theoretically rendering the collapsed state a non-repelling
valley just like a linear head", and ReLU heads "rely entirely on higher-order
heuristics — specifically finite step-size noise and BatchNorm".

**Quantifiers and scope.**
- Universally quantified over collapsed states z* satisfying Assumptions 1, 5, 6, 7.
- Part 1 is an *exact algebraic* statement (∇²h = 0 ⟹ M = 0), plus a *conditional*
  conclusion (H_eff ⪰ 0) that requires ∇²_h L ⪰ 0.
- Part 2 is generic, not universal: "generically a strict instability region".
- The ReLU statement is explicitly scoped to *continuous* gradient flow, no BatchNorm.
  Section 4.2 and Figure 4 say ReLU heads survive in practice via discrete-time noise
  and BatchNorm — so a ReLU run *with* BatchNorm or a large step size is outside scope.

**Load-bearing assumptions requiring numerical audit.**
- *PSD loss Hessian.* Section 4.1: "In standard MSE-based objectives, the intrinsic
  Hessian of the loss is PSD (H_L ⪰ 0)." Part 1's conclusion is void without it.
- *Assumption 6, nonvanishing residual gradient.* If ρ = ∇_h L = 0 then M = 0 for every
  head and the theorem is vacuous.
- *Assumption 7, timescale separation.* The head must adapt faster than the backbone so
  it can be treated as a locally equilibrated preconditioner.
- *Assumption 1, C² loss and head.* Required for ∇²h to exist at all.

---

## Claim 2 — Theorem 3.1 (Local Subspace Whitening), Section 3

> Let z* be a fixed point. Under Assumptions 1 and 3, let r = rank(∇²_h L|_{h(z*)}) be
> the intrinsic rank of the loss. **For any subspace S ⊂ T_{z*}Z of dimension r**, there
> exists a linear projection head W ∈ ℝ^{k×d} (with k ≥ r) such that the effective
> Hessian restricted to S is isometric to the identity on S:
> v^T H_eff(z*) v = ‖v‖₂², ∀ v ∈ S.

**Quantifiers.** ∀ S of dimension r, ∃ W with k ≥ r. This is a universally quantified
existence statement over a continuum of subspaces: finitely many random draws are
corroboration only, so a symbolic discharge of the quantifier is required.

**Scope note.** The isotropy is claimed *on S only*; nothing is claimed off S. The
judged claim wording — "achieving isotropy only on the active subspace determined by the
loss rank" — makes the off-S behaviour part of what must be shown.

**Assumption requiring audit.** r ≤ d is needed for an r-dimensional subspace of
T_{z*}Z ≅ ℝ^d to exist at all; and ∇²_h L must be PSD for H_L^{+1/2} to be real.

---

## Claim 3 — Proposition 3.3 (Curvature Barrier for Linear Heads), Section 3

> Assume Assumption 1. Let g_L(u) := ∇²_u L(u) be the Riemannian metric induced by the
> loss. If the intrinsic geometry (H, g_L) has nonvanishing Riemann curvature R_L ≢ 0,
> there exists **no** global constant linear map W ∈ ℝ^{k×d} such that the induced
> effective Hessian H_eff(z) = W^T g_L(Wz) W is **everywhere** nondegenerate and
> isotropic on the optimization manifold Z.

**Quantifiers.** A non-existence statement, universally quantified over all constant W.
No finite search over W can establish it; a proof-level argument is required. Fitting one
W at one point and observing failure elsewhere is corroboration, not proof.

**Hypothesis requiring audit.** R_L ≢ 0 must be established for the loss geometry in
question, otherwise the proposition is vacuous — a flat metric *is* globally whitenable,
which is the required negative control.

---

## Claim 4 — Figure 3, Section 6.1

> Geometric preconditioning and collapse recovery (CIFAR-10, ResNet-18). Smooth heads
> (Swish) natively inject negative curvature (λ_min < 0) to navigate the landscape and
> aggressively escape collapsed equilibria, whereas pure ReLU networks lack this
> intrinsic mechanism.

Stated quantifiers in the surrounding text:
- normal init + Swish: Spearman ρ_s = **0.339** between representation variance and
  condition number;
- pseudo-collapsed + Swish: ρ_s = **0.609**;
- pseudo-collapsed + ReLU: ρ_s = **0.669**, with "no negative curvature" and variance
  and condition number "oscillating statically around their starting values".

**Method as stated (Appendix C.3).** λ_min of the effective Hessian estimated *without
materializing the matrix*, by exact Hessian-vector products with shifted power iteration
(λ_max first, then the dominant eigenvector of H − λ_max I), 20 iterations per estimate,
computed on the first few batches of each epoch. Training: SimSiam-style objective,
CIFAR-10, ResNet-18, batch 256, SGD lr 0.05 momentum 0.9, BatchNorm removed,
pseudo-collapsed init = projector linear weights scaled by α = 0.1, 100 epochs.

**Precision caveat that must be audited.** The estimator is a 20-step float32 power
iteration; the released λ_min values are of order 1e-7 to 1e-6. Whether the reported
sign structure is resolvable at that scale is itself a question the reproduction must
answer, not assume.

---

## Claim 5 — Section 6 generality

> The geometric analysis is applied across both contrastive methods (InfoNCE, SimCLR,
> MoCo, DINO) and non-contrastive/decorrelation-based methods (BYOL, SimSiam, VICReg,
> Barlow Twins).

Section 6.3 / Appendix D.6 instantiate this on official public checkpoints released
*with* the projection head: DINO ViT-S/16, DINO ResNet-50, VICReg ResNet-50, Barlow
Twins ResNet-50, evaluated on CIFAR-10 test images resized to 224×224 with ImageNet
normalization, sweeping four continuous orbits — rotation [0°,45°], hue [−0.4,0.4],
saturation [0,2], Gaussian blur σ ∈ [0.1,3.0] — with 12 interpolation steps and mean
statistics over 50 sampled image trajectories.

**Scope.** The claim is that the *analysis applies* to both families, i.e. the geometric
mechanisms (whitening, metric singularity, curvature-induced destabilization) are
well-defined and hold for each objective — not that every method produces the same
numbers.

---

## Claim 6 — Figure 5 / Table 1, Section 6.2

> Figure 5 shows PCA visualization demonstrating **21.85×** compression of augmentation
> orbits under the learned projection head geometry.

Exact quantifiers from Table 1 and the surrounding text (CIFAR-10, ResNet-18):
- Mean orbit spread (×10⁻²): backbone **2.25 ± 1.07**, head **0.10 ± 0.06**,
  "21.85× Collapse"; the text states the reduction is "from 0.0225 to 0.0010".
- Local curvature: backbone 0.125 ± 0.002, head 0.242 ± 0.004, ratio **1.93×**.
- D_intra: 0.211 ± 0.045 → 0.044 ± 0.011, **4.76×** reduction.
- D_inter: 0.432 ± 0.052 → 0.111 ± 0.014, **3.89×** reduction.
- Class/orbit ratio: 2.04 ± 0.52 → 2.50 ± 0.72, **1.22×** separation.
- Orbit metrics over **n = 15** independent augmentation orbits; curvature over n = 3
  seeds; probes from the final 50-epoch SimCLR checkpoint.
- ViT-Tiny counterpart (Table 5): **4015.33×** compression, curvature ratio 2.59×,
  D_intra 64.13×, D_inter 55.95×, class/orbit 1.14×.

**Protocol (Appendix C.3).** Orbit construction: 5 classes, 3 images per class, 12
rotations spanning [0°,360°). Representations L2-normalized; head applied to the
normalized backbone output. Mean orbit spread = mean squared Euclidean distance of the
points of an orbit from that orbit's own centroid.

**The paper's own falsifiable side-claim.** "This compression is not merely a byproduct
of dimensionality reduction (the head is actually higher-dimensional, 2048 vs 512), but
rather a targeted geometric collapse induced by the learned metric." That sentence is
what makes an untrained-head control the right negative control.
