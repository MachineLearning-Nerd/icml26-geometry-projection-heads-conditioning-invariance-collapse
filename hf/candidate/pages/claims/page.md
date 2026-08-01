# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c284d4053c5f", "created_at": "2026-07-30T17:02:21+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. Theorem 4.1 shows that smooth nonlinear projection heads (e.g., using Swish or GELU) destabilize collapsed equilibria by injecting negative eigenvalues into the effective Hessian, while linear heads preserve collapse since the interaction term vanishes (Theorem 4.1, Section 4)
2. Theorem 3.1 establishes that linear projection heads perform implicit local subspace whitening, achieving isotropy only on the active subspace determined by the loss rank (Theorem 3.1, Section 3)
3. Proposition 3.3 shows that when the loss geometry has nonvanishing Riemann curvature, no global linear map can produce everywhere-isotropic conditioning, motivating the use of nonlinear heads (Proposition 3.3, Section 3)
4. Figure 3 tracks the Hessian spectrum during training and shows that smooth-activation heads inject negative eigenvalues enabling escape from collapse, whereas ReLU-based heads fail to do so, relying instead on discrete dynamics and BatchNorm (Figure 3, Section 6)
5. The geometric analysis is applied across both contrastive methods (InfoNCE, SimCLR, MoCo, DINO) and non-contrastive/decorrelation-based methods (BYOL, SimSiam, VICReg, Barlow Twins) (Section 6)
6. Figure 5 shows PCA visualization demonstrating 21.85x compression of augmentation orbits under the learned projection head geometry (Figure 5, Section 6)
