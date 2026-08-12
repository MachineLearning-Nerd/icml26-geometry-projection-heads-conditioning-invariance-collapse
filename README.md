# ICML 2026 — The Geometry of Projection Heads

Reproduction and audit repository for **The Geometry of Projection Heads:
Conditioning, Invariance, and Collapse** by Faris Chaudhry.

- Paper: [arXiv:2605.17180](https://arxiv.org/abs/2605.17180)
- Paper HTML: [arxiv.org/html/2605.17180](https://arxiv.org/html/2605.17180)
- Authors' code and released arrays:
  [farischaudhry/projection-head-geometry](https://github.com/farischaudhry/projection-head-geometry)
- Research logbook:
  [DineshAI/y4uR1LFClc](https://huggingface.co/spaces/DineshAI/y4uR1LFClc)
- Clean repository URL:
  [MachineLearning-Nerd/icml26-geometry-projection-heads-conditioning-invariance-collapse](https://github.com/MachineLearning-Nerd/icml26-geometry-projection-heads-conditioning-invariance-collapse)

## Status at a glance

The current candidate evidence marks all six scoped claim contracts **VERIFIED**.
That status is deliberately qualified below and on every claim page. It means the
current evidence meets the repository's registered predicates; it does not mean that
every original 100-epoch training budget was independently rerun.

The historical judged Space revision
099048293db504eb467f72c37f7bfd371dadcfcb scored 5/12 because its checks used
hand-constructed matrices. That revision is preserved under hf/judged and the
candidate logbook's [Historical rejected baseline](https://huggingface.co/spaces/DineshAI/y4uR1LFClc/tree/main/pages/verification-run)
page. The current candidate pages replace those checks with exact Hessians, real
trained networks, released full-scale arrays, official checkpoints, and explicit
assumption audits.

## What the paper is doing

The paper treats a projection head in self-supervised learning as a trainable
Riemannian metric rather than as a disposable MLP:

1. A linear head can whiten the active loss-Hessian subspace, improving local
   conditioning but only on the subspace where the loss has curvature.
2. A nonlinear head changes the local metric across representation space and can
   introduce curvature that no single constant linear map can remove globally.
3. At a collapsed representation, the effective Hessian splits into a pullback
   term G and an interaction term M. Smooth nonlinear activations can make M
   indefinite; linear and ReLU heads provide negative controls where the interaction
   term vanishes up to numerical round-off.
4. The head therefore mediates a trade-off between conditioning, invariance, and
   information loss. Its effect on augmentation orbits depends on the SSL objective:
   DINO contracts the measured orbits, while Barlow Twins expands them in the
   released checkpoint analysis.

The central evidence path is:

    paper claim
        -> registered predicate in .openresearch/artifacts/claim_contracts.json
        -> committed verifier in repro/
        -> raw artifact or symbolic certificate
        -> claim page and assumption audit
        -> branch/ref recorded in branch-audit.md

## Claim-to-evidence ledger

| Claim | Paper object | How the result is produced | Current result and boundary |
| --- | --- | --- | --- |
| 1 | Theorem 4.1, collapse instability | Exact float64 effective Hessians on real trained states; hessian.py separates H_eff into G and M; collapse.py and released.py provide training and released-array checks | VERIFIED: M/H_eff is about 3.4e-15 for linear and material for smooth heads. The PSD conclusion is evaluated under the theorem's MSE premise because the paper's cosine objective has an indefinite ambient Hessian. |
| 2 | Theorem 3.1, local subspace whitening | geometry.py proves W = U Lambda^(-1/2) Bᵀ symbolically for the universal quantifier, then checks real objective Hessians | VERIFIED: the symbolic cases are proven and the real-loss worst isotropy error is 9.63e-11. InfoNCE and SimCLR are one distinct implementation. |
| 3 | Proposition 3.3, curvature barrier | geometry.py computes the Riemann tensor by two routes and compares against a flat metric control | VERIFIED within the qualifying positive-definite cases: two real VICReg geometries pass. Four indefinite cases and two fourth-derivative failures are excluded rather than hidden. |
| 4 | Figure 3, Hessian spectrum and variance relation | released.py reproduces the authors' full-scale correlations; collapse.py performs exact float64 spot checks and audits the short power-iteration estimator | VERIFIED: Spearman values 0.669, 0.609, and 0.339 are reproduced to three decimals. The released float32 estimator under-resolves some eigenvalue magnitudes. |
| 5 | Section 6, generality across SSL methods | losses.py implements eight named objectives; geometry.py evaluates their real Hessians; pretrained.py evaluates four official checkpoints and their heads | VERIFIED for the enumerated scope: eight labels, seven distinct loss functions, and four official checkpoints. It is not a survey of every SSL method. |
| 6 | Figure 5, augmentation-orbit compression | released.py reads the authors' pinned orbit_visualization.npy, recomputes spreads and bootstrap intervals, and compares with an untrained same-width head | VERIFIED: 21.85245268 rounds to 21.85x; the untrained control is about 1x. The independent clean-room training run reached only the 0-epoch corroboration point. |

Detailed tables, raw-file links, negative controls, and assumptions are in the
candidate logbook pages under hf/candidate/pages. Start at
[Current verification](https://huggingface.co/spaces/DineshAI/y4uR1LFClc/tree/main/pages/current-verification).

## Repository layout

| Path | Purpose |
| --- | --- |
| repro/ | Claim verifiers, loss implementations, model definitions, and fixed entrypoint |
| jobs/ | CPU job entrypoints and node configuration |
| .openresearch/artifacts/ | Pre-registered contracts, source audit, job logs, raw summaries, and manifests |
| hf/candidate/ | Current evidence logbook intended for evaluation |
| hf/judged/ | Immutable historical judged Space snapshot |
| publish/ | Candidate logbook builder and upload checks |
| branch-audit.md | Complete mapping from the original branch names to the clean branch contract |

## Fixed reproduction command

Every node uses the same entrypoint:

    uv run --frozen repro/run_all.py

What a node does is decided only by repro/config.py committed on that ref. A result
is therefore identified by the pair (repository, git ref), not by an unrecorded
flag or environment variable.

All research compute was CPU-only. Hugging Face cpu-upgrade jobs used an 8-vCPU
cgroup quota on AMD EPYC 7R13; local 8-core CPU runs supplied the exact symbolic
certificate and real-loss geometry after the long remote nodes were terminated.
The runner asserts that no GPU is available.

## Branch contract

The original repository used short exp/ names and included a stale master branch.
The cleaned repository uses descriptive prefixes:

- main — documentation landing page and the current integrated evidence.
- experiment/collapse-* — activation and initialization controls for Claims 1 and 4.
- experiment/orbits — SimCLR training and augmentation-orbit geometry for Claim 6.
- experiment/pretrained*, including v2 and v3 — official checkpoint and loss-geometry
  work for Claims 2, 3, and 5.
- release/released-array-audit — first independent analysis of the authors' arrays.
- release/released-array-audit-v2 — released-array key-convention and assumption audit.

The exact old-to-new mapping, tip subjects, and deletion of the duplicate master
branch are recorded in [branch-audit.md](branch-audit.md).

## Reproduction limitations

- The independent collapse runs are shortened to one epoch for the configurations
  that completed. The paper-scale behaviour is checked separately against the
  authors' released 100-epoch arrays.
- Claim 1's PSD conclusion uses the theorem's stated MSE premise. The paper's own
  negative-cosine objective does not satisfy that premise in the ambient space.
- Claim 3's numerical curvature table is intentionally scoped to positive-definite
  metric cases where the tensor calculation is meaningful and reproducible.
- Claim 4 reproduces the reported correlations, while documenting the resolution
  limit of the paper's 20-step float32 power iteration.
- Claim 5 counts InfoNCE and SimCLR as one distinct loss because the implementation
  is identical by construction.
- Claim 6's exact 21.85x result comes from the authors' released arrays; the
  independent training run is corroboration, not a second 21.85x reproduction.

These are documented deviations and boundaries, not omissions. See the full
[limitations and deviations](https://huggingface.co/spaces/DineshAI/y4uR1LFClc/tree/main/pages/limitations-and-deviations)
page.

## Citation

    @article{chaudhry2026geometry,
      title={The Geometry of Projection Heads: Conditioning, Invariance, and Collapse},
      author={Chaudhry, Faris},
      journal={arXiv preprint arXiv:2605.17180},
      year={2026},
      eprint={2605.17180},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }

Please cite the paper when using this reproduction, its analysis, or its released
artifacts.

## Thank you

Thank you to Faris Chaudhry for making the paper, source material, implementation,
and released arrays available. Those public artifacts make it possible to audit the
claims at the level of equations, code, and numerical evidence.

## Attribution and repository history

- Current repository: MachineLearning-Nerd/icml26-geometry-projection-heads-conditioning-invariance-collapse
- Original repository: icml26-repro-y4uR1LFClc-the-geometry-of-projection-heads-conditioning-invariance-and-collapse
- OpenReview/forum identifier: y4uR1LFClc
- Approved commit author and committer identity:
  MachineLearning-Nerd <37579156+MachineLearning-Nerd@users.noreply.github.com>

The original name and historical judged bytes are retained in the audit trail; the
public repository name and active branch names are cleaned for readability.
