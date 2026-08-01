# Visibility matrix

A claim can be scientifically strong and still fail an evaluation because the evidence
cannot be found. This page is the checklist: for every claim, whether an evaluator
starting from [the index](#/index) can reach each required item **without** any knowledge
of the reproduction repository's internals.

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [link](#/claim-1-theorem-4-1) | yes | yes | yes | yes | yes | yes | **BLOCKED** |
| 2 | [link](#/claim-2-theorem-3-1) | yes | yes | yes | yes | yes | yes | **BLOCKED** |
| 3 | [link](#/claim-3-proposition-3-3) | yes | yes | yes | yes | yes | yes | **BLOCKED** |
| 4 | [link](#/claim-4-figure-3) | yes | yes | yes | yes | yes | yes | **VERIFIED** |
| 5 | [link](#/claim-5-section-6-generality) | yes | yes | yes | yes | yes | yes | **BLOCKED** |
| 6 | [link](#/claim-6-orbit-compression) | yes | yes | yes | yes | yes | yes | **VERIFIED** |

## What "visible" means here

- **Code visible** — the verifying source is either quoted on the claim page or named by
  file and reachable at a stated git revision of the public reproduction repository.
- **Data inline** — the numbers that decide the verdict appear in a table on the claim
  page, not only in a downloadable file.
- **Raw link** — a downloadable CSV or JSON on this Space, linked from the claim page.
- **Checker** — an executable verifier that exits non-zero when its evidence fails.
- **Control** — a negative control that fails for the intended reason.
- **Exact claim tested** — the quantity measured is the one the paper's sentence names,
  not a nearby proxy.

Every claim page also carries the fixed command, the pinned environment, the git
revision, the seeds, and the `cpu-upgrade` allocation and runtime.
