"""The one place an experiment-tree node differs from its parent.

The run command is fixed (`uv run repro/run_all.py`) and identical on every node;
every variant is a committed edit to this file on the node's own git branch.  Nothing
here may be overridden by an environment variable or a command-line flag.
"""

# Which stage this node runs.  One of:
#   "smoke"      cheap end-to-end validation of the Hessian machinery + a few steps
#   "collapse"   Claims 1 & 4 — effective-Hessian spectrum during real SSL training
#   "released"   independent re-analysis of the authors' released full-scale arrays
#   "orbits"     Claim 6 — SimCLR pretraining then augmentation-orbit geometry
#   "pretrained" Claims 2, 3 & 5 — geometry of official SSL checkpoints with heads
STAGE = "pretrained"

# --- stage "collapse" -------------------------------------------------------
COLLAPSE = {
    "activation": "swish",   # linear | relu | gelu | swish
    "init": "collapsed",     # collapsed | normal
    "epochs": 15,
    "batch_size": 256,
    "seed": 0,
    "lr": 0.05,
    "hess_batches": 5,       # matches the paper: first few batches of each epoch
    "hess_samples": 8,       # exact 512x512 Hessians per tracked batch
    "use_bn": False,         # the paper's Figure 3/4 setting: BatchNorm removed
}

# --- stage "orbits" ---------------------------------------------------------
ORBITS = {
    "epochs": 50,            # the paper's SimCLR pretraining budget
    "batch_size": 512,
    "lr": 1e-3,
    "temperature": 0.1,
    "seed": 0,
    "num_classes": 5,        # Appendix C.3: 5 classes, 3 images each, 12 rotations
    "images_per_class": 3,
    "num_rotations": 12,
}
