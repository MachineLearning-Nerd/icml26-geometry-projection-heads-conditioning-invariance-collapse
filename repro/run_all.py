"""Fixed reproduction entrypoint.  Every experiment-tree node runs exactly:

    uv run repro/run_all.py

with no arguments and no environment overrides.  What the node does is decided by
`repro/config.py`, which is committed on the node's own branch.
"""

import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import threads  # noqa: E402  (must precede numpy/torch)

import numpy as np  # noqa: E402
import torch  # noqa: E402

threads.pin_torch(torch)

from repro import config  # noqa: E402


def provenance():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:
        sha = os.environ.get("REPRO_GIT_SHA", "(not a git checkout)")
    info = {
        "git_sha": sha,
        "stage": config.STAGE,
        "cgroup_cpu_quota": threads.NUM_THREADS,
        "os_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "hf_job_flavor": os.environ.get("HF_JOB_FLAVOR", "cpu-upgrade"),
    }
    print("PROVENANCE " + json.dumps(info), flush=True)
    assert not torch.cuda.is_available(), "GPU detected; this campaign is CPU-only"
    return info


def main():
    t0 = time.perf_counter()
    provenance()
    stage = config.STAGE

    if stage == "smoke":
        from repro import collapse
        _, ok = collapse.verify_machinery()
        for act in ("linear", "relu", "gelu", "swish"):
            collapse.run(act, "collapsed", epochs=1, batch_size=64, seed=0,
                         hess_batches=2, hess_samples=2, max_steps_per_epoch=3)
        print(f"SMOKE machinery_ok={ok}", flush=True)

    elif stage == "collapse":
        from repro import collapse
        collapse.verify_machinery(activation=config.COLLAPSE["activation"])
        collapse.run(**config.COLLAPSE)

    elif stage == "released":
        from repro import released
        released.main()

    elif stage == "orbits":
        from repro import orbits
        orbits.main(**config.ORBITS)

    elif stage == "pretrained":
        from repro import pretrained
        pretrained.main()

    else:
        raise SystemExit(f"unknown stage {stage!r}")

    print(f"RUN_ALL COMPLETE stage={stage} in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
