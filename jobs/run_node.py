# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Hugging Face `cpu-upgrade` job bootstrap.

Clones this reproduction repository at a pinned git ref and runs the fixed
reproduction command inside it against the committed `uv.lock`:

    uv run --frozen repro/run_all.py

The bootstrap adds nothing scientific: which experiment runs is decided entirely by
`repro/config.py` on the cloned ref, so a job is fully identified by (repo, ref).

    python run_node.py <git-ref>
"""

import os
import subprocess
import sys

REPO = ("https://github.com/MachineLearning-Nerd/"
        "icml26-repro-y4uR1LFClc-the-geometry-of-projection-heads-conditioning-invariance-and-collapse")

ref = sys.argv[1] if len(sys.argv) > 1 else "main"
subprocess.run(["git", "clone", "--quiet", REPO, "repo"], check=True)
subprocess.run(["git", "-C", "repo", "checkout", "--quiet", ref], check=True)
sha = subprocess.run(["git", "-C", "repo", "rev-parse", "HEAD"],
                     capture_output=True, text=True, check=True).stdout.strip()
print(f"BOOTSTRAP ref={ref} sha={sha}", flush=True)

env = dict(os.environ, REPRO_GIT_SHA=sha, UV_INDEX_STRATEGY="unsafe-best-match")
sys.exit(subprocess.run(["uv", "run", "--frozen", "repro/run_all.py"],
                        cwd="repo", env=env).returncode)
