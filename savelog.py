#!/usr/bin/env python3
"""Persist a finished job's stdout to durable evidence.  Orchestration only.

Job filesystems are discarded when the job ends, so the log *is* the raw output.
Structured `KEY {json}` lines are also split out into a machine-readable JSONL file.

Usage:  <hf-python> savelog.py <job_id> <artifact_dir>
"""
import json
import os
import re
import sys

from huggingface_hub import HfApi, get_token

api, tok = HfApi(), get_token()
job_id, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)

info = api.inspect_job(job_id=job_id, token=tok)
lines = list(api.fetch_job_logs(job_id=job_id, token=tok))

# drop tqdm/progress carriage-return spam so the durable log stays readable
keep = [ln for ln in lines if not re.search(r"\|[\s▏▎▍▌▋▊▉█]*\|", ln)]
with open(os.path.join(outdir, f"{job_id}.log"), "w") as f:
    f.write("\n".join(keep))

records = []
for ln in keep:
    m = re.match(r"^([A-Z][A-Z0-9_]*)\s+(\{.*\})\s*$", ln)
    if m:
        try:
            records.append({"key": m.group(1), **json.loads(m.group(2))})
        except json.JSONDecodeError:
            pass
with open(os.path.join(outdir, f"{job_id}.jsonl"), "w") as f:
    for r in records:
        f.write(json.dumps(r, sort_keys=True) + "\n")

print(json.dumps({
    "job_id": job_id,
    "stage": str(info.status.stage),
    "flavor": str(getattr(info, "flavor", None)),
    "created_at": str(getattr(info, "created_at", None)),
    "log_lines_kept": len(keep),
    "structured_records": len(records),
    "keys": sorted({r["key"] for r in records}),
}, indent=1))
