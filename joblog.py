#!/usr/bin/env python3
"""Read a Hugging Face job's status and log tail.  Orchestration only — no research compute.

Usage:  <hf-python> joblog.py <job_id> [tail_lines]
"""
import sys

from huggingface_hub import HfApi, get_token

api, tok = HfApi(), get_token()
job_id = sys.argv[1]
tail = int(sys.argv[2]) if len(sys.argv) > 2 else 60

info = api.inspect_job(job_id=job_id, token=tok)
print(f"### {job_id}  stage={info.status.stage}  msg={info.status.message}")
print(f"### flavor={getattr(info, 'flavor', None)}  created={getattr(info, 'created_at', None)}")
lines = []
try:
    for line in api.fetch_job_logs(job_id=job_id, token=tok):
        lines.append(line)
except Exception as exc:  # log stream can close abruptly on finished jobs
    print(f"### log stream ended: {type(exc).__name__}: {exc}")
print(f"### {len(lines)} log lines, showing last {tail}")
print("\n".join(lines[-tail:]))
