#!/usr/bin/env python3
"""Read a Hugging Face job's status and log tail.  Orchestration only — no research compute.

`fetch_job_logs` keeps streaming while a job is RUNNING, so the read is bounded by a
wall-clock alarm and whatever arrived before it fires is printed.

Usage:  <hf-python> joblog.py <job_id> [tail_lines] [read_seconds]
"""
import signal
import sys

from huggingface_hub import HfApi, get_token

api, tok = HfApi(), get_token()
job_id = sys.argv[1]
tail = int(sys.argv[2]) if len(sys.argv) > 2 else 60
budget = int(sys.argv[3]) if len(sys.argv) > 3 else 25

info = api.inspect_job(job_id=job_id, token=tok)
print(f"### {job_id}  stage={info.status.stage}  msg={info.status.message}")
print(f"### flavor={getattr(info, 'flavor', None)}  created={getattr(info, 'created_at', None)}")

lines = []


class _Timeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
signal.alarm(budget)
try:
    for line in api.fetch_job_logs(job_id=job_id, token=tok):
        lines.append(line)
except _Timeout:
    pass
except Exception as exc:
    print(f"### log stream ended: {type(exc).__name__}: {exc}")
finally:
    signal.alarm(0)

print(f"### {len(lines)} log lines, showing last {tail}")
print("\n".join(lines[-tail:]))
