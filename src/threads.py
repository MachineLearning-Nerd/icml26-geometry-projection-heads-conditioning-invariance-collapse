"""Pin BLAS/OpenMP thread pools to the container's real CPU quota.

Must be imported before numpy or torch: both read these env vars at import time.
Inside a Hugging Face job container ``os.cpu_count()`` reports the *host's* core
count (32-64) while the cgroup grants only a few vCPUs, so unpinned pools spawn
far more threads than runnable cores and spin-contention dominates.
"""

import os


def cgroup_quota() -> int:
    """Cores actually grantable to this process, from cgroup v2, then v1, then affinity."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(float(quota) / float(period)))
    except OSError:
        pass
    try:
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            return max(1, quota // period)
    except OSError:
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def pin(n: int | None = None) -> int:
    n = n or cgroup_quota()
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(n)
    return n


NUM_THREADS = pin()


def pin_torch(torch) -> int:
    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(1)
    return NUM_THREADS
