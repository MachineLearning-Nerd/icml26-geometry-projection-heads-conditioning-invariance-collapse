"""CIFAR-10 staging.

torchvision's downloader streams from www.cs.toronto.edu at ~180 kB/s from inside a
Hugging Face job and emits a tqdm line per chunk, which floods the job log that is our
only durable output channel.  Fetching the archive once, quietly, into ./data lets
torchvision find and verify it (it checks the MD5) and just extract.
"""

import os
import time
import urllib.request

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE = "cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"


def stage(root="./data"):
    os.makedirs(root, exist_ok=True)
    dest = os.path.join(root, ARCHIVE)
    if os.path.exists(os.path.join(root, "cifar-10-batches-py", "data_batch_1")):
        print("DATA cifar10 already extracted", flush=True)
        return root
    if not os.path.exists(dest):
        t0 = time.perf_counter()
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        print(f"DATA fetched {ARCHIVE} ({os.path.getsize(dest):,} B) "
              f"in {time.perf_counter() - t0:.1f}s", flush=True)
    return root
