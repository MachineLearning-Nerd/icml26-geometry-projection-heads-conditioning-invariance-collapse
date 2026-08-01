#!/usr/bin/env python3
"""Publish the candidate logbook to the existing Space, text-only, then verify.

Guards, in order:
  1. the judged file set must be a subset of the candidate tree (paths), and the
     rejected-baseline page's judged bytes must survive verbatim;
  2. every uploaded path must match a text-only allowlist — no binary is ever written;
  3. no file may contain anything that looks like a token;
  4. the live Space HEAD must still equal the expected parent, so a concurrent session's
     commit is never silently clobbered;
  5. after upload, every published path is re-downloaded **at the new revision** and
     compared by SHA-256 against the staged bytes.

Usage:
    python3 publish/upload.py --dry-run      # gates + manifest only, no write
    python3 publish/upload.py --publish --expect-parent <sha>
"""
import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "hf", "candidate")
JUDGED = os.path.join(ROOT, "hf", "judged")
REPO_ID = "DineshAI/y4uR1LFClc"

# Text-only. Existing binaries in the judged revision are left untouched on the Space:
# they are neither re-uploaded nor deleted, because upload_folder without delete_patterns
# only adds and overwrites the paths it is given.
ALLOW_SUFFIXES = (".md", ".json", ".csv", ".txt", ".py")
ALLOW_EXACT = ("logbook.json",)

TOKEN_PATTERNS = [
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bapi_org_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\b(authorization|bearer)\b\s*[:=]\s*\S{16,}"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def relfiles(root):
    out = []
    for dp, _dn, fn in os.walk(root):
        if ".cache" in dp:
            continue
        for f in fn:
            out.append(os.path.relpath(os.path.join(dp, f), root))
    return sorted(out)


def gate():
    judged = set(relfiles(JUDGED))
    cand = set(relfiles(OUT))
    missing = sorted(judged - cand)

    vr = "pages/verification-run/page.md"
    preserved = open(os.path.join(OUT, vr), "rb").read().endswith(
        open(os.path.join(JUDGED, vr), "rb").read())

    allow, skipped = [], []
    for p in sorted(cand):
        if p in ALLOW_EXACT or p.endswith(ALLOW_SUFFIXES):
            allow.append(p)
        else:
            skipped.append(p)

    secrets = []
    for p in allow:
        try:
            text = open(os.path.join(OUT, p), encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for pat in TOKEN_PATTERNS:
            if pat.search(text):
                secrets.append(p)
                break

    manifest = {p: sha256(os.path.join(OUT, p)) for p in allow}
    report = {
        "judged_files": len(judged),
        "candidate_files": len(cand),
        "old_not_in_new": missing,
        "judged_set_is_subset": not missing,
        "rejected_baseline_bytes_preserved_verbatim": preserved,
        "upload_allowlist": allow,
        "not_uploaded_left_untouched_on_space": skipped,
        "files_with_possible_secrets": secrets,
        "manifest_sha256": manifest,
    }
    ok = (not missing) and preserved and not secrets
    return ok, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--expect-parent", default=None)
    args = ap.parse_args()

    ok, report = gate()
    os.makedirs(os.path.join(ROOT, ".openresearch", "artifacts"), exist_ok=True)
    with open(os.path.join(ROOT, ".openresearch", "artifacts", "upload_manifest.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items() if k != "manifest_sha256"}, indent=1))
    if not ok:
        print("GATE FAILED", file=sys.stderr)
        return 1
    print(f"GATE PASSED — {len(report['upload_allowlist'])} text files to upload")
    if not args.publish:
        return 0

    from huggingface_hub import HfApi, get_token, hf_hub_download
    token = get_token()
    api = HfApi()
    who = api.whoami(token=token)
    print(f"authenticated as {who['name']}")

    live = api.repo_info(REPO_ID, repo_type="space", token=token).sha
    print(f"live HEAD before upload: {live}")
    if args.expect_parent and live != args.expect_parent:
        print(f"ABORT: live HEAD {live} != expected parent {args.expect_parent}; "
              "another session may have published. Inspect before overwriting.",
              file=sys.stderr)
        return 1

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    api.upload_folder(
        repo_id=REPO_ID, repo_type="space", folder_path=OUT,
        allow_patterns=report["upload_allowlist"], token=token,
        commit_message="Replace toy 4x4 verification with full-scale evidence: exact "
                       "float64 effective Hessians of trained networks, the authors' "
                       "released full-scale arrays, and official pretrained SSL heads",
    )
    new = api.repo_info(REPO_ID, repo_type="space", token=token).sha
    print(f"published revision: {new}")

    bad = []
    for p, want in report["manifest_sha256"].items():
        got = sha256(hf_hub_download(REPO_ID, p, repo_type="space", revision=new,
                                     token=token, force_download=True))
        if got != want:
            bad.append(p)
    print(json.dumps({"published_revision": new, "verified_paths":
                      len(report["manifest_sha256"]), "hash_mismatches": bad}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
