#!/usr/bin/env python3
"""Verify the scoped geometry dossier and its live normalized GitHub state."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANONICAL = ("MachineLearning-Nerd", "MachineLearning-Nerd@users.noreply.github.com")
REPOSITORY = "icml26-geometry-projection-heads-conditioning-invariance-collapse"
EXPECTED_OVERALL = "SCOPED_CLAIMS_1_TO_6_VERIFIED_WITH_EXPLICIT_BUDGET_AND_ASSUMPTION_BOUNDARIES"
EXPECTED_BOUNDARY = "NO_FULL_BUDGET_INDEPENDENT_TRAINING_RELEASED_ARRAY_AND_CHECKPOINT_DEPENDENCIES"
EXPECTED_BRANCHES = {
    "main",
    "experiment/collapse-gelu-collapsed",
    "experiment/collapse-linear-collapsed",
    "experiment/collapse-relu-collapsed",
    "experiment/collapse-swish-collapsed",
    "experiment/collapse-swish-normal",
    "experiment/orbits",
    "experiment/pretrained",
    "experiment/pretrained-v2",
    "experiment/pretrained-v3",
    "release/released-array-audit",
    "release/released-array-audit-v2",
}
REQUIRED_PATHS = [
    "README.md",
    "STATUS.md",
    "CLAIM_EVIDENCE.md",
    "SOURCE_AUDIT.md",
    "ENVIRONMENT.md",
    "REPORT.md",
    "branch-audit.md",
    "claims.json",
    "reproduction_verdicts.json",
    "AUTONOMOUS_STATE.json",
    "EVIDENCE_MANIFEST.json",
    "CITATION.cff",
    "AUTHOR_THANK_YOU.md",
    "verify_final.py",
]
EXPECTED_STATUSES = ["verified_scoped"] * 6


def fail(message: str) -> None:
    print(f"FINAL_AUDIT=FAILED {message}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def run(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        fail(f"command failed: {' '.join(args)}\n{result.stderr.strip()}")
    return result.stdout


def current_bytes(path: str) -> bytes:
    local = ROOT / path
    if local.exists():
        return local.read_bytes()
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        fail(f"main path unavailable: {path}")
    return result.stdout


def git_object_bytes(ref: str, path: str) -> bytes:
    for candidate in (ref, f"origin/{ref}"):
        result = subprocess.run(["git", "show", f"{candidate}:{path}"], cwd=ROOT, capture_output=True, check=False)
        if result.returncode == 0:
            return result.stdout
    fail(f"branch evidence unavailable: {ref}:{path}")
    return b""


def current_json(path: str) -> object:
    try:
        return json.loads(current_bytes(path))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    return None


def verify_manifest() -> None:
    manifest = current_json("EVIDENCE_MANIFEST.json")
    require(isinstance(manifest, dict), "manifest is not an object")
    require(manifest.get("schema_version") == 1, "manifest schema changed")
    require(manifest.get("hash_algorithm") == "sha256", "manifest algorithm changed")
    require(manifest.get("overall_verdict") == EXPECTED_OVERALL, "manifest overall verdict changed")
    require(manifest.get("publication_allowed") is False, "manifest publication boundary changed")
    entries = manifest.get("entries", [])
    require(isinstance(entries, list) and entries, "manifest is empty")
    seen = set()
    for entry in entries:
        require(isinstance(entry, dict), "manifest entry is not an object")
        path = entry.get("path")
        expected = entry.get("sha256")
        key = (entry.get("ref"), path)
        require(isinstance(path, str), "manifest path missing")
        require(key not in seen, f"duplicate manifest entry: {key}")
        require(isinstance(expected, str) and len(expected) == 64, f"bad manifest hash: {key}")
        seen.add(key)
        raw = git_object_bytes(entry["ref"], path) if entry.get("ref") else current_bytes(path)
        require(hashlib.sha256(raw).hexdigest() == expected, f"manifest hash mismatch: {key}")


def main() -> None:
    origin = run("git", "config", "--get", "remote.origin.url").strip()
    require(origin in {f"https://github.com/MachineLearning-Nerd/{REPOSITORY}", f"https://github.com/MachineLearning-Nerd/{REPOSITORY}.git", f"git@github.com:MachineLearning-Nerd/{REPOSITORY}.git"}, f"unexpected origin: {origin}")
    require("ref: refs/heads/main\tHEAD" in run("git", "ls-remote", "--symref", "origin", "HEAD"), "origin/HEAD is not main")

    remote_heads = {}
    for line in run("git", "ls-remote", "--heads", "origin").splitlines():
        commit, ref = line.split("\t", 1)
        require(ref.startswith("refs/heads/"), f"unexpected remote ref: {ref}")
        remote_heads[ref.removeprefix("refs/heads/")] = commit
    require(set(remote_heads) == EXPECTED_BRANCHES, "remote branch set is not the 12-branch contract")
    require(not any(name.startswith("exp/") or name == "master" for name in remote_heads), "legacy branch remains live")

    local_heads = set(run("git", "for-each-ref", "--format=%(refname:strip=2)", "refs/heads").splitlines())
    require("main" in local_heads and local_heads <= EXPECTED_BRANCHES, "unexpected local branch name")
    require(not any("refs/original/" in ref for ref in run("git", "for-each-ref", "--format=%(refname)", "refs").splitlines()), "refs/original remains")

    identities = set()
    for line in run("git", "log", "--all", "--format=%an\t%ae\t%cn\t%ce").splitlines():
        if line.strip():
            identities.add(tuple(line.split("\t")))
    require(identities == {(CANONICAL[0], CANONICAL[1], CANONICAL[0], CANONICAL[1])}, f"non-canonical identity: {sorted(identities)}")
    require("co-authored-by:" not in run("git", "log", "--all", "--format=%B").lower(), "co-author trailer found")
    commit_count = int(run("git", "rev-list", "--count", "--all").strip())
    require(commit_count >= 31, f"unexpectedly short history: {commit_count}")

    for path in REQUIRED_PATHS:
        require((ROOT / path).exists(), f"required path missing: {path}")

    claims = current_json("claims.json")
    require(isinstance(claims, dict), "claims.json is not an object")
    require(claims.get("repository") == f"MachineLearning-Nerd/{REPOSITORY}", "claims repository changed")
    require(claims.get("overall_verdict") == EXPECTED_OVERALL, "claims overall verdict changed")
    require(claims.get("publication_boundary") == EXPECTED_BOUNDARY, "claims publication boundary changed")
    require(claims.get("publication_allowed") is False and claims.get("score_claim") is False and claims.get("official_author_endorsement") is False, "claims publication flags changed")
    rows = claims.get("claims", [])
    require(len(rows) == 6, "claims.json must contain six claims")
    statuses = [row.get("status") for row in rows]
    require(statuses == EXPECTED_STATUSES, f"claim statuses changed: {statuses}")
    official = claims.get("official_rejudge", {})
    require(official.get("score") == "5/12" and official.get("current_score_not_claimed") is True, "historical score boundary changed")

    reproduction = current_json("reproduction_verdicts.json")
    require(isinstance(reproduction, dict), "reproduction verdicts is not an object")
    require(reproduction.get("overall_verdict") == EXPECTED_OVERALL and reproduction.get("publication_boundary") == EXPECTED_BOUNDARY, "reproduction status changed")
    require(reproduction.get("publication_allowed") is False and reproduction.get("score_claim") is False and reproduction.get("official_author_endorsement") is False, "reproduction flags changed")
    reproduction_rows = reproduction.get("verdicts", {})
    require([reproduction_rows[str(i)].get("verdict") for i in range(1, 7)] == statuses, "reproduction claim rows changed")

    state = current_json("AUTONOMOUS_STATE.json")
    require(isinstance(state, dict), "state is not an object")
    require(state.get("phase") == "published_scoped_all_six_claims_with_explicit_boundaries", "state phase changed")
    require(state.get("branch_count") == 12 and state.get("default_branch") == "main", "state branch contract changed")
    require(state.get("verified_reachable_commits") == commit_count, "state commit count does not match")
    require(state.get("publication_allowed") is False and state.get("overall_verdict") == EXPECTED_OVERALL and state.get("publication_boundary") == EXPECTED_BOUNDARY, "state verdict changed")
    require(state.get("attribution") == {"name": CANONICAL[0], "email": CANONICAL[1]}, "state attribution changed")
    require(isinstance(state.get("last_known_git_commit"), str) and len(state["last_known_git_commit"]) == 40, "state checkpoint missing")

    readme = current_bytes("README.md").decode()
    status = current_bytes("STATUS.md").decode()
    report = current_bytes("REPORT.md").decode()
    for text, label in ((readme, "README"), (status, "STATUS"), (report, "REPORT")):
        for marker in (EXPECTED_OVERALL, "publication_allowed=false", "score_claim=false", "official_author_endorsement=false"):
            require(marker in text, f"{label} missing marker: {marker}")
    require("CLAIM_EVIDENCE.md" in readme and "reproduction_verdicts.json" in readme, "README missing dossier links")
    require("MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>" in readme, "README attribution changed")
    branch_audit = current_bytes("branch-audit.md").decode()
    require("There are no exp/ branches" in branch_audit and CANONICAL[1] in branch_audit, "branch audit boundary changed")

    verify_manifest()
    print(f"FINAL_AUDIT=VERIFIED branches={len(remote_heads)} commits={commit_count} claims=" + ",".join(f"{row['id']}:{row['status']}" for row in rows) + " historical_score=5/12 publication_allowed=false")


if __name__ == "__main__":
    main()
