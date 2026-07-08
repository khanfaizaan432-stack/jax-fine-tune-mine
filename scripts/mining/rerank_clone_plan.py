#!/usr/bin/env python3
"""Build a clone plan using JAX relevance, not stars alone."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

GOOD_TERMS = {
    "jax": 8,
    "flax": 7,
    "optax": 7,
    "xla": 5,
    "stablehlo": 5,
    "pjit": 5,
    "pmap": 5,
    "vmap": 5,
    "jit": 4,
    "lax": 4,
    "jaxlib": 4,
    "equinox": 4,
    "orbax": 4,
    "haiku": 4,
    "chex": 3,
    "jaxtyping": 3,
}

BAD_TERMS = {
    "awesome": 16,
    "paper-list": 12,
    "papers": 8,
    "survey": 7,
    "course": 6,
    "tutorial": 5,
    "examples": 3,
    "tensorflow": 4,
    "pytorch": 4,
    "torch": 4,
    "cuda": 3,
}

ALLOWED_LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "mpl-2.0", None}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def term_score(text: str) -> tuple[int, list[str], list[str]]:
    low = text.lower()
    good_hits: list[str] = []
    bad_hits: list[str] = []
    score = 0
    for term, value in GOOD_TERMS.items():
        if re.search(r"\b" + re.escape(term) + r"\b", low):
            score += value
            good_hits.append(term)
    for term, value in BAD_TERMS.items():
        if re.search(r"\b" + re.escape(term) + r"\b", low):
            score -= value
            bad_hits.append(term)
    return score, good_hits, bad_hits


def score_repo(row: dict[str, Any]) -> dict[str, Any]:
    repo = row.get("repo", "")
    desc = row.get("description") or ""
    buckets = row.get("buckets") or []
    queries = row.get("matched_queries") or []
    text = " ".join([repo, desc, " ".join(buckets), " ".join(queries)])
    score, good_hits, bad_hits = term_score(text)
    stars = row.get("stars") or 0
    size_kb = row.get("size_kb") or 0
    score += min(10, int(math.log10(stars + 1) * 4))
    score += min(8, len(set(buckets)) * 2)
    if row.get("language") == "Python":
        score += 3
    if size_kb > 150_000:
        score -= 12
    elif size_kb > 80_000:
        score -= 5
    if row.get("license_key") not in ALLOWED_LICENSES:
        score -= 8
    out = dict(row)
    out["jax_relevance_score"] = score
    out["jax_relevance_good_hits"] = good_hits
    out["jax_relevance_bad_hits"] = bad_hits
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--per-bucket", type=int, default=50)
    p.add_argument("--min-score", type=int, default=6)
    args = p.parse_args()

    out = Path(args.out_dir)
    rows = [score_repo(r) for r in read_jsonl(out / "repo_candidates_final.jsonl")]
    kept = [r for r in rows if r.get("jax_relevance_score", 0) >= args.min_score]
    kept = [r for r in kept if (r.get("size_kb") or 0) <= 150_000]
    kept = [r for r in kept if r.get("license_key") in ALLOWED_LICENSES]

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        for bucket in row.get("buckets", []):
            by_bucket[bucket].append(row)

    selected: dict[str, dict[str, Any]] = {}
    selected_by_bucket: dict[str, list[str]] = {}
    for bucket, bucket_rows in sorted(by_bucket.items()):
        ranked = sorted(bucket_rows, key=lambda r: (r.get("jax_relevance_score", 0), r.get("stars") or 0), reverse=True)
        chosen = ranked[: args.per_bucket]
        selected_by_bucket[bucket] = [r["repo"] for r in chosen]
        for row in chosen:
            selected[row["repo"]] = row

    selected_rows = sorted(selected.values(), key=lambda r: (r.get("jax_relevance_score", 0), r.get("stars") or 0), reverse=True)
    output = {
        "selected_total_unique": len(selected_rows),
        "selection_method": "jax_relevance_score_then_stars",
        "min_score": args.min_score,
        "selected_by_bucket": selected_by_bucket,
        "repos": selected_rows,
    }
    write_json(out / "repo_clone_plan.json", output)
    write_json(out / "repo_rerank_report.json", {
        "candidate_repos": len(rows),
        "kept_after_score_license_size": len(kept),
        "selected_total_unique": len(selected_rows),
        "top_repos": [{"repo": r["repo"], "score": r.get("jax_relevance_score"), "stars": r.get("stars"), "bad_hits": r.get("jax_relevance_bad_hits")} for r in selected_rows[:40]],
    })
    print(json.dumps(output | {"repos": output["repos"][:5]}, indent=2))


if __name__ == "__main__":
    main()
