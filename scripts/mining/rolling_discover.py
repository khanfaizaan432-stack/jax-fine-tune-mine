#!/usr/bin/env python3
"""Rolling repository discovery for JAXFixBench.

This writes the same `repo_candidates_final.jsonl` contract expected by
`mine_jaxfixbench.py plan`, but rotates GitHub search pages and avoids repos
seen in previous workflow runs when a cache-backed seen file is provided.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

BUCKETS: dict[str, list[str]] = {
    "jax_general": [
        "jax in:name,description,readme language:Python stars:>5 archived:false fork:false",
        "\"import jax\" in:readme language:Python stars:>5 archived:false fork:false",
    ],
    "jax_jit": [
        "\"jax.jit\" in:readme language:Python stars:>3 archived:false fork:false",
        "\"@jax.jit\" in:readme language:Python stars:>3 archived:false fork:false",
        "\"jit\" \"jax\" in:readme language:Python stars:>3 archived:false fork:false",
    ],
    "jax_vmap": [
        "\"jax.vmap\" in:readme language:Python stars:>3 archived:false fork:false",
        "\"vmap\" \"jax\" in:readme language:Python stars:>3 archived:false fork:false",
    ],
    "flax": [
        "\"flax.linen\" in:readme language:Python stars:>3 archived:false fork:false",
        "flax in:name,description,readme language:Python stars:>3 archived:false fork:false",
    ],
    "optax": [
        "optax in:name,description,readme language:Python stars:>3 archived:false fork:false",
        "\"import optax\" in:readme language:Python stars:>3 archived:false fork:false",
    ],
    "xla_stablehlo": [
        "xla in:name,description,readme language:Python stars:>3 archived:false fork:false",
        "stablehlo in:name,description,readme stars:>1 archived:false fork:false",
        "\"XLA\" \"JAX\" in:readme language:Python stars:>3 archived:false fork:false",
    ],
}


def gh_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PAT") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def read_seen(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data}
        if isinstance(data, dict):
            return {str(x) for x in data.get("repos", [])}
    except Exception:
        pass
    return set()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_repo(item: dict[str, Any], bucket: str, query: str) -> dict[str, Any]:
    lic = item.get("license") or {}
    return {
        "repo": item["full_name"],
        "html_url": item.get("html_url"),
        "clone_url": item.get("clone_url"),
        "description": item.get("description"),
        "stars": item.get("stargazers_count"),
        "forks": item.get("forks_count"),
        "language": item.get("language"),
        "size_kb": item.get("size") or 0,
        "license_key": lic.get("key"),
        "license_name": lic.get("name"),
        "updated_at": item.get("updated_at"),
        "pushed_at": item.get("pushed_at"),
        "default_branch": item.get("default_branch"),
        "buckets": [bucket],
        "matched_queries": [query],
        "source": "github_repository_search_rolling_seen_filtered",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seen-file", default=None)
    p.add_argument("--target-per-bucket", type=int, default=50)
    p.add_argument("--max-pages-per-query", type=int, default=2)
    p.add_argument("--run-offset", type=int, default=0)
    p.add_argument("--sleep-seconds", type=float, default=3.0)
    p.add_argument("--fallback-to-seen", action="store_true", help="Use already-seen repos only if fresh repos are insufficient.")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen_path = Path(args.seen_file) if args.seen_file else None
    seen_before = read_seen(seen_path)
    repo_map: dict[str, dict[str, Any]] = {}
    skipped_seen: list[str] = []
    errors: list[dict[str, Any]] = []
    fresh_by_bucket: dict[str, list[str]] = defaultdict(list)
    max_start_page = 8
    page_offset = (args.run_offset % max_start_page) * args.max_pages_per_query

    for bucket, queries in BUCKETS.items():
        bucket_fresh: set[str] = set()
        bucket_seen_candidates: list[dict[str, Any]] = []
        for query_i, query in enumerate(queries):
            if len(bucket_fresh) >= args.target_per_bucket:
                break
            start_page = 1 + ((page_offset + query_i * args.max_pages_per_query) % max_start_page)
            pages = list(range(start_page, start_page + args.max_pages_per_query))
            if 1 not in pages:
                pages.append(1)
            for page in pages:
                if len(bucket_fresh) >= args.target_per_bucket:
                    break
                resp = requests.get(
                    "https://api.github.com/search/repositories",
                    headers=gh_headers(),
                    params={"q": query, "sort": "stars", "order": "desc", "per_page": 50, "page": page},
                    timeout=40,
                )
                if resp.status_code in {403, 429}:
                    errors.append({"bucket": bucket, "query": query, "page": page, "status": resp.status_code, "body": resp.text[:300]})
                    continue
                resp.raise_for_status()
                for item in resp.json().get("items", []):
                    repo = item["full_name"]
                    if (item.get("size") or 0) > 300_000:
                        continue
                    row = normalize_repo(item, bucket, query)
                    if repo in seen_before:
                        skipped_seen.append(repo)
                        bucket_seen_candidates.append(row)
                        continue
                    if repo not in repo_map:
                        repo_map[repo] = row
                    else:
                        repo_map[repo]["buckets"] = sorted(set(repo_map[repo].get("buckets", [])) | {bucket})
                        repo_map[repo]["matched_queries"] = sorted(set(repo_map[repo].get("matched_queries", [])) | {query})
                    bucket_fresh.add(repo)
                    fresh_by_bucket[bucket].append(repo)
                    if len(bucket_fresh) >= args.target_per_bucket:
                        break
                time.sleep(args.sleep_seconds)
        if args.fallback_to_seen and len(bucket_fresh) < args.target_per_bucket:
            for row in bucket_seen_candidates:
                repo = row["repo"]
                if repo in repo_map:
                    continue
                row["source"] = "github_repository_search_rolling_seen_fallback"
                repo_map[repo] = row
                bucket_fresh.add(repo)
                if len(bucket_fresh) >= args.target_per_bucket:
                    break
        print(f"{bucket}: fresh={len(set(fresh_by_bucket[bucket]))} selected_total_so_far={len(repo_map)}")

    rows = sorted(repo_map.values(), key=lambda r: r.get("stars") or 0, reverse=True)
    write_jsonl(out / "repo_candidates_checkpoint.jsonl", rows)
    write_jsonl(out / "repo_candidates_final.jsonl", rows)
    seen_after = sorted(seen_before | {row["repo"] for row in rows})
    if seen_path:
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(seen_path, {"repos": seen_after})
    report = {
        "selected_repos": len(rows),
        "seen_before": len(seen_before),
        "seen_after": len(seen_after),
        "skipped_seen_count": len(set(skipped_seen)),
        "fresh_by_bucket": {k: len(set(v)) for k, v in fresh_by_bucket.items()},
        "page_offset": page_offset,
        "run_offset": args.run_offset,
        "errors": errors,
    }
    write_json(out / "rolling_discovery_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
