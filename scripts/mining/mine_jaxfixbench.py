#!/usr/bin/env python3
"""JAXFixBench public-source mining scaffold.

This script is intentionally conservative: it mines public repository metadata,
clones a capped slice, extracts JAX-ish snippets, scores/cleans them, mines
common JAX patterns, and builds repo-grounded synthetic mutation cases.

It does not claim that mined snippets are automatically dataset-ready. Keep
provenance, inspect outputs, respect licenses, and execute final cases in a
proper JAX environment before release.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
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

KEEP_EXTS = {".py", ".md", ".rst", ".ipynb"}
BAD_PATH_BITS = [".git", "__pycache__", "site-packages", "dist-info", "node_modules", ".venv", "/venv/", "/data/", "/datasets/", "/checkpoints/"]
KEYWORDS = ["import jax", "from jax", "jax.jit", "@jax.jit", "jax.vmap", "vmap(", "jax.grad", "lax.scan", "lax.cond", "jax.random", "PRNGKey", "flax.linen", "import flax", "optax", "pmap", "pjit", "sharding", "stablehlo"]

POS = {
    "import_jax": r"\bimport\s+jax\b|\bfrom\s+jax\b",
    "jnp": r"\bimport\s+jax\.numpy\s+as\s+jnp\b|\bjnp\.",
    "jit": r"\bjax\.jit\b|@jax\.jit\b|@jit\b",
    "vmap": r"\bjax\.vmap\b|\bvmap\(",
    "grad": r"\bjax\.grad\b|\bjax\.value_and_grad\b",
    "lax": r"\bjax\.lax\b|\blax\.scan\b|\blax\.cond\b|\blax\.while_loop\b",
    "random": r"\bjax\.random\b|\bPRNGKey\b|\brandom\.split\b",
    "flax": r"\bflax\b|\bflax\.linen\b|\bimport\s+flax\b",
    "optax": r"\boptax\b|\bimport\s+optax\b",
    "parallel": r"\bpmap\b|\bpjit\b|\bsharding\b|\bPartitionSpec\b",
}
NEG = {
    "cuda_only": r"\bcuda\b|\bcutlass\b|\btriton\b",
    "tensorflow_only": r"\btensorflow\b|\btf\.",
    "pytorch_only": r"\btorch\b|\bpytorch\b",
    "generic_mlir_only": r"\bmlir\b|\bstablehlo\b|\bxla\b",
}
PATTERNS = {
    "functional_update_at": [r"\.at\[[^\]]+\]\.set\(", r"\.at\[[^\]]+\]\.add\("],
    "jit_static_argnums": [r"jax\.jit\([^)]*static_argnums\s*=", r"jax\.jit\([^)]*static_argnames\s*=", r"functools\.partial\(\s*jax\.jit[^)]*static_arg"],
    "plain_jit": [r"@jax\.jit\b", r"jax\.jit\("],
    "lax_cond": [r"lax\.cond\(", r"jax\.lax\.cond\("],
    "lax_scan": [r"lax\.scan\(", r"jax\.lax\.scan\("],
    "vmap": [r"jax\.vmap\(", r"\bvmap\("],
    "random_split": [r"jax\.random\.split\(", r"random\.split\(", r"PRNGKey\("],
    "pmap_pjit_sharding": [r"jax\.pmap\(", r"\bpmap\(", r"\bpjit\(", r"PartitionSpec", r"NamedSharding", r"Mesh\("],
    "optax_update": [r"optax\.", r"apply_updates\(", r"\.update\([^)]*params"],
    "flax_module": [r"flax\.linen", r"import flax\.linen as nn", r"class\s+\w+\(nn\.Module\)", r"@nn\.compact"],
}
MAX_PER_PATTERN = {"functional_update_at": 80, "lax_cond": 50, "lax_scan": 50, "jit_static_argnums": 70, "vmap": 100, "random_split": 100, "pmap_pjit_sharding": 80, "optax_update": 80, "flax_module": 80, "plain_jit": 80}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


def gh_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def discover(args: argparse.Namespace) -> None:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "repo_candidates_checkpoint.jsonl"
    final = out / "repo_candidates_final.jsonl"
    repo_map = {row["repo"]: row for row in read_jsonl(checkpoint)}
    for bucket, queries in BUCKETS.items():
        current = {repo for repo, row in repo_map.items() if bucket in row.get("buckets", [])}
        print(f"BUCKET {bucket}: starting with {len(current)}")
        for query in queries:
            if len(current) >= args.target_per_bucket:
                break
            for page in range(1, args.max_pages_per_query + 1):
                if len(current) >= args.target_per_bucket:
                    break
                resp = requests.get(
                    "https://api.github.com/search/repositories",
                    headers=gh_headers(),
                    params={"q": query, "sort": "stars", "order": "desc", "per_page": 50, "page": page},
                    timeout=40,
                )
                if resp.status_code in {403, 429}:
                    print(f"Rate limited: {resp.status_code} {resp.text[:300]}", file=sys.stderr)
                    write_jsonl(final, sorted(repo_map.values(), key=lambda r: r.get("stars") or 0, reverse=True))
                    return
                resp.raise_for_status()
                for item in resp.json().get("items", []):
                    repo = item["full_name"]
                    size_kb = item.get("size") or 0
                    if size_kb > 300_000:
                        continue
                    lic = item.get("license") or {}
                    if repo not in repo_map:
                        repo_map[repo] = {
                            "repo": repo,
                            "html_url": item.get("html_url"),
                            "clone_url": item.get("clone_url"),
                            "description": item.get("description"),
                            "stars": item.get("stargazers_count"),
                            "forks": item.get("forks_count"),
                            "language": item.get("language"),
                            "size_kb": size_kb,
                            "license_key": lic.get("key"),
                            "license_name": lic.get("name"),
                            "updated_at": item.get("updated_at"),
                            "pushed_at": item.get("pushed_at"),
                            "default_branch": item.get("default_branch"),
                            "buckets": [],
                            "matched_queries": [],
                            "source": "github_repository_search_star_ranked",
                        }
                    row = repo_map[repo]
                    row["buckets"] = sorted(set(row.get("buckets", [])) | {bucket})
                    row["matched_queries"] = sorted(set(row.get("matched_queries", [])) | {query})
                    current.add(repo)
                    if len(current) >= args.target_per_bucket:
                        break
                write_jsonl(checkpoint, sorted(repo_map.values(), key=lambda r: r.get("stars") or 0, reverse=True))
                print(bucket, len(current), "unique in bucket;", len(repo_map), "total")
                time.sleep(args.sleep_seconds)
    rows = sorted(repo_map.values(), key=lambda r: r.get("stars") or 0, reverse=True)
    write_jsonl(final, rows)
    print("saved", final, len(rows))


def plan(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    rows = read_jsonl(out / "repo_candidates_final.jsonl")
    allowed = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "mpl-2.0", None}
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (row.get("size_kb") or 0) > 150_000:
            continue
        if row.get("license_key") not in allowed:
            continue
        for bucket in row.get("buckets", []):
            by_bucket[bucket].append(row)
    selected: dict[str, dict[str, Any]] = {}
    selected_by_bucket: dict[str, list[str]] = {}
    for bucket, bucket_rows in by_bucket.items():
        chosen = sorted(bucket_rows, key=lambda r: r.get("stars") or 0, reverse=True)[: args.per_bucket]
        selected_by_bucket[bucket] = [row["repo"] for row in chosen]
        for row in chosen:
            selected[row["repo"]] = row
    output = {"selected_total_unique": len(selected), "selected_by_bucket": selected_by_bucket, "repos": list(selected.values())}
    write_json(out / "repo_clone_plan.json", output)
    print("selected unique repos:", len(selected))


def clone(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    root = out / "public_repos"
    root.mkdir(parents=True, exist_ok=True)
    plan_path = out / "repo_clone_plan.json"
    clone_log = out / "repo_clone_log.jsonl"
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    done = {row["repo"] for row in read_jsonl(clone_log) if row.get("status") in {"ok", "exists"}}
    new_ok = 0
    with clone_log.open("a", encoding="utf-8") as log:
        for i, row in enumerate(plan_data["repos"], 1):
            if new_ok >= args.max_clones:
                break
            repo = row["repo"]
            if repo in done:
                continue
            target = root / repo.replace("/", "__")
            if target.exists():
                log.write(json.dumps({"repo": repo, "status": "exists"}) + "\n")
                continue
            print("cloning", i, repo)
            res = subprocess.run(["git", "clone", "--depth", "1", row.get("clone_url") or f"https://github.com/{repo}.git", str(target)], capture_output=True, text=True, timeout=240)
            status = "ok" if res.returncode == 0 else "failed"
            log.write(json.dumps({"repo": repo, "status": status, "returncode": res.returncode, "stderr_tail": res.stderr[-1000:]}, ensure_ascii=False) + "\n")
            log.flush()
            new_ok += int(status == "ok")
            time.sleep(1)
    print("new successful clones:", new_ok)


def safe_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > 1_200_000:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def should_keep(path: Path, text: str) -> bool:
    path_s = str(path).replace("\\", "/").lower()
    if any(bit in path_s for bit in BAD_PATH_BITS):
        return False
    if path.suffix.lower() not in KEEP_EXTS:
        return False
    low = text.lower()
    return any(k.lower() in low for k in KEYWORDS)


def chunk_text(text: str, max_chars: int = 5000) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines()
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for line in lines:
        n += len(line) + 1
        cur.append(line)
        if n >= max_chars:
            joined = "\n".join(cur)
            if any(k.lower() in joined.lower() for k in KEYWORDS):
                chunks.append(joined)
            cur, n = [], 0
    if cur:
        joined = "\n".join(cur)
        if any(k.lower() in joined.lower() for k in KEYWORDS):
            chunks.append(joined)
    return chunks


def extract(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    root = out / "public_repos"
    rows: list[dict[str, Any]] = []
    for repo_dir in root.iterdir() if root.exists() else []:
        if not repo_dir.is_dir():
            continue
        repo = repo_dir.name.replace("__", "/")
        for path in repo_dir.rglob("*"):
            if not path.is_file():
                continue
            text = safe_text(path)
            if not text or not should_keep(path, text):
                continue
            rel = str(path.relative_to(repo_dir)).replace("\\", "/")
            for part_i, chunk in enumerate(chunk_text(text)):
                rows.append({
                    "id": f"repo_snippet_{len(rows):06d}",
                    "repo": repo,
                    "path": rel,
                    "part": part_i,
                    "ext": path.suffix.lower(),
                    "text": chunk[:7000],
                    "metadata": {"source": "public_github_clone", "license_note": "Check repository license before redistribution."},
                })
    write_jsonl(out / "raw_snippets.jsonl", rows)
    print("raw snippets:", len(rows))


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    text = row.get("text", "")
    pos = {name: bool(re.search(regex, text, re.I | re.M)) for name, regex in POS.items()}
    neg = {name: bool(re.search(regex, text, re.I | re.M)) for name, regex in NEG.items()}
    score = sum(pos.values()) * 3
    if pos.get("import_jax") and pos.get("jnp"):
        score += 4
    if pos.get("jit") or pos.get("vmap") or pos.get("lax"):
        score += 3
    if neg.get("pytorch_only") and not pos.get("import_jax"):
        score -= 6
    if neg.get("tensorflow_only") and not pos.get("import_jax"):
        score -= 6
    if neg.get("generic_mlir_only") and not (pos.get("import_jax") or pos.get("flax") or pos.get("optax")):
        score -= 5
    row["jax_score"] = score
    row["positive_hits"] = pos
    row["negative_hits"] = neg
    return row


def score(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    rows = [score_row(row) for row in read_jsonl(out / "raw_snippets.jsonl")]
    rows = sorted(rows, key=lambda r: r.get("jax_score") or 0, reverse=True)
    clean = [r for r in rows if r.get("jax_score", 0) >= args.min_score and (r["positive_hits"].get("import_jax") or r["positive_hits"].get("jnp") or r["positive_hits"].get("flax") or r["positive_hits"].get("optax"))]
    write_jsonl(out / "scored_snippets.jsonl", rows)
    write_jsonl(out / "clean_snippets.jsonl", clean)
    print("clean snippets:", len(clean))


def line_window(text: str, char_pos: int, window: int = 8) -> tuple[str, int]:
    lines = text.splitlines()
    running = 0
    hit = 0
    for i, line in enumerate(lines):
        running += len(line) + 1
        if running >= char_pos:
            hit = i
            break
    return "\n".join(lines[max(0, hit - window): min(len(lines), hit + window + 1)]).strip(), hit + 1


def patterns(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    hits: list[dict[str, Any]] = []
    for row in read_jsonl(out / "clean_snippets.jsonl"):
        text = row.get("text", "")
        for pattern_name, regexes in PATTERNS.items():
            for regex in regexes:
                for match in re.finditer(regex, text, re.M):
                    snippet, line = line_window(text, match.start())
                    if len(snippet) < 20:
                        continue
                    hits.append({
                        "id": f"pattern_hit_{len(hits):06d}",
                        "pattern": pattern_name,
                        "regex": regex,
                        "repo": row["repo"],
                        "path": row["path"],
                        "line": line,
                        "snippet": snippet[:3000],
                        "jax_score": row.get("jax_score"),
                        "source_snippet_id": row.get("id"),
                    })
    dedup: list[dict[str, Any]] = []
    seen = set()
    for hit in hits:
        key = (hit["pattern"], hit["repo"], hit["path"], hit["snippet"][:300])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(hit)
    write_jsonl(out / "pattern_hits.jsonl", dedup)
    summary = {"total_hits": len(dedup), "patterns": dict(Counter(h["pattern"] for h in dedup).most_common()), "repos": dict(Counter(h["repo"] for h in dedup).most_common(50))}
    write_json(out / "pattern_summary.json", summary)
    print("pattern hits:", len(dedup))


def cap_pattern_hits(out: Path) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in read_jsonl(out / "pattern_hits.jsonl"):
        by[hit["pattern"]].append(hit)
    selected: list[dict[str, Any]] = []
    for pattern_name, hits in by.items():
        repo_counts: Counter[str] = Counter()
        kept = 0
        for hit in sorted(hits, key=lambda h: h.get("jax_score") or 0, reverse=True):
            if repo_counts[hit["repo"]] >= 15:
                continue
            hit = dict(hit)
            hit["capped_id"] = f"pattern_capped_{len(selected):06d}"
            selected.append(hit)
            repo_counts[hit["repo"]] += 1
            kept += 1
            if kept >= MAX_PER_PATTERN.get(pattern_name, 50):
                break
    write_jsonl(out / "pattern_hits_capped.jsonl", selected)
    return selected


def build_case(case_id: str, pattern: str, error_type: str, broken: str, fixed: str, root_cause: str, explanation: str, tags: list[str], hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case_id,
        "source": "repo_grounded_synthetic_mutation",
        "task_type": "jax_error_fix",
        "pattern": pattern,
        "error_type": error_type,
        "difficulty": "intermediate",
        "broken_code": broken,
        "fixed_code": fixed,
        "root_cause": root_cause,
        "explanation": explanation,
        "tags": tags,
        "source_evidence": {"repo": hit["repo"], "path": hit["path"], "line": hit["line"], "pattern": hit["pattern"], "snippet": hit["snippet"][:1200]},
        "messages": [
            {"role": "system", "content": "You are XLADoctor, a JAX/XLA debugging assistant."},
            {"role": "user", "content": f"Broken JAX code:\n```python\n{broken}\n```"},
            {"role": "assistant", "content": f"Error type: {error_type}\n\nRoot cause: {root_cause}\n\nFixed code:\n```python\n{fixed}\n```\n\nExplanation: {explanation}"},
        ],
    }


def mutations(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    selected = cap_pattern_hits(out)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in selected:
        by[hit["pattern"]].append(hit)
    templates = {
        "functional_update_at": ("JAX array immutability / item assignment error", "import jax.numpy as jnp\n\ndef update_value(x):\n    x[1] = 10.0\n    return x\n", "import jax.numpy as jnp\n\ndef update_value(x):\n    return x.at[1].set(10.0)\n", "JAX arrays are immutable, so direct item assignment is invalid.", "Use x.at[index].set(value) to return an updated array.", ["jax", "immutability", "at_set"], 40),
        "lax_cond": ("TracerBoolConversionError", "import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef choose_sign(x):\n    if jnp.sum(x) > 0:\n        return x\n    return -x\n", "import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef choose_sign(x):\n    return jax.lax.cond(jnp.sum(x) > 0, lambda y: y, lambda y: -y, x)\n", "A Python if depends on a traced value inside jit.", "Use jax.lax.cond for traced branch conditions.", ["jax", "jit", "lax_cond"], 40),
        "vmap": ("vmap in_axes axis-size mismatch", "import jax\nimport jax.numpy as jnp\n\ndef add_bias(x, bias):\n    return x + bias\n\nx = jnp.ones((3, 4))\nbias = jnp.ones((4,))\ny = jax.vmap(add_bias, in_axes=(0, 0))(x, bias)\n", "import jax\nimport jax.numpy as jnp\n\ndef add_bias(x, bias):\n    return x + bias\n\nx = jnp.ones((3, 4))\nbias = jnp.ones((4,))\ny = jax.vmap(add_bias, in_axes=(0, None))(x, bias)\n", "bias is shared across mapped examples, but in_axes maps over it.", "Use in_axes=(0, None) for shared arguments.", ["jax", "vmap", "in_axes"], 40),
        "jit_static_argnums": ("ConcretizationTypeError", "import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef make_range(n):\n    return jnp.arange(n)\n", "import functools\nimport jax\nimport jax.numpy as jnp\n\n@functools.partial(jax.jit, static_argnames=(\"n\",))\ndef make_range(n):\n    return jnp.arange(n)\n", "n controls an array shape but is traced under jit.", "Mark shape-controlling arguments static.", ["jax", "jit", "static_argnames"], 40),
        "random_split": ("PRNG key reuse", "import jax\n\nkey = jax.random.PRNGKey(0)\na = jax.random.normal(key, (3,))\nb = jax.random.uniform(key, (3,))\n", "import jax\n\nkey = jax.random.PRNGKey(0)\nkey, k1, k2 = jax.random.split(key, 3)\na = jax.random.normal(k1, (3,))\nb = jax.random.uniform(k2, (3,))\n", "The same PRNG key is reused for separate random draws.", "Split PRNG keys before separate stochastic operations.", ["jax", "random", "split"], 40),
        "lax_scan": ("Python loop/list accumulation under jit", "import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef cumulative_sum(xs):\n    total = 0.0\n    outputs = []\n    for x in xs:\n        total = total + x\n        outputs.append(total)\n    return jnp.array(outputs)\n", "import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef cumulative_sum(xs):\n    def step(total, x):\n        total = total + x\n        return total, total\n    _, ys = jax.lax.scan(step, 0.0, xs)\n    return ys\n", "Python list accumulation is a poor fit for jitted loops.", "Use lax.scan for loop-carried state and stacked outputs.", ["jax", "lax_scan"], 30),
    }
    cases: list[dict[str, Any]] = []
    for pattern_name, spec in templates.items():
        error_type, broken, fixed, root_cause, explanation, tags, limit = spec
        for hit in by.get(pattern_name, [])[:limit]:
            cases.append(build_case(f"repo_mutation_{len(cases):06d}", pattern_name, error_type, broken, fixed, root_cause, explanation, tags, hit))
    write_jsonl(out / "repo_grounded_mutation_cases_v0_2.jsonl", cases)
    print("mutation cases:", len(cases))


def validate(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    rows = read_jsonl(out / "repo_grounded_mutation_cases_v0_2.jsonl")
    reports: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []
    for row in rows:
        problems: list[str] = []
        for field in ["id", "broken_code", "fixed_code", "error_type", "root_cause", "source_evidence"]:
            if not row.get(field):
                problems.append(f"missing:{field}")
        for name in ["broken_code", "fixed_code"]:
            try:
                ast.parse(row.get(name, ""))
            except SyntaxError as exc:
                problems.append(f"syntax:{name}:{exc}")
        ok = not problems
        reports.append({"id": row.get("id"), "ok": ok, "problems": problems, "error_type": row.get("error_type"), "pattern": row.get("pattern")})
        if ok:
            passed.append(row)
    summary = {"total": len(rows), "ok": len(passed), "failed": len(rows) - len(passed), "by_pattern": dict(Counter(r.get("pattern") for r in reports)), "by_error_type": dict(Counter(r.get("error_type") for r in reports)), "note": "Structural validation only. Execute passed cases in a JAX environment before release."}
    write_json(out / "repo_grounded_mutation_validation_report.json", {"summary": summary, "reports": reports})
    write_jsonl(out / "repo_grounded_mutation_cases_v0_2_passed.jsonl", passed)
    print(summary)


def manifest(args: argparse.Namespace) -> None:
    root = Path(args.out_dir)
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "/.git/" not in str(path).replace("\\", "/"):
            data = path.read_bytes()
            files.append({"path": str(path.relative_to(root)).replace("\\", "/"), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    write_json(root / "manifest.json", {"files": files})
    print("manifest files:", len(files))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["discover", "plan", "clone", "extract", "score", "patterns", "mutations", "validate", "manifest"]:
        sp = sub.add_parser(name)
        sp.add_argument("--out-dir", required=True)
        if name == "discover":
            sp.add_argument("--target-per-bucket", type=int, default=50)
            sp.add_argument("--max-pages-per-query", type=int, default=2)
            sp.add_argument("--sleep-seconds", type=float, default=3.0)
        if name == "plan":
            sp.add_argument("--per-bucket", type=int, default=50)
        if name == "clone":
            sp.add_argument("--max-clones", type=int, default=60)
        if name == "score":
            sp.add_argument("--min-score", type=int, default=6)
    args = parser.parse_args()
    globals()[args.cmd](args)


if __name__ == "__main__":
    main()
