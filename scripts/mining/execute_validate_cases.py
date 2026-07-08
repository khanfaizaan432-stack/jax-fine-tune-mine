#!/usr/bin/env python3
"""Optional executable validation for generated JAXFixBench cases.

The workflow treats this as a quality filter, not as a hard benchmark. If JAX is
unavailable, the script reports that and keeps the static-passed cases so the
pipeline remains usable.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def has_jax() -> tuple[bool, str | None]:
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        return True, None
    except Exception as exc:
        return False, repr(exc)


def run_code(code: str, template_name: str, pattern: str) -> tuple[bool, str | None]:
    try:
        import jax
        import jax.numpy as jnp
        ns: dict[str, Any] = {"jax": jax, "jnp": jnp}
        exec(compile(code, "<jaxfixbench_case>", "exec"), ns, ns)
        if "update_value" in ns:
            ns["update_value"](jnp.array([1.0, 2.0, 3.0]))
        if "boost_prefix" in ns:
            ns["boost_prefix"](jnp.array([1.0, 2.0, 3.0]))
        if "fill_diag" in ns:
            ns["fill_diag"](jnp.zeros((3, 3)))
        if "choose_sign" in ns:
            ns["choose_sign"](jnp.array([1.0, -0.5]))
        if "normalize_or_zero" in ns:
            ns["normalize_or_zero"](jnp.array([2.0, 0.0]))
        if "maybe_dropout" in ns:
            ns["maybe_dropout"](jnp.ones((2,)), True)
        if "make_range" in ns:
            ns["make_range"](3)
        if "flatten_to" in ns:
            ns["flatten_to"](jnp.ones((2, 3)), 3)
        if "repeat_add" in ns:
            ns["repeat_add"](1, 3)
        if "sample_many" in ns:
            ns["sample_many"](jax.random.PRNGKey(0))
        if "sample_step" in ns:
            ns["sample_step"](jax.random.PRNGKey(0))
        if "cumulative_sum" in ns:
            ns["cumulative_sum"](jnp.array([1.0, 2.0, 3.0]))
        if "decay" in ns:
            ns["decay"](jnp.array([1.0, 2.0, 3.0]))
        if "noisy_walk" in ns:
            ns["noisy_walk"](jax.random.PRNGKey(0), jnp.array([1.0, 2.0, 3.0]))
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


BROKEN_SHOULD_FAIL = {"functional_update_at", "lax_cond", "vmap", "jit_static_argnums"}
SEMANTIC_ONLY = {"random_split", "lax_scan"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    src = out / "repo_grounded_mutation_cases_v0_2_passed.jsonl"
    if not src.exists():
        src = out / "repo_grounded_mutation_cases_v0_2.jsonl"
    cases = read_jsonl(src)
    available, import_error = has_jax()
    if not available:
        write_jsonl(out / "repo_grounded_mutation_cases_v0_2_exec_passed.jsonl", cases)
        write_json(out / "repo_grounded_mutation_execution_report.json", {"enabled": False, "jax_import_error": import_error, "cases_kept": len(cases)})
        print("JAX unavailable; kept static-passed cases")
        return

    passed: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for case in cases:
        fixed_ok, fixed_err = run_code(case.get("fixed_code", ""), case.get("template_name", ""), case.get("pattern", ""))
        broken_ok, broken_err = run_code(case.get("broken_code", ""), case.get("template_name", ""), case.get("pattern", ""))
        pattern = case.get("pattern", "")
        if pattern in BROKEN_SHOULD_FAIL:
            ok = fixed_ok and not broken_ok
        elif pattern in SEMANTIC_ONLY:
            ok = fixed_ok
        else:
            ok = fixed_ok
        case = dict(case)
        case["execution_validation"] = {"fixed_runs": fixed_ok, "fixed_error": fixed_err, "broken_runs": broken_ok, "broken_error": broken_err, "included": ok}
        reports.append({"id": case.get("id"), "pattern": pattern, "template_name": case.get("template_name"), **case["execution_validation"]})
        if ok:
            passed.append(case)
    write_jsonl(out / "repo_grounded_mutation_cases_v0_2_exec_validated.jsonl", cases)
    write_jsonl(out / "repo_grounded_mutation_cases_v0_2_exec_passed.jsonl", passed)
    write_json(out / "repo_grounded_mutation_execution_report.json", {"enabled": True, "total": len(cases), "passed": len(passed), "failed": len(cases) - len(passed), "reports": reports})
    print(json.dumps({"total": len(cases), "passed": len(passed), "failed": len(cases) - len(passed)}, indent=2))


if __name__ == "__main__":
    main()
