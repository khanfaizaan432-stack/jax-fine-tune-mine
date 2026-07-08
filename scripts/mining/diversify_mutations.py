#!/usr/bin/env python3
"""Replace repeated mutation rows with a broader template bank.

The earlier miner proved the pipeline but produced only a few behavioral
signatures. This script keeps the same repo-grounded evidence mechanism while
expanding the number of distinct JAX bug/fix behaviors.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


TEMPLATES: list[dict[str, Any]] = [
    {"name":"immutability_set_scalar","pattern":"functional_update_at","error_type":"JAX array immutability / item assignment error","root_cause":"JAX arrays are immutable, so direct item assignment cannot update an array in place.","explanation":"Return a new array with x.at[index].set(value).","broken":"import jax.numpy as jnp\n\ndef update_value(x):\n    x[1] = 10.0\n    return x\n","fixed":"import jax.numpy as jnp\n\ndef update_value(x):\n    return x.at[1].set(10.0)\n","tags":["jax","immutability","at_set"]},
    {"name":"immutability_add_slice","pattern":"functional_update_at","error_type":"JAX array immutability / item assignment error","root_cause":"A slice update uses Python mutation on a JAX array.","explanation":"Use x.at[start:stop].add(delta) to express a functional update.","broken":"import jax.numpy as jnp\n\ndef boost_prefix(x):\n    x[:2] += 1.0\n    return x\n","fixed":"import jax.numpy as jnp\n\ndef boost_prefix(x):\n    return x.at[:2].add(1.0)\n","tags":["jax","immutability","at_add"]},
    {"name":"immutability_loop_write","pattern":"functional_update_at","error_type":"JAX array immutability / item assignment error","root_cause":"The loop repeatedly mutates elements of a JAX array.","explanation":"Carry the updated array through functional .at updates.","broken":"import jax.numpy as jnp\n\ndef fill_diag(x):\n    for i in range(x.shape[0]):\n        x[i, i] = 1.0\n    return x\n","fixed":"import jax.numpy as jnp\n\ndef fill_diag(x):\n    for i in range(x.shape[0]):\n        x = x.at[i, i].set(1.0)\n    return x\n","tags":["jax","immutability","loop_update"]},
    {"name":"cond_scalar_sum","pattern":"lax_cond","error_type":"TracerBoolConversionError","root_cause":"A Python if statement depends on a traced JAX value inside jit.","explanation":"Use jax.lax.cond for data-dependent control flow under jit.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef choose_sign(x):\n    if jnp.sum(x) > 0:\n        return x\n    return -x\n","fixed":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef choose_sign(x):\n    return jax.lax.cond(jnp.sum(x) > 0, lambda y: y, lambda y: -y, x)\n","tags":["jax","jit","lax_cond"]},
    {"name":"cond_norm_clip","pattern":"lax_cond","error_type":"TracerBoolConversionError","root_cause":"The branch condition is computed from traced array data.","explanation":"Move both branches into lax.cond and pass operands explicitly.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef normalize_or_zero(x):\n    n = jnp.linalg.norm(x)\n    if n > 1.0:\n        return x / n\n    return jnp.zeros_like(x)\n","fixed":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef normalize_or_zero(x):\n    n = jnp.linalg.norm(x)\n    return jax.lax.cond(n > 1.0, lambda t: t[0] / t[1], lambda t: jnp.zeros_like(t[0]), (x, n))\n","tags":["jax","jit","lax_cond"]},
    {"name":"cond_training_flag_static","pattern":"lax_cond","error_type":"TracerBoolConversionError","root_cause":"A boolean argument is treated as traced data but used in Python control flow.","explanation":"Mark configuration booleans static when they control Python branches.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef maybe_dropout(x, training):\n    if training:\n        return x * 0.5\n    return x\n","fixed":"import functools\nimport jax\nimport jax.numpy as jnp\n\n@functools.partial(jax.jit, static_argnames=(\"training\",))\ndef maybe_dropout(x, training):\n    if training:\n        return x * 0.5\n    return x\n","tags":["jax","jit","static_argnames"]},
    {"name":"vmap_shared_bias","pattern":"vmap","error_type":"vmap in_axes axis-size mismatch","root_cause":"A shared bias vector is incorrectly mapped as if it had a batch dimension.","explanation":"Use in_axes=(0, None) for shared arguments.","broken":"import jax\nimport jax.numpy as jnp\n\ndef add_bias(x, bias):\n    return x + bias\n\nx = jnp.ones((3, 4))\nbias = jnp.ones((4,))\ny = jax.vmap(add_bias, in_axes=(0, 0))(x, bias)\n","fixed":"import jax\nimport jax.numpy as jnp\n\ndef add_bias(x, bias):\n    return x + bias\n\nx = jnp.ones((3, 4))\nbias = jnp.ones((4,))\ny = jax.vmap(add_bias, in_axes=(0, None))(x, bias)\n","tags":["jax","vmap","in_axes"]},
    {"name":"vmap_shared_params","pattern":"vmap","error_type":"vmap in_axes axis-size mismatch","root_cause":"Model parameters are shared across the batch but were mapped along axis 0.","explanation":"Map examples and keep params unmapped.","broken":"import jax\nimport jax.numpy as jnp\n\ndef linear(params, x):\n    return x @ params[\"w\"] + params[\"b\"]\n\nparams = {\"w\": jnp.ones((4, 2)), \"b\": jnp.zeros((2,))}\nx = jnp.ones((8, 4))\ny = jax.vmap(linear, in_axes=(0, 0))(params, x)\n","fixed":"import jax\nimport jax.numpy as jnp\n\ndef linear(params, x):\n    return x @ params[\"w\"] + params[\"b\"]\n\nparams = {\"w\": jnp.ones((4, 2)), \"b\": jnp.zeros((2,))}\nx = jnp.ones((8, 4))\ny = jax.vmap(linear, in_axes=(None, 0))(params, x)\n","tags":["jax","vmap","params"]},
    {"name":"vmap_output_axes","pattern":"vmap","error_type":"vmap out_axes mismatch","root_cause":"The function returns a tuple but out_axes does not match the output tree.","explanation":"Provide an out_axes tree with one entry per returned value.","broken":"import jax\nimport jax.numpy as jnp\n\ndef stats(x):\n    return x.sum(), x.mean()\n\nx = jnp.ones((5, 3))\ny = jax.vmap(stats, out_axes=0)(x)\n","fixed":"import jax\nimport jax.numpy as jnp\n\ndef stats(x):\n    return x.sum(), x.mean()\n\nx = jnp.ones((5, 3))\ny = jax.vmap(stats, out_axes=(0, 0))(x)\n","tags":["jax","vmap","out_axes"]},
    {"name":"static_arange_len","pattern":"jit_static_argnums","error_type":"ConcretizationTypeError","root_cause":"A traced value controls jnp.arange length inside jit.","explanation":"Mark shape-controlling arguments static.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef make_range(n):\n    return jnp.arange(n)\n","fixed":"import functools\nimport jax\nimport jax.numpy as jnp\n\n@functools.partial(jax.jit, static_argnames=(\"n\",))\ndef make_range(n):\n    return jnp.arange(n)\n","tags":["jax","jit","static_argnames"]},
    {"name":"static_reshape_dim","pattern":"jit_static_argnums","error_type":"ConcretizationTypeError","root_cause":"The reshape dimension is traced but must be known at compile time.","explanation":"Make the dimension static or pass arrays with fixed shapes.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef flatten_to(x, width):\n    return jnp.reshape(x, (-1, width))\n","fixed":"import functools\nimport jax\nimport jax.numpy as jnp\n\n@functools.partial(jax.jit, static_argnames=(\"width\",))\ndef flatten_to(x, width):\n    return jnp.reshape(x, (-1, width))\n","tags":["jax","jit","static_argnames","shape"]},
    {"name":"static_python_range","pattern":"jit_static_argnums","error_type":"TracerIntegerConversionError","root_cause":"Python range needs a concrete integer, but steps is traced by jit.","explanation":"Mark loop-count arguments static when using Python loops.","broken":"import jax\n\n@jax.jit\ndef repeat_add(x, steps):\n    for _ in range(steps):\n        x = x + 1\n    return x\n","fixed":"import functools\nimport jax\n\n@functools.partial(jax.jit, static_argnames=(\"steps\",))\ndef repeat_add(x, steps):\n    for _ in range(steps):\n        x = x + 1\n    return x\n","tags":["jax","jit","static_argnames","range"]},
    {"name":"random_reuse_two_draws","pattern":"random_split","error_type":"PRNG key reuse","root_cause":"The same PRNG key is reused for separate random draws.","explanation":"Split keys before independent stochastic operations.","broken":"import jax\n\nkey = jax.random.PRNGKey(0)\na = jax.random.normal(key, (3,))\nb = jax.random.uniform(key, (3,))\n","fixed":"import jax\n\nkey = jax.random.PRNGKey(0)\nkey, k1, k2 = jax.random.split(key, 3)\na = jax.random.normal(k1, (3,))\nb = jax.random.uniform(k2, (3,))\n","tags":["jax","random","split"]},
    {"name":"random_loop_reuse","pattern":"random_split","error_type":"PRNG key reuse","root_cause":"A loop repeatedly uses the same key, creating identical random draws.","explanation":"Split or fold in the loop index for each draw.","broken":"import jax\nimport jax.numpy as jnp\n\ndef sample_many(key):\n    xs = []\n    for _ in range(3):\n        xs.append(jax.random.normal(key, (2,)))\n    return jnp.stack(xs)\n","fixed":"import jax\nimport jax.numpy as jnp\n\ndef sample_many(key):\n    keys = jax.random.split(key, 3)\n    return jnp.stack([jax.random.normal(k, (2,)) for k in keys])\n","tags":["jax","random","split","loop"]},
    {"name":"random_jit_no_return_key","pattern":"random_split","error_type":"PRNG key state bug","root_cause":"The function consumes randomness but does not return the updated key.","explanation":"Return the next key so callers do not reuse stale randomness.","broken":"import jax\n\ndef sample_step(key):\n    key, subkey = jax.random.split(key)\n    return jax.random.normal(subkey, (4,))\n","fixed":"import jax\n\ndef sample_step(key):\n    key, subkey = jax.random.split(key)\n    return key, jax.random.normal(subkey, (4,))\n","tags":["jax","random","state"]},
    {"name":"scan_cumsum","pattern":"lax_scan","error_type":"Python loop/list accumulation under jit","root_cause":"Python list accumulation is a poor fit for jitted loop outputs.","explanation":"Use lax.scan for loop-carried state and stacked outputs.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef cumulative_sum(xs):\n    total = 0.0\n    outputs = []\n    for x in xs:\n        total = total + x\n        outputs.append(total)\n    return jnp.array(outputs)\n","fixed":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef cumulative_sum(xs):\n    def step(total, x):\n        total = total + x\n        return total, total\n    _, ys = jax.lax.scan(step, 0.0, xs)\n    return ys\n","tags":["jax","lax_scan"]},
    {"name":"scan_decay","pattern":"lax_scan","error_type":"Loop-carried state should use lax.scan","root_cause":"A recurrent computation is written as a Python loop instead of a JAX scan.","explanation":"Use lax.scan to make recurrent state explicit and transform-friendly.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef decay(xs):\n    state = 0.0\n    ys = []\n    for x in xs:\n        state = 0.9 * state + x\n        ys.append(state)\n    return jnp.stack(ys)\n","fixed":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef decay(xs):\n    def step(state, x):\n        state = 0.9 * state + x\n        return state, state\n    _, ys = jax.lax.scan(step, 0.0, xs)\n    return ys\n","tags":["jax","lax_scan","state"]},
    {"name":"scan_rng_keys","pattern":"lax_scan","error_type":"Random loop should carry PRNG state","root_cause":"A stochastic loop needs explicit PRNG state threading.","explanation":"Carry the key through lax.scan and split at each step.","broken":"import jax\nimport jax.numpy as jnp\n\ndef noisy_walk(key, xs):\n    state = 0.0\n    ys = []\n    for x in xs:\n        noise = jax.random.normal(key, ())\n        state = state + x + noise\n        ys.append(state)\n    return jnp.stack(ys)\n","fixed":"import jax\nimport jax.numpy as jnp\n\ndef noisy_walk(key, xs):\n    def step(carry, x):\n        state, key = carry\n        key, subkey = jax.random.split(key)\n        state = state + x + jax.random.normal(subkey, ())\n        return (state, key), state\n    (_, _), ys = jax.lax.scan(step, (0.0, key), xs)\n    return ys\n","tags":["jax","lax_scan","random"]},
]


def build_case(case_id: str, template: dict[str, Any], hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case_id,
        "source": "repo_grounded_diverse_synthetic_mutation",
        "task_type": "jax_error_fix",
        "pattern": template["pattern"],
        "template_name": template["name"],
        "error_type": template["error_type"],
        "difficulty": "intermediate",
        "broken_code": template["broken"],
        "fixed_code": template["fixed"],
        "root_cause": template["root_cause"],
        "explanation": template["explanation"],
        "tags": template.get("tags", []),
        "source_evidence": {"repo": hit.get("repo"), "path": hit.get("path"), "line": hit.get("line"), "pattern": hit.get("pattern"), "snippet": (hit.get("snippet") or "")[:1200]},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rows-per-template", type=int, default=16)
    args = p.parse_args()
    out = Path(args.out_dir)
    hits = read_jsonl(out / "pattern_hits_capped.jsonl")
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        by_pattern[hit.get("pattern", "")].append(hit)
    cases: list[dict[str, Any]] = []
    missing: dict[str, int] = {}
    for template in TEMPLATES:
        choices = by_pattern.get(template["pattern"], [])
        if not choices:
            missing[template["name"]] = 0
            continue
        for i, hit in enumerate(choices[: args.rows_per_template]):
            cases.append(build_case(f"repo_diverse_{len(cases):06d}_{template['name']}_{i:02d}", template, hit))
    write_jsonl(out / "repo_grounded_mutation_cases_v0_2.jsonl", cases)
    write_json(out / "diverse_mutation_report.json", {
        "templates": len(TEMPLATES),
        "rows_per_template": args.rows_per_template,
        "cases": len(cases),
        "by_pattern": dict(Counter(c["pattern"] for c in cases)),
        "by_template": dict(Counter(c["template_name"] for c in cases)),
        "missing_templates": missing,
    })
    print("diverse mutation cases:", len(cases))


if __name__ == "__main__":
    main()
