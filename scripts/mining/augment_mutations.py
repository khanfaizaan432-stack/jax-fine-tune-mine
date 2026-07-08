#!/usr/bin/env python3
"""Append a second wave of mutation templates for more behavior signatures."""

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


EXTRA = [
    {"name":"immutability_boolean_mask","pattern":"functional_update_at","error_type":"JAX array immutability / item assignment error","root_cause":"A boolean-mask assignment mutates a JAX array in place.","explanation":"Use x.at[mask].set(value) to return an updated array.","broken":"import jax.numpy as jnp\n\ndef clamp_positive(x):\n    x[x < 0] = 0.0\n    return x\n","fixed":"import jax.numpy as jnp\n\ndef clamp_positive(x):\n    return x.at[x < 0].set(0.0)\n","tags":["jax","immutability","mask"]},
    {"name":"immutability_index_add","pattern":"functional_update_at","error_type":"JAX array immutability / item assignment error","root_cause":"Indexed accumulation uses in-place mutation on a JAX array.","explanation":"Use x.at[idx].add(values) for indexed accumulation.","broken":"import jax.numpy as jnp\n\ndef add_updates(x, idx, values):\n    x[idx] += values\n    return x\n","fixed":"import jax.numpy as jnp\n\ndef add_updates(x, idx, values):\n    return x.at[idx].add(values)\n","tags":["jax","immutability","scatter_add"]},
    {"name":"cond_abs_scalar","pattern":"lax_cond","error_type":"TracerBoolConversionError","root_cause":"A Python branch compares a traced scalar inside jit.","explanation":"Use lax.cond so the branch is represented in JAXPR.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef abs_scalar(x):\n    if x < 0:\n        return -x\n    return x\n","fixed":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef abs_scalar(x):\n    return jax.lax.cond(x < 0, lambda z: -z, lambda z: z, x)\n","tags":["jax","jit","lax_cond"]},
    {"name":"cond_use_where_elementwise","pattern":"lax_cond","error_type":"TracerBoolConversionError","root_cause":"A vector predicate is used as a Python boolean.","explanation":"For elementwise choices, use jnp.where instead of a Python if.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef relu_bad(x):\n    if x > 0:\n        return x\n    return 0.0\n","fixed":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef relu_bad(x):\n    return jnp.where(x > 0, x, 0.0)\n","tags":["jax","jit","where"]},
    {"name":"vmap_two_batches_mismatch","pattern":"vmap","error_type":"vmap in_axes axis-size mismatch","root_cause":"Two mapped arguments have different leading batch sizes.","explanation":"Only map the batched argument and keep the other argument shared, or align batch sizes.","broken":"import jax\nimport jax.numpy as jnp\n\ndef dot_pair(x, y):\n    return jnp.dot(x, y)\n\nx = jnp.ones((3, 4))\ny = jnp.ones((4,))\nout = jax.vmap(dot_pair, in_axes=(0, 0))(x, y)\n","fixed":"import jax\nimport jax.numpy as jnp\n\ndef dot_pair(x, y):\n    return jnp.dot(x, y)\n\nx = jnp.ones((3, 4))\ny = jnp.ones((4,))\nout = jax.vmap(dot_pair, in_axes=(0, None))(x, y)\n","tags":["jax","vmap","in_axes"]},
    {"name":"vmap_rng_keys","pattern":"vmap","error_type":"PRNG key batching bug","root_cause":"A single PRNG key is shared across a vmapped stochastic function.","explanation":"Split a batch of keys and map over the key axis.","broken":"import jax\n\ndef sample(key):\n    return jax.random.normal(key, ())\n\nkey = jax.random.PRNGKey(0)\nout = jax.vmap(sample)(key)\n","fixed":"import jax\n\ndef sample(key):\n    return jax.random.normal(key, ())\n\nkey = jax.random.PRNGKey(0)\nkeys = jax.random.split(key, 8)\nout = jax.vmap(sample)(keys)\n","tags":["jax","vmap","random"]},
    {"name":"static_zeros_shape","pattern":"jit_static_argnums","error_type":"ConcretizationTypeError","root_cause":"A traced argument is used as an array shape.","explanation":"Mark shape arguments static or derive shape from input arrays.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef make_zeros(n):\n    return jnp.zeros((n, 4))\n","fixed":"import functools\nimport jax\nimport jax.numpy as jnp\n\n@functools.partial(jax.jit, static_argnames=(\"n\",))\ndef make_zeros(n):\n    return jnp.zeros((n, 4))\n","tags":["jax","jit","shape"]},
    {"name":"static_num_classes","pattern":"jit_static_argnums","error_type":"ConcretizationTypeError","root_cause":"num_classes controls one_hot output shape but is traced.","explanation":"Mark num_classes static because it changes output shape.","broken":"import jax\n\n@jax.jit\ndef labels_to_one_hot(y, num_classes):\n    return jax.nn.one_hot(y, num_classes)\n","fixed":"import functools\nimport jax\n\n@functools.partial(jax.jit, static_argnames=(\"num_classes\",))\ndef labels_to_one_hot(y, num_classes):\n    return jax.nn.one_hot(y, num_classes)\n","tags":["jax","jit","one_hot"]},
    {"name":"random_fold_in_step","pattern":"random_split","error_type":"PRNG key reuse across steps","root_cause":"The same base key is reused for each training step.","explanation":"Use fold_in or split so each step has distinct randomness.","broken":"import jax\n\ndef step_noise(key, step):\n    return jax.random.normal(key, (2,))\n","fixed":"import jax\n\ndef step_noise(key, step):\n    subkey = jax.random.fold_in(key, step)\n    return jax.random.normal(subkey, (2,))\n","tags":["jax","random","fold_in"]},
    {"name":"random_split_for_dropout","pattern":"random_split","error_type":"PRNG key reuse between dropout and noise","root_cause":"One key is reused for two stochastic operations in the same step.","explanation":"Split separate subkeys for separate random operations.","broken":"import jax\n\ndef stochastic_pair(key):\n    mask = jax.random.bernoulli(key, 0.5, (4,))\n    noise = jax.random.normal(key, (4,))\n    return mask, noise\n","fixed":"import jax\n\ndef stochastic_pair(key):\n    k1, k2 = jax.random.split(key)\n    mask = jax.random.bernoulli(k1, 0.5, (4,))\n    noise = jax.random.normal(k2, (4,))\n    return mask, noise\n","tags":["jax","random","dropout"]},
    {"name":"scan_tuple_state","pattern":"lax_scan","error_type":"Loop-carried tuple state should use lax.scan","root_cause":"A recurrent loop carries multiple pieces of state through Python control flow.","explanation":"Use lax.scan with a tuple carry.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef momentum(xs):\n    pos, vel = 0.0, 0.0\n    ys = []\n    for force in xs:\n        vel = 0.9 * vel + force\n        pos = pos + vel\n        ys.append(pos)\n    return jnp.stack(ys)\n","fixed":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef momentum(xs):\n    def step(carry, force):\n        pos, vel = carry\n        vel = 0.9 * vel + force\n        pos = pos + vel\n        return (pos, vel), pos\n    (_, _), ys = jax.lax.scan(step, (0.0, 0.0), xs)\n    return ys\n","tags":["jax","lax_scan","tuple_state"]},
    {"name":"scan_early_stop_flag","pattern":"lax_scan","error_type":"Stateful loop should carry flags explicitly","root_cause":"Loop state and stopping flags are implicit in Python variables.","explanation":"Carry flags explicitly through lax.scan or use lax.while_loop for dynamic stopping.","broken":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef threshold_accum(xs):\n    total = 0.0\n    done = False\n    ys = []\n    for x in xs:\n        if not done:\n            total = total + x\n            done = total > 3.0\n        ys.append(total)\n    return jnp.stack(ys)\n","fixed":"import jax\nimport jax.numpy as jnp\n\n@jax.jit\ndef threshold_accum(xs):\n    def step(carry, x):\n        total, done = carry\n        total = jax.lax.cond(done, lambda t: t, lambda t: t + x, total)\n        done = jnp.logical_or(done, total > 3.0)\n        return (total, done), total\n    (_, _), ys = jax.lax.scan(step, (0.0, False), xs)\n    return ys\n","tags":["jax","lax_scan","lax_cond"]},
]


def build(case_id: str, template: dict[str, Any], hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case_id,
        "source": "repo_grounded_augmented_synthetic_mutation",
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
    cases = read_jsonl(out / "repo_grounded_mutation_cases_v0_2.jsonl")
    hits = read_jsonl(out / "pattern_hits_capped.jsonl")
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        by_pattern[hit.get("pattern", "")].append(hit)
    start = len(cases)
    for template in EXTRA:
        for i, hit in enumerate(by_pattern.get(template["pattern"], [])[: args.rows_per_template]):
            cases.append(build(f"repo_aug_{len(cases):06d}_{template['name']}_{i:02d}", template, hit))
    write_jsonl(out / "repo_grounded_mutation_cases_v0_2.jsonl", cases)
    write_json(out / "augmented_mutation_report.json", {"extra_templates": len(EXTRA), "added_cases": len(cases) - start, "total_cases": len(cases), "by_template": dict(Counter(c.get("template_name") for c in cases))})
    print("augmented mutation cases:", len(cases))


if __name__ == "__main__":
    main()
