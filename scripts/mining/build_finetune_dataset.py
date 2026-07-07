#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + ('\n' if rows else ''), encoding='utf-8')


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def sig(c: dict[str, Any]) -> str:
    payload = '\n'.join([str(c.get('pattern','')), str(c.get('error_type','')), str(c.get('broken_code','')), str(c.get('fixed_code',''))])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def valid(c: dict[str, Any]) -> dict[str, Any]:
    probs = []
    for f in ['id','pattern','error_type','broken_code','fixed_code','root_cause','explanation','source_evidence']:
        if not c.get(f):
            probs.append('missing:' + f)
    for f in ['broken_code','fixed_code']:
        try:
            ast.parse(c.get(f,'') or '')
        except SyntaxError as e:
            probs.append('syntax:' + f + ':' + str(e))
    return {'static_ok': not probs, 'problems': probs, 'note': 'static validation only'}


def user_prompt(c: dict[str, Any]) -> str:
    ev = c.get('source_evidence') or {}
    return textwrap.dedent(f'''
    Diagnose and fix this JAX bug. Return the error type, root cause, fixed code, and a concise explanation.

    Source evidence:
    - repo: {ev.get('repo','unknown')}
    - path: {ev.get('path','unknown')}
    - line: {ev.get('line','unknown')}
    - mined pattern: {ev.get('pattern', c.get('pattern','unknown'))}

    Evidence snippet:
    ```python
    {ev.get('snippet','')[:1400]}
    ```

    Broken code:
    ```python
    {c.get('broken_code','').strip()}
    ```
    ''').strip()


def answer(c: dict[str, Any]) -> str:
    return textwrap.dedent(f'''
    Error type: {c.get('error_type','unknown')}

    Root cause: {c.get('root_cause','').strip()}

    Fixed code:
    ```python
    {c.get('fixed_code','').strip()}
    ```

    Explanation: {c.get('explanation','').strip()}
    ''').strip()


def row(c: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': c['id'],
        'messages': [
            {'role': 'system', 'content': 'You are XLADoctor, a precise JAX/XLA debugging assistant. Fix code without removing JAX transformations unless necessary.'},
            {'role': 'user', 'content': user_prompt(c)},
            {'role': 'assistant', 'content': answer(c)},
        ],
        'metadata': {'pattern': c.get('pattern'), 'error_type': c.get('error_type'), 'behavior_signature': c.get('behavior_signature'), 'validation': c.get('validation'), 'source_evidence': c.get('source_evidence')},
    }


def split(cid: str) -> str:
    b = h(cid) % 100
    return 'train' if b < 80 else ('val' if b < 90 else 'test')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--sft-dir', required=True)
    ap.add_argument('--max-per-signature', type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out_dir); sft = Path(a.sft_dir)
    src = out / 'repo_grounded_mutation_cases_v0_2_passed.jsonl'
    if not src.exists():
        src = out / 'repo_grounded_mutation_cases_v0_2.jsonl'
    cases = read_jsonl(src)
    enriched = []
    for c in cases:
        c = dict(c); c['behavior_signature'] = sig(c); c['validation'] = valid(c); enriched.append(c)
    good = [c for c in enriched if c['validation']['static_ok']]
    by = defaultdict(list)
    for c in sorted(good, key=lambda x: h(x.get('id',''))):
        by[c['behavior_signature']].append(c)
    clean = []
    for rows in by.values():
        clean.extend(rows[:a.max_per_signature])
    clean = sorted(clean, key=lambda x: x['id'])
    splits = {'train': [], 'val': [], 'test': []}
    for c in clean:
        splits[split(c['id'])].append(row(c))
    if len(clean) >= 10:
        for name in ['val','test']:
            if not splits[name] and splits['train']:
                splits[name].append(splits['train'].pop())
    write_jsonl(out / 'repo_grounded_mutation_cases_v0_2_static_validated.jsonl', enriched)
    write_jsonl(sft / 'clean_cases.jsonl', clean)
    for name, rows in splits.items():
        write_jsonl(sft / f'{name}.sft.jsonl', rows)
    eval_prompts = [{'id': r['id'], 'messages': r['messages'][:2], 'metadata': r['metadata']} for r in splits['test']]
    answer_key = [{'id': r['id'], 'answer': r['messages'][2], 'metadata': r['metadata']} for r in splits['test']]
    write_jsonl(sft / 'eval_prompts.jsonl', eval_prompts)
    write_jsonl(sft / 'eval_answer_key.jsonl', answer_key)
    stats = {
        'source_cases': len(cases),
        'static_valid_cases': len(good),
        'clean_cases_after_signature_cap': len(clean),
        'max_per_behavior_signature': a.max_per_signature,
        'unique_behavior_signatures_source': len(set(c['behavior_signature'] for c in enriched)),
        'unique_behavior_signatures_clean': len(set(c['behavior_signature'] for c in clean)),
        'split_counts': {k: len(v) for k, v in splits.items()},
        'by_pattern_clean': dict(Counter(c.get('pattern') for c in clean)),
        'by_error_type_clean': dict(Counter(c.get('error_type') for c in clean)),
        'warning': 'Static validation only; add executable JAX validation before public release.'
    }
    write_json(sft / 'dataset_stats.json', stats)
    (sft / 'DATASET_CARD.md').write_text('# JAXFixBench SFT Data\n\n```json\n' + json.dumps(stats, indent=2) + '\n```\n', encoding='utf-8')
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
