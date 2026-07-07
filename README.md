# JAX Fine-Tune Mine

Mining scaffold for building **JAXFixBench / XLADoctor** data.

The goal is to discover public JAX/Flax/Optax/XLA-related repositories, clone a capped subset, extract clean JAX snippets, mine recurring bug-fix patterns, and build repo-grounded mutation cases for a JAX debugging fine-tune/eval dataset.

## Quick start

Run locally:

```bash
python -m pip install -r requirements-mining.txt
python scripts/mining/mine_jaxfixbench.py discover --out-dir data/mined
python scripts/mining/mine_jaxfixbench.py plan --out-dir data/mined
python scripts/mining/mine_jaxfixbench.py clone --out-dir data/mined --max-clones 60
python scripts/mining/mine_jaxfixbench.py extract --out-dir data/mined
python scripts/mining/mine_jaxfixbench.py score --out-dir data/mined
python scripts/mining/mine_jaxfixbench.py patterns --out-dir data/mined
python scripts/mining/mine_jaxfixbench.py mutations --out-dir data/mined
python scripts/mining/mine_jaxfixbench.py validate --out-dir data/mined
python scripts/mining/mine_jaxfixbench.py manifest --out-dir data/mined
```

Or run the GitHub Actions workflow manually from the **Actions** tab.

## Outputs

Typical outputs under `data/mined/`:

- `repo_candidates_checkpoint.jsonl`
- `repo_candidates_final.jsonl`
- `repo_clone_plan.json`
- `repo_clone_log.jsonl`
- `raw_snippets.jsonl`
- `clean_snippets.jsonl`
- `pattern_hits.jsonl`
- `pattern_hits_capped.jsonl`
- `repo_grounded_mutation_cases_v0_2.jsonl`
- `repo_grounded_mutation_validation_report.json`
- `manifest.json`

## Notes

This scaffold is designed for public-source mining. Respect source licenses and keep provenance fields when turning mined examples into dataset rows.
