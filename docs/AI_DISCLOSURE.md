# AI Disclosure

Most of this project's code and documentation was written by **Claude Code** (Anthropic's AI coding agent),
directed by the team. The team set the research question, the experimental design and the rules in `CLAUDE.md`,
reviewed every phase, and made all git commits themselves.

No result in this repository was hand-written by a human or an AI: every number in the README and dashboard is
read from `artifacts/results.json`, which the pipeline produces.

## Running log

| Phase | What was AI-generated | Human role |
|---|---|---|
| 0 · Scaffold | `requirements.txt`, `pyproject.toml`, `.gitignore` additions, `Makefile`, `scripts/download_data.sh`, `src/driftguard/config.py`, `src/driftguard/pipeline.py` CLI skeleton, `tests/test_config.py` | Chose the CLI design (`python -m driftguard.pipeline` with stage flags), reviewed and committed |
| 1 · Data | `src/driftguard/data.py` (load, merge, label mapping, degree features, validation), `--data` stage in `pipeline.py`, `tests/test_data.py` | Supplied the Kaggle data locally, reviewed and committed |
| 2 · Static baselines | `src/driftguard/splits.py`, `models.py`, `evaluate.py` (metrics + static baseline experiment), `--train` stage, `tests/test_splits.py`, `tests/test_metrics.py`, minimal `app/app.py` | Reviewed results and committed |

## Dependency notes

- `matplotlib` was added to the fixed stack: SHAP's summary plot needs it, and it is used to save the PNG
  figures in `artifacts/figures/` without an extra image-export dependency.
