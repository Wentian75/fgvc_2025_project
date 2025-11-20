# Repository Guidelines

## Project Structure & Module Organization
- Root contains the ML project at `fgvc_2025_project/` and the upstream data repo `fgvc-comp-2025/`.
- Code: `fgvc_2025_project/src/` (`train.py`, `eval.py`, `infer.py`, `datasets.py`, `model.py`, `taxonomy_wandb_utils.py`, `utils.py`).
- Configs: `fgvc_2025_project/configs/` (e.g., `default.yaml`).
- Data: `fgvc_2025_project/data/` (training CSV + ROI images; do not commit).
- Outputs: `fgvc_2025_project/outputs/` (checkpoints, labels, model_config; keep together).
- Cache/Logs: `fgvc_2025_project/cache/`, `fgvc_2025_project/wandb/` (ignored).

## Build, Test, and Development Commands
- Setup: `cd fgvc_2025_project && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Train: `python -m src.train --config configs/default.yaml` (logs to W&B, saves `outputs/checkpoints/best.pt`).
- Eval: `python -m src.eval --config configs/default.yaml --model-dir outputs/checkpoints --data-root data/train`.
- Infer: `python -m src.infer --config configs/default.yaml --model-dir outputs/checkpoints --data-root data/test --output submission.csv`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, PEP8-ish style.
- Names: `snake_case` for functions/vars, `CapWords` for classes, modules `lower_snake.py`.
- Prefer type hints and small, focused functions. Avoid one-letter names.
- Keep changes minimal and localized; honor YAML config knobs instead of hard-coding.

## Testing Guidelines
- No formal test suite. Add lightweight smoke tests as needed (e.g., 1-epoch train on a tiny split, then `src.eval`).
- If adding tests, use `pytest` with files named `tests/test_*.py`; keep fast and deterministic (seeded).

## Commit & Pull Request Guidelines
- Commits: imperative, present tense (e.g., "Add early stopping for MHD"). Include the "why" in the body.
- PRs: describe scope, config used, dataset subset, hardware, and link to a W&B run. Include before/after metrics (Acc, MHD).
- Do not commit data, checkpoints, or secrets. Keep `best.pt`, `labels.json`, and `model_config.json` together when moving models.

## Security & Configuration Tips (Important)
- Never commit API keys; use `WANDB_API_KEY` env var. Data/cache/`wandb/` are ignored via `.gitignore`.
- If taxonomy seems wrong (e.g., Phylum confusion), delete `cache/taxonomy_reference.json` and rerun.

## Agent-Specific Instructions
- Follow these guidelines for any edits. Keep diffs focused, avoid repo-wide reformatting, and update `USAGE.md` when changing CLI or metrics.
