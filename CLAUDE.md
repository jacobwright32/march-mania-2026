# March Mania 2026

NCAA basketball tournament prediction for Kaggle March Machine Learning Mania 2026.

## Autoresearch Pattern

This repo follows the [autoresearch](https://github.com/karpathy/autoresearch) pattern for autonomous AI-driven experimentation.

### Key files:
- `prepare.py` — **FIXED**. Data loading, evaluation (Brier score), submission generation. Do not modify.
- `experiment.py` — **MODIFIABLE**. Feature engineering, model training, prediction. This is the only file the agent edits.
- `program.md` — Full research instructions and experiment loop protocol.
- `results.tsv` — Experiment log (untracked by git).

### Running an experiment:
```bash
uv run experiment.py > run.log 2>&1
grep "^val_brier:" run.log
```

### Metric:
- `val_brier` — Brier score via leave-one-season-out CV on 2016–2025 tournaments. **Lower is better.** Baseline: ~0.177.

### Experiment loop:
Read `program.md` for the full autonomous loop protocol. In short: modify experiment.py → commit → run → evaluate → keep if improved, discard if not → repeat.
