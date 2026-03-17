# March Mania 2026 — Autoresearch Program

This is an autonomous research loop for the Kaggle March Machine Learning Mania 2026 competition. The goal is to iteratively improve NCAA basketball tournament predictions by modifying features, feature engineering, and modeling techniques.

## Competition

- **Task**: Predict P(lower TeamID beats higher TeamID) for every possible NCAA tournament matchup (men's + women's)
- **Metric**: Brier score (mean squared error between predictions and 0/1 outcomes). **Lower is better.**
- **Submission**: CSV with `ID` column (format: `2026_TeamIDLow_TeamIDHigh`) and `Pred` column
- **Men's TeamIDs**: 1000–1999, **Women's TeamIDs**: 3000–3999
- **Deadline**: March 21, 2026 at 4PM UTC

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar17`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `program.md` — this file (research instructions)
   - `prepare.py` — **FIXED**. Data loading, evaluation harness, submission generation. Do not modify.
   - `experiment.py` — **THE FILE YOU MODIFY**. Feature engineering, model training, prediction.
4. **Verify data exists**: Check that `./data/` or `C:/dev/basketball_prediction/data/` contains the parquet files.
5. **Initialize results.tsv**: Create with just the header row. Baseline recorded after first run.
6. **Confirm and go**.

## Experimentation

Each experiment is a single run of `uv run experiment.py`. It builds features, cross-validates on historical tournaments (2016–2025, leave-one-season-out), and reports `val_brier`.

**What you CAN modify:**
- `experiment.py` — this is the ONLY file you edit. Everything is fair game:
  - Feature engineering (add/remove/transform features)
  - Feature creation (derived stats, rolling averages, rankings, strength of schedule, etc.)
  - Model type (logistic regression, XGBoost, LightGBM, random forest, ensemble, etc.)
  - Hyperparameters (regularization, learning rate, tree depth, etc.)
  - Training approach (stacking, blending, calibration, etc.)
  - Matchup feature construction (differences, ratios, interactions, etc.)

**What you CANNOT modify:**
- `prepare.py` — fixed evaluation harness. The `evaluate_brier()` function is the ground truth metric.
- `pyproject.toml` — no new dependencies.
- The evaluation methodology (leave-one-season-out cross-validation).

**The goal is simple: get the lowest val_brier.** Everything in `experiment.py` is fair game.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Always establish the baseline first by running experiment.py as-is.

## Available Data

The data directory contains parquet files with this structure:

### Men's data (M prefix, TeamIDs 1000–1999):
- `MTeams` — TeamID, TeamName, FirstD1Season, LastD1Season
- `MSeasons` — Season, DayZero, RegionW/X/Y/Z
- `MNCAATourneySeeds` — Season, Seed (e.g. "W01"), TeamID
- `MRegularSeasonCompactResults` — Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT
- `MRegularSeasonDetailedResults` — above + FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF (for both W and L)
- `MNCAATourneyCompactResults` / `MNCAATourneyDetailedResults` — same schema as regular season
- `MMasseyOrdinals` — Season, RankingDayNum, SystemName, TeamID, OrdinalRank (dozens of ranking systems)
- `MTeamCoaches` — Season, TeamID, FirstDayNum, LastDayNum, CoachName
- `MTeamConferences` — Season, TeamID, ConfAbbrev
- `MConferenceTourneyGames` — conference tournament results
- `MGameCities` — game locations
- `MSecondaryTourneyCompactResults` / `MSecondaryTourneyTeams` — NIT and other tournaments

### Women's data (W prefix, TeamIDs 3000–3999):
- Same files as men (except no MMasseyOrdinals for women)

### Shared:
- `Cities` — CityID, City, State
- `Conferences` — ConfAbbrev, Description
- `SampleSubmissionStage1` / `SampleSubmissionStage2` — ID, Pred

## Output format

The script prints a summary when done:

```
---
val_brier:        0.234567
train_brier:      0.198765
num_features:     42
model_type:       LogisticRegression
cv_seasons:       9
total_matchups:   1234
total_seconds:    12.3
```

Extract the key metric: `grep "^val_brier:" run.log`

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and 5 columns:

```
commit	val_brier	status	description
```

1. git commit hash (short, 7 chars)
2. val_brier achieved (e.g. 0.234567) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:
```
commit	val_brier	status	description
a1b2c3d	0.234567	keep	baseline (logistic regression + basic stats)
b2c3d4e	0.221000	keep	add seed number feature
c3d4e5f	0.240000	discard	switch to random forest (worse)
d4e5f6g	0.000000	crash	XGBoost feature interaction (import error)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar17`).

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Choose an experimental idea. Modify `experiment.py` with the change.
3. `git commit` the change.
4. Run the experiment: `uv run experiment.py > run.log 2>&1`
5. Read results: `grep "^val_brier:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback and attempt a fix.
7. Record results in results.tsv (do NOT commit results.tsv — leave it untracked).
8. If val_brier improved (lower), you "advance" the branch — keep the commit.
9. If val_brier is equal or worse, `git reset --hard HEAD~1` to discard.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder:
- Re-read the data files for new angles
- Try combining features that were individually weak
- Try different model architectures
- Try ensembling multiple models
- Try calibration techniques
- Try feature selection or dimensionality reduction
- Try interaction features

The loop runs until the human interrupts you.

## Research strategy: start simple, escalate gradually

**CRITICAL**: Always start with the fastest, simplest approaches first. The early experiments should be quick wins — adding obvious features, trying basic model swaps, tuning easy hyperparameters. Do NOT jump to complex solutions (ensembles, stacking, neural nets, elaborate feature engineering) until you have exhausted the simple ones.

**Phase 1 — Quick wins (do these first):**
- Add obvious features one at a time (seed, win%, scoring stats)
- Try the 2–3 most common model types (logistic regression, XGBoost, LightGBM)
- Basic hyperparameter tweaks (regularization strength, tree depth)
- Each experiment should be fast and targeted — change ONE thing at a time

**Phase 2 — Deeper feature engineering (once Phase 1 plateaus):**
- Detailed box score stats (shooting %, rebounds, turnovers, etc.)
- Derived stats (efficiency, pace, strength of schedule)
- Multi-season rolling features, momentum features
- Massey ordinal rankings integration
- Still keep individual experiments focused

**Phase 3 — Advanced modeling (only when you hit a metric floor with Phase 2):**
- Ensembling and stacking multiple models
- Probability calibration
- Feature interactions and polynomial features
- Hyperparameter search (grid/random on inner CV)
- Neural networks, Bayesian models
- These are slower and more complex — only worth it when simpler methods have truly plateaued

**How to know when to escalate**: If 3+ consecutive experiments at the current phase fail to improve val_brier, move to the next phase. If you're in Phase 3 and still stuck, try creative combinations or revisit earlier ideas with fresh perspective.

## Research ideas (organized by phase)

### Phase 1 — Feature additions (fast, one at a time):
1. Add detailed box score stats (FG%, 3P%, FT%, rebounds, assists, turnovers, steals, blocks)
2. Add offensive/defensive efficiency (points per possession)
3. Add strength of schedule (average opponent win pct)
4. Add conference strength features
5. Add momentum features (last N games performance)
6. Add Massey ordinal rankings (aggregate multiple ranking systems)
7. Add coach experience / tournament experience features
8. Add home/away/neutral performance splits
9. Add variance/consistency features (std of scoring)
10. Add pace features (possessions per game)

### Phase 2 — Deeper feature engineering (when Phase 1 plateaus):
1. Try ratios instead of (or in addition to) differences
2. Add offensive/defensive efficiency (points per possession)
3. Add strength of schedule (average opponent win pct)
4. Add conference strength features
5. Add momentum features (last N games performance)
6. Add Massey ordinal rankings (aggregate multiple ranking systems)
7. Try rolling averages over multiple seasons
8. Try weighting recent seasons more heavily

### Phase 3 — Advanced modeling (when Phase 2 plateaus):
1. Try XGBoost / LightGBM gradient boosting
2. Try ensemble of logistic regression + GBM
3. Try interaction features (e.g., seed * win_pct)
4. Try polynomial features
5. Try probability calibration (CalibratedClassifierCV)
6. Try stacking / blending multiple models
7. Try hyperparameter tuning (grid search on inner CV)
8. Try feature selection (mutual information, recursive elimination)
9. Try neural network (sklearn MLPClassifier)
10. Try Bayesian approaches (Bradley-Terry model)
