"""
Fixed data loading and evaluation harness for March Mania 2026.
DO NOT MODIFY THIS FILE. The agent modifies only experiment.py.

Provides:
  - load_all_data()         -> dict of DataFrames (all parquet files)
  - get_tourney_actuals()   -> historical tournament results as matchup outcomes
  - evaluate_brier()        -> Brier score (lower is better)
  - generate_submission()   -> write submission CSV
  - DATA_DIR                -> path to parquet data
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data directory — looks for data/ in this repo, falls back to basketball_prediction
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).parent
DATA_DIR = REPO_DIR / "data"

if not DATA_DIR.exists():
    FALLBACK = Path("C:/dev/basketball_prediction/data")
    if FALLBACK.exists():
        DATA_DIR = FALLBACK
    else:
        print("ERROR: No data directory found. Place parquet files in ./data/ or ensure C:/dev/basketball_prediction/data exists.")
        sys.exit(1)

# Target season for predictions
TARGET_SEASON = 2026

# Seasons to use for cross-validation evaluation (recent tournaments with known outcomes)
EVAL_SEASONS = list(range(2016, 2026))  # 2016-2025 (excluding 2020 — COVID cancelled)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all parquet files into a dict keyed by filename stem."""
    data = {}
    for subdir in [DATA_DIR, DATA_DIR / "men", DATA_DIR / "women"]:
        if not subdir.exists():
            continue
        for f in sorted(subdir.glob("*.parquet")):
            data[f.stem] = pd.read_parquet(f)
    return data


def load_tourney_results(data: dict) -> pd.DataFrame:
    """Combine men's and women's tournament compact results into one DataFrame.
    Returns columns: Season, WTeamID, LTeamID, WScore, LScore
    """
    frames = []
    for key in ["MNCAATourneyCompactResults", "WNCAATourneyCompactResults"]:
        if key in data:
            frames.append(data[key])
    if not frames:
        raise ValueError("No tournament results found in data")
    return pd.concat(frames, ignore_index=True)


def load_regular_season_results(data: dict) -> pd.DataFrame:
    """Combine men's and women's regular season compact results."""
    frames = []
    for key in ["MRegularSeasonCompactResults", "WRegularSeasonCompactResults"]:
        if key in data:
            frames.append(data[key])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_regular_season_detailed(data: dict) -> pd.DataFrame:
    """Combine men's and women's regular season detailed results."""
    frames = []
    for key in ["MRegularSeasonDetailedResults", "WRegularSeasonDetailedResults"]:
        if key in data:
            frames.append(data[key])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_seeds(data: dict) -> pd.DataFrame:
    """Combine men's and women's tournament seeds."""
    frames = []
    for key in ["MNCAATourneySeeds", "WNCAATourneySeeds"]:
        if key in data:
            frames.append(data[key])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_massey_ordinals(data: dict) -> pd.DataFrame:
    """Load Massey ordinals (men only — women don't have this)."""
    return data.get("MMasseyOrdinals", pd.DataFrame())


def load_sample_submission(data: dict, stage: int = 1) -> pd.DataFrame:
    """Load sample submission file."""
    key = f"SampleSubmissionStage{stage}"
    if key in data:
        return data[key]
    raise ValueError(f"Sample submission {key} not found")


# ---------------------------------------------------------------------------
# Tournament matchup creation for evaluation
# ---------------------------------------------------------------------------

def get_tourney_matchups(data: dict, seasons: list[int] | None = None) -> pd.DataFrame:
    """Build actual tournament matchup outcomes for evaluation.

    Returns DataFrame with columns:
        Season, TeamID_low, TeamID_high, Result (1 if low-ID team won, 0 otherwise)
    """
    tourney = load_tourney_results(data)
    if seasons is not None:
        tourney = tourney[tourney["Season"].isin(seasons)]

    rows = []
    for _, game in tourney.iterrows():
        w, l = int(game["WTeamID"]), int(game["LTeamID"])
        low, high = min(w, l), max(w, l)
        result = 1 if w == low else 0  # 1 if low-ID team won
        rows.append({"Season": int(game["Season"]), "TeamID_low": low, "TeamID_high": high, "Result": result})

    return pd.DataFrame(rows)


def parse_submission_ids(sample_sub: pd.DataFrame) -> pd.DataFrame:
    """Parse submission IDs into Season, TeamID_low, TeamID_high."""
    parts = sample_sub["ID"].str.split("_", expand=True).astype(int)
    return pd.DataFrame({
        "ID": sample_sub["ID"],
        "Season": parts[0],
        "TeamID_low": parts[1],
        "TeamID_high": parts[2],
    })


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_brier(predictions: dict[str, float], actuals: pd.DataFrame) -> float:
    """Compute Brier score (mean squared error) on tournament matchups.

    Args:
        predictions: dict mapping "Season_TeamLow_TeamHigh" -> P(low wins)
        actuals: DataFrame with Season, TeamID_low, TeamID_high, Result

    Returns:
        Brier score (lower is better). Perfect = 0.0, coin flip = 0.25
    """
    errors = []
    for _, row in actuals.iterrows():
        key = f"{int(row['Season'])}_{int(row['TeamID_low'])}_{int(row['TeamID_high'])}"
        pred = predictions.get(key, 0.5)  # default to 0.5 if missing
        actual = row["Result"]
        errors.append((pred - actual) ** 2)
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------------

def generate_submission(predictions: dict[str, float], output_path: str = "output/submission.csv"):
    """Write Kaggle submission CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = [{"ID": k, "Pred": v} for k, v in sorted(predictions.items())]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path} ({len(df)} rows)")
    return df
