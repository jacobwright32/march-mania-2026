"""
March Mania 2026 — Experiment File (AGENT MODIFIES THIS FILE)
=============================================================
Usage: uv run experiment.py

This is the ONLY file the agent modifies. It contains:
  1. Feature engineering (build_team_features)
  2. Model training (train_model)
  3. Prediction (predict_matchups)

The evaluation harness in prepare.py is fixed and measures Brier score
on held-out tournament seasons via leave-one-season-out cross-validation.

Goal: minimize val_brier (lower is better). Perfect = 0.0, coin flip = 0.25.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from prepare import (
    DATA_DIR,
    EVAL_SEASONS,
    TARGET_SEASON,
    evaluate_brier,
    generate_submission,
    get_tourney_matchups,
    load_all_data,
    load_massey_ordinals,
    load_regular_season_detailed,
    load_regular_season_results,
    load_sample_submission,
    load_seeds,
    parse_submission_ids,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
# Modify this section to add/remove/change features.
# build_team_features() should return a DataFrame indexed by (Season, TeamID)
# with numeric feature columns.

def build_team_features(data: dict) -> pd.DataFrame:
    """Build per-team, per-season feature matrix from raw data.

    Returns:
        DataFrame with index (Season, TeamID) and feature columns.
    """
    reg_results = load_regular_season_results(data)
    seeds = load_seeds(data)

    # --- Win/loss record ---
    wins = reg_results.groupby(["Season", "WTeamID"]).size().reset_index(name="Wins")
    wins.rename(columns={"WTeamID": "TeamID"}, inplace=True)
    losses = reg_results.groupby(["Season", "LTeamID"]).size().reset_index(name="Losses")
    losses.rename(columns={"LTeamID": "TeamID"}, inplace=True)

    record = pd.merge(wins, losses, on=["Season", "TeamID"], how="outer").fillna(0)
    record["WinPct"] = record["Wins"] / (record["Wins"] + record["Losses"])
    record["Games"] = record["Wins"] + record["Losses"]

    # --- Scoring stats ---
    w_scores = reg_results.groupby(["Season", "WTeamID"]).agg(
        WPtsFor=("WScore", "mean"), WPtsAgainst=("LScore", "mean")
    ).reset_index().rename(columns={"WTeamID": "TeamID"})

    l_scores = reg_results.groupby(["Season", "LTeamID"]).agg(
        LPtsFor=("LScore", "mean"), LPtsAgainst=("WScore", "mean")
    ).reset_index().rename(columns={"LTeamID": "TeamID"})

    scoring = pd.merge(w_scores, l_scores, on=["Season", "TeamID"], how="outer").fillna(0)
    scoring["AvgPtsFor"] = (scoring["WPtsFor"] * record["Wins"] + scoring["LPtsFor"] * record["Losses"]) / record["Games"]
    scoring["AvgPtsAgainst"] = (scoring["WPtsAgainst"] * record["Wins"] + scoring["LPtsAgainst"] * record["Losses"]) / record["Games"]
    scoring["AvgPtsDiff"] = scoring["AvgPtsFor"] - scoring["AvgPtsAgainst"]

    features = pd.merge(record, scoring[["Season", "TeamID", "AvgPtsFor", "AvgPtsAgainst", "AvgPtsDiff"]],
                        on=["Season", "TeamID"], how="left")

    # --- Strength of schedule (avg opponent win pct) ---
    # Build opponent list for each team
    games_as_winner = reg_results[["Season", "WTeamID", "LTeamID"]].rename(
        columns={"WTeamID": "TeamID", "LTeamID": "OppID"})
    games_as_loser = reg_results[["Season", "LTeamID", "WTeamID"]].rename(
        columns={"LTeamID": "TeamID", "WTeamID": "OppID"})
    all_games = pd.concat([games_as_winner, games_as_loser], ignore_index=True)
    # Merge opponent win pct
    opp_wp_lookup = record[["Season", "TeamID", "WinPct"]].rename(columns={"TeamID": "OppID", "WinPct": "OppWinPct"})
    opp_wp = pd.merge(all_games, opp_wp_lookup, on=["Season", "OppID"], how="left")
    sos = opp_wp.groupby(["Season", "TeamID"])["OppWinPct"].mean().reset_index()
    sos.columns = ["Season", "TeamID", "SOS"]
    features = pd.merge(features, sos, on=["Season", "TeamID"], how="left")
    features["SOS"] = features["SOS"].fillna(0.5)

    # --- Massey ordinal rankings (men only) ---
    massey = load_massey_ordinals(data)
    if not massey.empty:
        # Use only the latest ranking day per season (closest to tournament)
        last_day = massey.groupby(["Season", "SystemName"])["RankingDayNum"].max().reset_index()
        last_day.columns = ["Season", "SystemName", "LastDay"]
        massey = massey.merge(last_day, on=["Season", "SystemName"])
        massey_end = massey[massey["RankingDayNum"] == massey["LastDay"]]
        massey_latest = massey_end.groupby(["Season", "TeamID"]).agg(
            MasseyMeanRank=("OrdinalRank", "mean")
        ).reset_index()
        features = pd.merge(features, massey_latest, on=["Season", "TeamID"], how="left")
        features["MasseyMeanRank"] = features["MasseyMeanRank"].fillna(150.0)

    # --- Seed (numeric) ---
    if not seeds.empty:
        seeds = seeds.copy()
        seeds["SeedNum"] = seeds["Seed"].str.extract(r"(\d+)").astype(float)
        seeds = seeds[["Season", "TeamID", "SeedNum"]]
        features = pd.merge(features, seeds, on=["Season", "TeamID"], how="left")
        features["SeedNum"] = features["SeedNum"].fillna(16.0)  # unseeded = 16
    else:
        features["SeedNum"] = 16.0

    features = features.set_index(["Season", "TeamID"])
    return features


# ---------------------------------------------------------------------------
# Feature columns used for modeling (edit to add/remove features)
# ---------------------------------------------------------------------------

FEATURE_COLS = ["AvgPtsDiff", "SeedNum", "MasseyMeanRank", "SOS"]


# ---------------------------------------------------------------------------
# Matchup feature builder
# ---------------------------------------------------------------------------

RATIO_COLS = ["SeedNum", "MasseyMeanRank"]


def build_matchup_features(team_features: pd.DataFrame, matchups: pd.DataFrame) -> pd.DataFrame:
    """Build pairwise matchup features from team-level features.

    For each matchup (Season, TeamID_low, TeamID_high), computes:
      - Difference: low_feature - high_feature  (for each feature)
      - Ratio: low_feature / high_feature  (for selected features)

    Returns DataFrame aligned with matchups index.
    """
    rows = []
    for _, m in matchups.iterrows():
        season = m["Season"]
        t_low, t_high = m["TeamID_low"], m["TeamID_high"]
        try:
            f_low = team_features.loc[(season, t_low)]
            f_high = team_features.loc[(season, t_high)]
        except KeyError:
            row = {f"{c}_diff": 0.0 for c in FEATURE_COLS}
            for c in RATIO_COLS:
                row[f"{c}_ratio"] = 1.0
            rows.append(row)
            continue
        row = {}
        for c in FEATURE_COLS:
            row[f"{c}_diff"] = f_low[c] - f_high[c]
        for c in RATIO_COLS:
            denom = f_high[c] if f_high[c] != 0 else 1.0
            row[f"{c}_ratio"] = f_low[c] / denom
        rows.append(row)
    return pd.DataFrame(rows, index=matchups.index)


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
# Modify this section to change the model type, hyperparameters, or training approach.

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Train a model and return (model, scaler) tuple.

    Args:
        X_train: matchup features (differences between team stats)
        y_train: binary outcome (1 if low-ID team won)

    Returns:
        (model, scaler) — scaler transforms features before prediction
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(C=0.05, max_iter=1000, solver="lbfgs")
    model.fit(X_scaled, y_train)

    return model, scaler


def predict_proba(model, scaler, X: pd.DataFrame) -> np.ndarray:
    """Return P(low-ID team wins) for each matchup."""
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


# ---------------------------------------------------------------------------
# Main: Cross-validation evaluation + submission generation
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("Loading data...")
    data = load_all_data()
    print(f"  Loaded {len(data)} data files from {DATA_DIR}")

    print("Building team features...")
    team_features = build_team_features(data)
    print(f"  Feature matrix: {team_features.shape[0]} team-seasons, {len(FEATURE_COLS)} features")

    # --- Cross-validation on historical tournaments ---
    print(f"\nCross-validating on {len(EVAL_SEASONS)} tournament seasons...")
    all_predictions = {}
    all_actuals = []

    for hold_out_season in EVAL_SEASONS:
        # Skip 2020 (COVID — no tournament)
        tourney = get_tourney_matchups(data, seasons=[hold_out_season])
        if tourney.empty:
            continue

        # Train on all other eval seasons
        train_seasons = [s for s in EVAL_SEASONS if s != hold_out_season]
        train_tourney = get_tourney_matchups(data, seasons=train_seasons)
        if train_tourney.empty:
            continue

        X_train = build_matchup_features(team_features, train_tourney)
        y_train = train_tourney["Result"]

        model, scaler = train_model(X_train, y_train)

        # Predict on held-out season
        X_test = build_matchup_features(team_features, tourney)
        preds = predict_proba(model, scaler, X_test)

        for i, (_, row) in enumerate(tourney.iterrows()):
            key = f"{int(row['Season'])}_{int(row['TeamID_low'])}_{int(row['TeamID_high'])}"
            all_predictions[key] = float(np.clip(preds[i], 0.01, 0.99))

        all_actuals.append(tourney)

    actuals_df = pd.concat(all_actuals, ignore_index=True)
    val_brier = evaluate_brier(all_predictions, actuals_df)

    # --- Train final model on all eval data for submission ---
    print("\nTraining final model on all historical data...")
    all_tourney = get_tourney_matchups(data, seasons=EVAL_SEASONS)
    all_tourney = all_tourney[all_tourney["Season"] != 2020]  # skip COVID
    X_all = build_matchup_features(team_features, all_tourney)
    y_all = all_tourney["Result"]
    final_model, final_scaler = train_model(X_all, y_all)

    # --- Generate submission for 2026 ---
    try:
        sample_sub = load_sample_submission(data, stage=2)
    except ValueError:
        try:
            sample_sub = load_sample_submission(data, stage=1)
        except ValueError:
            print("WARNING: No sample submission found, skipping submission generation")
            sample_sub = None

    if sample_sub is not None:
        sub_matchups = parse_submission_ids(sample_sub)
        X_sub = build_matchup_features(team_features, sub_matchups)
        sub_preds = predict_proba(final_model, final_scaler, X_sub)

        submission_preds = {}
        for i, (_, row) in enumerate(sub_matchups.iterrows()):
            submission_preds[row["ID"]] = float(np.clip(sub_preds[i], 0.01, 0.99))

        generate_submission(submission_preds)

    # --- Train Brier (for reference, not the optimization target) ---
    X_train_all = build_matchup_features(team_features, all_tourney)
    train_preds_arr = predict_proba(final_model, final_scaler, X_train_all)
    train_predictions = {}
    for i, (_, row) in enumerate(all_tourney.iterrows()):
        key = f"{int(row['Season'])}_{int(row['TeamID_low'])}_{int(row['TeamID_high'])}"
        train_predictions[key] = float(np.clip(train_preds_arr[i], 0.01, 0.99))
    train_brier = evaluate_brier(train_predictions, all_tourney)

    t_end = time.time()

    # --- Summary (autoresearch-style output) ---
    print("\n---")
    print(f"val_brier:        {val_brier:.6f}")
    print(f"train_brier:      {train_brier:.6f}")
    print(f"num_features:     {len(FEATURE_COLS)}")
    print(f"model_type:       {type(final_model).__name__}")
    print(f"cv_seasons:       {len([s for s in EVAL_SEASONS if s != 2020])}")
    print(f"total_matchups:   {len(actuals_df)}")
    print(f"total_seconds:    {t_end - t_start:.1f}")


if __name__ == "__main__":
    main()
