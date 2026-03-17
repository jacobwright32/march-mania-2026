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
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

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

    # --- Offensive/Defensive efficiency (points per possession) ---
    detailed = load_regular_season_detailed(data)
    if not detailed.empty:
        # Possession estimate: FGA - OR + TO + 0.475*FTA
        # Winner possessions
        detailed["WPoss"] = detailed["WFGA"] - detailed["WOR"] + detailed["WTO"] + 0.475 * detailed["WFTA"]
        detailed["LPoss"] = detailed["LFGA"] - detailed["LOR"] + detailed["LTO"] + 0.475 * detailed["LFTA"]
        # Offensive efficiency = points / possessions * 100
        detailed["WOffEff"] = detailed["WScore"] / detailed["WPoss"].replace(0, 1) * 100
        detailed["LOffEff"] = detailed["LScore"] / detailed["LPoss"].replace(0, 1) * 100
        # Defensive efficiency = opponent points / own possessions * 100
        detailed["WDefEff"] = detailed["LScore"] / detailed["WPoss"].replace(0, 1) * 100
        detailed["LDefEff"] = detailed["WScore"] / detailed["LPoss"].replace(0, 1) * 100

        w_eff = detailed.groupby(["Season", "WTeamID"]).agg(
            WOE=("WOffEff", "mean"), WDE=("WDefEff", "mean")
        ).reset_index().rename(columns={"WTeamID": "TeamID"})
        l_eff = detailed.groupby(["Season", "LTeamID"]).agg(
            LOE=("LOffEff", "mean"), LDE=("LDefEff", "mean")
        ).reset_index().rename(columns={"LTeamID": "TeamID"})

        eff = pd.merge(w_eff, l_eff, on=["Season", "TeamID"], how="outer").fillna(0)
        eff = pd.merge(eff, record[["Season", "TeamID", "Wins", "Losses", "Games"]],
                         on=["Season", "TeamID"], how="left")
        eff["OffEff"] = (eff["WOE"] * eff["Wins"] + eff["LOE"] * eff["Losses"]) / eff["Games"]
        eff["DefEff"] = (eff["WDE"] * eff["Wins"] + eff["LDE"] * eff["Losses"]) / eff["Games"]
        eff["NetEff"] = eff["OffEff"] - eff["DefEff"]

        eff["EffRatio"] = np.log(eff["OffEff"] / eff["DefEff"].replace(0, 1) + 1e-6)
        features = pd.merge(features, eff[["Season", "TeamID", "NetEff", "EffRatio"]],
                             on=["Season", "TeamID"], how="left")
        features["NetEff"] = features["NetEff"].fillna(0.0)
        features["EffRatio"] = features["EffRatio"].fillna(0.0)

        # --- Turnover rate ---
        detailed["WTORate"] = detailed["WTO"] / detailed["WPoss"].replace(0, 1)
        detailed["LTORate"] = detailed["LTO"] / detailed["LPoss"].replace(0, 1)
        w_to = detailed.groupby(["Season", "WTeamID"])["WTORate"].mean().reset_index()
        w_to.columns = ["Season", "TeamID", "WTORate"]
        l_to = detailed.groupby(["Season", "LTeamID"])["LTORate"].mean().reset_index()
        l_to.columns = ["Season", "TeamID", "LTORate"]
        to_merged = pd.merge(w_to, l_to, on=["Season", "TeamID"], how="outer").fillna(0)
        to_merged = pd.merge(to_merged, record[["Season", "TeamID", "Wins", "Losses", "Games"]],
                              on=["Season", "TeamID"], how="left")
        to_merged["TORate"] = (to_merged["WTORate"] * to_merged["Wins"] +
                                to_merged["LTORate"] * to_merged["Losses"]) / to_merged["Games"]
        features = pd.merge(features, to_merged[["Season", "TeamID", "TORate"]],
                             on=["Season", "TeamID"], how="left")
        features["TORate"] = features["TORate"].fillna(features["TORate"].median())

    # --- Opponent-adjusted net efficiency (iterative, KenPom-style) ---
    # Start with raw NetEff per team, then iteratively adjust by opponent quality
    if "NetEff" in features.columns:
        adj_eff = features[["Season", "TeamID", "NetEff"]].copy() if "Season" in features.columns else None
    # Build from record + eff which are not yet indexed
    adj_eff_df = eff[["Season", "TeamID", "OffEff", "DefEff"]].copy() if not detailed.empty else None
    if adj_eff_df is not None:
        games_w = reg_results[["Season", "WTeamID", "LTeamID"]].rename(
            columns={"WTeamID": "TeamID", "LTeamID": "OppID"})
        games_l = reg_results[["Season", "LTeamID", "WTeamID"]].rename(
            columns={"LTeamID": "TeamID", "WTeamID": "OppID"})
        all_g = pd.concat([games_w, games_l], ignore_index=True)

        # Iterate: adjust OffEff by opponent DefEff and vice versa
        curr = adj_eff_df.copy()
        for _ in range(5):
            opp_lookup = curr.rename(columns={"TeamID": "OppID", "OffEff": "OppOE", "DefEff": "OppDE"})
            merged = pd.merge(all_g, opp_lookup, on=["Season", "OppID"], how="left")
            avg_opp = merged.groupby(["Season", "TeamID"]).agg(
                AvgOppOE=("OppOE", "mean"), AvgOppDE=("OppDE", "mean")
            ).reset_index()
            curr = pd.merge(adj_eff_df, avg_opp, on=["Season", "TeamID"], how="left")
            # Adjust: good offense against good defense is better
            curr["AdjOE"] = curr["OffEff"] + (curr["AvgOppDE"].fillna(100) - 100)
            curr["AdjDE"] = curr["DefEff"] - (curr["AvgOppOE"].fillna(100) - 100)
            curr["OffEff"] = curr["AdjOE"]
            curr["DefEff"] = curr["AdjDE"]

        curr["KenPomNetEff"] = curr["AdjOE"] - curr["AdjDE"]
        features = pd.merge(features, curr[["Season", "TeamID", "KenPomNetEff"]],
                             on=["Season", "TeamID"], how="left")
        features["KenPomNetEff"] = features["KenPomNetEff"].fillna(0.0)

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

    # --- Second-order SOS (avg opponent's SOS) ---
    opp_sos_lookup = sos.rename(columns={"TeamID": "OppID", "SOS": "OppSOS"})
    opp_sos = pd.merge(all_games, opp_sos_lookup, on=["Season", "OppID"], how="left")
    sos2 = opp_sos.groupby(["Season", "TeamID"])["OppSOS"].mean().reset_index()
    sos2.columns = ["Season", "TeamID", "SOS2"]
    features = pd.merge(features, sos2, on=["Season", "TeamID"], how="left")
    features["SOS2"] = features["SOS2"].fillna(0.5)

    # --- Third-order SOS (avg opponent's SOS2) ---
    opp_sos2_lookup = sos2.rename(columns={"TeamID": "OppID", "SOS2": "OppSOS2"})
    opp_sos3 = pd.merge(all_games, opp_sos2_lookup, on=["Season", "OppID"], how="left")
    sos3 = opp_sos3.groupby(["Season", "TeamID"])["OppSOS2"].mean().reset_index()
    sos3.columns = ["Season", "TeamID", "SOS3"]
    features = pd.merge(features, sos3, on=["Season", "TeamID"], how="left")
    features["SOS3"] = features["SOS3"].fillna(0.5)

    # --- Fourth-order SOS ---
    opp_sos3_lookup = sos3.rename(columns={"TeamID": "OppID", "SOS3": "OppSOS3"})
    opp_sos4 = pd.merge(all_games, opp_sos3_lookup, on=["Season", "OppID"], how="left")
    sos4 = opp_sos4.groupby(["Season", "TeamID"])["OppSOS3"].mean().reset_index()
    sos4.columns = ["Season", "TeamID", "SOS4"]
    features = pd.merge(features, sos4, on=["Season", "TeamID"], how="left")
    features["SOS4"] = features["SOS4"].fillna(0.5)

    # --- Fifth/Sixth-order SOS ---
    opp_sos4_lookup = sos4.rename(columns={"TeamID": "OppID", "SOS4": "OppSOS4"})
    opp_sos5 = pd.merge(all_games, opp_sos4_lookup, on=["Season", "OppID"], how="left")
    sos5 = opp_sos5.groupby(["Season", "TeamID"])["OppSOS4"].mean().reset_index()
    sos5.columns = ["Season", "TeamID", "SOS5"]
    features = pd.merge(features, sos5, on=["Season", "TeamID"], how="left")
    features["SOS5"] = features["SOS5"].fillna(0.5)

    opp_sos5_lookup = sos5.rename(columns={"TeamID": "OppID", "SOS5": "OppSOS5"})
    opp_sos6 = pd.merge(all_games, opp_sos5_lookup, on=["Season", "OppID"], how="left")
    sos6 = opp_sos6.groupby(["Season", "TeamID"])["OppSOS5"].mean().reset_index()
    sos6.columns = ["Season", "TeamID", "SOS6"]
    features = pd.merge(features, sos6, on=["Season", "TeamID"], how="left")
    features["SOS6"] = features["SOS6"].fillna(0.5)

    # --- Efficiency-based SOS (iterated, using EffRatio instead of NetEff) ---
    if "EffRatio" in features.columns:
        eff_sos_base = features[["Season", "TeamID", "EffRatio"]].copy() if "Season" in features.columns else None
    else:
        eff_sos_base = None
    if eff_sos_base is None and not detailed.empty:
        eff_sos_base = eff[["Season", "TeamID"]].copy()
        eff_sos_base["EffRatio"] = np.log(eff["OffEff"] / eff["DefEff"].replace(0, 1) + 1e-6)
    if eff_sos_base is not None:
        prev = eff_sos_base.copy()
        for order in range(1, 7):
            col = f"EffSOS{order}"
            opp_lkp = prev.rename(columns={"TeamID": "OppID", prev.columns[-1]: "OppVal"})
            m = pd.merge(all_games, opp_lkp[["Season", "OppID", "OppVal"]], on=["Season", "OppID"], how="left")
            new_sos = m.groupby(["Season", "TeamID"])["OppVal"].mean().reset_index()
            new_sos.columns = ["Season", "TeamID", col]
            features = pd.merge(features, new_sos, on=["Season", "TeamID"], how="left")
            features[col] = features[col].fillna(0.0)
            prev = new_sos

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

        # Massey percentile (normalized by number of teams per season)
        max_rank_per_season = massey_latest.groupby("Season")["MasseyMeanRank"].max().reset_index()
        max_rank_per_season.columns = ["Season", "MaxRank"]
        features = pd.merge(features, max_rank_per_season, on="Season", how="left")
        features["MasseyPctile"] = features["MasseyMeanRank"] / features["MaxRank"].fillna(350)
        features.drop(columns=["MaxRank"], inplace=True)

    # --- Seed (numeric) ---
    if not seeds.empty:
        seeds = seeds.copy()
        seeds["SeedNum"] = seeds["Seed"].str.extract(r"(\d+)").astype(float)
        seeds = seeds[["Season", "TeamID", "SeedNum"]]
        features = pd.merge(features, seeds, on=["Season", "TeamID"], how="left")
        features["SeedNum"] = features["SeedNum"].fillna(16.0)  # unseeded = 16
    else:
        features["SeedNum"] = 16.0

    # --- Historical seed win rate ---
    tourney_results = pd.concat([data.get(k, pd.DataFrame()) for k in ["MNCAATourneyCompactResults", "WNCAATourneyCompactResults"]], ignore_index=True)
    if not tourney_results.empty and not seeds.empty:
        all_seeds = load_seeds(data)
        all_seeds["SeedNum"] = all_seeds["Seed"].str.extract(r"(\d+)").astype(float)
        # Merge seeds for both winner and loser
        tr = tourney_results.merge(all_seeds[["Season", "TeamID", "SeedNum"]].rename(columns={"TeamID": "WTeamID", "SeedNum": "WSeed"}),
                                     on=["Season", "WTeamID"], how="left")
        tr = tr.merge(all_seeds[["Season", "TeamID", "SeedNum"]].rename(columns={"TeamID": "LTeamID", "SeedNum": "LSeed"}),
                        on=["Season", "LTeamID"], how="left")
        # Compute win rate per seed
        w_seed_wins = tr.groupby("WSeed").size().reset_index(name="Wins")
        l_seed_losses = tr.groupby("LSeed").size().reset_index(name="Losses")
        seed_wp = pd.merge(w_seed_wins.rename(columns={"WSeed": "SeedNum"}),
                            l_seed_losses.rename(columns={"LSeed": "SeedNum"}),
                            on="SeedNum", how="outer").fillna(0)
        seed_wp["SeedHistWinPct"] = seed_wp["Wins"] / (seed_wp["Wins"] + seed_wp["Losses"]).replace(0, 1)
        features = pd.merge(features, seed_wp[["SeedNum", "SeedHistWinPct"]], on="SeedNum", how="left")
        features["SeedHistWinPct"] = features["SeedHistWinPct"].fillna(0.5)

    # --- Pythagorean win expectation ---
    exp = 13.91  # basketball exponent
    features["PythWinPct"] = features["AvgPtsFor"] ** exp / (features["AvgPtsFor"] ** exp + features["AvgPtsAgainst"] ** exp + 1e-10)

    # --- Win rate vs better opponents (upset resistance) ---
    opp_wp_full = record[["Season", "TeamID", "WinPct"]].copy()
    # Games where team won against a better opponent (higher WinPct)
    w_upsets = reg_results[["Season", "WTeamID", "LTeamID"]].rename(
        columns={"WTeamID": "TeamID", "LTeamID": "OppID"})
    w_upsets = pd.merge(w_upsets, opp_wp_full.rename(columns={"TeamID": "OppID", "WinPct": "OppWP"}),
                         on=["Season", "OppID"], how="left")
    w_upsets = pd.merge(w_upsets, opp_wp_full, on=["Season", "TeamID"], how="left")
    w_upsets["BeatBetter"] = (w_upsets["OppWP"] > w_upsets["WinPct"]).astype(int)
    beat_better = w_upsets.groupby(["Season", "TeamID"]).agg(
        BeatBetterCount=("BeatBetter", "sum"), TotalWins=("BeatBetter", "count")
    ).reset_index()
    beat_better["UpsetRate"] = beat_better["BeatBetterCount"] / beat_better["TotalWins"].replace(0, 1)
    features = pd.merge(features, beat_better[["Season", "TeamID", "UpsetRate"]],
                         on=["Season", "TeamID"], how="left")
    features["UpsetRate"] = features["UpsetRate"].fillna(0.0)

    # --- Adjusted metrics ---
    features["AdjPtsDiff"] = features["AvgPtsDiff"] * features["SOS"]
    features["AdjNetEff"] = features["NetEff"] * features["SOS"]

    # --- Per-season standardization of key features ---
    season_norm_cols = ["AvgPtsDiff", "NetEff", "AdjNetEff", "KenPomNetEff", "EffSOS1"]
    for col in season_norm_cols:
        if col in features.columns:
            season_mean = features.groupby("Season")[col].transform("mean")
            season_std = features.groupby("Season")[col].transform("std").replace(0, 1)
            features[f"{col}_zseas"] = (features[col] - season_mean) / season_std

    # --- Interaction features ---
    features["KenPom_x_SeedWP"] = features["KenPomNetEff"] * features["SeedHistWinPct"]
    features["PtsDiff_x_SeedWP"] = features["AvgPtsDiff"] * features["SeedHistWinPct"]
    features["NetEff_x_SeedWP"] = features["NetEff"] * features["SeedHistWinPct"]
    features["Upset_x_SeedWP"] = features["UpsetRate"] * features["SeedHistWinPct"]
    features["EffRatio_x_SeedWP"] = features["EffRatio"] * features["SeedHistWinPct"]

    features = features.set_index(["Season", "TeamID"])
    return features


# ---------------------------------------------------------------------------
# Feature columns used for modeling (edit to add/remove features)
# ---------------------------------------------------------------------------

FEATURE_COLS = ["MasseyMeanRank", "EffRatio", "SOS",
                "EffSOS2", "EffSOS3", "EffSOS4", "EffSOS5", "EffSOS6",
                "NetEff_zseas", "KenPomNetEff_zseas",
                "SeedHistWinPct", "MasseyPctile", "KenPom_x_SeedWP", "NetEff_x_SeedWP",
                "UpsetRate", "Upset_x_SeedWP", "EffSOS1_zseas", "EffRatio_x_SeedWP",
                "PythWinPct"]


# ---------------------------------------------------------------------------
# Matchup feature builder
# ---------------------------------------------------------------------------

RATIO_COLS = []


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
    scaler = QuantileTransformer(output_distribution="normal", n_quantiles=50, random_state=42)
    X_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(C=0.03, max_iter=1000, solver="lbfgs")
    lr.fit(X_scaled, y_train)

    lgb_models = []
    for seed in [42, 123, 456]:
        m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.02,
            num_leaves=8, min_child_samples=10,
            reg_alpha=1.0, reg_lambda=2.0,
            subsample=0.8, colsample_bytree=0.6,
            verbose=-1, random_state=seed,
        )
        m.fit(X_scaled, y_train)
        lgb_models.append(m)

    return (lr, lgb_models), scaler


LR_WEIGHT = 0.65


def predict_proba(models, scaler, X: pd.DataFrame) -> np.ndarray:
    """Return P(low-ID team wins) — ensemble with bagged LGB."""
    lr, lgb_models = models
    X_scaled = scaler.transform(X)
    lr_preds = lr.predict_proba(X_scaled)[:, 1]
    lgb_preds = np.mean([m.predict_proba(X_scaled)[:, 1] for m in lgb_models], axis=0)
    return LR_WEIGHT * lr_preds + (1 - LR_WEIGHT) * lgb_preds


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
