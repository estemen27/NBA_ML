import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

FEAT_TEAM = [
    "is_home", "pts_last5", "pts_against_last5", "fg_pct_last5", "plus_minus_last5",
    "reb_last5", "ast_last5", "stl_last5", "winrate_last5", "pts_last10",
    "pts_against_last10", "fg_pct_last10", "plus_minus_last10", "reb_last10",
    "ast_last10", "stl_last10", "winrate_last10", "streak", "rest_days",
    "home_winrate", "away_winrate", "opp_pts_last5",
]
FEAT_PLAYER = [
    "is_home", "defense_rating", "pts_last5", "min_last5", "fg_pct_last5",
    "ft_pct_last5", "reb_last5", "ast_last5", "fg_pct_last10", "ft_pct_last10",
    "reb_last10", "pts_trend", "shot_volume_last5", "position",
]

FEAT_TEAM_LABELS = {
    "pts_last5": "Pts Scored (L5)", "pts_against_last5": "Pts Allowed (L5)",
    "fg_pct_last5": "FG% (L5)", "plus_minus_last5": "+/- (L5)",
    "reb_last5": "Rebounds (L5)", "ast_last5": "Assists (L5)",
    "stl_last5": "Steals (L5)", "winrate_last5": "Win Rate (L5)",
    "pts_last10": "Pts Scored (L10)", "pts_against_last10": "Pts Allowed (L10)",
    "fg_pct_last10": "FG% (L10)", "plus_minus_last10": "+/- (L10)",
    "reb_last10": "Rebounds (L10)", "ast_last10": "Assists (L10)",
    "stl_last10": "Steals (L10)", "winrate_last10": "Win Rate (L10)",
    "streak": "Current Streak", "rest_days": "Rest Days",
    "home_winrate": "Home Win Rate", "away_winrate": "Away Win Rate",
    "opp_pts_last5": "Opp Pts (L5)", "is_home": "Home Game",
}
FEAT_PLAYER_LABELS = {
    "pts_last5": "Points (L5)", "min_last5": "Minutes (L5)",
    "fg_pct_last5": "FG% (L5)", "ft_pct_last5": "FT% (L5)",
    "reb_last5": "Rebounds (L5)", "ast_last5": "Assists (L5)",
    "fg_pct_last10": "FG% (L10)", "ft_pct_last10": "FT% (L10)",
    "reb_last10": "Rebounds (L10)", "pts_trend": "Scoring Trend",
    "shot_volume_last5": "Shot Volume (L5)", "defense_rating": "Opp Defense Rating",
    "is_home": "Home Game", "position": "Position",
}


@lru_cache(maxsize=1)
def load_classifier():
    return joblib.load(MODELS_DIR / "calibrated_classifier.joblib")


@lru_cache(maxsize=1)
def load_regressor():
    return joblib.load(MODELS_DIR / "best_player_regressor.joblib")


def predict_team(features: dict) -> tuple[float, int]:
    clf = load_classifier()
    df = pd.DataFrame([features])[FEAT_TEAM]
    prob = float(clf.predict_proba(df)[0, 1])
    pred = int(prob >= 0.5)
    return prob, pred


def predict_player(features: dict) -> float:
    reg = load_regressor()
    df = pd.DataFrame([features])[FEAT_PLAYER]
    return float(reg.predict(df)[0])


def shap_clf(features: dict) -> pd.DataFrame:
    clf = load_classifier()
    base_pipe = clf.calibrated_classifiers_[0].estimator
    sc = base_pipe.named_steps["sc"]
    lr = base_pipe.named_steps["clf"]
    df = pd.DataFrame([features])[FEAT_TEAM]
    x_scaled = sc.transform(df)[0]
    contributions = lr.coef_[0] * x_scaled
    labels = [FEAT_TEAM_LABELS.get(f, f) for f in FEAT_TEAM]
    result = pd.DataFrame({"feature": labels, "contribution": contributions})
    result["abs"] = result["contribution"].abs()
    return result.sort_values("abs", ascending=False).head(10).reset_index(drop=True)


def shap_reg(features: dict) -> pd.DataFrame:
    reg = load_regressor()
    sc = reg.named_steps["sc"]
    ridge = reg.named_steps["reg"]
    df = pd.DataFrame([features])[FEAT_PLAYER]
    x_scaled = sc.transform(df)[0]
    contributions = ridge.coef_ * x_scaled
    labels = [FEAT_PLAYER_LABELS.get(f, f) for f in FEAT_PLAYER]
    result = pd.DataFrame({"feature": labels, "contribution": contributions})
    result["abs"] = result["contribution"].abs()
    return result.sort_values("abs", ascending=False).head(8).reset_index(drop=True)
