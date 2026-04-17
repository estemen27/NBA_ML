import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import psycopg2
import shap
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DB = dict(host="localhost", port=5432, user="nba_user", password="nba_pass", dbname="nba_database")

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


def get_conn():
    return psycopg2.connect(**DB)


def build_predictions_classification(clf, test_team, teams_df):
    X = test_team[FEAT_TEAM]
    y = test_team["target"]
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    out = test_team[["team_id", "game_date"]].copy()
    out = out.merge(teams_df[["team_id", "full_name"]], on="team_id", how="left")
    out["probability"] = probs.round(4)
    out["prediction"] = preds
    out["actual"] = y.values
    out.rename(columns={"full_name": "team", "game_date": "date"}, inplace=True)
    return out[["team", "date", "probability", "prediction", "actual"]]


def build_predictions_regression(reg, test_player, players_df):
    X = test_player[FEAT_PLAYER]
    y = test_player["target"]
    preds = reg.predict(X)
    out = test_player[["player_id", "game_date"]].copy()
    out = out.merge(players_df[["player_id", "full_name"]], on="player_id", how="left")
    out["actual_pts"] = y.values.round(1)
    out["predicted_pts"] = preds.round(2)
    out["abs_error"] = np.abs(preds - y.values).round(2)
    out.rename(columns={"full_name": "player", "game_date": "date"}, inplace=True)
    return out[["player", "date", "actual_pts", "predicted_pts", "abs_error"]]


def compute_shap_classifier(clf, X_test):
    base_pipe = clf.calibrated_classifiers_[0].estimator
    sc = base_pipe.named_steps["sc"]
    lr = base_pipe.named_steps["clf"]
    X_scaled = sc.transform(X_test)
    explainer = shap.LinearExplainer(lr, X_scaled, feature_perturbation="interventional")
    vals = explainer.shap_values(X_scaled)
    mean_abs = np.abs(vals).mean(axis=0)
    df = pd.DataFrame({"feature": FEAT_TEAM, "mean_shap": mean_abs.round(5)})
    return df.sort_values("mean_shap", ascending=False).reset_index(drop=True)


def compute_shap_regressor(reg, X_test):
    sc = reg.named_steps["sc"]
    ridge = reg.named_steps["reg"]
    X_scaled = sc.transform(X_test)
    explainer = shap.LinearExplainer(ridge, X_scaled, feature_perturbation="interventional")
    vals = explainer.shap_values(X_scaled)
    mean_abs = np.abs(vals).mean(axis=0)
    df = pd.DataFrame({"feature": FEAT_PLAYER, "mean_shap": mean_abs.round(5)})
    return df.sort_values("mean_shap", ascending=False).reset_index(drop=True)


def build_team_stats(conn):
    return pd.read_sql(
        """
        SELECT
            t.full_name AS team,
            t.abbreviation,
            COUNT(g.game_id) AS games_played,
            ROUND(AVG(g.pts)::numeric, 1) AS avg_pts,
            ROUND(AVG(g.reb)::numeric, 1) AS avg_reb,
            ROUND(AVG(g.ast)::numeric, 1) AS avg_ast,
            ROUND(AVG(g.fg_pct)::numeric, 3) AS avg_fg_pct,
            ROUND(AVG(g.plus_minus)::numeric, 1) AS avg_plus_minus,
            SUM(CASE WHEN g.wl = 'W' THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN g.wl = 'L' THEN 1 ELSE 0 END) AS losses,
            ROUND(AVG(CASE WHEN g.wl = 'W' THEN 1.0 ELSE 0.0 END)::numeric, 3) AS win_pct,
            ROUND(AVG(opp.pts)::numeric, 1) AS avg_pts_allowed
        FROM fact_team_game_logs g
        JOIN dim_teams t ON g.team_id = t.team_id
        JOIN fact_team_game_logs opp ON g.game_id = opp.game_id AND opp.team_id != g.team_id
        GROUP BY t.team_id, t.full_name, t.abbreviation
        ORDER BY win_pct DESC
        """,
        conn,
    )


def build_player_stats(conn):
    return pd.read_sql(
        """
        SELECT
            p.full_name AS player,
            COUNT(g.game_id) AS games_played,
            ROUND(AVG(g.pts)::numeric, 1) AS avg_pts,
            ROUND(AVG(g.min)::numeric, 1) AS avg_min,
            ROUND(AVG(g.reb)::numeric, 1) AS avg_reb,
            ROUND(AVG(g.ast)::numeric, 1) AS avg_ast,
            ROUND(AVG(g.fg_pct)::numeric, 3) AS avg_fg_pct,
            ROUND(AVG(g.ft_pct)::numeric, 3) AS avg_ft_pct,
            ROUND(MAX(g.pts)::numeric, 0) AS max_pts,
            ROUND(AVG(g.stl)::numeric, 1) AS avg_stl,
            ROUND(AVG(g.blk)::numeric, 1) AS avg_blk
        FROM fact_player_game_logs g
        JOIN dim_players p ON g.player_id = p.player_id
        WHERE g.min >= 10
        GROUP BY p.player_id, p.full_name
        HAVING COUNT(g.game_id) >= 10
        ORDER BY avg_pts DESC
        """,
        conn,
    )


def main():
    test_team = pd.read_csv(BASE_DIR / "data/processed/team_classification_test.csv")
    test_player = pd.read_csv(BASE_DIR / "data/processed/player_regression_test.csv")

    clf = joblib.load(BASE_DIR / "models/calibrated_classifier.joblib")
    reg = joblib.load(BASE_DIR / "models/best_player_regressor.joblib")

    conn = get_conn()
    teams_df = pd.read_sql("SELECT team_id, full_name, abbreviation FROM dim_teams", conn)
    players_df = pd.read_sql("SELECT player_id, full_name FROM dim_players", conn)

    build_predictions_classification(clf, test_team, teams_df).to_csv(
        REPORTS_DIR / "predictions_classification.csv", index=False
    )

    build_predictions_regression(reg, test_player, players_df).to_csv(
        REPORTS_DIR / "predictions_regression.csv", index=False
    )

    model_comp = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    model_comp.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    compute_shap_classifier(clf, test_team[FEAT_TEAM]).to_csv(
        REPORTS_DIR / "shap_values_classifier.csv", index=False
    )
    compute_shap_regressor(reg, test_player[FEAT_PLAYER]).to_csv(
        REPORTS_DIR / "shap_values_regressor.csv", index=False
    )

    build_team_stats(conn).to_csv(REPORTS_DIR / "team_stats_summary.csv", index=False)
    build_player_stats(conn).to_csv(REPORTS_DIR / "player_stats_summary.csv", index=False)

    conn.close()


if __name__ == "__main__":
    main()
