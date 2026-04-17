import pandas as pd
import numpy as np
from datetime import date


def _streak(wl_series: pd.Series) -> int:
    vals = wl_series.tolist()
    if not vals:
        return 0
    streak = 1 if vals[-1] else -1
    for v in reversed(vals[:-1]):
        if streak > 0 and v:
            streak += 1
        elif streak < 0 and not v:
            streak -= 1
        else:
            break
    return streak


def build_team_features(games: pd.DataFrame, is_home: int = 1) -> dict:
    if len(games) < 5:
        return {}

    g = games.sort_values("game_date").reset_index(drop=True)
    wl = (g["wl"] == "W").astype(int)
    last_date = pd.to_datetime(g["game_date"].iloc[-1]).date()
    rest = (date.today() - last_date).days

    home_mask = g["matchup"].str.contains(r"vs\.", na=False)
    home_wl = wl[home_mask]
    away_wl = wl[~home_mask]

    def safe_mean(s):
        return float(s.mean()) if len(s) > 0 else 0.5

    return {
        "is_home": is_home,
        "pts_last5": float(g["pts"].tail(5).mean()),
        "pts_against_last5": float(g["pts_against"].tail(5).mean()),
        "fg_pct_last5": float(g["fg_pct"].tail(5).mean()),
        "plus_minus_last5": float(g["plus_minus"].tail(5).mean()),
        "reb_last5": float(g["reb"].tail(5).mean()),
        "ast_last5": float(g["ast"].tail(5).mean()),
        "stl_last5": float(g["stl"].tail(5).mean()),
        "winrate_last5": float(wl.tail(5).mean()),
        "pts_last10": float(g["pts"].mean()),
        "pts_against_last10": float(g["pts_against"].mean()),
        "fg_pct_last10": float(g["fg_pct"].mean()),
        "plus_minus_last10": float(g["plus_minus"].mean()),
        "reb_last10": float(g["reb"].mean()),
        "ast_last10": float(g["ast"].mean()),
        "stl_last10": float(g["stl"].mean()),
        "winrate_last10": float(wl.mean()),
        "streak": _streak(wl),
        "rest_days": max(rest, 1),
        "home_winrate": safe_mean(home_wl),
        "away_winrate": safe_mean(away_wl),
        "opp_pts_last5": float(g["pts_against"].tail(5).mean()),
    }


def build_player_features(games: pd.DataFrame, is_home: int, defense_rating: float, position: int) -> dict:
    if len(games) < 5:
        return {}

    g = games.sort_values("game_date").reset_index(drop=True)
    fg5 = float(g["fg_pct"].tail(5).mean())
    pts5 = float(g["pts"].tail(5).mean())
    pts3 = float(g["pts"].tail(3).mean()) if len(g) >= 3 else pts5
    pts10 = float(g["pts"].mean())

    shot_vol = pts5 / (fg5 * 2) if fg5 > 0.01 else 0.0

    return {
        "is_home": is_home,
        "defense_rating": defense_rating,
        "pts_last5": pts5,
        "min_last5": float(g["min"].tail(5).mean()),
        "fg_pct_last5": fg5,
        "ft_pct_last5": float(g["ft_pct"].tail(5).mean()),
        "reb_last5": float(g["reb"].tail(5).mean()),
        "ast_last5": float(g["ast"].tail(5).mean()),
        "fg_pct_last10": float(g["fg_pct"].mean()),
        "ft_pct_last10": float(g["ft_pct"].mean()),
        "reb_last10": float(g["reb"].mean()),
        "pts_trend": pts3 - pts10,
        "shot_volume_last5": shot_vol,
        "position": position,
    }
