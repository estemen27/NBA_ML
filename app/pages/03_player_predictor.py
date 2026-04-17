import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.styles import inject_css, sidebar_brand, plotly_layout
from utils.db_connection import query_df
from utils.model_loader import predict_player, shap_reg
from utils.feature_engineering import build_player_features

MODEL_RMSE = 6.55

inject_css()
sidebar_brand()

st.markdown(
    """
    <div style="margin-bottom:28px;">
        <div style="font-size:0.72rem;color:#C9082A;text-transform:uppercase;letter-spacing:4px;font-weight:700;">Motor de Predicción en Tiempo Real</div>
        <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:3rem;font-weight:900;color:#FFFFFF;line-height:1.1;margin-top:6px;">
            PREDICTOR DE PUNTOS
        </div>
        <div style="font-size:0.9rem;color:#8FA3BF;margin-top:6px;">
            Selecciona un jugador y el modelo consulta sus últimos partidos desde la base de datos para predecir su anotación en el próximo encuentro.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    players_df = query_df(
        """
        SELECT p.player_id, p.full_name,
               COALESCE(s.team_abbrev, 'N/A') AS team,
               COUNT(g.game_id) AS games
        FROM dim_players p
        LEFT JOIN fact_player_game_logs g ON p.player_id = g.player_id AND g.min >= 10
        LEFT JOIN fact_player_season_stats s ON p.player_id = s.player_id
        WHERE p.is_active = true
        GROUP BY p.player_id, p.full_name, s.team_abbrev
        HAVING COUNT(g.game_id) >= 10
        ORDER BY AVG(g.pts) DESC NULLS LAST
        """
    )
    teams_df = query_df("SELECT team_id, full_name, abbreviation FROM dim_teams ORDER BY full_name")
except Exception as e:
    st.error(f"Error de conexión a la base de datos: {e}")
    st.stop()

player_options = players_df["full_name"].tolist()

col_sel, col_loc, col_opp = st.columns([2, 1, 1])
with col_sel:
    selected_player = st.selectbox("Seleccionar Jugador", player_options, index=0)
with col_loc:
    game_location = st.radio("Sede del Partido", ["Local", "Visitante"], horizontal=True)
with col_opp:
    opp_options = ["Media de la Liga"] + teams_df["full_name"].tolist()
    selected_opp = st.selectbox("Equipo Rival", opp_options, index=0)

is_home = 1 if game_location == "Local" else 0
player_row = players_df[players_df["full_name"] == selected_player].iloc[0]
player_id = int(player_row["player_id"])

try:
    if selected_opp != "Media de la Liga":
        opp_team_id = int(teams_df[teams_df["full_name"] == selected_opp]["team_id"].iloc[0])
        defense_df = query_df(
            """
            SELECT AVG(opp.pts) AS defense_rating
            FROM fact_team_game_logs t
            JOIN fact_team_game_logs opp ON t.game_id = opp.game_id AND opp.team_id != t.team_id
            WHERE t.team_id = %s
            """,
            params=(opp_team_id,),
        )
        defense_rating = float(defense_df["defense_rating"].iloc[0]) if not defense_df.empty else 115.0
    else:
        avg_df = query_df("SELECT AVG(pts) AS avg_pts FROM fact_team_game_logs")
        defense_rating = float(avg_df["avg_pts"].iloc[0])
except Exception:
    defense_rating = 115.0

try:
    position_df = query_df(
        "SELECT ast, blk, reb FROM fact_player_season_stats WHERE player_id = %s",
        params=(player_id,),
    )
    if not position_df.empty:
        row = position_df.iloc[0]
        if float(row["ast"]) >= 4.5 and float(row["blk"]) <= 0.5:
            position = 0
        elif float(row["blk"]) >= 1.0 and float(row["reb"]) >= 7.0:
            position = 2
        else:
            position = 1
    else:
        position = 1
except Exception:
    position = 1

try:
    games = query_df(
        """
        SELECT game_date, wl, pts, min, reb, ast, stl, fg_pct, ft_pct, plus_minus, matchup
        FROM fact_player_game_logs
        WHERE player_id = %s AND min >= 5
        ORDER BY game_date DESC
        LIMIT 20
        """,
        params=(player_id,),
    )
except Exception as e:
    st.error(f"No se pudieron obtener los datos del jugador: {e}")
    st.stop()

if len(games) < 5:
    st.warning(f"{selected_player} tiene menos de 5 partidos válidos en la base de datos.")
    st.stop()

features = build_player_features(games, is_home, defense_rating, position)
if not features:
    st.warning("Historial insuficiente para generar una predicción.")
    st.stop()

predicted_pts = predict_player(features)
shap_df = shap_reg(features)

ci_half = MODEL_RMSE * 1.0
ci_low = max(0, predicted_pts - ci_half)
ci_high = predicted_pts + ci_half
player_avg = float(games["pts"].head(10).mean())

pos_labels = {0: "Base / Escolta", 1: "Alero", 2: "Pívot"}

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_pred, col_shap = st.columns([1, 1])

with col_pred:
    st.markdown(
        f"""
        <div class="prediction-hero">
            <div style="font-size:0.72rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">
                {selected_player} &nbsp;·&nbsp; {pos_labels.get(position, 'Jugador')}
                &nbsp;·&nbsp; {'Local' if is_home else 'Visitante'}
            </div>
            <div style="font-size:0.75rem;color:#C9082A;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">Puntos Predichos</div>
            <div class="prediction-number">{predicted_pts:.1f}</div>
            <div style="font-size:0.8rem;color:#8FA3BF;margin-top:10px;">
                Intervalo de confianza (68%):
                <span style="color:#FDB927;font-weight:700;">{ci_low:.1f} — {ci_high:.1f} pts</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f"""<div style="background:rgba(253,185,39,0.1);border:1px solid rgba(253,185,39,0.3);border-radius:12px;padding:14px;text-align:center;">
                <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.8rem;font-weight:900;color:#FDB927;">{player_avg:.1f}</div>
                <div style="font-size:0.68rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:1px;">Media (Ú10)</div>
            </div>""", unsafe_allow_html=True,
        )
    with col_b:
        diff = predicted_pts - player_avg
        diff_color = "#4ade80" if diff >= 0 else "#f87171"
        st.markdown(
            f"""<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:14px;text-align:center;">
                <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.8rem;font-weight:900;color:{diff_color};">{diff:+.1f}</div>
                <div style="font-size:0.68rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:1px;">vs Media</div>
            </div>""", unsafe_allow_html=True,
        )
    with col_c:
        opp_label = teams_df[teams_df["full_name"] == selected_opp]["abbreviation"].iloc[0] if selected_opp != "Media de la Liga" else "LIG"
        st.markdown(
            f"""<div style="background:rgba(23,64,139,0.15);border:1px solid rgba(23,64,139,0.4);border-radius:12px;padding:14px;text-align:center;">
                <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.8rem;font-weight:900;color:#60a5fa;">{defense_rating:.0f}</div>
                <div style="font-size:0.68rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:1px;">Def. Rival ({opp_label})</div>
            </div>""", unsafe_allow_html=True,
        )

with col_shap:
    st.markdown(
        '<div class="section-header" style="margin-top:0;">Factores de Predicción</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.78rem;color:#6B82A8;margin-bottom:10px;">Contribución de variables al resultado del modelo (SHAP)</div>',
        unsafe_allow_html=True,
    )
    colors = ["#C9082A" if v > 0 else "#17408B" for v in shap_df["contribution"]]
    fig_shap = go.Figure(go.Bar(
        x=shap_df["contribution"],
        y=shap_df["feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.05)", width=1)),
        text=[f"{v:+.2f}" for v in shap_df["contribution"]],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=10),
    ))
    fig_shap.add_vline(x=0, line_color="#FDB927", line_width=1.5, line_dash="dot")
    fig_shap.update_layout(
        **plotly_layout(
            title="Contribuciones SHAP",
            height=340,
            xaxis=dict(title="Contribución (pts)", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
            yaxis=dict(autorange="reversed", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#FFFFFF", size=10), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
            margin=dict(l=10, r=70, t=50, b=20),
            showlegend=False,
        )
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Historial Anotador</div>', unsafe_allow_html=True)

games_plot = games.sort_values("game_date").copy()
games_plot["game_date"] = pd.to_datetime(games_plot["game_date"])
n = len(games_plot)
x_vals = list(range(n))
x_labels = games_plot["game_date"].dt.strftime("%d/%m").tolist()
x_pred = n

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=x_vals,
    y=games_plot["pts"],
    mode="lines+markers",
    name="Puntos Anotados",
    line=dict(color="#17408B", width=2.5),
    marker=dict(
        color=["#4ade80" if v == "W" else "#f87171" for v in games_plot["wl"]],
        size=9,
        line=dict(color="#FFFFFF", width=1.5),
    ),
    hovertemplate="<b>%{text}</b><br>Puntos: %{y}<extra></extra>",
    text=[f"{d} ({'V' if wl=='W' else 'D'})" for d, wl in zip(x_labels, games_plot["wl"])],
))
fig_line.add_trace(go.Scatter(
    x=[x_vals[-1], x_pred],
    y=[float(games_plot["pts"].iloc[-1]), predicted_pts],
    mode="lines",
    line=dict(color="#FDB927", width=2, dash="dash"),
    showlegend=False,
    hoverinfo="skip",
))
fig_line.add_trace(go.Scatter(
    x=[x_pred],
    y=[predicted_pts],
    mode="markers",
    name=f"Prediccion: {predicted_pts:.1f} pts",
    marker=dict(color="#FDB927", size=14, symbol="star", line=dict(color="#FFFFFF", width=2)),
    error_y=dict(type="constant", value=ci_half, color="rgba(253,185,39,0.4)", thickness=2, width=8),
))
avg_line = float(games_plot["pts"].mean())
fig_line.add_hline(
    y=avg_line,
    line_dash="dot",
    line_color="rgba(143,163,191,0.5)",
    annotation_text=f"Media {avg_line:.1f}",
    annotation_font_color="#8FA3BF",
    annotation_position="top right",
)
tick_step = max(1, n // 8)
fig_line.update_layout(
    **plotly_layout(
        title=f"{selected_player} — Puntos por Partido y Prediccion del Proximo",
        height=340,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(0, n, tick_step)) + [x_pred],
            ticktext=[x_labels[i] for i in range(0, n, tick_step)] + ["PROXIMO"],
            gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False,
        ),
        yaxis=dict(title="Puntos", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(22,32,64,0.85)", bordercolor="rgba(31,48,96,0.8)", borderwidth=1, font=dict(color="#FFFFFF", size=11)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown('<div class="section-header">Registro Últimos 10 Partidos</div>', unsafe_allow_html=True)

log = games.head(10).copy()
log["game_date"] = pd.to_datetime(log["game_date"]).dt.strftime("%d %b")
log["min"] = log["min"].apply(lambda m: f"{int(m)}:{int((m % 1) * 60):02d}")
log["fg_pct"] = (log["fg_pct"] * 100).round(1).astype(str) + "%"
log["ft_pct"] = (log["ft_pct"] * 100).round(1).astype(str) + "%"
log["plus_minus"] = log["plus_minus"].apply(lambda x: f"+{int(x)}" if x > 0 else str(int(x)))
log["wl"] = log["wl"].apply(
    lambda v: f'<span class="{"win" if v=="W" else "loss"}-badge">{"V" if v=="W" else "D"}</span>'
)

display_log = pd.DataFrame({
    "Fecha": log["game_date"].values,
    "V/D": log["wl"].values,
    "PTS": log["pts"].astype(int).values,
    "MIN": log["min"].values,
    "TC%": log["fg_pct"].values,
    "TL%": log["ft_pct"].values,
    "REB": log["reb"].astype(int).values,
    "AST": log["ast"].astype(int).values,
    "+/-": log["plus_minus"].values,
})

st.markdown(
    display_log.to_html(escape=False, index=False).replace(
        '<table', '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;"'
    ).replace(
        '<th>', '<th style="background:#162040;color:#8FA3BF;text-transform:uppercase;font-size:0.7rem;letter-spacing:1px;padding:10px 12px;border-bottom:1px solid rgba(31,48,96,0.8);text-align:center;">'
    ).replace(
        '<td>', '<td style="padding:8px 12px;border-bottom:1px solid rgba(31,48,96,0.4);color:#FFFFFF;text-align:center;">'
    ),
    unsafe_allow_html=True,
)
