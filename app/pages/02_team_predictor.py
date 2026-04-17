import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.styles import inject_css, sidebar_brand, plotly_layout
from utils.db_connection import query_df
from utils.model_loader import predict_team, shap_clf
from utils.feature_engineering import build_team_features

inject_css()
sidebar_brand()

st.markdown(
    """
    <div style="margin-bottom:28px;">
        <div style="font-size:0.72rem;color:#C9082A;text-transform:uppercase;letter-spacing:4px;font-weight:700;">Motor de Predicción en Tiempo Real</div>
        <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:3rem;font-weight:900;color:#FFFFFF;line-height:1.1;margin-top:6px;">
            PREDICTOR DE VICTORIAS
        </div>
        <div style="font-size:0.9rem;color:#8FA3BF;margin-top:6px;">
            Selecciona un equipo — el modelo consulta sus últimos 10 partidos desde la base de datos y calcula la probabilidad de victoria en tiempo real.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    teams_df = query_df("SELECT team_id, full_name, abbreviation, city FROM dim_teams ORDER BY full_name")
except Exception as e:
    st.error(f"Error de conexión a la base de datos: {e}")
    st.stop()

col_sel, col_opt = st.columns([2, 1])
with col_sel:
    team_names = teams_df["full_name"].tolist()
    selected_team = st.selectbox("Seleccionar Equipo", team_names, index=0)
with col_opt:
    game_location = st.radio("Próximo Partido", ["Local", "Visitante"], horizontal=True)

is_home = 1 if game_location == "Local" else 0
team_row = teams_df[teams_df["full_name"] == selected_team].iloc[0]
team_id = int(team_row["team_id"])

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

try:
    games = query_df(
        """
        SELECT a.game_date, a.wl, a.pts, a.reb, a.ast, a.stl,
               a.fg_pct, a.plus_minus, a.matchup,
               b.pts AS pts_against
        FROM fact_team_game_logs a
        JOIN fact_team_game_logs b ON a.game_id = b.game_id AND b.team_id != a.team_id
        WHERE a.team_id = %s
        ORDER BY a.game_date DESC
        LIMIT 10
        """,
        params=(team_id,),
    )
except Exception as e:
    st.error(f"No se pudieron obtener los datos del equipo: {e}")
    st.stop()

if len(games) < 5:
    st.warning(f"{selected_team} tiene menos de 5 partidos en la base de datos. No se puede calcular la predicción.")
    st.stop()

features = build_team_features(games, is_home)
if not features:
    st.warning("Historial de partidos insuficiente para generar una predicción.")
    st.stop()

prob, pred = predict_team(features)
shap_df = shap_clf(features)

col_gauge, col_stats = st.columns([1, 1])

with col_gauge:
    color_bar = "#4ade80" if prob >= 0.6 else ("#FDB927" if prob >= 0.4 else "#f87171")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={
            "suffix": "%",
            "font": {"size": 52, "color": "#FFFFFF", "family": "Barlow Condensed, Arial Narrow, sans-serif"},
        },
        title={
            "text": f"{'VICTORIA' if pred == 1 else 'DERROTA'}<br><span style='font-size:0.8em;color:#8FA3BF'>{selected_team}</span>",
            "font": {"size": 16, "color": "#FFFFFF", "family": "Barlow Condensed, Arial Narrow, sans-serif"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#8FA3BF",
                "tickfont": {"color": "#8FA3BF", "size": 11},
            },
            "bar": {"color": color_bar, "thickness": 0.22},
            "bgcolor": "#0B1426",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(248,113,113,0.12)"},
                {"range": [35, 65], "color": "rgba(253,185,39,0.08)"},
                {"range": [65, 100], "color": "rgba(74,222,128,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#FDB927", "width": 3},
                "thickness": 0.72,
                "value": 50,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#162040",
        font={"color": "#FFFFFF", "family": "Inter, sans-serif"},
        height=320,
        margin=dict(l=40, r=40, t=60, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    outcome_color = "#4ade80" if pred == 1 else "#f87171"
    outcome_text = "VICTORIA" if pred == 1 else "DERROTA"
    conf_level = "ALTA" if prob > 0.65 or prob < 0.35 else ("MEDIA" if prob > 0.55 or prob < 0.45 else "BAJA")
    conf_color = "#4ade80" if conf_level == "ALTA" else ("#FDB927" if conf_level == "MEDIA" else "#f87171")

    st.markdown(
        f"""
        <div style="display:flex;gap:10px;justify-content:center;margin-top:4px;">
            <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:12px 20px;text-align:center;flex:1;">
                <div style="font-size:0.68rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:1.5px;">Predicción</div>
                <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.6rem;font-weight:900;color:{outcome_color};">{outcome_text}</div>
            </div>
            <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:12px 20px;text-align:center;flex:1;">
                <div style="font-size:0.68rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:1.5px;">Confianza</div>
                <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.6rem;font-weight:900;color:{conf_color};">{conf_level}</div>
            </div>
            <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:12px 20px;text-align:center;flex:1;">
                <div style="font-size:0.68rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:1.5px;">Sede</div>
                <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.6rem;font-weight:900;color:#60a5fa;">{'LOCAL' if is_home else 'VISIT.'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_stats:
    st.markdown('<div class="section-header" style="margin-top:0;">Factores Clave</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.78rem;color:#6B82A8;margin-bottom:10px;">Contribución de variables a la probabilidad de victoria (SHAP)</div>',
        unsafe_allow_html=True,
    )

    colors = ["#C9082A" if v > 0 else "#17408B" for v in shap_df["contribution"]]
    fig_shap = go.Figure(go.Bar(
        x=shap_df["contribution"],
        y=shap_df["feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.05)", width=1)),
        text=[f"{v:+.3f}" for v in shap_df["contribution"]],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=10),
    ))
    fig_shap.add_vline(x=0, line_color="#FDB927", line_width=1.5, line_dash="dot")
    fig_shap.update_layout(
        **plotly_layout(
            title="Contribuciones SHAP",
            height=320,
            xaxis=dict(title="Contribución a la Probabilidad", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
            yaxis=dict(autorange="reversed", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#FFFFFF", size=10), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
            margin=dict(l=10, r=60, t=50, b=20),
            showlegend=False,
        )
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_hist, col_trend = st.columns([3, 2])

with col_hist:
    st.markdown('<div class="section-header" style="margin-top:0;">Historial Reciente</div>', unsafe_allow_html=True)

    display_games = games.copy()
    display_games["game_date"] = pd.to_datetime(display_games["game_date"]).dt.strftime("%d %b")
    display_games["pts"] = display_games["pts"].astype(int)
    display_games["pts_against"] = display_games["pts_against"].astype(int)
    display_games["fg_pct"] = (display_games["fg_pct"] * 100).round(1).astype(str) + "%"
    display_games["plus_minus"] = display_games["plus_minus"].apply(lambda x: f"+{int(x)}" if x > 0 else str(int(x)))

    wl_html = display_games.apply(
        lambda r: f'<span class="{"win" if r["wl"]=="W" else "loss"}-badge">{"V" if r["wl"]=="W" else "D"}</span>', axis=1
    )

    table_df = pd.DataFrame({
        "Fecha": display_games["game_date"].values,
        "Resultado": wl_html.values,
        "PTS": display_games["pts"].values,
        "PTS RIV": display_games["pts_against"].values,
        "TC%": display_games["fg_pct"].values,
        "+/-": display_games["plus_minus"].values,
        "REB": display_games["reb"].astype(int).values,
        "AST": display_games["ast"].astype(int).values,
    })

    st.markdown(
        table_df.to_html(escape=False, index=False).replace(
            '<table', '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;"'
        ).replace(
            '<th>', '<th style="background:#162040;color:#8FA3BF;text-transform:uppercase;font-size:0.7rem;letter-spacing:1px;padding:10px 12px;border-bottom:1px solid rgba(31,48,96,0.8);text-align:center;">'
        ).replace(
            '<td>', '<td style="padding:8px 12px;border-bottom:1px solid rgba(31,48,96,0.4);color:#FFFFFF;text-align:center;">'
        ),
        unsafe_allow_html=True,
    )

with col_trend:
    st.markdown('<div class="section-header" style="margin-top:0;">Tendencia Anotadora</div>', unsafe_allow_html=True)
    games_plot = games.sort_values("game_date").tail(10).copy()
    dates = pd.to_datetime(games_plot["game_date"]).dt.strftime("%d/%m")

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=dates, y=games_plot["pts_against"],
        mode="lines",
        name="Pts Encajados",
        line=dict(color="#C9082A", width=2, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(201,8,42,0.08)",
    ))
    fig_trend.add_trace(go.Scatter(
        x=dates, y=games_plot["pts"],
        mode="lines+markers",
        name="Pts Anotados",
        line=dict(color="#17408B", width=2.5),
        marker=dict(
            color=["#4ade80" if v == "W" else "#f87171" for v in games_plot["wl"]],
            size=10,
            line=dict(color="#FFFFFF", width=1.5),
        ),
        fill="tozeroy",
        fillcolor="rgba(23,64,139,0.1)",
    ))
    fig_trend.update_layout(
        **plotly_layout(
            title="Últimos 10 Partidos — Puntos Anotados vs Encajados",
            height=300,
            yaxis=dict(title="Puntos", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
            xaxis=dict(gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
            legend=dict(orientation="h", y=1.12, bgcolor="rgba(22,32,64,0.85)", bordercolor="rgba(31,48,96,0.8)", borderwidth=1, font=dict(color="#FFFFFF", size=11)),
            margin=dict(l=20, r=20, t=55, b=20),
        )
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    wins_last10 = sum(1 for v in games["wl"] if v == "W")
    st.markdown(
        f"""
        <div style="display:flex;gap:8px;margin-top:8px;">
            <div style="flex:1;background:rgba(74,222,128,0.1);border:1px solid rgba(74,222,128,0.3);border-radius:10px;padding:14px;text-align:center;">
                <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:2rem;font-weight:900;color:#4ade80;">{wins_last10}</div>
                <div style="font-size:0.7rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:1px;">Victorias (Ú10)</div>
            </div>
            <div style="flex:1;background:rgba(248,113,113,0.1);border:1px solid rgba(248,113,113,0.3);border-radius:10px;padding:14px;text-align:center;">
                <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:2rem;font-weight:900;color:#f87171;">{10 - wins_last10}</div>
                <div style="font-size:0.7rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:1px;">Derrotas (Ú10)</div>
            </div>
            <div style="flex:1;background:rgba(23,64,139,0.15);border:1px solid rgba(23,64,139,0.4);border-radius:10px;padding:14px;text-align:center;">
                <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:2rem;font-weight:900;color:#FDB927;">{features.get('streak', 0):+d}</div>
                <div style="font-size:0.7rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:1px;">Racha</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
