import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.styles import inject_css, nba_card, sidebar_brand, plotly_layout
from utils.db_connection import query_df

inject_css()
sidebar_brand()

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #0F1E40 0%, #17408B 50%, #0B1426 100%);
        border-radius: 20px;
        padding: 40px 48px 36px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(23,64,139,0.4);
    ">
        <div style="
            font-family:'Barlow Condensed',Arial,sans-serif;
            font-size:0.75rem;
            color:#C9082A;
            text-transform:uppercase;
            letter-spacing:4px;
            font-weight:700;
            margin-bottom:10px;
        ">Analítica de Machine Learning · Temporada 2025–26</div>
        <div style="
            font-family:'Barlow Condensed',Arial,sans-serif;
            font-size:3.8rem;
            font-weight:900;
            color:#FFFFFF;
            line-height:1.05;
            margin-bottom:14px;
        ">NBA ML<br><span style='color:#FDB927;'>PREDICTOR</span></div>
        <div style="font-size:1rem;color:#8FA3BF;max-width:600px;line-height:1.6;">
            Plataforma de analítica predictiva para la NBA — probabilidades de victoria por equipo
            y puntuación individual de jugadores, impulsada por modelos de machine learning entrenados
            con datos en tiempo real de la temporada regular 2025–26.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-header">Rendimiento del Modelo</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(nba_card("68,4%", "Precisión de Predicción", "Regresión Logística Calibrada"), unsafe_allow_html=True)
with col2:
    st.markdown(nba_card("75,4%", "Puntuación AUC-ROC", "Clasificación de Equipos"), unsafe_allow_html=True)
with col3:
    st.markdown(nba_card("39,0%", "Puntuación R²", "Regresión Ridge"), unsafe_allow_html=True)
with col4:
    st.markdown(nba_card("±4,99", "Error Abs. Medio", "Predicción de Puntos"), unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Resumen del Dataset</div>', unsafe_allow_html=True)

try:
    stats = query_df("""
        SELECT
            (SELECT COUNT(DISTINCT team_id) FROM dim_teams) AS teams,
            (SELECT COUNT(*) FROM dim_players WHERE is_active = true) AS players,
            (SELECT COUNT(*) FROM fact_team_game_logs) AS team_games,
            (SELECT COUNT(*) FROM fact_player_game_logs) AS player_games
    """)
    n_teams = int(stats["teams"].iloc[0])
    n_players = int(stats["players"].iloc[0])
    n_team_games = int(stats["team_games"].iloc[0])
    n_player_games = int(stats["player_games"].iloc[0])
except Exception:
    n_teams, n_players, n_team_games, n_player_games = 30, 582, 2460, 26651

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(nba_card(str(n_teams), "Equipos NBA", "Las 30 franquicias"), unsafe_allow_html=True)
with col2:
    st.markdown(nba_card(f"{n_players:,}", "Jugadores Activos", "Plantilla temporada 2025–26"), unsafe_allow_html=True)
with col3:
    st.markdown(nba_card(f"{n_team_games:,}", "Partidos de Equipo", "Temporada regular completa"), unsafe_allow_html=True)
with col4:
    st.markdown(nba_card(f"{n_player_games:,}", "Registros de Jugadores", "Actuaciones individuales"), unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-header">Clasificación — Victorias de Equipo</div>', unsafe_allow_html=True)

    clf_data = {
        "Modelo": ["Línea Base", "Regresión Logística", "Random Forest", "XGBoost", "RL Calibrada (*)"],
        "Precisión (%)": [50.0, 69.5, 69.5, 68.4, 68.4],
        "AUC-ROC (%)": [50.0, 76.4, 74.2, 73.7, 75.4],
    }
    df_clf = pd.DataFrame(clf_data)

    fig_clf = go.Figure()
    fig_clf.add_trace(go.Bar(
        name="Precisión (%)",
        x=df_clf["Modelo"],
        y=df_clf["Precisión (%)"],
        marker=dict(
            color=["#2d3a5c", "#2d3a5c", "#2d3a5c", "#2d3a5c", "#C9082A"],
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=df_clf["Precisión (%)"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11),
    ))
    fig_clf.add_trace(go.Bar(
        name="AUC-ROC (%)",
        x=df_clf["Modelo"],
        y=df_clf["AUC-ROC (%)"],
        marker=dict(
            color=["#1a2a5c", "#1a2a5c", "#1a2a5c", "#1a2a5c", "#17408B"],
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=df_clf["AUC-ROC (%)"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11),
    ))
    fig_clf.update_layout(
        **plotly_layout(
            title="Predicción de Victorias — Todos los Modelos",
            barmode="group",
            height=340,
            yaxis=dict(range=[40, 85], title="Puntuación (%)", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=11), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
            xaxis=dict(tickangle=-15, gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
            legend=dict(orientation="h", y=1.12, bgcolor="rgba(22,32,64,0.85)", bordercolor="rgba(31,48,96,0.8)", borderwidth=1, font=dict(color="#FFFFFF", size=12)),
            margin=dict(l=20, r=20, t=60, b=20),
        )
    )
    st.plotly_chart(fig_clf, use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">Regresión — Puntos por Jugador</div>', unsafe_allow_html=True)

    reg_data = {
        "Modelo": ["Línea Base", "Reg. Lineal", "Random Forest", "XGBoost", "Ridge (*)"],
        "RMSE (pts)": [8.38, 6.55, 6.56, 6.56, 6.55],
        "MAE (pts)": [6.61, 4.99, 5.03, 5.03, 4.99],
    }
    df_reg = pd.DataFrame(reg_data)

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Bar(
        name="RMSE (pts)",
        x=df_reg["Modelo"],
        y=df_reg["RMSE (pts)"],
        marker=dict(
            color=["#2d3a5c", "#2d3a5c", "#2d3a5c", "#2d3a5c", "#C9082A"],
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=df_reg["RMSE (pts)"].apply(lambda x: f"{x:.2f}"),
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11),
    ))
    fig_reg.add_trace(go.Bar(
        name="MAE (pts)",
        x=df_reg["Modelo"],
        y=df_reg["MAE (pts)"],
        marker=dict(
            color=["#1a2a5c", "#1a2a5c", "#1a2a5c", "#1a2a5c", "#17408B"],
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=df_reg["MAE (pts)"].apply(lambda x: f"{x:.2f}"),
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11),
    ))
    fig_reg.update_layout(
        **plotly_layout(
            title="Predicción de Puntos — Todos los Modelos",
            barmode="group",
            height=340,
            yaxis=dict(range=[0, 10.5], title="Error (pts)", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=11), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
            xaxis=dict(tickangle=-15, gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
            legend=dict(orientation="h", y=1.12, bgcolor="rgba(22,32,64,0.85)", bordercolor="rgba(31,48,96,0.8)", borderwidth=1, font=dict(color="#FFFFFF", size=12)),
            margin=dict(l=20, r=20, t=60, b=20),
        )
    )
    st.plotly_chart(fig_reg, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Metodología</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="metric-card" style="text-align:left;padding:24px;">
            <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.15rem;font-weight:900;color:#FDB927;margin-bottom:8px;text-transform:uppercase;">Pipeline de Datos</div>
            <div style="font-size:0.85rem;color:#8FA3BF;line-height:1.6;">
                Ingesta en tiempo real desde la NBA Stats API hacia un almacén de datos PostgreSQL.
                Seis tablas normalizadas con registros de partidos de equipo y jugador
                para toda la temporada regular 2025–26.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="metric-card" style="text-align:left;padding:24px;">
            <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.15rem;font-weight:900;color:#FDB927;margin-bottom:8px;text-transform:uppercase;">Ingeniería de Variables</div>
            <div style="font-size:0.85rem;color:#8FA3BF;line-height:1.6;">
                22 variables de equipo y 14 de jugador — medias móviles de 5 y 10 partidos,
                rachas de victorias, días de descanso, rendimiento local/visitante y
                métricas avanzadas. Desplazamiento temporal para evitar fuga de datos.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div class="metric-card" style="text-align:left;padding:24px;">
            <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.15rem;font-weight:900;color:#FDB927;margin-bottom:8px;text-transform:uppercase;">Selección del Modelo</div>
            <div style="font-size:0.85rem;color:#8FA3BF;line-height:1.6;">
                Metodología CRISP-DM con validación cruzada temporal (TimeSeriesSplit).
                Regresión Logística Calibrada para predicción de victorias y
                Ridge para puntos — seleccionados por calibración de probabilidades y generalización.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
