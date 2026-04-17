import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.styles import inject_css, sidebar_brand, plotly_layout
from utils.db_connection import query_df

BASE_DIR = Path(__file__).parent.parent.parent
REPORTS = BASE_DIR / "reports"

inject_css()
sidebar_brand()

st.markdown(
    """
    <div style="margin-bottom:28px;">
        <div style="font-size:0.72rem;color:#C9082A;text-transform:uppercase;letter-spacing:4px;font-weight:700;">Base de Datos en Tiempo Real · Temporada Regular 2025-26</div>
        <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:3rem;font-weight:900;color:#FFFFFF;line-height:1.1;margin-top:6px;">
            EXPLORADOR DE DATOS
        </div>
        <div style="font-size:0.9rem;color:#8FA3BF;margin-top:6px;">
            Explora estadisticas directamente desde la base de datos PostgreSQL — filtra, clasifica y compara equipos y jugadores.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_teams, tab_players, tab_compare = st.tabs(["Clasificacion por Equipo", "Clasificacion por Jugador", "Comparativa de Equipos"])

with tab_teams:
    st.markdown('<div class="section-header" style="margin-top:8px;">Clasificacion de Equipos — 2025-26</div>', unsafe_allow_html=True)

    try:
        team_stats = pd.read_csv(REPORTS / "team_stats_summary.csv")
    except Exception:
        try:
            team_stats = query_df(
                """
                SELECT t.full_name AS team, t.abbreviation,
                       COUNT(g.game_id) AS games_played,
                       ROUND(AVG(g.pts)::numeric,1) AS avg_pts,
                       ROUND(AVG(g.reb)::numeric,1) AS avg_reb,
                       ROUND(AVG(g.ast)::numeric,1) AS avg_ast,
                       ROUND(AVG(g.fg_pct)::numeric,3) AS avg_fg_pct,
                       ROUND(AVG(g.plus_minus)::numeric,1) AS avg_plus_minus,
                       SUM(CASE WHEN g.wl='W' THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN g.wl='L' THEN 1 ELSE 0 END) AS losses,
                       ROUND(AVG(CASE WHEN g.wl='W' THEN 1.0 ELSE 0.0 END)::numeric,3) AS win_pct,
                       ROUND(AVG(opp.pts)::numeric,1) AS avg_pts_allowed
                FROM fact_team_game_logs g
                JOIN dim_teams t ON g.team_id = t.team_id
                JOIN fact_team_game_logs opp ON g.game_id = opp.game_id AND opp.team_id != g.team_id
                GROUP BY t.team_id, t.full_name, t.abbreviation
                ORDER BY win_pct DESC
                """
            )
        except Exception as e:
            st.error(f"No se pudieron cargar los datos de equipos: {e}")
            st.stop()

    col_filter, col_metric = st.columns([2, 1])
    with col_filter:
        search = st.text_input("Buscar equipo", placeholder="Ej: Lakers, Boston...")
    with col_metric:
        sort_options = {
            "% Victorias": "win_pct",
            "Puntos por Partido": "avg_pts",
            "Net Rating": "avg_plus_minus",
            "Tiro de Campo (%)": "avg_fg_pct",
            "Rebotes": "avg_reb",
            "Asistencias": "avg_ast",
        }
        sort_label = st.selectbox("Ordenar por", list(sort_options.keys()))
        sort_by = sort_options[sort_label]

    if search:
        team_stats = team_stats[team_stats["team"].str.contains(search, case=False, na=False)]

    team_stats = team_stats.sort_values(sort_by, ascending=False).reset_index(drop=True)
    team_stats.index = range(1, len(team_stats) + 1)

    display_ts = team_stats.copy()
    display_ts["Balance"] = display_ts.apply(lambda r: f"{int(r['wins'])}-{int(r['losses'])}", axis=1)
    display_ts["% Vic"] = (display_ts["win_pct"] * 100).round(1).astype(str) + "%"
    display_ts["TC%"] = (display_ts["avg_fg_pct"] * 100).round(1).astype(str) + "%"
    display_ts["+/-"] = display_ts["avg_plus_minus"].apply(lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}")
    display_ts["PF"] = display_ts["avg_pts"]
    display_ts["PC"] = display_ts["avg_pts_allowed"]

    st.dataframe(
        display_ts[["team", "abbreviation", "Balance", "% Vic", "PF", "PC", "+/-", "avg_reb", "avg_ast", "TC%"]]
        .rename(columns={"team": "Equipo", "abbreviation": "Abr.", "avg_reb": "REB", "avg_ast": "AST"}),
        hide_index=False,
        use_container_width=True,
        height=600,
    )

    st.markdown('<div class="section-header">Top y Bottom 5 — Net Rating</div>', unsafe_allow_html=True)
    top5 = team_stats.nlargest(5, "avg_plus_minus")
    bot5 = team_stats.nsmallest(5, "avg_plus_minus")
    combined = pd.concat([top5, bot5]).sort_values("avg_plus_minus", ascending=True)

    colors = ["#C9082A" if v < 0 else "#17408B" for v in combined["avg_plus_minus"]]
    fig_net = go.Figure(go.Bar(
        x=combined["avg_plus_minus"],
        y=combined["abbreviation"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.06)", width=1)),
        text=[f"{v:+.1f}" for v in combined["avg_plus_minus"]],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11),
    ))
    fig_net.add_vline(x=0, line_color="#FDB927", line_width=1.5)
    fig_net.update_layout(
        **plotly_layout(
            title="Net Rating Medio — Diferencial de Puntos por Partido",
            height=340,
            xaxis=dict(title="Net Rating (+/-)", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
            yaxis=dict(gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#FFFFFF", size=11), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
            showlegend=False,
            margin=dict(l=20, r=80, t=50, b=20),
        )
    )
    st.plotly_chart(fig_net, use_container_width=True)


with tab_players:
    st.markdown('<div class="section-header" style="margin-top:8px;">Clasificacion de Jugadores — 2025-26</div>', unsafe_allow_html=True)

    try:
        player_stats = pd.read_csv(REPORTS / "player_stats_summary.csv")
    except Exception:
        try:
            player_stats = query_df(
                """
                SELECT p.full_name AS player,
                       COUNT(g.game_id) AS games_played,
                       ROUND(AVG(g.pts)::numeric,1) AS avg_pts,
                       ROUND(AVG(g.min)::numeric,1) AS avg_min,
                       ROUND(AVG(g.reb)::numeric,1) AS avg_reb,
                       ROUND(AVG(g.ast)::numeric,1) AS avg_ast,
                       ROUND(AVG(g.fg_pct)::numeric,3) AS avg_fg_pct,
                       ROUND(AVG(g.ft_pct)::numeric,3) AS avg_ft_pct,
                       ROUND(MAX(g.pts)::numeric,0) AS max_pts,
                       ROUND(AVG(g.stl)::numeric,1) AS avg_stl,
                       ROUND(AVG(g.blk)::numeric,1) AS avg_blk
                FROM fact_player_game_logs g
                JOIN dim_players p ON g.player_id = p.player_id
                WHERE g.min >= 10
                GROUP BY p.player_id, p.full_name
                HAVING COUNT(g.game_id) >= 10
                ORDER BY avg_pts DESC
                """
            )
        except Exception as e:
            st.error(f"No se pudieron cargar los datos de jugadores: {e}")
            st.stop()

    col_s, col_m, col_gp = st.columns([2, 1, 1])
    with col_s:
        search_p = st.text_input("Buscar jugador", placeholder="Ej: LeBron, Curry...")
    with col_m:
        sort_options_p = {
            "Puntos por Partido": "avg_pts",
            "Rebotes por Partido": "avg_reb",
            "Asistencias por Partido": "avg_ast",
            "Minutos por Partido": "avg_min",
            "Tiro de Campo (%)": "avg_fg_pct",
            "Maximo Temporada": "max_pts",
        }
        sort_label_p = st.selectbox("Ordenar por", list(sort_options_p.keys()))
        sort_p = sort_options_p[sort_label_p]
    with col_gp:
        min_games = st.slider("Min. partidos disputados", 10, 60, 20)

    filtered_p = player_stats[player_stats["games_played"] >= min_games].copy()
    if search_p:
        filtered_p = filtered_p[filtered_p["player"].str.contains(search_p, case=False, na=False)]
    filtered_p = filtered_p.sort_values(sort_p, ascending=False).reset_index(drop=True)
    filtered_p.index = range(1, len(filtered_p) + 1)

    display_p = filtered_p.copy()
    display_p["TC%"] = (display_p["avg_fg_pct"] * 100).round(1).astype(str) + "%"
    display_p["TL%"] = (display_p["avg_ft_pct"] * 100).round(1).astype(str) + "%"

    st.dataframe(
        display_p[["player", "games_played", "avg_pts", "avg_min", "avg_reb", "avg_ast", "TC%", "TL%", "avg_stl", "avg_blk", "max_pts"]]
        .rename(columns={
            "player": "Jugador", "games_played": "PJ", "avg_pts": "PPP",
            "avg_min": "MPP", "avg_reb": "RPP", "avg_ast": "APP",
            "avg_stl": "RPB", "avg_blk": "TAP", "max_pts": "Max. Temporada",
        }),
        hide_index=False,
        use_container_width=True,
        height=550,
    )

    st.markdown('<div class="section-header">Lideres Anotadores</div>', unsafe_allow_html=True)
    top20 = filtered_p.head(20).sort_values("avg_pts")

    fig_scorers = go.Figure(go.Bar(
        x=top20["avg_pts"],
        y=top20["player"],
        orientation="h",
        marker=dict(
            color=top20["avg_pts"],
            colorscale=[[0, "#17408B"], [0.5, "#8B2566"], [1, "#C9082A"]],
            line=dict(color="rgba(255,255,255,0.05)", width=0.5),
        ),
        text=top20["avg_pts"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=10),
    ))
    fig_scorers.update_layout(
        **plotly_layout(
            title="Top 20 Anotadores — Puntos por Partido",
            height=520,
            xaxis=dict(title="PPP", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
            yaxis=dict(gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#FFFFFF", size=10), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
            showlegend=False,
            margin=dict(l=20, r=60, t=50, b=20),
        )
    )
    st.plotly_chart(fig_scorers, use_container_width=True)


with tab_compare:
    st.markdown('<div class="section-header" style="margin-top:8px;">Comparativa Directa de Equipos</div>', unsafe_allow_html=True)

    try:
        team_stats_c = pd.read_csv(REPORTS / "team_stats_summary.csv")
        all_teams = sorted(team_stats_c["team"].tolist())
    except Exception:
        all_teams = []

    if not all_teams:
        st.warning("Los datos de equipos no estan disponibles.")
    else:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            team1 = st.selectbox("Equipo 1", all_teams, index=0, key="cmp1")
        with col_t2:
            default2 = 1 if len(all_teams) > 1 else 0
            team2 = st.selectbox("Equipo 2", all_teams, index=default2, key="cmp2")

        t1 = team_stats_c[team_stats_c["team"] == team1].iloc[0]
        t2 = team_stats_c[team_stats_c["team"] == team2].iloc[0]

        categories = ["PPP", "RPP", "APP", "TC%", "Net Rating", "% Vic."]

        def norm(v, mn, mx):
            if mx == mn:
                return 0.5
            return (v - mn) / (mx - mn)

        metrics_raw = {
            "PPP": ("avg_pts", 95, 130),
            "RPP": ("avg_reb", 38, 55),
            "APP": ("avg_ast", 20, 35),
            "TC%": ("avg_fg_pct", 0.42, 0.52),
            "Net Rating": ("avg_plus_minus", -15, 15),
            "% Vic.": ("win_pct", 0.2, 0.8),
        }

        vals1, vals2, raw1, raw2 = [], [], [], []
        for cat, (col, mn, mx) in metrics_raw.items():
            v1 = float(t1[col])
            v2 = float(t2[col])
            vals1.append(norm(v1, mn, mx))
            vals2.append(norm(v2, mn, mx))
            if col in ["avg_fg_pct", "win_pct"]:
                fmt1 = f"{v1*100:.1f}%"
                fmt2 = f"{v2*100:.1f}%"
            elif col == "avg_plus_minus":
                fmt1 = f"{v1:+.1f}"
                fmt2 = f"{v2:+.1f}"
            else:
                fmt1 = f"{v1:.1f}"
                fmt2 = f"{v2:.1f}"
            raw1.append(fmt1)
            raw2.append(fmt2)

        vals1_closed = vals1 + [vals1[0]]
        vals2_closed = vals2 + [vals2[0]]
        cats_closed = categories + [categories[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals1_closed,
            theta=cats_closed,
            fill="toself",
            name=t1["abbreviation"],
            line=dict(color="#17408B", width=2.5),
            fillcolor="rgba(23,64,139,0.25)",
            marker=dict(size=7, color="#17408B"),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals2_closed,
            theta=cats_closed,
            fill="toself",
            name=t2["abbreviation"],
            line=dict(color="#C9082A", width=2.5),
            fillcolor="rgba(201,8,42,0.2)",
            marker=dict(size=7, color="#C9082A"),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#0B1426",
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#8FA3BF", size=9), gridcolor="rgba(31,48,96,0.5)", linecolor="rgba(31,48,96,0.5)"),
                angularaxis=dict(tickfont=dict(color="#FFFFFF", size=12), linecolor="rgba(31,48,96,0.5)", gridcolor="rgba(31,48,96,0.5)"),
            ),
            paper_bgcolor="#162040",
            font=dict(color="#FFFFFF", family="Inter, sans-serif"),
            title=dict(text=f"{t1['abbreviation']} vs {t2['abbreviation']} — Perfil Estadistico", font=dict(size=17, color="#FFFFFF", family="Barlow Condensed, Arial Narrow, sans-serif")),
            legend=dict(bgcolor="rgba(22,32,64,0.85)", bordercolor="rgba(31,48,96,0.8)", borderwidth=1, font=dict(color="#FFFFFF", size=12)),
            height=480,
            margin=dict(l=60, r=60, t=70, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            rec1 = f"{int(t1['wins'])}-{int(t1['losses'])}"
            rows1 = "".join([
                f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(31,48,96,0.4);"><span style="color:#8FA3BF;font-size:0.85rem;">{cat}</span><span style="color:#FFFFFF;font-weight:700;">{r}</span></div>'
                for cat, r in zip(categories, raw1)
            ])
            st.markdown(
                f"""
                <div style="background:rgba(23,64,139,0.15);border:1px solid rgba(23,64,139,0.4);border-radius:14px;padding:20px;">
                    <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.6rem;font-weight:900;color:#17408B;margin-bottom:12px;">{team1}</div>
                    {rows1}
                    <div style="display:flex;justify-content:space-between;padding:6px 0;"><span style="color:#8FA3BF;font-size:0.85rem;">Balance</span><span style="color:#FDB927;font-weight:700;">{rec1}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_right:
            rec2 = f"{int(t2['wins'])}-{int(t2['losses'])}"
            rows2 = "".join([
                f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(31,48,96,0.4);"><span style="color:#8FA3BF;font-size:0.85rem;">{cat}</span><span style="color:#FFFFFF;font-weight:700;">{r}</span></div>'
                for cat, r in zip(categories, raw2)
            ])
            st.markdown(
                f"""
                <div style="background:rgba(201,8,42,0.1);border:1px solid rgba(201,8,42,0.3);border-radius:14px;padding:20px;">
                    <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:1.6rem;font-weight:900;color:#C9082A;margin-bottom:12px;">{team2}</div>
                    {rows2}
                    <div style="display:flex;justify-content:space-between;padding:6px 0;"><span style="color:#8FA3BF;font-size:0.85rem;">Balance</span><span style="color:#FDB927;font-weight:700;">{rec2}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Tendencia Anotadora en la Temporada</div>', unsafe_allow_html=True)
        try:
            t1_id = query_df("SELECT team_id FROM dim_teams WHERE full_name = %s", params=(team1,))["team_id"].iloc[0]
            t2_id = query_df("SELECT team_id FROM dim_teams WHERE full_name = %s", params=(team2,))["team_id"].iloc[0]

            t1_games = query_df(
                "SELECT game_date, pts FROM fact_team_game_logs WHERE team_id = %s ORDER BY game_date",
                params=(int(t1_id),),
            )
            t2_games = query_df(
                "SELECT game_date, pts FROM fact_team_game_logs WHERE team_id = %s ORDER BY game_date",
                params=(int(t2_id),),
            )

            for df_g in [t1_games, t2_games]:
                df_g["game_date"] = pd.to_datetime(df_g["game_date"])
                df_g["rolling_avg"] = df_g["pts"].rolling(10, min_periods=3).mean()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=t1_games["game_date"], y=t1_games["rolling_avg"],
                mode="lines", name=t1["abbreviation"],
                line=dict(color="#17408B", width=2.5),
                fill="tozeroy", fillcolor="rgba(23,64,139,0.08)",
            ))
            fig_trend.add_trace(go.Scatter(
                x=t2_games["game_date"], y=t2_games["rolling_avg"],
                mode="lines", name=t2["abbreviation"],
                line=dict(color="#C9082A", width=2.5),
                fill="tozeroy", fillcolor="rgba(201,8,42,0.08)",
            ))
            fig_trend.update_layout(
                **plotly_layout(
                    title="Media Movil de 10 Partidos — Puntos Anotados",
                    height=320,
                    xaxis=dict(title="Fecha", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
                    yaxis=dict(title="Puntos (Media Movil 10P)", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
                    legend=dict(orientation="h", y=1.12, bgcolor="rgba(22,32,64,0.85)", bordercolor="rgba(31,48,96,0.8)", borderwidth=1, font=dict(color="#FFFFFF", size=12)),
                    margin=dict(l=20, r=20, t=55, b=20),
                )
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        except Exception:
            pass
