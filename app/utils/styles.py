import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

_NBA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;800;900&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: #0B1426;
}

h1, h2, h3, .nba-title {
    font-family: 'Barlow Condensed', 'Arial Narrow', Arial, sans-serif !important;
    font-weight: 900 !important;
    letter-spacing: 0.5px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1020 0%, #0D1830 100%);
    border-right: 1px solid rgba(23, 64, 139, 0.25);
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.stButton > button {
    background: linear-gradient(135deg, #17408B 0%, #1a50a8 100%);
    color: #FFFFFF;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.5px;
    padding: 0.55rem 1.8rem;
    transition: all 0.25s ease;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #C9082A 0%, #e00a30 100%);
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(201, 8, 42, 0.45);
    border-color: transparent;
}

div[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', Arial, sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    color: #FDB927 !important;
}

div[data-testid="stMetricLabel"] {
    color: #8FA3BF !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

div[data-testid="stMetricDelta"] svg {
    display: none;
}

.stSelectbox label, .stRadio label {
    color: #8FA3BF !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stSelectbox > div > div {
    background: #162040 !important;
    border: 1px solid rgba(23, 64, 139, 0.4) !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
}

.stDataFrame, .stTable {
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}

.metric-card {
    background: linear-gradient(135deg, #162040 0%, #1b2a55 100%);
    border: 1px solid rgba(23, 64, 139, 0.35);
    border-radius: 16px;
    padding: 28px 24px 24px;
    position: relative;
    overflow: hidden;
    text-align: center;
    margin-bottom: 4px;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #17408B, #C9082A);
}

.metric-value {
    font-family: 'Barlow Condensed', Arial, sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: #FDB927;
    line-height: 1;
    margin: 8px 0 6px;
}

.metric-label {
    font-size: 0.73rem;
    color: #8FA3BF;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-weight: 600;
}

.metric-sub {
    font-size: 0.82rem;
    color: #6B82A8;
    margin-top: 4px;
}

.section-header {
    font-family: 'Barlow Condensed', Arial, sans-serif;
    font-size: 1.8rem;
    font-weight: 900;
    color: #FFFFFF;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-left: 4px solid #C9082A;
    padding-left: 14px;
    margin: 32px 0 16px;
    line-height: 1.1;
}

.prediction-hero {
    background: linear-gradient(135deg, #0F1E40 0%, #162040 100%);
    border: 1px solid rgba(23, 64, 139, 0.4);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.prediction-hero::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #C9082A, #17408B);
}

.prediction-number {
    font-family: 'Barlow Condensed', Arial, sans-serif;
    font-size: 5rem;
    font-weight: 900;
    color: #FDB927;
    line-height: 1;
}

.prediction-label {
    font-size: 0.8rem;
    color: #8FA3BF;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 6px;
}

.win-badge {
    display: inline-block;
    background: rgba(74, 222, 128, 0.15);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.35);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.loss-badge {
    display: inline-block;
    background: rgba(248, 113, 113, 0.15);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.35);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.info-pill {
    display: inline-block;
    background: rgba(23, 64, 139, 0.2);
    color: #60a5fa;
    border: 1px solid rgba(23, 64, 139, 0.4);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(23,64,139,0.4), transparent);
    margin: 24px 0;
}

.sidebar-brand {
    text-align: center;
    padding: 20px 0 10px;
}

.sidebar-brand-title {
    font-family: 'Barlow Condensed', Arial, sans-serif;
    font-size: 1.5rem;
    font-weight: 900;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #17408B, #C9082A);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-transform: uppercase;
}

.sidebar-brand-sub {
    font-size: 0.65rem;
    color: #6B82A8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 2px;
}

.stAlert {
    border-radius: 12px;
}

div[data-testid="column"] {
    padding: 0 6px;
}

.stPlotlyChart {
    border-radius: 16px;
    overflow: hidden;
}

.model-tag {
    display: inline-block;
    background: rgba(201, 8, 42, 0.15);
    color: #f87171;
    border: 1px solid rgba(201, 8, 42, 0.35);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
"""


def inject_css():
    st.markdown(_NBA_CSS, unsafe_allow_html=True)


def nba_card(value: str, label: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """


def sidebar_brand():
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">NBA Predictor</div>
            <div class="sidebar-brand-sub">Analítica ML · 2025–26</div>
        </div>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True,
    )


_LAYOUT = dict(
    paper_bgcolor="#162040",
    plot_bgcolor="#0B1426",
    font=dict(color="#FFFFFF", family="Inter, sans-serif", size=13),
    title_font=dict(size=17, color="#FFFFFF", family="Barlow Condensed, Arial Narrow, sans-serif"),
    margin=dict(l=20, r=20, t=50, b=20),
    colorway=["#17408B", "#C9082A", "#FDB927", "#4ade80", "#f87171", "#60a5fa", "#a78bfa"],
    xaxis=dict(
        gridcolor="rgba(31,48,96,0.5)",
        linecolor="#1f3060",
        tickfont=dict(color="#8FA3BF", size=11),
        title_font=dict(color="#8FA3BF"),
        showgrid=True,
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="rgba(31,48,96,0.5)",
        linecolor="#1f3060",
        tickfont=dict(color="#8FA3BF", size=11),
        title_font=dict(color="#8FA3BF"),
        showgrid=True,
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(22,32,64,0.85)",
        bordercolor="rgba(31,48,96,0.8)",
        borderwidth=1,
        font=dict(color="#FFFFFF", size=12),
    ),
)


def plotly_layout(**kwargs) -> dict:
    layout = dict(_LAYOUT)
    layout.update(kwargs)
    return layout
