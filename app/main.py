import streamlit as st

st.set_page_config(
    page_title="NBA ML Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = st.navigation(
    [
        st.Page("pages/01_overview.py", title="Resumen", default=True),
        st.Page("pages/02_team_predictor.py", title="Predictor de Victorias"),
        st.Page("pages/03_player_predictor.py", title="Predictor de Puntos"),
        st.Page("pages/04_model_performance.py", title="Rendimiento del Modelo"),
        st.Page("pages/05_data_explorer.py", title="Explorador de Datos"),
    ]
)
pages.run()
