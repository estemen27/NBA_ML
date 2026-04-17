import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix
from utils.styles import inject_css, sidebar_brand, plotly_layout

BASE_DIR = Path(__file__).parent.parent.parent
REPORTS = BASE_DIR / "reports"

inject_css()
sidebar_brand()

st.markdown(
    """
    <div style="margin-bottom:28px;">
        <div style="font-size:0.72rem;color:#C9082A;text-transform:uppercase;letter-spacing:4px;font-weight:700;">Evaluacion · Conjunto de Prueba Temporal</div>
        <div style="font-family:'Barlow Condensed',Arial,sans-serif;font-size:3rem;font-weight:900;color:#FFFFFF;line-height:1.1;margin-top:6px;">
            RENDIMIENTO DEL MODELO
        </div>
        <div style="font-size:0.9rem;color:#8FA3BF;margin-top:6px;">
            Resultados completos sobre el conjunto de prueba temporal (9 de marzo – 10 de abril de 2026).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model_type = st.radio(
    "Seleccionar Modelo",
    ["Prediccion de Victorias (Clasificacion)", "Puntos por Jugador (Regresion)"],
    horizontal=True,
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

try:
    clf_preds = pd.read_csv(REPORTS / "predictions_classification.csv")
    reg_preds = pd.read_csv(REPORTS / "predictions_regression.csv")
    shap_clf_df = pd.read_csv(REPORTS / "shap_values_classifier.csv")
    shap_reg_df = pd.read_csv(REPORTS / "shap_values_regressor.csv")
except Exception as e:
    st.error(f"No se pudieron cargar los archivos de resultados: {e}")
    st.stop()

if "Clasificacion" in model_type:
    st.markdown('<div class="section-header">Metricas de Clasificacion</div>', unsafe_allow_html=True)

    y_true = clf_preds["actual"].values
    y_prob = clf_preds["probability"].values
    y_pred = clf_preds["prediction"].values

    accuracy = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)

    col1, col2, col3, col4, col5 = st.columns(5)
    metric_list = [
        (col1, accuracy, "Precision"),
        (col2, precision, "Exactitud"),
        (col3, recall, "Exhaustividad"),
        (col4, f1, "F1-Score"),
        (col5, auc_score, "AUC-ROC"),
    ]
    for col, val, label in metric_list:
        with col:
            color = "#4ade80" if val >= 0.7 else ("#FDB927" if val >= 0.6 else "#f87171")
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{val*100:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown('<div class="section-header" style="margin-top:0;">Curva ROC</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"RL Calibrada (AUC = {auc_score:.3f})",
            line=dict(color="#C9082A", width=3),
            fill="tozeroy",
            fillcolor="rgba(201,8,42,0.08)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Clasificador Aleatorio",
            line=dict(color="#8FA3BF", width=1.5, dash="dash"),
        ))
        fig_roc.update_layout(
            **plotly_layout(
                title="Curva Caracteristica del Receptor (ROC)",
                height=360,
                xaxis=dict(title="Tasa de Falsos Positivos", range=[0, 1], gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
                yaxis=dict(title="Tasa de Verdaderos Positivos", range=[0, 1], gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
                legend=dict(x=0.45, y=0.1, bgcolor="rgba(22,32,64,0.85)", bordercolor="rgba(31,48,96,0.8)", borderwidth=1, font=dict(color="#FFFFFF", size=11)),
                margin=dict(l=20, r=20, t=50, b=20),
            )
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown('<div class="section-header" style="margin-top:0;">Matriz de Confusion</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_true, y_pred)

        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Pred. Derrota", "Pred. Victoria"],
            y=["Real Derrota", "Real Victoria"],
            colorscale=[[0, "#0B1426"], [0.5, "#17408B"], [1, "#C9082A"]],
            showscale=False,
            text=[[f"{cm[i][j]:,}\n{cm[i][j]/len(y_true)*100:.1f}%" for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
            textfont=dict(size=16, color="white"),
            hovertemplate="<b>%{y} -> %{x}</b><br>Total: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            **plotly_layout(
                title="Predicciones vs Resultados Reales",
                height=360,
                xaxis=dict(side="bottom", tickfont=dict(color="#FFFFFF", size=12), gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
                yaxis=dict(autorange="reversed", tickfont=dict(color="#FFFFFF", size=12), gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
                margin=dict(l=20, r=20, t=50, b=20),
            )
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="section-header">Comparativa de Modelos — Clasificacion</div>', unsafe_allow_html=True)
    comp_data = {
        "Modelo": ["Linea Base", "Reg. Logistica", "Random Forest", "XGBoost", "RL Calibrada"],
        "Precision (%)": [50.0, 69.5, 69.5, 68.4, 68.4],
        "F1-Score (%)": [66.7, 69.9, 72.1, 68.8, 68.5],
        "AUC-ROC (%)": [50.0, 76.4, 74.2, 73.7, 75.4],
        "T. Entrenamiento": ["0.00s", "0.19s", "4.85s", "2.11s", "0.05s"],
        "Seleccionado": ["", "", "", "", "SI"],
    }
    comp_df = pd.DataFrame(comp_data)

    def color_row(row):
        if row["Seleccionado"] == "SI":
            return ["background-color: rgba(201,8,42,0.15); color: #FFFFFF"] * len(row)
        return ["color: #FFFFFF"] * len(row)

    st.dataframe(
        comp_df.style.apply(color_row, axis=1).format({
            "Precision (%)": "{:.1f}%",
            "F1-Score (%)": "{:.1f}%",
            "AUC-ROC (%)": "{:.1f}%",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('<div class="section-header">Importancia de Variables SHAP — Clasificador</div>', unsafe_allow_html=True)
    top_shap = shap_clf_df.head(12).copy()
    top_shap["feature"] = top_shap["feature"].str.replace("_", " ").str.title()

    fig_shap_clf = go.Figure(go.Bar(
        x=top_shap["mean_shap"],
        y=top_shap["feature"],
        orientation="h",
        marker=dict(
            color=top_shap["mean_shap"],
            colorscale=[[0, "#17408B"], [0.5, "#8B1740"], [1, "#C9082A"]],
            line=dict(color="rgba(255,255,255,0.05)", width=1),
        ),
        text=top_shap["mean_shap"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=10),
    ))
    fig_shap_clf.update_layout(
        **plotly_layout(
            title="Valor |SHAP| Medio por Variable",
            height=380,
            xaxis=dict(title="Valor SHAP Absoluto Medio", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
            yaxis=dict(autorange="reversed", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#FFFFFF", size=10), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
            showlegend=False,
            margin=dict(l=10, r=80, t=50, b=20),
        )
    )
    st.plotly_chart(fig_shap_clf, use_container_width=True)

else:
    st.markdown('<div class="section-header">Metricas de Regresion</div>', unsafe_allow_html=True)

    y_true = reg_preds["actual_pts"].values
    y_pred_vals = reg_preds["predicted_pts"].values
    errors = reg_preds["abs_error"].values

    rmse = np.sqrt(np.mean((y_true - y_pred_vals) ** 2))
    mae = np.mean(errors)
    ss_res = np.sum((y_true - y_pred_vals) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    col1, col2, col3, col4 = st.columns(4)
    metric_list_r = [
        (col1, f"{rmse:.2f} pts", "RMSE", "#FDB927"),
        (col2, f"{mae:.2f} pts", "MAE", "#FDB927"),
        (col3, f"{r2:.1%}", "R²", "#4ade80" if r2 >= 0.35 else "#FDB927"),
        (col4, f"{np.mean(errors <= 5):.1%}", "Error menor 5 pts", "#4ade80" if np.mean(errors <= 5) >= 0.5 else "#FDB927"),
    ]
    for col, display, label, color in metric_list_r:
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{display}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    col_scatter, col_residual = st.columns(2)

    with col_scatter:
        st.markdown('<div class="section-header" style="margin-top:0;">Real vs Predicho</div>', unsafe_allow_html=True)
        sample = reg_preds.sample(min(800, len(reg_preds)), random_state=42)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=sample["actual_pts"],
            y=sample["predicted_pts"],
            mode="markers",
            marker=dict(
                color=sample["abs_error"],
                colorscale=[[0, "#17408B"], [0.5, "#FDB927"], [1, "#C9082A"]],
                size=5,
                opacity=0.65,
                colorbar=dict(
                    title=dict(text="Error (pts)", font=dict(color="#8FA3BF")),
                    tickfont=dict(color="#8FA3BF"),
                    thickness=12,
                ),
                line=dict(color="rgba(255,255,255,0.1)", width=0.5),
            ),
            hovertemplate="<b>Real: %{x:.0f} pts</b><br>Predicho: %{y:.1f} pts<extra></extra>",
            showlegend=False,
        ))
        max_val = max(sample["actual_pts"].max(), sample["predicted_pts"].max()) + 5
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            name="Prediccion Perfecta",
            line=dict(color="#FDB927", width=1.5, dash="dash"),
        ))
        fig_scatter.update_layout(
            **plotly_layout(
                title="Puntos Reales vs Puntos Predichos",
                height=360,
                xaxis=dict(title="Puntos Reales", range=[0, max_val], gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
                yaxis=dict(title="Puntos Predichos", range=[0, max_val], gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
                legend=dict(x=0.05, y=0.95, bgcolor="rgba(22,32,64,0.85)", bordercolor="rgba(31,48,96,0.8)", borderwidth=1, font=dict(color="#FFFFFF", size=11)),
                margin=dict(l=20, r=20, t=50, b=20),
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_residual:
        st.markdown('<div class="section-header" style="margin-top:0;">Distribucion de Residuos</div>', unsafe_allow_html=True)
        residuals = y_pred_vals - y_true

        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=residuals,
            nbinsx=50,
            name="Residuos",
            marker=dict(
                color="#17408B",
                line=dict(color="rgba(255,255,255,0.05)", width=0.5),
            ),
            opacity=0.85,
        ))
        fig_res.add_vline(x=0, line_color="#FDB927", line_width=2, line_dash="dot",
                          annotation_text="Error Cero", annotation_font_color="#FDB927")
        fig_res.add_vline(x=np.mean(residuals), line_color="#C9082A", line_width=1.5,
                          annotation_text=f"Media {np.mean(residuals):.2f}", annotation_font_color="#C9082A")
        fig_res.update_layout(
            **plotly_layout(
                title="Distribucion del Error de Prediccion",
                height=360,
                xaxis=dict(title="Residuo (Predicho menos Real)", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
                yaxis=dict(title="Frecuencia", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=20),
            )
        )
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown('<div class="section-header">Comparativa de Modelos — Regresion</div>', unsafe_allow_html=True)
    comp_data_reg = {
        "Modelo": ["Linea Base (Media)", "Reg. Lineal", "Random Forest", "XGBoost", "Ridge"],
        "RMSE (pts)": [8.38, 6.55, 6.56, 6.56, 6.55],
        "MAE (pts)": [6.61, 4.99, 5.03, 5.03, 4.99],
        "R²": [-0.0001, 0.3901, 0.3865, 0.3877, 0.3901],
        "T. Entrenamiento": ["0.00s", "0.01s", "12.85s", "3.48s", "0.12s"],
        "Seleccionado": ["", "", "", "", "SI"],
    }
    comp_df_reg = pd.DataFrame(comp_data_reg)

    def color_row_reg(row):
        if row["Seleccionado"] == "SI":
            return ["background-color: rgba(201,8,42,0.15); color: #FFFFFF"] * len(row)
        return ["color: #FFFFFF"] * len(row)

    st.dataframe(
        comp_df_reg.style.apply(color_row_reg, axis=1).format({
            "RMSE (pts)": "{:.2f}",
            "MAE (pts)": "{:.2f}",
            "R²": "{:.4f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('<div class="section-header">Importancia de Variables SHAP — Regresor</div>', unsafe_allow_html=True)
    top_shap_r = shap_reg_df.head(10).copy()
    top_shap_r["feature"] = top_shap_r["feature"].str.replace("_", " ").str.title()

    fig_shap_r = go.Figure(go.Bar(
        x=top_shap_r["mean_shap"],
        y=top_shap_r["feature"],
        orientation="h",
        marker=dict(
            color=top_shap_r["mean_shap"],
            colorscale=[[0, "#17408B"], [0.5, "#8B1740"], [1, "#C9082A"]],
            line=dict(color="rgba(255,255,255,0.05)", width=1),
        ),
        text=top_shap_r["mean_shap"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=10),
    ))
    fig_shap_r.update_layout(
        **plotly_layout(
            title="Valor |SHAP| Medio por Variable",
            height=340,
            xaxis=dict(title="Valor SHAP Absoluto Medio", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#8FA3BF", size=10), title_font=dict(color="#8FA3BF"), showgrid=True, zeroline=False),
            yaxis=dict(autorange="reversed", gridcolor="rgba(31,48,96,0.5)", linecolor="#1f3060", tickfont=dict(color="#FFFFFF", size=10), title_font=dict(color="#8FA3BF"), showgrid=False, zeroline=False),
            showlegend=False,
            margin=dict(l=10, r=80, t=50, b=20),
        )
    )
    st.plotly_chart(fig_shap_r, use_container_width=True)
