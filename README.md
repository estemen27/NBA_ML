# Predictor NBA con Machine Learning

Un pipeline de machine learning que predice los resultados de partidos NBA y el rendimiento anotador individual de jugadores para la temporada regular 2025–26. El proyecto cubre el ciclo de vida completo de ciencia de datos: ingesta, exploración, ingeniería de características, modelado, evaluación y un dashboard interactivo listo para producción.



## Descripción del Proyecto

Se abordan dos problemas predictivos:

1. **Predicción de Victoria del Equipo** — Clasificación binaria que estima la probabilidad de que un equipo gane su próximo partido basándose en métricas de rendimiento reciente.
2. **Predicción de Puntos del Jugador** — Regresión que pronostica cuántos puntos anotará un jugador en su próximo partido basándose en sus registros recientes de partidos.

Ambos modelos se sirven a través de un dashboard multipágina en Streamlit que consulta una base de datos PostgreSQL en tiempo real.



## Resultados

| Tarea | Modelo | Métrica | Valor |
|---|---|---|---|
| Predicción de Victoria | Regresión Logística Calibrada | Exactitud | 68,35% |
| Predicción de Victoria | Regresión Logística Calibrada | AUC-ROC | 75,36% |
| Predicción de Puntos | Regresión Ridge | RMSE | 6,55 pts |
| Predicción de Puntos | Regresión Ridge | R² | 0,3901 |
| Predicción de Puntos | Regresión Ridge | MAE | 4,99 pts |



## Dataset

Datos obtenidos de la API oficial de Estadísticas NBA mediante la librería `nba_api`, cubriendo la temporada regular 2025–26 completa.

| Tabla | Registros |
|---|---|
| `dim_teams` | 30 equipos |
| `dim_players` | 582 jugadores activos |
| `fact_team_game_logs` | 2.460 registros de partido por equipo |
| `fact_player_game_logs` | 26.651 registros de partido por jugador |
| `fact_team_season_stats` | 30 agregados de temporada |
| `fact_player_season_stats` | 582 agregados de temporada |



## Estructura del Proyecto

```
nba-ml-predictor/
├── app/                          # Dashboard Streamlit
│   ├── main.py                   # Punto de entrada y navegación
│   ├── .streamlit/config.toml    # Configuración del tema
│   ├── pages/
│   │   ├── 01_overview.py        # Métricas del modelo y resumen del proyecto
│   │   ├── 02_team_predictor.py  # Probabilidad de victoria en tiempo real
│   │   ├── 03_player_predictor.py # Pronóstico de puntos en tiempo real
│   │   ├── 04_model_performance.py # Métricas de evaluación y SHAP
│   │   └── 05_data_explorer.py   # Exploración interactiva de datos
│   └── utils/
│       ├── db_connection.py      # Helpers de conexión a PostgreSQL
│       ├── model_loader.py       # Carga del modelo e inferencia
│       ├── feature_engineering.py # Cálculo de características en tiempo real
│       └── styles.py             # CSS y tema de Plotly
├── config/
│   ├── db_config.yml             # Configuración de conexión a la base de datos
│   └── nba_config.yml            # Configuración de temporada y API
├── data/
│   ├── processed/                # Datasets con ingeniería de características
│   │   ├── team_classification_train.csv
│   │   ├── team_classification_test.csv
│   │   ├── player_regression_train.csv
│   │   └── player_regression_test.csv
│   └── raw/                      # No versionado — se obtiene desde la API
├── models/                       # Modelos entrenados serializados
│   ├── best_team_classifier.joblib
│   ├── best_player_regressor.joblib
│   ├── calibrated_classifier.joblib
│   ├── scaler_team.joblib
│   └── scaler_player.joblib
├── notebooks/                    # Notebooks de análisis CRISP-DM
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── reports/                      # Resultados exportados para el dashboard
│   ├── predictions_classification.csv
│   ├── predictions_regression.csv
│   ├── model_comparison.csv
│   ├── shap_values_classifier.csv
│   ├── shap_values_regressor.csv
│   ├── team_stats_summary.csv
│   └── player_stats_summary.csv
├── scripts/
│   └── generate_reports.py       # Genera todos los CSVs de reportes
├── src/
│   └── data_ingestion.py         # Pipeline de datos de la API NBA
├── docker-compose.yml            # Contenedor PostgreSQL
└── requirements.txt
```

---

## Configuración y Ejecución

### 1. Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd nba-ml-predictor
```

### 2. Crear un Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Iniciar la Base de Datos

El proyecto usa PostgreSQL mediante Docker. Inicia el contenedor con:

```bash
docker-compose up -d
```

La base de datos estará disponible en `localhost:5432` con las credenciales del archivo `.env`.

### 5. Cargar los Datos

Obtén los datos de la temporada actual desde la API de Estadísticas NBA:

```bash
python src/data_ingestion.py
```

Esto llena las seis tablas de la base de datos. El script respeta los límites de velocidad de la API NBA con un retraso de 1 segundo entre solicitudes.

### 6. Ejecutar los Notebooks

Ejecuta los notebooks en orden para reproducir el pipeline completo:

```
01_business_understanding.ipynb  — Definición del problema y objetivos
02_data_understanding.ipynb      — EDA y calidad de datos
03_data_preparation.ipynb        — Ingeniería de características
04_modeling.ipynb                — Entrenamiento y selección de modelos
05_evaluation.ipynb              — Evaluación en conjunto de prueba y conclusiones
```

Si omites los notebooks y deseas trabajar directamente con los modelos preentrenados, ya están guardados en `models/`.

### 7. Generar los CSVs de Reportes

```bash
python scripts/generate_reports.py
```

Este script genera siete archivos CSV en `reports/` que consume el dashboard. Carga los conjuntos de prueba, realiza predicciones, calcula los valores SHAP para ambos modelos y consulta la base de datos para obtener estadísticas agregadas.

### 8. Lanzar el Dashboard

```bash
cd app
streamlit run main.py
```

La aplicación se abre en `http://localhost:8501`.



## Páginas del Dashboard

### Vista General

Página de inicio que muestra las métricas clave del modelo (exactitud, AUC-ROC, R², MAE) como tarjetas, estadísticas del dataset (equipos, jugadores, registros de partidos), gráficos comparativos de modelos para ambas tareas y un resumen de la metodología.

### Predictor de Victoria del Equipo

Selecciona cualquiera de los 30 equipos NBA y la ubicación del partido (local o visitante). La aplicación consulta los últimos 10 partidos del equipo desde la base de datos, calcula las 22 características en tiempo real y ejecuta el clasificador calibrado para producir una probabilidad de victoria. Los resultados se muestran como un gráfico de gauge junto con la tabla del historial reciente de partidos y un gráfico de barras SHAP que explica qué factores influyeron más en la predicción.

### Predictor de Puntos del Jugador

Selecciona cualquier jugador activo con al menos 10 partidos jugados. Opcionalmente selecciona el equipo rival para obtener su rating defensivo. La aplicación consulta los últimos 20 partidos del jugador, construye las 14 características de regresión y predice su producción anotadora en el próximo partido. La predicción se muestra con un intervalo de confianza del 68%, un gráfico de líneas de su historial anotador con la predicción marcada como punto futuro y las contribuciones de características SHAP.

### Rendimiento del Modelo

Evaluación completa en el conjunto de prueba holdout (9 de marzo – 10 de abril de 2026). Para clasificación: exactitud, precisión, recall, F1, AUC-ROC, curva ROC interactiva y matriz de confusión. Para regresión: RMSE, MAE, R², gráfico de dispersión de valores reales vs predichos e histograma de distribución de residuos. Ambas secciones incluyen gráficos de importancia de características SHAP y tablas comparativas de modelos.

### Explorador de Datos

Tres pestañas para explorar la base de datos en vivo:
- **Clasificación de Equipos**: Tabla de posiciones ordenable con los 30 equipos y gráfico de barras de net rating.
- **Clasificación de Jugadores**: Tabla de líderes de jugadores filtrable con mínimo de partidos ajustable y ordenamiento por cualquier estadística.
- **Comparación de Equipos**: Gráfico de radar cara a cara comparando dos equipos seleccionados en seis dimensiones, con un gráfico de tendencia anotadora durante toda la temporada.



## Ingeniería de Características

Todas las características usan un desplazamiento temporal (`shift(1)`) antes de los cálculos de ventana deslizante para prevenir la fuga de datos. El conjunto de prueba contiene únicamente partidos disputados después del 9 de marzo de 2026.

### Características del Equipo (22)

| Característica | Descripción |
|---|---|
| `pts_last5 / pts_last10` | Promedio móvil de puntos anotados |
| `pts_against_last5 / pts_against_last10` | Promedio móvil de puntos encajados |
| `fg_pct_last5 / fg_pct_last10` | Promedio móvil de porcentaje de tiros de campo |
| `plus_minus_last5 / plus_minus_last10` | Promedio móvil de diferencial de puntos |
| `reb_last5 / reb_last10` | Promedio móvil de rebotes |
| `ast_last5 / ast_last10` | Promedio móvil de asistencias |
| `stl_last5 / stl_last10` | Promedio móvil de robos |
| `winrate_last5 / winrate_last10` | Porcentaje de victorias móvil |
| `streak` | Racha actual de victorias/derrotas (+N o −N) |
| `rest_days` | Días desde el último partido |
| `home_winrate / away_winrate` | Porcentaje acumulado de victorias como local y visitante |
| `opp_pts_last5` | Puntos anotados por el rival en sus últimos 5 partidos |
| `is_home` | Indicador binario local/visitante |

### Características del Jugador (14)

| Característica | Descripción |
|---|---|
| `pts_last5` | Promedio móvil de puntos (últimos 5 partidos) |
| `min_last5` | Promedio móvil de minutos jugados |
| `fg_pct_last5 / fg_pct_last10` | Promedio móvil de porcentaje de tiros de campo |
| `ft_pct_last5 / ft_pct_last10` | Promedio móvil de porcentaje de tiros libres |
| `reb_last5 / reb_last10` | Promedio móvil de rebotes |
| `ast_last5` | Promedio móvil de asistencias |
| `pts_trend` | Forma reciente: promedio últimos 3 partidos − promedio últimos 10 partidos |
| `shot_volume_last5` | Indicador de carga ofensiva: pts / (fg_pct × 2) |
| `defense_rating` | Promedio de puntos encajados por partido del rival en la temporada |
| `position` | Posición inferida (0 = Base, 1 = Alero, 2 = Pívot) |
| `is_home` | Indicador binario local/visitante |



## Modelos

### Clasificación — Regresión Logística Calibrada

Seleccionada por sus probabilidades bien calibradas, esenciales para que el gráfico de gauge refleje probabilidades reales de victoria en lugar de puntuaciones brutas. Entrenada con validación cruzada `TimeSeriesSplit` (5 particiones) y calibrada mediante regresión isotónica.

Alternativas evaluadas: Regresión Logística, Random Forest, XGBoost.

### Regresión — Regresión Ridge

Seleccionada por su capacidad para manejar características de ventana deslizante correlacionadas (las ventanas L5 y L10 comparten datos solapados). Su regularización L2 previene el sobreajuste ante un conjunto de características con alta multicolinealidad. Entrenada como un pipeline con `StandardScaler`.

Alternativas evaluadas: Regresión Lineal, Random Forest, XGBoost.


## Conclusiones del Proyecto

1. **La predicción de victorias con un 68,35% de exactitud supera significativamente la línea base del 50%**, lo que demuestra que las métricas de rendimiento reciente (puntos anotados, encajados, porcentaje de victorias) contienen señal predictiva relevante para los resultados de partidos NBA.

2. **El AUC-ROC de 0,7536 indica que el clasificador ordena con fiabilidad a los equipos por probabilidad real de victoria**, lo cual es más importante que la exactitud bruta en aplicaciones de toma de decisiones donde la magnitud importa.

3. **El RMSE de 6,55 puntos del regresor Ridge reduce el error en un 21,9% respecto a una línea base ingenua de media (RMSE 8,38)**, confirmando que el rendimiento de los jugadores muestra inercia que los promedios móviles pueden capturar.

4. **La ventana móvil de 5 partidos fue el conjunto de características más informativo** para ambos modelos. La forma reciente (pts_last5, winrate_last5) superó consistentemente a los promedios de 10 partidos en importancia SHAP, reflejando la naturaleza momentum del calendario NBA.

5. **La prevención de fuga de datos temporales mediante shift(1) fue crítica**: experimentos preliminares sin el desplazamiento produjeron cifras de exactitud infladas que se derrumbaron en el conjunto de holdout, confirmando la necesidad de una disciplina temporal estricta en la predicción de series temporales.

6. **El paso de calibración mejoró la confiabilidad respecto a la regresión logística sin calibrar**: tras la calibración, las probabilidades predichas entre 60–70% correspondieron empíricamente a tasas de victoria reales dentro de ese rango, haciendo que la salida sea adecuada para interpretación probabilística en el gauge del dashboard.



## Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Fuente de datos | API de Estadísticas NBA (`nba_api`) |
| Almacén de datos | PostgreSQL 15 (Docker) |
| Procesamiento de datos | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Explicabilidad | SHAP |
| Dashboard | Streamlit 1.56 |
| Visualización | Plotly |
| Entorno | Python 3.10+ |



## Variables de Entorno

El proyecto lee las credenciales de la base de datos desde el archivo `.env` en la raíz del proyecto:

```
POSTGRES_USER=nba_user
POSTGRES_PASSWORD=nba_pass
POSTGRES_DB=nba_database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

