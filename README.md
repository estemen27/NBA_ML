# NBA ML Predictor

A machine learning pipeline that predicts NBA game outcomes and individual player scoring for the 2025–26 regular season. The project covers the full data science lifecycle — ingestion, exploration, feature engineering, modeling, evaluation, and a production-ready interactive dashboard.

---

## Project Overview

Two predictive problems are addressed:

1. **Team Win Prediction** — Binary classification that estimates the probability a team wins their next game based on recent performance metrics.
2. **Player Points Prediction** — Regression that forecasts how many points a player will score in their next game based on their recent game logs.

Both models are served through a multi-page Streamlit dashboard that queries a live PostgreSQL database in real time.

---

## Results

| Task | Model | Metric | Value |
|---|---|---|---|
| Team Win Prediction | Calibrated Logistic Regression | Accuracy | 68.35% |
| Team Win Prediction | Calibrated Logistic Regression | AUC-ROC | 75.36% |
| Player Points Prediction | Ridge Regression | RMSE | 6.55 pts |
| Player Points Prediction | Ridge Regression | R² | 0.3901 |
| Player Points Prediction | Ridge Regression | MAE | 4.99 pts |

---

## Dataset

Data sourced from the official NBA Stats API via the `nba_api` library, covering the complete 2025–26 regular season.

| Table | Records |
|---|---|
| `dim_teams` | 30 teams |
| `dim_players` | 582 active players |
| `fact_team_game_logs` | 2,460 team-game records |
| `fact_player_game_logs` | 26,651 player-game records |
| `fact_team_season_stats` | 30 season aggregates |
| `fact_player_season_stats` | 582 season aggregates |

---

## Project Structure

```
nba-ml-predictor/
├── app/                          # Streamlit dashboard
│   ├── main.py                   # Navigation entry point
│   ├── .streamlit/config.toml    # Theme configuration
│   ├── pages/
│   │   ├── 01_overview.py        # Model metrics and project summary
│   │   ├── 02_team_predictor.py  # Real-time team win probability
│   │   ├── 03_player_predictor.py # Real-time player points forecast
│   │   ├── 04_model_performance.py # Evaluation metrics and SHAP
│   │   └── 05_data_explorer.py   # Interactive data exploration
│   └── utils/
│       ├── db_connection.py      # PostgreSQL connection helpers
│       ├── model_loader.py       # Model loading and inference
│       ├── feature_engineering.py # Real-time feature computation
│       └── styles.py             # CSS and Plotly theme
├── config/
│   ├── db_config.yml             # Database connection settings
│   └── nba_config.yml            # Season and API configuration
├── data/
│   ├── processed/                # Feature-engineered datasets
│   │   ├── team_classification_train.csv
│   │   ├── team_classification_test.csv
│   │   ├── player_regression_train.csv
│   │   └── player_regression_test.csv
│   └── raw/                      # Not tracked — fetched from API
├── models/                       # Serialized trained models
│   ├── best_team_classifier.joblib
│   ├── best_player_regressor.joblib
│   ├── calibrated_classifier.joblib
│   ├── scaler_team.joblib
│   └── scaler_player.joblib
├── notebooks/                    # CRISP-DM analysis notebooks
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── reports/                      # Exported results for the dashboard
│   ├── predictions_classification.csv
│   ├── predictions_regression.csv
│   ├── model_comparison.csv
│   ├── shap_values_classifier.csv
│   ├── shap_values_regressor.csv
│   ├── team_stats_summary.csv
│   └── player_stats_summary.csv
├── scripts/
│   └── generate_reports.py       # Generates all report CSVs
├── src/
│   └── data_ingestion.py         # NBA API data pipeline
├── docker-compose.yml            # PostgreSQL container
└── requirements.txt
```

---

## Setup and Execution

### 1. Clone the Repository

```bash
git clone <repository-url>
cd nba-ml-predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Database

The project uses PostgreSQL via Docker. Start the container with:

```bash
docker-compose up -d
```

The database will be available at `localhost:5432` with the credentials in `.env`.

### 5. Load the Data

Fetch current season data from the NBA Stats API:

```bash
python src/data_ingestion.py
```

This populates all six tables in the database. The script respects the NBA API rate limits with a 1-second delay between requests.

### 6. Run the Notebooks

Execute the notebooks in order to reproduce the full pipeline:

```
01_business_understanding.ipynb  — Problem framing and objectives
02_data_understanding.ipynb      — EDA and data quality
03_data_preparation.ipynb        — Feature engineering
04_modeling.ipynb                — Model training and selection
05_evaluation.ipynb              — Holdout test evaluation and conclusions
```

If you skip the notebooks and want to work directly with the pre-trained models, they are already saved in `models/`.

### 7. Generate the Report CSVs

```bash
python scripts/generate_reports.py
```

This script generates seven CSV files in `reports/` that the dashboard consumes. It loads the test sets, makes predictions, computes SHAP values for both models, and queries the database for aggregated statistics.

### 8. Launch the Dashboard

```bash
cd app
streamlit run main.py
```

The app opens at `http://localhost:8501`.

---

## Dashboard Pages

### Overview

Landing page showing the key model metrics (accuracy, AUC-ROC, R², MAE) as cards, dataset statistics (teams, players, game logs), model comparison charts for both tasks, and a summary of the methodology.

### Team Win Predictor

Select any of the 30 NBA teams and a game location (home or away). The app queries the team's last 10 games from the database, computes all 22 features in real time, and runs the calibrated classifier to produce a win probability. Results are shown as a gauge chart alongside the recent game history table and a SHAP bar chart explaining which factors most influenced the prediction.

### Player Points Predictor

Select any active player with at least 10 games played. Optionally select the opponent team to fetch their defensive rating. The app queries the player's last 20 games, builds the 14 regression features, and predicts their next scoring output. The prediction is shown with a 68% confidence interval, a line chart of their scoring history with the prediction marked as a future point, and SHAP feature contributions.

### Model Performance

Full evaluation on the holdout test set (March 9 – April 10, 2026). For classification: accuracy, precision, recall, F1, AUC-ROC, interactive ROC curve, and confusion matrix. For regression: RMSE, MAE, R², scatter plot of real vs predicted, and residual distribution histogram. Both sections include SHAP feature importance charts and model comparison tables.

### Data Explorer

Three tabs for exploring the live database:
- **Team Rankings**: Sortable standings table with all 30 teams and net rating bar chart.
- **Player Rankings**: Filtered player leaderboard with adjustable minimum games and sorting by any stat.
- **Team Comparison**: Head-to-head radar chart comparing two selected teams across six dimensions, with a season-long scoring trend chart.

---

## Feature Engineering

All features use a temporal shift (`shift(1)`) before rolling calculations to prevent data leakage. The test set contains only games played after March 9, 2026.

### Team Features (22)

| Feature | Description |
|---|---|
| `pts_last5 / pts_last10` | Rolling average points scored |
| `pts_against_last5 / pts_against_last10` | Rolling average points allowed |
| `fg_pct_last5 / fg_pct_last10` | Rolling average field goal percentage |
| `plus_minus_last5 / plus_minus_last10` | Rolling average point differential |
| `reb_last5 / reb_last10` | Rolling average rebounds |
| `ast_last5 / ast_last10` | Rolling average assists |
| `stl_last5 / stl_last10` | Rolling average steals |
| `winrate_last5 / winrate_last10` | Rolling win rate |
| `streak` | Current win/loss streak (+N or −N) |
| `rest_days` | Days since last game |
| `home_winrate / away_winrate` | Cumulative home and away win percentage |
| `opp_pts_last5` | Opponent points scored in last 5 games |
| `is_home` | Binary home/away indicator |

### Player Features (14)

| Feature | Description |
|---|---|
| `pts_last5` | Rolling average points (last 5 games) |
| `min_last5` | Rolling average minutes played |
| `fg_pct_last5 / fg_pct_last10` | Rolling average field goal percentage |
| `ft_pct_last5 / ft_pct_last10` | Rolling average free throw percentage |
| `reb_last5 / reb_last10` | Rolling average rebounds |
| `ast_last5` | Rolling average assists |
| `pts_trend` | Short-term form: avg last 3 games − avg last 10 games |
| `shot_volume_last5` | Offensive load proxy: pts / (fg_pct × 2) |
| `defense_rating` | Opponent's season-average points allowed per game |
| `position` | Inferred position (0 = Guard, 1 = Forward, 2 = Center) |
| `is_home` | Binary home/away indicator |

---

## Models

### Classification — Calibrated Logistic Regression

Selected for its well-calibrated probabilities, which are essential for the gauge chart to reflect actual win likelihoods rather than raw scores. Trained with `TimeSeriesSplit` cross-validation (5 folds) and calibrated using isotonic regression.

Alternatives evaluated: Logistic Regression, Random Forest, XGBoost.

### Regression — Ridge Regression

Selected for its ability to handle correlated rolling window features (L5 and L10 windows share overlapping data). Its L2 regularization prevents overfitting to a high-multicollinearity feature set. Trained as a pipeline with `StandardScaler`.

Alternatives evaluated: Linear Regression, Random Forest, XGBoost.

---

## Project Conclusions

1. **Win prediction at 68.35% accuracy significantly exceeds the 50% baseline**, demonstrating that recent performance metrics (points scored, allowed, win rate) carry meaningful predictive signal for NBA game outcomes.

2. **The AUC-ROC of 0.7536 indicates the classifier reliably ranks teams by true win probability**, which is more important than raw accuracy for decision-making applications where magnitude matters.

3. **The Ridge regressor's RMSE of 6.55 points reduces error by 21.9% relative to a naive mean baseline (RMSE 8.38)**, confirming that player performance exhibits inertia that rolling averages can capture.

4. **The 5-game rolling window was the most informative feature set** for both models. Short-term form (pts_last5, winrate_last5) consistently outweighed 10-game averages in SHAP importance, reflecting the momentum-driven nature of the NBA schedule.

5. **Temporal data leakage prevention via shift(1) was critical**: preliminary experiments without the shift produced inflated accuracy figures that collapsed on the holdout set, confirming the necessity of strict temporal discipline in time-series prediction.

6. **The calibration step improved trustworthiness over raw logistic regression**: after calibration, predicted probabilities between 60–70% corresponded empirically to actual win rates within that range, making the output suitable for probabilistic interpretation in the dashboard gauge.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Data source | NBA Stats API (`nba_api`) |
| Data warehouse | PostgreSQL 15 (Docker) |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit 1.56 |
| Visualization | Plotly |
| Environment | Python 3.10+ |

---

## Environment Variables

The project reads database credentials from the `.env` file in the project root:

```
POSTGRES_USER=nba_user
POSTGRES_PASSWORD=nba_pass
POSTGRES_DB=nba_database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

---

## License

MIT
