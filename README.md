# AirSense AI

**AI-Powered Air Quality Prediction & Environmental Analytics Platform**

AirSense AI is an end-to-end machine learning project that predicts the Air Quality Index (AQI) for Indian cities, classifies air quality categories, analyzes pollution trends, and provides health recommendations through an interactive Streamlit dashboard.

Developed as part of **M.Sc. Data Science** coursework at **Savitribai Phule Pune University**.

---

## Features

- **AQI Prediction** — Predict AQI and category from pollutant readings using trained ML models
- **City Explorer** — Explore historical AQI trends, seasonal patterns, and pollutant profiles
- **Analytics Dashboard** — Filter by city, year, and AQI category with interactive charts
- **Model Performance** — Compare regression and classification models with evaluation plots
- **Health Advisory** — Personalized recommendations based on predicted AQI levels

---

## Model Performance

| Task | Best Model | Metric | Score |
|------|------------|--------|-------|
| Regression | Random Forest | R² | 0.9023 |
| Regression | Random Forest | RMSE | 42.15 |
| Regression | Random Forest | MAE | 20.43 |
| Classification | Random Forest | Accuracy | 82.21% |
| Classification | Random Forest | F1 Score | 0.8208 |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9+ |
| Dashboard | Streamlit, Altair |
| Data | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, Joblib |
| Visualization | Matplotlib, Altair |
| Testing | pytest |

---

## Project Structure

```
AirSense AI/
├── app/
│   └── app.py                  # Streamlit dashboard
├── assets/
│   ├── logo.png
│   ├── background.png
│   └── style.css
├── data/
│   ├── city_day.csv            # Main dataset (CPCB)
│   └── pune_sample_rows.csv    # Sample rows for reference
├── models/                     # Trained model artifacts (generated locally)
│   └── README.md
├── outputs/                    # Evaluation reports and charts
├── src/
│   ├── config.py               # Paths and feature configuration
│   ├── data_loader.py          # Dataset loading
│   ├── preprocessing.py        # Data cleaning
│   ├── feature_engineering.py  # Feature creation
│   ├── train_regression.py     # Regression model training
│   ├── train_classification.py # Classification model training
│   ├── regression_analysis.py  # Residual and error analysis
│   └── predictor.py            # Unified prediction engine
├── tests/
│   └── test_prediction_payload.py
├── .github/workflows/ci.yml    # GitHub Actions CI
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Dataset

- **Source:** [Central Pollution Control Board (CPCB)](https://cpcb.nic.in/) — City Day AQI dataset
- **File:** `data/city_day.csv`
- **Coverage:** 27 cities, 24,858 records (2015–2024)
- **Target:** AQI (regression) and AQI_Bucket (classification)

Place `city_day.csv` inside the `data/` folder before training or running the app.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AirSense-AI.git
cd AirSense-AI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

Trained model files (`.pkl`) are **not included** in this repository because they exceed GitHub's file size limit (~1 GB total). Generate them locally:

```bash
python scripts/train_all.py
```

Or run each script individually:

```bash
cd src
python train_regression.py
python train_classification.py
python regression_analysis.py
```

This creates model files in `models/` and evaluation outputs in `outputs/`.

### 5. Run the dashboard

```bash
streamlit run app/app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Usage

### Streamlit Dashboard

| Page | Description |
|------|-------------|
| Home | Project overview, KPIs, and feature summary |
| AQI Prediction | Enter pollutant values and get AQI + health advice |
| City Explorer | Historical trends and pollutant analysis by city |
| Analytics | Filterable dataset exploration and charts |
| Model Performance | Model comparison tables and evaluation plots |
| About | Project details, pipeline, and developer info |

### Prediction Engine (CLI)

```bash
cd src
python predictor.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## ML Pipeline

```
Raw Dataset (city_day.csv)
        │
        ▼
   Data Loading
        │
        ▼
   Preprocessing (missing values, duplicates, sorting)
        │
        ▼
   Feature Engineering (date, season, pollution ratios)
        │
        ▼
   Train / Test Split (80/20)
        │
        ├── Regression Models (Linear, Random Forest, XGBoost)
        │
        └── Classification Models (Logistic, Random Forest, XGBoost)
        │
        ▼
   Model Evaluation & Reports
        │
        ▼
   Prediction Engine + Streamlit Dashboard
```

---

## GitHub Upload Notes

- **Included in repo:** source code, assets, dataset, evaluation outputs (CSVs and PNGs)
- **Excluded from repo:** trained `.pkl` model files, virtual environments, caches (see `.gitignore`)
- **After cloning:** run the training scripts once to generate models before launching the app

---

## Author

**Khan Shabbir**  
M.Sc. Data Science — Savitribai Phule Pune University

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Disclaimer

Predictions are generated using machine learning models trained on historical air quality data. Results are intended for **educational and analytical purposes** only.
