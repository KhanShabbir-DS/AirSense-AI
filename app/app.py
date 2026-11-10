"""
=========================================================
AirSense AI
Intelligent Air Quality Monitoring Platform

Author : Khan Shabbir
=========================================================
"""

from pathlib import Path
import sys
import base64
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LOGO_PATH = ASSETS_DIR / "logo.png"
BACKGROUND_PATH = ASSETS_DIR / "background.png"
CSS_PATH = ASSETS_DIR / "style.css"

st.set_page_config(
    page_title="AirSense AI",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(SRC_DIR))

try:
    from config import OUTPUT_DIR as CONFIG_OUTPUT_DIR
    OUTPUT_DIR = CONFIG_OUTPUT_DIR
except ImportError:
    pass

try:
    from predictor import AirSensePredictor
    from feature_engineering import get_season, compute_pollution_features
except Exception:
    import traceback

    st.error("Failed to import prediction engine modules.")
    st.code(traceback.format_exc())
    AirSensePredictor = None
    get_season = None
    compute_pollution_features = None

AQIPredictor = AirSensePredictor


@st.cache_resource
def load_prediction_engine():
    if AQIPredictor is None:
        return None
    return AQIPredictor()


# Set when the Streamlit app starts. Keeping module import side-effect free lets
# utility functions be imported without requiring locally trained model files.
predictor = None

# =========================================================
# Load CSS
# =========================================================


def load_css():
    if CSS_PATH.exists():
        css = CSS_PATH.read_text(encoding="utf-8")
        if BACKGROUND_PATH.exists():
            bg = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode()
            css = css.replace(
                'url("BACKGROUND_IMAGE")',
                f'url("data:image/png;base64,{bg}")',
            )
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        return

    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background: linear-gradient(90deg,#1d4ed8,#06b6d4) !important;
            color: white !important;
            padding: 0.6rem 1.1rem !important;
            border-radius: 12px !important;
            border: none !important;
            font-weight: 700 !important;
            box-shadow: 0 6px 18px rgba(13, 110, 253, 0.12) !important;
        }
        div.stButton > button:first-child:hover { filter: brightness(0.95); }
        h2.kpi { font-size: 44px !important; color: #0f172a !important; margin: 0.25rem 0 0.1rem !important; font-weight: 800 !important; }
        h4.kpi-label { font-size: 0.95rem !important; color: #475569 !important; margin: 0 !important; font-weight: 600 !important; }
        .kpi-card-1 { background: #e6f2ff !important; }
        .kpi-card-2 { background: #fff7e6 !important; }
        .kpi-card-3 { background: #e7fff5 !important; }
        .kpi-card-4 { background: #f6eefb !important; }
        .project-overview { background: #ffffff; color: #0f172a; padding: 1.2rem; border-radius: 14px; border: 1px solid #e6eef9; }
        .airsense-footer p { margin: 0.12rem 0; color: #94a3b8; }
        .airsense-footer h4 { margin: 0 0 0.2rem 0; }
        .sidebar-logo { display: block; margin-bottom: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# Sidebar
# =========================================================

def sidebar():
    with st.sidebar:
        if LOGO_PATH.exists():
            try:
                st.image(str(LOGO_PATH), width=250)
            except Exception:
                st.image(str(LOGO_PATH), use_container_width=True)

        st.markdown(
            """
            <h2 style="text-align:center;">
            AirSense AI
            </h2>

            <p style="text-align:center;color:#C8D6E5;">
            Environmental Intelligence Platform
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("## 🌍 AirSense AI")
        st.markdown("**Version 1.0**")
        st.markdown("Intelligent Air Quality\nMonitoring Platform")
        st.markdown("---")

        options = [
            "🏠 Home",
            "🤖 AQI Prediction",
            "🌍 City Explorer",
            "📊 Analytics",
            "🧠 Model Performance",
            "ℹ About",
        ]

        page_key = "page"
        if "nav_to" in st.session_state:
            st.session_state[page_key] = st.session_state.pop("nav_to")

        current_page = st.session_state.get(page_key, "🏠 Home")
        default_index = options.index(current_page) if current_page in options else 0
        page = st.radio(" ", options, index=default_index, key=page_key)

        st.markdown("---")
        st.caption("Intelligent Air Quality Monitoring")

    return page

# =========================================================
# Hero Section
# =========================================================

def hero():

    left, right = st.columns([1,4])

    with left:

        st.image(
            str(LOGO_PATH),
            width=460
        )

    with right:

        st.markdown("""
        # 🌍 AirSense AI

        ### AI-Powered Air Quality Prediction & Environmental Analytics Platform

        Predict • Analyze • Visualize • Protect

        A complete Machine Learning platform that predicts Air Quality Index (AQI),
        analyzes pollution trends, compares ML models and provides intelligent
        health recommendations using historical environmental data.
        """)

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("🔮 Predict AQI", use_container_width=True):
            st.session_state["nav_to"] = "🤖 AQI Prediction"
            st.rerun()

    with c2:
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state["nav_to"] = "📊 Analytics"
            st.rerun()

    with c3:
        if st.button("🌍 City Explorer", use_container_width=True):
            st.session_state["nav_to"] = "🌍 City Explorer"
            st.rerun()

    st.divider()

# =========================================================
# Project Overview
# =========================================================

def project_overview():
    st.markdown(
        """
        <div class="project-overview">
            <h3 style="margin-top: 0;">Project Overview</h3>
            <p style="margin-bottom: 0.6rem; line-height: 1.5;">AirSense AI combines Machine Learning, Air Pollution Analytics, and Health Intelligence to predict AQI, classify air quality, analyze city trends, and provide personalized health recommendations.</p>
            <ul style="margin: 0; padding-left: 1.1rem; line-height: 1.6;">
                <li>✔ 27 Cities</li>
                <li>✔ 24,858 Records</li>
                <li>✔ Random Forest (R² = 0.906)</li>
                <li>✔ Random Forest Classifier (82.21%)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# KPI Cards
# =========================================================

def kpis():

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="kpi-card-1" style="border: 1px solid #bfdbfe; border-radius: 16px; padding: 1rem; text-align: center; min-height: 140px;">
                <div style="font-size: 2rem;">📍</div>
                <h2 class="kpi" style="margin: 0.4rem 0 0.2rem;">27</h2>
                <h4 class="kpi-label" style="margin: 0;">Cities</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="kpi-card-2" style="border: 1px solid #fde68a; border-radius: 16px; padding: 1rem; text-align: center; min-height: 140px;">
                <div style="font-size: 2rem;">📄</div>
                <h2 class="kpi" style="margin: 0.4rem 0 0.2rem;">24,858</h2>
                <h4 class="kpi-label" style="margin: 0;">Records</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="kpi-card-3" style="border: 1px solid #a7f3d0; border-radius: 16px; padding: 1rem; text-align: center; min-height: 140px;">
                <div style="font-size: 2rem;">📈</div>
                <h2 class="kpi" style="margin: 0.4rem 0 0.2rem;">0.906</h2>
                <h4 class="kpi-label" style="margin: 0;">Regression R²</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="kpi-card-4" style="border: 1px solid #ddd6fe; border-radius: 16px; padding: 1rem; text-align: center; min-height: 140px;">
                <div style="font-size: 2rem;">🧠</div>
                <h2 class="kpi" style="margin: 0.4rem 0 0.2rem;">82.21%</h2>
                <h4 class="kpi-label" style="margin: 0;">Classification Accuracy</h4>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# Features
# =========================================================

def features():

    st.markdown(
        '<div class="section-title">✨ Platform Features</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="feature">
        <h1>🤖</h1>
        <h3>AQI Prediction</h3>
        <p>
        Predict Air Quality Index using
        Machine Learning.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature">
        <h1>🌍</h1>
        <h3>City Intelligence</h3>
        <p>
        Explore historical AQI,
        trends and pollution levels.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature">
        <h1>📊</h1>
        <h3>ML Analytics</h3>
        <p>
        Regression, Classification,
        Feature Importance &
        Model Comparison.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="feature">
        <h1>🩺</h1>
        <h3>Health Advisory</h3>
        <p>
        Personalized health
        recommendations based
        on predicted AQI.
        </p>
        </div>
        """, unsafe_allow_html=True)
# =========================================================
# Technology Stack
# =========================================================

def tech_stack():

    st.markdown(
        '<div style="margin-top: 1.5rem; font-size: 1.2rem; font-weight: 700;">💻 Technology Stack</div>',
        unsafe_allow_html=True
    )

    technologies = [
        "Python",
        "Streamlit",
        "Pandas",
        "NumPy",
        "Scikit-Learn",
        "XGBoost",
        "Matplotlib",
        "Joblib"
    ]

    badge_html = "".join(
        f"<span style='display: inline-block; background: #e2e8f0; color: #0f172a; padding: 0.45rem 0.8rem; border-radius: 999px; margin: 0.3rem; font-weight: 600;'> {tech} </span>"
        for tech in technologies
    )

    st.markdown(
        f"<div style='margin-top: 0.8rem;'>{badge_html}</div>",
        unsafe_allow_html=True
    )
# =========================================================
# Footer
# =========================================================

def footer():
    st.markdown("<hr style='margin-top: 1.6rem; opacity:0.18;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="airsense-footer" style="text-align:center; padding:0.6rem 0 1.2rem 0;">
            <h4 style="margin-bottom: 0.2rem;">🌍 AirSense AI</h4>
            <p style="margin: 0.08rem 0;">Developed by <strong>Khan Shabbir</strong></p>
            <p style="margin: 0.08rem 0;">M.Sc Data Science — Savitribai Phule Pune University</p>
            <p style="margin-top: 0.4rem; color:#94a3b8;">© 2026</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# Home Page
# =========================================================

def home():

    hero()

    st.write("")

    project_overview()

    st.write("")

    kpis()

    st.write("")

    features()

    st.write("")

    tech_stack()

    st.write("")

    footer()

# =========================================================
# Prediction Helpers
# =========================================================

def build_prediction_payload(values):

    return {
        "PM2.5": float(values["pm25"]),
        "PM10": float(values["pm10"]),
        "NO": float(values["no"]),
        "NO2": float(values["no2"]),
        "NOx": float(values["nox"]),
        "NH3": float(values["nh3"]),
        "CO": float(values["co"]),
        "SO2": float(values["so2"]),
        "O3": float(values["o3"]),
        "Benzene": float(values["benzene"]),
        "Toluene": float(values["toluene"]),
        "Xylene": float(values["xylene"]),
        "Year": int(values["year"]),
        "Month": int(values["month"]),
        "Quarter": int(values["quarter"]),
        "DayOfWeek": int(values["day_of_week"]),
        "Weekend": int(values["weekend"]),
        "Season": values["season"],
        "PM_Ratio": float(values["pm_ratio"]),
        "NOx_Ratio": float(values["nox_ratio"]),
        "Pollution_Index": float(values["pollution_index"]),
        "City": values["city"]
    }


def render_prediction_result(result):
    prediction = result["Prediction"]
    health = result["Health"]
    status = result["Status"]
    insights = result.get("City Insights", {})

    st.success("Prediction completed successfully")

    # Top KPI row: AQI (colored), Category, Confidence + progress
    kpi_aqi, kpi_cat, kpi_conf = st.columns([2, 1, 1])

    with kpi_aqi:
        st.markdown(
            f"<div style='background:{status.get('Color', '#111827')}; color: white; padding: 18px; border-radius: 12px; text-align: center;'>"
            f"<h2 style='margin:0; font-size:38px'>{prediction['AQI']:.1f}</h2>"
            f"<div style='opacity:0.9'>{prediction['Category']}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with kpi_cat:
        st.metric("Category", prediction["Category"]) 

    with kpi_conf:
        conf = prediction.get("Confidence") or 0.0
        st.metric("Confidence", f"{conf*100:.1f}%")
        try:
            st.progress(float(conf))
        except Exception:
            pass

    st.write("")

    # Two-column layout: Health guidance (left) and City Insights (right)
    left, right = st.columns([2, 2])

    with left:
        st.markdown("### 🩺 Health Recommendation")
        st.markdown(
            f"<div style='background:#1a3030; border:1px solid #e6eef9; padding:12px; border-radius:10px;'>"
            f"<strong>{status.get('Icon','')}</strong> {health.get('Advice','')}<br><br>"
            f"<strong>Mask:</strong> {health.get('Mask','N/A')}<br>"
            f"<strong>Outdoor Activity:</strong> {health.get('Outdoor Activity','N/A')}<br>"
            f"<strong>Sensitive Groups:</strong> {health.get('Sensitive Groups','N/A')}"
            f"</div>",
            unsafe_allow_html=True
        )

    with right:
        st.markdown("### 🌍 City Insights")
        if insights:
            st.metric("Average AQI", insights.get('Historical Average AQI', 'N/A'))
            st.metric("Highest AQI", insights.get('Highest AQI', 'N/A'))
            st.metric("Lowest AQI", insights.get('Lowest AQI', 'N/A'))
            st.write(f"**Years Covered:** {insights.get('Years Covered', 'N/A')}")
            st.write(f"**Trend:** {insights.get('Trend', 'N/A')}")
        else:
            st.info("No historical data available.")

    st.write("")

    # Technical details
    with st.expander("⚙ Technical Details"):
        st.write(f"Regression Model : {result['Technical'].get('Regression Model','N/A')}")
        st.write(f"Classification Model : {result['Technical'].get('Classification Model','N/A')}")
        st.write(f"Prediction Time : {result['Technical'].get('Prediction Time','N/A')}")


def prediction_page():

    st.title("🤖 AQI Prediction")

    st.write(
        "Enter the pollutant values below and predict the Air Quality Index."
    )

    st.divider()

    left, right = st.columns([1, 1])

    with left:

        st.subheader("🌍 Location")

        city = st.selectbox(
            "City",
            [
                "Ahmedabad",
                "Amaravati",
                "Bengaluru",
                "Chandigarh",
                "Chennai",
                "Coimbatore",
                "Delhi",
                "Ernakulam",
                "Gurugram",
                "Guwahati",
                "Hyderabad",
                "Jaipur",
                "Kochi",
                "Kolkata",
                "Lucknow",
                "Mumbai",
                "Patna",
                "Pune",
                "Shillong",
                "Talcher",
                "Thiruvananthapuram",
                "Visakhapatnam"
            ]
        )

        st.subheader("🧪 Pollutant Inputs")

        pm25 = st.number_input(
            "PM2.5",
            min_value=0.0,
            value=80.0
        )

        pm10 = st.number_input(
            "PM10",
            min_value=0.0,
            value=120.0
        )

        no = st.number_input(
            "NO",
            min_value=0.0,
            value=15.0
        )

        no2 = st.number_input(
            "NO2",
            min_value=0.0,
            value=35.0
        )

        nox = st.number_input(
            "NOx",
            min_value=0.0,
            value=55.0
        )

        nh3 = st.number_input(
            "NH3",
            min_value=0.0,
            value=25.0
        )

    with right:

        st.subheader("🌫 Additional Pollutants")

        co = st.number_input(
            "CO",
            min_value=0.0,
            value=1.2
        )

        so2 = st.number_input(
            "SO2",
            min_value=0.0,
            value=18.0
        )

        o3 = st.number_input(
            "O3",
            min_value=0.0,
            value=40.0
        )

        benzene = st.number_input(
            "Benzene",
            min_value=0.0,
            value=2.5
        )

        toluene = st.number_input(
            "Toluene",
            min_value=0.0,
            value=10.0
        )

        xylene = st.number_input(
            "Xylene",
            min_value=0.0,
            value=3.0
        )

        st.markdown("###")

        predict_btn = st.button(
            "🚀 Predict AQI",
            use_container_width=True
        )

    if predict_btn:
        if predictor is None or get_season is None or compute_pollution_features is None:
            st.error("Prediction engine is not available.")
            return

        with st.spinner("Running AI Prediction..."):
            try:
                now = datetime.now()
                year = now.year
                month = now.month
                quarter = ((month - 1) // 3) + 1
                day_of_week = now.weekday()
                weekend = 1 if day_of_week >= 5 else 0
                season = get_season(month)
                pm_ratio, nox_ratio, pollution_index = compute_pollution_features(
                    pm25, pm10, no2, nox, co, so2
                )

                values = {
                    "pm25": pm25,
                    "pm10": pm10,
                    "no": no,
                    "no2": no2,
                    "nox": nox,
                    "nh3": nh3,
                    "co": co,
                    "so2": so2,
                    "o3": o3,
                    "benzene": benzene,
                    "toluene": toluene,
                    "xylene": xylene,
                    "year": year,
                    "month": month,
                    "quarter": quarter,
                    "day_of_week": day_of_week,
                    "weekend": weekend,
                    "season": season,
                    "pm_ratio": pm_ratio,
                    "nox_ratio": nox_ratio,
                    "pollution_index": pollution_index,
                    "city": city
                }

                payload = build_prediction_payload(values)
                result = predictor.predict(payload)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        st.session_state["prediction"] = result
        try:
            # Render the result immediately for better UX
            render_prediction_result(result)
        except Exception:
            # If rendering fails for any reason, continue silently (the stored result remains)
            pass

        # ==========================================
        # Prediction History
        # ==========================================

        if "history" not in st.session_state:
            st.session_state["history"] = []

        st.session_state["history"].append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "City": city,
            "AQI": round(result["Prediction"]["AQI"], 2),
            "Category": result["Prediction"]["Category"]
        })

        st.write("")
        st.markdown("## 📜 Prediction History")

        hcol1, hcol2 = st.columns([1, 3])
        with hcol1:
            if st.button("🗑 Clear History"):
                st.session_state["history"] = []
                st.rerun()

        history_df = pd.DataFrame(st.session_state["history"])

        with hcol2:
            if not history_df.empty:
                csv = history_df.to_csv(index=False)
                st.download_button(
                    "📥 Download History",
                    csv,
                    "prediction_history.csv",
                    "text/csv"
                )

        st.dataframe(
            history_df,
            use_container_width=True
        )


def city_explorer_page():

    st.title("🌍 City Explorer")

    if predictor is None or not hasattr(predictor, 'city_data'):
        st.error("City data is not available. Ensure the prediction engine loaded correctly.")
        return

    df = predictor.city_data.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    cities = sorted(df['City'].dropna().unique())

    city = st.selectbox("Select City", cities, index=cities.index('Pune') if 'Pune' in cities else 0)

    city_df = df[df['City'].str.lower() == str(city).lower()].dropna(subset=['Date'])

    if city_df.empty:
        st.info("No historical records for this city.")
        return

    # Summary
    avg_aqi = round(city_df['AQI'].mean(), 2)
    max_aqi = round(city_df['AQI'].max(), 2)
    min_aqi = round(city_df['AQI'].min(), 2)
    years = f"{city_df['Date'].dt.year.min()} - {city_df['Date'].dt.year.max()}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average AQI", avg_aqi)
    c2.metric("Highest AQI", max_aqi)
    c3.metric("Lowest AQI", min_aqi)
    c4.metric("Years Covered", years)

    st.markdown("---")

    # AQI Timeline
    st.subheader("📈 AQI Timeline")
    timeseries = (
        city_df
        .set_index("Date")
        .resample("D")["AQI"]
        .mean()
        .ffill()
    )
    st.line_chart(timeseries)

    # Monthly Average
    st.subheader("📅 Monthly Average AQI")
    monthly = city_df.copy()
    monthly['Month'] = monthly['Date'].dt.to_period('M').dt.to_timestamp()
    monthly_avg = monthly.groupby('Month')['AQI'].mean().reset_index()
    chart = alt.Chart(monthly_avg).mark_bar().encode(x='Month:T', y='AQI:Q').properties(height=300)
    st.altair_chart(chart, use_container_width=True)

    # Seasonal AQI
    st.subheader("🌦 Seasonal AQI")
    def month_to_season(m):
        if m in (12,1,2): return 'Winter'
        if m in (3,4,5): return 'Summer'
        if m in (6,7,8,9): return 'Monsoon'
        return 'Post-Monsoon'

    seasonal = city_df.copy()
    seasonal['Season'] = seasonal['Date'].dt.month.apply(month_to_season)
    season_avg = seasonal.groupby('Season')['AQI'].mean().reindex(['Winter','Summer','Monsoon','Post-Monsoon']).reset_index()
    schart = alt.Chart(season_avg).mark_bar().encode(x='Season:N', y='AQI:Q', color='Season:N').properties(height=250)
    st.altair_chart(schart, use_container_width=True)

    # Top Pollutants
    st.subheader("🏆 Top Pollutants")
    pollutants = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]
    pollutant_means = city_df[pollutants].mean().sort_values(ascending=False).reset_index()
    pollutant_means.columns = ['Pollutant','Mean']
    pchart = alt.Chart(pollutant_means).mark_bar().encode(x='Mean:Q', y=alt.Y('Pollutant:N', sort='-x')).properties(height=300)
    st.altair_chart(pchart, use_container_width=True)

    # Correlation Heatmap
    st.subheader("📊 Pollutant Correlation")
    corr = city_df[pollutants].corr()
    st.dataframe(corr.style.background_gradient(cmap='RdYlBu_r'))

    # AQI Distribution
    st.subheader("📉 AQI Distribution")
    hist = alt.Chart(city_df).mark_bar().encode(alt.X('AQI:Q', bin=alt.Bin(maxbins=30)), y='count()').properties(height=250)
    st.altair_chart(hist, use_container_width=True)

    # Compare two cities
    st.markdown("---")
    st.subheader("🌍 Compare Two Cities")
    compare_cities = st.multiselect("Select up to two cities to compare", cities, default=[city, cities[0] if cities[0] != city else (cities[1] if len(cities)>1 else city)])
    if compare_cities:
        comp_df = df[df['City'].isin(compare_cities)].copy()
        comp_ts = comp_df.groupby(['City', pd.Grouper(key='Date', freq='M')])['AQI'].mean().reset_index()
        comp_chart = alt.Chart(comp_ts).mark_line(point=True).encode(x='Date:T', y='AQI:Q', color='City:N').properties(height=300)
        st.altair_chart(comp_chart, use_container_width=True)

# =========================================================
# Analytics Page
# =========================================================

def analytics_page():

    st.title("📊 Analytics Dashboard")

    if predictor is None or not hasattr(predictor, "city_data"):
        st.error("Analytics data is unavailable.")
        return

    df = predictor.city_data.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    # =====================================================
    # Create Season Column
    # =====================================================

    month_to_season = {
        12: "Winter",
        1: "Winter",
        2: "Winter",

        3: "Summer",
        4: "Summer",
        5: "Summer",

        6: "Monsoon",
        7: "Monsoon",
        8: "Monsoon",
        9: "Monsoon",

        10: "Post-Monsoon",
        11: "Post-Monsoon"
    }

    df["Season"] = (
        df["Date"]
        .dt.month
        .map(month_to_season)
    )
    st.sidebar.markdown("## 📊 Analytics Filters")

    selected_city = st.sidebar.selectbox(
        "City",
        ["All"] + sorted(df["City"].unique().tolist())
    )

    selected_bucket = st.sidebar.selectbox(
        "AQI Category",
        ["All"] + sorted(df["AQI_Bucket"].unique().tolist())
    )

    years = sorted(df["Date"].dt.year.unique())

    selected_year = st.sidebar.selectbox(
        "Year",
        ["All"] + years
    )

    filtered_df = df.copy()
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df["City"] == selected_city]
    if selected_bucket != "All":
        filtered_df = filtered_df[filtered_df["AQI_Bucket"] == selected_bucket]
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df["Date"].dt.year == int(selected_year)]

    if filtered_df.empty:
        st.info("No records match the current filters.")
        return

    total_records = len(filtered_df)

    total_cities = filtered_df["City"].nunique()

    avg_aqi = round(filtered_df["AQI"].mean(), 2)

    median_aqi = round(filtered_df["AQI"].median(), 2)

    max_aqi = round(filtered_df["AQI"].max(), 2)

    min_year = filtered_df["Date"].dt.year.min()

    max_year = filtered_df["Date"].dt.year.max()

    st.markdown("## 📈 Dataset Overview")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "📄 Records",
        f"{total_records:,}"
    )

    c2.metric(
        "🌍 Cities",
        total_cities
    )

    c3.metric(
        "📅 Years",
        f"{min_year}-{max_year}"
    )

    c4.metric(
        "📊 Avg AQI",
        avg_aqi
    )

    c5.metric(
        "📈 Median AQI",
        median_aqi
    )

    c6.metric(
        "🔥 Max AQI",
        max_aqi
    )

    st.divider()

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Dataset",
        data=csv,
        file_name="airsense_analytics.csv",
        mime="text/csv"
    )

    st.divider()

    # =====================================================
    # AQI Distribution
    # =====================================================

    st.subheader("📊 AQI Distribution")

    hist = (
        alt.Chart(filtered_df)
        .mark_bar(color="#25B0EB")
        .encode(
            alt.X(
                "AQI:Q",
                bin=alt.Bin(maxbins=35),
                title="AQI"
            ),
            y=alt.Y(
                "count()",
                title="Records"
            )
        )
        .properties(height=350)
    )

    st.altair_chart(
        hist,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # Most Polluted Cities
    # =====================================================

    st.subheader("🏆 Most Polluted Cities")

    city_avg = (
        filtered_df.groupby("City")["AQI"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    chart = (
        alt.Chart(city_avg.head(10))
        .mark_bar(color="#DF9F1F")
        .encode(
            x=alt.X(
                "AQI:Q",
                title="Average AQI"
            ),
            y=alt.Y(
                "City:N",
                sort="-x"
            ),
            tooltip=["City", "AQI"]
        )
        .properties(height=400)
    )

    st.altair_chart(
        chart,
        use_container_width=True
    )
    # =====================================================
    # Monthly AQI Trend
    # =====================================================

    st.divider()

    st.subheader("📅 Monthly Average AQI Trend")

    monthly_df = filtered_df.copy()

    monthly_df["Month"] = monthly_df["Date"].dt.month_name()

    month_order = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]

    monthly_avg = (
        monthly_df.groupby("Month")["AQI"]
        .mean()
        .reindex(month_order)
        .reset_index()
    )

    line_chart = (
        alt.Chart(monthly_avg)
        .mark_line(point=True, color="#3B82F6", strokeWidth=4)
        .encode(
            x=alt.X("Month:N", sort=month_order),
            y=alt.Y("AQI:Q", title="Average AQI"),
            tooltip=["Month","AQI"]
        )
        .properties(height=400)
    )

    st.altair_chart(line_chart, use_container_width=True)
    # =====================================================
    # Seasonal AQI
    # =====================================================

    st.divider()

    st.subheader("🌦 Seasonal AQI Analysis")

    seasonal_avg = (
        filtered_df.groupby("Season")["AQI"]
        .mean()
        .reindex([
            "Winter",
            "Summer",
            "Monsoon",
            "Post-Monsoon"
        ])
        .reset_index()
    )

    season_chart = (
        alt.Chart(seasonal_avg)
        .mark_bar(color="#5610B9")
        .encode(
            x="Season",
            y="AQI",
            tooltip=["Season","AQI"]
        )
        .properties(height=350)
    )

    st.altair_chart(
        season_chart,
        use_container_width=True
    )

    # =====================================================
    # Cleanest Cities
    # =====================================================

    st.divider()

    st.subheader("🌿 Top 10 Cleanest Cities")

    cleanest = (
        filtered_df.groupby("City")["AQI"]
        .mean()
        .sort_values()
        .head(10)
        .reset_index()
    )

    clean_chart = (
        alt.Chart(cleanest)
        .mark_bar(color="#4CBF76")
        .encode(
            x="AQI",
            y=alt.Y("City", sort="x"),
            tooltip=["City","AQI"]
        )
        .properties(height=400)
    )

    st.altair_chart(
        clean_chart,
        use_container_width=True
    )

    # =====================================================
    # AQI Categories
    # =====================================================

    st.divider()

    st.subheader("🥧 AQI Category Distribution")

    bucket = (
        filtered_df["AQI_Bucket"]
        .value_counts()
        .reset_index()
    )

    bucket.columns = ["AQI_Bucket","Count"]

    pie = (
        alt.Chart(bucket)
        .mark_arc()
        .encode(
            theta="Count",
            color="AQI_Bucket",
            tooltip=["AQI_Bucket","Count"]
        )
        .properties(height=450)
    )

    st.altair_chart(
        pie,
        use_container_width=True
    )

    st.divider()
    st.subheader("🔥 Pollutant Correlation Heatmap")

    corr_cols = [
        "PM2.5", "PM10", "NO", "NO2", "NOx",
        "NH3", "CO", "SO2", "O3",
        "Benzene", "Toluene", "Xylene", "AQI"
    ]

    corr = df[corr_cols].corr().round(2)

    st.dataframe(
        corr.style.background_gradient(cmap="RdYlGn_r"),
        use_container_width=True
    )

    st.divider()
    st.subheader("🧪 Pollutant Statistics")

    stats = df[corr_cols].describe().T

    st.dataframe(
        stats,
        use_container_width=True
    )

def model_performance_page():
    reg_path = OUTPUT_DIR / "regression_results.csv"
    clf_path = OUTPUT_DIR / "classification_results.csv"

    if not reg_path.exists() or not clf_path.exists():
        st.error("Model performance reports are not available yet.")
        return

    reg_results = pd.read_csv(reg_path)
    clf_results = pd.read_csv(clf_path)

    reg_results.columns = [col.strip() for col in reg_results.columns]
    clf_results.columns = [col.strip() for col in clf_results.columns]

    def get_model_name(df):
        for col in ["Model", "model", "Model Name", "Name"]:
            if col in df.columns:
                return col
        return df.columns[0]

    def get_metric_value(row, candidates):
        for candidate in candidates:
            if candidate in row.index:
                return row[candidate]
        return None

    reg_model_col = get_model_name(reg_results)
    clf_model_col = get_model_name(clf_results)

    reg_results = reg_results.copy()
    clf_results = clf_results.copy()

    reg_results = reg_results.rename(columns={reg_model_col: "Model"})
    clf_results = clf_results.rename(columns={clf_model_col: "Model"})

    best_reg = reg_results.iloc[0]
    best = clf_results.iloc[0]

    st.subheader("📈 Regression Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Best Model", best_reg["Model"])
    c2.metric("R²", f"{get_metric_value(best_reg, ['R2', 'R^2', 'r2']):.3f}")
    c3.metric("RMSE", f"{get_metric_value(best_reg, ['RMSE', 'rmse']):.2f}")
    c4.metric("MAE", f"{get_metric_value(best_reg, ['MAE', 'mae']):.2f}")

    st.subheader("🧠 Classification Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Best Model", best["Model"])
    c2.metric("Accuracy", f"{(val * 100):.2f}%" if (val := get_metric_value(best, ['Accuracy', 'accuracy'])) is not None else "N/A")

    c3.metric("Precision", f"{get_metric_value(best, ['Precision', 'precision']):.3f}")
    c4.metric("F1 Score", f"{get_metric_value(best, ['F1', 'f1', 'F1 Score']):.3f}")

    st.divider()

    st.subheader("Regression Model Comparison")

    st.dataframe(
        reg_results,
        use_container_width=True
    )
    st.divider()
    st.subheader("📈 Actual vs Predicted AQI")

    st.image(
        OUTPUT_DIR / "actual_vs_predicted.png",
        use_container_width=True
    )

    st.divider()
    st.subheader("📉 Residual Analysis")

    st.image(
        OUTPUT_DIR / "residual_analysis.png",
        use_container_width=True
    )

    st.divider()
    st.subheader("📊 Prediction Error Distribution")

    st.image(
        OUTPUT_DIR / "prediction_error_distribution.png",
        use_container_width=True
    )

    st.divider()

    st.subheader("Classification Model Comparison")

    st.dataframe(
        clf_results,
        use_container_width=True
    )

    st.divider()
    st.subheader("🧠 Confusion Matrix")

    st.image(
        OUTPUT_DIR / "confusion_matrix.png",
        use_container_width=True
    )

    st.divider()

    st.subheader("📊 Feature Importance")

    st.image(
        OUTPUT_DIR / "feature_importance.png",
        use_container_width=True
    )

    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        with open(
            OUTPUT_DIR/"regression_results.csv",
            "rb"
        ) as f:

            st.download_button(
                "📥 Regression Report",
                f,
                file_name="regression_results.csv"
            )

    with col2:

        with open(
            OUTPUT_DIR/"classification_results.csv",
            "rb"
        ) as f:

            st.download_button(
                "📥 Classification Report",
                f,
                file_name="classification_results.csv"
            )
# ==========================================================
# About Page
# ==========================================================

def about_page():

    st.title("🌍 About AirSense AI")

    st.markdown("""
    # AirSense AI

    ### AI-Powered Air Quality Prediction & Environmental Analytics Platform
    """)

    st.success("Version 1.0 • Machine Learning Powered • Built with Python & Streamlit")

    st.divider()
    col1, col2 = st.columns([2,1])

    with col1:

        st.markdown("""
        ## 🚀 Project Overview

        AirSense AI is an end-to-end Machine Learning application designed to
        predict the Air Quality Index (AQI) of Indian cities using historical
        pollution data.

        The system combines advanced data preprocessing, feature engineering,
        machine learning models, and interactive analytics into a modern
        dashboard for environmental monitoring.

        The platform enables users to:

        - 🔮 Predict Air Quality Index (AQI)
        - 🌍 Explore city-wise pollution statistics
        - 📊 Analyze long-term pollution trends
        - 🤖 Compare Machine Learning models
        - ❤️ Receive health recommendations based on AQI levels
        """)

    with col2:

        st.metric("🏙 Cities", "27")

        st.metric("📄 Records", "24,858")

        st.metric("🧠 ML Models", "6")

        st.metric("📈 Best Regression", "Random Forest")

        st.metric("🎯 Best Accuracy", "82.21%")

    st.divider()

    st.subheader("✨ Key Highlights")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info("""
        ### 🔮 Prediction

        • AQI Prediction

        • AQI Category

        • Confidence Score

        • Health Advice
        """)

    with c2:

            st.info("""
        ### 📊 Analytics

        • AQI Distribution

        • Seasonal Trends

        • Monthly Trends

        • Correlation Analysis
        """)

    with c3:

            st.info("""
        ### 🤖 Machine Learning

        • Random Forest

        • XGBoost

        • Linear Regression

        • Logistic Regression
        """)

    st.divider()

    st.subheader("📊 Dataset Information")

    dataset = pd.DataFrame({
        "Property":[
            "Dataset Source",
            "Total Cities",
            "Total Records",
            "Years Covered",
            "Features",
            "Target Variable",
            "Missing Values",
            "Duplicates Removed"
        ],

        "Value":[
            "Central Pollution Control Board (CPCB)",
            "27",
            "24,858",
            "2015 - 2024",
            "27",
            "AQI",
            "Handled During Preprocessing",
            "Yes"
        ]
    })

    st.dataframe(
        dataset,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🧠 Machine Learning Pipeline")

    st.markdown("""
        Raw Air Quality Dataset
        │
        ▼
        Data Loading
        │
        ▼
        Data Preprocessing
        │
        ▼
        Feature Engineering
        │
        ▼
        Train/Test Split
        │
        ▼
        Regression Models
        │
        ▼
        Classification Models
        │
        ▼
        Model Evaluation
        │
        ▼
        Prediction Engine
        │
        ▼
        Interactive Dashboard


    """)

    st.divider()

    st.subheader("🛠 Technology Stack")

    col1,col2,col3,col4 = st.columns(4)

    with col1:

        st.success("""
        ### Programming

        Python

        Pandas

        NumPy
        """)

    with col2:

         st.success("""
        ### Machine Learning

        Scikit-Learn

        XGBoost

        Joblib
        """)

    with col3:

        st.success("""
        ### Visualization

        Altair

        Matplotlib

        Plotly
        """)

    with col4:

        st.success("""
        ### Application

        Streamlit

        GitHub

        VS Code
        """)

    st.divider()

    st.subheader("📁 Project Architecture")

    st.code("""
AirSense AI/

├── app/
│   └── app.py
├── assets/
│   ├── logo.png
│   ├── background.png
│   └── style.css
├── data/
│   └── city_day.csv
├── models/
├── outputs/
├── scripts/
│   └── train_all.py
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_regression.py
│   ├── train_classification.py
│   ├── regression_analysis.py
│   └── predictor.py
├── tests/
├── requirements.txt
└── README.md
    """)

    st.divider()

    st.subheader("📈 Model Performance Summary")

    left,right = st.columns(2)

    with left:

        st.info("""
        ## Regression

        🏆 Best Model

        Random Forest

        R² : **0.9023**

        RMSE : **42.15**

        MAE : **20.43**
        """)

    with right:

        st.info("""
        ## Classification

        🏆 Best Model

        Random Forest

        Accuracy : **82.21%**

        Precision : **82.14%**

        Recall : **82.21%**

        F1 Score : **82.08%**
        """)

    st.divider()

    st.subheader("🚀 Future Enhancements")

    col1, col2 = st.columns(2)

    with col1:

        st.success("""
        ### 🌍 Planned Features

        • Live AQI API Integration

        • Weather Forecast Integration

        • Interactive Pollution Maps

        • Mobile Application

        • IoT Sensor Connectivity

        • Real-Time AQI Monitoring
        """)

    with col2:

        st.success("""
        ### 🤖 AI Enhancements

        • Deep Learning Models

        • LSTM AQI Forecasting

        • Explainable AI (SHAP)

        • Auto Model Retraining

        • Satellite Data Integration

        • Personalized Health Alerts
        """)

    st.divider()

    st.subheader("📈 Project Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Python Files", "12+")

    c2.metric("ML Models", "6")

    c3.metric("Charts", "15+")

    c4.metric("Dataset Size", "24,858 Records")

    st.divider()

    st.subheader("👨‍💻 Developer")

    left, right = st.columns([1,3])

    with left:

        st.markdown("# 👨‍💻")

    with right:

        st.markdown("""
        ## KHAN SHABBIR

        **M.Sc. Data Science**

        Savitribai Phule Pune University

        Passionate about Artificial Intelligence, Machine Learning,
        Data Science and Environmental Analytics.

        ### Project

        AirSense AI

        An end-to-end Machine Learning platform for Air Quality
        Prediction, Environmental Analytics and Decision Support.
        """)

    st.divider()

    st.subheader("🙏 Acknowledgements")

    st.info("""
        This project was developed as part of the M.Sc. Data Science curriculum.

        Special thanks to:

        • Central Pollution Control Board (CPCB)
        • Savitribai Phule Pune University
        • Open-source Python Community
        • Scikit-Learn
        • XGBoost
        • Streamlit
        """)

    st.divider()

    st.markdown(
        """
        <div style="text-align:center; padding:25px;">

        <h2>🌍 AirSense AI</h2>

        <p>
        AI-Powered Air Quality Prediction & Environmental Analytics Platform
        </p>

        <hr>

        <p>
        Version <b>1.0</b>
        </p>

        <p>
        Built with ❤️ using Python, Streamlit and Machine Learning
        </p>

        <p>
        © 2026 KHAN SHABBIR
        </p>

        </div>
        """,
        unsafe_allow_html=True
        )

    st.caption(
    "Disclaimer: Predictions are generated using machine learning models trained on historical air quality data and are intended for educational and analytical purposes."
)
 
    

# =========================================================
# Main
# =========================================================

def main():
    global predictor
    predictor = load_prediction_engine()

    load_css()
    page = sidebar()

    if page == "🏠 Home":
        home()
    elif page == "🤖 AQI Prediction":
        prediction_page()
    elif page == "🌍 City Explorer":
        city_explorer_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "🧠 Model Performance":
        st.title("🧠 Model Performance")
        model_performance_page()
    elif page == "ℹ About":
        about_page()


if __name__ == "__main__":
    main()

