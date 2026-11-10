import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "app.py"

spec = importlib.util.spec_from_file_location("airsense_app", APP_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_build_prediction_payload():
    values = {
        "pm25": 82.0,
        "pm10": 120.0,
        "no": 18.0,
        "no2": 32.0,
        "nox": 60.0,
        "nh3": 28.0,
        "co": 1.2,
        "so2": 16.0,
        "o3": 44.0,
        "benzene": 2.5,
        "toluene": 5.0,
        "xylene": 1.0,
        "year": 2024,
        "month": 7,
        "quarter": 3,
        "day_of_week": 1,
        "weekend": 0,
        "season": "Monsoon",
        "pm_ratio": 0.68,
        "nox_ratio": 1.87,
        "pollution_index": 305,
        "city": "Pune",
    }

    payload = module.build_prediction_payload(values)

    assert payload["PM2.5"] == 82.0
    assert payload["City"] == "Pune"
    assert payload["Season"] == "Monsoon"
    assert payload["Pollution_Index"] == 305
