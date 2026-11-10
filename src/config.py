""" ========================================================= AirSense AI Configuration File ========================================================= """ 
from pathlib import Path 

# -------------------------------------------------------- # Project Paths # -------------------------------------------------------- 
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
DATA_DIR = PROJECT_ROOT / "data" 
MODEL_DIR = PROJECT_ROOT / "models" 
OUTPUT_DIR = PROJECT_ROOT / "outputs" 

# Create folders automatically 
MODEL_DIR.mkdir(exist_ok=True) 
OUTPUT_DIR.mkdir(exist_ok=True) 

# -------------------------------------------------------- # Dataset # -------------------------------------------------------- 
DATA_FILE = DATA_DIR / "city_day.csv" 

# -------------------------------------------------------- # Target Columns # -------------------------------------------------------- 
TARGET_REGRESSION = "AQI" 
TARGET_CLASSIFICATION = "AQI_Bucket" 

# -------------------------------------------------------- # Date Column # -------------------------------------------------------- 
DATE_COLUMN = "Date" 
CITY_COLUMN = "City" 

# -------------------------------------------------------- # Random Seed # -------------------------------------------------------- 
RANDOM_STATE = 42 
TEST_SIZE = 0.20 

# -------------------------------------------------------- # Pollutants & Features # -------------------------------------------------------- 
POLLUTANT_COLUMNS = [ 
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

NUMERICAL_FEATURES = POLLUTANT_COLUMNS + [
    "Year", "Month", "Quarter", "DayOfWeek", "Weekend", "PM_Ratio", "NOx_Ratio", "Pollution_Index" 
] 

CATEGORICAL_FEATURES = [ 
    "City", "Season" 
]
