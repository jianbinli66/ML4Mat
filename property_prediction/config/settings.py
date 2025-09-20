import os
from pathlib import Path
from datetime import datetime

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"

# Create directories
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data settings
DATA_FILE = DATA_DIR / "clean_property_data.csv"
FEATURE_PREFIXES = ["Formula_", "Condition_"]
PROPERTY_PREFIXES = ["Property_"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
MIN_SAMPLES_THRESHOLD = 50

# Model settings
MODEL_CONFIGS = {
    "RandomForest": {
        "class": "RandomForestRegressor",
        "params": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
    },
    "GradientBoosting": {
        "class": "GradientBoostingRegressor",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": RANDOM_STATE
        }
    },
    "SVR": {
        "class": "SVR",
        "params": {
            "kernel": "rbf",
            "C": 1.0,
            "epsilon": 0.1
        }
    },
    "LinearRegression": {
        "class": "LinearRegression",
        "params": {"n_jobs": -1}
    },
    "Ridge": {
        "class": "Ridge",
        "params": {"random_state": RANDOM_STATE}
    }
}

# Hyperparameter tuning
HYPERPARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "GradientBoosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }
}

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Evaluation metrics
METRICS = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]