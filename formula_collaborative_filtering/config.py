import os
from pathlib import Path
from datetime import datetime

# Base paths
BASE_DIR = Path(__file__).resolve().parent
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
FEATURE_PREFIXES = ["Formula_"]
PROPERTY_PREFIXES = ["Property_"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
MIN_SAMPLES_THRESHOLD = 50

# PCA settings
PCA_VARIANCE_RETAINED = 0.95  # Retain 95% of variance

# Collaborative Filtering settings
NEIGHBORS_COUNT = 5

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / f"formula_cf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Model persistence
MODEL_SAVE_FORMAT = "joblib"  # Options: "joblib", "pickle"
MODEL_VERSION = "v1.0"

# Visualization settings
PLOT_FORMAT = "png"           # Options: "png", "svg", "pdf"
PLOT_DPI = 300
COLOR_PALETTE = "viridis"     # Options: "viridis", "plasma", "inferno", "magma", "cividis"

# Performance optimization
USE_GPU = False               # Set to True if GPU acceleration is available
BATCH_SIZE = 64
N_JOBS = -1                   # Use all available cores (-1)

# Evaluation metrics
EVALUATION_METRICS = ["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
RECOMMENDATION_METRICS = ["precision_at_k", "recall_at_k", "ndcg_at_k"]

# API settings (if applicable)
API_HOST = "localhost"
API_PORT = 8000
API_DEBUG = False

# Export settings
EXPORT_FORMATS = ["csv", "json", "excel"]

# Notification settings (for monitoring)
EMAIL_NOTIFICATIONS = False
SLACK_NOTIFICATIONS = False

# Add any additional project-specific settings below