import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def setup_logging(log_file, log_level="INFO"):
    """Setup logging configuration"""
    log_file = Path(log_file)
    log_file.parent.mkdir(exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def save_results(results, filename):
    """Save results to JSON file"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    try:
        with open(results_dir / filename, 'w') as f:
            json.dump(convert_types(results), f, indent=2)
        logging.info(f"Results saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")


def load_results(filename):
    """Load results from JSON file"""
    try:
        with open(Path("results") / filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading results: {e}")
        return None


def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")