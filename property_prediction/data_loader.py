import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, data_path, feature_prefixes, property_prefixes):
        self.data_path = Path(data_path)
        self.feature_prefixes = feature_prefixes
        self.property_prefixes = property_prefixes
        self.data = None
        self.feature_columns = None
        self.property_columns = None

    def load_data(self):
        """Load and validate the dataset"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Dataset shape: {self.data.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def identify_columns(self):
        """Identify feature and property columns"""
        self.feature_columns = [
            col for col in self.data.columns
            if any(col.startswith(prefix) for prefix in self.feature_prefixes)
        ]

        self.property_columns = [
            col for col in self.data.columns
            if any(col.startswith(prefix) for prefix in self.property_prefixes)
        ]

        logger.info(f"Found {len(self.feature_columns)} feature columns")
        logger.info(f"Found {len(self.property_columns)} property columns")

        return self.feature_columns, self.property_columns

    def get_property_stats(self):
        """Get statistics about property data availability"""
        stats = {}
        for prop in self.property_columns:
            non_null = self.data[prop].notna().sum()
            stats[prop] = {
                'non_null_count': non_null,
                'null_count': len(self.data) - non_null,
                'completeness_ratio': non_null / len(self.data)
            }

        # Sort by completeness
        return dict(sorted(stats.items(), key=lambda x: x[1]['completeness_ratio'], reverse=True))

    def prepare_property_data(self, property_name, test_size=0.2, random_state=42):
        """Prepare data for a specific property prediction"""
        logger.info(f"Preparing data for property: {property_name}")

        # Filter data where property is not NaN
        valid_data = self.data.dropna(subset=[property_name])
        X = valid_data[self.feature_columns].dropna()
        y = valid_data.loc[X.index, property_name]

        logger.info(f"Available samples for {property_name}: {len(X)}")

        if len(X) == 0:
            return None, None, None, None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def get_feature_names(self):
        """Get feature column names"""
        return self.feature_columns