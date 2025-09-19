import logging
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, models_dir, model_configs, hyperparam_grids):
        self.models_dir = Path(models_dir)
        self.model_configs = model_configs
        self.hyperparam_grids = hyperparam_grids
        self.models = self._initialize_models()

    def _initialize_models(self):
        """Initialize model instances"""
        model_classes = {
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "SVR": SVR,
            "LinearRegression": LinearRegression,
            "Ridge": Ridge
        }

        models = {}
        for name, config in self.model_configs.items():
            try:
                model_class = model_classes[config["class"]]
                models[name] = model_class(**config["params"])
                logger.debug(f"Initialized model: {name}")
            except KeyError:
                logger.warning(f"Unknown model class: {config['class']}")

        return models

    def train_models(self, X_train, y_train, cv_folds=5):
        """Train all models and return results"""
        results = {}

        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")

                # Create pipeline with scaling for certain models
                if name in ["SVR", "Ridge"]:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                else:
                    pipeline = Pipeline([('model', model)])

                # Train model
                pipeline.fit(X_train, y_train)

                # Store results
                results[name] = {
                    'model': pipeline,
                    'trained': True
                }

                logger.info(f"Completed training {name}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'model': None, 'trained': False, 'error': str(e)}

        return results

    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid, cv_folds=5):
        """Perform hyperparameter tuning for a specific model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None

        logger.info(f"Starting hyperparameter tuning for {model_name}")

        try:
            # Create base model
            base_model = self.models[model_name]

            # Setup grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )

            # Perform grid search
            grid_search.fit(X_train, y_train)

            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

            return grid_search.best_estimator_, grid_search.best_params_

        except Exception as e:
            logger.error(f"Error during hyperparameter tuning for {model_name}: {e}")
            return None, None

    def save_model(self, model, property_name, model_name, metrics, feature_names):
        """Save trained model to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{property_name}_{model_name}_{timestamp}.pkl"
        filepath = self.models_dir / filename

        model_data = {
            'model': model,
            'property_name': property_name,
            'model_name': model_name,
            'metrics': metrics,
            'feature_names': feature_names,
            'timestamp': timestamp,
            'version': '1.0'
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None

    def load_model(self, filepath):
        """Load a saved model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model_data
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None