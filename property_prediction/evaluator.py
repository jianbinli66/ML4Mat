import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self):
        self.metrics = {
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error,
            'r2': r2_score
        }

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate a single model"""
        try:
            y_pred = model.predict(X_test)

            results = {}
            for metric_name, metric_func in self.metrics.items():
                try:
                    score = metric_func(y_test, y_pred)
                    results[metric_name] = score
                except Exception as e:
                    logger.warning(f"Error calculating {metric_name}: {e}")
                    results[metric_name] = None

            return results, y_pred

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None, None

    def cross_validate(self, model, X, y, cv=5, scoring='r2'):
        """Perform cross-validation"""
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return None

    def compare_models(self, models_results, X_test, y_test):
        """Compare multiple models"""
        comparison = {}

        for model_name, model_data in models_results.items():
            if model_data.get('trained', False):
                metrics, _ = self.evaluate_model(model_data['model'], X_test, y_test)
                comparison[model_name] = metrics
            else:
                comparison[model_name] = {'error': model_data.get('error', 'Unknown error')}

        return comparison

    def get_best_model(self, comparison_results, metric='r2', higher_is_better=True):
        """Get the best model based on specified metric"""
        valid_models = {
            name: metrics for name, metrics in comparison_results.items()
            if metric in metrics and metrics[metric] is not None
        }

        if not valid_models:
            logger.warning(f"No valid models found for metric {metric}")
            return None

        if higher_is_better:
            best_model = max(valid_models.items(), key=lambda x: x[1][metric])
        else:
            best_model = min(valid_models.items(), key=lambda x: x[1][metric])

        return best_model