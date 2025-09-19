import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from sklearn.metrics import r2_score
import shap

logger = logging.getLogger(__name__)


class ResultVisualizer:
    def __init__(self, plots_dir):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

    def _clip_r2_score(self, r2):
        """Clip R2 score to handle extreme negative values"""
        return max(-1.0, r2)

    def plot_model_comparison(self, comparison_results, property_name, save=True):
        """Create comparison plot of model performances"""
        # Extract R2 scores
        r2_scores = {}
        for model_name, metrics in comparison_results.items():
            if 'r2' in metrics and metrics['r2'] is not None:
                r2_scores[model_name] = self._clip_r2_score(metrics['r2'])

        if not r2_scores:
            logger.warning("No R2 scores available for comparison")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        models = list(r2_scores.keys())
        scores = list(r2_scores.values())

        bars = ax.barh(models, scores, color=sns.color_palette("husl", len(models)))
        ax.set_xlabel('R² Score')
        ax.set_title(f'Model Comparison for {property_name}')
        ax.set_xlim(-1.1, 1.1)  # Extended to show negative values

        # Add vertical line at zero
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{score:.3f}', va='center')

        plt.tight_layout()

        if save:
            filename = f"{property_name}_model_comparison.png"
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved model comparison plot: {filename}")
        else:
            plt.show()

    def plot_prediction_vs_actual(self, y_true, y_pred, model_name, property_name, save=True):
        """Plot predicted vs actual values"""
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(y_true, y_pred, alpha=0.6, s=50)
        lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
        ax.plot(lims, lims, 'r--', alpha=0.8)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name} - {property_name}\nPredicted vs Actual')

        # Add R2 score to plot (clipped)
        r2 = self._clip_r2_score(r2_score(y_true, y_pred))
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        plt.tight_layout()

        if save:
            filename = f"{property_name}_{model_name}_prediction_plot.png"
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved prediction plot: {filename}")
        else:
            plt.show()

    def plot_residuals(self, y_true, y_pred, model_name, property_name, save=True):
        """Plot residuals"""
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.6, s=50)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{model_name} - {property_name}\nResidual Plot')

        # Add residual statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax.text(0.05, 0.95, f'Mean = {mean_residual:.3f}\nStd = {std_residual:.3f}',
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        plt.tight_layout()

        if save:
            filename = f"{property_name}_{model_name}_residuals.png"
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved residuals plot: {filename}")
        else:
            plt.show()

    def plot_feature_importance(self, model, feature_names, model_name, property_name, top_n=15, save=True):
        """Plot feature importance for tree-based models"""
        try:
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                importances = model.named_steps['model'].feature_importances_
            elif hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                logger.warning(f"Model {model_name} doesn't support feature importance")
                return

            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            top_indices = indices[:top_n]

            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(len(top_indices)), importances[top_indices])
            ax.set_yticks(range(len(top_indices)))
            ax.set_yticklabels([feature_names[i] for i in top_indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} - {property_name}\nTop {top_n} Feature Importance')
            ax.invert_yaxis()

            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, importances[top_indices])):
                ax.text(bar.get_width() + 0.001, i, f'{importance:.4f}', va='center')

            plt.tight_layout()

            if save:
                filename = f"{property_name}_{model_name}_feature_importance.png"
                plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved feature importance plot: {filename}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")

    def plot_shap_summary(self, model, X, feature_names, model_name, property_name, save=True):
        """Create SHAP summary plot for model interpretation"""
        try:
            # Extract the actual model from the pipeline if needed
            if hasattr(model, 'named_steps'):
                model_to_explain = model.named_steps['model']
            else:
                model_to_explain = model

            # Create SHAP explainer
            if hasattr(model_to_explain, 'predict_proba'):  # For classification
                explainer = shap.Explainer(model_to_explain, X)
            else:  # For regression
                explainer = shap.Explainer(model_to_explain, X)

            # Calculate SHAP values
            shap_values = explainer(X)

            # Create summary plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            ax.set_title(f'{model_name} - {property_name}\nSHAP Feature Importance')
            plt.tight_layout()

            if save:
                filename = f"{property_name}_{model_name}_shap_summary.png"
                plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved SHAP summary plot: {filename}")
            else:
                plt.show()

            # Create bar plot of mean absolute SHAP values
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
            ax.set_title(f'{model_name} - {property_name}\nMean |SHAP| Values')
            plt.tight_layout()

            if save:
                filename = f"{property_name}_{model_name}_shap_bar.png"
                plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved SHAP bar plot: {filename}")
            else:
                plt.show()

            # Return SHAP values for further analysis
            return shap_values

        except Exception as e:
            logger.error(f"Error creating SHAP plot: {e}")
            return None

    def plot_shap_dependence(self, shap_values, X, feature_names, model_name, property_name,
                             target_feature, interaction_index=None, save=True):
        """Create SHAP dependence plot for a specific feature"""
        try:
            # Find the index of the target feature
            feature_idx = feature_names.index(target_feature) if isinstance(target_feature, str) else target_feature

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.dependence_plot(feature_idx, shap_values.values, X,
                                 feature_names=feature_names,
                                 interaction_index=interaction_index,
                                 show=False)
            ax.set_title(f'{model_name} - {property_name}\nSHAP Dependence: {target_feature}')
            plt.tight_layout()

            if save:
                filename = f"{property_name}_{model_name}_shap_dependence_{target_feature}.png"
                plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved SHAP dependence plot: {filename}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {e}")

    def create_shap_analysis_report(self, shap_values, feature_names, model_name, property_name, save=True):
        """Create a comprehensive SHAP analysis report"""
        try:
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

            # Create DataFrame with feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)

            # Calculate global feature impacts
            global_impacts = pd.DataFrame({
                'feature': feature_names,
                'mean_shap': np.mean(shap_values.values, axis=0),
                'std_shap': np.std(shap_values.values, axis=0)
            })

            # Save to CSV if requested
            if save:
                # Create reports directory if it doesn't exist
                reports_dir = self.plots_dir.parent / "reports"
                reports_dir.mkdir(exist_ok=True)

                # Save importance data
                filename = f"{property_name}_{model_name}_shap_importance.csv"
                importance_df.to_csv(reports_dir / filename, index=False)

                # Save global impacts
                filename = f"{property_name}_{model_name}_shap_global_impacts.csv"
                global_impacts.to_csv(reports_dir / filename, index=False)

                logger.info(f"Saved SHAP analysis reports for {model_name}")

            return importance_df, global_impacts

        except Exception as e:
            logger.error(f"Error creating SHAP analysis report: {e}")
            return None, None

    def create_summary_report(self, all_results, save=True):
        """Create a summary report of all properties"""
        report_data = []

        for property_name, results in all_results.items():
            if 'best_model' in results and 'metrics' in results:
                # Clip R2 score if needed
                r2_score_val = results['metrics'].get('r2', 'N/A')
                if r2_score_val != 'N/A':
                    r2_score_val = self._clip_r2_score(r2_score_val)

                report_data.append({
                    'Property': property_name,
                    'Best Model': results['best_model'],
                    'R2 Score': r2_score_val,
                    'RMSE': results['metrics'].get('rmse', 'N/A'),
                    'MAE': results['metrics'].get('mae', 'N/A'),
                    'Samples': results.get('samples', 'N/A')
                })

        report_df = pd.DataFrame(report_data)

        if save and not report_df.empty:
            # Create reports directory if it doesn't exist
            reports_dir = self.plots_dir.parent / "reports"
            reports_dir.mkdir(exist_ok=True)

            filename = "prediction_summary_report.csv"
            report_df.to_csv(reports_dir / filename, index=False)
            logger.info(f"Saved summary report: {filename}")

        return report_df