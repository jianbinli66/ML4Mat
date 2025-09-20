#!/usr/bin/env python3
"""
Material Property Prediction System
Predicts material properties from formula and condition features
"""

import logging
from pathlib import Path
import argparse
# Import project modules
from config.settings import *
from data_loader import DataLoader
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
from utils import setup_logging, save_results, get_timestamp


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Material Property Prediction System")
    parser.add_argument("--properties", nargs="+", help="Specific properties to process")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_THRESHOLD,
                        help="Minimum samples required to process a property")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(LOG_FILE, LOG_LEVEL)
    logger.info("Starting Material Property Prediction System")

    try:
        # Initialize components
        data_loader = DataLoader(DATA_FILE, FEATURE_PREFIXES, PROPERTY_PREFIXES)
        model_trainer = ModelTrainer(MODELS_DIR, MODEL_CONFIGS, HYPERPARAM_GRIDS)
        evaluator = ModelEvaluator()
        visualizer = ResultVisualizer(PLOTS_DIR)

        # Load data
        if not data_loader.load_data():
            raise Exception("Failed to load data")

        # Identify columns
        feature_columns, property_columns = data_loader.identify_columns()
        feature_names = data_loader.get_feature_names()

        # Get property statistics
        property_stats = data_loader.get_property_stats()
        logger.info("Property statistics:")
        for prop, stats in list(property_stats.items())[:10]:  # Show first 10
            logger.info(f"  {prop}: {stats['non_null_count']} samples ({stats['completeness_ratio']:.1%})")

        # Determine which properties to process
        if args.properties:
            properties_to_process = [p for p in args.properties if p in property_columns]
        else:
            properties_to_process = [
                p for p, stats in property_stats.items()
                if stats['non_null_count'] >= args.min_samples
            ]

        logger.info(f"Processing {len(properties_to_process)} properties")

        # Process each property
        all_results = {}
        for property_name in properties_to_process:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing property: {property_name}")
            logger.info(f"{'=' * 60}")

            try:
                # Prepare data
                X_train, X_test, y_train, y_test, scaler = data_loader.prepare_property_data(
                    property_name, TEST_SIZE, RANDOM_STATE
                )

                if X_train is None:
                    logger.warning(f"Skipping {property_name} - no valid data")
                    continue

                # Train models
                training_results = model_trainer.train_models(X_train, y_train, CV_FOLDS)

                # Evaluate models
                comparison = evaluator.compare_models(training_results, X_test, y_test)

                # Get best model
                best_model_name, best_metrics = evaluator.get_best_model(comparison)

                if best_model_name:
                    best_model = training_results[best_model_name]['model']

                    # Hyperparameter tuning if requested
                    if args.tune and best_model_name in HYPERPARAM_GRIDS:
                        logger.info(f"Tuning hyperparameters for {best_model_name}")
                        tuned_model, best_params = model_trainer.hyperparameter_tuning(
                            best_model_name, X_train, y_train,
                            HYPERPARAM_GRIDS[best_model_name], CV_FOLDS
                        )
                        if tuned_model:
                            best_model = tuned_model
                            # Re-evaluate tuned model
                            best_metrics, y_pred = evaluator.evaluate_model(best_model, X_test, y_test)

                    # Cross-validation
                    cv_results = evaluator.cross_validate(best_model, X_train, y_train, CV_FOLDS)

                    # Save best model
                    model_path = model_trainer.save_model(
                        best_model, property_name, best_model_name,
                        best_metrics, feature_names
                    )

                    # Create visualizations
                    _, y_pred = evaluator.evaluate_model(best_model, X_test, y_test)
                    visualizer.plot_model_comparison(comparison, property_name)
                    visualizer.plot_prediction_vs_actual(y_test, y_pred, best_model_name, property_name)
                    visualizer.plot_residuals(y_test, y_pred, best_model_name, property_name)
                    visualizer.plot_feature_importance(best_model, feature_names, best_model_name, property_name)
                    visualizer.plot_shap_summary(best_model, X_train, feature_names, best_model_name, property_name)
                    # visualizer.plot_shap_dependence(best_model, X_train, feature_names, best_model_name, property_name)
                    visualizer.create_shap_analysis_report(best_model, X_train, feature_names, property_name)
                    # Store results
                    all_results[property_name] = {
                        'best_model': best_model_name,
                        'metrics': best_metrics,
                        'cv_results': cv_results,
                        'model_path': str(model_path) if model_path else None,
                        'samples': len(X_train) + len(X_test)
                    }

                    logger.info(
                        f"Completed {property_name}: Best model {best_model_name} with R² {best_metrics['r2']:.3f}")

            except Exception as e:
                logger.error(f"Error processing {property_name}: {e}")
                continue

        # Create summary report
        if all_results:
            summary = visualizer.create_summary_report(all_results)
            save_results(all_results, f"prediction_results_{get_timestamp()}.json")
            logger.info("\nPrediction completed successfully!")
            logger.info(summary)
            logger.info(f"Results saved for {len(all_results)} properties")
        else:
            logger.warning("No properties were successfully processed")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()