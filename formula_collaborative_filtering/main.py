import numpy as np
from data_loader import PropertyDataLoader
from preprocessing import DataPreprocessor
from pca_reduction import PCAReducer
from collaborative_filtering import FormulaCollaborativeFiltering
import config


def main():
    # Load data
    data_loader = PropertyDataLoader(config.DATA_FILE)
    data = data_loader.load_data()

    # Extract formula and property data
    formula_data = data_loader.get_formula_data()
    condition_data = data_loader.get_condition_data()
    property_data = data_loader.get_property_data()
    # Preprocess data
    preprocessor = DataPreprocessor()
    formula_processed = preprocessor.preprocess_formula_data(formula_data)
    condition_processed = preprocessor.preprocess_condition_data(condition_data)

    # Apply PCA for dimensionality reduction
    pca_reducer = PCAReducer(
        n_components=config.PCA_VARIANCE_RETAINED)
    formula_reduced = pca_reducer.fit_transform_formula(formula_processed)
    condition_reduced = pca_reducer.fit_transform_condition(condition_processed)
    # combine reduced formula and condition data
    combined_reduced = np.hstack((formula_reduced, condition_reduced))

    print(f"Original formula dimensions: {formula_processed.shape}")
    print(f"Reduced formula dimensions: {formula_reduced.shape}")
    print(f"Variance explained by formula PCA: {sum(pca_reducer.formula_explained_variance_):.3f}")

    print(f"Original condition dimensions: {condition_processed.shape}")
    print(f"Reduced condition dimensions: {condition_reduced.shape}")
    print(f"Variance explained by property PCA: {sum(pca_reducer.condition_explained_variance_):.3f}")


    # Initialize collaborative filtering
    cf = FormulaCollaborativeFiltering(
        n_neighbors=config.NEIGHBORS_COUNT)
    cf.fit(combined_reduced)

    # Test with a sample formula
    sample_idx = 0    # combine sample formula and condition
    sample_data = np.hstack((formula_reduced[sample_idx], condition_reduced[sample_idx]))
    # Get recommendations, excluding the sample itself
    similar_indices, similar_samples, distances = cf.recommend(sample_data)


    print(f"\nSample formula (index {sample_idx})")
    # print original formula and conditions and properties as key-value pairs where values > 0
    original_sample_formula = formula_data.iloc[sample_idx].to_dict()
    original_sample_condition = condition_data.iloc[sample_idx].to_dict()
    original_sample_property = property_data.iloc[sample_idx].to_dict()
    print("Original Formula:", {k: v for k, v in original_sample_formula.items() if v > 0})
    print("Original Condition:", {k: v for k, v in original_sample_condition.items() if v > 0})
    print("Original Property:", {k: v for k, v in original_sample_property.items() if v > 0})
    print(f"Top {config.NEIGHBORS_COUNT} similar formulas:")
    for i, idx in enumerate(similar_indices):
        print(f"{i + 1}. Index: {idx}, Distance: {distances[i]:.3f}")
        # print original formula and conditions and properties as key-value pairs where values > 0
        original_formula = formula_data.iloc[idx].to_dict()
        original_condition = condition_data.iloc[idx].to_dict()
        original_property = property_data.iloc[idx].to_dict()
        print("   Formula:", {k: v for k, v in original_formula.items() if v > 0})
        print("   Condition:", {k: v for k, v in original_condition.items() if v > 0})
        print("   Property:", {k: v for k, v in original_property.items() if v > 0})




    # Save results
    np.savez(
        config.MODELS_DIR / 'model_data.npz',
        formula_reduced=formula_reduced,
        condition_processed=condition_processed,
        pca_formula_components=pca_reducer.formula_pca.components_,
        pca_property_components=pca_reducer.condition_pca.components_
    )

    print("\nModel training completed and results saved!")


if __name__ == "__main__":
    main()