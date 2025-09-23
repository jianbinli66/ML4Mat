import numpy as np
from matplotlib import pyplot as plt

from data_loader import PropertyDataLoader
from preprocessing import DataPreprocessor
from pca_reduction import PCAReducer
from collaborative_filtering import FormulaCollaborativeFiltering
import config
def plot_pca_performance(pca_reducer, formula_processed, condition_processed):
    """Visualize PCA performance with multiple plots"""

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Cumulative variance explained for formula PCA
    formula_cumulative_variance = np.cumsum(pca_reducer.formula_explained_variance_)
    ax1.plot(range(1, len(formula_cumulative_variance) + 1), formula_cumulative_variance,
             marker='o', linestyle='-', color='blue')
    ax1.axhline(y=config.PCA_VARIANCE_RETAINED, color='red', linestyle='--',
                label=f'Target variance ({config.PCA_VARIANCE_RETAINED})')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Variance Explained')
    ax1.set_title('Formula PCA: Cumulative Variance Explained')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Cumulative variance explained for condition PCA
    condition_cumulative_variance = np.cumsum(pca_reducer.condition_explained_variance_)
    ax2.plot(range(1, len(condition_cumulative_variance) + 1), condition_cumulative_variance,
             marker='o', linestyle='-', color='green')
    ax2.axhline(y=config.PCA_VARIANCE_RETAINED, color='red', linestyle='--',
                label=f'Target variance ({config.PCA_VARIANCE_RETAINED})')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Condition PCA: Cumulative Variance Explained')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Variance explained per component (formula)
    formula_variance = pca_reducer.formula_explained_variance_
    ax3.bar(range(1, len(formula_variance) + 1), formula_variance, color='lightblue', alpha=0.7)
    ax3.plot(range(1, len(formula_variance) + 1), formula_variance, marker='o', color='blue')
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Variance Explained Ratio')
    ax3.set_title('Formula PCA: Variance per Component')
    ax3.grid(True, alpha=0.3)

    # 4. Variance explained per component (condition)
    condition_variance = pca_reducer.condition_explained_variance_
    ax4.bar(range(1, len(condition_variance) + 1), condition_variance, color='lightgreen', alpha=0.7)
    ax4.plot(range(1, len(condition_variance) + 1), condition_variance, marker='o', color='green')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Variance Explained Ratio')
    ax4.set_title('Condition PCA: Variance per Component')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / 'pca_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("PCA PERFORMANCE ANALYSIS")
    print("=" * 60)

    print(f"\nFORMULA PCA:")
    print(f"Original dimensions: {formula_processed.shape[1]}")
    print(f"Reduced dimensions: {pca_reducer.formula_pca.n_components_}")
    print(f"Total variance explained: {sum(pca_reducer.formula_explained_variance_):.3f}")
    print(
        f"Components needed for {config.PCA_VARIANCE_RETAINED * 100}% variance: {np.where(formula_cumulative_variance >= config.PCA_VARIANCE_RETAINED)[0][0] + 1}")

    print(f"\nCONDITION PCA:")
    print(f"Original dimensions: {condition_processed.shape[1]}")
    print(f"Reduced dimensions: {pca_reducer.condition_pca.n_components_}")
    print(f"Total variance explained: {sum(pca_reducer.condition_explained_variance_):.3f}")
    print(
        f"Components needed for {config.PCA_VARIANCE_RETAINED * 100}% variance: {np.where(condition_cumulative_variance >= config.PCA_VARIANCE_RETAINED)[0][0] + 1}")


def plot_pca_2d_projection(formula_reduced, condition_reduced, property_data, sample_idx, similar_indices):
    """Plot 2D projection of PCA results"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot formula PCA projection (first 2 components)
    if formula_reduced.shape[1] >= 2:
        ax1.scatter(formula_reduced[:, 0], formula_reduced[:, 1], alpha=0.6,
                    c='blue', label='All samples')
        # Highlight sample
        ax1.scatter(formula_reduced[sample_idx, 0], formula_reduced[sample_idx, 1],
                    c='red', s=100, marker='*', label='Query sample')
        # Highlight similar samples
        ax1.scatter(formula_reduced[similar_indices, 0], formula_reduced[similar_indices, 1],
                    c='orange', s=80, marker='s', label='Similar samples')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title('Formula PCA: 2D Projection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot condition PCA projection (first 2 components)
    if condition_reduced.shape[1] >= 2:
        ax2.scatter(condition_reduced[:, 0], condition_reduced[:, 1], alpha=0.6,
                    c='green', label='All samples')
        # Highlight sample
        ax2.scatter(condition_reduced[sample_idx, 0], condition_reduced[sample_idx, 1],
                    c='red', s=100, marker='*', label='Query sample')
        # Highlight similar samples
        ax2.scatter(condition_reduced[similar_indices, 0], condition_reduced[similar_indices, 1],
                    c='orange', s=80, marker='s', label='Similar samples')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('Condition PCA: 2D Projection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / 'pca_2d_projection.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Load data
    data_loader = PropertyDataLoader(config.DATA_FILE)
    data = data_loader.load_data()

    # Extract formula and property data
    formula_data = data_loader.get_formula_data()
    condition_data = data_loader.get_condition_data()
    property_data = data_loader.get_property_data()

    # Fill NaN values with 0 for formula and condition data
    formula_data = formula_data.fillna(0)
    condition_data = condition_data.fillna(0)

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
    # visualize PCA performance
    plot_pca_performance(pca_reducer, formula_processed, condition_processed)
    plot_pca_2d_projection(formula_reduced, condition_reduced, property_data, sample_idx=0, similar_indices=[])

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
    sample_idx = 0  # combine sample formula and condition
    sample_data = np.hstack((formula_reduced[sample_idx], condition_reduced[sample_idx]))
    # Get recommendations, excluding the sample itself
    similar_indices, similar_samples, distances = cf.recommend(sample_data)

    print(f"\nSample formula (index {sample_idx})")
    # print original formula and conditions (properties will be shown after search)
    original_sample_formula = formula_data.iloc[sample_idx].to_dict()
    original_sample_condition = condition_data.iloc[sample_idx].to_dict()
    print("Original Formula:", {k: v for k, v in original_sample_formula.items() if v > 0})
    print("Original Condition:", {k: v for k, v in original_sample_condition.items() if v > 0})

    print(f"Top {config.NEIGHBORS_COUNT} similar formulas:")
    for i, idx in enumerate(similar_indices):
        print(f"{i + 1}. Index: {idx}, Distance: {distances[i]:.3f}")
        # print original formula and conditions
        original_formula = formula_data.iloc[idx].to_dict()
        original_condition = condition_data.iloc[idx].to_dict()
        print("   Formula:", {k: v for k, v in original_formula.items() if v > 0})
        print("   Condition:", {k: v for k, v in original_condition.items() if v > 0})

        # Show property data only after search results
        original_property = property_data.iloc[idx].to_dict()
        print("   Property:", {k: v for k, v in original_property.items() if v > 0})
        print()  # Add empty line for readability

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