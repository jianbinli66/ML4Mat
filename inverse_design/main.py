import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import logging
import os
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union
import json
import joblib
from tqdm import tqdm
import warnings

from inverse_design.data_loader import MaterialDataset
from inverse_design.forward_model import ForwardModel
from inverse_design.inverse_model import InvertibleNeuralNetwork

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaterialDesignFramework:
    """Comprehensive framework for material inverse design using INN"""

    def __init__(self, feature_cols: List[str], property_cols: List[str]):
        self.feature_cols = feature_cols
        self.property_cols = property_cols
        self.formula_cols = [col for col in feature_cols if col.startswith('Formula_')]
        self.condition_cols = [col for col in feature_cols if col.startswith('Condition_')]

        # Get indices for formula and condition columns
        self.formula_indices = [i for i, col in enumerate(feature_cols) if col in self.formula_cols]
        self.condition_indices = [i for i, col in enumerate(feature_cols) if col in self.condition_cols]

        self.feature_scaler = StandardScaler()
        self.property_scaler = StandardScaler()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Create results directory
        os.makedirs('results/models', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)
        os.makedirs('results/designs', exist_ok=True)

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for training"""
        logger.info("Preparing data for training")

        # Handle missing values
        df_clean = df.dropna(subset=self.property_cols)

        # Separate features and properties
        features = df_clean[self.feature_cols].values
        properties = df_clean[self.property_cols].values

        # Scale features and properties
        features_scaled = self.feature_scaler.fit_transform(features)
        properties_scaled = self.property_scaler.fit_transform(properties)

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, properties_scaled, test_size=test_size, random_state=42
        )

        # Create datasets and data loaders
        train_dataset = MaterialDataset(X_train, y_train)
        test_dataset = MaterialDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        logger.info(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        return train_loader, test_loader

    def _calculate_inn_loss(self, features: torch.Tensor, properties: torch.Tensor,
                            rec_features: torch.Tensor, rec_properties: torch.Tensor,
                            mse_loss: nn.Module, bce_loss: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate INN loss components"""
        # Split reconstruction into formula and condition components
        rec_formula = rec_features[:, self.formula_indices]
        rec_conditions = rec_features[:, self.condition_indices]
        true_formula = features[:, self.formula_indices]
        true_conditions = features[:, self.condition_indices]

        # Different loss functions for formula and conditions
        formula_loss = mse_loss(rec_formula, true_formula)

        # For conditions, use BCE loss and ensure binary outputs
        # Apply sigmoid to get probabilities, then use straight-through estimator
        condition_probs = torch.sigmoid(rec_conditions)
        condition_output = condition_probs + (torch.round(condition_probs) - condition_probs).detach()
        condition_loss = bce_loss(condition_output, true_conditions)

        # Property reconstruction loss
        property_loss = mse_loss(rec_properties, properties)

        # log likelihood loss
        # nll_loss = -self.inn_model.log_prob(features, properties).mean()

        # Total INN loss with weighted components
        inn_loss = formula_loss + condition_loss + property_loss

        # Store individual loss components
        loss_components = {
            'formula_loss': formula_loss.item(),
            'condition_loss': condition_loss.item(),
            'property_loss': property_loss.item(),
            # 'nll_loss': nll_loss.item(),
            'total_loss': inn_loss.item()
        }

        return inn_loss, loss_components

    def train_models(self, train_loader: DataLoader, test_loader: DataLoader,
                     epochs: int = 100, lr: float = 1e-3, inn_weight: float = 0.5):
        """Train both forward and inverse models jointly"""
        logger.info("Training forward and inverse models jointly")

        input_dim = len(self.feature_cols)
        output_dim = len(self.property_cols)

        # Initialize models
        self.forward_model = ForwardModel(input_dim, output_dim).to(self.device)
        self.inn_model = InvertibleNeuralNetwork(input_dim, output_dim).to(self.device)

        # Optimizers
        forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=lr, weight_decay=1e-5)
        inn_optimizer = optim.Adam(self.inn_model.parameters(), lr=lr, weight_decay=1e-5)

        # Learning rate schedulers
        forward_scheduler = optim.lr_scheduler.ReduceLROnPlateau(forward_optimizer, patience=10, factor=0.5,
                                                                 verbose=True)
        inn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(inn_optimizer, patience=10, factor=0.5, verbose=True)

        # Loss functions
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()  # For binary condition outputs

        # Training history
        forward_losses, inn_losses = [], []
        forward_test_losses, inn_test_losses = [], []

        # Track INN sub-losses
        inn_sub_losses = {
            'formula': [],
            'condition': [],
            'property': [],
            'nll': [],
            'total': []
        }

        for epoch in range(epochs):
            # Training phase
            self.forward_model.train()
            self.inn_model.train()

            forward_train_loss, inn_train_loss = 0, 0
            epoch_inn_sub_losses = {
                'formula': 0,
                'condition': 0,
                'property': 0,
                'nll': 0,
                'total': 0
            }

            for features, properties in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                features = features.to(self.device)
                properties = properties.to(self.device)

                # Train forward model
                forward_optimizer.zero_grad()
                pred_properties = self.forward_model(features)
                forward_loss = mse_loss(pred_properties, properties)
                forward_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), max_norm=1.0)
                forward_optimizer.step()
                forward_train_loss += forward_loss.item()

                # Train INN model
                inn_optimizer.zero_grad()

                # Forward pass (features + properties → latent)
                z, log_det_jacobian = self.inn_model(features, properties)

                # Inverse pass (latent → features + properties)
                rec_features, rec_properties = self.inn_model.inverse(z)

                # Calculate INN loss components
                inn_loss, loss_components = self._calculate_inn_loss(
                    features, properties, rec_features, rec_properties, mse_loss, bce_loss
                )

                inn_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.inn_model.parameters(), max_norm=1.0)
                inn_optimizer.step()
                inn_train_loss += inn_loss.item()

                # Accumulate sub-losses
                epoch_inn_sub_losses['formula'] += loss_components['formula_loss']
                epoch_inn_sub_losses['condition'] += loss_components['condition_loss']
                epoch_inn_sub_losses['property'] += loss_components['property_loss']
                # epoch_inn_sub_losses['nll'] += loss_components['nll_loss']
                epoch_inn_sub_losses['total'] += loss_components['total_loss']

            # Calculate average sub-losses for the epoch
            num_batches = len(train_loader)
            for key in epoch_inn_sub_losses:
                epoch_inn_sub_losses[key] /= num_batches
                inn_sub_losses[key].append(epoch_inn_sub_losses[key])

            # Validation phase
            forward_test_loss, inn_test_loss = self.evaluate_models(test_loader, mse_loss, bce_loss)

            # Record losses
            forward_losses.append(forward_train_loss / len(train_loader))
            inn_losses.append(inn_train_loss / len(train_loader))
            forward_test_losses.append(forward_test_loss)
            inn_test_losses.append(inn_test_loss)

            # Update learning rates
            forward_scheduler.step(forward_test_loss)
            inn_scheduler.step(inn_test_loss)

            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch}: Forward Loss: {forward_losses[-1]:.4f}, INN Loss: {inn_losses[-1]:.4f}')
                logger.info(f'Test - Forward Loss: {forward_test_loss:.4f}, INN Loss: {inn_test_loss:.4f}')
                logger.info(f'INN Sub-losses - Formula: {epoch_inn_sub_losses["formula"]:.4f}, '
                            f'Condition: {epoch_inn_sub_losses["condition"]:.4f}, '
                            f'Property: {epoch_inn_sub_losses["property"]:.4f}, '
                            # f'NLL: {epoch_inn_sub_losses["nll"]:.4f}'
                            )

        # Plot training history
        self._plot_training_history(forward_losses, forward_test_losses, "forward_model")
        self._plot_training_history(inn_losses, inn_test_losses, "inn_model")

        # Plot INN sub-loss history
        self._plot_inn_sub_losses(inn_sub_losses)

        logger.info("Model training completed")
        return forward_losses, inn_losses

    def _plot_inn_sub_losses(self, inn_sub_losses: Dict[str, List[float]]):
        """Plot INN sub-loss trends"""
        plt.figure(figsize=(12, 8))

        epochs = range(len(inn_sub_losses['total']))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, inn_sub_losses['formula'], label='Formula Loss', linestyle='--')
        plt.plot(epochs, inn_sub_losses['condition'], label='Condition Loss', linestyle='--')
        plt.plot(epochs, inn_sub_losses['property'], label='Property Loss', linestyle='--')
        # plt.plot(epochs, inn_sub_losses['nll'], label='NLL Loss', linestyle='--')
        plt.title('INN Sub-Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(epochs, inn_sub_losses['total'], label='Total INN Loss', color='black', linewidth=2)
        plt.title('Total INN Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('results/plots/inn_sub_losses.png', dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate_models(self, test_loader: DataLoader, mse_loss: nn.Module, bce_loss: nn.Module) -> Tuple[float, float]:
        """Evaluate both forward and INN models"""
        self.forward_model.eval()
        self.inn_model.eval()

        forward_loss, inn_loss = 0, 0
        all_true_properties, all_pred_properties = [], []
        all_true_features, all_pred_features = [], []

        with torch.no_grad():
            for features, properties in test_loader:
                features = features.to(self.device)
                properties = properties.to(self.device)

                # Forward model evaluation
                pred_properties = self.forward_model(features)
                forward_loss += mse_loss(pred_properties, properties).item()

                # INN model evaluation
                z, _ = self.inn_model(features, properties)
                rec_features, rec_properties = self.inn_model.inverse(z)

                # Calculate INN loss components
                inn_loss_batch, _ = self._calculate_inn_loss(
                    features, properties, rec_features, rec_properties, mse_loss, bce_loss
                )
                inn_loss += inn_loss_batch.item()

                # Collect data for R² calculation
                all_true_properties.append(properties.cpu().numpy())
                all_pred_properties.append(pred_properties.cpu().numpy())
                all_true_features.append(features.cpu().numpy())
                all_pred_features.append(rec_features.cpu().numpy())

        # Calculate R² scores
        true_properties = np.vstack(all_true_properties)
        pred_properties = np.vstack(all_pred_properties)
        true_features = np.vstack(all_true_features)
        pred_features = np.vstack(all_pred_features)

        forward_r2 = r2_score(true_properties, pred_properties)

        # Calculate R² separately for formula and condition components
        formula_r2 = r2_score(true_features[:, self.formula_indices], pred_features[:, self.formula_indices])

        # For conditions, calculate accuracy instead of R²
        condition_accuracy = np.mean(
            np.round(pred_features[:, self.condition_indices]) == true_features[:, self.condition_indices]
        )

        logger.info(f"Forward model R²: {forward_r2:.4f}")
        logger.info(f"INN model - Formula R²: {formula_r2:.4f}, Condition Accuracy: {condition_accuracy:.4f}")

        # Plot predicted vs actual
        self._plot_predicted_vs_actual(true_properties, pred_properties, self.property_cols, "forward_model")

        # For features, plot formula and conditions separately
        self._plot_predicted_vs_actual(
            true_features[:, self.formula_indices],
            pred_features[:, self.formula_indices],
            [self.feature_cols[i] for i in self.formula_indices][:6],
            "inn_model_formula"
        )

        # Plot condition accuracy
        self._plot_condition_accuracy(
            true_features[:, self.condition_indices],
            pred_features[:, self.condition_indices],
            [self.feature_cols[i] for i in self.condition_indices]
        )

        return forward_loss / len(test_loader), inn_loss / len(test_loader)

    def _plot_training_history(self, train_losses: List[float], test_losses: List[float], model_name: str):
        """Plot training history for a model"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title(f'{model_name.title()} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/plots/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_predicted_vs_actual(self, true_values: np.ndarray, pred_values: np.ndarray,
                                  column_names: List[str], model_type: str):
        """Plot predicted vs actual values for evaluation"""
        n_cols = min(3, len(column_names))
        n_rows = (len(column_names) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (ax, col_name) in enumerate(zip(axes, column_names)):
            if i < len(column_names):
                ax.scatter(true_values[:, i], pred_values[:, i], alpha=0.5)

                # Add perfect prediction line
                min_val = min(true_values[:, i].min(), pred_values[:, i].min())
                max_val = max(true_values[:, i].max(), pred_values[:, i].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')

                ax.set_xlabel('True Values')
                ax.set_ylabel('Predicted Values')

                if "condition" in col_name.lower():
                    # For conditions, show accuracy instead of R²
                    accuracy = np.mean(np.round(pred_values[:, i]) == true_values[:, i])
                    ax.set_title(f'{col_name}\nAccuracy = {accuracy:.4f}')
                else:
                    ax.set_title(f'{col_name}\nR² = {r2_score(true_values[:, i], pred_values[:, i]):.4f}')

                ax.grid(True)
            else:
                ax.set_visible(False)

        plt.suptitle(f'{model_type.title()} Model: Predicted vs True Values', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'results/plots/{model_type}_predicted_vs_true.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_condition_accuracy(self, true_conditions: np.ndarray, pred_conditions: np.ndarray,
                                 condition_names: List[str]):
        """Plot accuracy for each condition"""
        accuracies = []
        for i in range(true_conditions.shape[1]):
            accuracy = np.mean(np.round(pred_conditions[:, i]) == true_conditions[:, i])
            accuracies.append(accuracy)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(condition_names)), accuracies, color='skyblue')
        plt.title('Condition Prediction Accuracy')
        plt.xlabel('Conditions')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(condition_names)), condition_names, rotation=45, ha='right')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{accuracy:.3f}',
                     ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('results/plots/condition_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_designs(self, target_properties: Dict[str, float], num_designs: int = 5) -> pd.DataFrame:
        """Generate material designs for target properties with binary conditions"""
        logger.info(f"Generating {num_designs} designs for target properties: {target_properties}")

        self.inn_model.eval()

        # Prepare target properties
        target_properties_arr = np.array([target_properties[prop] for prop in self.property_cols])
        target_properties_scaled = self.property_scaler.transform([target_properties_arr])[0]
        target_properties_tensor = torch.FloatTensor(target_properties_scaled).to(self.device)
        target_properties_tensor = target_properties_tensor.repeat(num_designs, 1)

        # Sample from prior
        random_features = torch.randn(num_designs, len(self.feature_cols)).to(self.device)

        # Generate designs using INN
        with torch.no_grad():
            # Use INN to generate features from properties
            latent_rep = torch.randn(num_designs, self.inn_model.total_dim).to(self.device)
            latent_rep[:, len(self.feature_cols):] = target_properties_tensor

            generated_features, _ = self.inn_model.inverse(latent_rep)

            # Apply sigmoid and threshold to condition components to make them binary
            if self.condition_indices:
                condition_features = generated_features[:, self.condition_indices]
                condition_probs = torch.sigmoid(condition_features)
                condition_binary = torch.round(condition_probs)
                generated_features = generated_features.clone()
                generated_features[:, self.condition_indices] = condition_binary

        # Convert to original scale
        generated_features = generated_features.cpu().numpy()
        generated_features = self.feature_scaler.inverse_transform(generated_features)

        # Ensure conditions are exactly 0 or 1
        if self.condition_indices:
            generated_features[:, self.condition_indices] = np.clip(
                np.round(generated_features[:, self.condition_indices]), 0, 1
            )

        # Create result DataFrame
        designs = []
        for i in range(num_designs):
            design = {}

            # Add features
            for j, col in enumerate(self.feature_cols):
                design[col] = generated_features[i, j]

            # Predict properties for this design using forward model
            design_features = generated_features[i:i + 1]
            design_features_scaled = self.feature_scaler.transform(design_features)
            design_features_tensor = torch.FloatTensor(design_features_scaled).to(self.device)

            with torch.no_grad():
                predicted_properties = self.forward_model(design_features_tensor)
                predicted_properties = predicted_properties.cpu().numpy()
                predicted_properties = self.property_scaler.inverse_transform(predicted_properties)[0]

            for j, col in enumerate(self.property_cols):
                design[f'predicted_{col}'] = predicted_properties[j]

            # Calculate error
            for j, col in enumerate(self.property_cols):
                design[f'error_{col}'] = abs(predicted_properties[j] - target_properties[col])

            designs.append(design)

        return pd.DataFrame(designs)

    def save_models(self):
        """Save trained models and scalers"""
        torch.save(self.forward_model.state_dict(), 'results/models/forward_model.pth')
        torch.save(self.inn_model.state_dict(), 'results/models/inn_model.pth')

        # Save scalers
        joblib.dump(self.feature_scaler, 'results/models/feature_scaler.pkl')
        joblib.dump(self.property_scaler, 'results/models/property_scaler.pkl')

        # Save column information
        with open('results/models/column_info.json', 'w') as f:
            json.dump({
                'feature_cols': self.feature_cols,
                'property_cols': self.property_cols,
                'formula_cols': self.formula_cols,
                'condition_cols': self.condition_cols,
                'formula_indices': self.formula_indices,
                'condition_indices': self.condition_indices
            }, f)

        logger.info("Models and scalers saved")

    def load_models(self):
        """Load trained models and scalers"""
        # Load column information
        with open('results/models/column_info.json', 'r') as f:
            column_info = json.load(f)
            self.feature_cols = column_info['feature_cols']
            self.property_cols = column_info['property_cols']
            self.formula_cols = column_info.get('formula_cols', [])
            self.condition_cols = column_info.get('condition_cols', [])
            self.formula_indices = column_info.get('formula_indices', [])
            self.condition_indices = column_info.get('condition_indices', [])

        # Load scalers
        self.feature_scaler = joblib.load('results/models/feature_scaler.pkl')
        self.property_scaler = joblib.load('results/models/property_scaler.pkl')

        # Load models
        input_dim = len(self.feature_cols)
        output_dim = len(self.property_cols)

        self.forward_model = ForwardModel(input_dim, output_dim).to(self.device)
        self.forward_model.load_state_dict(torch.load('results/models/forward_model.pth', map_location=self.device))

        self.inn_model = InvertibleNeuralNetwork(input_dim, output_dim).to(self.device)
        self.inn_model.load_state_dict(torch.load('results/models/inn_model.pth', map_location=self.device))

        logger.info("Models and scalers loaded")

# The analyze_dataset and main functions remain the same as in the previous implementation

def analyze_dataset(df: pd.DataFrame, feature_cols: List[str], property_cols: List[str]):
    """Perform exploratory data analysis on the dataset"""
    logger.info("Performing exploratory data analysis")

    # Create plots directory
    os.makedirs('results/plots', exist_ok=True)

    # Property distributions
    fig, axes = plt.subplots(1, len(property_cols), figsize=(5 * len(property_cols), 4))
    if len(property_cols) == 1:
        axes = [axes]

    for i, prop in enumerate(property_cols):
        axes[i].hist(df[prop].dropna(), bins=30, alpha=0.7, color='skyblue')
        axes[i].set_title(f'Distribution of {prop}')
        axes[i].set_xlabel(prop)
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('results/plots/property_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Feature correlations
    formula_cols = [col for col in feature_cols if col.startswith('Formula_')]
    condition_cols = [col for col in feature_cols if col.startswith('Condition_')]

    if formula_cols:
        formula_corr = df[formula_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(formula_corr, annot=False, cmap='coolwarm', center=0)
        plt.title('Formula Features Correlation')
        plt.tight_layout()
        plt.savefig('results/plots/formula_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

    if condition_cols:
        condition_stats = df[condition_cols].sum(axis=0)
        plt.figure(figsize=(10, 6))
        condition_stats.plot(kind='bar', color='lightcoral')
        plt.title('Condition Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/plots/condition_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Feature-property relationships
    for prop in property_cols:
        # Find most correlated features
        correlations = {}
        for feat in feature_cols:
            valid_idx = df[feat].notna() & df[prop].notna()
            if valid_idx.sum() > 10:  # Only calculate if enough data
                correlation = df.loc[valid_idx, feat].corr(df.loc[valid_idx, prop])
                correlations[feat] = correlation

        # Get top 10 features by absolute correlation
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        features, corr_values = zip(*sorted_correlations)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(features)), corr_values,
                       color=['lightcoral' if x < 0 else 'skyblue' for x in corr_values])
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.title(f'Top Features Correlated with {prop}')
        plt.ylabel('Correlation Coefficient')

        # Add value labels on bars
        for bar, value in zip(bars, corr_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{value:.2f}',
                     ha='center', va='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'results/plots/correlation_with_{prop}.png', dpi=300, bbox_inches='tight')
        plt.close()

    logger.info("Exploratory data analysis completed")


def main():
    """Main function to run the material design framework"""
    logger.info("Starting Material Design Framework with INN")

    # Load data
    df = pd.read_csv('data/clean_property_data.csv')

    # Identify feature and property columns
    feature_cols = [col for col in df.columns if col.startswith('Formula_') or col.startswith('Condition_')]
    property_cols = [col for col in df.columns if col.startswith('Property_')]

    # Remove FID if present
    if 'FID' in df.columns:
        df = df.drop(columns=['FID'])

    # Perform EDA
    # analyze_dataset(df, feature_cols, property_cols)

    # Initialize framework
    framework = MaterialDesignFramework(feature_cols, property_cols)

    # Prepare data
    train_loader, test_loader = framework.prepare_data(df,test_size=0.3)

    # Train models
    framework.train_models(train_loader, test_loader, epochs=100)

    # Evaluate models
    framework.evaluate_models(test_loader, nn.MSELoss(), nn.BCEWithLogitsLoss())

    # Save models
    framework.save_models()

    # Generate designs for target properties (using median properties as example)
    median_properties = df[property_cols].median().to_dict()
    designs = framework.generate_designs(median_properties, num_designs=5)

    # Save designs
    designs.to_csv('results/designs/generated_designs.csv', index=False)
    logger.info(f"Generated designs saved to results/designs/generated_designs.csv")

    # Analyze results
    logger.info("Generated designs analysis:")
    for i, design in designs.iterrows():
        logger.info(f"Design {i + 1}:")

        # Show significant formula components (>1%)
        formula_cols = [col for col in feature_cols if col.startswith('Formula_')]
        for col in formula_cols:
            if design[col] > 0.01:
                logger.info(f"  {col}: {design[col]:.4f}")

        # Show conditions (only active ones)
        condition_cols = [col for col in feature_cols if col.startswith('Condition_')]
        for col in condition_cols:
            if design[col] > 0:  # Only show active conditions
                logger.info(f"  {col}: {design[col]:.0f}")

        # Show predicted properties and errors
        for prop in property_cols:
            logger.info(f"  Predicted {prop}: {design[f'predicted_{prop}']:.4f} "
                        f"(Error: {design[f'error_{prop}']:.4f})")

    logger.info("Material Design Framework with INN completed")


if __name__ == "__main__":
    main()