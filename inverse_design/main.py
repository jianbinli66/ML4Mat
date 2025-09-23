from datetime import datetime

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
from pathlib import Path

from inverse_design.data_loader import MaterialDataset
from inverse_design.forward_model import ForwardModel
from inverse_design.inverse_model import InvertibleNeuralNetwork

warnings.filterwarnings('ignore')
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

# Set up logging
logger = setup_logging(f'results/logs/material_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log', 'INFO')


class MaterialDesignFramework:
    """Comprehensive framework for material inverse design using INN with condition embeddings"""

    def __init__(self, formula_cols: List[str], condition_cols: List[str], property_cols: List[str]):
        self.formula_cols = formula_cols
        self.condition_cols = condition_cols
        self.property_cols = property_cols

        # Get indices for formula and condition columns
        self.formula_indices = list(range(len(formula_cols)))
        self.condition_indices = list(range(len(condition_cols)))

        self.formula_scaler = StandardScaler()
        self.property_scaler = StandardScaler()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Create results directory
        os.makedirs('results/models', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)
        os.makedirs('results/designs', exist_ok=True)

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for training with condition embeddings"""
        logger.info("Preparing data for training")

        # Handle missing values
        df_clean = df.dropna(subset=self.property_cols)

        # Separate features and properties
        formula_features = df_clean[self.formula_cols].values
        condition_features = df_clean[self.condition_cols].values.astype(int)  # Convert to int for embedding
        properties = df_clean[self.property_cols].values

        # Scale formula features and properties
        formula_features_scaled = self.formula_scaler.fit_transform(formula_features)
        properties_scaled = self.property_scaler.fit_transform(properties)

        # Split into train and test
        X_formula_train, X_formula_test, X_condition_train, X_condition_test, y_train, y_test = train_test_split(
            formula_features_scaled, condition_features, properties_scaled, test_size=test_size, random_state=42
        )

        # Create datasets and data loaders
        train_dataset = MaterialDataset(X_formula_train, X_condition_train, y_train)
        test_dataset = MaterialDataset(X_formula_test, X_condition_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        logger.info(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        return train_loader, test_loader

    def _calculate_inn_loss(self, formula_features: torch.Tensor, condition_features: torch.Tensor,
                            properties: torch.Tensor, rec_formula: torch.Tensor, rec_condition_logits: torch.Tensor,
                            rec_properties: torch.Tensor, mse_loss: nn.Module, bce_loss: nn.Module) -> Tuple[
        torch.Tensor, Dict[str, float]]:
        """Calculate INN loss components with condition embeddings"""
        # Formula reconstruction loss
        formula_loss = mse_loss(rec_formula, formula_features)

        # Condition reconstruction loss (binary cross entropy)
        condition_loss = bce_loss(rec_condition_logits, condition_features.float())

        # Property reconstruction loss
        property_loss = mse_loss(rec_properties, properties)

        # Total INN loss with weighted components
        inn_loss = formula_loss + condition_loss + property_loss

        # Store individual loss components
        loss_components = {
            'formula_loss': formula_loss.item(),
            'condition_loss': condition_loss.item(),
            'property_loss': property_loss.item(),
            'total_loss': inn_loss.item()
        }

        return inn_loss, loss_components

    def train_models(self, train_loader: DataLoader, test_loader: DataLoader,
                     epochs: int = 100, lr: float = 1e-3, inn_weight: float = 0.5):
        """Train both forward and inverse models jointly with condition embeddings"""
        logger.info("Training forward and inverse models jointly")

        formula_dim = len(self.formula_cols)
        condition_dim = len(self.condition_cols)
        property_dim = len(self.property_cols)

        # Initialize models
        self.forward_model = ForwardModel(formula_dim, condition_dim, property_dim).to(self.device)
        self.inn_model = InvertibleNeuralNetwork(formula_dim, condition_dim, property_dim).to(self.device)

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

        # Track INN sub-losses for both train and test
        inn_train_sub_losses = {
            'formula': [],
            'condition': [],
            'property': [],
            'total': []
        }

        inn_test_sub_losses = {
            'formula': [],
            'condition': [],
            'property': [],
            'total': []
        }

        for epoch in range(epochs):
            # Training phase
            self.forward_model.train()
            self.inn_model.train()

            forward_train_loss, inn_train_loss = 0, 0
            epoch_inn_train_sub_losses = {
                'formula': 0,
                'condition': 0,
                'property': 0,
                'total': 0
            }

            for formula_features, condition_features, properties in tqdm(train_loader,
                                                                         desc=f"Epoch {epoch + 1}/{epochs}"):
                formula_features = formula_features.to(self.device)
                condition_features = condition_features.to(self.device)
                properties = properties.to(self.device)

                # Train forward model
                forward_optimizer.zero_grad()
                pred_properties = self.forward_model(formula_features, condition_features)
                forward_loss = mse_loss(pred_properties, properties)
                forward_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), max_norm=1.0)
                forward_optimizer.step()
                forward_train_loss += forward_loss.item()

                # Train INN model
                inn_optimizer.zero_grad()

                # Use detached pred_properties to break the computation graph
                pred_properties_detached = pred_properties.detach()

                # Forward pass (latent)
                random_latent = torch.randn(len(properties), self.inn_model.total_dim).to(self.device)
                latent_rep = random_latent
                latent_rep[:, self.inn_model.formula_dim + self.inn_model.embedding_dim:] = pred_properties_detached

                # Forward pass through INN to get latent representation
                _ = self.inn_model.forward(formula_features, condition_features, properties)

                # Inverse pass (latent → features + properties)
                rec_formula, rec_condition_logits, rec_properties = self.inn_model.inverse(latent_rep)

                # Calculate INN loss components
                inn_loss, loss_components = self._calculate_inn_loss(
                    formula_features, condition_features, properties,
                    rec_formula, rec_condition_logits, rec_properties,
                    mse_loss, bce_loss
                )
                inn_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.inn_model.parameters(), max_norm=1.0)
                inn_optimizer.step()
                inn_train_loss += inn_loss.item()

                # Accumulate sub-losses
                epoch_inn_train_sub_losses['formula'] += loss_components['formula_loss']
                epoch_inn_train_sub_losses['condition'] += loss_components['condition_loss']
                epoch_inn_train_sub_losses['property'] += loss_components['property_loss']
                epoch_inn_train_sub_losses['total'] += loss_components['total_loss']

            # Calculate average train sub-losses for the epoch
            num_batches = len(train_loader)
            for key in epoch_inn_train_sub_losses:
                epoch_inn_train_sub_losses[key] /= num_batches
                inn_train_sub_losses[key].append(epoch_inn_train_sub_losses[key])

            # Validation phase - get test sub-losses
            forward_test_loss, inn_test_loss, epoch_inn_test_sub_losses = self.evaluate_models_with_sub_losses(
                test_loader, mse_loss, bce_loss)

            # Store test sub-losses
            for key in epoch_inn_test_sub_losses:
                inn_test_sub_losses[key].append(epoch_inn_test_sub_losses[key])

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
                logger.info(f'Train INN Sub-losses - Formula: {epoch_inn_train_sub_losses["formula"]:.4f}, '
                            f'Condition: {epoch_inn_train_sub_losses["condition"]:.4f}, '
                            f'Property: {epoch_inn_train_sub_losses["property"]:.4f}')
                logger.info(f'Test INN Sub-losses - Formula: {epoch_inn_test_sub_losses["formula"]:.4f}, '
                            f'Condition: {epoch_inn_test_sub_losses["condition"]:.4f}, '
                            f'Property: {epoch_inn_test_sub_losses["property"]:.4f}')

        # Plot training history
        self._plot_training_history(forward_losses, forward_test_losses, "forward_model")
        self._plot_training_history(inn_losses, inn_test_losses, "inn_model")

        # Plot INN sub-loss history with both train and test
        self._plot_inn_sub_losses(inn_train_sub_losses, inn_test_sub_losses)

        logger.info("Model training completed")
        return forward_losses, inn_losses

    def evaluate_models(self, test_loader: DataLoader, mse_loss: nn.Module, bce_loss: nn.Module) -> Tuple[float, float]:
        """Evaluate both forward and INN models with condition embeddings"""
        self.forward_model.eval()
        self.inn_model.eval()

        forward_loss, inn_loss = 0, 0

        with torch.no_grad():
            for formula_features, condition_features, properties in test_loader:
                formula_features = formula_features.to(self.device)
                condition_features = condition_features.to(self.device)
                properties = properties.to(self.device)

                # Forward model evaluation
                pred_properties = self.forward_model(formula_features, condition_features)
                forward_loss += mse_loss(pred_properties, properties).item()

                # Sample from prior
                random_latent = torch.randn(len(properties), self.inn_model.total_dim).to(self.device)
                latent_rep = random_latent
                latent_rep[:,
                self.inn_model.formula_dim + self.inn_model.embedding_dim:] = pred_properties  # No detach needed in eval

                rec_formula, rec_condition_logits, rec_properties = self.inn_model.inverse(latent_rep)

                # Calculate INN loss components
                inn_loss_batch, _ = self._calculate_inn_loss(
                    formula_features, condition_features, properties,
                    rec_formula, rec_condition_logits, rec_properties,
                    mse_loss, bce_loss
                )
                inn_loss += inn_loss_batch.item()

        return forward_loss / len(test_loader), inn_loss / len(test_loader)

    def evaluate_models_with_sub_losses(self, test_loader: DataLoader, mse_loss: nn.Module, bce_loss: nn.Module) -> \
    Tuple[float, float, Dict[str, float]]:
        """Evaluate both forward and INN models with condition embeddings and return sub-losses"""
        self.forward_model.eval()
        self.inn_model.eval()

        forward_loss, inn_loss = 0, 0
        inn_sub_losses = {
            'formula': 0,
            'condition': 0,
            'property': 0,
            'total': 0
        }

        with torch.no_grad():
            for formula_features, condition_features, properties in test_loader:
                formula_features = formula_features.to(self.device)
                condition_features = condition_features.to(self.device)
                properties = properties.to(self.device)

                # Forward model evaluation
                pred_properties = self.forward_model(formula_features, condition_features)
                forward_loss += mse_loss(pred_properties, properties).item()

                # Sample from prior
                random_latent = torch.randn(len(properties), self.inn_model.total_dim).to(self.device)
                latent_rep = random_latent
                latent_rep[:, self.inn_model.formula_dim + self.inn_model.embedding_dim:] = pred_properties

                rec_formula, rec_condition_logits, rec_properties = self.inn_model.inverse(latent_rep)

                # Calculate INN loss components
                inn_loss_batch, loss_components = self._calculate_inn_loss(
                    formula_features, condition_features, properties,
                    rec_formula, rec_condition_logits, rec_properties,
                    mse_loss, bce_loss
                )
                inn_loss += inn_loss_batch.item()

                # Accumulate sub-losses
                inn_sub_losses['formula'] += loss_components['formula_loss']
                inn_sub_losses['condition'] += loss_components['condition_loss']
                inn_sub_losses['property'] += loss_components['property_loss']
                inn_sub_losses['total'] += loss_components['total_loss']

        # Average the sub-losses
        num_batches = len(test_loader)
        for key in inn_sub_losses:
            inn_sub_losses[key] /= num_batches

        return forward_loss / len(test_loader), inn_loss / len(test_loader), inn_sub_losses

    def _plot_inn_sub_losses(self, train_sub_losses: Dict[str, List[float]], test_sub_losses: Dict[str, List[float]]):
        """Plot INN sub-loss trends for both train and test sets"""
        plt.figure(figsize=(15, 10))

        epochs = range(len(train_sub_losses['total']))

        # Plot individual sub-loss components
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_sub_losses['formula'], label='Train Formula Loss', linestyle='-', linewidth=2)
        plt.plot(epochs, test_sub_losses['formula'], label='Test Formula Loss', linestyle='--', linewidth=2)
        plt.title('Formula Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_sub_losses['condition'], label='Train Condition Loss', linestyle='-', linewidth=2)
        plt.plot(epochs, test_sub_losses['condition'], label='Test Condition Loss', linestyle='--', linewidth=2)
        plt.title('Condition Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_sub_losses['property'], label='Train Property Loss', linestyle='-', linewidth=2)
        plt.plot(epochs, test_sub_losses['property'], label='Test Property Loss', linestyle='--', linewidth=2)
        plt.title('Property Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(epochs, train_sub_losses['total'], label='Train Total INN Loss', linestyle='-', linewidth=2,
                 color='black')
        plt.plot(epochs, test_sub_losses['total'], label='Test Total INN Loss', linestyle='--', linewidth=2,
                 color='red')
        plt.title('Total INN Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('results/plots/inn_sub_losses_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Also create a separate plot with all components together for better comparison
        plt.figure(figsize=(12, 8))

        # Plot all train losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_sub_losses['formula'], label='Formula Loss', linestyle='-')
        plt.plot(epochs, train_sub_losses['condition'], label='Condition Loss', linestyle='-')
        plt.plot(epochs, train_sub_losses['property'], label='Property Loss', linestyle='-')
        plt.plot(epochs, train_sub_losses['total'], label='Total Loss', linestyle='-', linewidth=2, color='black')
        plt.title('Train INN Sub-Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot all test losses
        plt.subplot(1, 2, 2)
        plt.plot(epochs, test_sub_losses['formula'], label='Formula Loss', linestyle='-')
        plt.plot(epochs, test_sub_losses['condition'], label='Condition Loss', linestyle='-')
        plt.plot(epochs, test_sub_losses['property'], label='Property Loss', linestyle='-')
        plt.plot(epochs, test_sub_losses['total'], label='Total Loss', linestyle='-', linewidth=2, color='black')
        plt.title('Test INN Sub-Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('results/plots/inn_sub_losses_separate.png', dpi=300, bbox_inches='tight')
        plt.close()
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
        """Generate material designs for target properties with condition embeddings"""
        logger.info(f"Generating {num_designs} designs for target properties: {target_properties}")

        self.inn_model.eval()

        # Prepare target properties
        target_properties_arr = np.array([target_properties[prop] for prop in self.property_cols])
        target_properties_scaled = self.property_scaler.transform([target_properties_arr])[0]
        target_properties_tensor = torch.FloatTensor(target_properties_scaled).to(self.device)
        target_properties_tensor = target_properties_tensor.repeat(num_designs, 1)

        # Sample from prior
        random_latent = torch.randn(num_designs, self.inn_model.total_dim).to(self.device)

        # Generate designs using INN
        with torch.no_grad():
            # Use INN to generate features from properties
            latent_rep = random_latent
            latent_rep[:, self.inn_model.formula_dim + self.inn_model.embedding_dim:] = target_properties_tensor

            generated_formula, generated_condition_logits, _ = self.inn_model.inverse(latent_rep)

            # Apply sigmoid and threshold to condition components to make them binary
            condition_probs = torch.sigmoid(generated_condition_logits)
            # choose the max probability as 1, others as 0
            condition_binary = torch.zeros_like(condition_probs)
            max_indices = torch.argmax(condition_probs, dim=1)
            condition_binary[torch.arange(num_designs), max_indices] = 1


            # condition_binary = torch.round(condition_probs)

        # Convert to original scale
        generated_formula = generated_formula.cpu().numpy()
        generated_formula = self.formula_scaler.inverse_transform(generated_formula)
        # all formula components should be non-negative
        generated_formula[generated_formula < 0] = 0
        condition_binary = condition_binary.cpu().numpy()

        # Create result DataFrame
        designs = []
        for i in range(num_designs):
            design = {}

            # Add formula features
            for j, col in enumerate(self.formula_cols):
                design[col] = generated_formula[i, j]

            # Add condition features
            for j, col in enumerate(self.condition_cols):
                design[col] = condition_binary[i, j]

            # Predict properties for this design using forward model
            design_formula = generated_formula[i:i + 1]
            design_condition = condition_binary[i:i + 1]

            design_formula_scaled = self.formula_scaler.transform(design_formula)
            design_formula_tensor = torch.FloatTensor(design_formula_scaled).to(self.device)
            design_condition_tensor = torch.LongTensor(design_condition).to(self.device)

            with torch.no_grad():
                predicted_properties = self.forward_model(design_formula_tensor, design_condition_tensor)
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
        joblib.dump(self.formula_scaler, 'results/models/formula_scaler.pkl')
        joblib.dump(self.property_scaler, 'results/models/property_scaler.pkl')

        # Save column information
        with open('results/models/column_info.json', 'w') as f:
            json.dump({
                'formula_cols': self.formula_cols,
                'condition_cols': self.condition_cols,
                'property_cols': self.property_cols,
                'formula_indices': self.formula_indices,
                'condition_indices': self.condition_indices
            }, f)

        logger.info("Models and scalers saved")

    def load_models(self):
        """Load trained models and scalers"""
        # Load column information
        with open('results/models/column_info.json', 'r') as f:
            column_info = json.load(f)
            self.formula_cols = column_info['formula_cols']
            self.condition_cols = column_info['condition_cols']
            self.property_cols = column_info['property_cols']
            self.formula_indices = column_info.get('formula_indices', [])
            self.condition_indices = column_info.get('condition_indices', [])

        # Load scalers
        self.formula_scaler = joblib.load('results/models/formula_scaler.pkl')
        self.property_scaler = joblib.load('results/models/property_scaler.pkl')

        # Load models
        formula_dim = len(self.formula_cols)
        condition_dim = len(self.condition_cols)
        property_dim = len(self.property_cols)

        self.forward_model = ForwardModel(formula_dim, condition_dim, property_dim).to(self.device)
        self.forward_model.load_state_dict(torch.load('results/models/forward_model.pth', map_location=self.device))

        self.inn_model = InvertibleNeuralNetwork(formula_dim, condition_dim, property_dim).to(self.device)
        self.inn_model.load_state_dict(torch.load('results/models/inn_model.pth', map_location=self.device))

        logger.info("Models and scalers loaded")
# The analyze_dataset and main functions remain the same as in the previous implementation

def main():
    """Main function to run the material design framework"""
    logger.info("Starting Material Design Framework with INN")

    # Load data
    df = pd.read_csv('data/clean_property_data.csv')

    # Identify feature and property columns
    # feature_cols = [col for col in df.columns if col.startswith('Formula_') or col.startswith('Condition_')]
    formula_cols = [col for col in df.columns if col.startswith('Formula_')]
    condition_cols = [col for col in df.columns if col.startswith('Condition_')]
    property_cols = [col for col in df.columns if col.startswith('Property_')]

    # Remove FID if present
    if 'FID' in df.columns:
        df = df.drop(columns=['FID'])

    # Initialize framework
    framework = MaterialDesignFramework(formula_cols, condition_cols,property_cols)

    # Prepare data
    train_loader, test_loader = framework.prepare_data(df,test_size=0.3)

    # Train models
    framework.train_models(train_loader, test_loader, epochs=300)

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
        formula_cols = [col for col in formula_cols]
        for col in formula_cols:
            if design[col] > 0.01:
                logger.info(f"  {col}: {design[col]:.4f}")

        # Show conditions (only active ones)
        condition_cols = [col for col in condition_cols]
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