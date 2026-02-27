"""
College Football Predictor using PyTorch and Deep Framework
Replaces TensorFlow-based predictors with PyTorch using core deep module
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Import from deep framework
from core.training import Trainer, EarlyStopping, ModelCheckpoint
from core.utils import set_seed, get_device

# Import custom CFB models
from cfb_models import CFBFeedForwardModel, CFBLSTMModel


class CFBPredictorPyTorch:
    """
    College football predictor using PyTorch and deep framework

    This class provides a similar interface to the TensorFlow-based predictors
    but uses the shared deep learning framework.
    """

    def __init__(self, model_type='feedforward', task='classification'):
        """
        Args:
            model_type: 'feedforward' or 'lstm'
            task: 'classification' (win/loss) or 'regression' (spread)
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.trainer = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.history = None
        self.device = get_device()

    def load_data(self, csv_file='ml_features_v2_2025.csv', min_week=3):
        """
        Load and prepare data for training

        Args:
            csv_file: Path to features CSV
            min_week: Minimum week to include (teams need some games for stats)

        Returns:
            X, y_win, y_spread, df for training
        """
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)

        print(f"Initial data shape: {df.shape}")
        print(f"Weeks available: {df['week'].min()} to {df['week'].max()}")

        # Filter to games where teams have played some games
        df = df[df['week'] >= min_week].copy()
        print(f"After filtering to week >= {min_week}: {df.shape}")

        # Remove rows with missing outcomes
        df = df.dropna(subset=['home_win', 'point_differential'])
        print(f"After removing missing outcomes: {df.shape}")

        # Select features (exclude IDs and outcomes)
        exclude_cols = [
            'game_id', 'season', 'week', 'date',
            'home_team_id', 'away_team_id',
            'home_team', 'away_team',
            'home_score', 'away_score',
            'home_win', 'point_differential',
            'prediction', 'predicted_spread'  # If these exist from previous runs
        ]

        # Get available feature columns
        available_features = [col for col in df.columns if col not in exclude_cols]
        print(f"\nAvailable features: {len(available_features)}")
        print(f"First 10 features: {available_features[:10]}")

        self.feature_columns = available_features

        X = df[self.feature_columns].values
        y_win = df['home_win'].values
        y_spread = df['point_differential'].values

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Win labels shape: {y_win.shape}")
        print(f"Spread labels shape: {y_spread.shape}")
        print(f"Home team wins: {y_win.sum()}/{len(y_win)} ({100*y_win.mean():.1f}%)")

        return X, y_win, y_spread, df

    def build_model(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.3):
        """
        Build neural network model using deep framework

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        print(f"\nBuilding {self.model_type} model for {self.task}...")
        print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Layers: {num_layers}")

        if self.model_type == 'feedforward':
            model = CFBFeedForwardModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=num_layers,
                dropout=dropout,
                task_type=self.task
            )
        elif self.model_type == 'lstm':
            model = CFBLSTMModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=2,
                dropout=dropout,
                task_type=self.task
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model = model.to(self.device)
        print(f"Model built successfully!")
        print(f"Total parameters: {model.count_parameters():,}")
        print(f"Using device: {self.device}")

        return model

    def prepare_data_loaders(self, X, y, validation_split=0.2, batch_size=32):
        """
        Prepare PyTorch data loaders

        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction for validation
            batch_size: Batch size

        Returns:
            train_loader, val_loader
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"\nData prepared:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Batch size: {batch_size}")

        return train_loader, val_loader

    def train(
        self,
        X,
        y,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        patience=15,
        save_path='best_cfb_model.pt',
        verbose=True
    ):
        """
        Train the model using deep framework's Trainer

        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience
            save_path: Path to save best model
            verbose: Verbosity level
        """
        # Build model if not exists
        if self.model is None:
            self.build_model(input_dim=X.shape[1])

        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(
            X, y, validation_split, batch_size
        )

        # Setup optimizer and loss
        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        if self.task == 'classification':
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()

        # Setup callbacks
        callbacks = [
            EarlyStopping(patience=patience, verbose=verbose),
            ModelCheckpoint(filepath=save_path, verbose=verbose)
        ]

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            callbacks=callbacks
        )

        # Define metrics
        def accuracy(y_pred, y_true):
            """Calculate accuracy for classification"""
            pred_labels = (y_pred > 0.5).float()
            return (pred_labels == y_true).float().mean()

        def mae(y_pred, y_true):
            """Calculate MAE for regression"""
            return torch.abs(y_pred - y_true).mean()

        metrics = {'accuracy': accuracy} if self.task == 'classification' else {'mae': mae}

        # Train
        print(f"\n{'='*50}")
        print(f"Training {self.task} model...")
        print(f"{'='*50}\n")

        self.history = self.trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            metrics=metrics,
            verbose=verbose
        )

        print(f"\n{'='*50}")
        print("Training completed!")
        print(f"Best validation loss: {min(self.history['val_loss']):.4f}")
        print(f"{'='*50}\n")

        return self.history

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Feature matrix

        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().flatten()

    def save(self, model_path, scaler_path=None):
        """Save model and scaler"""
        if self.model is None:
            raise ValueError("No model to save!")

        # Save model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

        # Save scaler
        if scaler_path is None:
            scaler_path = model_path.replace('.pt', '_scaler.pkl')

        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
        print(f"Scaler saved to {scaler_path}")

    def load(self, model_path, scaler_path=None):
        """Load model and scaler"""
        # Load scaler
        if scaler_path is None:
            scaler_path = model_path.replace('.pt', '_scaler.pkl')

        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']

        print(f"Scaler loaded from {scaler_path}")

        # Build model with correct input dimension
        input_dim = len(self.feature_columns)
        self.build_model(input_dim=input_dim)

        # Load weights
        self.model.load(model_path)
        print(f"Model loaded from {model_path}")
