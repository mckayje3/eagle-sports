"""
Comprehensive College Football Score Predictor using Deep Learning

This module predicts actual game scores by combining:
1. Point differential (spread) prediction
2. Total points (over/under) prediction

From these two predictions, we calculate actual scores for both teams.
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
from cfb_models import CFBFeedForwardModel


class CFBScorePredictor:
    """
    Comprehensive score predictor using deep learning

    Trains three models:
    1. Win/Loss classifier (binary classification)
    2. Point differential predictor (regression) - spread
    3. Total points predictor (regression) - over/under

    Uses models 2 and 3 to calculate actual predicted scores.
    """

    def __init__(self):
        """Initialize the score predictor"""
        self.win_model = None
        self.spread_model = None
        self.total_model = None

        self.win_trainer = None
        self.spread_trainer = None
        self.total_trainer = None

        self.scaler = StandardScaler()
        self.feature_columns = []
        self.device = get_device()

        print(f"ScorePredictor initialized - Using device: {self.device}")

    def load_data(self, csv_file='ml_features_v2_2025.csv', min_week=3):
        """
        Load and prepare data for training

        Args:
            csv_file: Path to features CSV
            min_week: Minimum week to include

        Returns:
            X, y_win, y_spread, y_total, df
        """
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)

        print(f"Initial data shape: {df.shape}")
        print(f"Weeks available: {df['week'].min()} to {df['week'].max()}")

        # Filter to games where teams have played some games
        df = df[df['week'] >= min_week].copy()
        print(f"After filtering to week >= {min_week}: {df.shape}")

        # Calculate total points if not already present
        if 'total_points' not in df.columns:
            df['total_points'] = df['home_score'] + df['away_score']
            print("Added 'total_points' column")

        # Remove rows with missing outcomes
        df = df.dropna(subset=['home_win', 'point_differential', 'home_score', 'away_score'])
        print(f"After removing missing outcomes: {df.shape}")

        # Select features (exclude IDs, outcomes, and Vegas odds columns)
        exclude_cols = [
            'game_id', 'season', 'week', 'date',
            'home_team_id', 'away_team_id',
            'home_team', 'away_team',
            'home_score', 'away_score',
            'home_win', 'point_differential', 'total_points',
            'prediction', 'predicted_spread', 'predicted_total',
            'spread', 'total', 'covered_spread'  # Exclude Vegas odds (may have NaN)
        ]

        # Get available feature columns
        available_features = [col for col in df.columns if col not in exclude_cols]
        print(f"\nAvailable features: {len(available_features)}")
        print(f"First 10 features: {available_features[:10]}")

        self.feature_columns = available_features

        X = df[self.feature_columns].values
        y_win = df['home_win'].values
        y_spread = df['point_differential'].values
        y_total = df['total_points'].values

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Win labels shape: {y_win.shape}")
        print(f"Spread labels shape: {y_spread.shape}")
        print(f"Total labels shape: {y_total.shape}")
        print(f"\nTarget statistics:")
        print(f"  Home team wins: {y_win.sum()}/{len(y_win)} ({100*y_win.mean():.1f}%)")
        print(f"  Avg point differential: {y_spread.mean():.2f} (±{y_spread.std():.2f})")
        print(f"  Avg total points: {y_total.mean():.2f} (±{y_total.std():.2f})")
        print(f"  Total points range: {y_total.min():.0f} - {y_total.max():.0f}")

        return X, y_win, y_spread, y_total, df

    def build_models(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.3):
        """
        Build all three neural network models

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        print(f"\nBuilding models...")
        print(f"Architecture: Input={input_dim}, Hidden={hidden_dim}, Layers={num_layers}, Dropout={dropout}")

        # Win/Loss classifier
        print("\n1. Win/Loss Classifier")
        self.win_model = CFBFeedForwardModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            task_type='regression'  # Use regression (no sigmoid) with BCEWithLogitsLoss
        ).to(self.device)
        print(f"   Parameters: {self.win_model.count_parameters():,}")

        # Point differential (spread) predictor
        print("\n2. Point Differential (Spread) Predictor")
        self.spread_model = CFBFeedForwardModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            task_type='regression'
        ).to(self.device)
        print(f"   Parameters: {self.spread_model.count_parameters():,}")

        # Total points predictor
        print("\n3. Total Points (Over/Under) Predictor")
        self.total_model = CFBFeedForwardModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            task_type='regression'
        ).to(self.device)
        print(f"   Parameters: {self.total_model.count_parameters():,}")

        print(f"\nAll models built successfully!")
        print(f"Total parameters across all models: {sum([
            self.win_model.count_parameters(),
            self.spread_model.count_parameters(),
            self.total_model.count_parameters()
        ]):,}")

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

        return train_loader, val_loader

    def train_all_models(
        self,
        X, y_win, y_spread, y_total,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        patience=15,
        save_dir='models',
        verbose=True
    ):
        """
        Train all three models

        Args:
            X: Feature matrix
            y_win: Win/loss labels
            y_spread: Point differential labels
            y_total: Total points labels
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience
            save_dir: Directory to save models
            verbose: Verbosity level

        Returns:
            Dictionary with training histories
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Build models if not exists
        if self.win_model is None:
            self.build_models(input_dim=X.shape[1])

        histories = {}

        # 1. Train Win/Loss Classifier
        print("\n" + "="*80)
        print("TRAINING WIN/LOSS CLASSIFIER")
        print("="*80 + "\n")

        train_loader, val_loader = self.prepare_data_loaders(X, y_win, validation_split, batch_size)

        optimizer = Adam(self.win_model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()  # Includes sigmoid internally
        callbacks = [
            EarlyStopping(patience=patience, verbose=verbose),
            ModelCheckpoint(filepath=f'{save_dir}/win_model.pt', verbose=verbose)
        ]

        self.win_trainer = Trainer(self.win_model, optimizer, criterion, self.device, callbacks)

        def accuracy(y_pred, y_true):
            pred_probs = torch.sigmoid(y_pred)  # Apply sigmoid for accuracy calculation
            pred_labels = (pred_probs > 0.5).float()
            return (pred_labels == y_true).float().mean()

        histories['win'] = self.win_trainer.fit(
            train_loader, val_loader, epochs=epochs, metrics={'accuracy': accuracy}, verbose=verbose
        )

        # 2. Train Point Differential Predictor
        print("\n" + "="*80)
        print("TRAINING POINT DIFFERENTIAL (SPREAD) PREDICTOR")
        print("="*80 + "\n")

        train_loader, val_loader = self.prepare_data_loaders(X, y_spread, validation_split, batch_size)

        optimizer = Adam(self.spread_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        callbacks = [
            EarlyStopping(patience=patience, verbose=verbose),
            ModelCheckpoint(filepath=f'{save_dir}/spread_model.pt', verbose=verbose)
        ]

        self.spread_trainer = Trainer(self.spread_model, optimizer, criterion, self.device, callbacks)

        def mae(y_pred, y_true):
            return torch.abs(y_pred - y_true).mean()

        histories['spread'] = self.spread_trainer.fit(
            train_loader, val_loader, epochs=epochs, metrics={'mae': mae}, verbose=verbose
        )

        # 3. Train Total Points Predictor
        print("\n" + "="*80)
        print("TRAINING TOTAL POINTS (OVER/UNDER) PREDICTOR")
        print("="*80 + "\n")

        train_loader, val_loader = self.prepare_data_loaders(X, y_total, validation_split, batch_size)

        optimizer = Adam(self.total_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        callbacks = [
            EarlyStopping(patience=patience, verbose=verbose),
            ModelCheckpoint(filepath=f'{save_dir}/total_model.pt', verbose=verbose)
        ]

        self.total_trainer = Trainer(self.total_model, optimizer, criterion, self.device, callbacks)

        histories['total'] = self.total_trainer.fit(
            train_loader, val_loader, epochs=epochs, metrics={'mae': mae}, verbose=verbose
        )

        print("\n" + "="*80)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*80)
        print("\nBest Validation Results:")
        print(f"  Win Classifier - Loss: {min(histories['win']['val_loss']):.4f}")
        print(f"  Spread Predictor - MAE: {min([h for h in histories['spread']['val_loss']]):.2f} points")
        print(f"  Total Predictor - MAE: {min([h for h in histories['total']['val_loss']]):.2f} points")
        print("="*80 + "\n")

        return histories

    def predict_scores(self, X):
        """
        Predict actual game scores

        Args:
            X: Feature matrix

        Returns:
            Dictionary with predictions:
            - home_win_prob: Probability of home team winning
            - spread: Predicted point differential (positive = home favored)
            - total: Predicted total points
            - home_score: Predicted home team score
            - away_score: Predicted away team score
        """
        if self.win_model is None or self.spread_model is None or self.total_model is None:
            raise ValueError("Models not trained yet!")

        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Make predictions
        self.win_model.eval()
        self.spread_model.eval()
        self.total_model.eval()

        with torch.no_grad():
            win_logits = self.win_model(X_tensor)
            win_prob = torch.sigmoid(win_logits).cpu().numpy().flatten()  # Apply sigmoid to logits
            spread = self.spread_model(X_tensor).cpu().numpy().flatten()
            total = self.total_model(X_tensor).cpu().numpy().flatten()

        # Calculate actual scores
        # If spread = +7 and total = 55:
        # home_score = (55 + 7) / 2 = 31
        # away_score = (55 - 7) / 2 = 24
        home_score = (total + spread) / 2
        away_score = (total - spread) / 2

        # Round to nearest integer
        home_score = np.round(home_score).astype(int)
        away_score = np.round(away_score).astype(int)

        return {
            'home_win_prob': win_prob,
            'spread': spread,
            'total': total,
            'home_score': home_score,
            'away_score': away_score
        }

    def save(self, save_dir='models'):
        """Save all models and scaler"""
        os.makedirs(save_dir, exist_ok=True)

        # Save models
        self.win_model.save(f'{save_dir}/win_model.pt')
        self.spread_model.save(f'{save_dir}/spread_model.pt')
        self.total_model.save(f'{save_dir}/total_model.pt')

        # Save scaler and feature columns
        with open(f'{save_dir}/score_predictor_data.pkl', 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)

        print(f"\nAll models saved to {save_dir}/")

    def load(self, save_dir='models'):
        """Load all models and scaler"""
        # Load scaler and feature columns
        with open(f'{save_dir}/score_predictor_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']

        # Build models
        input_dim = len(self.feature_columns)
        self.build_models(input_dim=input_dim)

        # Load weights
        self.win_model.load(f'{save_dir}/win_model.pt')
        self.spread_model.load(f'{save_dir}/spread_model.pt')
        self.total_model.load(f'{save_dir}/total_model.pt')

        print(f"All models loaded from {save_dir}/")
