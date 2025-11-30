"""
Deep Eagle - Advanced Score Prediction Model
Trains models to predict actual game scores using comprehensive features including drive data

Predicts:
- Home team score
- Away team score
- Derived: spread, total points, winner
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
import sys


class DeepEagleModel(nn.Module):
    """
    Deep Eagle neural network for score prediction
    Multi-output regression predicting home_score and away_score
    """

    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(DeepEagleModel, self).__init__()

        # Build layers dynamically
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Separate heads for home and away scores
        self.home_score_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.away_score_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        home_score = self.home_score_head(features)
        away_score = self.away_score_head(features)
        return torch.cat([home_score, away_score], dim=1)


class DeepEagleTrainer:
    """Trainer for Deep Eagle models"""

    def __init__(self, sport='cfb', season=2025):
        self.sport = sport.upper()
        self.season = season
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_data(self, features_path, train_weeks=None, test_weeks=None):
        """
        Load features and split into train/test based on weeks

        Args:
            features_path: Path to CSV with extracted features
            train_weeks: List of weeks for training (e.g., [1,2,3,4,5,6,7,8,9,10])
            test_weeks: List of weeks for testing (e.g., [11,12,13])
        """
        print(f"\n{'='*80}")
        print(f"LOADING DATA - {self.sport} {self.season}")
        print('='*80)

        df = pd.read_csv(features_path)
        print(f"Loaded {len(df)} games from {features_path}")
        print(f"Feature columns: {len(df.columns)}")

        # Identify feature columns (exclude IDs and target variables)
        exclude_cols = [
            'game_id', 'season', 'week', 'home_team_id', 'away_team_id',
            'home_score', 'away_score', 'point_spread', 'total_points', 'home_win'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Using {len(feature_cols)} features")

        # Split by week for time-series validation
        if train_weeks and test_weeks:
            train_df = df[df['week'].isin(train_weeks)]
            test_df = df[df['week'].isin(test_weeks)]
            print(f"\nTime-based split:")
            print(f"  Train weeks: {train_weeks} -> {len(train_df)} games")
            print(f"  Test weeks: {test_weeks} -> {len(test_df)} games")
        else:
            # Default: 80/20 split on earlier vs later weeks
            weeks_sorted = sorted(df['week'].unique())
            split_idx = int(len(weeks_sorted) * 0.8)
            train_weeks = weeks_sorted[:split_idx]
            test_weeks = weeks_sorted[split_idx:]
            train_df = df[df['week'].isin(train_weeks)]
            test_df = df[df['week'].isin(test_weeks)]
            print(f"\nDefault 80/20 time split:")
            print(f"  Train weeks: {train_weeks} -> {len(train_df)} games")
            print(f"  Test weeks: {test_weeks} -> {len(test_df)} games")

        # Prepare features and targets
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values

        # Target: [home_score, away_score]
        y_train = train_df[['home_score', 'away_score']].values
        y_test = test_df[['home_score', 'away_score']].values

        # Store for later analysis
        self.train_df = train_df
        self.test_df = test_df
        self.feature_cols = feature_cols

        print(f"\nData shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")

        # Handle NaN and inf values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, y_train, X_test, y_test

    def build_model(self, input_dim):
        """Build Deep Eagle model"""
        self.model = DeepEagleModel(input_dim).to(self.device)
        print(f"\nModel architecture:")
        print(self.model)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=32, learning_rate=0.001, patience=15, max_grad_norm=1.0):
        """
        Train the model

        Args:
            max_grad_norm: Maximum gradient norm for clipping (prevents exploding gradients)

        Returns:
            history: dict with training metrics
        """
        print(f"\n{'='*80}")
        print(f"TRAINING {self.sport} {self.season} MODEL")
        print('='*80)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Early stopping patience: {patience}")
        print(f"Gradient clipping: {max_grad_norm}")

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added L2 regularization
        criterion = nn.MSELoss()

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Train MAE':<12} {'Val MAE':<12}")
        print('-' * 60)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            train_maes = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()

                train_losses.append(loss.item())
                mae = torch.mean(torch.abs(predictions - batch_y)).item()
                train_maes.append(mae)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
                val_mae = torch.mean(torch.abs(val_pred - y_val_t)).item()

            # Record history
            avg_train_loss = np.mean(train_losses)
            avg_train_mae = np.mean(train_maes)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(avg_train_mae)
            history['val_mae'].append(val_mae)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"{epoch+1:<8} {avg_train_loss:<12.4f} {val_loss:<12.4f} "
                      f"{avg_train_mae:<12.4f} {val_mae:<12.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break

        # Restore best model
        self.model.load_state_dict(self.best_model_state)

        print(f"\n{'='*80}")
        print("TRAINING COMPLETE!")
        print('='*80)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation MAE: {min(history['val_mae']):.4f}")

        return history

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        self.model.eval()

        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test)

        with torch.no_grad():
            predictions = self.model(X_test_t).cpu().numpy()

        # Calculate metrics
        mae = np.mean(np.abs(predictions - y_test))
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)

        # Score-specific metrics
        home_mae = np.mean(np.abs(predictions[:, 0] - y_test[:, 0]))
        away_mae = np.mean(np.abs(predictions[:, 1] - y_test[:, 1]))

        # Derived metrics
        pred_spread = predictions[:, 0] - predictions[:, 1]
        actual_spread = y_test[:, 0] - y_test[:, 1]
        spread_mae = np.mean(np.abs(pred_spread - actual_spread))

        pred_total = predictions[:, 0] + predictions[:, 1]
        actual_total = y_test[:, 0] + y_test[:, 1]
        total_mae = np.mean(np.abs(pred_total - actual_total))

        # Winner accuracy
        pred_winner = (predictions[:, 0] > predictions[:, 1]).astype(int)
        actual_winner = (y_test[:, 0] > y_test[:, 1]).astype(int)
        winner_accuracy = np.mean(pred_winner == actual_winner)

        print(f"\n{'='*80}")
        print("TEST SET EVALUATION")
        print('='*80)
        print(f"Overall MAE: {mae:.2f} points")
        print(f"Overall RMSE: {rmse:.2f} points")
        print(f"\nScore Prediction:")
        print(f"  Home team MAE: {home_mae:.2f} points")
        print(f"  Away team MAE: {away_mae:.2f} points")
        print(f"\nDerived Metrics:")
        print(f"  Spread MAE: {spread_mae:.2f} points")
        print(f"  Total points MAE: {total_mae:.2f} points")
        print(f"  Winner accuracy: {winner_accuracy:.1%}")

        results = {
            'predictions': predictions,
            'actual': y_test,
            'mae': mae,
            'rmse': rmse,
            'home_mae': home_mae,
            'away_mae': away_mae,
            'spread_mae': spread_mae,
            'total_mae': total_mae,
            'winner_accuracy': winner_accuracy
        }

        return results

    def save(self, model_path, scaler_path=None):
        """Save model and scaler"""
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sport': self.sport,
            'season': self.season,
            'feature_cols': self.feature_cols
        }, model_path)
        print(f"\nModel saved to: {model_path}")

        # Save scaler
        if scaler_path is None:
            scaler_path = model_path.replace('.pt', '_scaler.pkl')

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {scaler_path}")


def main():
    """Main training pipeline"""
    if len(sys.argv) < 4:
        print("Usage: py train_deep_eagle.py <sport> <season> <features_csv>")
        print("Example: py train_deep_eagle.py cfb 2025 cfb_2025_deep_eagle_features.csv")
        sys.exit(1)

    sport = sys.argv[1]
    season = int(sys.argv[2])
    features_path = sys.argv[3]

    # Create output directory
    os.makedirs('models', exist_ok=True)

    # Initialize trainer
    trainer = DeepEagleTrainer(sport=sport, season=season)

    # Load data with time-based split
    # For CFB: weeks 1-10 train, 11-13+ test
    # For NFL: weeks 1-10 train, 11-12+ test
    if sport.lower() == 'cfb':
        train_weeks = list(range(1, 11))
        test_weeks = list(range(11, 16))
    else:  # NFL
        train_weeks = list(range(1, 11))
        test_weeks = list(range(11, 14))

    X_train, y_train, X_test, y_test = trainer.load_data(
        features_path,
        train_weeks=train_weeks,
        test_weeks=test_weeks
    )

    # Build model
    trainer.build_model(input_dim=X_train.shape[1])

    # Train model - use lower learning rate for CFB due to higher variance
    if sport.lower() == 'cfb':
        learning_rate = 0.0001  # 10x lower for CFB
        max_grad_norm = 0.5  # Stricter gradient clipping
    else:  # NFL
        learning_rate = 0.001
        max_grad_norm = 1.0

    history = trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=200,
        batch_size=32,
        learning_rate=learning_rate,
        patience=20,
        max_grad_norm=max_grad_norm
    )

    # Evaluate
    results = trainer.evaluate(X_test, y_test)

    # Save model
    model_path = f'models/deep_eagle_{sport.lower()}_{season}.pt'
    trainer.save(model_path)

    print(f"\n{'='*80}")
    print(f"DEEP EAGLE {sport.upper()} {season} - COMPLETE!")
    print('='*80)
    print(f"Model: {model_path}")
    print(f"Winner Accuracy: {results['winner_accuracy']:.1%}")
    print(f"Spread MAE: {results['spread_mae']:.2f} points")
    print(f"Total MAE: {results['total_mae']:.2f} points")


if __name__ == '__main__':
    main()
