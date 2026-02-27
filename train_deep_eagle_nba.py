"""
NBA Deep Eagle - Score Prediction Model
Trains models to predict NBA game scores using comprehensive features

Predicts:
- Home team score
- Away team score
- Derived: spread, total points, winner

Key differences from football models:
- Uses date-based splits instead of weeks
- Higher scoring games (avg ~110 pts per team vs ~25 for football)
- More games per season (82 per team vs 17/12)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
from nba_feature_extractor import NBAFeatureExtractor


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


class NBADeepEagleTrainer:
    """Trainer for NBA Deep Eagle models"""

    def __init__(self, season=2024):
        self.season = season
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_data(self, features_path, train_pct=0.75):
        """
        Load features and split into train/test based on date

        Args:
            features_path: Path to CSV with extracted features
            train_pct: Percentage of season for training (time-based split)
        """
        print(f"\n{'='*80}")
        print(f"LOADING DATA - NBA {self.season}-{str(self.season+1)[-2:]}")
        print('='*80)

        df = pd.read_csv(features_path)
        print(f"Loaded {len(df)} games from {features_path}")
        print(f"Feature columns: {len(df.columns)}")

        # Sort by date for proper time-based split
        df = df.sort_values('date').reset_index(drop=True)

        # Identify feature columns (exclude IDs and target variables)
        exclude_cols = [
            'game_id', 'season', 'date', 'games_into_season', 'home_team_id', 'away_team_id',
            'home_score', 'away_score', 'point_spread', 'total_points', 'home_win'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Using {len(feature_cols)} features")

        # Time-based split
        split_idx = int(len(df) * train_pct)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        print(f"\nTime-based split ({train_pct:.0%}/{1-train_pct:.0%}):")
        print(f"  Train: {len(train_df)} games ({train_df['date'].min()} to {train_df['date'].max()})")
        print(f"  Test: {len(test_df)} games ({test_df['date'].min()} to {test_df['date'].max()})")

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
              epochs=100, batch_size=64, learning_rate=0.001, patience=15, max_grad_norm=1.0):
        """Train the model"""
        print(f"\n{'='*80}")
        print(f"TRAINING NBA {self.season}-{str(self.season+1)[-2:]} MODEL")
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
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
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

                # Gradient clipping
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

        # ATS performance (if we have Vegas spread data)
        if hasattr(self, 'test_df') and 'odds_latest_spread' in self.test_df.columns:
            vegas_spread = self.test_df['odds_latest_spread'].values
            # We beat the spread if our predicted winner is correct vs Vegas favorite
            # or if our spread prediction is closer to actual
            pred_covers = pred_spread < vegas_spread
            actual_covers = actual_spread < vegas_spread
            ats_accuracy = np.mean((pred_covers == actual_covers) | (vegas_spread == 0))
        else:
            ats_accuracy = None

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
        if ats_accuracy:
            print(f"  ATS accuracy: {ats_accuracy:.1%}")

        results = {
            'predictions': predictions,
            'actual': y_test,
            'mae': mae,
            'rmse': rmse,
            'home_mae': home_mae,
            'away_mae': away_mae,
            'spread_mae': spread_mae,
            'total_mae': total_mae,
            'winner_accuracy': winner_accuracy,
            'ats_accuracy': ats_accuracy
        }

        return results

    def save(self, model_path, scaler_path=None):
        """Save model and scaler"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sport': 'nba',
            'season': self.season,
            'feature_cols': self.feature_cols
        }, model_path)
        print(f"\nModel saved to: {model_path}")

        if scaler_path is None:
            scaler_path = model_path.replace('.pt', '_scaler.pkl')

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {scaler_path}")


def extract_and_train(season):
    """Extract features and train model for a season"""
    print(f"\n{'#'*80}")
    print(f"# NBA DEEP EAGLE - {season}-{str(season+1)[-2:]} SEASON")
    print(f"{'#'*80}")

    # Step 1: Extract features
    print("\n[1/4] Extracting features...")
    extractor = NBAFeatureExtractor()
    features_df = extractor.extract_season_features(season)

    features_path = f'nba_{season}_deep_eagle_features.csv'
    extractor.save_features(features_df, features_path)
    extractor.close()

    # Step 2: Load and prepare data
    print("\n[2/4] Loading data...")
    trainer = NBADeepEagleTrainer(season=season)
    X_train, y_train, X_test, y_test = trainer.load_data(features_path, train_pct=0.75)

    # Step 3: Build and train model
    print("\n[3/4] Training model...")
    trainer.build_model(input_dim=X_train.shape[1])

    # NBA-specific hyperparameters
    history = trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=150,
        batch_size=64,  # Larger batch size for NBA (more data)
        learning_rate=0.0005,  # Moderate learning rate
        patience=20,
        max_grad_norm=1.0
    )

    # Step 4: Evaluate
    print("\n[4/4] Evaluating...")
    results = trainer.evaluate(X_test, y_test)

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/deep_eagle_nba_{season}.pt'
    trainer.save(model_path)

    return trainer, results


def main():
    """Main training pipeline"""
    os.makedirs('models', exist_ok=True)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'both':
            # Train on both seasons
            print("Training on 2023-24 season...")
            trainer_2023, results_2023 = extract_and_train(2023)

            print("\n\nTraining on 2024-25 season...")
            trainer_2024, results_2024 = extract_and_train(2024)

            print(f"\n{'='*80}")
            print("FINAL SUMMARY")
            print('='*80)
            print(f"\n2023-24 Season:")
            print(f"  Winner Accuracy: {results_2023['winner_accuracy']:.1%}")
            print(f"  Spread MAE: {results_2023['spread_mae']:.2f} points")
            print(f"\n2024-25 Season:")
            print(f"  Winner Accuracy: {results_2024['winner_accuracy']:.1%}")
            print(f"  Spread MAE: {results_2024['spread_mae']:.2f} points")
        else:
            season = int(sys.argv[1])
            extract_and_train(season)
    else:
        # Default: train on 2024-25 season (more recent)
        extract_and_train(2024)


if __name__ == '__main__':
    main()
