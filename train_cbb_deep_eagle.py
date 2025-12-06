"""
CBB Deep Eagle - Score Prediction Model
Trains models to predict College Basketball game scores using comprehensive features

Predicts:
- Home team score
- Away team score
- Derived: spread, total points, winner

Key characteristics:
- Uses week-based date encoding (CBB runs Nov-Apr)
- More games per season than NBA (362 D1 teams)
- Higher variance due to skill disparities
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
from cbb_feature_extractor import CBBFeatureExtractor


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


class CBBDeepEagleTrainer:
    """Trainer for CBB Deep Eagle models"""

    def __init__(self, season=2025):
        self.season = season
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_data(self, features_path, train_pct=0.75):
        """
        Load features and split into train/test based on week

        Args:
            features_path: Path to CSV with extracted features
            train_pct: Percentage of season for training (time-based split)
        """
        print(f"\n{'='*80}")
        print(f"LOADING DATA - CBB {self.season-1}-{str(self.season)[-2:]}")
        print('='*80)

        df = pd.read_csv(features_path)
        print(f"Loaded {len(df)} games from {features_path}")
        print(f"Feature columns: {len(df.columns)}")

        # Sort by week for proper time-based split
        df = df.sort_values('week').reset_index(drop=True)

        # Identify feature columns (exclude IDs and target variables)
        exclude_cols = [
            'game_id', 'season', 'week', 'week_normalized', 'home_team_id', 'away_team_id',
            'home_score', 'away_score', 'point_spread', 'total_points', 'home_win'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Using {len(feature_cols)} features")

        # Time-based split
        split_idx = int(len(df) * train_pct)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        print(f"\nTime-based split ({train_pct:.0%}/{1-train_pct:.0%}):")
        print(f"  Train: {len(train_df)} games (weeks {train_df['week'].min()}-{train_df['week'].max()})")
        print(f"  Test: {len(test_df)} games (weeks {test_df['week'].min()}-{test_df['week'].max()})")

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
              epochs=100, batch_size=128, learning_rate=0.001, patience=15, max_grad_norm=1.0):
        """Train the model"""
        print(f"\n{'='*80}")
        print(f"TRAINING CBB {self.season-1}-{str(self.season)[-2:]} MODEL")
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
        if hasattr(self, 'test_df') and 'odds_spread' in self.test_df.columns:
            vegas_spread = self.test_df['odds_spread'].values
            # Model covers if predicted margin matches actual vs Vegas better
            pred_covers = pred_spread < vegas_spread
            actual_covers = actual_spread < vegas_spread
            ats_accuracy = np.mean((pred_covers == actual_covers) | (np.isnan(vegas_spread)))
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
            'sport': 'cbb',
            'season': self.season,
            'feature_cols': self.feature_cols
        }, model_path)
        print(f"\nModel saved to: {model_path}")

        if scaler_path is None:
            scaler_path = model_path.replace('.pt', '_scaler.pkl')

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {scaler_path}")


def train_from_features(features_path, season):
    """Train model from existing feature file"""
    print(f"\n{'#'*80}")
    print(f"# CBB DEEP EAGLE - {season-1}-{str(season)[-2:]} SEASON")
    print(f"{'#'*80}")

    # Load and prepare data
    print("\n[1/3] Loading data...")
    trainer = CBBDeepEagleTrainer(season=season)
    X_train, y_train, X_test, y_test = trainer.load_data(features_path, train_pct=0.75)

    # Build and train model
    print("\n[2/3] Training model...")
    trainer.build_model(input_dim=X_train.shape[1])

    # CBB-specific hyperparameters
    # Larger batch size due to more games, moderate learning rate
    history = trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=150,
        batch_size=128,  # Larger batch for CBB (lots of games)
        learning_rate=0.0005,
        patience=20,
        max_grad_norm=1.0
    )

    # Evaluate
    print("\n[3/3] Evaluating...")
    results = trainer.evaluate(X_test, y_test)

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/deep_eagle_cbb_{season}.pt'
    trainer.save(model_path)

    return trainer, results


def extract_and_train(season):
    """Extract features and train model for a season"""
    print(f"\n{'#'*80}")
    print(f"# CBB DEEP EAGLE - {season-1}-{str(season)[-2:]} SEASON")
    print(f"{'#'*80}")

    # Step 1: Extract features
    print("\n[1/4] Extracting features...")
    extractor = CBBFeatureExtractor()
    features_df = extractor.extract_season_features(season)

    features_path = f'cbb_{season}_deep_eagle_features.csv'
    extractor.save_features(features_df, features_path)
    extractor.close()

    # Step 2: Load and prepare data
    print("\n[2/4] Loading data...")
    trainer = CBBDeepEagleTrainer(season=season)
    X_train, y_train, X_test, y_test = trainer.load_data(features_path, train_pct=0.75)

    # Step 3: Build and train model
    print("\n[3/4] Training model...")
    trainer.build_model(input_dim=X_train.shape[1])

    # CBB-specific hyperparameters
    history = trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=150,
        batch_size=128,
        learning_rate=0.0005,
        patience=20,
        max_grad_norm=1.0
    )

    # Step 4: Evaluate
    print("\n[4/4] Evaluating...")
    results = trainer.evaluate(X_test, y_test)

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/deep_eagle_cbb_{season}.pt'
    trainer.save(model_path)

    return trainer, results


def main():
    """Main training pipeline"""
    os.makedirs('models', exist_ok=True)

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == 'both':
            # Train on both seasons from existing feature files
            print("Training on 2023-24 season (from existing features)...")
            trainer_2024, results_2024 = train_from_features('cbb_2024_deep_eagle_features.csv', 2024)

            print("\n\nTraining on 2024-25 season (from existing features)...")
            trainer_2025, results_2025 = train_from_features('cbb_2025_deep_eagle_features.csv', 2025)

            print(f"\n{'='*80}")
            print("FINAL SUMMARY")
            print('='*80)
            print(f"\n2023-24 Season:")
            print(f"  Winner Accuracy: {results_2024['winner_accuracy']:.1%}")
            print(f"  Spread MAE: {results_2024['spread_mae']:.2f} points")
            print(f"\n2024-25 Season:")
            print(f"  Winner Accuracy: {results_2025['winner_accuracy']:.1%}")
            print(f"  Spread MAE: {results_2025['spread_mae']:.2f} points")

        elif arg == 'combined':
            # Combine both seasons for training
            print("Combining 2023-24 and 2024-25 seasons...")

            df_2024 = pd.read_csv('cbb_2024_deep_eagle_features.csv')
            df_2025 = pd.read_csv('cbb_2025_deep_eagle_features.csv')

            # Combine
            combined = pd.concat([df_2024, df_2025], ignore_index=True)
            combined_path = 'cbb_combined_deep_eagle_features.csv'
            combined.to_csv(combined_path, index=False)
            print(f"Combined {len(df_2024)} + {len(df_2025)} = {len(combined)} games")

            # Train on combined data
            trainer, results = train_from_features(combined_path, 2025)

            print(f"\n{'='*80}")
            print("COMBINED MODEL RESULTS")
            print('='*80)
            print(f"Winner Accuracy: {results['winner_accuracy']:.1%}")
            print(f"Spread MAE: {results['spread_mae']:.2f} points")
            print(f"Total MAE: {results['total_mae']:.2f} points")

        elif arg.isdigit():
            # Train specific season from existing feature file
            season = int(arg)
            features_path = f'cbb_{season}_deep_eagle_features.csv'
            if os.path.exists(features_path):
                train_from_features(features_path, season)
            else:
                print(f"Feature file not found: {features_path}")
                print("Extracting features and training...")
                extract_and_train(season)
        else:
            print(f"Usage: python train_cbb_deep_eagle.py [2024|2025|both|combined]")

    else:
        # Default: train on current season
        features_path = 'cbb_2025_deep_eagle_features.csv'
        if os.path.exists(features_path):
            train_from_features(features_path, 2025)
        else:
            print("Feature file not found. Extracting features...")
            extract_and_train(2025)


if __name__ == '__main__':
    main()
