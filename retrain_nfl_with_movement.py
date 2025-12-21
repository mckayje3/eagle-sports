"""
Retrain NFL Deep Eagle model with line movement features
Uses feedforward architecture matching nfl_predictor.py
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


class DeepEagleModel(nn.Module):
    """Deep Eagle neural network - matches nfl_predictor.py architecture"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(DeepEagleModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

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


def train_nfl_model(features_path, season=2025):
    """Train NFL model with movement features"""
    print("=" * 80)
    print(f"RETRAINING NFL {season} WITH LINE MOVEMENT FEATURES")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load features
    df = pd.read_csv(features_path)
    print(f"\nLoaded {len(df)} games from {features_path}")

    # Exclude IDs and target columns
    exclude_cols = [
        'game_id', 'season', 'week', 'home_team_id', 'away_team_id',
        'home_score', 'away_score', 'point_spread', 'total_points', 'home_win'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"Using {len(feature_cols)} features")

    # Check for movement features
    movement_cols = [c for c in feature_cols if 'movement' in c.lower()]
    print(f"Movement features: {movement_cols}")

    # Split by week (train on weeks 3-13, test on weeks 14-16)
    train_weeks = list(range(3, 14))
    test_weeks = list(range(14, 17))

    train_df = df[df['week'].isin(train_weeks)]
    test_df = df[df['week'].isin(test_weeks)]

    print(f"\nTrain: weeks {min(train_weeks)}-{max(train_weeks)} -> {len(train_df)} games")
    print(f"Test: weeks {min(test_weeks)}-{max(test_weeks)} -> {len(test_df)} games")

    # Prepare features and targets
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[['home_score', 'away_score']].values
    y_test = test_df[['home_score', 'away_score']].values

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nData shapes: X_train={X_train.shape}, y_train={y_train.shape}")

    # Build model
    model = DeepEagleModel(input_dim=len(feature_cols)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training setup
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    # Train
    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Test Loss':<12} {'Test MAE':<12}")
    print('-' * 50)

    best_test_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(200):
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t).cpu().numpy()
            test_loss = np.mean((test_pred - y_test) ** 2)
            test_mae = np.mean(np.abs(test_pred - y_test))

        avg_train_loss = np.mean(train_losses)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"{epoch+1:<8} {avg_train_loss:<12.4f} {test_loss:<12.4f} {test_mae:<12.4f}")

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).cpu().numpy()

    home_mae = np.mean(np.abs(test_pred[:, 0] - y_test[:, 0]))
    away_mae = np.mean(np.abs(test_pred[:, 1] - y_test[:, 1]))

    pred_spread = test_pred[:, 0] - test_pred[:, 1]
    actual_spread = y_test[:, 0] - y_test[:, 1]
    spread_mae = np.mean(np.abs(pred_spread - actual_spread))

    pred_total = test_pred[:, 0] + test_pred[:, 1]
    actual_total = y_test[:, 0] + y_test[:, 1]
    total_mae = np.mean(np.abs(pred_total - actual_total))

    pred_winner = (test_pred[:, 0] > test_pred[:, 1]).astype(int)
    actual_winner = (y_test[:, 0] > y_test[:, 1]).astype(int)
    winner_accuracy = np.mean(pred_winner == actual_winner)

    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print('='*80)
    print(f"Home Score MAE: {home_mae:.2f} points")
    print(f"Away Score MAE: {away_mae:.2f} points")
    print(f"Spread MAE: {spread_mae:.2f} points")
    print(f"Total MAE: {total_mae:.2f} points")
    print(f"Winner Accuracy: {winner_accuracy:.1%}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/deep_eagle_nfl_{season}.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'sport': 'NFL',
        'season': season,
        'feature_cols': feature_cols
    }, model_path)
    print(f"\nModel saved: {model_path}")

    # Save scaler
    scaler_path = f'models/deep_eagle_nfl_{season}_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {scaler_path}")

    return {
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'winner_accuracy': winner_accuracy,
        'feature_cols': feature_cols
    }


if __name__ == '__main__':
    import sys

    features_path = sys.argv[1] if len(sys.argv) > 1 else 'nfl_2025_deep_eagle_features.csv'
    season = int(sys.argv[2]) if len(sys.argv) > 2 else 2025

    results = train_nfl_model(features_path, season)

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print('='*80)
