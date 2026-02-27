"""
Train NFL Deep Eagle Model - Simple Feed-Forward Version
Uses features from deep_eagle_feature_extractor.py directly
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pickle
from pathlib import Path


class NFLNet(nn.Module):
    """Simple feed-forward network for NFL predictions"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output: home_score, away_score
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_and_prepare_data(features_path):
    """Load feature extraction output and prepare for training"""
    print(f"\nLoading features from: {features_path}")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} games")

    # Targets: predict actual scores
    # home_score and away_score are included in the feature extraction for evaluation
    df['target_home'] = df['home_score']
    df['target_away'] = df['away_score']

    return df


def get_feature_columns(df):
    """Select feature columns for training - exclude identifiers and targets"""
    exclude_cols = [
        # Identifiers
        'game_id', 'season', 'week', 'date',
        'home_team_id', 'away_team_id',

        # Actual game results (TARGET LEAKAGE)
        'home_score', 'away_score',
        'point_spread', 'total_points', 'home_win',

        # Our constructed targets
        'target_home', 'target_away',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def train_nfl_model():
    """Main training function"""
    print("=" * 80)
    print("NFL DEEP EAGLE MODEL - SIMPLE NETWORK")
    print("=" * 80)

    # Load data
    df = load_and_prepare_data('nfl_2022_2024_deep_eagle_features.csv')

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nFeatures: {len(feature_cols)} columns")

    # Prepare features and targets
    X = df[feature_cols].values
    y_home = df['target_home'].values
    y_away = df['target_away'].values
    y = np.column_stack([y_home, y_away])

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=21.0)  # Default to ~21 points

    # Train/val split (time-based: use season for splitting)
    train_mask = df['season'].isin([2022, 2023])
    val_mask = df['season'] == 2024

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    print(f"\nData Split:")
    print(f"  Train: {len(X_train)} games (2022-2023)")
    print(f"  Val: {len(X_val)} games (2024)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val)

    # Create model
    input_dim = X_train.shape[1]
    model = NFLNet(input_dim, hidden_dims=[256, 128, 64], dropout=0.3)

    print(f"\nModel Architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: [256, 128, 64]")
    print(f"  Output dim: 2 (home_score, away_score)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    model = model.to(device)
    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)
    X_val_t = X_val_t.to(device)
    y_val_t = y_val_t.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    # Training
    print(f"\n{'='*60}")
    print("TRAINING...")
    print('='*60)

    batch_size = 64
    epochs = 200
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        total_loss = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / (len(indices) // batch_size + 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val_loss - 0.0001:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'epoch': epoch
            }, 'models/deep_eagle_nfl_2025_best.pt')
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.2f}, Val: {val_loss:.2f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    checkpoint = torch.load('models/deep_eagle_nfl_2025_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()

    # Calculate MAE
    home_mae = mean_absolute_error(y_val[:, 0], val_pred[:, 0])
    away_mae = mean_absolute_error(y_val[:, 1], val_pred[:, 1])

    # Calculate spread and total MAE
    actual_spread = y_val[:, 0] - y_val[:, 1]  # home - away
    pred_spread = val_pred[:, 0] - val_pred[:, 1]
    spread_mae = mean_absolute_error(actual_spread, pred_spread)

    actual_total = y_val[:, 0] + y_val[:, 1]
    pred_total = val_pred[:, 0] + val_pred[:, 1]
    total_mae = mean_absolute_error(actual_total, pred_total)

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print('='*60)
    print(f"  Home Score MAE: {home_mae:.2f} points")
    print(f"  Away Score MAE: {away_mae:.2f} points")
    print(f"  Spread MAE: {spread_mae:.2f} points")
    print(f"  Total MAE: {total_mae:.2f} points")
    print(f"\n  Vegas Benchmarks:")
    print(f"    Spread: ~5.5 pts (NFL)")
    print(f"    Total: ~7.0 pts (NFL)")

    if spread_mae < 5.5:
        print(f"\n  Spread model BEATS Vegas benchmark!")
    if total_mae < 7.0:
        print(f"  Total model BEATS Vegas benchmark!")

    # Save final model
    print(f"\nSaving model...")

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': [256, 128, 64],
        'dropout': 0.3,
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'home_mae': home_mae,
        'away_mae': away_mae
    }, 'models/deep_eagle_nfl_2025.pt')

    # Save scaler
    with open('models/deep_eagle_nfl_2025_scaler.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_columns': feature_cols
        }, f)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print('='*60)
    print(f"  Model saved: models/deep_eagle_nfl_2025.pt")
    print(f"  Scaler saved: models/deep_eagle_nfl_2025_scaler.pkl")
    print(f"\n  Training data: 2022-2023 ({len(X_train)} games)")
    print(f"  Validation: 2024 ({len(X_val)} games)")
    print(f"  Features: {len(feature_cols)}")
    print(f"\n  Final Spread MAE: {spread_mae:.2f}")
    print(f"  Final Total MAE: {total_mae:.2f}")


if __name__ == '__main__':
    train_nfl_model()
