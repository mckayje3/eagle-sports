"""
Train NFL Deep Eagle Model - V2
Uses features from deep_eagle_feature_extractor.py directly
Trains on 2022-2024 data with new features (rest, bye, form, day of week)
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r'C:\Users\jbeast\documents\coding\deep')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
from pathlib import Path

from core import (
    TimeSeriesDataset,
    TimeSeriesDataLoader,
    LSTMModel,
    Trainer
)
from core.training.callbacks import EarlyStopping, ModelCheckpoint


def load_and_prepare_data(features_path):
    """Load feature extraction output and prepare for training"""
    print(f"\nLoading features from: {features_path}")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} games")

    # Create targets from actual scores (these are in the feature output for evaluation)
    # point_spread in extractor = away_score - home_score
    # We want spread from home team perspective = home_score - away_score = -point_spread
    df['spread_target'] = -df['point_spread']  # Home team spread
    df['total_target'] = df['total_points']

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
        'spread_target', 'total_target',

        # Normalized week (keep for now, but may exclude)
        # 'week_normalized',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def train_nfl_model():
    """Main training function"""
    print("=" * 80)
    print("NFL DEEP EAGLE MODEL TRAINING - V2")
    print("=" * 80)

    # Load data
    df = load_and_prepare_data('nfl_2022_2024_deep_eagle_features.csv')

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nFeatures: {len(feature_cols)} columns")

    # Prepare features and targets
    X = df[feature_cols].values
    y_spread = df['spread_target'].values
    y_total = df['total_target'].values

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    y_spread = np.nan_to_num(y_spread, nan=0.0)
    y_total = np.nan_to_num(y_total, nan=0.0)

    # Train/val split (time-based: use season for splitting)
    # Train on 2022-2023, validate on 2024
    train_mask = df['season'].isin([2022, 2023])
    val_mask = df['season'] == 2024

    X_train, X_val = X[train_mask], X[val_mask]
    y_spread_train, y_spread_val = y_spread[train_mask], y_spread[val_mask]
    y_total_train, y_total_val = y_total[train_mask], y_total[val_mask]

    print(f"\nData Split:")
    print(f"  Train: {len(X_train)} games (2022-2023)")
    print(f"  Val: {len(X_val)} games (2024)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create combined targets: [spread, total]
    y_train = np.column_stack([y_spread_train, y_total_train])
    y_val = np.column_stack([y_spread_val, y_total_val])

    # Model parameters
    sequence_length = 8  # Use fewer games for sequence (NFL has fewer games)
    hidden_dim = 128
    num_layers = 2
    dropout = 0.3

    # Create datasets
    train_dataset = TimeSeriesDataset(
        data=X_train_scaled,
        targets=y_train,
        sequence_length=sequence_length,
        forecast_horizon=1
    )

    val_dataset = TimeSeriesDataset(
        data=X_val_scaled,
        targets=y_val,
        sequence_length=sequence_length,
        forecast_horizon=1
    )

    print(f"\nDatasets:")
    print(f"  Train sequences: {len(train_dataset)}")
    print(f"  Val sequences: {len(val_dataset)}")

    # Create loaders
    batch_size = 32
    train_loader = TimeSeriesDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TimeSeriesDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = X_train.shape[1]
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=2,  # Spread and total
        num_layers=num_layers,
        dropout=dropout
    )

    print(f"\nModel Architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output dim: 2 (spread, total)")
    print(f"  Num layers: {num_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    callbacks = [
        EarlyStopping(patience=20, min_delta=0.0001),
        ModelCheckpoint(
            filepath='models/deep_eagle_nfl_2025_best.pt',
            monitor='val_loss',
            save_best_only=True
        )
    ]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks
    )

    # Train
    print(f"\n{'='*60}")
    print("TRAINING...")
    print('='*60)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=150
    )

    print(f"\nTraining Complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Best val loss: {min(history['val_loss']):.4f}")

    # Evaluate on validation set
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            predictions.extend(pred.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    spread_mae = mean_absolute_error(actuals[:, 0], predictions[:, 0])
    total_mae = mean_absolute_error(actuals[:, 1], predictions[:, 1])

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print('='*60)
    print(f"  Spread MAE: {spread_mae:.2f} points")
    print(f"  Total MAE: {total_mae:.2f} points")
    print(f"\n  Vegas Benchmarks:")
    print(f"    Spread: ~5.5 pts (NFL)")
    print(f"    Total: ~7.0 pts (NFL)")

    if spread_mae < 5.5:
        print(f"\n  Spread model BEATS Vegas benchmark!")
    if total_mae < 7.0:
        print(f"  Total model BEATS Vegas benchmark!")

    # Save model and scaler
    print(f"\nSaving model...")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'output_dim': 2,
        'sequence_length': sequence_length,
        'spread_mae': spread_mae,
        'total_mae': total_mae
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
