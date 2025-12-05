"""
Train Deep Eagle CFB v3 Model
- Uses odds data (opening/latest spread/total) as features
- NO target leak (point_spread, total_points, home_win are NOT used as input features)
- Trains on 2024 and completed 2025 games
- Predicts home_score and away_score
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

from deep_eagle_feature_extractor import DeepEagleFeatureExtractor


class DeepEagleModelV3(nn.Module):
    """Deep Eagle v3 - score prediction model"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(DeepEagleModelV3, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Separate prediction heads for home and away scores
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


def extract_and_prepare_data():
    """Extract features and prepare training data"""
    print("="*80)
    print("EXTRACTING FEATURES FOR CFB V3 MODEL")
    print("="*80)

    extractor = DeepEagleFeatureExtractor('cfb_games.db', 'cfb')

    # Extract 2024 (full season) and 2025 (completed games)
    print("\nExtracting 2024 features...")
    features_2024 = extractor.extract_season_features(2024)

    print("\nExtracting 2025 features...")
    features_2025 = extractor.extract_season_features(2025)

    extractor.close()

    # Combine
    all_features = pd.concat([features_2024, features_2025], ignore_index=True)
    print(f"\nTotal games: {len(all_features)}")

    return all_features


def prepare_training_data(df):
    """Prepare features and targets, excluding leak columns"""

    # TARGET columns - what we're predicting
    target_cols = ['home_score', 'away_score']

    # LEAK columns - these are derived from targets, must NOT be used as input
    leak_cols = ['point_spread', 'total_points', 'home_win']

    # ID/metadata columns - not features
    meta_cols = ['game_id', 'season', 'home_team_id', 'away_team_id']

    # All columns to exclude from features
    exclude_cols = target_cols + leak_cols + meta_cols

    # Get feature columns (everything else)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:2}. {col}")

    # Verify no leak columns are included
    leak_check = [col for col in feature_cols if col in leak_cols]
    if leak_check:
        raise ValueError(f"LEAK DETECTED! These columns should not be features: {leak_check}")

    # Extract X and y
    X = df[feature_cols].copy()
    y = df[target_cols].copy()

    # Fill NaN with 0
    X = X.fillna(0)
    y = y.fillna(0)

    # Replace inf with 0
    X = X.replace([np.inf, -np.inf], 0)

    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"Sample features: {feature_cols[:5]}...")

    return X, y, feature_cols


def train_model(X, y, feature_cols):
    """Train the Deep Eagle v3 model"""
    print("\n" + "="*80)
    print("TRAINING DEEP EAGLE CFB V3 MODEL")
    print("="*80)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val.values)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = DeepEagleModelV3(len(feature_cols))
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None

    epochs = 200
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor.to(device))
            val_loss = criterion(val_predictions, y_val_tensor.to(device)).item()

        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor.to(device)).cpu().numpy()
        y_val_np = y_val.values

        # Calculate metrics
        home_mae = np.mean(np.abs(val_pred[:, 0] - y_val_np[:, 0]))
        away_mae = np.mean(np.abs(val_pred[:, 1] - y_val_np[:, 1]))

        spread_pred = val_pred[:, 0] - val_pred[:, 1]
        spread_actual = y_val_np[:, 0] - y_val_np[:, 1]
        spread_mae = np.mean(np.abs(spread_pred - spread_actual))

        total_pred = val_pred[:, 0] + val_pred[:, 1]
        total_actual = y_val_np[:, 0] + y_val_np[:, 1]
        total_mae = np.mean(np.abs(total_pred - total_actual))

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Home Score MAE: {home_mae:.2f} points")
    print(f"Away Score MAE: {away_mae:.2f} points")
    print(f"Spread MAE: {spread_mae:.2f} points")
    print(f"Total MAE: {total_mae:.2f} points")

    return model, scaler, feature_cols


def save_model(model, scaler, feature_cols, model_path, scaler_path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'input_dim': len(feature_cols),
        'version': 'v3'
    }

    torch.save(checkpoint, model_path)
    print(f"\nModel saved to: {model_path}")

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")


def main():
    # Extract features
    df = extract_and_prepare_data()

    # Prepare training data (excludes leak columns)
    X, y, feature_cols = prepare_training_data(df)

    # Train model
    model, scaler, feature_cols = train_model(X, y, feature_cols)

    # Save model
    save_model(
        model, scaler, feature_cols,
        'models/deep_eagle_cfb_2025_v3.pt',
        'models/deep_eagle_cfb_2025_v3_scaler.pkl'
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
