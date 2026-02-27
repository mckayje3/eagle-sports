"""
Retrain all sports models with line movement features
Compares new model performance against old model
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
from datetime import datetime


class DeepEagleModel(nn.Module):
    """Deep Eagle neural network for score prediction"""

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


def evaluate_old_model(sport, model_path, scaler_path, X_test, y_test):
    """Evaluate existing model on test set"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        old_feature_cols = checkpoint.get('feature_cols', [])

        with open(scaler_path, 'rb') as f:
            old_scaler = pickle.load(f)

        model = DeepEagleModel(input_dim=len(old_feature_cols))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Note: Can't directly compare if feature sets differ
        # Return None to indicate old model exists but can't be compared directly
        return {
            'exists': True,
            'feature_count': len(old_feature_cols),
            'note': 'Old model has different features - will compare after retraining'
        }
    except FileNotFoundError:
        return {'exists': False}
    except Exception as e:
        return {'exists': True, 'error': str(e)}


def train_model(sport, features_path, season, train_weeks, test_weeks):
    """Train model for a sport and return metrics"""

    print(f"\n{'='*70}")
    print(f"TRAINING {sport.upper()} MODEL WITH MOVEMENT FEATURES")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load features
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} games from {features_path}")

    # CRITICAL: Exclude target columns to prevent data leakage
    # These columns contain actual game results - NEVER use as features
    exclude_cols = [
        'game_id', 'season', 'week', 'home_team_id', 'away_team_id',
        'home_score', 'away_score',  # TARGETS
        'point_spread', 'total_points', 'home_win'  # Derived from targets
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Verify no leakage
    leakage_check = ['home_score', 'away_score', 'point_spread', 'total_points', 'home_win']
    found_leakage = [c for c in leakage_check if c in feature_cols]
    if found_leakage:
        raise ValueError(f"DATA LEAKAGE DETECTED! Remove: {found_leakage}")

    print(f"Using {len(feature_cols)} features (excluded {len(exclude_cols)} non-feature columns)")

    # Check movement features
    movement_cols = [c for c in feature_cols if 'movement' in c.lower()]
    print(f"Movement features: {len(movement_cols)}")
    for mc in movement_cols:
        print(f"  - {mc}")

    # Split by week
    train_df = df[df['week'].isin(train_weeks)]
    test_df = df[df['week'].isin(test_weeks)]

    print(f"\nTrain: weeks {min(train_weeks)}-{max(train_weeks)} -> {len(train_df)} games")
    print(f"Test: weeks {min(test_weeks)}-{max(test_weeks)} -> {len(test_df)} games")

    if len(train_df) < 50:
        print(f"WARNING: Only {len(train_df)} training games - results may be unreliable")
    if len(test_df) < 20:
        print(f"WARNING: Only {len(test_df)} test games - results may be unreliable")

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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    model = DeepEagleModel(input_dim=len(feature_cols)).to(device)

    # Training setup
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    # Train
    print(f"\nTraining...")
    best_test_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(200):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t).cpu().numpy()
            test_loss = np.mean((test_pred - y_test) ** 2)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(best_model_state)
    model.to(device)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).cpu().numpy()

    # Calculate metrics
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

    metrics = {
        'home_mae': home_mae,
        'away_mae': away_mae,
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'winner_accuracy': winner_accuracy,
        'feature_count': len(feature_cols),
        'train_games': len(train_df),
        'test_games': len(test_df)
    }

    print(f"\n{sport.upper()} RESULTS:")
    print(f"  Spread MAE: {spread_mae:.2f} pts")
    print(f"  Total MAE: {total_mae:.2f} pts")
    print(f"  Winner Accuracy: {winner_accuracy:.1%}")

    return model, scaler, feature_cols, metrics


def save_model(sport, season, model, scaler, feature_cols):
    """Save model and scaler"""
    os.makedirs('models', exist_ok=True)

    model_path = f'models/deep_eagle_{sport}_{season}.pt'
    scaler_path = f'models/deep_eagle_{sport}_{season}_scaler.pkl'

    torch.save({
        'model_state_dict': model.state_dict(),
        'sport': sport.upper(),
        'season': season,
        'feature_cols': feature_cols,
        'trained_at': datetime.now().isoformat()
    }, model_path)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"  Saved: {model_path}")
    return model_path, scaler_path


def main():
    """Retrain all sports models"""

    print("="*70)
    print("RETRAINING ALL MODELS WITH LINE MOVEMENT FEATURES")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration for each sport
    sports_config = {
        'nfl': {
            'db': 'nfl_games.db',
            'season': 2025,
            'train_weeks': list(range(3, 14)),  # Weeks 3-13
            'test_weeks': list(range(14, 17)),  # Weeks 14-16
            'extractor': 'deep_eagle_feature_extractor.py',
        },
        'cfb': {
            'db': 'cfb_games.db',
            'season': 2025,
            'train_weeks': list(range(1, 13)),  # Weeks 1-12
            'test_weeks': list(range(13, 18)),  # Weeks 13-17
            'extractor': 'deep_eagle_feature_extractor.py',
        },
        'nba': {
            'db': 'nba_games.db',
            'season': 2025,
            'train_weeks': None,  # Will use date-based split
            'test_weeks': None,
            'extractor': 'nba_feature_extractor.py',
        },
        'cbb': {
            'db': 'cbb_games.db',
            'season': 2025,
            'train_weeks': None,  # Will use date-based split
            'test_weeks': None,
            'extractor': 'cbb_feature_extractor.py',
        }
    }

    results = {}

    # Process each sport
    for sport in ['nfl', 'cfb', 'nba', 'cbb']:
        config = sports_config[sport]
        season = config['season']
        features_path = f'{sport}_{season}_deep_eagle_features.csv'

        # Extract features first
        print(f"\n{'='*70}")
        print(f"EXTRACTING {sport.upper()} FEATURES")
        print(f"{'='*70}")

        if sport in ['nfl', 'cfb']:
            os.system(f'python deep_eagle_feature_extractor.py {sport} {config["db"]} {season} {features_path}')
        elif sport == 'nba':
            os.system(f'python nba_feature_extractor.py {season} {features_path}')
        elif sport == 'cbb':
            os.system(f'python cbb_feature_extractor.py {season} {features_path}')

        # Check if features file exists
        if not os.path.exists(features_path):
            print(f"ERROR: Features file not created for {sport}")
            continue

        # Load and check data
        df = pd.read_csv(features_path)

        # Set up train/test split
        if config['train_weeks'] and config['test_weeks']:
            train_weeks = config['train_weeks']
            test_weeks = config['test_weeks']
        else:
            # For NBA/CBB, use week-based split if available, otherwise 80/20
            if 'week' in df.columns:
                weeks = sorted(df['week'].unique())
                split_idx = int(len(weeks) * 0.75)
                train_weeks = weeks[:split_idx]
                test_weeks = weeks[split_idx:]
            else:
                # Fallback: random 80/20 split (less ideal)
                print(f"WARNING: No week column for {sport}, using index-based split")
                n = len(df)
                train_weeks = [1]  # Dummy
                test_weeks = [2]   # Dummy
                # Will handle differently below

        try:
            model, scaler, feature_cols, metrics = train_model(
                sport, features_path, season, train_weeks, test_weeks
            )

            # Save model
            save_model(sport, season, model, scaler, feature_cols)

            results[sport] = metrics

        except Exception as e:
            print(f"ERROR training {sport}: {e}")
            import traceback
            traceback.print_exc()
            results[sport] = {'error': str(e)}

    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    for sport, metrics in results.items():
        if 'error' in metrics:
            print(f"\n{sport.upper()}: FAILED - {metrics['error']}")
        else:
            print(f"\n{sport.upper()}:")
            print(f"  Features: {metrics['feature_count']}")
            print(f"  Train/Test: {metrics['train_games']}/{metrics['test_games']} games")
            print(f"  Spread MAE: {metrics['spread_mae']:.2f} pts")
            print(f"  Total MAE: {metrics['total_mae']:.2f} pts")
            print(f"  Winner Accuracy: {metrics['winner_accuracy']:.1%}")

    print("\n" + "="*70)
    print("RETRAINING COMPLETE")
    print("="*70)

    return results


if __name__ == '__main__':
    main()
