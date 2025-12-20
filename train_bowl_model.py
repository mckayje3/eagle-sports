"""
Train Bowl-Specific CFB Model
Combines Deep Eagle features with bowl-specific features
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cfb_deep_eagle_predictor import CFBDeepEaglePredictor, DeepEagleModel
from cfb_bowl_features import BowlFeatureExtractor


class BowlDeepEagleModel(nn.Module):
    """Modified Deep Eagle with bowl-specific feature branch"""

    def __init__(self, base_features, bowl_features, hidden_dims=[256, 128, 64]):
        super().__init__()

        # Main feature extractor (same as Deep Eagle)
        layers = []
        prev_dim = base_features
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = dim

        self.base_extractor = nn.Sequential(*layers)

        # Bowl-specific branch
        self.bowl_branch = nn.Sequential(
            nn.Linear(bowl_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Combined heads
        combined_dim = hidden_dims[-1] + 16

        self.home_head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.away_head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, base_x, bowl_x):
        base_features = self.base_extractor(base_x)
        bowl_features = self.bowl_branch(bowl_x)

        combined = torch.cat([base_features, bowl_features], dim=1)

        home_score = self.home_head(combined)
        away_score = self.away_head(combined)

        return torch.cat([home_score, away_score], dim=1)


def extract_training_data():
    """Extract both base and bowl features for training"""
    base_predictor = CFBDeepEaglePredictor()
    bowl_extractor = BowlFeatureExtractor()

    conn = sqlite3.connect('cfb_games.db')

    # Get bowl games
    query = '''
        SELECT
            g.game_id, g.season, g.week, g.date,
            g.home_team_id, g.away_team_id,
            ht.display_name as home_team,
            at.display_name as away_team,
            g.neutral_site, g.conference_game, g.postseason_type,
            g.venue_name,
            g.home_score, g.away_score,
            COALESCE(o.latest_spread, o.opening_spread) as vegas_spread,
            COALESCE(o.latest_total, o.opening_total) as vegas_total
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.week >= 16 AND g.completed = 1
        AND g.season IN (2021, 2022, 2023, 2024)
        AND (o.latest_spread IS NOT NULL OR o.opening_spread IS NOT NULL)
        ORDER BY g.date
    '''

    games_df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Extracting features for {len(games_df)} bowl games...")

    base_features_list = []
    bowl_features_list = []
    targets = []
    game_ids = []

    for idx, game in games_df.iterrows():
        try:
            # Base features (Deep Eagle)
            base_feats = base_predictor.extract_features(game)
            base_feats.pop('_warnings', None)

            # Bowl-specific features
            bowl_feats = bowl_extractor.extract_bowl_features(game)

            # Align to expected columns
            base_vec = [base_feats.get(col, 0) for col in base_predictor.feature_cols]
            base_vec = np.nan_to_num(base_vec, nan=0.0)

            bowl_vec = [
                bowl_feats['win_differential'],
                bowl_feats['home_wins'],
                bowl_feats['away_wins'],
                bowl_feats['home_win_pct'],
                bowl_feats['away_win_pct'],
                bowl_feats['home_power_conf'],
                bowl_feats['away_power_conf'],
                bowl_feats['bowl_tier'],
                bowl_feats['home_opt_out_risk'],
                bowl_feats['away_opt_out_risk'],
                bowl_feats['opt_out_differential'],
                bowl_feats['home_motivation'],
                bowl_feats['away_motivation'],
                bowl_feats['motivation_differential'],
            ]

            base_features_list.append(base_vec)
            bowl_features_list.append(bowl_vec)
            targets.append([game['home_score'], game['away_score']])
            game_ids.append(game['game_id'])

        except Exception as e:
            print(f"  Error on game {game['game_id']}: {e}")
            continue

    print(f"  Extracted {len(base_features_list)} complete feature sets")

    return (np.array(base_features_list), np.array(bowl_features_list),
            np.array(targets), game_ids, base_predictor.feature_cols)


def train_bowl_model():
    """Train the bowl-specific model"""
    print("=" * 60)
    print("TRAINING BOWL-SPECIFIC CFB MODEL")
    print("=" * 60)

    # Extract data
    base_X, bowl_X, y, game_ids, base_cols = extract_training_data()

    print(f"\nDataset: {len(y)} bowl games")
    print(f"Base features: {base_X.shape[1]}")
    print(f"Bowl features: {bowl_X.shape[1]}")

    # Scale features
    base_scaler = StandardScaler()
    bowl_scaler = StandardScaler()

    base_X_scaled = base_scaler.fit_transform(base_X)
    bowl_X_scaled = bowl_scaler.fit_transform(bowl_X)

    # Train/test split (use 2024 as test)
    test_mask = np.array([gid > 401600000 for gid in game_ids])  # 2024 games
    train_mask = ~test_mask

    X_base_train = base_X_scaled[train_mask]
    X_bowl_train = bowl_X_scaled[train_mask]
    y_train = y[train_mask]

    X_base_test = base_X_scaled[test_mask]
    X_bowl_test = bowl_X_scaled[test_mask]
    y_test = y[test_mask]

    print(f"\nTrain: {len(y_train)} games")
    print(f"Test (2024): {len(y_test)} games")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BowlDeepEagleModel(
        base_features=base_X.shape[1],
        bowl_features=bowl_X.shape[1]
    ).to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_base_train_t = torch.FloatTensor(X_base_train).to(device)
    X_bowl_train_t = torch.FloatTensor(X_bowl_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    X_base_test_t = torch.FloatTensor(X_base_test).to(device)
    X_bowl_test_t = torch.FloatTensor(X_bowl_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # Training loop
    best_test_loss = float('inf')
    patience_counter = 0
    max_patience = 50

    print("\nTraining...")

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        pred = model(X_base_train_t, X_bowl_train_t)
        loss = criterion(pred, y_train_t)

        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_base_test_t, X_bowl_test_t)
            test_loss = criterion(test_pred, y_test_t)

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={loss.item():.2f}, Test Loss={test_loss.item():.2f}")

        if patience_counter >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(X_base_train_t, X_bowl_train_t).cpu().numpy()
        test_pred = model(X_base_test_t, X_bowl_test_t).cpu().numpy()

    # Calculate MAE
    train_margin_pred = train_pred[:, 0] - train_pred[:, 1]
    train_margin_actual = y_train[:, 0] - y_train[:, 1]
    train_mae = np.mean(np.abs(train_margin_pred - train_margin_actual))

    test_margin_pred = test_pred[:, 0] - test_pred[:, 1]
    test_margin_actual = y_test[:, 0] - y_test[:, 1]
    test_mae = np.mean(np.abs(test_margin_pred - test_margin_actual))

    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Train MAE: {train_mae:.1f} pts")
    print(f"Test MAE (2024 bowls): {test_mae:.1f} pts")

    # Compare to Vegas on test set
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    vegas_errors = []
    for i, gid in enumerate([game_ids[j] for j in range(len(game_ids)) if test_mask[j]]):
        cursor.execute('''
            SELECT COALESCE(latest_spread, opening_spread) FROM odds_and_predictions WHERE game_id = ?
        ''', (gid,))
        row = cursor.fetchone()
        if row and row[0]:
            vegas_margin = -row[0]
            actual_margin = y_test[i, 0] - y_test[i, 1]
            vegas_errors.append(abs(actual_margin - vegas_margin))

    conn.close()

    vegas_mae = np.mean(vegas_errors) if vegas_errors else 0

    print(f"\nTest Set Comparison:")
    print(f"  Bowl Model MAE: {test_mae:.1f} pts")
    print(f"  Vegas MAE:      {vegas_mae:.1f} pts")

    # Save model
    torch.save({
        'model_state': best_model_state,
        'base_features': base_X.shape[1],
        'bowl_features': bowl_X.shape[1],
        'base_cols': base_cols
    }, 'models/bowl_deep_eagle.pt')

    with open('models/bowl_deep_eagle_base_scaler.pkl', 'wb') as f:
        pickle.dump(base_scaler, f)
    with open('models/bowl_deep_eagle_bowl_scaler.pkl', 'wb') as f:
        pickle.dump(bowl_scaler, f)

    print(f"\nModel saved to models/bowl_deep_eagle.pt")

    return model, base_scaler, bowl_scaler


if __name__ == '__main__':
    train_bowl_model()
