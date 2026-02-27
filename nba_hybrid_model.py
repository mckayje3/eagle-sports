"""
NBA Hybrid Deep Learning Model

Uses statistical model features as inputs to a neural network:
- Pre-computed team ratings (offensive, defensive)
- Rest/B2B status
- Previous season ratings
- GRU for recent game sequence (optional enhancement)

This leverages domain knowledge while allowing the NN to find non-linear patterns.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import pickle

DB_PATH = Path(__file__).parent / 'nba_games.db'
MODEL_DIR = Path(__file__).parent / 'models'


class StatisticalFeatureExtractor:
    """Extract features using our proven statistical model logic."""

    def __init__(self, decay: float = 0.93, prev_hl: float = 6.0):
        self.decay = decay
        self.prev_hl = prev_hl
        self.team_games = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'weights': []
        }))
        self.prev_ratings = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}
        self.last_game_date = {}

    def _wavg(self, vals, wts):
        return float(np.average(vals, weights=wts)) if vals else None

    def get_team_rating(self, tid: int, season: int) -> tuple[float, float]:
        td = self.team_games[tid][season]
        gp = len(td['ppg'])
        ppg = self._wavg(td['ppg'], td['weights'])
        papg = self._wavg(td['papg'], td['weights'])
        prev_off = self.prev_ratings.get(tid, {}).get('off', self.league_avg['ppg'])
        prev_def = self.prev_ratings.get(tid, {}).get('def', self.league_avg['papg'])
        if ppg is None:
            return prev_off, prev_def
        blend = 0.5 ** (gp / self.prev_hl)
        return blend * prev_off + (1 - blend) * ppg, blend * prev_def + (1 - blend) * papg

    def get_rest(self, tid: int, gdate: str) -> int:
        if tid not in self.last_game_date:
            return 3
        curr = datetime.strptime(gdate[:10], '%Y-%m-%d')
        last = datetime.strptime(self.last_game_date[tid][:10], '%Y-%m-%d')
        return max(0, min((curr - last).days - 1, 5))

    def get_games_played(self, tid: int, season: int) -> int:
        return len(self.team_games[tid][season]['ppg'])

    def update(self, hid: int, aid: int, hs: int, aws: int, season: int, gdate: str):
        for tid in [hid, aid]:
            self.team_games[tid][season]['weights'] = [
                w * self.decay for w in self.team_games[tid][season]['weights']
            ]
        self.team_games[hid][season]['ppg'].append(hs)
        self.team_games[hid][season]['papg'].append(aws)
        self.team_games[hid][season]['weights'].append(1.0)
        self.team_games[aid][season]['ppg'].append(aws)
        self.team_games[aid][season]['papg'].append(hs)
        self.team_games[aid][season]['weights'].append(1.0)
        self.last_game_date[hid] = gdate
        self.last_game_date[aid] = gdate

    def set_prev_season(self, season: int):
        ps = season - 1
        for tid in self.team_games:
            if ps in self.team_games[tid] and self.team_games[tid][ps]['ppg']:
                self.prev_ratings[tid] = {
                    'off': np.mean(self.team_games[tid][ps]['ppg']),
                    'def': np.mean(self.team_games[tid][ps]['papg'])
                }
        self.last_game_date.clear()

    def set_league_avg(self, ppg: float, papg: float):
        self.league_avg = {'ppg': ppg, 'papg': papg}


def extract_features(games: pd.DataFrame) -> pd.DataFrame:
    """Extract features for all games using statistical model."""
    extractor = StatisticalFeatureExtractor()
    features = []

    for season in sorted(games.season.unique()):
        if season > games.season.min():
            extractor.set_prev_season(season)
            pg = games[games.season == season - 1]
            if len(pg) > 0:
                extractor.set_league_avg(pg['home_score'].mean(), pg['away_score'].mean())

        season_games = games[games.season == season].sort_values('game_date_eastern')

        for _, g in season_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            gdate = g['game_date_eastern']

            # Get features BEFORE update
            h_off, h_def = extractor.get_team_rating(hid, season)
            a_off, a_def = extractor.get_team_rating(aid, season)
            h_rest = extractor.get_rest(hid, gdate)
            a_rest = extractor.get_rest(aid, gdate)
            h_gp = extractor.get_games_played(hid, season)
            a_gp = extractor.get_games_played(aid, season)

            # Statistical model prediction (for reference)
            hca = 2.0
            stat_home = (h_off + a_def) / 2 + hca / 2
            stat_away = (a_off + h_def) / 2 - hca / 2
            adj = 0.0
            if h_rest == 0:
                adj -= 1.0
            if a_rest == 0:
                adj += 1.0
            stat_spread = (stat_away - adj/2) - (stat_home + adj/2)

            features.append({
                'game_id': g['game_id'],
                'season': season,
                # Team ratings
                'home_off': h_off,
                'home_def': h_def,
                'away_off': a_off,
                'away_def': a_def,
                # Rating differentials
                'off_diff': h_off - a_off,
                'def_diff': h_def - a_def,  # Lower is better for home
                # Rest
                'home_rest': h_rest,
                'away_rest': a_rest,
                'rest_diff': h_rest - a_rest,
                'home_b2b': 1 if h_rest == 0 else 0,
                'away_b2b': 1 if a_rest == 0 else 0,
                # Games played (sample size proxy)
                'home_gp': h_gp,
                'away_gp': a_gp,
                'min_gp': min(h_gp, a_gp),
                # Statistical model prediction
                'stat_spread': stat_spread,
                'stat_total': stat_home + stat_away,
                # Targets
                'actual_spread': g['away_score'] - g['home_score'],
                'actual_total': g['home_score'] + g['away_score']
            })

            # Update AFTER extracting features
            extractor.update(hid, aid, g['home_score'], g['away_score'], season, gdate)

    return pd.DataFrame(features)


class HybridDataset(Dataset):
    """Dataset with statistical features."""

    def __init__(self, features_df: pd.DataFrame, feature_cols: list[str], scaler=None):
        self.feature_cols = feature_cols
        self.X = features_df[feature_cols].values.astype(np.float32)
        self.spread = features_df['actual_spread'].values.astype(np.float32)
        self.total = features_df['actual_total'].values.astype(np.float32)

        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.X[idx], dtype=torch.float32),
            'spread': torch.tensor(self.spread[idx], dtype=torch.float32),
            'total': torch.tensor(self.total[idx], dtype=torch.float32)
        }


class HybridNNModel(nn.Module):
    """Simple feedforward NN on statistical features."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = [64, 32], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.spread_head = nn.Linear(prev_dim, 1)
        self.total_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        spread = self.spread_head(features).squeeze(-1)
        total = self.total_head(features).squeeze(-1)
        return spread, total


def train_model(
    train_dataset: HybridDataset,
    val_dataset: HybridDataset,
    input_dim: int,
    hidden_dims: list[int] = [64, 32],
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 15
) -> tuple[HybridNNModel, dict]:
    """Train the hybrid model."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = HybridNNModel(input_dim, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'val_spread_mae': [], 'val_total_mae': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch['features'].to(device)
            spread_target = batch['spread'].to(device)
            total_target = batch['total'].to(device)

            optimizer.zero_grad()
            spread_pred, total_pred = model(features)

            # Weighted loss: spread is harder, weight it more
            loss_spread = nn.functional.mse_loss(spread_pred, spread_target)
            loss_total = nn.functional.mse_loss(total_pred, total_target)
            loss = loss_spread * 2 + loss_total  # Weight spread more

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        spread_errors = []
        total_errors = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                spread_target = batch['spread'].to(device)
                total_target = batch['total'].to(device)

                spread_pred, total_pred = model(features)

                loss_spread = nn.functional.mse_loss(spread_pred, spread_target)
                loss_total = nn.functional.mse_loss(total_pred, total_target)
                val_loss += (loss_spread * 2 + loss_total).item()

                spread_errors.extend((spread_pred - spread_target).abs().cpu().numpy())
                total_errors.extend((total_pred - total_target).abs().cpu().numpy())

        val_loss /= len(val_loader)
        val_spread_mae = np.mean(spread_errors)
        val_total_mae = np.mean(total_errors)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_spread_mae'].append(val_spread_mae)
        history['val_total_mae'].append(val_total_mae)

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f'Epoch {epoch+1:3d}: train_loss={train_loss:.2f}, val_loss={val_loss:.2f}, '
                  f'spread_MAE={val_spread_mae:.2f}, total_MAE={val_total_mae:.2f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    model.load_state_dict(best_model_state)
    return model, history


def evaluate(model: HybridNNModel, dataset: HybridDataset, device: torch.device,
             stat_spread: np.ndarray = None) -> dict:
    """Evaluate model."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()

    all_spread_pred = []
    all_spread_actual = []
    all_total_pred = []
    all_total_actual = []

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            spread_pred, total_pred = model(features)

            all_spread_pred.extend(spread_pred.cpu().numpy())
            all_spread_actual.extend(batch['spread'].numpy())
            all_total_pred.extend(total_pred.cpu().numpy())
            all_total_actual.extend(batch['total'].numpy())

    spread_pred = np.array(all_spread_pred)
    spread_actual = np.array(all_spread_actual)
    total_pred = np.array(all_total_pred)
    total_actual = np.array(all_total_actual)

    # Metrics
    spread_mae = np.abs(spread_pred - spread_actual).mean()
    total_mae = np.abs(total_pred - total_actual).mean()

    pred_home_win = spread_pred < 0
    actual_home_win = spread_actual < 0
    winner_acc = (pred_home_win == actual_home_win).mean()

    # Compare with statistical baseline if provided
    if stat_spread is not None:
        stat_home_win = stat_spread < 0
        stat_acc = (stat_home_win == actual_home_win).mean()
    else:
        stat_acc = None

    return {
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'winner_acc': winner_acc,
        'stat_acc': stat_acc
    }


def main():
    print('=' * 60)
    print('NBA HYBRID MODEL (Statistical Features + NN)')
    print('=' * 60)

    # Load data
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT game_id, season, game_date_eastern, home_team_id, away_team_id,
               home_score, away_score
        FROM games WHERE home_score > 0
        ORDER BY game_date_eastern, game_id
    ''', conn)
    conn.close()

    print(f'\nLoaded {len(games)} games')
    print(f'Seasons: {sorted(games.season.unique())}')

    # Extract features
    print('\nExtracting statistical features...')
    features = extract_features(games)
    print(f'Features extracted: {len(features)} samples')

    # Feature columns (excluding targets and identifiers)
    feature_cols = [
        'home_off', 'home_def', 'away_off', 'away_def',
        'off_diff', 'def_diff',
        'home_rest', 'away_rest', 'rest_diff',
        'home_b2b', 'away_b2b',
        'home_gp', 'away_gp', 'min_gp',
        'stat_spread', 'stat_total'
    ]
    print(f'Feature columns: {len(feature_cols)}')

    # Split data
    seasons = sorted(features.season.unique())
    test_season = seasons[-1]
    val_season = seasons[-2] if len(seasons) > 2 else None

    if val_season:
        train_df = features[~features.season.isin([val_season, test_season])]
        val_df = features[features.season == val_season]
    else:
        train_df = features[features.season != test_season]
        n_val = int(len(train_df) * 0.2)
        val_df = train_df.tail(n_val)
        train_df = train_df.head(len(train_df) - n_val)

    test_df = features[features.season == test_season]

    print(f'\nTrain: {len(train_df)} samples')
    print(f'Val: {len(val_df)} samples')
    print(f'Test: {len(test_df)} samples (season {test_season})')

    # Create datasets
    train_dataset = HybridDataset(train_df, feature_cols)
    val_dataset = HybridDataset(val_df, feature_cols, scaler=train_dataset.scaler)
    test_dataset = HybridDataset(test_df, feature_cols, scaler=train_dataset.scaler)

    # Statistical baseline on test set
    stat_spread_test = test_df['stat_spread'].values
    stat_home_win = stat_spread_test < 0
    actual_home_win = test_df['actual_spread'].values < 0
    stat_acc = (stat_home_win == actual_home_win).mean()
    stat_spread_mae = np.abs(stat_spread_test - test_df['actual_spread'].values).mean()

    print(f'\nStatistical baseline on test: {stat_acc*100:.1f}% accuracy, MAE={stat_spread_mae:.2f}')

    # Train model
    print('\n' + '=' * 60)
    print('TRAINING')
    print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, history = train_model(
        train_dataset, val_dataset,
        input_dim=len(feature_cols),
        hidden_dims=[64, 32],
        num_epochs=150,
        batch_size=32,
        learning_rate=0.001,
        patience=20
    )

    # Evaluate
    print('\n' + '=' * 60)
    print('TEST SET EVALUATION')
    print('=' * 60)

    test_metrics = evaluate(model, test_dataset, device, stat_spread_test)

    print(f"\nHybrid NN:")
    print(f"  Spread MAE: {test_metrics['spread_mae']:.2f}")
    print(f"  Total MAE: {test_metrics['total_mae']:.2f}")
    print(f"  Winner Accuracy: {test_metrics['winner_acc']*100:.1f}%")

    print(f"\nStatistical Baseline:")
    print(f"  Spread MAE: {stat_spread_mae:.2f}")
    print(f"  Winner Accuracy: {stat_acc*100:.1f}%")

    improvement = test_metrics['winner_acc'] - stat_acc
    print(f"\nImprovement: {improvement*100:+.1f}%")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / 'nba_hybrid_v1.pt'
    scaler_path = MODEL_DIR / 'nba_hybrid_v1_scaler.pkl'

    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'hidden_dims': [64, 32],
        'history': history
    }, model_path)

    with open(scaler_path, 'wb') as f:
        pickle.dump(train_dataset.scaler, f)

    print(f'\nModel saved to {model_path}')


if __name__ == '__main__':
    main()
