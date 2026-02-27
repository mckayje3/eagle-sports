"""
NBA Vegas-Blend Model

Uses neural network to learn optimal blending of:
1. Our statistical model predictions
2. Vegas lines
3. Context features (rest, games played, etc.)

Goal: Learn when to trust our model vs Vegas.
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


def extract_features_with_vegas(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Extract features including Vegas odds."""
    # Merge games with odds
    games = games.merge(odds[['game_id', 'latest_spread', 'latest_total']],
                        on='game_id', how='inner')

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

            # Statistical model prediction
            hca = 2.0
            stat_home = (h_off + a_def) / 2 + hca / 2
            stat_away = (a_off + h_def) / 2 - hca / 2
            adj = 0.0
            if h_rest == 0:
                adj -= 1.0
            if a_rest == 0:
                adj += 1.0
            stat_spread = (stat_away - adj/2) - (stat_home + adj/2)
            stat_total = stat_home + stat_away

            # Vegas odds (already in away-home convention)
            vegas_spread = g['latest_spread']
            vegas_total = g['latest_total'] if pd.notna(g['latest_total']) else stat_total

            features.append({
                'game_id': g['game_id'],
                'season': season,
                # Our model predictions
                'stat_spread': stat_spread,
                'stat_total': stat_total,
                # Vegas predictions
                'vegas_spread': vegas_spread,
                'vegas_total': vegas_total,
                # Difference between our model and Vegas
                'spread_diff': stat_spread - vegas_spread,
                'total_diff': stat_total - vegas_total,
                # Context features
                'home_rest': h_rest,
                'away_rest': a_rest,
                'rest_diff': h_rest - a_rest,
                'home_b2b': 1 if h_rest == 0 else 0,
                'away_b2b': 1 if a_rest == 0 else 0,
                'min_gp': min(h_gp, a_gp),
                'gp_diff': abs(h_gp - a_gp),
                # Targets
                'actual_spread': g['away_score'] - g['home_score'],
                'actual_total': g['home_score'] + g['away_score']
            })

            extractor.update(hid, aid, g['home_score'], g['away_score'], season, gdate)

    return pd.DataFrame(features)


class BlendDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], scaler=None):
        self.X = df[feature_cols].values.astype(np.float32)
        self.spread = df['actual_spread'].values.astype(np.float32)
        self.total = df['actual_total'].values.astype(np.float32)

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


class BlendModel(nn.Module):
    """Learn to blend statistical model with Vegas."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 16)
        self.spread_head = nn.Linear(16, 1)
        self.total_head = nn.Linear(16, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        spread = self.spread_head(h).squeeze(-1)
        total = self.total_head(h).squeeze(-1)
        return spread, total


def train_model(train_ds, val_ds, input_dim, epochs=100, patience=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = BlendModel(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch['features'].to(device)
            spread_t = batch['spread'].to(device)
            total_t = batch['total'].to(device)

            optimizer.zero_grad()
            spread_p, total_p = model(x)
            loss = nn.functional.mse_loss(spread_p, spread_t) + nn.functional.mse_loss(total_p, total_t)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        spread_errs, total_errs = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['features'].to(device)
                spread_t = batch['spread'].to(device)
                total_t = batch['total'].to(device)
                spread_p, total_p = model(x)
                val_loss += (nn.functional.mse_loss(spread_p, spread_t) +
                            nn.functional.mse_loss(total_p, total_t)).item()
                spread_errs.extend((spread_p - spread_t).abs().cpu().numpy())
                total_errs.extend((total_p - total_t).abs().cpu().numpy())

        val_loss /= len(val_loader)

        if (epoch + 1) % 20 == 0 or epoch < 3:
            print(f'Epoch {epoch+1}: val_loss={val_loss:.2f}, spread_MAE={np.mean(spread_errs):.2f}, total_MAE={np.mean(total_errs):.2f}')

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    model.load_state_dict(best_state)
    return model


def main():
    print('=' * 60)
    print('NBA VEGAS-BLEND MODEL')
    print('=' * 60)

    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT game_id, season, game_date_eastern, home_team_id, away_team_id,
               home_score, away_score
        FROM games WHERE home_score > 0
        ORDER BY game_date_eastern, game_id
    ''', conn)
    odds = pd.read_sql_query('''
        SELECT game_id, latest_spread, latest_total
        FROM odds_and_predictions
        WHERE latest_spread IS NOT NULL
    ''', conn)
    conn.close()

    print(f'Games: {len(games)}, with Vegas odds: {len(odds)}')

    # Extract features
    features = extract_features_with_vegas(games, odds)
    print(f'Features extracted: {len(features)} samples')

    feature_cols = [
        'stat_spread', 'stat_total',
        'vegas_spread', 'vegas_total',
        'spread_diff', 'total_diff',
        'home_rest', 'away_rest', 'rest_diff',
        'home_b2b', 'away_b2b',
        'min_gp', 'gp_diff'
    ]

    # Split
    seasons = sorted(features.season.unique())
    test_season = seasons[-1]
    val_season = seasons[-2] if len(seasons) > 2 else None

    if val_season:
        train_df = features[~features.season.isin([val_season, test_season])]
        val_df = features[features.season == val_season]
    else:
        train_df = features[features.season != test_season]
        val_df = train_df.tail(int(len(train_df) * 0.2))
        train_df = train_df.head(len(train_df) - len(val_df))

    test_df = features[features.season == test_season]

    print(f'\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')

    # Baselines on test set
    test_spread = test_df['actual_spread'].values
    test_total = test_df['actual_total'].values

    stat_spread = test_df['stat_spread'].values
    vegas_spread = test_df['vegas_spread'].values
    vegas_total = test_df['vegas_total'].values

    stat_acc = ((stat_spread < 0) == (test_spread < 0)).mean()
    vegas_acc = ((vegas_spread < 0) == (test_spread < 0)).mean()
    blend_50 = (stat_spread + vegas_spread) / 2
    blend_acc = ((blend_50 < 0) == (test_spread < 0)).mean()

    print(f'\n=== BASELINES (Test Set) ===')
    print(f'Statistical model: {stat_acc*100:.1f}%')
    print(f'Vegas:             {vegas_acc*100:.1f}%')
    print(f'50/50 blend:       {blend_acc*100:.1f}%')

    # Train NN
    print('\n=== TRAINING BLEND MODEL ===')
    train_ds = BlendDataset(train_df, feature_cols)
    val_ds = BlendDataset(val_df, feature_cols, train_ds.scaler)
    test_ds = BlendDataset(test_df, feature_cols, train_ds.scaler)

    model = train_model(train_ds, val_ds, len(feature_cols))

    # Evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    loader = DataLoader(test_ds, batch_size=64)
    spread_preds, total_preds = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['features'].to(device)
            sp, tp = model(x)
            spread_preds.extend(sp.cpu().numpy())
            total_preds.extend(tp.cpu().numpy())

    spread_preds = np.array(spread_preds)
    total_preds = np.array(total_preds)

    nn_acc = ((spread_preds < 0) == (test_spread < 0)).mean()
    nn_spread_mae = np.abs(spread_preds - test_spread).mean()
    nn_total_mae = np.abs(total_preds - test_total).mean()

    vegas_spread_mae = np.abs(vegas_spread - test_spread).mean()
    vegas_total_mae = np.abs(vegas_total - test_total).mean()

    print(f'\n=== RESULTS ===')
    print(f"{'Model':<20} {'Winner Acc':<12} {'Spread MAE':<12} {'Total MAE'}")
    print('-' * 56)
    print(f"{'Statistical':<20} {stat_acc*100:.1f}%{'':<7} {np.abs(stat_spread - test_spread).mean():.2f}")
    print(f"{'Vegas':<20} {vegas_acc*100:.1f}%{'':<7} {vegas_spread_mae:.2f}{'':<8} {vegas_total_mae:.2f}")
    print(f"{'50/50 Blend':<20} {blend_acc*100:.1f}%")
    print(f"{'NN Blend':<20} {nn_acc*100:.1f}%{'':<7} {nn_spread_mae:.2f}{'':<8} {nn_total_mae:.2f}")

    # Save
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols
    }, MODEL_DIR / 'nba_vegas_blend_v1.pt')
    with open(MODEL_DIR / 'nba_vegas_blend_v1_scaler.pkl', 'wb') as f:
        pickle.dump(train_ds.scaler, f)

    print(f'\nModel saved.')


if __name__ == '__main__':
    main()
