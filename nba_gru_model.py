"""
NBA GRU-based Deep Learning Model

Predicts spread and total using:
- GRU to encode each team's recent game history
- Context features (rest days, B2B)
- Shared encoder weights for home/away teams
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


class NBAGameDataset(Dataset):
    """Dataset that provides game sequences for each team."""

    def __init__(self, games_df: pd.DataFrame, seq_length: int = 10):
        self.seq_length = seq_length
        self.samples = []
        self.scaler = StandardScaler()

        # Build team game histories
        team_history = defaultdict(list)  # team_id -> [(date, score, opp_score, was_home)]
        last_game_date = {}

        # Process games chronologically
        games_sorted = games_df.sort_values('game_date_eastern').reset_index(drop=True)

        for _, g in games_sorted.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            gdate = g['game_date_eastern']
            hs, aws = g['home_score'], g['away_score']

            # Calculate rest days
            home_rest = self._calc_rest(hid, gdate, last_game_date)
            away_rest = self._calc_rest(aid, gdate, last_game_date)

            # Get sequences BEFORE this game (for prediction)
            home_seq = self._get_sequence(team_history[hid])
            away_seq = self._get_sequence(team_history[aid])

            # Target: spread (away - home) and total
            spread = aws - hs
            total = hs + aws

            # Store sample
            self.samples.append({
                'home_seq': home_seq,
                'away_seq': away_seq,
                'home_rest': home_rest,
                'away_rest': away_rest,
                'home_b2b': 1.0 if home_rest == 0 else 0.0,
                'away_b2b': 1.0 if away_rest == 0 else 0.0,
                'spread': spread,
                'total': total,
                'game_id': g['game_id'],
                'season': g['season']
            })

            # Update histories AFTER creating sample
            team_history[hid].append({
                'score': hs, 'opp_score': aws, 'was_home': 1.0
            })
            team_history[aid].append({
                'score': aws, 'opp_score': hs, 'was_home': 0.0
            })
            last_game_date[hid] = gdate
            last_game_date[aid] = gdate

        # Fit scaler on sequence data
        all_seqs = np.concatenate([
            np.array(s['home_seq']).flatten() for s in self.samples
        ] + [
            np.array(s['away_seq']).flatten() for s in self.samples
        ]).reshape(-1, 1)
        self.scaler.fit(all_seqs)

    def _calc_rest(self, team_id: int, game_date: str, last_game_date: dict) -> int:
        if team_id not in last_game_date:
            return 3
        curr = datetime.strptime(game_date[:10], '%Y-%m-%d')
        last = datetime.strptime(last_game_date[team_id][:10], '%Y-%m-%d')
        return max(0, min((curr - last).days - 1, 5))

    def _get_sequence(self, history: list) -> np.ndarray:
        """Get last N games as sequence, padded if needed."""
        seq = np.zeros((self.seq_length, 3))  # [score, opp_score, was_home]

        if len(history) == 0:
            # No history - use league average placeholders
            seq[:, 0] = 115.0  # score
            seq[:, 1] = 115.0  # opp_score
            seq[:, 2] = 0.5    # was_home (neutral)
        else:
            # Fill from most recent backwards
            n = min(len(history), self.seq_length)
            recent = history[-n:]
            for i, g in enumerate(recent):
                idx = self.seq_length - n + i  # Right-align sequence
                seq[idx, 0] = g['score']
                seq[idx, 1] = g['opp_score']
                seq[idx, 2] = g['was_home']

            # Pad earlier slots with averages from available history
            if n < self.seq_length:
                avg_score = np.mean([g['score'] for g in history])
                avg_opp = np.mean([g['opp_score'] for g in history])
                for i in range(self.seq_length - n):
                    seq[i, 0] = avg_score
                    seq[i, 1] = avg_opp
                    seq[i, 2] = 0.5

        return seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Scale sequences (scores only, not was_home)
        home_seq = s['home_seq'].copy()
        away_seq = s['away_seq'].copy()

        home_seq[:, :2] = self.scaler.transform(home_seq[:, :2].reshape(-1, 1)).reshape(-1, 2)
        away_seq[:, :2] = self.scaler.transform(away_seq[:, :2].reshape(-1, 1)).reshape(-1, 2)

        # Context features (normalized)
        context = np.array([
            s['home_rest'] / 5.0,
            s['away_rest'] / 5.0,
            s['home_b2b'],
            s['away_b2b']
        ], dtype=np.float32)

        return {
            'home_seq': torch.tensor(home_seq, dtype=torch.float32),
            'away_seq': torch.tensor(away_seq, dtype=torch.float32),
            'context': torch.tensor(context, dtype=torch.float32),
            'spread': torch.tensor(s['spread'], dtype=torch.float32),
            'total': torch.tensor(s['total'], dtype=torch.float32)
        }


class NBAGRUModel(nn.Module):
    """
    GRU-based model for NBA game prediction.

    Architecture:
    - Shared GRU encoder for team game sequences
    - Concatenate home/away encodings with context
    - Dense layers to predict spread and total
    """

    def __init__(
        self,
        seq_features: int = 3,
        hidden_size: int = 32,
        num_layers: int = 1,
        context_size: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Shared GRU encoder
        self.gru = nn.GRU(
            input_size=seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Combined dimension: home_state + away_state + context
        combined_dim = hidden_size * 2 + context_size

        # Dense layers
        self.fc1 = nn.Linear(combined_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_spread = nn.Linear(32, 1)
        self.fc_total = nn.Linear(32, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def encode_team(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode team's game sequence to a state vector."""
        # seq: (batch, seq_len, features)
        _, hidden = self.gru(seq)
        # hidden: (num_layers, batch, hidden_size)
        return hidden[-1]  # Last layer's hidden state

    def forward(self, home_seq: torch.Tensor, away_seq: torch.Tensor,
                context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode both teams
        home_state = self.encode_team(home_seq)
        away_state = self.encode_team(away_seq)

        # Combine with context
        combined = torch.cat([home_state, away_state, context], dim=1)

        # Dense layers
        x = self.dropout(self.relu(self.fc1(combined)))
        x = self.dropout(self.relu(self.fc2(x)))

        # Separate heads for spread and total
        spread = self.fc_spread(x).squeeze(-1)
        total = self.fc_total(x).squeeze(-1)

        return spread, total


def load_games() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT game_id, season, game_date_eastern, home_team_id, away_team_id,
               home_score, away_score
        FROM games
        WHERE home_score > 0
        ORDER BY game_date_eastern, game_id
    ''', conn)
    conn.close()
    return games


def train_model(
    train_dataset: NBAGameDataset,
    val_dataset: NBAGameDataset,
    hidden_size: int = 32,
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 10
) -> tuple[NBAGRUModel, dict]:
    """Train the GRU model."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = NBAGRUModel(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
            home_seq = batch['home_seq'].to(device)
            away_seq = batch['away_seq'].to(device)
            context = batch['context'].to(device)
            spread_target = batch['spread'].to(device)
            total_target = batch['total'].to(device)

            optimizer.zero_grad()
            spread_pred, total_pred = model(home_seq, away_seq, context)

            # Combined loss (equal weight for spread and total)
            loss_spread = nn.functional.mse_loss(spread_pred, spread_target)
            loss_total = nn.functional.mse_loss(total_pred, total_target)
            loss = loss_spread + loss_total

            loss.backward()
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
                home_seq = batch['home_seq'].to(device)
                away_seq = batch['away_seq'].to(device)
                context = batch['context'].to(device)
                spread_target = batch['spread'].to(device)
                total_target = batch['total'].to(device)

                spread_pred, total_pred = model(home_seq, away_seq, context)

                loss_spread = nn.functional.mse_loss(spread_pred, spread_target)
                loss_total = nn.functional.mse_loss(total_pred, total_target)
                val_loss += (loss_spread + loss_total).item()

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

        print(f'Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, '
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

    # Load best model
    model.load_state_dict(best_model_state)
    return model, history


def evaluate_model(model: NBAGRUModel, dataset: NBAGameDataset, device: torch.device) -> dict:
    """Evaluate model and return metrics."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()

    all_spread_pred = []
    all_spread_actual = []
    all_total_pred = []
    all_total_actual = []

    with torch.no_grad():
        for batch in loader:
            home_seq = batch['home_seq'].to(device)
            away_seq = batch['away_seq'].to(device)
            context = batch['context'].to(device)

            spread_pred, total_pred = model(home_seq, away_seq, context)

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

    # Winner accuracy (spread < 0 means home wins)
    pred_home_win = spread_pred < 0
    actual_home_win = spread_actual < 0
    winner_acc = (pred_home_win == actual_home_win).mean()

    # Over/under accuracy (using median total as rough line)
    median_total = np.median(total_actual)
    pred_over = total_pred > median_total
    actual_over = total_actual > median_total
    ou_acc = (pred_over == actual_over).mean()

    return {
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'winner_acc': winner_acc,
        'ou_acc': ou_acc,
        'spread_pred': spread_pred,
        'spread_actual': spread_actual,
        'total_pred': total_pred,
        'total_actual': total_actual
    }


def main():
    print('=' * 60)
    print('NBA GRU MODEL TRAINING')
    print('=' * 60)

    # Load data
    games = load_games()
    print(f'\nLoaded {len(games)} games')
    print(f'Seasons: {sorted(games.season.unique())}')

    # Split by season (use last season for test)
    seasons = sorted(games.season.unique())
    train_seasons = seasons[:-1]
    test_season = seasons[-1]

    train_games = games[games.season.isin(train_seasons)]
    test_games = games[games.season == test_season]

    # Further split train into train/val (use second-to-last season for val)
    if len(train_seasons) > 1:
        val_season = train_seasons[-1]
        train_seasons = train_seasons[:-1]
        train_games = games[games.season.isin(train_seasons)]
        val_games = games[games.season == val_season]
    else:
        # If only 2 seasons, split last season 80/20
        val_games = train_games.tail(int(len(train_games) * 0.2))
        train_games = train_games.head(int(len(train_games) * 0.8))

    print(f'\nTrain: {len(train_games)} games (seasons {train_seasons})')
    print(f'Val: {len(val_games)} games')
    print(f'Test: {len(test_games)} games (season {test_season})')

    # Create datasets
    seq_length = 10
    print(f'\nSequence length: {seq_length} games')

    # Build datasets with full history
    full_train = games[games.season.isin(train_seasons)]
    full_val = games[games.season.isin(train_seasons + [val_season] if isinstance(train_seasons, list) else [train_seasons, val_season])]

    train_dataset = NBAGameDataset(full_train, seq_length=seq_length)
    val_dataset = NBAGameDataset(full_val, seq_length=seq_length)
    test_dataset = NBAGameDataset(games, seq_length=seq_length)

    # Filter val/test to only games from those seasons
    val_dataset.samples = [s for s in val_dataset.samples
                          if s['season'] == val_season]
    test_dataset.samples = [s for s in test_dataset.samples
                           if s['season'] == test_season]

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')

    # Train model
    print('\n' + '=' * 60)
    print('TRAINING')
    print('=' * 60)

    model, history = train_model(
        train_dataset, val_dataset,
        hidden_size=32,
        num_epochs=100,
        batch_size=64,
        learning_rate=0.001,
        patience=15
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Evaluate on test set
    print('\n' + '=' * 60)
    print('TEST SET EVALUATION')
    print('=' * 60)

    test_metrics = evaluate_model(model, test_dataset, device)

    print(f"\nSpread MAE: {test_metrics['spread_mae']:.2f}")
    print(f"Total MAE: {test_metrics['total_mae']:.2f}")
    print(f"Winner Accuracy: {test_metrics['winner_acc']*100:.1f}%")
    print(f"Over/Under Accuracy: {test_metrics['ou_acc']*100:.1f}%")

    # Compare with baseline
    print('\n' + '=' * 60)
    print('COMPARISON WITH STATISTICAL BASELINE')
    print('=' * 60)
    print(f"{'Model':<30} {'Winner Acc':<15} {'Spread MAE'}")
    print('-' * 55)
    print(f"{'Statistical Baseline':<30} {'65.2%':<15} {'~11.4'}")
    print(f"{'GRU Model':<30} {test_metrics['winner_acc']*100:.1f}%{'':<10} {test_metrics['spread_mae']:.2f}")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / 'nba_gru_v1.pt'
    scaler_path = MODEL_DIR / 'nba_gru_v1_scaler.pkl'

    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': 32,
        'seq_length': seq_length,
        'history': history
    }, model_path)

    with open(scaler_path, 'wb') as f:
        pickle.dump(train_dataset.scaler, f)

    print(f'\nModel saved to {model_path}')


if __name__ == '__main__':
    main()
