"""
NBA Deep Learning Models - GRU, LSTM, and Transformer

Comparison of sequence models for NBA game prediction.
Key challenge: ~1400 training samples requires aggressive anti-overfitting.

Anti-Overfitting Strategy:
1. TINY models (< 10K parameters vs typical 100K+)
2. Heavy dropout (0.3-0.5)
3. Strong L2 regularization (weight decay)
4. Early stopping with patience
5. Sequence length limited to 5-10 games
6. Batch normalization for stability
7. Learning rate scheduling
8. Vegas blend as regularization (anchor predictions)
"""
from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'nba_games.db'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters - tuned for small dataset
SEQ_LEN = 5          # Last N games per team
HIDDEN_DIM = 32      # Very small hidden dimension
NUM_LAYERS = 1       # Single layer to reduce parameters
DROPOUT = 0.4        # Aggressive dropout
WEIGHT_DECAY = 1e-3  # Strong L2 regularization
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15        # Early stopping patience


class NBASequenceDataset(Dataset):
    """
    Dataset that provides game sequences for each team.

    For each game, we provide:
    - Home team's last N games (as sequence)
    - Away team's last N games (as sequence)
    - Vegas odds (for blending)
    - Target: actual spread and total
    """

    # Features per game in sequence
    GAME_FEATURES = [
        'points_for', 'points_against', 'fg_pct', 'three_pct', 'ft_pct',
        'rebounds', 'assists', 'steals', 'blocks', 'turnovers',
        'was_home', 'won', 'rest_days'
    ]
    NUM_FEATURES = len(GAME_FEATURES)

    def __init__(self, games_df: pd.DataFrame, team_histories: dict,
                 scaler: StandardScaler = None, fit_scaler: bool = False):
        self.games = games_df.reset_index(drop=True)
        self.team_histories = team_histories
        self.seq_len = SEQ_LEN

        # Build sequences
        self.sequences = []
        self.targets = []
        self.vegas = []

        for idx, game in self.games.iterrows():
            home_seq = self._get_team_sequence(
                game['home_team_id'], game['date'], game['season']
            )
            away_seq = self._get_team_sequence(
                game['away_team_id'], game['date'], game['season']
            )

            if home_seq is None or away_seq is None:
                continue

            # Stack home and away sequences
            combined = np.concatenate([home_seq, away_seq], axis=1)
            self.sequences.append(combined)

            # Targets
            actual_spread = game['away_score'] - game['home_score']
            actual_total = game['home_score'] + game['away_score']
            self.targets.append([actual_spread, actual_total])

            # Vegas
            vegas_spread = game['vegas_spread'] if pd.notna(game['vegas_spread']) else 0
            vegas_total = game['vegas_total'] if pd.notna(game['vegas_total']) else 220
            self.vegas.append([vegas_spread, vegas_total])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.vegas = np.array(self.vegas, dtype=np.float32)

        # Normalize sequences
        if fit_scaler:
            # Reshape to 2D for scaler
            n_samples, seq_len, n_features = self.sequences.shape
            flat = self.sequences.reshape(-1, n_features)
            self.scaler = StandardScaler()
            flat_scaled = self.scaler.fit_transform(flat)
            self.sequences = flat_scaled.reshape(n_samples, seq_len, n_features)
        elif scaler is not None:
            self.scaler = scaler
            n_samples, seq_len, n_features = self.sequences.shape
            flat = self.sequences.reshape(-1, n_features)
            flat_scaled = self.scaler.transform(flat)
            self.sequences = flat_scaled.reshape(n_samples, seq_len, n_features)
        else:
            self.scaler = None

        log.info(f"  Dataset: {len(self.sequences)} games, seq_len={self.seq_len}, "
                 f"features={self.sequences.shape[2]}")

    def _get_team_sequence(self, team_id: int, game_date: str, season: int) -> np.ndarray | None:
        """Get team's last N games as a sequence."""
        key = (team_id, season)
        if key not in self.team_histories:
            return None

        history = self.team_histories[key]
        # Filter to games before this date
        prior_games = [g for g in history if g['date'] < game_date]

        if len(prior_games) < 2:  # Need at least 2 games of history
            return None

        # Take last N games
        recent = prior_games[-self.seq_len:]

        # Pad if needed
        if len(recent) < self.seq_len:
            padding = [self._empty_game() for _ in range(self.seq_len - len(recent))]
            recent = padding + recent

        # Convert to feature matrix
        seq = np.array([[
            g['points_for'], g['points_against'], g['fg_pct'], g['three_pct'],
            g['ft_pct'], g['rebounds'], g['assists'], g['steals'],
            g['blocks'], g['turnovers'], g['was_home'], g['won'], g['rest_days']
        ] for g in recent], dtype=np.float32)

        return seq

    def _empty_game(self) -> dict:
        """Return empty game for padding."""
        return {
            'date': '1900-01-01',
            'points_for': 110, 'points_against': 110,
            'fg_pct': 46, 'three_pct': 36, 'ft_pct': 78,
            'rebounds': 44, 'assists': 25, 'steals': 7.5,
            'blocks': 5, 'turnovers': 14,
            'was_home': 0.5, 'won': 0.5, 'rest_days': 2
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.targets[idx]),
            torch.tensor(self.vegas[idx])
        )


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class GRUPredictor(nn.Module):
    """
    GRU-based predictor.

    Architecture:
    - Separate GRU for home and away team sequences
    - Concatenate final hidden states
    - Small MLP head for spread/total prediction

    Parameters: ~5K (very small)
    """

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Shared GRU for both teams (parameter efficient)
        self.gru = nn.GRU(
            input_size=input_dim // 2,  # Half for each team
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Batch norm for stability
        self.bn = nn.BatchNorm1d(hidden_dim * 2)

        # Small prediction head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # spread, total
        )

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"  GRU parameters: {total:,}")

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # Split into home and away
        home_x = x[:, :, :features // 2]
        away_x = x[:, :, features // 2:]

        # Process each team
        _, home_h = self.gru(home_x)
        _, away_h = self.gru(away_x)

        # Concatenate final hidden states
        combined = torch.cat([home_h[-1], away_h[-1]], dim=1)

        # Batch norm
        combined = self.bn(combined)

        # Predict
        return self.head(combined)


class LSTMPredictor(nn.Module):
    """
    LSTM-based predictor.

    Similar to GRU but with LSTM cells.
    LSTMs have separate cell state which may help with longer-term patterns.

    Parameters: ~7K
    """

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Shared LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim // 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.bn = nn.BatchNorm1d(hidden_dim * 2)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"  LSTM parameters: {total:,}")

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        home_x = x[:, :, :features // 2]
        away_x = x[:, :, features // 2:]

        _, (home_h, _) = self.lstm(home_x)
        _, (away_h, _) = self.lstm(away_x)

        combined = torch.cat([home_h[-1], away_h[-1]], dim=1)
        combined = self.bn(combined)

        return self.head(combined)


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor.

    Uses self-attention to weight which games in history matter most.
    Cross-attention between teams could capture matchup dynamics.

    VERY prone to overfitting - uses aggressive regularization.

    Parameters: ~8K
    """

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_heads: int = 2, num_layers: int = 1, dropout: float = DROPOUT):
        super().__init__()

        self.hidden_dim = hidden_dim
        team_features = input_dim // 2

        # Project to hidden dim
        self.input_proj = nn.Linear(team_features, hidden_dim)

        # Positional encoding (learnable, short sequence)
        self.pos_encoding = nn.Parameter(torch.randn(1, SEQ_LEN, hidden_dim) * 0.1)

        # Single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention between teams
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.bn = nn.BatchNorm1d(hidden_dim * 2)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"  Transformer parameters: {total:,}")

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        home_x = x[:, :, :features // 2]
        away_x = x[:, :, features // 2:]

        # Project and add positional encoding
        home_x = self.input_proj(home_x) + self.pos_encoding
        away_x = self.input_proj(away_x) + self.pos_encoding

        # Self-attention within each team's history
        home_x = self.transformer(home_x)
        away_x = self.transformer(away_x)

        # Cross-attention: how does home team match up against away team?
        home_attn, _ = self.cross_attn(home_x, away_x, away_x)
        away_attn, _ = self.cross_attn(away_x, home_x, home_x)

        # Pool: take mean of attended sequences
        home_repr = (home_x + home_attn).mean(dim=1)
        away_repr = (away_x + away_attn).mean(dim=1)

        combined = torch.cat([home_repr, away_repr], dim=1)
        combined = self.bn(combined)

        return self.head(combined)


class HybridPredictor(nn.Module):
    """
    Hybrid model that combines sequence model with static features.

    Uses GRU for sequences + direct differential features.
    This gives the model both temporal patterns and current state.

    Parameters: ~6K
    """

    def __init__(self, input_dim: int, static_dim: int = 10,
                 hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()

        self.hidden_dim = hidden_dim

        # GRU for sequences
        self.gru = nn.GRU(
            input_size=input_dim // 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Static feature processing
        self.static_proj = nn.Linear(static_dim, hidden_dim)

        self.bn = nn.BatchNorm1d(hidden_dim * 3)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"  Hybrid parameters: {total:,}")

    def forward(self, x, static_features=None):
        batch_size, seq_len, features = x.shape

        home_x = x[:, :, :features // 2]
        away_x = x[:, :, features // 2:]

        _, home_h = self.gru(home_x)
        _, away_h = self.gru(away_x)

        if static_features is not None:
            static_h = self.static_proj(static_features)
            combined = torch.cat([home_h[-1], away_h[-1], static_h], dim=1)
        else:
            # Use last timestep features as static
            last_diff = home_x[:, -1, :] - away_x[:, -1, :]
            static_h = self.static_proj(last_diff[:, :10])  # First 10 features
            combined = torch.cat([home_h[-1], away_h[-1], static_h], dim=1)

        combined = self.bn(combined)
        return self.head(combined)


# ============================================================================
# TRAINING
# ============================================================================

def build_team_histories(conn) -> dict:
    """Build game history for each team-season."""
    log.info("Building team game histories...")

    games = pd.read_sql_query('''
        SELECT
            g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
            g.home_score, g.away_score,
            hs.field_goal_pct as home_fg, hs.three_point_pct as home_three,
            hs.free_throw_pct as home_ft, hs.total_rebounds as home_reb,
            hs.assists as home_ast, hs.steals as home_stl,
            hs.blocks as home_blk, hs.turnovers as home_tov,
            aws.field_goal_pct as away_fg, aws.three_point_pct as away_three,
            aws.free_throw_pct as away_ft, aws.total_rebounds as away_reb,
            aws.assists as away_ast, aws.steals as away_stl,
            aws.blocks as away_blk, aws.turnovers as away_tov
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        WHERE g.home_score > 0 AND g.completed = 1
        ORDER BY g.date
    ''', conn)

    histories = defaultdict(list)
    last_game_date = {}

    for _, g in games.iterrows():
        # Calculate rest days
        home_rest = 2
        away_rest = 2

        home_key = g['home_team_id']
        away_key = g['away_team_id']

        if home_key in last_game_date:
            curr = datetime.strptime(g['date'][:10], '%Y-%m-%d')
            last = datetime.strptime(last_game_date[home_key][:10], '%Y-%m-%d')
            home_rest = max(0, min((curr - last).days - 1, 5))

        if away_key in last_game_date:
            curr = datetime.strptime(g['date'][:10], '%Y-%m-%d')
            last = datetime.strptime(last_game_date[away_key][:10], '%Y-%m-%d')
            away_rest = max(0, min((curr - last).days - 1, 5))

        # Add to home team history
        histories[(g['home_team_id'], g['season'])].append({
            'date': g['date'],
            'points_for': g['home_score'],
            'points_against': g['away_score'],
            'fg_pct': g['home_fg'] if pd.notna(g['home_fg']) else 46,
            'three_pct': g['home_three'] if pd.notna(g['home_three']) else 36,
            'ft_pct': g['home_ft'] if pd.notna(g['home_ft']) else 78,
            'rebounds': g['home_reb'] if pd.notna(g['home_reb']) else 44,
            'assists': g['home_ast'] if pd.notna(g['home_ast']) else 25,
            'steals': g['home_stl'] if pd.notna(g['home_stl']) else 7.5,
            'blocks': g['home_blk'] if pd.notna(g['home_blk']) else 5,
            'turnovers': g['home_tov'] if pd.notna(g['home_tov']) else 14,
            'was_home': 1,
            'won': 1 if g['home_score'] > g['away_score'] else 0,
            'rest_days': home_rest
        })

        # Add to away team history
        histories[(g['away_team_id'], g['season'])].append({
            'date': g['date'],
            'points_for': g['away_score'],
            'points_against': g['home_score'],
            'fg_pct': g['away_fg'] if pd.notna(g['away_fg']) else 46,
            'three_pct': g['away_three'] if pd.notna(g['away_three']) else 36,
            'ft_pct': g['away_ft'] if pd.notna(g['away_ft']) else 78,
            'rebounds': g['away_reb'] if pd.notna(g['away_reb']) else 44,
            'assists': g['away_ast'] if pd.notna(g['away_ast']) else 25,
            'steals': g['away_stl'] if pd.notna(g['away_stl']) else 7.5,
            'blocks': g['away_blk'] if pd.notna(g['away_blk']) else 5,
            'turnovers': g['away_tov'] if pd.notna(g['away_tov']) else 14,
            'was_home': 0,
            'won': 1 if g['away_score'] > g['home_score'] else 0,
            'rest_days': away_rest
        })

        last_game_date[home_key] = g['date']
        last_game_date[away_key] = g['date']

    log.info(f"  Built histories for {len(histories)} team-seasons")
    return dict(histories)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                model_name: str) -> dict:
    """Train a model with early stopping."""

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for sequences, targets, vegas in train_loader:
            sequences = sequences.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets, vegas in val_loader:
                sequences = sequences.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = model(sequences)
                val_loss += criterion(predictions, targets).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            log.info(f"  Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 20 == 0:
            log.info(f"  Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}")

    # Restore best model
    model.load_state_dict(best_state)

    return {'best_val_loss': best_val_loss, 'epochs': epoch + 1}


def evaluate_model(model: nn.Module, test_loader: DataLoader,
                   model_weight: float = 0.9) -> dict:
    """Evaluate model on test set."""

    model.eval()

    all_preds = []
    all_targets = []
    all_vegas = []

    with torch.no_grad():
        for sequences, targets, vegas in test_loader:
            sequences = sequences.to(DEVICE)
            preds = model(sequences).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())
            all_vegas.append(vegas.numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    vegas = np.vstack(all_vegas)

    # Model-only metrics
    spread_mae = np.abs(preds[:, 0] - targets[:, 0]).mean()
    total_mae = np.abs(preds[:, 1] - targets[:, 1]).mean()
    winner_acc = ((preds[:, 0] < 0) == (targets[:, 0] < 0)).mean()

    # Blended metrics
    blended_spread = model_weight * preds[:, 0] + (1 - model_weight) * vegas[:, 0]
    blended_total = 0.6 * preds[:, 1] + 0.4 * vegas[:, 1]

    blend_spread_mae = np.abs(blended_spread - targets[:, 0]).mean()
    blend_total_mae = np.abs(blended_total - targets[:, 1]).mean()
    blend_winner_acc = ((blended_spread < 0) == (targets[:, 0] < 0)).mean()

    # Vegas metrics
    vegas_spread_mae = np.abs(vegas[:, 0] - targets[:, 0]).mean()
    vegas_total_mae = np.abs(vegas[:, 1] - targets[:, 1]).mean()
    vegas_winner_acc = ((vegas[:, 0] < 0) == (targets[:, 0] < 0)).mean()

    return {
        'model_spread_mae': spread_mae,
        'model_total_mae': total_mae,
        'model_winner_acc': winner_acc,
        'blend_spread_mae': blend_spread_mae,
        'blend_total_mae': blend_total_mae,
        'blend_winner_acc': blend_winner_acc,
        'vegas_spread_mae': vegas_spread_mae,
        'vegas_total_mae': vegas_total_mae,
        'vegas_winner_acc': vegas_winner_acc,
    }


def main():
    log.info("=" * 70)
    log.info("NBA DEEP LEARNING MODEL COMPARISON")
    log.info("=" * 70)
    log.info(f"Device: {DEVICE}")
    log.info(f"Sequence length: {SEQ_LEN}")
    log.info(f"Hidden dim: {HIDDEN_DIM}")
    log.info(f"Dropout: {DROPOUT}")
    log.info(f"Weight decay: {WEIGHT_DECAY}")

    conn = sqlite3.connect(str(DB_PATH))

    # Build team histories
    team_histories = build_team_histories(conn)

    # Get games with Vegas odds
    games = pd.read_sql_query('''
        SELECT
            g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
            g.home_score, g.away_score,
            o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.home_score > 0 AND g.completed = 1
            AND o.latest_spread IS NOT NULL
        ORDER BY g.date
    ''', conn)
    conn.close()

    log.info(f"\nTotal games with Vegas odds: {len(games)}")

    # Split by season
    seasons = sorted(games['season'].unique())
    if len(seasons) >= 3:
        test_season = seasons[-2]
        train_seasons = [s for s in seasons if s < test_season]
    else:
        test_season = seasons[-1]
        train_seasons = seasons[:-1]

    train_games = games[games['season'].isin(train_seasons)]
    test_games = games[games['season'] == test_season]

    log.info(f"Train seasons: {train_seasons}")
    log.info(f"Test season: {test_season}")

    # Create datasets
    log.info("\nCreating datasets...")
    train_dataset = NBASequenceDataset(train_games, team_histories, fit_scaler=True)
    test_dataset = NBASequenceDataset(test_games, team_histories, scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Input dimension
    input_dim = train_dataset.sequences.shape[2]

    # Models to test
    models = {
        'GRU': GRUPredictor(input_dim),
        'LSTM': LSTMPredictor(input_dim),
        'Transformer': TransformerPredictor(input_dim),
        'Hybrid': HybridPredictor(input_dim),
    }

    results = {}

    for name, model in models.items():
        log.info(f"\n{'=' * 70}")
        log.info(f"TRAINING {name}")
        log.info('=' * 70)

        # Reset model
        model = models[name].__class__(input_dim)

        # Train
        train_info = train_model(model, train_loader, test_loader, name)

        # Evaluate
        metrics = evaluate_model(model, test_loader)
        results[name] = {**train_info, **metrics}

    # Print comparison
    log.info("\n" + "=" * 70)
    log.info("RESULTS COMPARISON")
    log.info("=" * 70)

    log.info(f"\n{'Model':<15} {'Spread MAE':<12} {'Total MAE':<12} {'Winner Acc':<12}")
    log.info("-" * 55)

    for name, r in results.items():
        log.info(f"{name:<15} {r['model_spread_mae']:.2f}         "
                 f"{r['model_total_mae']:.2f}         {r['model_winner_acc']*100:.1f}%")

    log.info(f"\n{'Vegas':<15} {results['GRU']['vegas_spread_mae']:.2f}         "
             f"{results['GRU']['vegas_total_mae']:.2f}         "
             f"{results['GRU']['vegas_winner_acc']*100:.1f}%")

    log.info("\n--- WITH VEGAS BLEND ---")
    log.info(f"\n{'Model':<15} {'Spread MAE':<12} {'Total MAE':<12} {'Winner Acc':<12}")
    log.info("-" * 55)

    for name, r in results.items():
        log.info(f"{name:<15} {r['blend_spread_mae']:.2f}         "
                 f"{r['blend_total_mae']:.2f}         {r['blend_winner_acc']*100:.1f}%")

    # Best model
    best_model = min(results.items(), key=lambda x: x[1]['blend_spread_mae'])
    log.info(f"\nBest model (by blended spread MAE): {best_model[0]}")
    log.info(f"  Spread MAE: {best_model[1]['blend_spread_mae']:.2f} "
             f"(vs Vegas {best_model[1]['vegas_spread_mae']:.2f})")
    log.info(f"  Total MAE: {best_model[1]['blend_total_mae']:.2f} "
             f"(vs Vegas {best_model[1]['vegas_total_mae']:.2f})")


if __name__ == '__main__':
    main()
