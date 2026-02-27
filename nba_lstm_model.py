"""
NBA LSTM Model - Learns when Ridge V2 is right/wrong using sequence data.

Architecture:
    Input: Ridge V2 prediction + Vegas line + team game sequences (last 10 games)
    LSTM: Processes team form trajectories
    Dense: Learns adjustment patterns
    Output: Adjusted spread, confidence, fade signal

Usage:
    python nba_lstm_model.py          # Train and evaluate
    python nba_lstm_model.py --predict # Generate predictions for upcoming games
"""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# GPU setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SEQUENCE_LENGTH = 10  # Last 10 games per team
DB_PATH = Path('nba_games.db')
MODEL_PATH = Path('models/nba_lstm_v1.pt')
SCALER_PATH = Path('models/nba_lstm_v1_scaler.pkl')


@dataclass
class GameSequence:
    """Sequence features for a single game."""
    game_id: int
    season: int
    home_team_id: int
    away_team_id: int
    home_sequence: np.ndarray  # (SEQUENCE_LENGTH, seq_features)
    away_sequence: np.ndarray  # (SEQUENCE_LENGTH, seq_features)
    static_features: np.ndarray  # Ridge prediction, Vegas line, etc.
    actual_margin: float
    vegas_spread: float
    ridge_spread: float


class NBASequenceDataset(Dataset):
    """PyTorch dataset for NBA game sequences."""

    def __init__(self, sequences: list[GameSequence], scaler: StandardScaler = None, fit_scaler: bool = False):
        self.sequences = sequences

        # Extract all static features for scaling
        static = np.array([s.static_features for s in sequences])

        if fit_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(static)
        else:
            self.scaler = scaler

        # Scale static features
        self.static_scaled = self.scaler.transform(static)

        # Targets: actual margin (for regression) and ATS result (for classification)
        self.margins = np.array([s.actual_margin for s in sequences])
        self.vegas_spreads = np.array([s.vegas_spread for s in sequences])
        self.ridge_spreads = np.array([s.ridge_spread for s in sequences])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        return {
            'home_seq': torch.FloatTensor(seq.home_sequence),
            'away_seq': torch.FloatTensor(seq.away_sequence),
            'static': torch.FloatTensor(self.static_scaled[idx]),
            'margin': torch.FloatTensor([self.margins[idx]]),
            'vegas_spread': torch.FloatTensor([self.vegas_spreads[idx]]),
            'ridge_spread': torch.FloatTensor([self.ridge_spreads[idx]]),
        }


class NBALSTM(nn.Module):
    """LSTM model for NBA spread prediction."""

    def __init__(
        self,
        seq_input_size: int,
        static_input_size: int,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.seq_input_size = seq_input_size
        self.static_input_size = static_input_size
        self.lstm_hidden = lstm_hidden

        # Separate LSTMs for home and away team sequences
        self.home_lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )

        self.away_lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )

        # Combine LSTM outputs with static features
        combined_size = lstm_hidden * 2 + static_input_size  # home + away LSTM + static

        # Dense layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.spread_head = nn.Linear(64, 1)  # Adjusted spread prediction
        self.confidence_head = nn.Linear(64, 1)  # Confidence score (0-1)
        self.fade_head = nn.Linear(64, 1)  # Fade signal (should we fade Ridge?)

        self.sigmoid = nn.Sigmoid()

    def forward(self, home_seq, away_seq, static):
        # Process sequences through LSTMs
        # Take final hidden state
        _, (home_hidden, _) = self.home_lstm(home_seq)
        _, (away_hidden, _) = self.away_lstm(away_seq)

        # Use last layer's hidden state
        home_out = home_hidden[-1]  # (batch, lstm_hidden)
        away_out = away_hidden[-1]  # (batch, lstm_hidden)

        # Concatenate LSTM outputs with static features
        combined = torch.cat([home_out, away_out, static], dim=1)

        # Dense layers
        features = self.fc(combined)

        # Output heads
        spread_adj = self.spread_head(features)  # Adjustment to Ridge spread
        confidence = self.sigmoid(self.confidence_head(features))
        fade = self.sigmoid(self.fade_head(features))

        return spread_adj, confidence, fade


class NBALSTMPredictor:
    """Main class for training and predicting with LSTM model."""

    def __init__(self):
        self.model: NBALSTM | None = None
        self.scaler: StandardScaler | None = None
        self.ridge_model = None
        self.seq_features = 8  # Features per game in sequence
        self.static_features = 12  # Static features per matchup

    def _get_team_sequence(
        self,
        cursor: sqlite3.Cursor,
        team_id: int,
        before_game_id: int,
        season: int,
    ) -> np.ndarray:
        """Get last N games for a team as sequence features."""

        cursor.execute('''
            SELECT
                g.game_id,
                CASE WHEN g.home_team_id = ? THEN g.home_score ELSE g.away_score END as team_score,
                CASE WHEN g.home_team_id = ? THEN g.away_score ELSE g.home_score END as opp_score,
                CASE WHEN g.home_team_id = ? THEN 1 ELSE 0 END as is_home,
                tgs.field_goal_pct,
                tgs.three_point_pct,
                tgs.total_rebounds,
                tgs.turnovers
            FROM games g
            JOIN team_game_stats tgs ON g.game_id = tgs.game_id AND tgs.team_id = ?
            WHERE g.completed = 1
                AND g.game_id < ?
                AND (g.home_team_id = ? OR g.away_team_id = ?)
                AND g.season >= ? - 1
            ORDER BY g.game_date_eastern DESC
            LIMIT ?
        ''', (team_id, team_id, team_id, team_id, before_game_id, team_id, team_id, season, SEQUENCE_LENGTH))

        rows = cursor.fetchall()

        if len(rows) < 3:  # Need at least 3 games
            return None

        # Build sequence (most recent first, then reverse for chronological order)
        sequence = []
        for row in rows:
            game_id, team_score, opp_score, is_home, fg_pct, three_pct, reb, tov = row
            margin = team_score - opp_score

            features = [
                margin / 20.0,  # Normalized margin
                (team_score - 110) / 15.0,  # Normalized score (centered on ~110)
                is_home,
                (fg_pct or 0.45) - 0.45,  # FG% deviation from average
                (three_pct or 0.35) - 0.35,  # 3P% deviation
                (reb or 44) / 50.0,  # Normalized rebounds
                (tov or 14) / 20.0,  # Normalized turnovers
                1.0 if margin > 0 else 0.0,  # Win indicator
            ]
            sequence.append(features)

        # Pad if needed
        while len(sequence) < SEQUENCE_LENGTH:
            sequence.append([0.0] * self.seq_features)

        # Reverse to chronological order (oldest to newest)
        sequence = sequence[::-1]

        return np.array(sequence, dtype=np.float32)

    def _get_ridge_prediction(
        self,
        home_id: int,
        away_id: int,
        season: int,
        game_date: str,
        game_id: int,
        vegas_spread: float,
        vegas_total: float,
    ) -> dict | None:
        """Get Ridge V2 prediction for a game."""

        if self.ridge_model is None:
            try:
                from nba_ridge_v2 import NBARidgeV2
                self.ridge_model = NBARidgeV2.load()
            except Exception as e:
                logger.warning(f"Could not load Ridge V2: {e}")
                return None

        try:
            return self.ridge_model.predict(
                home_id, away_id, season, game_date, game_id, vegas_spread, vegas_total
            )
        except Exception as e:
            return None

    def build_sequences(
        self,
        seasons: list[int],
        cursor: sqlite3.Cursor,
    ) -> list[GameSequence]:
        """Build training sequences for given seasons."""

        sequences = []

        # Get all completed games with odds
        placeholders = ','.join('?' * len(seasons))
        cursor.execute(f'''
            SELECT
                g.game_id, g.season, g.home_team_id, g.away_team_id,
                g.game_date_eastern, g.home_score, g.away_score,
                o.latest_spread, o.latest_total
            FROM games g
            LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.completed = 1
                AND g.season IN ({placeholders})
                AND o.latest_spread IS NOT NULL
            ORDER BY g.game_date_eastern
        ''', seasons)

        games = cursor.fetchall()
        logger.info(f"Processing {len(games)} games from seasons {seasons}")

        for i, game in enumerate(games):
            game_id, season, home_id, away_id, game_date, home_score, away_score, vegas_spread, vegas_total = game

            # Get team sequences
            home_seq = self._get_team_sequence(cursor, home_id, game_id, season)
            away_seq = self._get_team_sequence(cursor, away_id, game_id, season)

            if home_seq is None or away_seq is None:
                continue

            # Get Ridge V2 prediction
            ridge_pred = self._get_ridge_prediction(
                home_id, away_id, season, game_date, game_id, vegas_spread, vegas_total
            )

            if ridge_pred is None:
                continue

            ridge_spread = ridge_pred['predicted_spread']

            # Build static features
            edge = ridge_spread - vegas_spread
            static = np.array([
                ridge_spread / 15.0,  # Normalized ridge spread
                vegas_spread / 15.0,  # Normalized vegas spread
                edge / 10.0,  # Normalized edge
                ridge_pred.get('dynamic_hca', 2.0) / 5.0,  # HCA
                ridge_pred.get('srs_diff', 0) / 10.0,  # SRS difference
                1.0 if ridge_pred.get('is_road_fav', False) else 0.0,  # Road fav indicator
                1.0 if vegas_spread > 0 else 0.0,  # Vegas road fav
                1.0 if abs(vegas_spread) > 7 else 0.0,  # Big spread
                1.0 if abs(edge) > 5 else 0.0,  # Big edge
                home_seq[-1, 0],  # Home team recent margin (last game)
                away_seq[-1, 0],  # Away team recent margin
                (home_seq[:, 7].sum() - away_seq[:, 7].sum()) / 10.0,  # Win differential
            ], dtype=np.float32)

            actual_margin = home_score - away_score

            sequences.append(GameSequence(
                game_id=game_id,
                season=season,
                home_team_id=home_id,
                away_team_id=away_id,
                home_sequence=home_seq,
                away_sequence=away_seq,
                static_features=static,
                actual_margin=actual_margin,
                vegas_spread=vegas_spread,
                ridge_spread=ridge_spread,
            ))

            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{len(games)} games, {len(sequences)} valid sequences")

        logger.info(f"Built {len(sequences)} sequences")
        return sequences

    def train(
        self,
        train_sequences: list[GameSequence],
        val_sequences: list[GameSequence],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 15,
    ):
        """Train the LSTM model."""

        # Create datasets
        train_dataset = NBASequenceDataset(train_sequences, fit_scaler=True)
        self.scaler = train_dataset.scaler

        val_dataset = NBASequenceDataset(val_sequences, scaler=self.scaler)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        self.model = NBALSTM(
            seq_input_size=self.seq_features,
            static_input_size=self.static_features,
            lstm_hidden=64,
            lstm_layers=2,
            dropout=0.3,
        ).to(DEVICE)

        # Loss functions
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        logger.info(f"Training on {len(train_sequences)} games, validating on {len(val_sequences)}")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                home_seq = batch['home_seq'].to(DEVICE)
                away_seq = batch['away_seq'].to(DEVICE)
                static = batch['static'].to(DEVICE)
                margin = batch['margin'].to(DEVICE)
                vegas = batch['vegas_spread'].to(DEVICE)
                ridge = batch['ridge_spread'].to(DEVICE)

                optimizer.zero_grad()

                spread_adj, confidence, fade = self.model(home_seq, away_seq, static)

                # Compute targets
                # 1. Spread adjustment target: how much to adjust Ridge spread
                target_spread = margin  # We want to predict actual margin
                ridge_error = margin - ridge  # How wrong was Ridge?

                # 2. ATS target: did the pick cover?
                edge = ridge - vegas
                pick_away = (edge > 0).float()
                away_covered = ((-margin) > vegas).float()
                home_covered = (margin > (-vegas)).float()
                ats_result = pick_away * away_covered + (1 - pick_away) * home_covered

                # Combined loss
                # Primary: predict the adjustment needed
                loss_spread = mse_loss(spread_adj, ridge_error)

                # Secondary: predict when to fade
                should_fade = ((edge > 0) & (away_covered < 0.5)) | ((edge < 0) & (home_covered < 0.5))
                loss_fade = bce_loss(fade, should_fade.float())

                # Confidence should correlate with correct picks
                loss_conf = bce_loss(confidence, ats_result)

                loss = loss_spread + 0.3 * loss_fade + 0.3 * loss_conf

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_ats_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    home_seq = batch['home_seq'].to(DEVICE)
                    away_seq = batch['away_seq'].to(DEVICE)
                    static = batch['static'].to(DEVICE)
                    margin = batch['margin'].to(DEVICE)
                    vegas = batch['vegas_spread'].to(DEVICE)
                    ridge = batch['ridge_spread'].to(DEVICE)

                    spread_adj, confidence, fade = self.model(home_seq, away_seq, static)

                    ridge_error = margin - ridge
                    loss_spread = mse_loss(spread_adj, ridge_error)
                    val_loss += loss_spread.item()

                    # Compute ATS accuracy with adjusted spread
                    adjusted_spread = ridge + spread_adj
                    edge = adjusted_spread - vegas

                    for i in range(len(margin)):
                        m = margin[i].item()
                        v = vegas[i].item()
                        e = edge[i].item()

                        if e > 0:  # Pick away
                            if (-m) > v:
                                val_ats_correct += 1
                        else:  # Pick home
                            if m > (-v):
                                val_ats_correct += 1
                        val_total += 1

            val_loss /= len(val_loader)
            val_ats = val_ats_correct / val_total if val_total > 0 else 0

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val ATS: {val_ats:.1%} ({val_ats_correct}/{val_total})"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    def evaluate(self, sequences: list[GameSequence]) -> dict:
        """Evaluate model on a set of sequences."""

        if self.model is None:
            raise ValueError("Model not trained")

        dataset = NBASequenceDataset(sequences, scaler=self.scaler)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        self.model.eval()

        results = {
            'ridge_ats': [],
            'lstm_ats': [],
            'edges': [],
            'confidences': [],
            'fades': [],
        }

        with torch.no_grad():
            for batch in loader:
                home_seq = batch['home_seq'].to(DEVICE)
                away_seq = batch['away_seq'].to(DEVICE)
                static = batch['static'].to(DEVICE)
                margin = batch['margin'].cpu().numpy()
                vegas = batch['vegas_spread'].cpu().numpy()
                ridge = batch['ridge_spread'].cpu().numpy()

                spread_adj, confidence, fade = self.model(home_seq, away_seq, static)
                spread_adj = spread_adj.cpu().numpy()
                confidence = confidence.cpu().numpy()
                fade = fade.cpu().numpy()

                for i in range(len(margin)):
                    m = margin[i, 0]
                    v = vegas[i, 0]
                    r = ridge[i, 0]
                    adj = spread_adj[i, 0]
                    conf = confidence[i, 0]
                    f = fade[i, 0]

                    ridge_edge = r - v
                    lstm_spread = r + adj
                    lstm_edge = lstm_spread - v

                    # Ridge ATS
                    if ridge_edge > 0:
                        ridge_covers = (-m) > v
                    else:
                        ridge_covers = m > (-v)

                    # LSTM ATS
                    if lstm_edge > 0:
                        lstm_covers = (-m) > v
                    else:
                        lstm_covers = m > (-v)

                    results['ridge_ats'].append(ridge_covers)
                    results['lstm_ats'].append(lstm_covers)
                    results['edges'].append(abs(lstm_edge))
                    results['confidences'].append(conf)
                    results['fades'].append(f)

        return results

    def save(self):
        """Save model and scaler."""

        MODEL_PATH.parent.mkdir(exist_ok=True)

        torch.save({
            'model_state': self.model.state_dict(),
            'seq_features': self.seq_features,
            'static_features': self.static_features,
        }, MODEL_PATH)

        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)

        logger.info(f"Saved model to {MODEL_PATH}")

    @classmethod
    def load(cls) -> 'NBALSTMPredictor':
        """Load saved model."""

        predictor = cls()

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        predictor.seq_features = checkpoint['seq_features']
        predictor.static_features = checkpoint['static_features']

        predictor.model = NBALSTM(
            seq_input_size=predictor.seq_features,
            static_input_size=predictor.static_features,
        ).to(DEVICE)
        predictor.model.load_state_dict(checkpoint['model_state'])
        predictor.model.eval()

        with open(SCALER_PATH, 'rb') as f:
            predictor.scaler = pickle.load(f)

        return predictor

    def predict(
        self,
        home_id: int,
        away_id: int,
        season: int,
        game_date: str,
        game_id: int,
        vegas_spread: float,
        vegas_total: float,
    ) -> dict | None:
        """Predict for a single game."""

        if self.model is None:
            raise ValueError("Model not loaded")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get sequences
        home_seq = self._get_team_sequence(cursor, home_id, game_id + 1, season)  # +1 to include current
        away_seq = self._get_team_sequence(cursor, away_id, game_id + 1, season)

        if home_seq is None or away_seq is None:
            conn.close()
            return None

        # Get Ridge prediction
        ridge_pred = self._get_ridge_prediction(
            home_id, away_id, season, game_date, game_id, vegas_spread, vegas_total
        )

        if ridge_pred is None:
            conn.close()
            return None

        ridge_spread = ridge_pred['predicted_spread']
        edge = ridge_spread - vegas_spread

        # Build static features
        static = np.array([
            ridge_spread / 15.0,
            vegas_spread / 15.0,
            edge / 10.0,
            ridge_pred.get('dynamic_hca', 2.0) / 5.0,
            ridge_pred.get('srs_diff', 0) / 10.0,
            1.0 if ridge_pred.get('is_road_fav', False) else 0.0,
            1.0 if vegas_spread > 0 else 0.0,
            1.0 if abs(vegas_spread) > 7 else 0.0,
            1.0 if abs(edge) > 5 else 0.0,
            home_seq[-1, 0],
            away_seq[-1, 0],
            (home_seq[:, 7].sum() - away_seq[:, 7].sum()) / 10.0,
        ], dtype=np.float32)

        conn.close()

        # Scale and predict
        static_scaled = self.scaler.transform(static.reshape(1, -1))

        self.model.eval()
        with torch.no_grad():
            home_t = torch.FloatTensor(home_seq).unsqueeze(0).to(DEVICE)
            away_t = torch.FloatTensor(away_seq).unsqueeze(0).to(DEVICE)
            static_t = torch.FloatTensor(static_scaled).to(DEVICE)

            spread_adj, confidence, fade = self.model(home_t, away_t, static_t)

        adjusted_spread = ridge_spread + spread_adj.item()

        return {
            'ridge_spread': ridge_spread,
            'lstm_adjustment': spread_adj.item(),
            'lstm_spread': adjusted_spread,
            'confidence': confidence.item(),
            'fade_signal': fade.item(),
            'vegas_spread': vegas_spread,
            'edge': adjusted_spread - vegas_spread,
        }


def main():
    """Train and evaluate the LSTM model."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    predictor = NBALSTMPredictor()

    # Build sequences
    logger.info("Building training sequences (2024-2025)...")
    train_sequences = predictor.build_sequences([2024, 2025], cursor)

    logger.info("Building validation sequences (2026)...")
    val_sequences = predictor.build_sequences([2026], cursor)

    conn.close()

    if len(train_sequences) < 100 or len(val_sequences) < 50:
        logger.error("Not enough data for training")
        return

    # Train
    logger.info("Training LSTM model...")
    predictor.train(train_sequences, val_sequences, epochs=100, patience=15)

    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON 2026 SEASON")
    logger.info("=" * 60)

    results = predictor.evaluate(val_sequences)

    ridge_ats = results['ridge_ats']
    lstm_ats = results['lstm_ats']
    edges = results['edges']
    confidences = results['confidences']

    # Overall comparison
    ridge_wins = sum(ridge_ats)
    lstm_wins = sum(lstm_ats)
    total = len(ridge_ats)

    print(f"\nOverall ATS:")
    print(f"  Ridge V2: {ridge_wins}/{total} = {ridge_wins/total:.1%}")
    print(f"  LSTM:     {lstm_wins}/{total} = {lstm_wins/total:.1%}")

    # By edge threshold
    print(f"\nBy Edge Threshold:")
    for threshold in [3, 5, 7]:
        ridge_high = sum(1 for i, e in enumerate(edges) if e >= threshold and ridge_ats[i])
        lstm_high = sum(1 for i, e in enumerate(edges) if e >= threshold and lstm_ats[i])
        total_high = sum(1 for e in edges if e >= threshold)

        if total_high > 0:
            print(f"  {threshold}+ pts: Ridge {ridge_high}/{total_high} = {ridge_high/total_high:.1%}, "
                  f"LSTM {lstm_high}/{total_high} = {lstm_high/total_high:.1%}")

    # By confidence
    print(f"\nBy LSTM Confidence:")
    for conf_thresh in [0.5, 0.6, 0.7]:
        high_conf = [(lstm_ats[i], edges[i]) for i, c in enumerate(confidences) if c >= conf_thresh]
        if high_conf:
            wins = sum(1 for ats, _ in high_conf if ats)
            print(f"  Conf >= {conf_thresh}: {wins}/{len(high_conf)} = {wins/len(high_conf):.1%}")

    # Save model
    predictor.save()
    logger.info(f"\nModel saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()
