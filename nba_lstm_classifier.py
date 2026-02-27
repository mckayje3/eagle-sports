"""
NBA LSTM Edge Classifier - Predicts when Ridge V2 picks will cover.

Instead of predicting spread adjustments, this model directly classifies
whether Ridge V2's pick will cover ATS. This better aligns training
with the actual betting objective.

Architecture:
    Input: Team sequences + Ridge features + Vegas line
    LSTM: Processes team form
    Output: P(Ridge pick covers)

Usage:
    python nba_lstm_classifier.py
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
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
SEQUENCE_LENGTH = 10
DB_PATH = Path('nba_games.db')
MODEL_PATH = Path('models/nba_lstm_classifier.pt')
SCALER_PATH = Path('models/nba_lstm_classifier_scaler.pkl')


class SimpleLSTMClassifier(nn.Module):
    """Simpler LSTM that directly predicts ATS outcome."""

    def __init__(self, seq_features: int, static_features: int, hidden: int = 32):
        super().__init__()

        # Single bidirectional LSTM for combined sequence
        self.lstm = nn.LSTM(
            input_size=seq_features * 2,  # Home + away concatenated
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Simple classifier
        combined_size = hidden * 2 + static_features  # Bidirectional + static
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, home_seq, away_seq, static):
        # Concatenate home and away sequences along feature dimension
        combined_seq = torch.cat([home_seq, away_seq], dim=2)  # (batch, seq_len, features*2)

        # LSTM
        _, (hidden, _) = self.lstm(combined_seq)

        # Concatenate forward and backward hidden states
        lstm_out = torch.cat([hidden[0], hidden[1]], dim=1)  # (batch, hidden*2)

        # Combine with static features
        combined = torch.cat([lstm_out, static], dim=1)

        # Classify
        return self.classifier(combined)


class NBALSTMClassifier:
    """LSTM classifier for NBA edge detection."""

    def __init__(self):
        self.model: SimpleLSTMClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.ridge_model = None
        self.seq_features = 6  # Simplified sequence features
        self.static_features = 8  # Simplified static features

    def _get_team_sequence(
        self,
        cursor: sqlite3.Cursor,
        team_id: int,
        before_game_id: int,
        season: int,
    ) -> np.ndarray | None:
        """Get simplified sequence for a team."""

        cursor.execute('''
            SELECT
                g.game_id,
                CASE WHEN g.home_team_id = ? THEN g.home_score ELSE g.away_score END as team_score,
                CASE WHEN g.home_team_id = ? THEN g.away_score ELSE g.home_score END as opp_score,
                CASE WHEN g.home_team_id = ? THEN 1 ELSE 0 END as is_home
            FROM games g
            WHERE g.completed = 1
                AND g.game_id < ?
                AND (g.home_team_id = ? OR g.away_team_id = ?)
                AND g.season >= ? - 1
            ORDER BY g.game_date_eastern DESC
            LIMIT ?
        ''', (team_id, team_id, team_id, before_game_id, team_id, team_id, season, SEQUENCE_LENGTH))

        rows = cursor.fetchall()

        if len(rows) < 5:
            return None

        sequence = []
        for row in rows:
            game_id, team_score, opp_score, is_home = row
            margin = team_score - opp_score

            features = [
                margin / 15.0,  # Normalized margin
                (team_score - 112) / 12.0,  # Centered score
                (opp_score - 112) / 12.0,  # Opponent score
                is_home,
                1.0 if margin > 0 else 0.0,  # Win
                1.0 if margin > 10 else (0.0 if margin < -10 else 0.5),  # Blowout indicator
            ]
            sequence.append(features)

        # Pad
        while len(sequence) < SEQUENCE_LENGTH:
            sequence.append([0.0] * self.seq_features)

        # Reverse to chronological
        return np.array(sequence[::-1], dtype=np.float32)

    def _get_ridge_prediction(self, home_id, away_id, season, game_date, game_id, vegas_spread, vegas_total):
        """Get Ridge V2 prediction."""
        if self.ridge_model is None:
            try:
                from nba_ridge_v2 import NBARidgeV2
                self.ridge_model = NBARidgeV2.load()
            except Exception as e:
                logger.warning(f"Could not load Ridge V2: {e}")
                return None

        try:
            return self.ridge_model.predict(home_id, away_id, season, game_date, game_id, vegas_spread, vegas_total)
        except:
            return None

    def build_dataset(self, seasons: list[int], cursor: sqlite3.Cursor) -> tuple:
        """Build training data."""

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

        home_seqs = []
        away_seqs = []
        static_features = []
        targets = []
        edges = []

        for i, game in enumerate(games):
            game_id, season, home_id, away_id, game_date, home_score, away_score, vegas_spread, vegas_total = game

            home_seq = self._get_team_sequence(cursor, home_id, game_id, season)
            away_seq = self._get_team_sequence(cursor, away_id, game_id, season)

            if home_seq is None or away_seq is None:
                continue

            ridge_pred = self._get_ridge_prediction(
                home_id, away_id, season, game_date, game_id, vegas_spread, vegas_total
            )
            if ridge_pred is None:
                continue

            ridge_spread = ridge_pred['predicted_spread']
            edge = ridge_spread - vegas_spread
            actual_margin = home_score - away_score

            # Determine if Ridge pick covers
            if edge > 0:  # Ridge picks away
                covers = (-actual_margin) > vegas_spread
            else:  # Ridge picks home
                covers = actual_margin > (-vegas_spread)

            # Static features
            static = np.array([
                edge / 10.0,  # Normalized edge
                abs(edge) / 10.0,  # Edge magnitude
                vegas_spread / 15.0,  # Vegas spread
                1.0 if vegas_spread > 0 else 0.0,  # Vegas road fav
                1.0 if ridge_pred.get('is_road_fav', False) else 0.0,  # Model road fav
                ridge_pred.get('srs_diff', 0) / 10.0,  # SRS diff
                home_seq[-1, 0],  # Home recent form
                away_seq[-1, 0],  # Away recent form
            ], dtype=np.float32)

            home_seqs.append(home_seq)
            away_seqs.append(away_seq)
            static_features.append(static)
            targets.append(float(covers))
            edges.append(abs(edge))

            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{len(games)} games")

        return (
            np.array(home_seqs),
            np.array(away_seqs),
            np.array(static_features),
            np.array(targets),
            np.array(edges),
        )

    def train(
        self,
        train_data: tuple,
        val_data: tuple,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 20,
    ):
        """Train the classifier."""

        train_home, train_away, train_static, train_y, train_edges = train_data
        val_home, val_away, val_static, val_y, val_edges = val_data

        # Scale static features
        self.scaler = StandardScaler()
        train_static_scaled = self.scaler.fit_transform(train_static)
        val_static_scaled = self.scaler.transform(val_static)

        # Convert to tensors
        train_home_t = torch.FloatTensor(train_home)
        train_away_t = torch.FloatTensor(train_away)
        train_static_t = torch.FloatTensor(train_static_scaled)
        train_y_t = torch.FloatTensor(train_y).unsqueeze(1)

        val_home_t = torch.FloatTensor(val_home)
        val_away_t = torch.FloatTensor(val_away)
        val_static_t = torch.FloatTensor(val_static_scaled)
        val_y_t = torch.FloatTensor(val_y).unsqueeze(1)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_home_t, train_away_t, train_static_t, train_y_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        self.model = SimpleLSTMClassifier(
            seq_features=self.seq_features,
            static_features=self.static_features,
            hidden=32,
        ).to(DEVICE)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

        best_val_acc = 0
        best_state = None
        patience_counter = 0

        logger.info(f"Training on {len(train_y)} samples, validating on {len(val_y)}")
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Train class balance: {train_y.mean():.1%} covers")

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for batch_home, batch_away, batch_static, batch_y in train_loader:
                batch_home = batch_home.to(DEVICE)
                batch_away = batch_away.to(DEVICE)
                batch_static = batch_static.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad()
                pred = self.model(batch_home, batch_away, batch_static)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_home_d = val_home_t.to(DEVICE)
                val_away_d = val_away_t.to(DEVICE)
                val_static_d = val_static_t.to(DEVICE)

                val_pred = self.model(val_home_d, val_away_d, val_static_d).cpu().numpy().flatten()

            # Compute accuracy at different thresholds
            val_acc = ((val_pred > 0.5) == val_y).mean()

            # Accuracy on high-edge games
            high_edge_mask = val_edges >= 5
            if high_edge_mask.sum() > 0:
                high_edge_acc = ((val_pred[high_edge_mask] > 0.5) == val_y[high_edge_mask]).mean()
            else:
                high_edge_acc = 0

            scheduler.step(-val_acc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, "
                    f"Val Acc: {val_acc:.1%}, High Edge: {high_edge_acc:.1%}"
                )

            # Early stopping based on high-edge accuracy
            if high_edge_acc > best_val_acc:
                best_val_acc = high_edge_acc
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        logger.info(f"Best high-edge accuracy: {best_val_acc:.1%}")

    def evaluate(self, data: tuple) -> dict:
        """Evaluate model performance."""

        home_seqs, away_seqs, static, targets, edges = data
        static_scaled = self.scaler.transform(static)

        self.model.eval()
        with torch.no_grad():
            home_t = torch.FloatTensor(home_seqs).to(DEVICE)
            away_t = torch.FloatTensor(away_seqs).to(DEVICE)
            static_t = torch.FloatTensor(static_scaled).to(DEVICE)

            probs = self.model(home_t, away_t, static_t).cpu().numpy().flatten()

        # Analyze by probability threshold
        print("\n=== LSTM Classifier Results ===")
        print(f"Total games: {len(targets)}")
        print(f"Ridge base rate: {targets.mean():.1%} ATS")

        print("\n--- By Edge Threshold (Ridge picks) ---")
        for edge_thresh in [3, 5, 7]:
            mask = edges >= edge_thresh
            if mask.sum() > 0:
                acc = targets[mask].mean()
                print(f"  {edge_thresh}+ pt edges: {int(targets[mask].sum())}/{mask.sum()} = {acc:.1%}")

        print("\n--- By LSTM Confidence ---")
        for prob_thresh in [0.55, 0.6, 0.65, 0.7]:
            # High confidence agrees with Ridge
            agree_mask = probs >= prob_thresh
            if agree_mask.sum() > 0:
                acc = targets[agree_mask].mean()
                print(f"  P(cover) >= {prob_thresh}: {int(targets[agree_mask].sum())}/{agree_mask.sum()} = {acc:.1%}")

            # High confidence disagrees (fade Ridge)
            fade_mask = probs <= (1 - prob_thresh)
            if fade_mask.sum() > 0:
                # When fading, success = Ridge DOESN'T cover
                fade_acc = (1 - targets[fade_mask]).mean()
                print(f"  Fade (P <= {1-prob_thresh:.2f}): {int((1-targets[fade_mask]).sum())}/{fade_mask.sum()} = {fade_acc:.1%}")

        print("\n--- Combined: High Edge + High Confidence ---")
        for edge_thresh in [3, 5]:
            edge_mask = edges >= edge_thresh
            for prob_thresh in [0.55, 0.6]:
                combined_mask = edge_mask & (probs >= prob_thresh)
                if combined_mask.sum() > 10:
                    acc = targets[combined_mask].mean()
                    print(f"  {edge_thresh}+ edge, P >= {prob_thresh}: {int(targets[combined_mask].sum())}/{combined_mask.sum()} = {acc:.1%}")

        return {'probs': probs, 'targets': targets, 'edges': edges}

    def save(self):
        """Save model."""
        MODEL_PATH.parent.mkdir(exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'seq_features': self.seq_features,
            'static_features': self.static_features,
        }, MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved to {MODEL_PATH}")

    @classmethod
    def load(cls) -> 'NBALSTMClassifier':
        """Load model."""
        predictor = cls()
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        predictor.model = SimpleLSTMClassifier(
            seq_features=checkpoint['seq_features'],
            static_features=checkpoint['static_features'],
        ).to(DEVICE)
        predictor.model.load_state_dict(checkpoint['model_state'])
        predictor.model.eval()
        with open(SCALER_PATH, 'rb') as f:
            predictor.scaler = pickle.load(f)
        return predictor


def main():
    """Train and evaluate."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    classifier = NBALSTMClassifier()

    logger.info("Building training data (2024-2025)...")
    train_data = classifier.build_dataset([2024, 2025], cursor)

    logger.info("Building validation data (2026)...")
    val_data = classifier.build_dataset([2026], cursor)

    conn.close()

    logger.info("Training classifier...")
    classifier.train(train_data, val_data, epochs=100, patience=20)

    logger.info("\nEvaluating on 2026...")
    classifier.evaluate(val_data)

    classifier.save()


if __name__ == '__main__':
    main()
