"""
NBA Edge Classifier - Neural Network to Find Betting Edges

Instead of predicting game outcomes directly, this model predicts:
"Given the model-Vegas disagreement and situational factors, should we bet with the model, fade it, or pass?"

The key insight: our model has predictable error patterns that a meta-model can exploit.

Architecture:
- Input: Meta-features (model-Vegas diffs, situational factors, recent accuracy)
- Output: Probabilities for SPREAD and TOTAL betting decisions
- Training: Walk-forward to avoid lookahead bias
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DB_PATH = Path(__file__).parent / 'nba_games.db'
MODEL_DIR = Path(__file__).parent / 'models'


class BasePredictor:
    """Simple predictor to generate model predictions for training data."""
    DECAY = 0.97
    MIN_GAMES = 10

    def __init__(self):
        self.team_stats = defaultdict(lambda: {'ppg': [], 'papg': [], 'wts': []})
        self.last_game = {}
        self.model_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

    def _wavg(self, vals, wts):
        if not vals or not wts:
            return None
        return np.average(vals[-len(wts):], weights=wts[-len(vals):])

    def _get_rest(self, tid, date):
        if tid not in self.last_game:
            return 3
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 2

    def get_recent_accuracy(self, n=20):
        """Get model's recent ATS accuracy."""
        total = sum(s['total'] for s in self.model_accuracy.values())
        if total == 0:
            return 0.5
        correct = sum(s['correct'] for s in self.model_accuracy.values())
        return correct / total

    def extract_spread_features(self, hid, aid, date):
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]
        if len(hs['ppg']) < self.MIN_GAMES or len(aws['ppg']) < self.MIN_GAMES:
            return None

        h_ppg = self._wavg(hs['ppg'], hs['wts'])
        h_papg = self._wavg(hs['papg'], hs['wts'])
        a_ppg = self._wavg(aws['ppg'], aws['wts'])
        a_papg = self._wavg(aws['papg'], aws['wts'])

        h_recent = np.mean(hs['ppg'][-5:]) if len(hs['ppg']) >= 5 else h_ppg
        a_recent = np.mean(aws['ppg'][-5:]) if len(aws['ppg']) >= 5 else a_ppg

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            h_ppg - a_ppg,
            h_papg - a_papg,
            (h_ppg - h_papg) - (a_ppg - a_papg),
            h_recent - a_recent,
            hr - ar,
            1 if hr == 0 else 0,
            1 if ar == 0 else 0,
            1.8,
            min(len(hs['ppg']) / 30, 1),
            min(len(aws['ppg']) / 30, 1),
        ])

    def extract_total_features(self, hid, aid, date):
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]
        if len(hs['ppg']) < self.MIN_GAMES or len(aws['ppg']) < self.MIN_GAMES:
            return None

        h_ppg = self._wavg(hs['ppg'], hs['wts'])
        h_papg = self._wavg(hs['papg'], hs['wts'])
        a_ppg = self._wavg(aws['ppg'], aws['wts'])
        a_papg = self._wavg(aws['papg'], aws['wts'])

        h_recent = np.mean(hs['ppg'][-5:]) if len(hs['ppg']) >= 5 else h_ppg
        a_recent = np.mean(aws['ppg'][-5:]) if len(aws['ppg']) >= 5 else a_ppg

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            h_ppg + a_ppg,
            h_papg + a_papg,
            h_recent + a_recent,
            1 if hr == 0 else 0,
            1 if ar == 0 else 0,
            min(len(hs['ppg']) / 30, 1),
            min(len(aws['ppg']) / 30, 1),
        ])

    def update(self, tid, pf, pa, date):
        ts = self.team_stats[tid]
        ts['wts'] = [w * self.DECAY for w in ts['wts']]
        ts['ppg'].append(pf)
        ts['papg'].append(pa)
        ts['wts'].append(1.0)
        self.last_game[tid] = date


class EdgeClassifier(nn.Module):
    """
    Neural network to predict betting edges.

    Input: Meta-features about model-Vegas disagreement and game context
    Output: Probabilities for each betting action
    """

    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3),
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Separate heads for spread and total decisions
        # Output: [P(pass), P(bet_with_model), P(fade_model)]
        self.spread_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

        self.total_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        shared = self.shared(x)
        spread_logits = self.spread_head(shared)
        total_logits = self.total_head(shared)
        return spread_logits, total_logits


def generate_training_data():
    """
    Generate training data with meta-features and labels.

    For each game:
    - Run base predictor to get model spread/total
    - Compare to Vegas
    - Create meta-features
    - Label: did betting WITH model win? did FADING model win?
    """
    print("=" * 70)
    print("GENERATING TRAINING DATA")
    print("=" * 70)

    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
    ''', conn)
    conn.close()

    # Filter to games with Vegas lines
    games = games[games['vegas_spread'].notna() & games['vegas_total'].notna()].copy()
    print(f"Games with Vegas lines: {len(games)}")

    # Walk-forward prediction
    predictor = BasePredictor()
    spread_ridge = None
    total_ridge = None
    spread_scaler = StandardScaler()
    total_scaler = StandardScaler()

    X_spread_train, y_spread_train = [], []
    X_total_train, y_total_train = [], []

    training_data = []
    recent_spread_results = []  # Track recent ATS results
    recent_total_results = []   # Track recent O/U results

    for idx, g in games.iterrows():
        actual_spread = g['away_score'] - g['home_score']
        actual_total = g['home_score'] + g['away_score']
        vegas_spread = g['vegas_spread']
        vegas_total = g['vegas_total']

        # Get features
        spread_feat = predictor.extract_spread_features(
            g['home_team_id'], g['away_team_id'], g['date']
        )
        total_feat = predictor.extract_total_features(
            g['home_team_id'], g['away_team_id'], g['date']
        )

        # Make predictions if we have enough training data
        if spread_feat is not None and len(X_spread_train) >= 100:
            # Fit ridge models
            X_s = spread_scaler.fit_transform(np.array(X_spread_train))
            spread_ridge = Ridge(alpha=0.1).fit(X_s, np.array(y_spread_train))
            model_spread = spread_ridge.predict(
                spread_scaler.transform(spread_feat.reshape(1, -1))
            )[0]

            X_t = total_scaler.fit_transform(np.array(X_total_train))
            total_ridge = Ridge(alpha=0.1).fit(X_t, np.array(y_total_train))
            model_total = total_ridge.predict(
                total_scaler.transform(total_feat.reshape(1, -1))
            )[0]

            # Calculate meta-features
            spread_edge = model_spread - vegas_spread  # Positive = model favors away more
            total_edge = model_total - vegas_total      # Positive = model expects higher scoring

            # Recent accuracy (last 50 games)
            recent_spread_acc = np.mean(recent_spread_results[-50:]) if recent_spread_results else 0.5
            recent_total_acc = np.mean(recent_total_results[-50:]) if recent_total_results else 0.5

            # Team games played
            home_games = len(predictor.team_stats[g['home_team_id']]['ppg'])
            away_games = len(predictor.team_stats[g['away_team_id']]['ppg'])

            # Rest days
            home_rest = predictor._get_rest(g['home_team_id'], g['date'])
            away_rest = predictor._get_rest(g['away_team_id'], g['date'])

            # Season progress
            season_games = len([x for x in games.itertuples()
                               if x.season == g['season'] and x.date < g['date']])

            meta_features = {
                # Core disagreement features
                'spread_edge': spread_edge,
                'spread_edge_abs': abs(spread_edge),
                'spread_edge_positive': 1 if spread_edge > 0 else 0,
                'total_edge': total_edge,
                'total_edge_abs': abs(total_edge),
                'total_edge_positive': 1 if total_edge > 0 else 0,

                # Model predictions
                'model_spread': model_spread,
                'model_total': model_total,
                'vegas_spread': vegas_spread,
                'vegas_total': vegas_total,

                # Recent model accuracy
                'recent_spread_acc': recent_spread_acc,
                'recent_total_acc': recent_total_acc,

                # Team reliability
                'home_games': min(home_games / 30, 1),
                'away_games': min(away_games / 30, 1),
                'combined_games': min((home_games + away_games) / 60, 1),

                # Situational
                'home_rest': home_rest,
                'away_rest': away_rest,
                'rest_diff': home_rest - away_rest,
                'home_b2b': 1 if home_rest == 0 else 0,
                'away_b2b': 1 if away_rest == 0 else 0,

                # Season context
                'season_progress': min(season_games / 1230, 1),  # ~1230 games per season
                'early_season': 1 if season_games < 300 else 0,
                'mid_season': 1 if 300 <= season_games < 800 else 0,
                'late_season': 1 if season_games >= 800 else 0,

                # Spread magnitude features
                'vegas_spread_abs': abs(vegas_spread),
                'big_favorite': 1 if abs(vegas_spread) > 8 else 0,
                'close_game': 1 if abs(vegas_spread) < 3 else 0,

                # Total magnitude features
                'vegas_total_high': 1 if vegas_total > 230 else 0,
                'vegas_total_low': 1 if vegas_total < 215 else 0,
            }

            # Labels: did betting with model win? did fading win?
            # Spread labels
            spread_result = actual_spread - vegas_spread
            if spread_edge > 0:  # Model says away will do better than Vegas thinks
                spread_with_model_wins = 1 if spread_result > 0.5 else 0
                spread_fade_model_wins = 1 if spread_result < -0.5 else 0
            else:  # Model says home will do better
                spread_with_model_wins = 1 if spread_result < -0.5 else 0
                spread_fade_model_wins = 1 if spread_result > 0.5 else 0

            # Total labels
            total_result = actual_total - vegas_total
            if total_edge > 0:  # Model says over
                total_with_model_wins = 1 if total_result > 0.5 else 0
                total_fade_model_wins = 1 if total_result < -0.5 else 0
            else:  # Model says under
                total_with_model_wins = 1 if total_result < -0.5 else 0
                total_fade_model_wins = 1 if total_result > 0.5 else 0

            # Track recent results for accuracy features
            recent_spread_results.append(spread_with_model_wins)
            recent_total_results.append(total_with_model_wins)

            training_data.append({
                **meta_features,
                'actual_spread': actual_spread,
                'actual_total': actual_total,
                'spread_with_model_wins': spread_with_model_wins,
                'spread_fade_model_wins': spread_fade_model_wins,
                'total_with_model_wins': total_with_model_wins,
                'total_fade_model_wins': total_fade_model_wins,
                'season': g['season'],
            })

        # Update training data
        if spread_feat is not None:
            X_spread_train.append(spread_feat)
            y_spread_train.append(actual_spread)
        if total_feat is not None:
            X_total_train.append(total_feat)
            y_total_train.append(actual_total)

        # Update predictor
        predictor.update(g['home_team_id'], g['home_score'], g['away_score'], g['date'])
        predictor.update(g['away_team_id'], g['away_score'], g['home_score'], g['date'])

    df = pd.DataFrame(training_data)
    print(f"Training samples generated: {len(df)}")

    return df


def train_classifier(df: pd.DataFrame):
    """Train the edge classifier on generated data."""
    print("\n" + "=" * 70)
    print("TRAINING EDGE CLASSIFIER")
    print("=" * 70)

    # Feature columns
    feature_cols = [
        'spread_edge', 'spread_edge_abs', 'spread_edge_positive',
        'total_edge', 'total_edge_abs', 'total_edge_positive',
        'model_spread', 'model_total', 'vegas_spread', 'vegas_total',
        'recent_spread_acc', 'recent_total_acc',
        'home_games', 'away_games', 'combined_games',
        'home_rest', 'away_rest', 'rest_diff', 'home_b2b', 'away_b2b',
        'season_progress', 'early_season', 'mid_season', 'late_season',
        'vegas_spread_abs', 'big_favorite', 'close_game',
        'vegas_total_high', 'vegas_total_low',
    ]

    X = df[feature_cols].values

    # Create labels: 0=pass, 1=bet_with_model, 2=fade_model
    # For spread
    y_spread = np.zeros(len(df), dtype=np.int64)
    y_spread[df['spread_with_model_wins'] == 1] = 1
    y_spread[df['spread_fade_model_wins'] == 1] = 2

    # For total
    y_total = np.zeros(len(df), dtype=np.int64)
    y_total[df['total_with_model_wins'] == 1] = 1
    y_total[df['total_fade_model_wins'] == 1] = 2

    # Train/test split by season
    train_seasons = [2023, 2024]
    test_season = 2025

    train_mask = df['season'].isin(train_seasons)
    test_mask = df['season'] == test_season

    X_train, X_test = X[train_mask], X[test_mask]
    y_spread_train, y_spread_test = y_spread[train_mask], y_spread[test_mask]
    y_total_train, y_total_test = y_total[train_mask], y_total[test_mask]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_spread_train_t = torch.LongTensor(y_spread_train)
    y_total_train_t = torch.LongTensor(y_total_train)

    # Create model
    model = EdgeClassifier(input_dim=len(feature_cols), hidden_dims=[64, 32])

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Training loop
    batch_size = 64
    n_epochs = 150

    dataset = TensorDataset(X_train_t, y_spread_train_t, y_total_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\nTraining for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y_spread, batch_y_total in loader:
            optimizer.zero_grad()

            spread_logits, total_logits = model(batch_X)

            loss_spread = criterion(spread_logits, batch_y_spread)
            loss_total = criterion(total_logits, batch_y_total)
            loss = loss_spread + loss_total

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET (2025 Season)")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        spread_logits, total_logits = model(X_test_t)
        spread_probs = torch.softmax(spread_logits, dim=1).numpy()
        total_probs = torch.softmax(total_logits, dim=1).numpy()

    # Analyze spread predictions
    print("\nSPREAD BETTING ANALYSIS:")
    analyze_predictions(spread_probs, y_spread_test, df[test_mask], 'spread')

    # Analyze total predictions
    print("\nTOTAL BETTING ANALYSIS:")
    analyze_predictions(total_probs, y_total_test, df[test_mask], 'total')

    # Save model and scaler
    save_path = MODEL_DIR / 'nba_edge_classifier.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
    }, save_path)
    print(f"\nModel saved to {save_path}")

    return model, scaler, feature_cols


def analyze_predictions(probs, y_true, df_test, bet_type):
    """Analyze classifier predictions and find profitable strategies."""

    # probs: [P(pass), P(bet_with), P(fade)]
    pred_action = np.argmax(probs, axis=1)

    # Confidence thresholds to test
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    print(f"\n{'Threshold':<12} {'Action':<15} {'Bets':>6} {'Wins':>6} {'Losses':>6} {'Win%':>8} {'ROI':>10}")
    print("-" * 75)

    for threshold in thresholds:
        for action_idx, action_name in [(1, 'BET_WITH'), (2, 'FADE')]:
            # Games where model predicts this action with confidence >= threshold
            confident_mask = probs[:, action_idx] >= threshold

            if confident_mask.sum() == 0:
                continue

            # Check outcomes
            wins = (y_true[confident_mask] == action_idx).sum()
            losses = confident_mask.sum() - wins - (y_true[confident_mask] == 0).sum()
            pushes = (y_true[confident_mask] == 0).sum()

            total_decided = wins + losses
            if total_decided > 0:
                win_pct = wins / total_decided * 100
                roi = (wins * 0.909 - losses) / total_decided * 100
                profitable = " ***" if win_pct > 52.4 else ""

                print(f">= {threshold:<9} {action_name:<15} {confident_mask.sum():>6} {wins:>6} "
                      f"{losses:>6} {win_pct:>7.1f}% {roi:>+9.1f}%{profitable}")

    # Best strategy summary
    print("\n" + "-" * 75)
    print("Looking for combinations...")

    best_roi = -100
    best_config = None

    for threshold in thresholds:
        for action_idx, action_name in [(1, 'BET_WITH'), (2, 'FADE')]:
            confident_mask = probs[:, action_idx] >= threshold

            if confident_mask.sum() < 20:  # Need minimum sample size
                continue

            wins = (y_true[confident_mask] == action_idx).sum()
            losses = confident_mask.sum() - wins - (y_true[confident_mask] == 0).sum()

            total_decided = wins + losses
            if total_decided > 0:
                roi = (wins * 0.909 - losses) / total_decided * 100
                win_pct = wins / total_decided * 100

                if roi > best_roi and win_pct > 50:
                    best_roi = roi
                    best_config = {
                        'threshold': threshold,
                        'action': action_name,
                        'bets': confident_mask.sum(),
                        'wins': wins,
                        'losses': losses,
                        'win_pct': win_pct,
                        'roi': roi
                    }

    if best_config:
        print(f"\nBest {bet_type.upper()} strategy:")
        print(f"  Action: {best_config['action']} when confidence >= {best_config['threshold']}")
        print(f"  Record: {best_config['wins']}-{best_config['losses']} ({best_config['win_pct']:.1f}%)")
        print(f"  ROI: {best_config['roi']:+.1f}%")


def find_edge_patterns(df: pd.DataFrame):
    """
    Use the classifier to find interesting edge patterns.
    Look at feature importance and decision boundaries.
    """
    print("\n" + "=" * 70)
    print("EDGE PATTERN ANALYSIS")
    print("=" * 70)

    # Analyze by different cuts of the data
    print("\n1. BY SPREAD EDGE SIZE:")
    for min_edge, max_edge in [(0, 3), (3, 5), (5, 7), (7, 100)]:
        mask = (df['spread_edge_abs'] >= min_edge) & (df['spread_edge_abs'] < max_edge)
        subset = df[mask]
        if len(subset) < 20:
            continue

        # When model favors away (positive edge)
        pos_mask = subset['spread_edge_positive'] == 1
        if pos_mask.sum() > 10:
            pos_wr = subset.loc[pos_mask, 'spread_with_model_wins'].mean()
            print(f"  Edge {min_edge}-{max_edge}, AWAY bias: {pos_mask.sum()} games, "
                  f"with_model WR: {pos_wr*100:.1f}%")

        # When model favors home (negative edge)
        neg_mask = subset['spread_edge_positive'] == 0
        if neg_mask.sum() > 10:
            neg_wr = subset.loc[neg_mask, 'spread_with_model_wins'].mean()
            print(f"  Edge {min_edge}-{max_edge}, HOME bias: {neg_mask.sum()} games, "
                  f"with_model WR: {neg_wr*100:.1f}%")

    print("\n2. BY TOTAL EDGE SIZE:")
    for min_edge, max_edge in [(0, 3), (3, 5), (5, 7), (7, 100)]:
        mask = (df['total_edge_abs'] >= min_edge) & (df['total_edge_abs'] < max_edge)
        subset = df[mask]
        if len(subset) < 20:
            continue

        # When model says over
        over_mask = subset['total_edge_positive'] == 1
        if over_mask.sum() > 10:
            over_wr = subset.loc[over_mask, 'total_with_model_wins'].mean()
            fade_wr = subset.loc[over_mask, 'total_fade_model_wins'].mean()
            print(f"  Edge {min_edge}-{max_edge}, OVER signal: {over_mask.sum()} games, "
                  f"with: {over_wr*100:.1f}%, fade: {fade_wr*100:.1f}%")

        # When model says under
        under_mask = subset['total_edge_positive'] == 0
        if under_mask.sum() > 10:
            under_wr = subset.loc[under_mask, 'total_with_model_wins'].mean()
            fade_wr = subset.loc[under_mask, 'total_fade_model_wins'].mean()
            print(f"  Edge {min_edge}-{max_edge}, UNDER signal: {under_mask.sum()} games, "
                  f"with: {under_wr*100:.1f}%, fade: {fade_wr*100:.1f}%")

    print("\n3. BY SEASON SEGMENT:")
    for segment, col in [('Early', 'early_season'), ('Mid', 'mid_season'), ('Late', 'late_season')]:
        mask = df[col] == 1
        subset = df[mask]
        if len(subset) < 20:
            continue

        spread_wr = subset['spread_with_model_wins'].mean()
        total_wr = subset['total_with_model_wins'].mean()
        total_fade = subset['total_fade_model_wins'].mean()
        print(f"  {segment} season: {len(subset)} games")
        print(f"    Spread with_model: {spread_wr*100:.1f}%")
        print(f"    Total with_model: {total_wr*100:.1f}%, fade: {total_fade*100:.1f}%")

    print("\n4. BY RECENT MODEL ACCURACY:")
    for min_acc, max_acc in [(0, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 1.0)]:
        mask = (df['recent_spread_acc'] >= min_acc) & (df['recent_spread_acc'] < max_acc)
        subset = df[mask]
        if len(subset) < 20:
            continue

        spread_wr = subset['spread_with_model_wins'].mean()
        print(f"  Recent acc {min_acc:.0%}-{max_acc:.0%}: {len(subset)} games, "
              f"next game with_model WR: {spread_wr*100:.1f}%")

    print("\n5. INTERACTION: SPREAD EDGE + FAVORITE SIZE")
    for big_fav in [0, 1]:
        fav_label = "Big favorite (>8)" if big_fav else "Close/Small favorite"
        mask = df['big_favorite'] == big_fav

        # High positive edge (model likes away)
        high_away = mask & (df['spread_edge'] >= 5)
        if high_away.sum() > 10:
            wr = df.loc[high_away, 'spread_with_model_wins'].mean()
            print(f"  {fav_label}, Model likes away (+5): {high_away.sum()} games, WR: {wr*100:.1f}%")

        # High negative edge (model likes home)
        high_home = mask & (df['spread_edge'] <= -5)
        if high_home.sum() > 10:
            wr = df.loc[high_home, 'spread_with_model_wins'].mean()
            print(f"  {fav_label}, Model likes home (-5): {high_home.sum()} games, WR: {wr*100:.1f}%")


def main():
    """Run the full edge classifier pipeline."""
    # Generate training data
    df = generate_training_data()

    # Save training data for analysis
    df.to_csv('nba_edge_training_data.csv', index=False)
    print(f"Training data saved to nba_edge_training_data.csv")

    # Find edge patterns in data
    find_edge_patterns(df)

    # Train classifier
    model, scaler, feature_cols = train_classifier(df)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review edge patterns above for manual strategies")
    print("2. Use classifier for automated edge detection")
    print("3. Backtest combined strategies")


if __name__ == '__main__':
    main()
