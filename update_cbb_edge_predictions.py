"""
Update CBB 2026 predictions with edge classifier outputs.

This script:
1. Walks through all 2026 games (completed and upcoming)
2. Generates predictions from both Simple and Enhanced models
3. Runs the edge classifier to get spread action/confidence
4. Updates the database with all predictions
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DB_PATH = Path(__file__).parent / 'cbb_games.db'
MODEL_DIR = Path(__file__).parent / 'models'

# Constants matching production models
SIMPLE_DECAY = 0.95
ENHANCED_DECAY = 0.92
PREV_HALF_LIFE = 6.0
MIN_GAMES = 5


class DualModelPredictor:
    """Maintains state for both Simple and Enhanced models."""

    def __init__(self):
        self.simple_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'fg_pct': [], 'fg_wts': [],
            'rebounds': [], 'reb_wts': [],
            'turnovers': [], 'tov_wts': [],
        }))

        self.enhanced_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'fg_pct': [], 'fg_wts': [],
            'rebounds': [], 'reb_wts': [],
            'turnovers': [], 'tov_wts': [],
            'margins': [], 'wins': [],
        }))

        self.team_hca_data = defaultdict(lambda: defaultdict(lambda: {
            'home_margins': [], 'away_margins': []
        }))

        self.prev_ratings = {}
        self.prev_hca = {}
        self.last_game = {}
        self.league_avg = {'ppg': 72.0, 'papg': 72.0}

        self.simple_spread_model = None
        self.simple_total_model = None
        self.enhanced_spread_model = None
        self.enhanced_total_model = None

        self.simple_spread_scaler = StandardScaler()
        self.simple_total_scaler = StandardScaler()
        self.enhanced_spread_scaler = StandardScaler()
        self.enhanced_total_scaler = StandardScaler()

        self.simple_spread_X, self.simple_spread_y = [], []
        self.simple_total_X, self.simple_total_y = [], []
        self.enhanced_spread_X, self.enhanced_spread_y = [], []
        self.enhanced_total_X, self.enhanced_total_y = [], []

    def _wavg(self, vals, wts):
        if not vals or not wts or len(vals) != len(wts):
            return None
        return float(np.average(vals, weights=wts))

    def _get_rest(self, tid, date):
        if tid not in self.last_game:
            return 3
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 2

    def _get_simple_stats(self, tid, season):
        td = self.simple_stats[tid][season]
        n = len(td['ppg'])
        if n == 0:
            prev = self.prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', self.league_avg['ppg']),
                'papg': prev.get('papg', self.league_avg['papg']),
                'fg_pct': prev.get('fg_pct', 45.0),
                'rebounds': prev.get('rebounds', 35.0),
                'turnovers': prev.get('turnovers', 13.0),
                'games': 0,
            }

        ppg = self._wavg(td['ppg'], td['wts'])
        papg = self._wavg(td['papg'], td['wts'])
        fg = self._wavg(td['fg_pct'], td['fg_wts']) if td['fg_pct'] else None
        reb = self._wavg(td['rebounds'], td['reb_wts']) if td['rebounds'] else None
        tov = self._wavg(td['turnovers'], td['tov_wts']) if td['turnovers'] else None

        prev = self.prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', self.league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', self.league_avg['papg']) + (1 - blend) * papg,
            'fg_pct': fg if fg else prev.get('fg_pct', 45.0),
            'rebounds': reb if reb else prev.get('rebounds', 35.0),
            'turnovers': tov if tov else prev.get('turnovers', 13.0),
            'games': n,
        }

    def _get_enhanced_stats(self, tid, season):
        td = self.enhanced_stats[tid][season]
        n = len(td['ppg'])
        if n == 0:
            prev = self.prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', self.league_avg['ppg']),
                'papg': prev.get('papg', self.league_avg['papg']),
                'fg_pct': prev.get('fg_pct', 45.0),
                'rebounds': prev.get('rebounds', 35.0),
                'turnovers': prev.get('turnovers', 13.0),
                'games': 0,
                'margins': [],
                'wins': [],
            }

        ppg = self._wavg(td['ppg'], td['wts'])
        papg = self._wavg(td['papg'], td['wts'])
        fg = self._wavg(td['fg_pct'], td['fg_wts']) if td['fg_pct'] else None
        reb = self._wavg(td['rebounds'], td['reb_wts']) if td['rebounds'] else None
        tov = self._wavg(td['turnovers'], td['tov_wts']) if td['turnovers'] else None

        prev = self.prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', self.league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', self.league_avg['papg']) + (1 - blend) * papg,
            'fg_pct': fg if fg else prev.get('fg_pct', 45.0),
            'rebounds': reb if reb else prev.get('rebounds', 35.0),
            'turnovers': tov if tov else prev.get('turnovers', 13.0),
            'games': n,
            'margins': td['margins'],
            'wins': td['wins'],
        }

    def _get_dynamic_hca(self, home_id, season, neutral_site=False):
        if neutral_site:
            return 0.0

        hd = self.team_hca_data[home_id][season]
        n_home = len(hd['home_margins'])
        n_away = len(hd['away_margins'])
        total = n_home + n_away

        if total == 0:
            return self.prev_hca.get(home_id, 3.5)  # CBB HCA typically ~3.5

        if n_home > 0 and n_away > 0:
            raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
            raw = max(-2, min(raw, 10))
        else:
            raw = 3.5

        shrunk = 3.5 + 0.5 * (raw - 3.5)
        prev = self.prev_hca.get(home_id, 3.5)
        blend = 0.5 ** (total / 20.0)  # Faster adaptation for CBB

        return blend * prev + (1 - blend) * shrunk

    def _recent_form(self, margins, n=5):
        if len(margins) < n:
            return 0.0
        return float(np.mean(margins[-n:]))

    def _momentum(self, margins, n=6):
        if len(margins) < n:
            return 0.0
        recent = margins[-n:]
        return np.mean(recent[n//2:]) - np.mean(recent[:n//2])

    def _streak(self, wins):
        if not wins:
            return 0
        s, last = 0, wins[-1]
        for w in reversed(wins):
            if w == last:
                s += 1
            else:
                break
        return s if last == 1 else -s

    def extract_simple_spread_features(self, hid, aid, season, date, neutral_site=False):
        hs = self._get_simple_stats(hid, season)
        aws = self._get_simple_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        net = (hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg'])

        return np.array([
            hs['ppg'] - aws['ppg'],
            hs['papg'] - aws['papg'],
            hs['fg_pct'] - aws['fg_pct'],
            hs['rebounds'] - aws['rebounds'],
            hs['turnovers'] - aws['turnovers'],
            hr - ar,
            1.0 if hr == 0 else 0.0,
            1.0 if ar == 0 else 0.0,
            net,
            net,
            min(hs['games'] / 15.0, 1.0),
            min(aws['games'] / 15.0, 1.0),
        ])

    def extract_simple_total_features(self, hid, aid, season, date):
        hs = self._get_simple_stats(hid, season)
        aws = self._get_simple_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            hs['ppg'] + aws['ppg'],
            hs['papg'] + aws['papg'],
            1.0 if hr == 0 else 0.0,
            1.0 if ar == 0 else 0.0,
            hs['rebounds'] + aws['rebounds'],
            hs['turnovers'] + aws['turnovers'],
        ])

    def extract_enhanced_spread_features(self, hid, aid, season, date, neutral_site=False):
        hs = self._get_enhanced_stats(hid, season)
        aws = self._get_enhanced_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        net = (hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg'])
        hca = self._get_dynamic_hca(hid, season, neutral_site)

        return np.array([
            hs['ppg'] - aws['ppg'],
            hs['papg'] - aws['papg'],
            net,
            self._recent_form(hs['margins']) - self._recent_form(aws['margins']),
            self._momentum(hs['margins']) - self._momentum(aws['margins']),
            self._streak(hs['wins']) - self._streak(aws['wins']),
            hr - ar,
            1.0 if hr == 0 else 0.0,
            1.0 if ar == 0 else 0.0,
            hca,
            min(hs['games'] / 25.0, 1.0),
            min(aws['games'] / 25.0, 1.0),
            1.0 if neutral_site else 0.0,
            (hs['games'] + aws['games']) / 60.0,
            hs['fg_pct'] - aws['fg_pct'],
            hs['rebounds'] - aws['rebounds'],
            hs['turnovers'] - aws['turnovers'],
        ])

    def extract_enhanced_total_features(self, hid, aid, season, date):
        hs = self._get_enhanced_stats(hid, season)
        aws = self._get_enhanced_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        home_recent = abs(self._recent_form(hs['margins']))
        away_recent = abs(self._recent_form(aws['margins']))
        home_mom = abs(self._momentum(hs['margins']))
        away_mom = abs(self._momentum(aws['margins']))

        return np.array([
            hs['ppg'] + aws['ppg'],
            hs['papg'] + aws['papg'],
            (hs['ppg'] + hs['papg']) / 2,
            (aws['ppg'] + aws['papg']) / 2,
            1.0 if hr == 0 else 0.0,
            1.0 if ar == 0 else 0.0,
            min(hs['games'] / 25.0, 1.0),
            min(aws['games'] / 25.0, 1.0),
            hs['fg_pct'] + aws['fg_pct'],
            hs['rebounds'] + aws['rebounds'],
            hs['turnovers'] + aws['turnovers'],
            home_recent + away_recent,
            home_mom + away_mom,
            (hs['games'] + aws['games']) / 60.0,
            0.0,  # No injury adjustment in CBB
        ])

    def predict(self, hid, aid, season, date, neutral_site=False):
        simple_spread_feat = self.extract_simple_spread_features(hid, aid, season, date, neutral_site)
        simple_spread = None
        if simple_spread_feat is not None and self.simple_spread_model is not None:
            X = self.simple_spread_scaler.transform(simple_spread_feat.reshape(1, -1))
            simple_spread = self.simple_spread_model.predict(X)[0]

        simple_total_feat = self.extract_simple_total_features(hid, aid, season, date)
        simple_total = None
        if simple_total_feat is not None and self.simple_total_model is not None:
            X = self.simple_total_scaler.transform(simple_total_feat.reshape(1, -1))
            simple_total = self.simple_total_model.predict(X)[0]

        enhanced_spread_feat = self.extract_enhanced_spread_features(hid, aid, season, date, neutral_site)
        enhanced_spread = None
        if enhanced_spread_feat is not None and self.enhanced_spread_model is not None:
            X = self.enhanced_spread_scaler.transform(enhanced_spread_feat.reshape(1, -1))
            enhanced_spread = self.enhanced_spread_model.predict(X)[0]

        enhanced_total_feat = self.extract_enhanced_total_features(hid, aid, season, date)
        enhanced_total = None
        if enhanced_total_feat is not None and self.enhanced_total_model is not None:
            X = self.enhanced_total_scaler.transform(enhanced_total_feat.reshape(1, -1))
            enhanced_total = self.enhanced_total_model.predict(X)[0]

        return {
            'simple_spread': simple_spread,
            'simple_total': simple_total,
            'enhanced_spread': enhanced_spread,
            'enhanced_total': enhanced_total,
            'simple_spread_feat': simple_spread_feat,
            'simple_total_feat': simple_total_feat,
            'enhanced_spread_feat': enhanced_spread_feat,
            'enhanced_total_feat': enhanced_total_feat,
        }

    def update(self, tid, season, date, pf, pa, is_home, fg=None, reb=None, tov=None):
        margin = pf - pa

        ts = self.simple_stats[tid][season]
        ts['wts'] = [w * SIMPLE_DECAY for w in ts['wts']]
        ts['ppg'].append(pf)
        ts['papg'].append(pa)
        ts['wts'].append(1.0)
        if pd.notna(fg):
            ts['fg_wts'] = [w * SIMPLE_DECAY for w in ts['fg_wts']]
            ts['fg_pct'].append(fg)
            ts['fg_wts'].append(1.0)
        if pd.notna(reb):
            ts['reb_wts'] = [w * SIMPLE_DECAY for w in ts['reb_wts']]
            ts['rebounds'].append(reb)
            ts['reb_wts'].append(1.0)
        if pd.notna(tov):
            ts['tov_wts'] = [w * SIMPLE_DECAY for w in ts['tov_wts']]
            ts['turnovers'].append(tov)
            ts['tov_wts'].append(1.0)

        te = self.enhanced_stats[tid][season]
        te['wts'] = [w * ENHANCED_DECAY for w in te['wts']]
        te['ppg'].append(pf)
        te['papg'].append(pa)
        te['wts'].append(1.0)
        te['margins'].append(margin)
        te['wins'].append(1 if margin > 0 else 0)
        if pd.notna(fg):
            te['fg_wts'] = [w * ENHANCED_DECAY for w in te['fg_wts']]
            te['fg_pct'].append(fg)
            te['fg_wts'].append(1.0)
        if pd.notna(reb):
            te['reb_wts'] = [w * ENHANCED_DECAY for w in te['reb_wts']]
            te['rebounds'].append(reb)
            te['reb_wts'].append(1.0)
        if pd.notna(tov):
            te['tov_wts'] = [w * ENHANCED_DECAY for w in te['tov_wts']]
            te['turnovers'].append(tov)
            te['tov_wts'].append(1.0)

        hd = self.team_hca_data[tid][season]
        if is_home:
            hd['home_margins'].append(margin)
        else:
            hd['away_margins'].append(-margin)

        self.last_game[tid] = date

    def set_previous_season(self, season):
        prev = season - 1

        for tid in self.simple_stats:
            if prev in self.simple_stats[tid]:
                td = self.simple_stats[tid][prev]
                if td['ppg']:
                    self.prev_ratings[tid] = {
                        'ppg': np.mean(td['ppg']),
                        'papg': np.mean(td['papg']),
                        'fg_pct': np.mean(td['fg_pct']) if td['fg_pct'] else 45.0,
                        'rebounds': np.mean(td['rebounds']) if td['rebounds'] else 35.0,
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 13.0,
                    }

        for tid in self.team_hca_data:
            if prev in self.team_hca_data[tid]:
                hd = self.team_hca_data[tid][prev]
                if hd['home_margins'] and hd['away_margins']:
                    raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
                    raw = max(-2, min(raw, 10))
                    self.prev_hca[tid] = 3.5 + 0.5 * (raw - 3.5)

        self.last_game.clear()

    def train_models(self):
        if len(self.simple_spread_X) >= 100:
            X = np.array(self.simple_spread_X)
            y = np.array(self.simple_spread_y)
            self.simple_spread_scaler.fit(X)
            X_s = self.simple_spread_scaler.transform(X)
            self.simple_spread_model = Ridge(alpha=1.0).fit(X_s, y)

        if len(self.simple_total_X) >= 100:
            X = np.array(self.simple_total_X)
            y = np.array(self.simple_total_y)
            self.simple_total_scaler.fit(X)
            X_s = self.simple_total_scaler.transform(X)
            self.simple_total_model = Ridge(alpha=1.0).fit(X_s, y)

        if len(self.enhanced_spread_X) >= 100:
            X = np.array(self.enhanced_spread_X)
            y = np.array(self.enhanced_spread_y)
            self.enhanced_spread_scaler.fit(X)
            X_s = self.enhanced_spread_scaler.transform(X)
            self.enhanced_spread_model = Ridge(alpha=1.0).fit(X_s, y)

        if len(self.enhanced_total_X) >= 100:
            X = np.array(self.enhanced_total_X)
            y = np.array(self.enhanced_total_y)
            self.enhanced_total_scaler.fit(X)
            X_s = self.enhanced_total_scaler.transform(X)
            self.enhanced_total_model = Ridge(alpha=1.0).fit(X_s, y)


def load_edge_classifier():
    """Load the trained edge classifier."""
    import torch.nn as nn

    class EdgeClassifier(nn.Module):
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
            self.spread_head = nn.Sequential(
                nn.Linear(prev_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
            )

        def forward(self, x):
            shared = self.shared(x)
            return self.spread_head(shared)

    model_path = MODEL_DIR / 'cbb_edge_classifier.pt'
    if not model_path.exists():
        print(f"Edge classifier not found at {model_path}")
        return None, None, None

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    feature_cols = checkpoint['feature_cols']
    scaler = checkpoint['scaler']

    model = EdgeClassifier(input_dim=len(feature_cols))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, scaler, feature_cols


def main():
    print("=" * 70)
    print("UPDATING CBB 2026 PREDICTIONS WITH EDGE CLASSIFIER")
    print("=" * 70)

    # Load edge classifier
    edge_model, edge_scaler, feature_cols = load_edge_classifier()
    if edge_model is None:
        print("ERROR: Could not load edge classifier")
        return

    print(f"Loaded edge classifier with {len(feature_cols)} features")

    conn = sqlite3.connect(str(DB_PATH))

    # Ensure columns exist
    cursor = conn.cursor()
    new_cols = [
        ('spread_action', 'TEXT'),
        ('spread_confidence', 'REAL'),
        ('spread_edge', 'REAL'),
        ('simple_pred_spread', 'REAL'),
        ('enhanced_pred_spread', 'REAL'),
        ('avg_pred_spread', 'REAL'),
        ('simple_pred_total', 'REAL'),
        ('enhanced_pred_total', 'REAL'),
        ('avg_pred_total', 'REAL'),
    ]

    existing = {r[1] for r in cursor.execute("PRAGMA table_info(odds_and_predictions)")}
    for col, dtype in new_cols:
        if col not in existing:
            cursor.execute(f"ALTER TABLE odds_and_predictions ADD COLUMN {col} {dtype}")
            print(f"  Added column: {col}")
    conn.commit()

    # Load all games (2024-2026 to build history)
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.completed, g.neutral_site,
               hs.field_goal_pct as home_fg, hs.total_rebounds as home_reb,
               hs.turnovers as home_tov,
               aws.field_goal_pct as away_fg, aws.total_rebounds as away_reb,
               aws.turnovers as away_tov,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total,
               ht.conference_id as home_conf, at.conference_id as away_conf
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id
            AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id
            AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        LEFT JOIN teams ht ON g.home_team_id = ht.team_id
        LEFT JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.season >= 2024
        ORDER BY g.date
    ''', conn)

    print(f"Loaded {len(games)} games (2024-2026)")

    # Initialize predictor
    predictor = DualModelPredictor()

    # Track recent results for accuracy features
    recent_spread_results = []
    seasons = sorted(games['season'].unique())

    # Predictions to update
    updates = []
    season_game_count = {}

    print("\nProcessing games...")

    for season in seasons:
        if season > seasons[0]:
            predictor.set_previous_season(season)

        season_game_count[season] = 0
        season_games = games[games['season'] == season].copy()

        for _, g in season_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            vegas_spread = g['vegas_spread']
            vegas_total = g['vegas_total']
            is_completed = g['completed'] == 1
            neutral = g['neutral_site'] == 1
            conf_game = (g['home_conf'] == g['away_conf']) if pd.notna(g['home_conf']) else 0

            # Get predictions
            preds = predictor.predict(hid, aid, season, g['date'], neutral)

            # Accumulate training features for walk-forward
            if is_completed and preds['simple_spread_feat'] is not None:
                actual_spread = g['away_score'] - g['home_score']
                actual_total = g['home_score'] + g['away_score']

                predictor.simple_spread_X.append(preds['simple_spread_feat'])
                predictor.simple_spread_y.append(actual_spread)

                if preds['simple_total_feat'] is not None:
                    predictor.simple_total_X.append(preds['simple_total_feat'])
                    predictor.simple_total_y.append(actual_total)

                if preds['enhanced_spread_feat'] is not None:
                    predictor.enhanced_spread_X.append(preds['enhanced_spread_feat'])
                    predictor.enhanced_spread_y.append(actual_spread)

                if preds['enhanced_total_feat'] is not None:
                    predictor.enhanced_total_X.append(preds['enhanced_total_feat'])
                    predictor.enhanced_total_y.append(actual_total)

                # Retrain periodically
                if len(predictor.simple_spread_X) % 200 == 0:
                    predictor.train_models()

            # For 2026 games with predictions, calculate edge classifier features
            if season == 2026 and all(preds[k] is not None for k in
                                       ['simple_spread', 'simple_total',
                                        'enhanced_spread', 'enhanced_total']):

                if pd.notna(vegas_spread) and pd.notna(vegas_total):
                    simple_spread_edge = preds['simple_spread'] - vegas_spread
                    enhanced_spread_edge = preds['enhanced_spread'] - vegas_spread
                    avg_spread = (preds['simple_spread'] + preds['enhanced_spread']) / 2
                    avg_total = (preds['simple_total'] + preds['enhanced_total']) / 2
                    avg_spread_edge = avg_spread - vegas_spread

                    spread_agreement = 1 if (simple_spread_edge * enhanced_spread_edge > 0) else 0

                    recent_spread_acc = np.mean(recent_spread_results[-50:]) if recent_spread_results else 0.5

                    hs = predictor._get_simple_stats(hid, season)
                    aws = predictor._get_simple_stats(aid, season)
                    hr = predictor._get_rest(hid, g['date'])
                    ar = predictor._get_rest(aid, g['date'])

                    meta = {
                        'simple_spread_edge': simple_spread_edge,
                        'simple_spread_edge_abs': abs(simple_spread_edge),
                        'simple_spread': preds['simple_spread'],
                        'enhanced_spread_edge': enhanced_spread_edge,
                        'enhanced_spread_edge_abs': abs(enhanced_spread_edge),
                        'enhanced_spread': preds['enhanced_spread'],
                        'avg_spread_edge': avg_spread_edge,
                        'avg_spread_edge_abs': abs(avg_spread_edge),
                        'spread_agreement': spread_agreement,
                        'spread_model_diff': abs(preds['simple_spread'] - preds['enhanced_spread']),
                        'vegas_spread': vegas_spread,
                        'vegas_spread_abs': abs(vegas_spread),
                        'big_favorite': 1 if abs(vegas_spread) > 12 else 0,
                        'close_game': 1 if abs(vegas_spread) < 4 else 0,
                        'recent_spread_acc': recent_spread_acc,
                        'home_games': min(hs['games'] / 25, 1),
                        'away_games': min(aws['games'] / 25, 1),
                        'combined_games': min((hs['games'] + aws['games']) / 50, 1),
                        'home_rest': hr,
                        'away_rest': ar,
                        'rest_diff': hr - ar,
                        'home_b2b': 1 if hr == 0 else 0,
                        'away_b2b': 1 if ar == 0 else 0,
                        'season_progress': min(season_game_count[season] / 5000, 1),
                        'early_season': 1 if season_game_count[season] < 1000 else 0,
                        'mid_season': 1 if 1000 <= season_game_count[season] < 3500 else 0,
                        'late_season': 1 if season_game_count[season] >= 3500 else 0,
                        'neutral_site': 1 if neutral else 0,
                        'conference_game': 1 if conf_game else 0,
                    }

                    # Build feature vector for edge classifier
                    X = np.array([[meta.get(col, 0) for col in feature_cols]])
                    X_scaled = edge_scaler.transform(X)

                    with torch.no_grad():
                        logits = edge_model(torch.FloatTensor(X_scaled))
                        probs = torch.softmax(logits, dim=1).numpy()[0]

                    # Determine action (use 0.55 threshold for FADE based on training)
                    if probs[2] >= 0.55:  # FADE threshold
                        action = 'FADE'
                        confidence = probs[2]
                    elif probs[1] >= 0.55:  # BET_WITH threshold
                        action = 'BET_WITH'
                        confidence = probs[1]
                    else:
                        action = 'PASS'
                        confidence = probs[0]

                    updates.append({
                        'game_id': int(g['game_id']),
                        'spread_action': action,
                        'spread_confidence': float(confidence),
                        'spread_edge': float(avg_spread_edge),
                        'simple_pred_spread': float(preds['simple_spread']),
                        'enhanced_pred_spread': float(preds['enhanced_spread']),
                        'avg_pred_spread': float(avg_spread),
                        'simple_pred_total': float(preds['simple_total']),
                        'enhanced_pred_total': float(preds['enhanced_total']),
                        'avg_pred_total': float(avg_total),
                    })

            # Update state for completed games
            if is_completed:
                predictor.update(hid, season, g['date'], g['home_score'], g['away_score'],
                               is_home=not neutral, fg=g['home_fg'], reb=g['home_reb'], tov=g['home_tov'])
                predictor.update(aid, season, g['date'], g['away_score'], g['home_score'],
                               is_home=False, fg=g['away_fg'], reb=g['away_reb'], tov=g['away_tov'])

                # Track recent accuracy
                if all(preds[k] is not None for k in ['simple_spread', 'enhanced_spread']) and pd.notna(vegas_spread):
                    actual_spread = g['away_score'] - g['home_score']
                    avg_spread_edge = ((preds['simple_spread'] + preds['enhanced_spread']) / 2) - vegas_spread
                    spread_result = actual_spread - vegas_spread
                    if avg_spread_edge > 0:
                        recent_spread_results.append(1 if spread_result > 0.5 else 0)
                    else:
                        recent_spread_results.append(1 if spread_result < -0.5 else 0)

                season_game_count[season] += 1

    print(f"\nGenerated {len(updates)} predictions to update")

    # Update database
    update_count = 0

    for upd in updates:
        cursor.execute('''
            UPDATE odds_and_predictions
            SET spread_action = ?,
                spread_confidence = ?,
                spread_edge = ?,
                simple_pred_spread = ?,
                enhanced_pred_spread = ?,
                avg_pred_spread = ?,
                simple_pred_total = ?,
                enhanced_pred_total = ?,
                avg_pred_total = ?
            WHERE game_id = ?
        ''', (
            upd['spread_action'],
            upd['spread_confidence'],
            upd['spread_edge'],
            upd['simple_pred_spread'],
            upd['enhanced_pred_spread'],
            upd['avg_pred_spread'],
            upd['simple_pred_total'],
            upd['enhanced_pred_total'],
            upd['avg_pred_total'],
            upd['game_id'],
        ))
        update_count += cursor.rowcount

    conn.commit()
    conn.close()

    print(f"Updated {update_count} rows in database")

    # Summary stats
    actions = pd.DataFrame(updates)
    if not actions.empty:
        print("\nAction Distribution:")
        print(actions['spread_action'].value_counts())
        print(f"\nAverage confidence: {actions['spread_confidence'].mean():.3f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
