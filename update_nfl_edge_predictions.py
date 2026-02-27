"""
Update NFL 2025 predictions with edge classifier outputs.

This script:
1. Walks through all 2025 games (completed and upcoming)
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

DB_PATH = Path(__file__).parent / 'nfl_games.db'
MODEL_DIR = Path(__file__).parent / 'models'

# Constants matching production models
SIMPLE_DECAY = 0.92
ENHANCED_DECAY = 0.90
PREV_HALF_LIFE = 4.0
MIN_GAMES = 3


class DualModelPredictor:
    """Maintains state for both Simple and Enhanced NFL models."""

    def __init__(self):
        self.simple_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'yards': [], 'yards_wts': [],
            'turnovers': [], 'to_wts': [],
            'third_down_pct': [], 'td_wts': [],
        }))

        self.enhanced_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'yards': [], 'yards_wts': [],
            'turnovers': [], 'to_wts': [],
            'margins': [], 'wins': [],
            'ppd': [], 'ppd_wts': [],
            'ypd': [], 'ypd_wts': [],
            'scoring_pct': [], 'sp_wts': [],
        }))

        self.team_hca_data = defaultdict(lambda: defaultdict(lambda: {
            'home_margins': [], 'away_margins': []
        }))

        self.prev_ratings = {}
        self.prev_hca = {}
        self.last_game = {}
        self.league_avg = {
            'ppg': 22.0, 'papg': 22.0, 'yards': 330.0, 'turnovers': 1.3,
            'ppd': 1.9, 'ypd': 28.0, 'scoring_pct': 0.35
        }

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
            return 7
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return min((curr - last).days, 14)
        except Exception:
            return 7

    def _get_simple_stats(self, tid, season):
        td = self.simple_stats[tid][season]
        n = len(td['ppg'])
        if n == 0:
            prev = self.prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', self.league_avg['ppg']),
                'papg': prev.get('papg', self.league_avg['papg']),
                'yards': prev.get('yards', self.league_avg['yards']),
                'turnovers': prev.get('turnovers', self.league_avg['turnovers']),
                'games': 0,
            }

        ppg = self._wavg(td['ppg'], td['wts'])
        papg = self._wavg(td['papg'], td['wts'])
        yards = self._wavg(td['yards'], td['yards_wts']) if td['yards'] else None
        to = self._wavg(td['turnovers'], td['to_wts']) if td['turnovers'] else None

        prev = self.prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', self.league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', self.league_avg['papg']) + (1 - blend) * papg,
            'yards': yards if yards else prev.get('yards', self.league_avg['yards']),
            'turnovers': to if to else prev.get('turnovers', self.league_avg['turnovers']),
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
                'ppd': prev.get('ppd', self.league_avg['ppd']),
                'games': 0,
                'margins': [],
                'wins': [],
            }

        ppg = self._wavg(td['ppg'], td['wts'])
        papg = self._wavg(td['papg'], td['wts'])
        ppd = self._wavg(td['ppd'], td['ppd_wts']) if td['ppd'] else None

        prev = self.prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', self.league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', self.league_avg['papg']) + (1 - blend) * papg,
            'ppd': ppd if ppd else prev.get('ppd', self.league_avg['ppd']),
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
            return self.prev_hca.get(home_id, 2.5)
        if n_home > 0 and n_away > 0:
            raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
            raw = max(-3, min(raw, 8))
        else:
            raw = 2.5
        shrunk = 2.5 + 0.5 * (raw - 2.5)
        prev = self.prev_hca.get(home_id, 2.5)
        blend = 0.5 ** (total / 10.0)
        return blend * prev + (1 - blend) * shrunk

    def _recent_form(self, margins, n=4):
        if len(margins) < n:
            return 0.0
        return float(np.mean(margins[-n:]))

    def extract_simple_spread_features(self, hid, aid, season, date, week):
        hs = self._get_simple_stats(hid, season)
        aws = self._get_simple_stats(aid, season)
        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None
        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        net = (hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg'])
        return np.array([
            hs['ppg'] - aws['ppg'], hs['papg'] - aws['papg'], net,
            hs['yards'] - aws['yards'], hs['turnovers'] - aws['turnovers'], 0.0,
            min(hr, 10) - min(ar, 10),
            1.0 if hr >= 13 else 0.0, 1.0 if ar >= 13 else 0.0,
            (1.0 if hr >= 13 else 0.0) - (1.0 if ar >= 13 else 0.0),
            min(hs['games'] / 10.0, 1.0), min(aws['games'] / 10.0, 1.0),
        ])

    def extract_simple_total_features(self, hid, aid, season, date, week):
        hs = self._get_simple_stats(hid, season)
        aws = self._get_simple_stats(aid, season)
        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None
        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        return np.array([
            hs['ppg'] + aws['ppg'], hs['papg'] + aws['papg'],
            hs['yards'] + aws['yards'], hs['turnovers'] + aws['turnovers'],
            1.0 if hr >= 13 else 0.0, 1.0 if ar >= 13 else 0.0,
            min(hs['games'] / 10.0, 1.0), min(aws['games'] / 10.0, 1.0),
        ])

    def extract_enhanced_spread_features(self, hid, aid, season, date, week, **kwargs):
        hs = self._get_enhanced_stats(hid, season)
        aws = self._get_enhanced_stats(aid, season)
        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None
        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        net = (hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg'])
        hca = self._get_dynamic_hca(hid, season, kwargs.get('neutral_site', False))
        return np.array([
            hs['ppg'] - aws['ppg'], hs['papg'] - aws['papg'], net,
            0.0, 0.0,  # yards, turnovers placeholders
            self._recent_form(hs['margins']) - self._recent_form(aws['margins']),
            0.0, 0.0,  # momentum, streak placeholders
            min(hr, 10) - min(ar, 10),
            (1.0 if hr >= 13 else 0.0) - (1.0 if ar >= 13 else 0.0),
            hca,
            hs['ppd'] - aws['ppd'], 0.0,  # scoring_pct placeholder
            1.0 if kwargs.get('is_dome', False) else 0.0,
            1.0 if kwargs.get('is_primetime', False) else 0.0,
            min(week / 17.0, 1.0),
            min(hs['games'] / 10.0, 1.0), min(aws['games'] / 10.0, 1.0),
        ])

    def extract_enhanced_total_features(self, hid, aid, season, date, week, is_dome=False):
        hs = self._get_enhanced_stats(hid, season)
        aws = self._get_enhanced_stats(aid, season)
        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None
        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        return np.array([
            hs['ppg'] + aws['ppg'], hs['papg'] + aws['papg'], 0.0, 0.0,
            hs['ppd'] + aws['ppd'], 0.0,
            abs(self._recent_form(hs['margins'])) + abs(self._recent_form(aws['margins'])),
            1.0 if hr >= 13 else 0.0, 1.0 if ar >= 13 else 0.0,
            1.0 if is_dome else 0.0, min(week / 17.0, 1.0),
            min(hs['games'] / 10.0, 1.0), min(aws['games'] / 10.0, 1.0),
            (hs['games'] + aws['games']) / 34.0,
        ])

    def predict(self, hid, aid, season, date, week, **kwargs):
        simple_spread_feat = self.extract_simple_spread_features(hid, aid, season, date, week)
        simple_spread = None
        if simple_spread_feat is not None and self.simple_spread_model is not None:
            X = self.simple_spread_scaler.transform(simple_spread_feat.reshape(1, -1))
            simple_spread = self.simple_spread_model.predict(X)[0]

        simple_total_feat = self.extract_simple_total_features(hid, aid, season, date, week)
        simple_total = None
        if simple_total_feat is not None and self.simple_total_model is not None:
            X = self.simple_total_scaler.transform(simple_total_feat.reshape(1, -1))
            simple_total = self.simple_total_model.predict(X)[0]

        enhanced_spread_feat = self.extract_enhanced_spread_features(hid, aid, season, date, week, **kwargs)
        enhanced_spread = None
        if enhanced_spread_feat is not None and self.enhanced_spread_model is not None:
            X = self.enhanced_spread_scaler.transform(enhanced_spread_feat.reshape(1, -1))
            enhanced_spread = self.enhanced_spread_model.predict(X)[0]

        enhanced_total_feat = self.extract_enhanced_total_features(
            hid, aid, season, date, week, is_dome=kwargs.get('is_dome', False)
        )
        enhanced_total = None
        if enhanced_total_feat is not None and self.enhanced_total_model is not None:
            X = self.enhanced_total_scaler.transform(enhanced_total_feat.reshape(1, -1))
            enhanced_total = self.enhanced_total_model.predict(X)[0]

        return {
            'simple_spread': simple_spread,
            'simple_total': simple_total,
            'enhanced_spread': enhanced_spread,
            'enhanced_total': enhanced_total,
        }

    def update(self, tid, season, date, pf, pa, is_home, yards=None, turnovers=None, ppd=None):
        margin = pf - pa
        ts = self.simple_stats[tid][season]
        ts['wts'] = [w * SIMPLE_DECAY for w in ts['wts']]
        ts['ppg'].append(pf)
        ts['papg'].append(pa)
        ts['wts'].append(1.0)
        if pd.notna(yards):
            ts['yards_wts'] = [w * SIMPLE_DECAY for w in ts['yards_wts']]
            ts['yards'].append(yards)
            ts['yards_wts'].append(1.0)
        if pd.notna(turnovers):
            ts['to_wts'] = [w * SIMPLE_DECAY for w in ts['to_wts']]
            ts['turnovers'].append(turnovers)
            ts['to_wts'].append(1.0)

        te = self.enhanced_stats[tid][season]
        te['wts'] = [w * ENHANCED_DECAY for w in te['wts']]
        te['ppg'].append(pf)
        te['papg'].append(pa)
        te['wts'].append(1.0)
        te['margins'].append(margin)
        te['wins'].append(1 if margin > 0 else 0)
        if pd.notna(ppd):
            te['ppd_wts'] = [w * ENHANCED_DECAY for w in te['ppd_wts']]
            te['ppd'].append(ppd)
            te['ppd_wts'].append(1.0)

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
                        'yards': np.mean(td['yards']) if td['yards'] else 330.0,
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 1.3,
                        'ppd': 1.9,
                    }
        for tid in self.enhanced_stats:
            if prev in self.enhanced_stats[tid]:
                te = self.enhanced_stats[tid][prev]
                if te['ppd']:
                    self.prev_ratings.setdefault(tid, {})['ppd'] = np.mean(te['ppd'])
        for tid in self.team_hca_data:
            if prev in self.team_hca_data[tid]:
                hd = self.team_hca_data[tid][prev]
                if hd['home_margins'] and hd['away_margins']:
                    raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
                    raw = max(-3, min(raw, 8))
                    self.prev_hca[tid] = 2.5 + 0.5 * (raw - 2.5)
        self.last_game.clear()

    def train_models(self):
        if len(self.simple_spread_X) >= 50:
            X = np.array(self.simple_spread_X)
            y = np.array(self.simple_spread_y)
            self.simple_spread_scaler.fit(X)
            self.simple_spread_model = Ridge(alpha=1.0).fit(self.simple_spread_scaler.transform(X), y)
        if len(self.simple_total_X) >= 50:
            X = np.array(self.simple_total_X)
            y = np.array(self.simple_total_y)
            self.simple_total_scaler.fit(X)
            self.simple_total_model = Ridge(alpha=1.0).fit(self.simple_total_scaler.transform(X), y)
        if len(self.enhanced_spread_X) >= 50:
            X = np.array(self.enhanced_spread_X)
            y = np.array(self.enhanced_spread_y)
            self.enhanced_spread_scaler.fit(X)
            self.enhanced_spread_model = Ridge(alpha=1.0).fit(self.enhanced_spread_scaler.transform(X), y)
        if len(self.enhanced_total_X) >= 50:
            X = np.array(self.enhanced_total_X)
            y = np.array(self.enhanced_total_y)
            self.enhanced_total_scaler.fit(X)
            self.enhanced_total_model = Ridge(alpha=1.0).fit(self.enhanced_total_scaler.transform(X), y)


def load_edge_classifier():
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

    model_path = MODEL_DIR / 'nfl_edge_classifier.pt'
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
    print("UPDATING NFL 2025 PREDICTIONS WITH EDGE CLASSIFIER")
    print("=" * 70)

    edge_model, edge_scaler, feature_cols = load_edge_classifier()
    if edge_model is None:
        print("ERROR: Could not load edge classifier")
        return

    print(f"Loaded edge classifier with {len(feature_cols)} features")

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Ensure columns exist
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

    # Load games
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.completed, g.neutral_site, g.is_dome,
               hs.total_yards as home_yards, hs.turnovers as home_to,
               aws.total_yards as away_yards, aws.turnovers as away_to,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.season >= 2022
        ORDER BY g.date, g.week
    ''', conn)

    # Load drive stats
    drives = pd.read_sql_query('''
        SELECT d.game_id, d.team_id, d.yards, d.is_score
        FROM drives d JOIN games g ON d.game_id = g.game_id WHERE g.completed = 1
    ''', conn)

    if not drives.empty:
        drive_stats = drives.groupby(['game_id', 'team_id']).agg({
            'yards': ['sum', 'count'], 'is_score': 'sum'
        }).reset_index()
        drive_stats.columns = ['game_id', 'team_id', 'total_yards', 'num_drives', 'scores']
        drive_stats['ppd'] = drive_stats['scores'] * 7 / drive_stats['num_drives'].replace(0, 1)

        home_drives = drive_stats[['game_id', 'team_id', 'ppd']].copy()
        home_drives.columns = ['game_id', 'team_id', 'home_ppd']
        games = games.merge(home_drives, left_on=['game_id', 'home_team_id'],
                           right_on=['game_id', 'team_id'], how='left').drop(columns=['team_id'], errors='ignore')

        away_drives = drive_stats[['game_id', 'team_id', 'ppd']].copy()
        away_drives.columns = ['game_id', 'team_id', 'away_ppd']
        games = games.merge(away_drives, left_on=['game_id', 'away_team_id'],
                           right_on=['game_id', 'team_id'], how='left').drop(columns=['team_id'], errors='ignore')

    games['day_of_week'] = games['date'].str[:10].apply(
        lambda x: pd.to_datetime(x).dayofweek if pd.notna(x) else 6
    )
    games['is_primetime'] = games['day_of_week'].isin([0, 3])

    print(f"Loaded {len(games)} games (2022-2025)")

    predictor = DualModelPredictor()
    recent_spread_results = []
    updates = []
    seasons = sorted(games['season'].unique())

    print("\nProcessing games...")

    for season in seasons:
        if season > seasons[0]:
            predictor.set_previous_season(season)

        season_games = games[games['season'] == season].copy()

        for _, g in season_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            vegas_spread = g['vegas_spread']
            vegas_total = g['vegas_total']
            is_completed = g['completed'] == 1
            neutral = g['neutral_site'] == 1
            is_dome = g.get('is_dome', 0) == 1
            is_primetime = g.get('is_primetime', False)
            week = g['week']

            preds = predictor.predict(
                hid, aid, season, g['date'], week,
                is_dome=is_dome, is_primetime=is_primetime, neutral_site=neutral
            )

            # Accumulate training
            if is_completed:
                actual_spread = g['away_score'] - g['home_score']
                actual_total = g['home_score'] + g['away_score']

                sfeat = predictor.extract_simple_spread_features(hid, aid, season, g['date'], week)
                if sfeat is not None:
                    predictor.simple_spread_X.append(sfeat)
                    predictor.simple_spread_y.append(actual_spread)
                stfeat = predictor.extract_simple_total_features(hid, aid, season, g['date'], week)
                if stfeat is not None:
                    predictor.simple_total_X.append(stfeat)
                    predictor.simple_total_y.append(actual_total)
                efeat = predictor.extract_enhanced_spread_features(
                    hid, aid, season, g['date'], week,
                    is_dome=is_dome, is_primetime=is_primetime, neutral_site=neutral
                )
                if efeat is not None:
                    predictor.enhanced_spread_X.append(efeat)
                    predictor.enhanced_spread_y.append(actual_spread)
                etfeat = predictor.extract_enhanced_total_features(
                    hid, aid, season, g['date'], week, is_dome=is_dome
                )
                if etfeat is not None:
                    predictor.enhanced_total_X.append(etfeat)
                    predictor.enhanced_total_y.append(actual_total)

                if len(predictor.simple_spread_X) % 50 == 0:
                    predictor.train_models()

            # For 2025 games with predictions
            if season == 2025 and all(preds[k] is not None for k in
                                       ['simple_spread', 'simple_total', 'enhanced_spread', 'enhanced_total']):
                if pd.notna(vegas_spread) and pd.notna(vegas_total):
                    simple_spread_edge = preds['simple_spread'] - vegas_spread
                    enhanced_spread_edge = preds['enhanced_spread'] - vegas_spread
                    avg_spread = (preds['simple_spread'] + preds['enhanced_spread']) / 2
                    avg_total = (preds['simple_total'] + preds['enhanced_total']) / 2
                    avg_spread_edge = avg_spread - vegas_spread

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
                        'spread_agreement': 1 if simple_spread_edge * enhanced_spread_edge > 0 else 0,
                        'spread_model_diff': abs(preds['simple_spread'] - preds['enhanced_spread']),
                        'vegas_spread': vegas_spread,
                        'vegas_spread_abs': abs(vegas_spread),
                        'big_favorite': 1 if abs(vegas_spread) > 7 else 0,
                        'close_game': 1 if abs(vegas_spread) < 3.5 else 0,
                        'recent_spread_acc': np.mean(recent_spread_results[-20:]) if recent_spread_results else 0.5,
                        'home_games': min(hs['games'] / 10, 1),
                        'away_games': min(aws['games'] / 10, 1),
                        'combined_games': min((hs['games'] + aws['games']) / 20, 1),
                        'home_rest': hr,
                        'away_rest': ar,
                        'rest_diff': hr - ar,
                        'home_post_bye': 1 if hr >= 13 else 0,
                        'away_post_bye': 1 if ar >= 13 else 0,
                        'bye_advantage': (1 if hr >= 13 else 0) - (1 if ar >= 13 else 0),
                        'week_normalized': min(week / 17.0, 1.0),
                        'early_season': 1 if week <= 4 else 0,
                        'mid_season': 1 if 5 <= week <= 12 else 0,
                        'late_season': 1 if week >= 13 else 0,
                        'neutral_site': 1 if neutral else 0,
                        'is_dome': 1 if is_dome else 0,
                        'is_primetime': 1 if is_primetime else 0,
                    }

                    X = np.array([[meta.get(col, 0) for col in feature_cols]])
                    X_scaled = edge_scaler.transform(X)

                    with torch.no_grad():
                        logits = edge_model(torch.FloatTensor(X_scaled))
                        probs = torch.softmax(logits, dim=1).numpy()[0]

                    if probs[2] >= 0.4:
                        action = 'FADE'
                        confidence = probs[2]
                    elif probs[1] >= 0.5:
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

            # Update state
            if is_completed:
                predictor.update(
                    hid, season, g['date'], g['home_score'], g['away_score'],
                    is_home=not neutral,
                    yards=g['home_yards'], turnovers=g['home_to'],
                    ppd=g.get('home_ppd')
                )
                predictor.update(
                    aid, season, g['date'], g['away_score'], g['home_score'],
                    is_home=False,
                    yards=g['away_yards'], turnovers=g['away_to'],
                    ppd=g.get('away_ppd')
                )

                if all(preds[k] is not None for k in ['simple_spread', 'enhanced_spread']) and pd.notna(vegas_spread):
                    actual_spread = g['away_score'] - g['home_score']
                    avg_spread_edge = ((preds['simple_spread'] + preds['enhanced_spread']) / 2) - vegas_spread
                    spread_result = actual_spread - vegas_spread
                    if avg_spread_edge > 0:
                        recent_spread_results.append(1 if spread_result > 0.5 else 0)
                    else:
                        recent_spread_results.append(1 if spread_result < -0.5 else 0)

    print(f"\nGenerated {len(updates)} predictions to update")

    # Update database
    update_count = 0
    for upd in updates:
        cursor.execute('''
            UPDATE odds_and_predictions
            SET spread_action = ?, spread_confidence = ?, spread_edge = ?,
                simple_pred_spread = ?, enhanced_pred_spread = ?, avg_pred_spread = ?,
                simple_pred_total = ?, enhanced_pred_total = ?, avg_pred_total = ?
            WHERE game_id = ?
        ''', (
            upd['spread_action'], upd['spread_confidence'], upd['spread_edge'],
            upd['simple_pred_spread'], upd['enhanced_pred_spread'], upd['avg_pred_spread'],
            upd['simple_pred_total'], upd['enhanced_pred_total'], upd['avg_pred_total'],
            upd['game_id'],
        ))
        update_count += cursor.rowcount

    conn.commit()
    conn.close()

    print(f"Updated {update_count} rows in database")

    actions = pd.DataFrame(updates)
    if not actions.empty:
        print("\nAction Distribution:")
        print(actions['spread_action'].value_counts())
        print(f"\nAverage confidence: {actions['spread_confidence'].mean():.3f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
