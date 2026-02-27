"""
CBB Edge Classifier - Neural Network to Find Betting Edges (SPREAD ONLY)

Uses predictions from BOTH Simple and Enhanced models to identify when to bet.
Similar to NBA edge classifier but adapted for CBB:
- No player injury data
- Neutral site games handled
- Conference game indicator

Architecture:
- Input: Meta-features from both models, Vegas, situational factors
- Output: Probabilities for SPREAD betting decisions only
- Training: Walk-forward to avoid lookahead bias
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
    """
    Maintains state for both Simple and Enhanced CBB models.
    Generates predictions from both for edge classifier training.
    """

    def __init__(self):
        # Simple model state (decay 0.95)
        self.simple_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'fg_pct': [], 'fg_wts': [],
            'rebounds': [], 'reb_wts': [],
            'turnovers': [], 'tov_wts': [],
        }))

        # Enhanced model state (decay 0.92)
        self.enhanced_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'fg_pct': [], 'fg_wts': [],
            'rebounds': [], 'reb_wts': [],
            'turnovers': [], 'tov_wts': [],
            'margins': [], 'wins': [],
        }))

        # Dynamic HCA for enhanced model
        self.team_hca_data = defaultdict(lambda: defaultdict(lambda: {
            'home_margins': [], 'away_margins': []
        }))

        self.prev_ratings = {}
        self.prev_hca = {}
        self.last_game = {}
        self.league_avg = {'ppg': 72.0, 'papg': 72.0}

        # Ridge models (trained walk-forward)
        self.simple_spread_model = None
        self.simple_total_model = None
        self.enhanced_spread_model = None
        self.enhanced_total_model = None

        self.simple_spread_scaler = StandardScaler()
        self.simple_total_scaler = StandardScaler()
        self.enhanced_spread_scaler = StandardScaler()
        self.enhanced_total_scaler = StandardScaler()

        # Training data accumulators
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

    def _get_dynamic_hca(self, home_id, season):
        hd = self.team_hca_data[home_id][season]
        n_home = len(hd['home_margins'])
        n_away = len(hd['away_margins'])
        total = n_home + n_away

        default_hca = 3.5  # CBB default

        if total == 0:
            return self.prev_hca.get(home_id, default_hca)

        if n_home > 0 and n_away > 0:
            raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
            raw = max(-2, min(raw, 10))
        else:
            raw = default_hca

        shrunk = default_hca + 0.5 * (raw - default_hca)
        prev = self.prev_hca.get(home_id, default_hca)
        blend = 0.5 ** (total / 20.0)

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

    def extract_simple_spread_features(self, hid, aid, season, date):
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

    def extract_enhanced_spread_features(self, hid, aid, season, date,
                                          neutral_site=False, conference_game=False,
                                          season_games_played=0):
        hs = self._get_enhanced_stats(hid, season)
        aws = self._get_enhanced_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        net = (hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg'])
        hca = 0.0 if neutral_site else self._get_dynamic_hca(hid, season)

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
            min(hs['games'] / 20.0, 1.0),
            min(aws['games'] / 20.0, 1.0),
            1.0 if neutral_site else 0.0,
            1.0 if conference_game else 0.0,
            min(season_games_played / 5000, 1.0),
            hs['fg_pct'] - aws['fg_pct'],
            hs['turnovers'] - aws['turnovers'],
        ])

    def extract_enhanced_total_features(self, hid, aid, season, date, season_games_played=0):
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
            min(hs['games'] / 20.0, 1.0),
            min(aws['games'] / 20.0, 1.0),
            hs['fg_pct'] + aws['fg_pct'],
            hs['rebounds'] + aws['rebounds'],
            hs['turnovers'] + aws['turnovers'],
            home_recent + away_recent,
            home_mom + away_mom,
            min(season_games_played / 5000, 1.0),
            0.0,
        ])

    def predict(self, hid, aid, season, date, neutral_site=False,
                conference_game=False, season_games_played=0):
        simple_spread_feat = self.extract_simple_spread_features(hid, aid, season, date)
        simple_spread = None
        if simple_spread_feat is not None and self.simple_spread_model is not None:
            X = self.simple_spread_scaler.transform(simple_spread_feat.reshape(1, -1))
            simple_spread = self.simple_spread_model.predict(X)[0]

        simple_total_feat = self.extract_simple_total_features(hid, aid, season, date)
        simple_total = None
        if simple_total_feat is not None and self.simple_total_model is not None:
            X = self.simple_total_scaler.transform(simple_total_feat.reshape(1, -1))
            simple_total = self.simple_total_model.predict(X)[0]

        enhanced_spread_feat = self.extract_enhanced_spread_features(
            hid, aid, season, date, neutral_site, conference_game, season_games_played
        )
        enhanced_spread = None
        if enhanced_spread_feat is not None and self.enhanced_spread_model is not None:
            X = self.enhanced_spread_scaler.transform(enhanced_spread_feat.reshape(1, -1))
            enhanced_spread = self.enhanced_spread_model.predict(X)[0]

        enhanced_total_feat = self.extract_enhanced_total_features(
            hid, aid, season, date, season_games_played
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
            'simple_spread_feat': simple_spread_feat,
            'simple_total_feat': simple_total_feat,
            'enhanced_spread_feat': enhanced_spread_feat,
            'enhanced_total_feat': enhanced_total_feat,
        }

    def update(self, tid, season, date, pf, pa, is_home, fg=None, reb=None, tov=None):
        margin = pf - pa

        # Simple model update
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

        # Enhanced model update
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

        # HCA update
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


class EdgeClassifier(nn.Module):
    """Neural network to predict spread betting edges (spreads only)."""

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
            nn.Linear(16, 3),  # [P(pass), P(bet_with), P(fade)]
        )

    def forward(self, x):
        shared = self.shared(x)
        return self.spread_head(shared)


def generate_training_data():
    """Generate training data using both models."""
    print("=" * 70)
    print("GENERATING CBB TRAINING DATA (Dual Model)")
    print("=" * 70)

    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site, g.conference_game,
               hs.field_goal_pct as home_fg, hs.total_rebounds as home_reb,
               hs.turnovers as home_tov,
               aws.field_goal_pct as away_fg, aws.total_rebounds as away_reb,
               aws.turnovers as away_tov,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id
            AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id
            AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
    ''', conn)
    conn.close()

    # Filter to games with Vegas lines
    games = games[games['vegas_spread'].notna() & games['vegas_total'].notna()].copy()
    print(f"Games with Vegas lines: {len(games)}")

    predictor = DualModelPredictor()

    training_data = []
    recent_spread_results = []
    seasons = sorted(games['season'].unique())
    season_game_counts = {}

    for season in seasons:
        if season > seasons[0]:
            predictor.set_previous_season(season)

        season_game_counts[season] = 0
        season_games = games[games['season'] == season]

        for _, g in season_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_spread = g['away_score'] - g['home_score']
            actual_total = g['home_score'] + g['away_score']
            vegas_spread = g['vegas_spread']
            vegas_total = g['vegas_total']
            neutral = g['neutral_site'] == 1
            conf = g['conference_game'] == 1

            # Get predictions
            preds = predictor.predict(hid, aid, season, g['date'], neutral, conf,
                                       season_game_counts[season])

            # Accumulate training features
            if preds['simple_spread_feat'] is not None:
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

            # Retrain models periodically
            if len(predictor.simple_spread_X) % 200 == 0:
                predictor.train_models()

            # Create training sample if all predictions available
            if all(preds[k] is not None for k in ['simple_spread', 'simple_total',
                                                    'enhanced_spread', 'enhanced_total']):

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
                    'big_favorite': 1 if abs(vegas_spread) > 12 else 0,  # CBB spreads larger
                    'close_game': 1 if abs(vegas_spread) < 4 else 0,
                    'recent_spread_acc': recent_spread_acc,
                    'home_games': min(hs['games'] / 20, 1),
                    'away_games': min(aws['games'] / 20, 1),
                    'combined_games': min((hs['games'] + aws['games']) / 40, 1),
                    'home_rest': hr,
                    'away_rest': ar,
                    'rest_diff': hr - ar,
                    'home_b2b': 1 if hr == 0 else 0,
                    'away_b2b': 1 if ar == 0 else 0,
                    'season_progress': min(season_game_counts[season] / 5000, 1),
                    'early_season': 1 if season_game_counts[season] < 1500 else 0,
                    'mid_season': 1 if 1500 <= season_game_counts[season] < 4000 else 0,
                    'late_season': 1 if season_game_counts[season] >= 4000 else 0,
                    'neutral_site': 1 if neutral else 0,
                    'conference_game': 1 if conf else 0,
                }

                # Labels
                spread_result = actual_spread - vegas_spread
                if avg_spread_edge > 0:
                    spread_with_wins = 1 if spread_result > 0.5 else 0
                    spread_fade_wins = 1 if spread_result < -0.5 else 0
                else:
                    spread_with_wins = 1 if spread_result < -0.5 else 0
                    spread_fade_wins = 1 if spread_result > 0.5 else 0

                recent_spread_results.append(spread_with_wins)

                training_data.append({
                    **meta,
                    'actual_spread': actual_spread,
                    'spread_with_model_wins': spread_with_wins,
                    'spread_fade_model_wins': spread_fade_wins,
                    'season': season,
                })

            # Update predictor state
            predictor.update(hid, season, g['date'], g['home_score'], g['away_score'],
                           is_home=True, fg=g['home_fg'], reb=g['home_reb'], tov=g['home_tov'])
            predictor.update(aid, season, g['date'], g['away_score'], g['home_score'],
                           is_home=False, fg=g['away_fg'], reb=g['away_reb'], tov=g['away_tov'])

            season_game_counts[season] += 1

    df = pd.DataFrame(training_data)
    print(f"Training samples: {len(df)}")

    return df


def train_classifier(df):
    """Train edge classifier (spread only)."""
    print("\n" + "=" * 70)
    print("TRAINING CBB EDGE CLASSIFIER (SPREAD ONLY)")
    print("=" * 70)

    feature_cols = [
        'simple_spread_edge', 'simple_spread_edge_abs', 'simple_spread',
        'enhanced_spread_edge', 'enhanced_spread_edge_abs', 'enhanced_spread',
        'avg_spread_edge', 'avg_spread_edge_abs',
        'spread_agreement', 'spread_model_diff',
        'vegas_spread', 'vegas_spread_abs', 'big_favorite', 'close_game',
        'recent_spread_acc',
        'home_games', 'away_games', 'combined_games',
        'home_rest', 'away_rest', 'rest_diff', 'home_b2b', 'away_b2b',
        'season_progress', 'early_season', 'mid_season', 'late_season',
        'neutral_site', 'conference_game',
    ]

    X = df[feature_cols].values

    y_spread = np.zeros(len(df), dtype=np.int64)
    y_spread[df['spread_with_model_wins'] == 1] = 1
    y_spread[df['spread_fade_model_wins'] == 1] = 2

    # Split - train on 2024-2025, test on 2026
    train_seasons = [2024, 2025]
    test_season = 2026
    train_mask = df['season'].isin(train_seasons)
    test_mask = df['season'] == test_season

    if test_mask.sum() < 50:
        train_seasons = [2024]
        test_season = 2025
        train_mask = df['season'] == 2024
        test_mask = df['season'] == test_season

    X_train, X_test = X[train_mask], X[test_mask]
    y_spread_train, y_spread_test = y_spread[train_mask], y_spread[test_mask]

    print(f"Train: {len(X_train)}, Test: {len(X_test)} (season {test_season})")
    print(f"Features: {len(feature_cols)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train_s)
    X_test_t = torch.FloatTensor(X_test_s)
    y_spread_t = torch.LongTensor(y_spread_train)

    model = EdgeClassifier(input_dim=len(feature_cols), hidden_dims=[64, 32])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    dataset = TensorDataset(X_train_t, y_spread_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"\nTraining for 150 epochs...")
    for epoch in range(150):
        model.train()
        total_loss = 0
        for bX, by_s in loader:
            optimizer.zero_grad()
            spread_out = model(bX)
            loss = criterion(spread_out, by_s)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    # Evaluate
    print("\n" + "=" * 70)
    print(f"EVALUATION ON TEST SET ({test_season} Season)")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        spread_logits = model(X_test_t)
        spread_probs = torch.softmax(spread_logits, dim=1).numpy()

    print("\nSPREAD BETTING:")
    analyze_predictions(spread_probs, y_spread_test, df[test_mask], 'spread')

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
    }, MODEL_DIR / 'cbb_edge_classifier.pt')
    print(f"\nModel saved to {MODEL_DIR / 'cbb_edge_classifier.pt'}")

    return model, scaler, feature_cols


def analyze_predictions(probs, y_true, df_test, bet_type):
    """Analyze betting performance at various thresholds."""
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    print(f"\n{'Threshold':<12} {'Action':<12} {'Bets':>6} {'Wins':>6} {'Loss':>6} {'Win%':>8} {'ROI':>10}")
    print("-" * 70)

    for thresh in thresholds:
        for action_idx, action in [(1, 'BET_WITH'), (2, 'FADE')]:
            mask = probs[:, action_idx] >= thresh
            if mask.sum() == 0:
                continue

            wins = (y_true[mask] == action_idx).sum()
            losses = mask.sum() - wins - (y_true[mask] == 0).sum()
            total = wins + losses

            if total > 0:
                win_pct = wins / total * 100
                roi = (wins * 0.909 - losses) / total * 100
                flag = " ***" if win_pct > 52.4 else ""
                print(f">= {thresh:<9} {action:<12} {mask.sum():>6} {wins:>6} {losses:>6} "
                      f"{win_pct:>7.1f}% {roi:>+9.1f}%{flag}")

    # Best strategy
    best_roi, best = -100, None
    for thresh in thresholds:
        for action_idx, action in [(1, 'BET_WITH'), (2, 'FADE')]:
            mask = probs[:, action_idx] >= thresh
            if mask.sum() < 20:
                continue
            wins = (y_true[mask] == action_idx).sum()
            losses = mask.sum() - wins - (y_true[mask] == 0).sum()
            total = wins + losses
            if total > 0:
                roi = (wins * 0.909 - losses) / total * 100
                win_pct = wins / total * 100
                if roi > best_roi and win_pct > 50:
                    best_roi = roi
                    best = {'thresh': thresh, 'action': action, 'wins': wins,
                           'losses': losses, 'win_pct': win_pct, 'roi': roi}

    if best:
        print(f"\nBest {bet_type.upper()}: {best['action']} @ {best['thresh']}")
        print(f"  Record: {best['wins']}-{best['losses']} ({best['win_pct']:.1f}%)")
        print(f"  ROI: {best['roi']:+.1f}%")


def find_patterns(df):
    """Find interesting patterns in spread data."""
    print("\n" + "=" * 70)
    print("CBB SPREAD PATTERN ANALYSIS")
    print("=" * 70)

    print("\n1. MODEL AGREEMENT:")
    for agree in [0, 1]:
        label = "Models AGREE" if agree else "Models DISAGREE"
        mask = df['spread_agreement'] == agree
        if mask.sum() > 20:
            wr = df.loc[mask, 'spread_with_model_wins'].mean()
            print(f"  {label}: {mask.sum()} games, WR: {wr*100:.1f}%")

    print("\n2. BY SPREAD EDGE SIZE:")
    for lo, hi in [(0, 3), (3, 5), (5, 7), (7, 100)]:
        mask = (df['avg_spread_edge_abs'] >= lo) & (df['avg_spread_edge_abs'] < hi)
        if mask.sum() > 20:
            wr = df.loc[mask, 'spread_with_model_wins'].mean()
            fade_wr = df.loc[mask, 'spread_fade_model_wins'].mean()
            print(f"  Edge {lo}-{hi}: {mask.sum()} games, with: {wr*100:.1f}%, fade: {fade_wr*100:.1f}%")

    print("\n3. BY GAME TYPE:")
    for label, mask in [('Neutral', df['neutral_site'] == 1),
                        ('Home/Away', df['neutral_site'] == 0),
                        ('Conference', df['conference_game'] == 1),
                        ('Non-conf', df['conference_game'] == 0)]:
        if mask.sum() > 20:
            wr = df.loc[mask, 'spread_with_model_wins'].mean()
            print(f"  {label}: {mask.sum()} games, WR: {wr*100:.1f}%")

    print("\n4. BY SEASON SEGMENT:")
    for seg, col in [('Early', 'early_season'), ('Mid', 'mid_season'), ('Late', 'late_season')]:
        mask = df[col] == 1
        if mask.sum() > 20:
            wr = df.loc[mask, 'spread_with_model_wins'].mean()
            print(f"  {seg}: {mask.sum()} games, WR: {wr*100:.1f}%")


def main():
    df = generate_training_data()
    df.to_csv('cbb_edge_training_data.csv', index=False)
    print(f"Saved to cbb_edge_training_data.csv")

    find_patterns(df)
    train_classifier(df)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
