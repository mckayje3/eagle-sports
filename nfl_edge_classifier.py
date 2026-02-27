"""
NFL Edge Classifier - Neural network to identify profitable betting opportunities.

Takes predictions from Simple and Enhanced Ridge models plus Vegas context
to classify games into: PASS, BET_WITH, or FADE.

NFL-specific features:
- Bye week indicators
- Week of season (early vs late)
- Primetime game indicator
- Dome/outdoor
- Model agreement/disagreement patterns
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DB_PATH = Path(__file__).parent / 'nfl_games.db'
MODEL_DIR = Path(__file__).parent / 'models'

# Model constants (must match training scripts) - OPTIMIZED
SIMPLE_DECAY = 0.96  # Higher decay works better in NFL
ENHANCED_DECAY = 0.96  # Both models use same decay now
PREV_HALF_LIFE = 4.0
MIN_GAMES = 2  # Lower threshold for better coverage


class DualModelPredictor:
    """Maintains state for both Simple and Enhanced NFL models."""

    def __init__(self):
        # Simple model - yards-focused features (optimized)
        self.simple_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'yards': [], 'yards_wts': [],
            'pass_yards': [], 'pass_wts': [],
            'rush_yards': [], 'rush_wts': [],
            'turnovers': [], 'to_wts': [],
            'first_downs': [], 'fd_wts': [],
        }))

        # Enhanced model - yards + drive efficiency + form
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
            'ppg': 22.0, 'papg': 22.0, 'yards': 330.0,
            'pass_yards': 220.0, 'rush_yards': 110.0,
            'turnovers': 1.3, 'first_downs': 20.0,
            'ppd': 1.9, 'ypd': 28.0, 'scoring_pct': 0.35
        }

        # Ridge models
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
            return 7
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return min((curr - last).days, 14)
        except Exception:
            return 7

    def _get_simple_stats(self, tid, season):
        """Get simple model stats - yards-focused features."""
        td = self.simple_stats[tid][season]
        n = len(td['ppg'])
        if n == 0:
            prev = self.prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', self.league_avg['ppg']),
                'papg': prev.get('papg', self.league_avg['papg']),
                'yards': prev.get('yards', self.league_avg['yards']),
                'pass_yards': prev.get('pass_yards', self.league_avg['pass_yards']),
                'rush_yards': prev.get('rush_yards', self.league_avg['rush_yards']),
                'turnovers': prev.get('turnovers', self.league_avg['turnovers']),
                'first_downs': prev.get('first_downs', self.league_avg['first_downs']),
                'games': 0,
            }

        ppg = self._wavg(td['ppg'], td['wts'])
        papg = self._wavg(td['papg'], td['wts'])
        yards = self._wavg(td['yards'], td['yards_wts']) if td['yards'] else None
        pass_yds = self._wavg(td['pass_yards'], td['pass_wts']) if td['pass_yards'] else None
        rush_yds = self._wavg(td['rush_yards'], td['rush_wts']) if td['rush_yards'] else None
        to = self._wavg(td['turnovers'], td['to_wts']) if td['turnovers'] else None
        fd = self._wavg(td['first_downs'], td['fd_wts']) if td['first_downs'] else None

        prev = self.prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', self.league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', self.league_avg['papg']) + (1 - blend) * papg,
            'yards': yards if yards else prev.get('yards', self.league_avg['yards']),
            'pass_yards': pass_yds if pass_yds else prev.get('pass_yards', self.league_avg['pass_yards']),
            'rush_yards': rush_yds if rush_yds else prev.get('rush_yards', self.league_avg['rush_yards']),
            'turnovers': to if to else prev.get('turnovers', self.league_avg['turnovers']),
            'first_downs': fd if fd else prev.get('first_downs', self.league_avg['first_downs']),
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
                'yards': prev.get('yards', self.league_avg['yards']),
                'turnovers': prev.get('turnovers', self.league_avg['turnovers']),
                'ppd': prev.get('ppd', self.league_avg['ppd']),
                'ypd': prev.get('ypd', self.league_avg['ypd']),
                'scoring_pct': prev.get('scoring_pct', self.league_avg['scoring_pct']),
                'games': 0,
                'margins': [],
                'wins': [],
            }

        ppg = self._wavg(td['ppg'], td['wts'])
        papg = self._wavg(td['papg'], td['wts'])
        yards = self._wavg(td['yards'], td['yards_wts']) if td['yards'] else None
        to = self._wavg(td['turnovers'], td['to_wts']) if td['turnovers'] else None
        ppd = self._wavg(td['ppd'], td['ppd_wts']) if td['ppd'] else None
        ypd = self._wavg(td['ypd'], td['ypd_wts']) if td['ypd'] else None
        sp = self._wavg(td['scoring_pct'], td['sp_wts']) if td['scoring_pct'] else None

        prev = self.prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', self.league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', self.league_avg['papg']) + (1 - blend) * papg,
            'yards': yards if yards else prev.get('yards', self.league_avg['yards']),
            'turnovers': to if to else prev.get('turnovers', self.league_avg['turnovers']),
            'ppd': ppd if ppd else prev.get('ppd', self.league_avg['ppd']),
            'ypd': ypd if ypd else prev.get('ypd', self.league_avg['ypd']),
            'scoring_pct': sp if sp else prev.get('scoring_pct', self.league_avg['scoring_pct']),
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

    def _momentum(self, margins, n=4):
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

    def extract_simple_spread_features(self, hid, aid, season, date, week):
        """Extract simple spread features - yards-focused (optimized)."""
        hs = self._get_simple_stats(hid, season)
        aws = self._get_simple_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        h_post_bye = 1.0 if hr >= 13 else 0.0
        a_post_bye = 1.0 if ar >= 13 else 0.0

        # Optimized yards-focused features
        return np.array([
            # Yards differentials (best performing combo)
            hs['yards'] - aws['yards'],
            hs['pass_yards'] - aws['pass_yards'],
            hs['rush_yards'] - aws['rush_yards'],  # Highest correlation

            # Supporting features
            hs['turnovers'] - aws['turnovers'],
            hs['first_downs'] - aws['first_downs'],

            # Rest and bye advantages
            min(hr, 10) - min(ar, 10),
            h_post_bye - a_post_bye,

            # Reliability
            min(hs['games'] / 10.0, 1.0),
            min(aws['games'] / 10.0, 1.0),
        ])

    def extract_simple_total_features(self, hid, aid, season, date, week):
        hs = self._get_simple_stats(hid, season)
        aws = self._get_simple_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            hs['ppg'] + aws['ppg'],
            hs['papg'] + aws['papg'],
            hs['yards'] + aws['yards'],
            hs['turnovers'] + aws['turnovers'],
            1.0 if hr >= 13 else 0.0,
            1.0 if ar >= 13 else 0.0,
            min(hs['games'] / 10.0, 1.0),
            min(aws['games'] / 10.0, 1.0),
        ])

    def extract_enhanced_spread_features(self, hid, aid, season, date, week,
                                          is_dome=False, is_primetime=False, neutral_site=False):
        """Extract enhanced spread features - yards + drive efficiency + form (optimized)."""
        hs = self._get_enhanced_stats(hid, season)
        aws = self._get_enhanced_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        hca = self._get_dynamic_hca(hid, season, neutral_site)
        h_post_bye = 1.0 if hr >= 13 else 0.0
        a_post_bye = 1.0 if ar >= 13 else 0.0

        return np.array([
            # Yards differentials (best performing in analysis)
            hs['yards'] - aws['yards'],
            hs['ypd'] - aws['ypd'],  # Key: yards per drive

            # Drive efficiency (also showed value)
            hs['scoring_pct'] - aws['scoring_pct'],
            hs['ppd'] - aws['ppd'],

            # Turnovers (supporting feature)
            hs['turnovers'] - aws['turnovers'],

            # Form and momentum
            self._recent_form(hs['margins']) - self._recent_form(aws['margins']),
            self._momentum(hs['margins']) - self._momentum(aws['margins']),
            self._streak(hs['wins']) - self._streak(aws['wins']),

            # Rest and bye
            min(hr, 10) - min(ar, 10),
            h_post_bye - a_post_bye,

            # HCA
            hca,

            # Context
            1.0 if is_dome else 0.0,
            1.0 if is_primetime else 0.0,
            min(week / 17.0, 1.0),  # Season progress

            # Reliability
            min(hs['games'] / 10.0, 1.0),
            min(aws['games'] / 10.0, 1.0),
        ])

    def extract_enhanced_total_features(self, hid, aid, season, date, week, is_dome=False):
        hs = self._get_enhanced_stats(hid, season)
        aws = self._get_enhanced_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            hs['ppg'] + aws['ppg'],
            hs['papg'] + aws['papg'],
            hs['yards'] + aws['yards'],
            hs['turnovers'] + aws['turnovers'],
            hs['ppd'] + aws['ppd'],
            hs['scoring_pct'] + aws['scoring_pct'],
            abs(self._recent_form(hs['margins'])) + abs(self._recent_form(aws['margins'])),
            1.0 if hr >= 13 else 0.0,
            1.0 if ar >= 13 else 0.0,
            1.0 if is_dome else 0.0,
            min(week / 17.0, 1.0),
            min(hs['games'] / 10.0, 1.0),
            min(aws['games'] / 10.0, 1.0),
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
            hid, aid, season, date, week,
            is_dome=kwargs.get('is_dome', False)
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

    def update(self, tid, season, date, pf, pa, is_home, yards=None, pass_yards=None,
               rush_yards=None, turnovers=None, first_downs=None,
               ppd=None, ypd=None, scoring_pct=None):
        margin = pf - pa

        # Simple stats - yards-focused features
        ts = self.simple_stats[tid][season]
        ts['wts'] = [w * SIMPLE_DECAY for w in ts['wts']]
        ts['ppg'].append(pf)
        ts['papg'].append(pa)
        ts['wts'].append(1.0)
        if pd.notna(yards):
            ts['yards_wts'] = [w * SIMPLE_DECAY for w in ts['yards_wts']]
            ts['yards'].append(yards)
            ts['yards_wts'].append(1.0)
        if pd.notna(pass_yards):
            ts['pass_wts'] = [w * SIMPLE_DECAY for w in ts['pass_wts']]
            ts['pass_yards'].append(pass_yards)
            ts['pass_wts'].append(1.0)
        if pd.notna(rush_yards):
            ts['rush_wts'] = [w * SIMPLE_DECAY for w in ts['rush_wts']]
            ts['rush_yards'].append(rush_yards)
            ts['rush_wts'].append(1.0)
        if pd.notna(turnovers):
            ts['to_wts'] = [w * SIMPLE_DECAY for w in ts['to_wts']]
            ts['turnovers'].append(turnovers)
            ts['to_wts'].append(1.0)
        if pd.notna(first_downs):
            ts['fd_wts'] = [w * SIMPLE_DECAY for w in ts['fd_wts']]
            ts['first_downs'].append(first_downs)
            ts['fd_wts'].append(1.0)

        # Enhanced stats - yards + drive efficiency + form
        te = self.enhanced_stats[tid][season]
        te['wts'] = [w * ENHANCED_DECAY for w in te['wts']]
        te['ppg'].append(pf)
        te['papg'].append(pa)
        te['wts'].append(1.0)
        te['margins'].append(margin)
        te['wins'].append(1 if margin > 0 else 0)
        if pd.notna(yards):
            te['yards_wts'] = [w * ENHANCED_DECAY for w in te['yards_wts']]
            te['yards'].append(yards)
            te['yards_wts'].append(1.0)
        if pd.notna(turnovers):
            te['to_wts'] = [w * ENHANCED_DECAY for w in te['to_wts']]
            te['turnovers'].append(turnovers)
            te['to_wts'].append(1.0)
        if pd.notna(ppd):
            te['ppd_wts'] = [w * ENHANCED_DECAY for w in te['ppd_wts']]
            te['ppd'].append(ppd)
            te['ppd_wts'].append(1.0)
        if pd.notna(ypd):
            te['ypd_wts'] = [w * ENHANCED_DECAY for w in te['ypd_wts']]
            te['ypd'].append(ypd)
            te['ypd_wts'].append(1.0)
        if pd.notna(scoring_pct):
            te['sp_wts'] = [w * ENHANCED_DECAY for w in te['sp_wts']]
            te['scoring_pct'].append(scoring_pct)
            te['sp_wts'].append(1.0)

        # HCA data
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
                        'pass_yards': np.mean(td['pass_yards']) if td['pass_yards'] else 220.0,
                        'rush_yards': np.mean(td['rush_yards']) if td['rush_yards'] else 110.0,
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 1.3,
                        'first_downs': np.mean(td['first_downs']) if td['first_downs'] else 20.0,
                    }

        for tid in self.enhanced_stats:
            if prev in self.enhanced_stats[tid]:
                te = self.enhanced_stats[tid][prev]
                if tid not in self.prev_ratings:
                    self.prev_ratings[tid] = {}
                if te['ppd']:
                    self.prev_ratings[tid]['ppd'] = np.mean(te['ppd'])
                if te['ypd']:
                    self.prev_ratings[tid]['ypd'] = np.mean(te['ypd'])
                if te['scoring_pct']:
                    self.prev_ratings[tid]['scoring_pct'] = np.mean(te['scoring_pct'])

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
            X_s = self.simple_spread_scaler.transform(X)
            self.simple_spread_model = Ridge(alpha=1.0).fit(X_s, y)

        if len(self.simple_total_X) >= 50:
            X = np.array(self.simple_total_X)
            y = np.array(self.simple_total_y)
            self.simple_total_scaler.fit(X)
            X_s = self.simple_total_scaler.transform(X)
            self.simple_total_model = Ridge(alpha=1.0).fit(X_s, y)

        if len(self.enhanced_spread_X) >= 50:
            X = np.array(self.enhanced_spread_X)
            y = np.array(self.enhanced_spread_y)
            self.enhanced_spread_scaler.fit(X)
            X_s = self.enhanced_spread_scaler.transform(X)
            self.enhanced_spread_model = Ridge(alpha=1.0).fit(X_s, y)

        if len(self.enhanced_total_X) >= 50:
            X = np.array(self.enhanced_total_X)
            y = np.array(self.enhanced_total_y)
            self.enhanced_total_scaler.fit(X)
            X_s = self.enhanced_total_scaler.transform(X)
            self.enhanced_total_model = Ridge(alpha=1.0).fit(X_s, y)


class EdgeClassifier(nn.Module):
    """Neural network for classifying betting opportunities."""

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
            nn.Linear(16, 3),  # PASS, BET_WITH, FADE
        )

    def forward(self, x):
        shared = self.shared(x)
        return self.spread_head(shared)


def generate_training_data():
    """Generate training data with dual model predictions."""
    print("=" * 70)
    print("GENERATING NFL TRAINING DATA (Dual Model)")
    print("=" * 70)

    conn = sqlite3.connect(str(DB_PATH))

    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site, g.is_dome,
               hs.total_yards as home_yards, hs.passing_yards as home_pass_yards,
               hs.rushing_yards as home_rush_yards, hs.turnovers as home_to,
               hs.first_downs as home_fd,
               aws.total_yards as away_yards, aws.passing_yards as away_pass_yards,
               aws.rushing_yards as away_rush_yards, aws.turnovers as away_to,
               aws.first_downs as away_fd,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id
            AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id
            AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date, g.week
    ''', conn)

    # Load drive stats
    drives = pd.read_sql_query('''
        SELECT d.game_id, d.team_id, d.yards, d.is_score
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE g.completed = 1
    ''', conn)
    conn.close()

    # Calculate drive efficiency
    if not drives.empty:
        drive_stats = drives.groupby(['game_id', 'team_id']).agg({
            'yards': ['sum', 'count'],
            'is_score': 'sum'
        }).reset_index()
        drive_stats.columns = ['game_id', 'team_id', 'total_yards', 'num_drives', 'scores']
        drive_stats['ppd'] = drive_stats['scores'] * 7 / drive_stats['num_drives'].replace(0, 1)
        drive_stats['ypd'] = drive_stats['total_yards'] / drive_stats['num_drives'].replace(0, 1)
        drive_stats['scoring_pct'] = drive_stats['scores'] / drive_stats['num_drives'].replace(0, 1)

        # Merge home drives
        home_drives = drive_stats[['game_id', 'team_id', 'ppd', 'ypd', 'scoring_pct']].copy()
        home_drives.columns = ['game_id', 'team_id', 'home_ppd', 'home_ypd', 'home_scoring_pct']
        games = games.merge(
            home_drives, left_on=['game_id', 'home_team_id'],
            right_on=['game_id', 'team_id'], how='left'
        ).drop(columns=['team_id'], errors='ignore')

        # Merge away drives
        away_drives = drive_stats[['game_id', 'team_id', 'ppd', 'ypd', 'scoring_pct']].copy()
        away_drives.columns = ['game_id', 'team_id', 'away_ppd', 'away_ypd', 'away_scoring_pct']
        games = games.merge(
            away_drives, left_on=['game_id', 'away_team_id'],
            right_on=['game_id', 'team_id'], how='left'
        ).drop(columns=['team_id'], errors='ignore')

    # Day of week
    games['day_of_week'] = games['date'].str[:10].apply(
        lambda x: pd.to_datetime(x).dayofweek if pd.notna(x) else 6
    )
    games['is_primetime'] = games['day_of_week'].isin([0, 3])  # Mon, Thu

    predictor = DualModelPredictor()
    training_data = []
    recent_spread_results = []
    seasons = sorted(games['season'].unique())

    for season in seasons:
        if season > seasons[0]:
            predictor.set_previous_season(season)

        season_games = games[games['season'] == season].copy()

        for _, g in season_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_spread = g['away_score'] - g['home_score']
            actual_total = g['home_score'] + g['away_score']
            vegas_spread = g['vegas_spread']
            vegas_total = g['vegas_total']
            neutral = g['neutral_site'] == 1
            is_dome = g.get('is_dome', 0) == 1
            is_primetime = g.get('is_primetime', False)
            week = g['week']

            # Get predictions
            preds = predictor.predict(
                hid, aid, season, g['date'], week,
                is_dome=is_dome, is_primetime=is_primetime, neutral_site=neutral
            )

            # Accumulate training data for models
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

            # Retrain periodically
            if len(predictor.simple_spread_X) % 50 == 0:
                predictor.train_models()

            # Record for edge classifier if we have predictions and Vegas
            if all(preds[k] is not None for k in
                   ['simple_spread', 'simple_total', 'enhanced_spread', 'enhanced_total']):
                if pd.notna(vegas_spread) and pd.notna(vegas_total):
                    simple_spread_edge = preds['simple_spread'] - vegas_spread
                    enhanced_spread_edge = preds['enhanced_spread'] - vegas_spread
                    avg_spread = (preds['simple_spread'] + preds['enhanced_spread']) / 2
                    avg_spread_edge = avg_spread - vegas_spread

                    spread_result = actual_spread - vegas_spread
                    push = abs(spread_result) < 0.5

                    if avg_spread_edge > 0:
                        spread_cover = 1 if spread_result > 0.5 else (0 if spread_result < -0.5 else None)
                    else:
                        spread_cover = 1 if spread_result < -0.5 else (0 if spread_result > 0.5 else None)

                    if not push:
                        recent_spread_results.append(1 if spread_cover == 1 else 0)

                    hs = predictor._get_simple_stats(hid, season)
                    aws = predictor._get_simple_stats(aid, season)
                    hr = predictor._get_rest(hid, g['date'])
                    ar = predictor._get_rest(aid, g['date'])

                    training_data.append({
                        'game_id': g['game_id'],
                        'season': season,
                        'week': week,
                        # Spread predictions
                        'simple_spread': preds['simple_spread'],
                        'enhanced_spread': preds['enhanced_spread'],
                        'simple_spread_edge': simple_spread_edge,
                        'enhanced_spread_edge': enhanced_spread_edge,
                        'simple_spread_edge_abs': abs(simple_spread_edge),
                        'enhanced_spread_edge_abs': abs(enhanced_spread_edge),
                        'avg_spread_edge': avg_spread_edge,
                        'avg_spread_edge_abs': abs(avg_spread_edge),
                        # Model agreement
                        'spread_agreement': 1 if simple_spread_edge * enhanced_spread_edge > 0 else 0,
                        'spread_model_diff': abs(preds['simple_spread'] - preds['enhanced_spread']),
                        # Vegas context
                        'vegas_spread': vegas_spread,
                        'vegas_spread_abs': abs(vegas_spread),
                        'big_favorite': 1 if abs(vegas_spread) > 7 else 0,
                        'close_game': 1 if abs(vegas_spread) < 3.5 else 0,
                        # Recent accuracy
                        'recent_spread_acc': np.mean(recent_spread_results[-20:]) if recent_spread_results else 0.5,
                        # Team context
                        'home_games': min(hs['games'] / 10, 1),
                        'away_games': min(aws['games'] / 10, 1),
                        'combined_games': min((hs['games'] + aws['games']) / 20, 1),
                        # Rest/bye
                        'home_rest': hr,
                        'away_rest': ar,
                        'rest_diff': hr - ar,
                        'home_post_bye': 1 if hr >= 13 else 0,
                        'away_post_bye': 1 if ar >= 13 else 0,
                        'bye_advantage': (1 if hr >= 13 else 0) - (1 if ar >= 13 else 0),
                        # Game context
                        'week_normalized': min(week / 17.0, 1.0),
                        'early_season': 1 if week <= 4 else 0,
                        'mid_season': 1 if 5 <= week <= 12 else 0,
                        'late_season': 1 if week >= 13 else 0,
                        'neutral_site': 1 if neutral else 0,
                        'is_dome': 1 if is_dome else 0,
                        'is_primetime': 1 if is_primetime else 0,
                        # Targets
                        'spread_cover': spread_cover,
                        'actual_spread': actual_spread,
                    })

            # Update state with new yards-focused features
            predictor.update(
                hid, season, g['date'], g['home_score'], g['away_score'],
                is_home=not neutral,
                yards=g['home_yards'], pass_yards=g['home_pass_yards'],
                rush_yards=g['home_rush_yards'], turnovers=g['home_to'],
                first_downs=g['home_fd'],
                ppd=g.get('home_ppd'), ypd=g.get('home_ypd'),
                scoring_pct=g.get('home_scoring_pct')
            )
            predictor.update(
                aid, season, g['date'], g['away_score'], g['home_score'],
                is_home=False,
                yards=g['away_yards'], pass_yards=g['away_pass_yards'],
                rush_yards=g['away_rush_yards'], turnovers=g['away_to'],
                first_downs=g['away_fd'],
                ppd=g.get('away_ppd'), ypd=g.get('away_ypd'),
                scoring_pct=g.get('away_scoring_pct')
            )

    df = pd.DataFrame(training_data)
    df = df.dropna(subset=['spread_cover'])
    print(f"Games with Vegas lines: {len(games[games['vegas_spread'].notna()])}")
    print(f"Training samples: {len(df)}")
    df.to_csv('nfl_edge_training_data.csv', index=False)
    print("Saved to nfl_edge_training_data.csv")

    return df


def analyze_patterns(df: pd.DataFrame):
    """Analyze betting patterns."""
    print("\n" + "=" * 70)
    print("NFL SPREAD PATTERN ANALYSIS")
    print("=" * 70)

    # Model agreement
    print("\n1. MODEL AGREEMENT:")
    for agree in [0, 1]:
        mask = df['spread_agreement'] == agree
        n = mask.sum()
        wr = df.loc[mask, 'spread_cover'].mean() * 100
        label = "AGREE" if agree == 1 else "DISAGREE"
        print(f"  Models {label}: {n} games, WR: {wr:.1f}%")

    # Edge size
    print("\n2. BY SPREAD EDGE SIZE:")
    for low, high in [(0, 2), (2, 4), (4, 6), (6, 100)]:
        mask = (df['avg_spread_edge_abs'] >= low) & (df['avg_spread_edge_abs'] < high)
        n = mask.sum()
        if n > 0:
            with_wr = df.loc[mask & (df['avg_spread_edge'] > 0), 'spread_cover'].mean() * 100
            fade_wr = df.loc[mask & (df['avg_spread_edge'] < 0), 'spread_cover'].mean() * 100
            print(f"  Edge {low}-{high}: {n} games, with: {with_wr:.1f}%, fade: {fade_wr:.1f}%")

    # By week range
    print("\n3. BY WEEK:")
    for low, high in [(1, 4), (5, 9), (10, 14), (15, 18)]:
        mask = (df['week'] >= low) & (df['week'] <= high)
        n = mask.sum()
        if n > 0:
            wr = df.loc[mask, 'spread_cover'].mean() * 100
            print(f"  Weeks {low}-{high}: {n} games, WR: {wr:.1f}%")

    # Bye advantage
    print("\n4. BYE WEEK ADVANTAGE:")
    for adv in [-1, 0, 1]:
        mask = df['bye_advantage'] == adv
        n = mask.sum()
        if n > 0:
            wr = df.loc[mask, 'spread_cover'].mean() * 100
            label = {-1: "Away bye", 0: "Neither", 1: "Home bye"}[adv]
            print(f"  {label}: {n} games, WR: {wr:.1f}%")

    # Primetime
    print("\n5. PRIMETIME:")
    for pt in [0, 1]:
        mask = df['is_primetime'] == pt
        n = mask.sum()
        if n > 0:
            wr = df.loc[mask, 'spread_cover'].mean() * 100
            label = "Primetime" if pt == 1 else "Regular"
            print(f"  {label}: {n} games, WR: {wr:.1f}%")


def train_edge_classifier(df: pd.DataFrame):
    """Train the edge classifier."""
    print("\n" + "=" * 70)
    print("TRAINING NFL EDGE CLASSIFIER (SPREAD ONLY)")
    print("=" * 70)

    feature_cols = [
        'simple_spread_edge', 'simple_spread_edge_abs', 'simple_spread',
        'enhanced_spread_edge', 'enhanced_spread_edge_abs', 'enhanced_spread',
        'avg_spread_edge', 'avg_spread_edge_abs',
        'spread_agreement', 'spread_model_diff',
        'vegas_spread', 'vegas_spread_abs', 'big_favorite', 'close_game',
        'recent_spread_acc',
        'home_games', 'away_games', 'combined_games',
        'home_rest', 'away_rest', 'rest_diff',
        'home_post_bye', 'away_post_bye', 'bye_advantage',
        'week_normalized', 'early_season', 'mid_season', 'late_season',
        'neutral_site', 'is_dome', 'is_primetime',
    ]

    # Split by season (train on 2022-2024, test on 2025)
    train_df = df[df['season'] < 2025].copy()
    test_df = df[df['season'] == 2025].copy()

    print(f"Train: {len(train_df)}, Test: {len(test_df)} (season 2025)")
    print(f"Features: {len(feature_cols)}")

    # Create labels
    def get_label(row):
        if row['avg_spread_edge'] > 0:
            return 1 if row['spread_cover'] == 1 else 2  # BET_WITH or FADE
        else:
            return 2 if row['spread_cover'] == 1 else 1  # FADE or BET_WITH

    train_df['label'] = train_df.apply(get_label, axis=1)

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train
    model = EdgeClassifier(input_dim=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.FloatTensor(X_train_scaled)
    y_tensor = torch.LongTensor(y_train)

    print(f"\nTraining for 150 epochs...")
    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET (2025 Season)")
    print("=" * 70)

    X_test = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test)

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test_scaled))
        probs = torch.softmax(logits, dim=1).numpy()

    test_df = test_df.copy()
    test_df['prob_pass'] = probs[:, 0]
    test_df['prob_with'] = probs[:, 1]
    test_df['prob_fade'] = probs[:, 2]

    print("\nSPREAD BETTING:\n")
    print(f"{'Threshold':<12} {'Action':<12} {'Bets':>6} {'Wins':>6} {'Loss':>6} {'Win%':>8} {'ROI':>10}")
    print("-" * 70)

    best = {'threshold': 0, 'action': '', 'roi': -999, 'record': ''}

    for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for action in ['BET_WITH', 'FADE']:
            prob_col = 'prob_with' if action == 'BET_WITH' else 'prob_fade'
            mask = test_df[prob_col] >= thresh
            n = mask.sum()
            if n == 0:
                continue

            subset = test_df[mask]
            if action == 'BET_WITH':
                wins = (subset['avg_spread_edge'] * (subset['actual_spread'] - subset['vegas_spread']) > 0).sum()
            else:
                wins = (subset['avg_spread_edge'] * (subset['actual_spread'] - subset['vegas_spread']) < 0).sum()

            losses = n - wins
            wr = wins / n * 100
            roi = (wins - losses * 1.1) / n * 100

            marker = " ***" if roi > 0 else ""
            print(f">= {thresh:<10} {action:<12} {n:>6} {wins:>6} {losses:>6} {wr:>7.1f}% {roi:>+9.1f}%{marker}")

            if roi > best['roi']:
                best = {'threshold': thresh, 'action': action, 'roi': roi, 'record': f"{wins}-{losses}"}

    print(f"\nBest SPREAD: {best['action']} @ {best['threshold']}")
    print(f"  Record: {best['record']} ({(int(best['record'].split('-')[0]) / sum(map(int, best['record'].split('-')))) * 100:.1f}%)")
    print(f"  ROI: {best['roi']:+.1f}%")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
    }, MODEL_DIR / 'nfl_edge_classifier.pt')
    print(f"\nModel saved to {MODEL_DIR / 'nfl_edge_classifier.pt'}")

    return model, scaler, feature_cols


def main():
    df = generate_training_data()
    analyze_patterns(df)
    train_edge_classifier(df)
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
