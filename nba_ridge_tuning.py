"""
NBA Ridge Model Comprehensive Tuning

Systematic grid search to minimize MAE on 2024 season.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from itertools import product
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).parent / 'nba_games.db'


@dataclass
class ModelConfig:
    decay: float = 0.95
    hca_shrink: float = 0.4
    alpha: float = 1.0
    min_games: int = 5
    use_defense: bool = True
    use_shooting: bool = True
    use_rest: bool = True
    use_recent_form: bool = True
    recent_window: int = 10


class NBAModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.team_stats = defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'fg': [], 'three': [], 'ft': [],
            'reb': [], 'ast': [], 'tov': [], 'stl': [], 'blk': [],
        })
        self.last_game = {}
        self.team_hca = {}
        self.league_hca = 2.5

    def reset(self):
        self.team_stats.clear()
        self.last_game.clear()

    def set_hca(self, raw_hca: dict, shrink: float = None):
        """Set per-team HCA with shrinkage."""
        if shrink is None:
            shrink = self.config.hca_shrink
        self.league_hca = np.mean(list(raw_hca.values()))
        self.team_hca = {
            tid: self.league_hca + shrink * (hca - self.league_hca)
            for tid, hca in raw_hca.items()
        }

    def _wavg(self, values: list, weights: list) -> float | None:
        if not values or not weights:
            return None
        n = min(len(values), len(weights))
        return float(np.average(values[-n:], weights=weights[-n:]))

    def get_stats(self, team_id: int) -> dict | None:
        ts = self.team_stats[team_id]
        if not ts['ppg']:
            return None

        wts = ts['wts']
        stats = {
            'ppg': self._wavg(ts['ppg'], wts),
            'papg': self._wavg(ts['papg'], wts),
            'games': len(ts['ppg']),
        }

        if self.config.use_shooting:
            stats['fg'] = self._wavg(ts['fg'], wts) or 47.0
            stats['three'] = self._wavg(ts['three'], wts) or 36.0
            stats['ft'] = self._wavg(ts['ft'], wts) or 78.0

        if self.config.use_defense:
            stats['reb'] = self._wavg(ts['reb'], wts) or 44.0
            stats['ast'] = self._wavg(ts['ast'], wts) or 25.0
            stats['tov'] = self._wavg(ts['tov'], wts) or 14.0
            stats['stl'] = self._wavg(ts['stl'], wts) or 7.5
            stats['blk'] = self._wavg(ts['blk'], wts) or 5.0

        if self.config.use_recent_form:
            n = min(self.config.recent_window, len(ts['ppg']))
            if n >= 3:
                stats['recent_ppg'] = np.mean(ts['ppg'][-n:])
                stats['recent_papg'] = np.mean(ts['papg'][-n:])
            else:
                stats['recent_ppg'] = stats['ppg']
                stats['recent_papg'] = stats['papg']

        return stats

    def get_rest_days(self, team_id: int, game_date: str) -> int:
        if team_id not in self.last_game:
            return 3
        from datetime import datetime
        try:
            curr = datetime.strptime(game_date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[team_id][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 2

    def extract_features(self, home_id: int, away_id: int, game_date: str) -> np.ndarray | None:
        hs = self.get_stats(home_id)
        aws = self.get_stats(away_id)

        if not hs or not aws:
            return None
        if hs['games'] < self.config.min_games or aws['games'] < self.config.min_games:
            return None

        features = []

        # Core: PPG differential
        features.append(hs['ppg'] - aws['ppg'])

        # Defense: PAPG differential (lower is better)
        if self.config.use_defense:
            features.append(hs['papg'] - aws['papg'])
            # Net rating
            features.append((hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg']))

        # Shooting
        if self.config.use_shooting:
            features.append(hs['fg'] - aws['fg'])
            features.append(hs['three'] - aws['three'])
            features.append(hs['ft'] - aws['ft'])

        # Box score diffs
        if self.config.use_defense:
            features.append(hs['reb'] - aws['reb'])
            features.append(hs['ast'] - aws['ast'])
            features.append(hs['tov'] - aws['tov'])  # More TO is bad
            features.append(hs['stl'] - aws['stl'])
            features.append(hs['blk'] - aws['blk'])

        # Rest
        if self.config.use_rest:
            hr = self.get_rest_days(home_id, game_date)
            ar = self.get_rest_days(away_id, game_date)
            features.append(hr - ar)
            features.append(1.0 if hr == 0 else 0.0)  # Home B2B
            features.append(1.0 if ar == 0 else 0.0)  # Away B2B

        # Recent form
        if self.config.use_recent_form:
            features.append(hs['recent_ppg'] - aws['recent_ppg'])
            features.append(hs['recent_papg'] - aws['recent_papg'])

        # HCA
        home_hca = self.team_hca.get(home_id, self.league_hca)
        features.append(home_hca)

        # Reliability
        features.append(min(hs['games'] / 20.0, 1.0))
        features.append(min(aws['games'] / 20.0, 1.0))

        return np.array(features)

    def update(self, team_id: int, pts_for: int, pts_against: int,
               game_date: str, box: dict):
        ts = self.team_stats[team_id]

        # Decay existing weights
        ts['wts'] = [w * self.config.decay for w in ts['wts']]

        # Add new game
        ts['ppg'].append(pts_for)
        ts['papg'].append(pts_against)
        ts['wts'].append(1.0)

        for key in ['fg', 'three', 'ft', 'reb', 'ast', 'tov', 'stl', 'blk']:
            if key in box and box[key] is not None and not np.isnan(box[key]):
                ts[key].append(box[key])

        self.last_game[team_id] = game_date


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT
            g.game_id, g.season, g.date,
            g.home_team_id, g.away_team_id,
            g.home_score, g.away_score,
            hs.field_goal_pct as home_fg, hs.three_point_pct as home_three,
            hs.free_throw_pct as home_ft, hs.total_rebounds as home_reb,
            hs.assists as home_ast, hs.turnovers as home_tov,
            hs.steals as home_stl, hs.blocks as home_blk,
            aws.field_goal_pct as away_fg, aws.three_point_pct as away_three,
            aws.free_throw_pct as away_ft, aws.total_rebounds as away_reb,
            aws.assists as away_ast, aws.turnovers as away_tov,
            aws.steals as away_stl, aws.blocks as away_blk,
            o.latest_spread as vegas_spread
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
    ''', conn)
    conn.close()

    games['actual_spread'] = games['away_score'] - games['home_score']
    return games


def calculate_raw_hca(games: pd.DataFrame, season: int) -> dict:
    sg = games[games['season'] == season]
    hca = {}
    for tid in sg['home_team_id'].unique():
        home_g = sg[sg['home_team_id'] == tid]
        away_g = sg[sg['away_team_id'] == tid]
        if len(home_g) >= 10 and len(away_g) >= 10:
            hm = (home_g['home_score'] - home_g['away_score']).mean()
            am = (away_g['away_score'] - away_g['home_score']).mean()
            hca[tid] = (hm - am) / 2
    return hca


def run_test(config: ModelConfig, games: pd.DataFrame, raw_hca: dict,
             test_season: int = 2024) -> tuple[float, float, int]:
    """Run walk-forward test, return (MAE, RMSE, N)."""
    model = NBAModel(config)
    model.set_hca(raw_hca, config.hca_shrink)

    test_games = games[games['season'] == test_season].copy()

    X, y, vegas = [], [], []

    for _, g in test_games.iterrows():
        feat = model.extract_features(g['home_team_id'], g['away_team_id'], g['date'])

        if feat is not None:
            X.append(feat)
            y.append(g['actual_spread'])
            vegas.append(g['vegas_spread'] if pd.notna(g['vegas_spread']) else np.nan)

        # Update model
        home_box = {
            'fg': g['home_fg'], 'three': g['home_three'], 'ft': g['home_ft'],
            'reb': g['home_reb'], 'ast': g['home_ast'], 'tov': g['home_tov'],
            'stl': g['home_stl'], 'blk': g['home_blk'],
        }
        away_box = {
            'fg': g['away_fg'], 'three': g['away_three'], 'ft': g['away_ft'],
            'reb': g['away_reb'], 'ast': g['away_ast'], 'tov': g['away_tov'],
            'stl': g['away_stl'], 'blk': g['away_blk'],
        }
        model.update(g['home_team_id'], g['home_score'], g['away_score'], g['date'], home_box)
        model.update(g['away_team_id'], g['away_score'], g['home_score'], g['date'], away_box)

    X = np.array(X)
    y = np.array(y)
    vegas = np.array(vegas)

    # Filter valid
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(vegas)
    X, y, vegas = X[valid], y[valid], vegas[valid]

    if len(X) < 100:
        return float('inf'), float('inf'), 0

    # Train/test split within season (first 80% train, last 20% test)
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    vegas_te = vegas[split:]

    # Scale and train
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    ridge = Ridge(alpha=config.alpha)
    ridge.fit(X_tr_s, y_tr)

    preds = ridge.predict(X_te_s)

    mae = np.abs(preds - y_te).mean()
    mse = ((preds - y_te) ** 2).mean()
    vegas_mae = np.abs(vegas_te - y_te).mean()

    return mae, np.sqrt(mse), len(y_te), vegas_mae


def grid_search():
    print("=" * 70)
    print("NBA RIDGE MODEL COMPREHENSIVE TUNING")
    print("=" * 70)

    games = load_data()
    raw_hca = calculate_raw_hca(games, 2023)
    print(f"Loaded {len(games)} games, {len(raw_hca)} teams with HCA data")

    # Get Vegas baseline
    baseline_config = ModelConfig()
    _, _, n, vegas_mae = run_test(baseline_config, games, raw_hca)
    print(f"\nVegas MAE baseline: {vegas_mae:.4f} (N={n})")

    # Grid search parameters
    decays = [0.90, 0.93, 0.95, 0.97, 0.99]
    hca_shrinks = [0.0, 0.2, 0.4, 0.6, 0.8]
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    min_games_list = [3, 5, 7, 10]

    results = []

    # Test decay and HCA shrink with default alpha
    print("\n" + "=" * 70)
    print("PHASE 1: Decay x HCA Shrink (alpha=1.0)")
    print("=" * 70)
    print(f"\n{'Decay':>6} {'Shrink':>8} {'MAE':>10} {'RMSE':>10} {'vs Vegas':>10}")
    print("-" * 50)

    best_mae = float('inf')
    best_config = None

    for decay, shrink in product(decays, hca_shrinks):
        config = ModelConfig(decay=decay, hca_shrink=shrink)
        mae, rmse, n, vg = run_test(config, games, raw_hca)
        if mae < float('inf'):
            results.append({'decay': decay, 'shrink': shrink, 'alpha': 1.0, 'mae': mae, 'rmse': rmse})
            if mae < best_mae:
                best_mae = mae
                best_config = config
            print(f"{decay:>6.2f} {shrink:>8.1f} {mae:>10.4f} {rmse:>10.4f} {mae - vg:>+10.4f}")

    print(f"\nBest so far: decay={best_config.decay}, shrink={best_config.hca_shrink}, MAE={best_mae:.4f}")

    # Test alpha with best decay/shrink
    print("\n" + "=" * 70)
    print(f"PHASE 2: Alpha tuning (decay={best_config.decay}, shrink={best_config.hca_shrink})")
    print("=" * 70)
    print(f"\n{'Alpha':>8} {'MAE':>10} {'RMSE':>10}")
    print("-" * 30)

    for alpha in alphas:
        config = ModelConfig(decay=best_config.decay, hca_shrink=best_config.hca_shrink, alpha=alpha)
        mae, rmse, n, vg = run_test(config, games, raw_hca)
        if mae < float('inf'):
            results.append({'decay': config.decay, 'shrink': config.hca_shrink, 'alpha': alpha, 'mae': mae, 'rmse': rmse})
            if mae < best_mae:
                best_mae = mae
                best_config = config
            print(f"{alpha:>8.1f} {mae:>10.4f} {rmse:>10.4f}")

    # Test min_games with best params
    print("\n" + "=" * 70)
    print("PHASE 3: Min Games tuning")
    print("=" * 70)
    print(f"\n{'Min Games':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 35)

    for mg in min_games_list:
        config = ModelConfig(
            decay=best_config.decay, hca_shrink=best_config.hca_shrink,
            alpha=best_config.alpha, min_games=mg
        )
        mae, rmse, n, vg = run_test(config, games, raw_hca)
        if mae < float('inf'):
            if mae < best_mae:
                best_mae = mae
                best_config = config
            print(f"{mg:>10} {mae:>10.4f} {rmse:>10.4f}")

    # Test feature combinations
    print("\n" + "=" * 70)
    print("PHASE 4: Feature Ablation")
    print("=" * 70)

    feature_combos = [
        {'use_defense': False, 'use_shooting': True, 'use_rest': True, 'use_recent_form': True},
        {'use_defense': True, 'use_shooting': False, 'use_rest': True, 'use_recent_form': True},
        {'use_defense': True, 'use_shooting': True, 'use_rest': False, 'use_recent_form': True},
        {'use_defense': True, 'use_shooting': True, 'use_rest': True, 'use_recent_form': False},
        {'use_defense': False, 'use_shooting': False, 'use_rest': True, 'use_recent_form': True},  # Just PPG + rest
        {'use_defense': True, 'use_shooting': True, 'use_rest': True, 'use_recent_form': True},  # All
    ]

    print(f"\n{'Features':<40} {'MAE':>10} {'RMSE':>10}")
    print("-" * 60)

    for combo in feature_combos:
        config = ModelConfig(
            decay=best_config.decay, hca_shrink=best_config.hca_shrink,
            alpha=best_config.alpha, min_games=best_config.min_games,
            **combo
        )
        mae, rmse, n, vg = run_test(config, games, raw_hca)
        label = f"D={combo['use_defense']}, S={combo['use_shooting']}, R={combo['use_rest']}, F={combo['use_recent_form']}"
        if mae < float('inf'):
            if mae < best_mae:
                best_mae = mae
                best_config = config
            print(f"{label:<40} {mae:>10.4f} {rmse:>10.4f}")

    # Final results
    print("\n" + "=" * 70)
    print("FINAL BEST CONFIGURATION")
    print("=" * 70)
    print(f"\nDecay: {best_config.decay}")
    print(f"HCA Shrink: {best_config.hca_shrink}")
    print(f"Alpha: {best_config.alpha}")
    print(f"Min Games: {best_config.min_games}")
    print(f"Use Defense: {best_config.use_defense}")
    print(f"Use Shooting: {best_config.use_shooting}")
    print(f"Use Rest: {best_config.use_rest}")
    print(f"Use Recent Form: {best_config.use_recent_form}")
    print(f"\nBest MAE: {best_mae:.4f}")
    print(f"Vegas MAE: {vegas_mae:.4f}")
    print(f"Improvement: {vegas_mae - best_mae:+.4f}")


if __name__ == '__main__':
    grid_search()
