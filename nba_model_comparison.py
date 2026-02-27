"""
NBA Model Comparison: Simple vs Enhanced Ridge

Simple = bare bones (PPG, PAPG, rest, flat HCA) - ~8 features
Enhanced = full featured (recent form, momentum, streaks, per-team HCA, totals)

Compare performance across 2024 and 2025 seasons, broken down by:
- Early season (games 1-300)
- Mid season (games 300-700)
- Late season (games 700+)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).parent / 'nba_games.db'


def load_games():
    """Load all completed games with box scores and odds."""
    conn = sqlite3.connect(str(DB_PATH))
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
            aws.blocks as away_blk, aws.turnovers as away_tov,
            o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
    ''', conn)
    conn.close()
    return games


class SimpleModel:
    """
    Bare-bones model with minimal features.
    ~8 features: PPG diff, PAPG diff, net rating, rest, B2B indicators, flat HCA, reliability.
    No box score stats, no momentum, no streaks.
    """
    DECAY = 0.97
    PREV_HALF_LIFE = 6.0
    HCA = 1.8  # Flat HCA for all teams

    def __init__(self):
        self.team_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': []
        }))
        self.prev_ratings = {}
        self.last_game = {}
        self.league_avg = {'ppg': 115.0}

    def reset(self):
        self.team_stats.clear()
        self.prev_ratings.clear()
        self.last_game.clear()

    def _wavg(self, vals, wts):
        if not vals or not wts:
            return None
        return float(np.average(vals, weights=wts))

    def _get_stats(self, tid, season):
        td = self.team_stats[tid][season]
        games = len(td['ppg'])

        ppg = self._wavg(td['ppg'], td['wts']) if td['ppg'] else None
        papg = self._wavg(td['papg'], td['wts']) if td['papg'] else None

        prev = self.prev_ratings.get(tid, {})
        prev_ppg = prev.get('ppg', self.league_avg['ppg'])
        prev_papg = prev.get('papg', self.league_avg['ppg'])

        if ppg is None:
            return {'ppg': prev_ppg, 'papg': prev_papg, 'games': 0}

        blend = 0.5 ** (games / self.PREV_HALF_LIFE)
        return {
            'ppg': blend * prev_ppg + (1 - blend) * ppg,
            'papg': blend * prev_papg + (1 - blend) * papg,
            'games': games
        }

    def _get_rest(self, tid, date):
        if tid not in self.last_game:
            return 3
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 1

    def extract_spread_features(self, hid, aid, season, date):
        """8 features for spread prediction."""
        home = self._get_stats(hid, season)
        away = self._get_stats(aid, season)
        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            home['ppg'] - away['ppg'],                              # 1: PPG diff
            home['papg'] - away['papg'],                            # 2: PAPG diff
            (home['ppg'] - home['papg']) - (away['ppg'] - away['papg']),  # 3: Net rating diff
            hr - ar,                                                 # 4: Rest diff
            1.0 if hr == 0 else 0.0,                                # 5: Home B2B
            1.0 if ar == 0 else 0.0,                                # 6: Away B2B
            min((home['games'] + away['games']) / 40.0, 1.0),       # 7: Combined reliability
            self.HCA,                                                # 8: Flat HCA
        ])

    def extract_total_features(self, hid, aid, season, date):
        """6 features for total prediction."""
        home = self._get_stats(hid, season)
        away = self._get_stats(aid, season)
        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            home['ppg'] + away['ppg'],                              # 1: Combined PPG
            home['papg'] + away['papg'],                            # 2: Combined PAPG
            (home['ppg'] + home['papg'] + away['ppg'] + away['papg']) / 4,  # 3: Pace proxy
            1.0 if hr == 0 else 0.0,                                # 4: Home B2B
            1.0 if ar == 0 else 0.0,                                # 5: Away B2B
            min((home['games'] + away['games']) / 40.0, 1.0),       # 6: Combined reliability
        ])

    def update(self, tid, season, date, pts_for, pts_against):
        td = self.team_stats[tid][season]
        td['wts'] = [w * self.DECAY for w in td['wts']]
        td['ppg'].append(pts_for)
        td['papg'].append(pts_against)
        td['wts'].append(1.0)
        self.last_game[tid] = date

    def set_prev_season(self, season, games_df):
        prev = season - 1
        prev_games = games_df[games_df['season'] == prev]
        if len(prev_games) == 0:
            return

        self.league_avg['ppg'] = prev_games['home_score'].mean()

        for tid in set(prev_games['home_team_id']) | set(prev_games['away_team_id']):
            home = prev_games[prev_games['home_team_id'] == tid]
            away = prev_games[prev_games['away_team_id'] == tid]
            pts = list(home['home_score']) + list(away['away_score'])
            pts_ag = list(home['away_score']) + list(away['home_score'])
            if pts:
                self.prev_ratings[tid] = {'ppg': np.mean(pts), 'papg': np.mean(pts_ag)}

        self.last_game.clear()


class EnhancedModel:
    """
    Full-featured model with all the bells and whistles.
    - Recent form (last 5 games)
    - Momentum/trend (last 6 games)
    - Win/loss streaks
    - Per-team HCA (scaled & shrunk toward league mean)
    - Both spread and total predictions
    """
    DECAY = 0.97
    MIN_GAMES = 10
    HCA_SCALE = 0.36
    HCA_SHRINK = 0.50
    HCA_DEFAULT = 1.8

    def __init__(self):
        self.team_stats = defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'margins': [], 'wins': []
        })
        self.prev_ratings = {}
        self.last_game = {}
        self.league_hca = self.HCA_DEFAULT
        self.team_hca = {}  # Per-team HCA values

    def reset(self):
        self.team_stats.clear()
        self.prev_ratings.clear()
        self.last_game.clear()
        self.team_hca.clear()

    def _wavg(self, vals, wts):
        if not vals or not wts:
            return None
        n = min(len(vals), len(wts))
        return np.average(vals[-n:], weights=wts[-n:])

    def _get_rest(self, tid, date):
        if tid not in self.last_game:
            return 3
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 2

    def get_team_hca(self, tid):
        """Get per-team HCA, falling back to league average."""
        return self.team_hca.get(tid, self.league_hca)

    def calculate_team_hca(self, games_df, season):
        """
        Calculate per-team HCA from a season's data.
        Formula: raw = home_margin - away_margin, then shrink toward league mean.
        """
        sg = games_df[games_df['season'] == season]
        raw_hca = {}

        for tid in sg['home_team_id'].unique():
            home_games = sg[sg['home_team_id'] == tid]
            away_games = sg[sg['away_team_id'] == tid]

            if len(home_games) >= 10 and len(away_games) >= 10:
                home_margin = (home_games['home_score'] - home_games['away_score']).mean()
                away_margin = (away_games['away_score'] - away_games['home_score']).mean()
                raw_hca[tid] = home_margin - away_margin

        if not raw_hca:
            return

        league_mean = np.mean(list(raw_hca.values()))
        self.league_hca = self.HCA_SCALE * league_mean

        for tid, raw in raw_hca.items():
            shrunk = league_mean + self.HCA_SHRINK * (raw - league_mean)
            self.team_hca[tid] = self.HCA_SCALE * shrunk

    def extract_spread_features(self, hid, aid, date):
        """16 features for spread prediction."""
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]

        if not hs['ppg'] or not aws['ppg']:
            return None
        if len(hs['ppg']) < self.MIN_GAMES or len(aws['ppg']) < self.MIN_GAMES:
            return None

        # Weighted season stats
        h_ppg = self._wavg(hs['ppg'], hs['wts'])
        h_papg = self._wavg(hs['papg'], hs['wts'])
        a_ppg = self._wavg(aws['ppg'], aws['wts'])
        a_papg = self._wavg(aws['papg'], aws['wts'])

        # Recent form (last 5 games, unweighted)
        h_recent_ppg = np.mean(hs['ppg'][-5:]) if len(hs['ppg']) >= 5 else h_ppg
        h_recent_papg = np.mean(hs['papg'][-5:]) if len(hs['papg']) >= 5 else h_papg
        a_recent_ppg = np.mean(aws['ppg'][-5:]) if len(aws['ppg']) >= 5 else a_ppg
        a_recent_papg = np.mean(aws['papg'][-5:]) if len(aws['papg']) >= 5 else a_papg

        # Momentum (trend over last 6 games)
        def get_trend(margins, n=6):
            if len(margins) < n:
                return 0
            recent = margins[-n:]
            return np.mean(recent[n//2:]) - np.mean(recent[:n//2])

        h_trend = get_trend(hs['margins'])
        a_trend = get_trend(aws['margins'])

        # Win streak
        def get_streak(wins):
            if not wins:
                return 0
            streak = 0
            last = wins[-1]
            for w in reversed(wins):
                if w == last:
                    streak += 1
                else:
                    break
            return streak if last == 1 else -streak

        h_streak = get_streak(hs['wins'])
        a_streak = get_streak(aws['wins'])

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        games_played = (len(hs['ppg']) + len(aws['ppg'])) / 2

        return np.array([
            h_ppg - a_ppg,                                          # 1: PPG diff
            h_papg - a_papg,                                        # 2: PAPG diff
            (h_ppg - h_papg) - (a_ppg - a_papg),                   # 3: Net rating diff
            h_recent_ppg - a_recent_ppg,                           # 4: Recent PPG diff
            h_recent_papg - a_recent_papg,                         # 5: Recent PAPG diff
            (h_recent_ppg - h_recent_papg) - (a_recent_ppg - a_recent_papg),  # 6: Recent net
            h_trend - a_trend,                                      # 7: Momentum diff
            h_streak - a_streak,                                    # 8: Streak diff
            hr - ar,                                                # 9: Rest diff
            1 if hr == 0 else 0,                                   # 10: Home B2B
            1 if ar == 0 else 0,                                   # 11: Away B2B
            self.get_team_hca(hid),                                # 12: Per-team HCA
            min(len(hs['ppg']) / 30, 1),                           # 13: Home reliability
            min(len(aws['ppg']) / 30, 1),                          # 14: Away reliability
            min(games_played / 82, 1),                             # 15: Season progress
            (h_ppg + a_ppg) / 2 - 115,                             # 16: Pace adjustment
        ])

    def extract_total_features(self, hid, aid, date):
        """12 features for total prediction."""
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]

        if not hs['ppg'] or not aws['ppg']:
            return None
        if len(hs['ppg']) < self.MIN_GAMES or len(aws['ppg']) < self.MIN_GAMES:
            return None

        # Weighted season stats
        h_ppg = self._wavg(hs['ppg'], hs['wts'])
        h_papg = self._wavg(hs['papg'], hs['wts'])
        a_ppg = self._wavg(aws['ppg'], aws['wts'])
        a_papg = self._wavg(aws['papg'], aws['wts'])

        # Recent form
        h_recent_ppg = np.mean(hs['ppg'][-5:]) if len(hs['ppg']) >= 5 else h_ppg
        h_recent_papg = np.mean(hs['papg'][-5:]) if len(hs['papg']) >= 5 else h_papg
        a_recent_ppg = np.mean(aws['ppg'][-5:]) if len(aws['ppg']) >= 5 else a_ppg
        a_recent_papg = np.mean(aws['papg'][-5:]) if len(aws['papg']) >= 5 else a_papg

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            h_ppg + a_ppg,                                          # 1: Combined PPG
            h_papg + a_papg,                                        # 2: Combined PAPG
            (h_ppg + h_papg) / 2,                                   # 3: Home pace proxy
            (a_ppg + a_papg) / 2,                                   # 4: Away pace proxy
            h_recent_ppg + a_recent_ppg,                           # 5: Recent combined PPG
            h_recent_papg + a_recent_papg,                         # 6: Recent combined PAPG
            1 if hr == 0 else 0,                                   # 7: Home B2B
            1 if ar == 0 else 0,                                   # 8: Away B2B
            min(len(hs['ppg']) / 30, 1),                           # 9: Home reliability
            min(len(aws['ppg']) / 30, 1),                          # 10: Away reliability
            min((len(hs['ppg']) + len(aws['ppg'])) / 164, 1),      # 11: Season progress
            (h_ppg + h_papg + a_ppg + a_papg) / 4 - 115,           # 12: Combined pace adjustment
        ])

    def update(self, tid, pts_for, pts_against, date, won):
        ts = self.team_stats[tid]
        ts['wts'] = [w * self.DECAY for w in ts['wts']]
        ts['ppg'].append(pts_for)
        ts['papg'].append(pts_against)
        ts['wts'].append(1.0)
        ts['margins'].append(pts_for - pts_against)
        ts['wins'].append(1 if won else 0)
        self.last_game[tid] = date

    def set_prev_season(self, season, games_df):
        """Set previous season ratings and calculate per-team HCA."""
        prev = season - 1
        prev_games = games_df[games_df['season'] == prev]
        if len(prev_games) == 0:
            return

        for tid in set(prev_games['home_team_id']) | set(prev_games['away_team_id']):
            home = prev_games[prev_games['home_team_id'] == tid]
            away = prev_games[prev_games['away_team_id'] == tid]
            pts = list(home['home_score']) + list(away['away_score'])
            pts_ag = list(home['away_score']) + list(away['home_score'])
            if pts:
                self.prev_ratings[tid] = {'ppg': np.mean(pts), 'papg': np.mean(pts_ag)}

        # Calculate per-team HCA from previous season
        self.calculate_team_hca(games_df, prev)
        self.last_game.clear()


def run_comparison():
    """Run both models on 2024 and 2025 seasons with walk-forward validation."""
    print("=" * 70)
    print("NBA MODEL COMPARISON: SIMPLE (8 feat) vs ENHANCED (16 feat)")
    print("=" * 70)

    games = load_games()
    print(f"Total games loaded: {len(games)}")

    # Filter to 2024 and 2025 seasons
    test_seasons = [2024, 2025]

    # Results storage - spreads
    results = {
        'simple': {'pred': [], 'actual': [], 'vegas': [], 'game_num': [], 'season': []},
        'enhanced': {'pred': [], 'actual': [], 'vegas': [], 'game_num': [], 'season': []}
    }

    # Results storage - totals
    total_results = {
        'simple': {'pred': [], 'actual': [], 'vegas': [], 'game_num': [], 'season': []},
        'enhanced': {'pred': [], 'actual': [], 'vegas': [], 'game_num': [], 'season': []}
    }

    for test_season in test_seasons:
        print(f"\n{'='*70}")
        print(f"TESTING SEASON {test_season}")
        print(f"{'='*70}")

        # Initialize models
        simple = SimpleModel()
        enhanced = EnhancedModel()

        # Warm up on previous season
        simple.set_prev_season(test_season, games)
        enhanced.set_prev_season(test_season, games)

        prev_season = test_season - 1
        for _, g in games[games['season'] == prev_season].iterrows():
            home_won = g['home_score'] > g['away_score']
            simple.update(g['home_team_id'], prev_season, g['date'], g['home_score'], g['away_score'])
            simple.update(g['away_team_id'], prev_season, g['date'], g['away_score'], g['home_score'])
            enhanced.update(g['home_team_id'], g['home_score'], g['away_score'], g['date'], home_won)
            enhanced.update(g['away_team_id'], g['away_score'], g['home_score'], g['date'], not home_won)

        # Walk-forward on test season
        test_games = games[games['season'] == test_season].copy()
        print(f"Test games: {len(test_games)}")

        # Training data for spread models
        X_simple_s, y_simple_s = [], []
        X_enhanced_s, y_enhanced_s = [], []

        # Training data for total models
        X_simple_t, y_simple_t = [], []
        X_enhanced_t, y_enhanced_t = [], []

        game_num = 0
        for _, g in test_games.iterrows():
            actual_spread = g['away_score'] - g['home_score']
            actual_total = g['home_score'] + g['away_score']
            vegas_spread = g['vegas_spread'] if pd.notna(g['vegas_spread']) else None
            vegas_total = g['vegas_total'] if pd.notna(g['vegas_total']) else None

            # Simple model features
            feat_simple_s = simple.extract_spread_features(g['home_team_id'], g['away_team_id'], test_season, g['date'])
            feat_simple_t = simple.extract_total_features(g['home_team_id'], g['away_team_id'], test_season, g['date'])

            # Enhanced model features
            feat_enhanced_s = enhanced.extract_spread_features(g['home_team_id'], g['away_team_id'], g['date'])
            feat_enhanced_t = enhanced.extract_total_features(g['home_team_id'], g['away_team_id'], g['date'])

            # Train and predict for simple model - SPREAD (after 50 games)
            if len(X_simple_s) >= 50 and not np.isnan(feat_simple_s).any():
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(np.array(X_simple_s))
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_scaled, np.array(y_simple_s))
                pred = ridge.predict(scaler.transform(feat_simple_s.reshape(1, -1)))[0]

                results['simple']['pred'].append(pred)
                results['simple']['actual'].append(actual_spread)
                results['simple']['vegas'].append(vegas_spread)
                results['simple']['game_num'].append(game_num)
                results['simple']['season'].append(test_season)

            # Train and predict for simple model - TOTAL (after 50 games)
            if len(X_simple_t) >= 50 and not np.isnan(feat_simple_t).any():
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(np.array(X_simple_t))
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_scaled, np.array(y_simple_t))
                pred = ridge.predict(scaler.transform(feat_simple_t.reshape(1, -1)))[0]

                total_results['simple']['pred'].append(pred)
                total_results['simple']['actual'].append(actual_total)
                total_results['simple']['vegas'].append(vegas_total)
                total_results['simple']['game_num'].append(game_num)
                total_results['simple']['season'].append(test_season)

            # Store for training - simple
            if not np.isnan(feat_simple_s).any():
                X_simple_s.append(feat_simple_s)
                y_simple_s.append(actual_spread)
            if not np.isnan(feat_simple_t).any():
                X_simple_t.append(feat_simple_t)
                y_simple_t.append(actual_total)

            # Train and predict for enhanced model - SPREAD (after 50 games)
            if feat_enhanced_s is not None and len(X_enhanced_s) >= 50:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(np.array(X_enhanced_s))
                ridge = Ridge(alpha=0.1)
                ridge.fit(X_scaled, np.array(y_enhanced_s))
                pred = ridge.predict(scaler.transform(feat_enhanced_s.reshape(1, -1)))[0]

                results['enhanced']['pred'].append(pred)
                results['enhanced']['actual'].append(actual_spread)
                results['enhanced']['vegas'].append(vegas_spread)
                results['enhanced']['game_num'].append(game_num)
                results['enhanced']['season'].append(test_season)

            # Train and predict for enhanced model - TOTAL (after 50 games)
            if feat_enhanced_t is not None and len(X_enhanced_t) >= 50:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(np.array(X_enhanced_t))
                ridge = Ridge(alpha=0.1)
                ridge.fit(X_scaled, np.array(y_enhanced_t))
                pred = ridge.predict(scaler.transform(feat_enhanced_t.reshape(1, -1)))[0]

                total_results['enhanced']['pred'].append(pred)
                total_results['enhanced']['actual'].append(actual_total)
                total_results['enhanced']['vegas'].append(vegas_total)
                total_results['enhanced']['game_num'].append(game_num)
                total_results['enhanced']['season'].append(test_season)

            # Store for training - enhanced
            if feat_enhanced_s is not None:
                X_enhanced_s.append(feat_enhanced_s)
                y_enhanced_s.append(actual_spread)
            if feat_enhanced_t is not None:
                X_enhanced_t.append(feat_enhanced_t)
                y_enhanced_t.append(actual_total)

            # Update both models
            home_won = g['home_score'] > g['away_score']
            simple.update(g['home_team_id'], test_season, g['date'], g['home_score'], g['away_score'])
            simple.update(g['away_team_id'], test_season, g['date'], g['away_score'], g['home_score'])
            enhanced.update(g['home_team_id'], g['home_score'], g['away_score'], g['date'], home_won)
            enhanced.update(g['away_team_id'], g['away_score'], g['home_score'], g['date'], not home_won)

            game_num += 1

    # Analyze results
    print("\n" + "=" * 70)
    print("OVERALL RESULTS - SPREADS")
    print("=" * 70)

    for model_name in ['simple', 'enhanced']:
        r = results[model_name]
        if not r['pred']:
            print(f"\n{model_name.upper()}: No predictions generated")
            continue

        pred = np.array(r['pred'])
        actual = np.array(r['actual'])
        vegas = np.array([v if v is not None else np.nan for v in r['vegas']])

        # Filter to games with Vegas lines
        valid = ~np.isnan(vegas)

        mae_model = np.abs(pred - actual).mean()
        mae_vegas = np.abs(vegas[valid] - actual[valid]).mean() if valid.sum() > 0 else np.nan

        winner_model = ((pred < 0) == (actual < 0)).mean()
        winner_vegas = ((vegas[valid] < 0) == (actual[valid] < 0)).mean() if valid.sum() > 0 else np.nan

        print(f"\n{model_name.upper()} MODEL (N={len(pred)}):")
        print(f"  Spread MAE: {mae_model:.3f}")
        print(f"  Vegas MAE:  {mae_vegas:.3f}")
        print(f"  Difference: {mae_model - mae_vegas:+.3f}")
        print(f"  Winner Acc: {winner_model*100:.1f}%")
        print(f"  Vegas Win:  {winner_vegas*100:.1f}%")

    print("\n" + "=" * 70)
    print("OVERALL RESULTS - TOTALS")
    print("=" * 70)

    for model_name in ['simple', 'enhanced']:
        r = total_results[model_name]
        if not r['pred']:
            print(f"\n{model_name.upper()}: No predictions generated")
            continue

        pred = np.array(r['pred'])
        actual = np.array(r['actual'])
        vegas = np.array([v if v is not None else np.nan for v in r['vegas']])

        valid = ~np.isnan(vegas)

        mae_model = np.abs(pred - actual).mean()
        mae_vegas = np.abs(vegas[valid] - actual[valid]).mean() if valid.sum() > 0 else np.nan

        # Bias check (does model under/overestimate?)
        bias = (pred - actual).mean()

        print(f"\n{model_name.upper()} MODEL (N={len(pred)}):")
        print(f"  Total MAE:  {mae_model:.3f}")
        print(f"  Vegas MAE:  {mae_vegas:.3f}")
        print(f"  Difference: {mae_model - mae_vegas:+.3f}")
        print(f"  Bias:       {bias:+.2f} (positive = overestimates)")

    # Segment analysis
    print("\n" + "=" * 70)
    print("PERFORMANCE BY SEASON SEGMENT")
    print("=" * 70)

    segments = [
        ('Early (1-300)', 0, 300),
        ('Mid (300-700)', 300, 700),
        ('Late (700+)', 700, 1500)
    ]

    for model_name in ['simple', 'enhanced']:
        r = results[model_name]
        if not r['pred']:
            continue

        print(f"\n{model_name.upper()} MODEL:")
        print(f"{'Segment':<20} {'N':>6} {'MAE':>8} {'Vegas':>8} {'Diff':>8} {'Win%':>8}")
        print("-" * 60)

        for seg_name, start, end in segments:
            mask = [(start <= gn < end) for gn in r['game_num']]
            if sum(mask) == 0:
                continue

            pred = np.array([r['pred'][i] for i, m in enumerate(mask) if m])
            actual = np.array([r['actual'][i] for i, m in enumerate(mask) if m])
            vegas = np.array([r['vegas'][i] if r['vegas'][i] is not None else np.nan
                             for i, m in enumerate(mask) if m])

            valid = ~np.isnan(vegas)
            mae_model = np.abs(pred - actual).mean()
            mae_vegas = np.abs(vegas[valid] - actual[valid]).mean() if valid.sum() > 0 else np.nan
            winner = ((pred < 0) == (actual < 0)).mean()

            diff = mae_model - mae_vegas if not np.isnan(mae_vegas) else np.nan
            print(f"{seg_name:<20} {len(pred):>6} {mae_model:>8.3f} {mae_vegas:>8.3f} {diff:>+8.3f} {winner*100:>7.1f}%")

    # By season
    print("\n" + "=" * 70)
    print("PERFORMANCE BY SEASON")
    print("=" * 70)

    for model_name in ['simple', 'enhanced']:
        r = results[model_name]
        if not r['pred']:
            continue

        print(f"\n{model_name.upper()} MODEL:")
        print(f"{'Season':<10} {'N':>6} {'MAE':>8} {'Vegas':>8} {'Diff':>8} {'Win%':>8}")
        print("-" * 50)

        for season in test_seasons:
            mask = [s == season for s in r['season']]
            if sum(mask) == 0:
                continue

            pred = np.array([r['pred'][i] for i, m in enumerate(mask) if m])
            actual = np.array([r['actual'][i] for i, m in enumerate(mask) if m])
            vegas = np.array([r['vegas'][i] if r['vegas'][i] is not None else np.nan
                             for i, m in enumerate(mask) if m])

            valid = ~np.isnan(vegas)
            mae_model = np.abs(pred - actual).mean()
            mae_vegas = np.abs(vegas[valid] - actual[valid]).mean() if valid.sum() > 0 else np.nan
            winner = ((pred < 0) == (actual < 0)).mean()

            diff = mae_model - mae_vegas if not np.isnan(mae_vegas) else np.nan
            print(f"{season:<10} {len(pred):>6} {mae_model:>8.3f} {mae_vegas:>8.3f} {diff:>+8.3f} {winner*100:>7.1f}%")

    # Direct comparison
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD: WHEN DO MODELS DIVERGE?")
    print("=" * 70)

    # Find games where both models made predictions
    simple_games = set(zip(results['simple']['game_num'], results['simple']['season']))
    enhanced_games = set(zip(results['enhanced']['game_num'], results['enhanced']['season']))
    common = simple_games & enhanced_games

    if common:
        simple_better = 0
        enhanced_better = 0

        for gn, season in common:
            idx_s = [(i, g, s) for i, (g, s) in enumerate(zip(results['simple']['game_num'],
                                                               results['simple']['season']))
                     if g == gn and s == season][0][0]
            idx_e = [(i, g, s) for i, (g, s) in enumerate(zip(results['enhanced']['game_num'],
                                                               results['enhanced']['season']))
                     if g == gn and s == season][0][0]

            actual = results['simple']['actual'][idx_s]
            err_s = abs(results['simple']['pred'][idx_s] - actual)
            err_e = abs(results['enhanced']['pred'][idx_e] - actual)

            if err_s < err_e:
                simple_better += 1
            elif err_e < err_s:
                enhanced_better += 1

        print(f"\nGames where both predicted: {len(common)}")
        print(f"Simple better:   {simple_better} ({simple_better/len(common)*100:.1f}%)")
        print(f"Enhanced better: {enhanced_better} ({enhanced_better/len(common)*100:.1f}%)")
        print(f"Tied:            {len(common) - simple_better - enhanced_better}")


if __name__ == '__main__':
    run_comparison()
