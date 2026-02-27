"""
NFL Edge Analysis: When Model Disagrees with Vegas, Who is Right?

Adapted from nba_edge_analysis.py to analyze NFL games.
NFL-specific considerations:
- Only 17 games per team per season (much smaller sample than NBA)
- Bye weeks affect team performance
- Higher variance in NFL outcomes
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = 'nfl_games.db'


class QuickEnhancedModel:
    """Simplified enhanced model for quick analysis (NFL version).

    SPREAD CONVENTION: actual_spread = away_score - home_score
    - Negative spread means home team won/favored
    - Positive spread means away team won/favored

    Features should be signed so that:
    - Positive feature value -> higher predicted spread (away does better)
    - Negative feature value -> lower predicted spread (home does better)
    """
    DECAY = 0.96  # Higher decay for NFL - recent games more predictive
    MIN_GAMES = 3  # Require at least 3 games for reliability

    def __init__(self):
        self.team_stats = defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [], 'margins': [], 'wins': [],
            'yards': [], 'yards_wts': [], 'turnovers': [], 'to_wts': []
        })
        self.last_game = {}
        self.prev_ratings = {}

    def _wavg(self, vals, wts):
        if not vals or not wts:
            return None
        n = min(len(vals), len(wts))
        return np.average(vals[-n:], weights=wts[-n:])

    def _get_rest(self, tid, date):
        if tid not in self.last_game:
            return 7  # Default NFL rest
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            rest = (curr - last).days
            return min(rest, 14)  # Cap at 14 for bye week
        except Exception:
            return 7

    def extract_features(self, hid, aid, date, season):
        """Extract features with proper signs for spread convention.

        All features are signed so that POSITIVE value -> higher spread (away does better).
        """
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]

        if len(hs['ppg']) < self.MIN_GAMES or len(aws['ppg']) < self.MIN_GAMES:
            return None

        h_ppg = self._wavg(hs['ppg'], hs['wts'])
        h_papg = self._wavg(hs['papg'], hs['wts'])
        a_ppg = self._wavg(aws['ppg'], aws['wts'])
        a_papg = self._wavg(aws['papg'], aws['wts'])

        # Net ratings (positive = that team is better)
        h_net = h_ppg - h_papg
        a_net = a_ppg - a_papg

        return np.array([
            # Offensive: away scores more -> higher spread (correct sign)
            a_ppg - h_ppg,  # NEGATED from before

            # Defensive: home allows more -> higher spread (correct sign)
            h_papg - a_papg,

            # Net rating: away better -> higher spread (correct sign)
            a_net - h_net,  # NEGATED from before

            # HCA: constant negative since home advantage lowers spread
            -2.5,  # NEGATED from before
        ])

    def update(self, tid, pf, pa, date, won):
        """Update team stats after a game."""
        ts = self.team_stats[tid]
        ts['wts'] = [w * self.DECAY for w in ts['wts']]
        ts['ppg'].append(pf)
        ts['papg'].append(pa)
        ts['wts'].append(1.0)
        ts['margins'].append(pf - pa)
        ts['wins'].append(1 if won else 0)
        self.last_game[tid] = date

    def set_previous_season(self, season):
        """Carry over previous season ratings."""
        prev = season - 1
        for tid in self.team_stats:
            td = self.team_stats[tid]
            if td['ppg']:
                self.prev_ratings[tid] = {
                    'ppg': np.mean(td['ppg']),
                    'papg': np.mean(td['papg']),
                }
        self.last_game.clear()


def run_edge_analysis():
    """Analyze when model disagreement with Vegas = profitable edge."""
    print("=" * 70)
    print("NFL EDGE ANALYSIS: When Model Disagrees with Vegas, Who is Right?")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date, g.week
    ''', conn)
    conn.close()

    # Filter to games with Vegas lines
    games = games[games['vegas_spread'].notna()].copy()
    games['actual_spread'] = games['away_score'] - games['home_score']

    print(f"Total games with Vegas lines: {len(games)}")

    # Run walk-forward prediction
    model = QuickEnhancedModel()
    X_all, y_all = [], []
    results = []
    current_season = None

    for _, g in games.iterrows():
        # Handle season transitions
        if current_season is not None and g['season'] != current_season:
            model.set_previous_season(g['season'])
        current_season = g['season']

        feat = model.extract_features(
            g['home_team_id'], g['away_team_id'], g['date'], g['season']
        )

        if feat is not None and len(X_all) >= 100:  # Need enough data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.array(X_all))
            ridge = Ridge(alpha=10.0)  # Higher regularization to prevent overfitting
            ridge.fit(X_scaled, np.array(y_all))
            pred = ridge.predict(scaler.transform(feat.reshape(1, -1)))[0]

            results.append({
                'pred': pred,
                'vegas': g['vegas_spread'],
                'actual': g['actual_spread'],
                'total': g['home_score'] + g['away_score'],
                'vegas_total': g['vegas_total'],
                'season': g['season'],
                'week': g['week'],
                'edge': pred - g['vegas_spread'],
            })

        if feat is not None:
            X_all.append(feat)
            y_all.append(g['actual_spread'])

        won = g['home_score'] > g['away_score']
        model.update(g['home_team_id'], g['home_score'], g['away_score'], g['date'], won)
        model.update(g['away_team_id'], g['away_score'], g['home_score'], g['date'], not won)

    df = pd.DataFrame(results)

    print(f"\nGames analyzed (with predictions): {len(df)}")
    print(f"Model MAE: {np.abs(df['pred'] - df['actual']).mean():.2f}")
    print(f"Vegas MAE: {np.abs(df['vegas'] - df['actual']).mean():.2f}")

    # ATS analysis by edge size
    print("\n" + "=" * 70)
    print("ATS RECORD BY EDGE SIZE (Enhanced Ridge)")
    print("(Betting WITH the model when it disagrees with Vegas)")
    print("=" * 70)
    print(f"{'Edge':<12} {'Games':>8} {'Wins':>8} {'Losses':>8} {'Push':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 70)

    for threshold in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]:
        wins, losses, pushes = 0, 0, 0

        mask_pos = df['edge'] >= threshold
        for _, r in df[mask_pos].iterrows():
            result = r['actual'] - r['vegas']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1
            else:
                pushes += 1

        mask_neg = df['edge'] <= -threshold
        for _, r in df[mask_neg].iterrows():
            result = r['vegas'] - r['actual']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1
            else:
                pushes += 1

        total = wins + losses
        if total > 0:
            win_pct = wins / total * 100
            profit = wins * 0.909 - losses
            roi = profit / (wins + losses) * 100

            marker = " <-- PROFITABLE" if win_pct > 52.4 else ""
            print(f">= {threshold:<8} {wins + losses + pushes:>8} {wins:>8} {losses:>8} {pushes:>8} "
                  f"{win_pct:>7.1f}% {roi:>+9.1f}%{marker}")

    # Totals analysis
    print("\n" + "=" * 70)
    print("TOTALS RECORD BY EDGE SIZE")
    print("=" * 70)
    print(f"{'Edge':<12} {'Games':>8} {'Wins':>8} {'Losses':>8} {'Push':>8} {'Win%':>8}")
    print("-" * 70)

    df_totals = df[df['vegas_total'].notna()].copy()
    df_totals['total_edge'] = df_totals['total'] - df_totals['vegas_total']
    df_totals['model_total_pred'] = df_totals['pred'].apply(lambda x: 0)  # Placeholder

    for threshold in [0, 3, 5, 7, 10]:
        # Over: model predicts higher total than Vegas
        # Under: model predicts lower total than Vegas
        # We don't have a total model here, so skip this for now
        pass

    # By season
    print("\n" + "=" * 70)
    print("EDGE PERFORMANCE BY SEASON (>= 5 pt edge)")
    print("=" * 70)

    for season in sorted(df['season'].unique()):
        sdf = df[df['season'] == season]
        wins, losses = 0, 0

        for _, r in sdf[sdf['edge'] >= 5].iterrows():
            result = r['actual'] - r['vegas']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        for _, r in sdf[sdf['edge'] <= -5].iterrows():
            result = r['vegas'] - r['actual']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        if wins + losses > 0:
            pct = wins / (wins + losses) * 100
            print(f"Season {season}: {wins}-{losses} ({pct:.1f}%)")

    # By week range
    print("\n" + "=" * 70)
    print("EDGE PERFORMANCE BY WEEK (>= 5 pt edge)")
    print("=" * 70)

    for start, end in [(1, 4), (5, 9), (10, 14), (15, 18)]:
        mask = (df['week'] >= start) & (df['week'] <= end)
        sdf = df[mask]
        wins, losses = 0, 0

        for _, r in sdf[sdf['edge'] >= 5].iterrows():
            result = r['actual'] - r['vegas']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        for _, r in sdf[sdf['edge'] <= -5].iterrows():
            result = r['vegas'] - r['actual']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        if wins + losses > 0:
            pct = wins / (wins + losses) * 100
            print(f"Weeks {start}-{end}: {wins}-{losses} ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Break-even at -110 odds: 52.4%")
    print("Look for thresholds where Win% consistently exceeds 52.4%")


if __name__ == '__main__':
    run_edge_analysis()
