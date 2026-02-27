"""
CBB Edge Analysis: When Model Disagrees with Vegas, Who is Right?

Adapted from nba_edge_analysis.py to analyze college basketball.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = 'cbb_games.db'


class QuickEnhancedModel:
    """Simplified enhanced model for quick analysis."""
    DECAY = 0.97
    MIN_GAMES = 8  # Lower threshold for CBB (more teams, fewer games each)

    def __init__(self):
        self.team_stats = defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [], 'margins': [], 'wins': []
        })
        self.last_game = {}

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
            return max(0, min((curr - last).days - 1, 7))
        except Exception:
            return 3

    def extract_features(self, hid, aid, date):
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

        def get_trend(margins):
            if len(margins) < 6:
                return 0
            return np.mean(margins[-3:]) - np.mean(margins[-6:-3])

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
            return streak if last else -streak

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            h_ppg - a_ppg,
            h_papg - a_papg,
            (h_ppg - h_papg) - (a_ppg - a_papg),
            h_recent - a_recent,
            get_trend(hs['margins']) - get_trend(aws['margins']),
            get_streak(hs['wins']) - get_streak(aws['wins']),
            hr - ar,
            1 if hr == 0 else 0,
            1 if ar == 0 else 0,
            3.5,  # Higher HCA for college
            min(len(hs['ppg']) / 20, 1),
            min(len(aws['ppg']) / 20, 1),
        ])

    def update(self, tid, pf, pa, date, won):
        ts = self.team_stats[tid]
        ts['wts'] = [w * self.DECAY for w in ts['wts']]
        ts['ppg'].append(pf)
        ts['papg'].append(pa)
        ts['wts'].append(1.0)
        ts['margins'].append(pf - pa)
        ts['wins'].append(1 if won else 0)
        self.last_game[tid] = date


def run_edge_analysis():
    """Analyze when model disagreement with Vegas = profitable edge."""
    print("=" * 70)
    print("CBB EDGE ANALYSIS: When Model Disagrees with Vegas, Who is Right?")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               o.latest_spread as vegas_spread
        FROM games g
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
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

    for _, g in games.iterrows():
        feat = model.extract_features(g['home_team_id'], g['away_team_id'], g['date'])

        if feat is not None and len(X_all) >= 200:  # Need more training data for CBB
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.array(X_all))
            ridge = Ridge(alpha=0.1)
            ridge.fit(X_scaled, np.array(y_all))
            pred = ridge.predict(scaler.transform(feat.reshape(1, -1)))[0]

            results.append({
                'pred': pred,
                'vegas': g['vegas_spread'],
                'actual': g['actual_spread'],
                'season': g['season'],
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

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Break-even at -110 odds: 52.4%")
    print("Look for thresholds where Win% consistently exceeds 52.4%")


if __name__ == '__main__':
    run_edge_analysis()
