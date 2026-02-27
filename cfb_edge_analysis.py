"""
CFB Edge Analysis: When Model Disagrees with Vegas, Who is Right?

CFB-specific considerations:
- 130+ teams with wildly varying skill levels
- Many games between mismatched opponents
- Neutral site games for bowls/playoffs
- Home field advantage varies by stadium/atmosphere
- More volatile spreads than NFL
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = 'cfb_games.db'


class QuickCFBModel:
    """Simplified model for quick CFB edge analysis.

    SPREAD CONVENTION: actual_spread = away_score - home_score
    - Negative spread means home team won/favored
    - Positive spread means away team won/favored

    Features are signed so POSITIVE value -> higher spread (away does better).
    """
    DECAY = 0.96
    MIN_GAMES = 3  # Lower threshold for CFB (short seasons)

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
            return 7  # Default CFB rest
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return min((curr - last).days, 30)  # Can be longer due to bye/bowl prep
        except Exception:
            return 7

    def extract_features(self, hid, aid, date, season, neutral_site=False):
        """Extract features with proper signs for spread convention.

        All features are signed so POSITIVE value -> higher spread (away does better).
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

        # HCA - negative for home advantage (lowers spread), zero for neutral site
        hca = 0 if neutral_site else -3.0  # CFB HCA ~3 points, negated

        return np.array([
            # Offensive: away scores more -> higher spread
            a_ppg - h_ppg,

            # Defensive: home allows more -> higher spread
            h_papg - a_papg,

            # Net rating: away better -> higher spread
            a_net - h_net,

            # HCA: home advantage lowers spread (negative constant)
            hca,
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
    print("CFB EDGE ANALYSIS: When Model Disagrees with Vegas, Who is Right?")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site,
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
    games['actual_total'] = games['home_score'] + games['away_score']

    print(f"Total games with Vegas lines: {len(games)}")
    if 'neutral_site' in games.columns:
        neutral_count = games['neutral_site'].sum()
        print(f"Neutral site games: {neutral_count} ({100*neutral_count/len(games):.1f}%)")

    # Run walk-forward prediction
    model = QuickCFBModel()
    X_all, y_all = [], []
    results = []
    current_season = None

    for _, g in games.iterrows():
        # Handle season transitions
        if current_season is not None and g['season'] != current_season:
            model.set_previous_season(g['season'])
        current_season = g['season']

        neutral = g.get('neutral_site', 0) == 1

        feat = model.extract_features(
            g['home_team_id'], g['away_team_id'], g['date'], g['season'],
            neutral_site=neutral
        )

        if feat is not None and len(X_all) >= 100:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.array(X_all))
            ridge = Ridge(alpha=10.0)  # Higher regularization to prevent overfitting
            ridge.fit(X_scaled, np.array(y_all))
            pred = ridge.predict(scaler.transform(feat.reshape(1, -1)))[0]

            results.append({
                'pred': pred,
                'vegas': g['vegas_spread'],
                'actual': g['actual_spread'],
                'actual_total': g['actual_total'],
                'vegas_total': g['vegas_total'],
                'season': g['season'],
                'week': g['week'],
                'neutral_site': neutral,
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

    # Neutral site analysis
    print("\n" + "=" * 70)
    print("NEUTRAL SITE vs REGULAR GAMES (>= 5 pt edge)")
    print("=" * 70)

    for is_neutral in [False, True]:
        sdf = df[df['neutral_site'] == is_neutral]
        if len(sdf) == 0:
            continue

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
            label = "Neutral" if is_neutral else "Regular"
            print(f"{label}: {wins}-{losses} ({pct:.1f}%)")

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

    for start, end in [(1, 4), (5, 8), (9, 12), (13, 20)]:
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
            label = "Bowl Season" if end == 20 else f"Weeks {start}-{end}"
            print(f"{label}: {wins}-{losses} ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Break-even at -110 odds: 52.4%")
    print("Look for thresholds where Win% consistently exceeds 52.4%")


if __name__ == '__main__':
    run_edge_analysis()
