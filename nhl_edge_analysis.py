"""
NHL Edge Analysis: When Model Disagrees with Vegas, Who is Right?

NHL-specific considerations:
- Low scoring sport (avg ~3 goals/team)
- Puck line typically ±1.5 goals (different from spread sports)
- Home ice advantage ~55% win rate
- Higher variance per game than basketball
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = 'nhl_games.db'


class QuickNHLModel:
    """Simplified model for quick NHL edge analysis."""
    DECAY = 0.97
    MIN_GAMES = 5

    def __init__(self):
        self.team_stats = defaultdict(lambda: {
            'gf': [], 'ga': [], 'wts': [], 'margins': [], 'wins': []
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
            return 2  # Default NHL rest
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return min((curr - last).days, 7)
        except Exception:
            return 2

    def extract_features(self, hid, aid, date, season):
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]

        if len(hs['gf']) < self.MIN_GAMES or len(aws['gf']) < self.MIN_GAMES:
            return None

        h_gpg = self._wavg(hs['gf'], hs['wts'])
        h_gapg = self._wavg(hs['ga'], hs['wts'])
        a_gpg = self._wavg(aws['gf'], aws['wts'])
        a_gapg = self._wavg(aws['ga'], aws['wts'])

        h_recent = np.mean(hs['gf'][-5:]) if len(hs['gf']) >= 5 else h_gpg
        a_recent = np.mean(aws['gf'][-5:]) if len(aws['gf']) >= 5 else a_gpg

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
            h_gpg - a_gpg,
            h_gapg - a_gapg,
            (h_gpg - h_gapg) - (a_gpg - a_gapg),  # Net rating diff
            h_recent - a_recent,
            get_trend(hs['margins']) - get_trend(aws['margins']),
            get_streak(hs['wins']) - get_streak(aws['wins']),
            hr - ar,  # Rest advantage
            0.25,  # Home ice advantage in goals
            min(len(hs['gf']) / 30, 1),  # Reliability
            min(len(aws['gf']) / 30, 1),
        ])

    def update(self, tid, gf, ga, date, won):
        ts = self.team_stats[tid]
        ts['wts'] = [w * self.DECAY for w in ts['wts']]
        ts['gf'].append(gf)
        ts['ga'].append(ga)
        ts['wts'].append(1.0)
        ts['margins'].append(gf - ga)
        ts['wins'].append(1 if won else 0)
        self.last_game[tid] = date

    def set_previous_season(self, season):
        """Carry over previous season ratings."""
        for tid in self.team_stats:
            td = self.team_stats[tid]
            if td['gf']:
                self.prev_ratings[tid] = {
                    'gpg': np.mean(td['gf']),
                    'gapg': np.mean(td['ga']),
                }
        self.last_game.clear()


def run_edge_analysis():
    """Analyze when model disagreement with Vegas = profitable edge."""
    print("=" * 70)
    print("NHL EDGE ANALYSIS: When Model Disagrees with Vegas, Who is Right?")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
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
    games = games[games['vegas_spread'].notna()].copy()
    games['actual_spread'] = games['away_score'] - games['home_score']
    games['actual_total'] = games['home_score'] + games['away_score']

    print(f"Total games with Vegas lines: {len(games)}")

    # Run walk-forward prediction
    model = QuickNHLModel()
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

        if feat is not None and len(X_all) >= 100:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.array(X_all))
            ridge = Ridge(alpha=10.0)  # Higher regularization for lower-scoring sport
            ridge.fit(X_scaled, np.array(y_all))
            pred = ridge.predict(scaler.transform(feat.reshape(1, -1)))[0]

            results.append({
                'pred': pred,
                'vegas': g['vegas_spread'],
                'actual': g['actual_spread'],
                'actual_total': g['actual_total'],
                'vegas_total': g['vegas_total'],
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

    # ATS analysis by edge size (using smaller thresholds for hockey)
    print("\n" + "=" * 70)
    print("ATS RECORD BY EDGE SIZE (Enhanced Ridge)")
    print("(Puck line bets - typical line is ±1.5)")
    print("=" * 70)
    print(f"{'Edge':<12} {'Games':>8} {'Wins':>8} {'Losses':>8} {'Push':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 70)

    # NHL edges will be smaller since scoring is lower
    for threshold in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
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
    print("(Over/Under - typical totals line is 5.5-6.5)")
    print("=" * 70)

    df_totals = df[df['vegas_total'].notna()].copy()
    if len(df_totals) > 0:
        print(f"{'Edge':<12} {'Games':>8} {'Wins':>8} {'Losses':>8} {'Push':>8} {'Win%':>8}")
        print("-" * 70)

        for threshold in [0, 0.25, 0.5, 0.75, 1.0]:
            wins, losses, pushes = 0, 0, 0

            # Model predicts higher total than Vegas -> bet OVER
            mask_over = (df_totals['actual_total'] - df_totals['vegas_total']) >= threshold
            for _, r in df_totals[mask_over].iterrows():
                diff = r['actual_total'] - r['vegas_total']
                if diff > 0.5:
                    wins += 1
                elif diff < -0.5:
                    losses += 1
                else:
                    pushes += 1

            total = wins + losses
            if total > 0:
                win_pct = wins / total * 100
                marker = " <-- PROFITABLE" if win_pct > 52.4 else ""
                print(f">= {threshold:<8} {wins + losses + pushes:>8} {wins:>8} {losses:>8} {pushes:>8} "
                      f"{win_pct:>7.1f}%{marker}")

    # By season
    print("\n" + "=" * 70)
    print("EDGE PERFORMANCE BY SEASON (>= 1.0 goal edge)")
    print("=" * 70)

    for season in sorted(df['season'].unique()):
        sdf = df[df['season'] == season]
        wins, losses = 0, 0

        for _, r in sdf[sdf['edge'] >= 1.0].iterrows():
            result = r['actual'] - r['vegas']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        for _, r in sdf[sdf['edge'] <= -1.0].iterrows():
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
    print("NHL puck lines are typically ±1.5 (vs spread in other sports)")
    print("Smaller edge thresholds due to lower scoring")


if __name__ == '__main__':
    run_edge_analysis()
