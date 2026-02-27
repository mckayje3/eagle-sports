"""
NHL Deep Eagle Edge Analysis: Neural Network vs Vegas

Uses MLP (Multi-Layer Perceptron) neural network for NHL predictions.

NHL-SPECIFIC NOTES:
- Puck line is almost always ±1.5 (not variable like basketball/football)
- Analysis focuses on predicting whether favorite covers -1.5 vs underdog covers +1.5
- Lower scoring sport = smaller edges in goals

SPREAD CONVENTION: actual_spread = away_score - home_score
- Negative = home team won
- Positive = away team won
- Vegas spread of -1.5 = home favorite (must win by 2+)
- Vegas spread of +1.5 = away favorite (home must lose by 2+)
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DB_PATH = 'nhl_games.db'

# NHL-specific constants
DECAY = 0.97
MIN_GAMES = 3  # Lower for NHL due to limited odds data
LEAGUE_AVG_GOALS = 3.0


class QuickNHLModel:
    """Simplified model for NHL edge analysis."""

    def __init__(self):
        self.team_stats = defaultdict(lambda: {
            'gf': [], 'ga': [], 'wts': [], 'margins': []
        })
        self.last_game = {}

    def _wavg(self, vals, wts):
        if not vals or not wts:
            return LEAGUE_AVG_GOALS
        n = min(len(vals), len(wts))
        return np.average(vals[-n:], weights=wts[-n:])

    def _get_rest(self, tid, date):
        if tid not in self.last_game:
            return 2
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return min((curr - last).days, 7)
        except Exception:
            return 2

    def extract_features(self, hid, aid, date, season):
        """Extract features for prediction.

        Features signed so POSITIVE value -> higher spread (away does better).
        """
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]

        if len(hs['gf']) < MIN_GAMES or len(aws['gf']) < MIN_GAMES:
            return None

        h_gpg = self._wavg(hs['gf'], hs['wts'])
        h_gapg = self._wavg(hs['ga'], hs['wts'])
        a_gpg = self._wavg(aws['gf'], aws['wts'])
        a_gapg = self._wavg(aws['ga'], aws['wts'])

        # Net ratings
        h_net = h_gpg - h_gapg
        a_net = a_gpg - a_gapg

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        # All features signed: positive = away does better = higher spread
        return np.array([
            # Offense: away scores more -> higher spread
            a_gpg - h_gpg,

            # Defense: home allows more -> higher spread
            h_gapg - a_gapg,

            # Net rating: away better -> higher spread
            a_net - h_net,

            # Rest: away has more rest -> higher spread
            ar - hr,

            # Home ice advantage (constant negative - home does better)
            -0.25,
        ])

    def update(self, tid, gf, ga, date):
        ts = self.team_stats[tid]
        ts['wts'] = [w * DECAY for w in ts['wts']]
        ts['gf'].append(gf)
        ts['ga'].append(ga)
        ts['wts'].append(1.0)
        ts['margins'].append(gf - ga)
        self.last_game[tid] = date


def run_edge_analysis():
    """Run edge analysis on NHL games with odds data."""
    print("=" * 70)
    print("NHL DEEP EAGLE EDGE ANALYSIS (MLP Neural Network)")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # First, build team stats from ALL completed games (not just those with odds)
    all_games = pd.read_sql_query('''
        SELECT game_id, date, season, home_team_id, away_team_id,
               home_score, away_score
        FROM games
        WHERE completed = 1
        ORDER BY date
    ''', conn)

    print(f"Total completed games for training: {len(all_games)}")

    # Now get games with Vegas odds for testing
    games_with_odds = pd.read_sql_query('''
        SELECT g.game_id, g.date, g.season, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND o.latest_spread IS NOT NULL
        ORDER BY g.date
    ''', conn)
    conn.close()

    games_with_odds['actual_spread'] = games_with_odds['away_score'] - games_with_odds['home_score']

    print(f"Games with Vegas odds: {len(games_with_odds)}")

    if len(games_with_odds) < 20:
        print("\nINSUFFICIENT DATA: Need at least 20 games with odds for analysis")
        return None

    # Build model using walk-forward on ALL games
    model = QuickNHLModel()
    X_all, y_all = [], []
    results = []

    # Track which game_ids have odds
    odds_game_ids = set(games_with_odds['game_id'].values)

    for _, g in all_games.iterrows():
        hid, aid = g['home_team_id'], g['away_team_id']
        date = g['date']
        actual_spread = g['away_score'] - g['home_score']

        feat = model.extract_features(hid, aid, date, g['season'])

        # If this game has odds AND we have enough training data, make prediction
        if feat is not None and g['game_id'] in odds_game_ids and len(X_all) >= 30:
            # Train model on accumulated data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.array(X_all))

            mlp = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                solver='adam',
                alpha=0.1,  # Higher regularization for small dataset
                max_iter=300,
                random_state=42,
                verbose=False
            )
            mlp.fit(X_scaled, np.array(y_all))

            pred = mlp.predict(scaler.transform(feat.reshape(1, -1)))[0]

            # Get vegas spread for this game
            odds_row = games_with_odds[games_with_odds['game_id'] == g['game_id']].iloc[0]
            vegas = odds_row['vegas_spread']

            results.append({
                'pred': pred,
                'vegas': vegas,
                'actual': actual_spread,
                'season': g['season'],
                'edge': pred - vegas,
            })

        # Add to training data
        if feat is not None:
            X_all.append(feat)
            y_all.append(actual_spread)

        # Update model
        model.update(hid, g['home_score'], g['away_score'], date)
        model.update(aid, g['away_score'], g['home_score'], date)

    if not results:
        print("\nNo predictions generated - insufficient training data")
        return None

    results_df = pd.DataFrame(results)

    print(f"\nGames analyzed: {len(results_df)}")
    print(f"Model MAE: {np.abs(results_df['pred'] - results_df['actual']).mean():.2f}")
    print(f"Vegas MAE: {np.abs(results_df['vegas'] - results_df['actual']).mean():.2f}")

    # Puck line analysis
    print("\n" + "=" * 70)
    print("PUCK LINE ANALYSIS (±1.5 spread)")
    print("=" * 70)

    # Calculate model's predicted cover rate
    # If vegas = -1.5 (home fav), home needs to win by 2+ to cover
    # If vegas = +1.5 (away fav), away needs to win by 2+ to cover

    # Analyze by edge magnitude
    print(f"\n{'Edge':<12} {'Games':>8} {'Wins':>8} {'Losses':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 70)

    # Use smaller thresholds for hockey (lower scoring)
    for threshold in [0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        wins, losses = 0, 0

        # Positive edge: model predicts higher spread than Vegas -> bet away/underdog
        mask_pos = results_df['edge'] >= threshold
        for _, r in results_df[mask_pos].iterrows():
            # If betting "with model" when edge is positive, we're betting away
            # For puck line: away covers if actual > vegas (actual > -1.5 or actual > 1.5)
            result = r['actual'] - r['vegas']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        # Negative edge: model predicts lower spread than Vegas -> bet home/favorite
        mask_neg = results_df['edge'] <= -threshold
        for _, r in results_df[mask_neg].iterrows():
            result = r['vegas'] - r['actual']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        total = wins + losses
        if total > 0:
            win_pct = wins / total * 100
            profit = wins * 0.909 - losses
            roi = profit / total * 100

            marker = " <-- PROFITABLE" if win_pct > 52.4 else ""
            print(f">= {threshold:<8} {total:>8} {wins:>8} {losses:>8} "
                  f"{win_pct:>7.1f}% {roi:>+9.1f}%{marker}")

    # Underdog analysis
    print("\n" + "=" * 70)
    print("UNDERDOG +1.5 ANALYSIS")
    print("=" * 70)

    underdog_wins, underdog_losses = 0, 0
    for _, r in results_df.iterrows():
        # Bet the underdog (+1.5) in all games
        if r['vegas'] < 0:  # Home is favorite, away is underdog
            # Away (underdog) covers if actual > -1.5
            if r['actual'] > -1.5:
                underdog_wins += 1
            else:
                underdog_losses += 1
        else:  # Away is favorite, home is underdog
            # Home (underdog) covers if actual < 1.5
            if r['actual'] < 1.5:
                underdog_wins += 1
            else:
                underdog_losses += 1

    if underdog_wins + underdog_losses > 0:
        underdog_pct = underdog_wins / (underdog_wins + underdog_losses) * 100
        underdog_roi = (underdog_wins * 0.909 - underdog_losses) / (underdog_wins + underdog_losses) * 100
        print(f"Underdog +1.5: {underdog_wins}-{underdog_losses} ({underdog_pct:.1f}%) ROI: {underdog_roi:+.1f}%")
        if underdog_pct > 52.4:
            print("  --> PROFITABLE: Consider always betting underdog +1.5")

    # Totals analysis
    print("\n" + "=" * 70)
    print("TOTALS ANALYSIS (Over/Under)")
    print("=" * 70)

    df_totals = results_df.copy()
    df_totals = df_totals.merge(
        games_with_odds[['game_id', 'vegas_total', 'home_score', 'away_score']].drop_duplicates(),
        left_index=True, right_index=True, how='left'
    )

    # Simple over/under analysis based on Vegas line
    overs, unders = 0, 0
    for _, row in games_with_odds.iterrows():
        actual_total = row['home_score'] + row['away_score']
        vegas_total = row['vegas_total']
        if pd.notna(vegas_total):
            if actual_total > vegas_total:
                overs += 1
            elif actual_total < vegas_total:
                unders += 1

    if overs + unders > 0:
        over_pct = overs / (overs + unders) * 100
        print(f"Overs hit: {overs}/{overs + unders} ({over_pct:.1f}%)")
        print(f"Unders hit: {unders}/{overs + unders} ({100-over_pct:.1f}%)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Break-even at -110 odds: 52.4%")
    print("NHL puck lines are typically ±1.5 (fixed, unlike other sports)")
    print(f"Sample size: {len(results_df)} games (limited - results may not be reliable)")
    print("\nNOTE: NHL odds data is limited. Backfill more historical odds for")
    print("reliable analysis. See backfill_nhl_historical_odds.py")

    return results_df


if __name__ == '__main__':
    run_edge_analysis()
