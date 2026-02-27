"""
NBA Vegas Blend Analysis

Test optimal blending of our model with Vegas.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).parent / 'nba_games.db'

DECAY = 0.97
MIN_GAMES = 10


def wavg(vals, wts):
    if not vals:
        return None
    n = min(len(vals), len(wts))
    return np.average(vals[-n:], weights=wts[-n:])


def main():
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.date,
            g.home_team_id, g.away_team_id,
            g.home_score, g.away_score,
            o.latest_spread as vegas_spread
        FROM games g
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
    ''', conn)
    conn.close()

    games['actual_spread'] = games['away_score'] - games['home_score']

    # Calculate league HCA
    s23 = games[games['season'] == 2023]
    hca_vals = []
    for tid in s23['home_team_id'].unique():
        hg = s23[s23['home_team_id'] == tid]
        ag = s23[s23['away_team_id'] == tid]
        if len(hg) >= 10 and len(ag) >= 10:
            hca = ((hg['home_score'] - hg['away_score']).mean() -
                   (ag['away_score'] - ag['home_score']).mean()) / 2
            hca_vals.append(hca)
    LEAGUE_HCA = np.mean(hca_vals)

    # Warm up team stats on 2023
    team_stats = defaultdict(lambda: {'ppg': [], 'papg': [], 'wts': []})
    last_game = {}

    for _, g in s23.iterrows():
        for tid, pf, pa in [(g['home_team_id'], g['home_score'], g['away_score']),
                            (g['away_team_id'], g['away_score'], g['home_score'])]:
            ts = team_stats[tid]
            ts['wts'] = [w * DECAY for w in ts['wts']]
            ts['ppg'].append(pf)
            ts['papg'].append(pa)
            ts['wts'].append(1.0)
            last_game[tid] = g['date']

    # Collect predictions for 2024
    s24 = games[games['season'] == 2024]
    predictions = []

    for _, g in s24.iterrows():
        hs = team_stats[g['home_team_id']]
        aws = team_stats[g['away_team_id']]

        if len(hs['ppg']) >= MIN_GAMES and len(aws['ppg']) >= MIN_GAMES and pd.notna(g['vegas_spread']):
            h_ppg = wavg(hs['ppg'], hs['wts'])
            h_papg = wavg(hs['papg'], hs['wts'])
            a_ppg = wavg(aws['ppg'], aws['wts'])
            a_papg = wavg(aws['papg'], aws['wts'])

            # Simple prediction: PPG diff + HCA
            simple_pred = -(h_ppg - a_ppg) - LEAGUE_HCA

            # Net rating based prediction
            net_diff = (h_ppg - h_papg) - (a_ppg - a_papg)
            net_pred = -net_diff / 2 - LEAGUE_HCA

            predictions.append({
                'simple': simple_pred,
                'net': net_pred,
                'vegas': g['vegas_spread'],
                'actual': g['actual_spread'],
                'games_h': len(hs['ppg']),
                'games_a': len(aws['ppg']),
            })

        # Update
        for tid, pf, pa in [(g['home_team_id'], g['home_score'], g['away_score']),
                            (g['away_team_id'], g['away_score'], g['home_score'])]:
            ts = team_stats[tid]
            ts['wts'] = [w * DECAY for w in ts['wts']]
            ts['ppg'].append(pf)
            ts['papg'].append(pa)
            ts['wts'].append(1.0)
            last_game[tid] = g['date']

    df = pd.DataFrame(predictions)

    print("=" * 60)
    print("OPTIMAL VEGAS BLEND ANALYSIS")
    print("=" * 60)

    # Baseline MAEs
    simple_mae = np.abs(df['simple'] - df['actual']).mean()
    net_mae = np.abs(df['net'] - df['actual']).mean()
    vegas_mae = np.abs(df['vegas'] - df['actual']).mean()

    print(f"\nBaseline MAEs (N={len(df)}):")
    print(f"  Simple (PPG diff + HCA): {simple_mae:.4f}")
    print(f"  Net Rating: {net_mae:.4f}")
    print(f"  Vegas: {vegas_mae:.4f}")

    # Optimal blend: Simple + Vegas
    print("\n" + "-" * 60)
    print("Blend: Simple Model + Vegas")
    print("-" * 60)
    print(f"{'Model Weight':<15} {'MAE':<10} {'vs Vegas':<12}")
    print("-" * 40)

    best_w, best_mae = 0, float('inf')
    for w in np.arange(0, 1.01, 0.05):
        blended = w * df['simple'] + (1 - w) * df['vegas']
        mae = np.abs(blended - df['actual']).mean()
        if mae < best_mae:
            best_mae = mae
            best_w = w
        if w in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            print(f"{w:<15.2f} {mae:<10.4f} {mae - vegas_mae:<+12.4f}")

    print(f"\nBest blend: {best_w:.2f} model + {1-best_w:.2f} Vegas")
    print(f"Best MAE: {best_mae:.4f}")
    print(f"Improvement over Vegas: {vegas_mae - best_mae:+.4f}")

    # Check if blending helps by season segment
    print("\n" + "=" * 60)
    print("BLEND PERFORMANCE BY SEASON SEGMENT")
    print("=" * 60)

    # Add game number
    df['game_num'] = range(len(df))

    for lo, hi, label in [(0, 200, 'Early'), (200, 600, 'Mid'), (600, 1300, 'Late')]:
        seg = df[(df['game_num'] >= lo) & (df['game_num'] < hi)]
        if len(seg) < 20:
            continue

        seg_vegas_mae = np.abs(seg['vegas'] - seg['actual']).mean()
        seg_simple_mae = np.abs(seg['simple'] - seg['actual']).mean()

        # Find best blend for this segment
        best_seg_w, best_seg_mae = 0, float('inf')
        for w in np.arange(0, 1.01, 0.05):
            blended = w * seg['simple'] + (1 - w) * seg['vegas']
            mae = np.abs(blended - seg['actual']).mean()
            if mae < best_seg_mae:
                best_seg_mae = mae
                best_seg_w = w

        print(f"\n{label} (games {lo}-{hi}, N={len(seg)}):")
        print(f"  Vegas MAE: {seg_vegas_mae:.4f}")
        print(f"  Simple MAE: {seg_simple_mae:.4f}")
        print(f"  Best blend: {best_seg_w:.2f} model + {1-best_seg_w:.2f} Vegas = {best_seg_mae:.4f}")
        print(f"  Improvement: {seg_vegas_mae - best_seg_mae:+.4f}")

    # What if we use model-only when we have lots of data, Vegas otherwise?
    print("\n" + "=" * 60)
    print("ADAPTIVE STRATEGY")
    print("=" * 60)

    # Use model more when we have more games played
    df['min_games'] = df[['games_h', 'games_a']].min(axis=1)

    # Adaptive weight: more model weight when more games
    for threshold in [20, 30, 40, 50]:
        adaptive_preds = []
        for _, row in df.iterrows():
            if row['min_games'] >= threshold:
                # Use blend
                w = 0.3  # 30% model
            else:
                # Use pure Vegas
                w = 0.0
            pred = w * row['simple'] + (1 - w) * row['vegas']
            adaptive_preds.append(pred)

        adaptive_mae = np.abs(np.array(adaptive_preds) - df['actual'].values).mean()
        print(f"Threshold={threshold}: Use 30% model when min_games >= {threshold}, else Vegas")
        print(f"  MAE: {adaptive_mae:.4f}, vs Vegas: {adaptive_mae - vegas_mae:+.4f}")


if __name__ == '__main__':
    main()
