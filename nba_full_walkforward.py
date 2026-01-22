"""
NBA Full Walk-Forward Validation

True online learning: train on all past games, predict next game, repeat.
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


def main():
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT
            g.game_id, g.season, g.date,
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

    # Calculate HCA from 2023
    s23 = games[games['season'] == 2023]
    raw_hca = {}
    for tid in s23['home_team_id'].unique():
        hg = s23[s23['home_team_id'] == tid]
        ag = s23[s23['away_team_id'] == tid]
        if len(hg) >= 10 and len(ag) >= 10:
            raw_hca[tid] = ((hg['home_score'] - hg['away_score']).mean() -
                            (ag['away_score'] - ag['home_score']).mean()) / 2

    league_hca = np.mean(list(raw_hca.values()))
    print(f"League HCA: {league_hca:.2f}")

    # Model settings
    DECAY = 0.97
    MIN_GAMES = 10
    ALPHA = 0.1

    team_stats = defaultdict(lambda: {'ppg': [], 'papg': [], 'wts': []})
    last_game = {}

    def get_rest(tid, date):
        if tid not in last_game:
            return 3
        from datetime import datetime
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(last_game[tid][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 2

    def wavg(vals, wts):
        if not vals:
            return None
        n = min(len(vals), len(wts))
        return np.average(vals[-n:], weights=wts[-n:])

    def get_features(hid, aid, date):
        hs = team_stats[hid]
        aws = team_stats[aid]
        if not hs['ppg'] or not aws['ppg']:
            return None
        if len(hs['ppg']) < MIN_GAMES or len(aws['ppg']) < MIN_GAMES:
            return None

        h_ppg = wavg(hs['ppg'], hs['wts'])
        h_papg = wavg(hs['papg'], hs['wts'])
        a_ppg = wavg(aws['ppg'], aws['wts'])
        a_papg = wavg(aws['papg'], aws['wts'])

        hr = get_rest(hid, date)
        ar = get_rest(aid, date)

        hca = league_hca

        return np.array([
            h_ppg - a_ppg,
            h_papg - a_papg,
            (h_ppg - h_papg) - (a_ppg - a_papg),
            hr - ar,
            1 if hr == 0 else 0,
            1 if ar == 0 else 0,
            hca,
            min(len(hs['ppg']) / 30, 1),
            min(len(aws['ppg']) / 30, 1),
        ])

    def update_team(tid, pf, pa, date):
        ts = team_stats[tid]
        ts['wts'] = [w * DECAY for w in ts['wts']]
        ts['ppg'].append(pf)
        ts['papg'].append(pa)
        ts['wts'].append(1.0)
        last_game[tid] = date

    # Warm up on 2023
    print("\nWarming up on 2023 season...")
    for _, g in games[games['season'] == 2023].iterrows():
        update_team(g['home_team_id'], g['home_score'], g['away_score'], g['date'])
        update_team(g['away_team_id'], g['away_score'], g['home_score'], g['date'])

    # Full walk-forward on 2024
    print("Running walk-forward on 2024...")
    s24 = games[games['season'] == 2024].copy()

    X_all, y_all = [], []
    results = []

    for idx, g in s24.iterrows():
        feat = get_features(g['home_team_id'], g['away_team_id'], g['date'])

        if feat is not None:
            if len(X_all) >= 50:
                X_train = np.array(X_all)
                y_train = np.array(y_all)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)

                ridge = Ridge(alpha=ALPHA)
                ridge.fit(X_scaled, y_train)

                feat_scaled = scaler.transform(feat.reshape(1, -1))
                pred = ridge.predict(feat_scaled)[0]

                results.append({
                    'pred': pred,
                    'actual': g['actual_spread'],
                    'vegas': g['vegas_spread'],
                    'games_trained': len(X_all)
                })

            X_all.append(feat)
            y_all.append(g['actual_spread'])

        update_team(g['home_team_id'], g['home_score'], g['away_score'], g['date'])
        update_team(g['away_team_id'], g['away_score'], g['home_score'], g['date'])

    df = pd.DataFrame(results)
    df = df[df['vegas'].notna()]

    print(f"\nTotal predictions: {len(df)}")

    mae_model = np.abs(df['pred'] - df['actual']).mean()
    mae_vegas = np.abs(df['vegas'] - df['actual']).mean()
    rmse_model = np.sqrt(((df['pred'] - df['actual']) ** 2).mean())
    rmse_vegas = np.sqrt(((df['vegas'] - df['actual']) ** 2).mean())

    print("\n" + "=" * 50)
    print("FULL WALK-FORWARD RESULTS")
    print("=" * 50)
    print(f"\n{'Model':<15} {'MAE':<10} {'RMSE':<10}")
    print("-" * 35)
    print(f"{'Ridge':<15} {mae_model:<10.4f} {rmse_model:<10.4f}")
    print(f"{'Vegas':<15} {mae_vegas:<10.4f} {rmse_vegas:<10.4f}")
    print(f"{'Difference':<15} {mae_model - mae_vegas:<+10.4f} {rmse_model - rmse_vegas:<+10.4f}")

    # By part of season
    print("\n" + "=" * 50)
    print("MAE BY GAMES TRAINED (season progress)")
    print("=" * 50)

    for lo, hi in [(50, 200), (200, 400), (400, 700), (700, 1200)]:
        mask = (df['games_trained'] >= lo) & (df['games_trained'] < hi)
        if mask.sum() > 20:
            m_mae = np.abs(df.loc[mask, 'pred'] - df.loc[mask, 'actual']).mean()
            v_mae = np.abs(df.loc[mask, 'vegas'] - df.loc[mask, 'actual']).mean()
            print(f"Games {lo:4d}-{hi:4d}: Model={m_mae:.3f}, Vegas={v_mae:.3f}, "
                  f"Diff={m_mae - v_mae:+.3f}, N={mask.sum()}")

    # Feature importance
    print("\n" + "=" * 50)
    print("RIDGE COEFFICIENTS (final model)")
    print("=" * 50)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.array(X_all))
    ridge = Ridge(alpha=ALPHA)
    ridge.fit(X_scaled, np.array(y_all))

    names = ['PPG diff', 'PAPG diff', 'Net rating', 'Rest diff',
             'Home B2B', 'Away B2B', 'HCA', 'Home rel', 'Away rel']
    for name, coef in sorted(zip(names, ridge.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:<15} {coef:+.4f}")

    # Simple baseline comparison
    print("\n" + "=" * 50)
    print("SIMPLE BASELINE COMPARISON")
    print("=" * 50)

    # What if we just used PPG diff + HCA?
    simple_preds = []
    team_stats2 = defaultdict(lambda: {'ppg': [], 'wts': []})

    for _, g in games[games['season'] == 2023].iterrows():
        for tid, pts in [(g['home_team_id'], g['home_score']), (g['away_team_id'], g['away_score'])]:
            ts = team_stats2[tid]
            ts['wts'] = [w * DECAY for w in ts['wts']]
            ts['ppg'].append(pts)
            ts['wts'].append(1.0)

    for _, g in s24.iterrows():
        hs = team_stats2[g['home_team_id']]
        aws = team_stats2[g['away_team_id']]

        if len(hs['ppg']) >= MIN_GAMES and len(aws['ppg']) >= MIN_GAMES:
            h_ppg = wavg(hs['ppg'], hs['wts'])
            a_ppg = wavg(aws['ppg'], aws['wts'])
            pred = -(h_ppg - a_ppg) - league_hca  # Simple: just PPG diff + HCA
            simple_preds.append({
                'pred': pred,
                'actual': g['actual_spread'],
                'vegas': g['vegas_spread']
            })

        for tid, pts in [(g['home_team_id'], g['home_score']), (g['away_team_id'], g['away_score'])]:
            ts = team_stats2[tid]
            ts['wts'] = [w * DECAY for w in ts['wts']]
            ts['ppg'].append(pts)
            ts['wts'].append(1.0)

    simple_df = pd.DataFrame(simple_preds)
    simple_df = simple_df[simple_df['vegas'].notna()]

    simple_mae = np.abs(simple_df['pred'] - simple_df['actual']).mean()
    simple_vegas_mae = np.abs(simple_df['vegas'] - simple_df['actual']).mean()

    print(f"\nSimple (PPG diff + HCA only):")
    print(f"  MAE: {simple_mae:.4f}")
    print(f"  Vegas MAE: {simple_vegas_mae:.4f}")
    print(f"  Difference: {simple_mae - simple_vegas_mae:+.4f}")

    print(f"\nRidge improvement over simple: {simple_mae - mae_model:+.4f}")


if __name__ == '__main__':
    main()
