"""Fair comparison - same test set for both models."""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from datetime import datetime

DB_PATH = 'nba_games.db'
DECAY = 0.97
PREV_HALF_LIFE = 6.0

# Load data WITH box score
conn = sqlite3.connect(DB_PATH)
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
        o.latest_spread as vegas_spread
    FROM games g
    LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
    LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
    LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
    WHERE g.home_score > 0 AND g.completed = 1
    ORDER BY g.date
''', conn)
conn.close()

team_games = defaultdict(lambda: defaultdict(lambda: {
    'ppg': [], 'papg': [], 'wts': [],
    'fg_pct': [], 'three_pct': [], 'ft_pct': [],
    'rebounds': [], 'assists': [], 'steals': [], 'blocks': [], 'turnovers': [],
    'margins': [], 'wins': []
}))
prev_ratings = {}
last_game = {}
league_avg = {'ppg': 115.0, 'papg': 115.0}

def wavg(vals, wts):
    if not vals or not wts:
        return None
    n = min(len(vals), len(wts))
    return float(np.average(vals[-n:], weights=wts[-n:])) if n > 0 else None

X_simple = []
X_enhanced = []
y_all = []
vegas_all = []
seasons_all = []

seasons = sorted(games['season'].unique())

for season in seasons:
    if season > seasons[0]:
        prev = season - 1
        for tid in team_games:
            if prev in team_games[tid] and team_games[tid][prev]['ppg']:
                td = team_games[tid][prev]
                prev_ratings[tid] = {
                    'ppg': np.mean(td['ppg']), 'papg': np.mean(td['papg']),
                    'fg_pct': np.mean(td['fg_pct']) if td['fg_pct'] else np.nan,
                    'three_pct': np.mean(td['three_pct']) if td['three_pct'] else np.nan,
                    'ft_pct': np.mean(td['ft_pct']) if td['ft_pct'] else np.nan,
                    'rebounds': np.mean(td['rebounds']) if td['rebounds'] else np.nan,
                    'assists': np.mean(td['assists']) if td['assists'] else np.nan,
                    'steals': np.mean(td['steals']) if td['steals'] else np.nan,
                    'blocks': np.mean(td['blocks']) if td['blocks'] else np.nan,
                    'turnovers': np.mean(td['turnovers']) if td['turnovers'] else np.nan,
                }
        last_game.clear()
        pg = games[games['season'] == prev]
        if len(pg) > 0:
            league_avg = {'ppg': pg['home_score'].mean(), 'papg': pg['away_score'].mean()}

    for _, g in games[games['season'] == season].iterrows():
        hid, aid = g['home_team_id'], g['away_team_id']
        htd = team_games[hid][season]
        atd = team_games[aid][season]
        hn, an = len(htd['ppg']), len(atd['ppg'])

        if hn >= 10 and an >= 10:
            def get_team_stats(tid, td):
                n = len(td['ppg'])
                ppg = wavg(td['ppg'], td['wts'])
                papg = wavg(td['papg'], td['wts'])
                prev = prev_ratings.get(tid, {})
                blend = 0.5 ** (n / PREV_HALF_LIFE) if ppg else 1.0

                return {
                    'ppg': blend * prev.get('ppg', league_avg['ppg']) + (1-blend) * ppg if ppg else prev.get('ppg', league_avg['ppg']),
                    'papg': blend * prev.get('papg', league_avg['papg']) + (1-blend) * papg if papg else prev.get('papg', league_avg['papg']),
                    'fg_pct': wavg(td['fg_pct'], td['wts'][-len(td['fg_pct']):]) if td['fg_pct'] else prev.get('fg_pct'),
                    'three_pct': wavg(td['three_pct'], td['wts'][-len(td['three_pct']):]) if td['three_pct'] else prev.get('three_pct'),
                    'ft_pct': wavg(td['ft_pct'], td['wts'][-len(td['ft_pct']):]) if td['ft_pct'] else prev.get('ft_pct'),
                    'rebounds': wavg(td['rebounds'], td['wts'][-len(td['rebounds']):]) if td['rebounds'] else prev.get('rebounds'),
                    'assists': wavg(td['assists'], td['wts'][-len(td['assists']):]) if td['assists'] else prev.get('assists'),
                    'steals': wavg(td['steals'], td['wts'][-len(td['steals']):]) if td['steals'] else prev.get('steals'),
                    'blocks': wavg(td['blocks'], td['wts'][-len(td['blocks']):]) if td['blocks'] else prev.get('blocks'),
                    'turnovers': wavg(td['turnovers'], td['wts'][-len(td['turnovers']):]) if td['turnovers'] else prev.get('turnovers'),
                    'n': n, 'margins': td['margins'], 'wins': td['wins'],
                }

            hs = get_team_stats(hid, htd)
            aws = get_team_stats(aid, atd)

            def get_rest(tid):
                if tid not in last_game: return 3
                curr = datetime.strptime(g['date'][:10], '%Y-%m-%d')
                last = datetime.strptime(last_game[tid][:10], '%Y-%m-%d')
                return max(0, min((curr - last).days - 1, 5))

            hr, ar = get_rest(hid), get_rest(aid)
            net_diff = (hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg'])

            # Simple (18)
            simple = [
                hs['ppg'] - aws['ppg'], hs['papg'] - aws['papg'],
                (hs['fg_pct'] - aws['fg_pct']) if hs['fg_pct'] and aws['fg_pct'] else np.nan,
                (hs['three_pct'] - aws['three_pct']) if hs['three_pct'] and aws['three_pct'] else np.nan,
                (hs['ft_pct'] - aws['ft_pct']) if hs['ft_pct'] and aws['ft_pct'] else np.nan,
                (hs['rebounds'] - aws['rebounds']) if hs['rebounds'] and aws['rebounds'] else np.nan,
                (hs['assists'] - aws['assists']) if hs['assists'] and aws['assists'] else np.nan,
                (hs['steals'] - aws['steals']) if hs['steals'] and aws['steals'] else np.nan,
                (hs['blocks'] - aws['blocks']) if hs['blocks'] and aws['blocks'] else np.nan,
                (hs['turnovers'] - aws['turnovers']) if hs['turnovers'] and aws['turnovers'] else np.nan,
                hr - ar, 1.0 if hr == 0 else 0.0, 1.0 if ar == 0 else 0.0,
                net_diff, net_diff, min(hn / 20.0, 1.0), min(an / 20.0, 1.0), 1.8
            ]

            # Enhanced (14)
            def recent_form(margins, n=5):
                return float(np.mean(margins[-n:])) if len(margins) >= n else 0.0
            def momentum(margins, n=6):
                if len(margins) < n: return 0.0
                return np.mean(margins[-n:][n//2:]) - np.mean(margins[-n:][:n//2])
            def streak(wins):
                if not wins: return 0
                s, last = 0, wins[-1]
                for w in reversed(wins):
                    if w == last: s += 1
                    else: break
                return s if last == 1 else -s

            enhanced = [
                hs['ppg'] - aws['ppg'], hs['papg'] - aws['papg'], net_diff,
                recent_form(hs['margins']) - recent_form(aws['margins']),
                momentum(hs['margins']) - momentum(aws['margins']),
                streak(hs['wins']) - streak(aws['wins']),
                hr - ar, 1.0 if hr == 0 else 0.0, 1.0 if ar == 0 else 0.0,
                2.2, min(hn / 30.0, 1.0), min(an / 30.0, 1.0), 0.0, (hn + an) / 164.0
            ]

            X_simple.append(simple)
            X_enhanced.append(enhanced)
            y_all.append(g['away_score'] - g['home_score'])
            vegas_all.append(g['vegas_spread'] if pd.notna(g['vegas_spread']) else np.nan)
            seasons_all.append(season)

        # Update
        for tid, pts_for, pts_ag, fg, three, ft, reb, ast, stl, blk, tov in [
            (hid, g['home_score'], g['away_score'], g['home_fg'], g['home_three'],
             g['home_ft'], g['home_reb'], g['home_ast'], g['home_stl'], g['home_blk'], g['home_tov']),
            (aid, g['away_score'], g['home_score'], g['away_fg'], g['away_three'],
             g['away_ft'], g['away_reb'], g['away_ast'], g['away_stl'], g['away_blk'], g['away_tov'])
        ]:
            td = team_games[tid][season]
            td['wts'] = [w * DECAY for w in td['wts']]
            td['ppg'].append(pts_for)
            td['papg'].append(pts_ag)
            td['wts'].append(1.0)
            margin = pts_for - pts_ag
            td['margins'].append(margin)
            td['wins'].append(1 if margin > 0 else 0)
            if pd.notna(fg): td['fg_pct'].append(fg)
            if pd.notna(three): td['three_pct'].append(three)
            if pd.notna(ft): td['ft_pct'].append(ft)
            if pd.notna(reb): td['rebounds'].append(reb)
            if pd.notna(ast): td['assists'].append(ast)
            if pd.notna(stl): td['steals'].append(stl)
            if pd.notna(blk): td['blocks'].append(blk)
            if pd.notna(tov): td['turnovers'].append(tov)
            last_game[tid] = g['date']

X_simple = np.array(X_simple)
X_enhanced = np.array(X_enhanced)
y_all = np.array(y_all)
vegas_all = np.array(vegas_all)
seasons_all = np.array(seasons_all)

# Drop NaN - same rows for both
nan_mask = np.isnan(X_simple).any(axis=1) | np.isnan(y_all) | np.isnan(vegas_all)
print(f'Total samples before NaN removal: {len(X_simple)}')
print(f'Samples with NaN: {nan_mask.sum()}')

X_simple = X_simple[~nan_mask]
X_enhanced = X_enhanced[~nan_mask]
y_all = y_all[~nan_mask]
vegas_all = vegas_all[~nan_mask]
seasons_all = seasons_all[~nan_mask]

print(f'Total samples after NaN removal: {len(X_simple)}')

test_mask = seasons_all == 2025
train_mask = seasons_all < 2025

print(f'Train samples: {train_mask.sum()}')
print(f'Test samples (2025): {test_mask.sum()}')

def evaluate(X, name):
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_all[train_mask], y_all[test_mask]
    vegas_test = vegas_all[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

    pred = model.predict(X_test_s)
    blended = 0.9 * pred + 0.1 * vegas_test

    mae = np.abs(pred - y_test).mean()
    blend_mae = np.abs(blended - y_test).mean()
    acc = ((pred < 0) == (y_test < 0)).mean()
    blend_acc = ((blended < 0) == (y_test < 0)).mean()

    return mae, blend_mae, acc, blend_acc

print()
print('='*70)
print('FAIR COMPARISON - Same test set (games with complete box score data)')
print('='*70)
print(f"{'Model':<25} {'Features':<10} {'MAE':<10} {'Blend MAE':<12} {'Acc':<10} {'Blend Acc'}")
print('-'*70)

s_mae, s_blend, s_acc, s_bacc = evaluate(X_simple, 'Simple')
e_mae, e_blend, e_acc, e_bacc = evaluate(X_enhanced, 'Enhanced')
v_mae = np.abs(vegas_all[test_mask] - y_all[test_mask]).mean()
v_acc = ((vegas_all[test_mask] < 0) == (y_all[test_mask] < 0)).mean()

print(f"{'Simple (18 feat)':<25} {18:<10} {s_mae:<10.3f} {s_blend:<12.3f} {s_acc*100:<10.1f} {s_bacc*100:.1f}%")
print(f"{'Enhanced (14 feat)':<25} {14:<10} {e_mae:<10.3f} {e_blend:<12.3f} {e_acc*100:<10.1f} {e_bacc*100:.1f}%")
print(f"{'Vegas':<25} {'-':<10} {v_mae:<10.3f} {'-':<12} {v_acc*100:<10.1f} -")

print()
print('DIFFERENCE (Enhanced - Simple):')
print(f'  MAE:       {e_mae - s_mae:+.3f}')
print(f'  Blend MAE: {e_blend - s_blend:+.3f}')
print(f'  Accuracy:  {(e_acc - s_acc)*100:+.2f}%')
