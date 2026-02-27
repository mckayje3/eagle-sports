"""Ablation study: compare feature sets."""
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

# Load data
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

# Build team stats
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
    if n == 0:
        return None
    return float(np.average(vals[-n:], weights=wts[-n:]))

def get_stats(tid, season):
    td = team_games[tid][season]
    n = len(td['ppg'])
    defaults = {'ppg': 115, 'papg': 115, 'fg_pct': 46, 'three_pct': 36, 'ft_pct': 78,
                'rebounds': 44, 'assists': 25, 'steals': 7.5, 'blocks': 5, 'turnovers': 14}

    if n == 0:
        prev = prev_ratings.get(tid, {})
        return {k: prev.get(k, defaults[k]) for k in defaults}, 0, [], []

    blend = 0.5 ** (n / PREV_HALF_LIFE)
    prev = prev_ratings.get(tid, {})

    stats = {}
    for k in defaults:
        curr = wavg(td[k], td['wts']) if td[k] else None
        prev_val = prev.get(k, defaults[k])
        stats[k] = prev_val if curr is None else blend * prev_val + (1 - blend) * curr

    return stats, n, td['margins'], td['wins']

def get_rest(tid, gdate):
    if tid not in last_game:
        return 3
    curr = datetime.strptime(gdate[:10], '%Y-%m-%d')
    last = datetime.strptime(last_game[tid][:10], '%Y-%m-%d')
    return max(0, min((curr - last).days - 1, 5))

def get_recent_form(margins, n=5):
    return float(np.mean(margins[-n:])) if len(margins) >= n else 0.0

def get_momentum(margins, n=6):
    if len(margins) < n:
        return 0.0
    recent = margins[-n:]
    return np.mean(recent[n//2:]) - np.mean(recent[:n//2])

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

# Build features
X_full = []
X_reduced = []
X_enhanced = []
X_enhanced_plus = []
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
                    'fg_pct': np.mean(td['fg_pct']) if td['fg_pct'] else 46,
                    'three_pct': np.mean(td['three_pct']) if td['three_pct'] else 36,
                    'ft_pct': np.mean(td['ft_pct']) if td['ft_pct'] else 78,
                    'rebounds': np.mean(td['rebounds']) if td['rebounds'] else 44,
                    'assists': np.mean(td['assists']) if td['assists'] else 25,
                    'steals': np.mean(td['steals']) if td['steals'] else 7.5,
                    'blocks': np.mean(td['blocks']) if td['blocks'] else 5,
                    'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 14,
                }
        last_game.clear()
        pg = games[games['season'] == prev]
        if len(pg) > 0:
            league_avg = {'ppg': pg['home_score'].mean(), 'papg': pg['away_score'].mean()}

    for _, g in games[games['season'] == season].iterrows():
        hid, aid = g['home_team_id'], g['away_team_id']
        hs, hn, hm, hw = get_stats(hid, season)
        aws, an, am, aw = get_stats(aid, season)

        if hn >= 10 and an >= 10:
            hr, ar = get_rest(hid, g['date']), get_rest(aid, g['date'])
            net_diff = (hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg'])

            # Full simple (18)
            full = [
                hs['ppg'] - aws['ppg'], hs['papg'] - aws['papg'],
                hs['fg_pct'] - aws['fg_pct'], hs['three_pct'] - aws['three_pct'],
                hs['ft_pct'] - aws['ft_pct'], hs['rebounds'] - aws['rebounds'],
                hs['assists'] - aws['assists'], hs['steals'] - aws['steals'],
                hs['blocks'] - aws['blocks'], hs['turnovers'] - aws['turnovers'],
                hr - ar, 1.0 if hr == 0 else 0.0, 1.0 if ar == 0 else 0.0,
                net_diff, net_diff, min(hn / 20.0, 1.0), min(an / 20.0, 1.0), 1.8
            ]

            # Reduced simple (12) - remove 3P%, FT%, Ast, Stl, Blk, HCA
            reduced = [full[i] for i in [0,1,2,5,9,10,11,12,13,14,15,16]]

            # Enhanced (14) - base enhanced features
            enhanced = [
                hs['ppg'] - aws['ppg'], hs['papg'] - aws['papg'], net_diff,
                get_recent_form(hm, 5) - get_recent_form(am, 5),
                get_momentum(hm, 6) - get_momentum(am, 6),
                get_streak(hw) - get_streak(aw),
                hr - ar, 1.0 if hr == 0 else 0.0, 1.0 if ar == 0 else 0.0,
                2.2, min(hn / 30.0, 1.0), min(an / 30.0, 1.0), 0.0, (hn + an) / 164.0
            ]

            # Enhanced + box score (22)
            enhanced_plus = enhanced + [
                hs['fg_pct'] - aws['fg_pct'], hs['three_pct'] - aws['three_pct'],
                hs['ft_pct'] - aws['ft_pct'], hs['rebounds'] - aws['rebounds'],
                hs['assists'] - aws['assists'], hs['steals'] - aws['steals'],
                hs['blocks'] - aws['blocks'], hs['turnovers'] - aws['turnovers']
            ]

            X_full.append(full)
            X_reduced.append(reduced)
            X_enhanced.append(enhanced)
            X_enhanced_plus.append(enhanced_plus)
            y_all.append(g['away_score'] - g['home_score'])
            vegas_all.append(g['vegas_spread'] if pd.notna(g['vegas_spread']) else 0)
            seasons_all.append(season)

        # Update team stats
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

# Convert
X_full = np.array(X_full)
X_reduced = np.array(X_reduced)
X_enhanced = np.array(X_enhanced)
X_enhanced_plus = np.array(X_enhanced_plus)
y_all = np.array(y_all)
vegas_all = np.array(vegas_all)
seasons_all = np.array(seasons_all)

# Remove NaN
mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(y_all))
X_full, X_reduced, X_enhanced, X_enhanced_plus = X_full[mask], X_reduced[mask], X_enhanced[mask], X_enhanced_plus[mask]
y_all, vegas_all, seasons_all = y_all[mask], vegas_all[mask], seasons_all[mask]

# Split
test_mask = seasons_all == 2025
train_mask = seasons_all < 2025

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

    return mae, blend_mae, acc, blend_acc, X.shape[1]

print('='*75)
print('ABLATION STUDY RESULTS (Test: 2025 season)')
print('='*75)
print(f"{'Model':<35} {'Feat':<6} {'MAE':<8} {'Blend':<8} {'Acc':<8} {'BlendAcc':<8}")
print('-'*75)

results = {}
for X, name in [
    (X_full, 'Simple (full 18)'),
    (X_reduced, 'Simple reduced (12)'),
    (X_enhanced, 'Enhanced (14)'),
    (X_enhanced_plus, 'Enhanced + box score (22)')
]:
    mae, blend_mae, acc, blend_acc, n = evaluate(X, name)
    results[name] = (mae, blend_mae, acc, blend_acc)
    print(f'{name:<35} {n:<6} {mae:<8.3f} {blend_mae:<8.3f} {acc*100:<8.1f} {blend_acc*100:.1f}%')

vegas_mae = np.abs(vegas_all[test_mask] - y_all[test_mask]).mean()
vegas_acc = ((vegas_all[test_mask] < 0) == (y_all[test_mask] < 0)).mean()
print(f"{'Vegas':<35} {'-':<6} {vegas_mae:<8.3f} {'-':<8} {vegas_acc*100:<8.1f} -")

print()
print('='*75)
print('ANALYSIS')
print('='*75)

f_mae, f_blend, f_acc, f_bacc = results['Simple (full 18)']
r_mae, r_blend, r_acc, r_bacc = results['Simple reduced (12)']
e_mae, e_blend, e_acc, e_bacc = results['Enhanced (14)']
ep_mae, ep_blend, ep_acc, ep_bacc = results['Enhanced + box score (22)']

print(f'\n1. Removing 6 weak features from Simple:')
print(f'   MAE:     {f_mae:.3f} -> {r_mae:.3f} ({r_mae - f_mae:+.3f})')
print(f'   Blend:   {f_blend:.3f} -> {r_blend:.3f} ({r_blend - f_blend:+.3f})')
print(f'   Acc:     {f_acc*100:.1f}% -> {r_acc*100:.1f}% ({(r_acc-f_acc)*100:+.1f}%)')

print(f'\n2. Adding 8 box score features to Enhanced:')
print(f'   MAE:     {e_mae:.3f} -> {ep_mae:.3f} ({ep_mae - e_mae:+.3f})')
print(f'   Blend:   {e_blend:.3f} -> {ep_blend:.3f} ({ep_blend - e_blend:+.3f})')
print(f'   Acc:     {e_acc*100:.1f}% -> {ep_acc*100:.1f}% ({(ep_acc-e_acc)*100:+.1f}%)')

print(f'\n3. Best model comparison:')
print(f'   Simple full:       Blend MAE = {f_blend:.3f}, Acc = {f_bacc*100:.1f}%')
print(f'   Enhanced:          Blend MAE = {e_blend:.3f}, Acc = {e_bacc*100:.1f}%')
print(f'   Enhanced + box:    Blend MAE = {ep_blend:.3f}, Acc = {ep_bacc*100:.1f}%')
