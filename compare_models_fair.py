"""Fair head-to-head comparison of Simple vs Enhanced models on same test set."""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from datetime import datetime

DB_PATH = 'nba_games.db'

# Load games
conn = sqlite3.connect(DB_PATH)
games = pd.read_sql_query('''
    SELECT
        g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
        g.home_score, g.away_score,
        hs.field_goal_pct as home_fg, hs.total_rebounds as home_reb,
        hs.turnovers as home_tov,
        aws.field_goal_pct as away_fg, aws.total_rebounds as away_reb,
        aws.turnovers as away_tov,
        o.latest_spread as vegas_spread, o.latest_total as vegas_total
    FROM games g
    LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
    LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
    LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
    WHERE g.home_score > 0 AND g.completed = 1
    ORDER BY g.date
''', conn)
conn.close()

print('=' * 70)
print('FAIR HEAD-TO-HEAD COMPARISON')
print('Same test set: 2025 season games WITH Vegas lines')
print('=' * 70)

DECAY_SIMPLE = 0.97
DECAY_ENHANCED = 0.93
PREV_HL = 6.0

team_stats = defaultdict(lambda: defaultdict(lambda: {
    'ppg': [], 'papg': [], 'wts_s': [], 'wts_e': [],
    'fg_pct': [], 'reb': [], 'tov': [],
    'margins': [], 'wins': []
}))
prev_ratings = {}
last_game = {}


def wavg(vals, wts):
    if not vals or not wts:
        return None
    n = min(len(vals), len(wts))
    return float(np.average(vals[-n:], weights=wts[-n:])) if n > 0 else None


X_simple, X_enhanced = [], []
X_total_simple, X_total_enhanced = [], []
y_spread, y_total = [], []
vegas_spreads, vegas_totals = [], []
seasons_all = []

seasons = sorted(games['season'].unique())

for season in seasons:
    if season > seasons[0]:
        prev = season - 1
        for tid in team_stats:
            if prev in team_stats[tid] and team_stats[tid][prev]['ppg']:
                td = team_stats[tid][prev]
                prev_ratings[tid] = {
                    'ppg': np.mean(td['ppg']), 'papg': np.mean(td['papg']),
                    'fg_pct': np.mean(td['fg_pct']) if td['fg_pct'] else 46.0,
                    'reb': np.mean(td['reb']) if td['reb'] else 44.0,
                    'tov': np.mean(td['tov']) if td['tov'] else 14.0,
                }
        last_game.clear()

    for _, g in games[games['season'] == season].iterrows():
        hid, aid = g['home_team_id'], g['away_team_id']
        htd, atd = team_stats[hid][season], team_stats[aid][season]
        hn, an = len(htd['ppg']), len(atd['ppg'])

        # Only include games with Vegas lines and enough history
        if hn >= 10 and an >= 10 and pd.notna(g['vegas_spread']):
            def get_stats(tid, td, decay):
                wts_key = 'wts_s' if decay == DECAY_SIMPLE else 'wts_e'
                n = len(td['ppg'])
                ppg = wavg(td['ppg'], td[wts_key])
                papg = wavg(td['papg'], td[wts_key])
                prev = prev_ratings.get(tid, {})
                blend = 0.5 ** (n / PREV_HL) if ppg else 1.0

                fg = wavg(td['fg_pct'], td[wts_key][-len(td['fg_pct']):]) if td['fg_pct'] else None
                reb = wavg(td['reb'], td[wts_key][-len(td['reb']):]) if td['reb'] else None
                tov = wavg(td['tov'], td[wts_key][-len(td['tov']):]) if td['tov'] else None

                return {
                    'ppg': blend * prev.get('ppg', 115) + (1 - blend) * ppg if ppg else prev.get('ppg', 115),
                    'papg': blend * prev.get('papg', 115) + (1 - blend) * papg if papg else prev.get('papg', 115),
                    'fg_pct': fg if fg else prev.get('fg_pct', 46.0),
                    'reb': reb if reb else prev.get('reb', 44.0),
                    'tov': tov if tov else prev.get('tov', 14.0),
                    'n': n, 'margins': td['margins'], 'wins': td['wins'],
                }

            def get_rest(tid):
                if tid not in last_game:
                    return 3
                curr = datetime.strptime(g['date'][:10], '%Y-%m-%d')
                last = datetime.strptime(last_game[tid][:10], '%Y-%m-%d')
                return max(0, min((curr - last).days - 1, 5))

            hr, ar = get_rest(hid), get_rest(aid)

            # Simple model stats (decay=0.97)
            hs_s = get_stats(hid, htd, DECAY_SIMPLE)
            as_s = get_stats(aid, atd, DECAY_SIMPLE)

            # Enhanced model stats (decay=0.93)
            hs_e = get_stats(hid, htd, DECAY_ENHANCED)
            as_e = get_stats(aid, atd, DECAY_ENHANCED)

            # Simple spread features (12)
            net_s = (hs_s['ppg'] - hs_s['papg']) - (as_s['ppg'] - as_s['papg'])
            simple_spread = [
                hs_s['ppg'] - as_s['ppg'], hs_s['papg'] - as_s['papg'],
                hs_s['fg_pct'] - as_s['fg_pct'], hs_s['reb'] - as_s['reb'],
                hs_s['tov'] - as_s['tov'], hr - ar,
                1.0 if hr == 0 else 0.0, 1.0 if ar == 0 else 0.0,
                net_s, net_s,
                min(hn / 20.0, 1.0), min(an / 20.0, 1.0),
            ]

            # Enhanced spread features (17)
            def recent_form(margins, n=5):
                return float(np.mean(margins[-n:])) if len(margins) >= n else 0.0

            def momentum(margins, n=6):
                if len(margins) < n:
                    return 0.0
                return np.mean(margins[-n:][n // 2:]) - np.mean(margins[-n:][:n // 2])

            def streak(wins):
                if not wins:
                    return 0
                s, last = 0, wins[-1]
                for w in reversed(wins):
                    if w == last:
                        s += 1
                    else:
                        break
                return s if last == 1 else -s

            net_e = (hs_e['ppg'] - hs_e['papg']) - (as_e['ppg'] - as_e['papg'])
            enhanced_spread = [
                hs_e['ppg'] - as_e['ppg'], hs_e['papg'] - as_e['papg'], net_e,
                recent_form(hs_e['margins']) - recent_form(as_e['margins']),
                momentum(hs_e['margins']) - momentum(as_e['margins']),
                streak(hs_e['wins']) - streak(as_e['wins']),
                hr - ar, 1.0 if hr == 0 else 0.0, 1.0 if ar == 0 else 0.0,
                2.2, min(hn / 30.0, 1.0), min(an / 30.0, 1.0),
                0.0, (hn + an) / 164.0,
                hs_e['fg_pct'] - as_e['fg_pct'], hs_e['reb'] - as_e['reb'],
                hs_e['tov'] - as_e['tov'],
            ]

            # Simple total features (6)
            simple_total = [
                hs_s['ppg'] + as_s['ppg'], hs_s['papg'] + as_s['papg'],
                1.0 if hr == 0 else 0.0, 1.0 if ar == 0 else 0.0,
                hs_s['reb'] + as_s['reb'], hs_s['tov'] + as_s['tov'],
            ]

            # Enhanced total features (15) - mirrors spread features for totals
            home_recent_total = abs(recent_form(hs_e['margins']))
            away_recent_total = abs(recent_form(as_e['margins']))
            home_momentum_abs = abs(momentum(hs_e['margins']))
            away_momentum_abs = abs(momentum(as_e['margins']))

            enhanced_total = [
                hs_e['ppg'] + as_e['ppg'],                   # 0: Combined PPG
                hs_e['papg'] + as_e['papg'],                 # 1: Combined PAPG
                (hs_e['ppg'] + hs_e['papg']) / 2,            # 2: Home pace proxy
                (as_e['ppg'] + as_e['papg']) / 2,            # 3: Away pace proxy
                1.0 if hr == 0 else 0.0,                     # 4: Home B2B
                1.0 if ar == 0 else 0.0,                     # 5: Away B2B
                min(hn / 30.0, 1.0),                         # 6: Home reliability
                min(an / 30.0, 1.0),                         # 7: Away reliability
                hs_e['fg_pct'] + as_e['fg_pct'],             # 8: Combined FG%
                hs_e['reb'] + as_e['reb'],                   # 9: Combined rebounds
                hs_e['tov'] + as_e['tov'],                   # 10: Combined TOV
                home_recent_total + away_recent_total,       # 11: Recent game intensity
                home_momentum_abs + away_momentum_abs,       # 12: Combined momentum (abs)
                (hn + an) / 164.0,                           # 13: Season progress
                0.0,                                         # 14: Injury adj (not available in this script)
            ]

            X_simple.append(simple_spread)
            X_enhanced.append(enhanced_spread)
            X_total_simple.append(simple_total)
            X_total_enhanced.append(enhanced_total)
            y_spread.append(g['away_score'] - g['home_score'])
            y_total.append(g['home_score'] + g['away_score'])
            vegas_spreads.append(g['vegas_spread'])
            vegas_totals.append(g['vegas_total'])
            seasons_all.append(season)

        # Update stats for both decay rates
        for tid, pf, pa, fg, reb, tov in [
            (hid, g['home_score'], g['away_score'], g['home_fg'], g['home_reb'], g['home_tov']),
            (aid, g['away_score'], g['home_score'], g['away_fg'], g['away_reb'], g['away_tov'])
        ]:
            td = team_stats[tid][season]
            td['wts_s'] = [w * DECAY_SIMPLE for w in td['wts_s']]
            td['wts_e'] = [w * DECAY_ENHANCED for w in td['wts_e']]
            td['ppg'].append(pf)
            td['papg'].append(pa)
            td['wts_s'].append(1.0)
            td['wts_e'].append(1.0)
            margin = pf - pa
            td['margins'].append(margin)
            td['wins'].append(1 if margin > 0 else 0)
            if pd.notna(fg):
                td['fg_pct'].append(fg)
            if pd.notna(reb):
                td['reb'].append(reb)
            if pd.notna(tov):
                td['tov'].append(tov)
            last_game[tid] = g['date']

# Convert to arrays
X_simple = np.array(X_simple)
X_enhanced = np.array(X_enhanced)
X_total_simple = np.array(X_total_simple)
X_total_enhanced = np.array(X_total_enhanced)
y_spread = np.array(y_spread)
y_total = np.array(y_total)
vegas_spreads = np.array(vegas_spreads)
vegas_totals = np.array(vegas_totals)
seasons_all = np.array(seasons_all)

# Drop NaN
nan_mask = (np.isnan(X_simple).any(axis=1) | np.isnan(X_enhanced).any(axis=1) |
            np.isnan(y_spread) | np.isnan(vegas_spreads))
X_simple = X_simple[~nan_mask]
X_enhanced = X_enhanced[~nan_mask]
X_total_simple = X_total_simple[~nan_mask]
X_total_enhanced = X_total_enhanced[~nan_mask]
y_spread = y_spread[~nan_mask]
y_total = y_total[~nan_mask]
vegas_spreads = vegas_spreads[~nan_mask]
vegas_totals = vegas_totals[~nan_mask]
seasons_all = seasons_all[~nan_mask]

# Train/test split
test_mask = seasons_all == 2025
train_mask = seasons_all < 2025

print(f'Train: {train_mask.sum()}, Test: {test_mask.sum()} (all with Vegas lines)')


def evaluate(X, y, vegas, train_mask, test_mask):
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    v_test = vegas[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)

    mae = np.abs(pred - y_test).mean()
    vegas_mae = np.abs(v_test - y_test).mean()

    # Winner accuracy (for spread)
    acc = ((pred < 0) == (y_test < 0)).mean()
    vegas_acc = ((v_test < 0) == (y_test < 0)).mean()

    return mae, acc, vegas_mae, vegas_acc


# Spread comparison
s_mae, s_acc, v_mae_s, v_acc = evaluate(X_simple, y_spread, vegas_spreads, train_mask, test_mask)
e_mae, e_acc, _, _ = evaluate(X_enhanced, y_spread, vegas_spreads, train_mask, test_mask)

# Total comparison
st_mae, _, vt_mae, _ = evaluate(X_total_simple, y_total, vegas_totals, train_mask, test_mask)
et_mae, _, _, _ = evaluate(X_total_enhanced, y_total, vegas_totals, train_mask, test_mask)

print()
print(f'SPREAD PREDICTION (same {test_mask.sum()} games):')
print(f'{"Model":<20} {"MAE":<10} {"Winner Acc":<12}')
print('-' * 45)
print(f'{"Simple (12 feat)":<20} {s_mae:<10.2f} {s_acc * 100:.1f}%')
print(f'{"Enhanced (17 feat)":<20} {e_mae:<10.2f} {e_acc * 100:.1f}%')
print(f'{"Vegas":<20} {v_mae_s:<10.2f} {v_acc * 100:.1f}%')

print()
print(f'TOTAL PREDICTION (same {test_mask.sum()} games):')
print(f'{"Model":<20} {"MAE":<10}')
print('-' * 30)
print(f'{"Simple (6 feat)":<20} {st_mae:.2f}')
print(f'{"Enhanced (15 feat)":<20} {et_mae:.2f}')
print(f'{"Vegas":<20} {vt_mae:.2f}')
