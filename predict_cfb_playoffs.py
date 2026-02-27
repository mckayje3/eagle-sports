"""Generate predictions for CFB playoff games."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).parent / 'cfb_games.db'

# CFB Simple model constants (optimized)
DECAY = 0.88
PREV_HALF_LIFE = 5.0
MIN_GAMES = 2


def main():
    team_stats = defaultdict(lambda: defaultdict(lambda: {
        'ppg': [], 'papg': [], 'wts': [],
        'yards': [], 'yards_wts': [],
        'pass_yards': [], 'pass_wts': [],
        'rush_yards': [], 'rush_wts': [],
        'turnovers': [], 'to_wts': [],
        'first_downs': [], 'fd_wts': [],
        'margins': [], 'wins': [],
    }))
    prev_ratings = {}
    last_game = {}
    league_avg = {
        'ppg': 28.0, 'papg': 28.0, 'yards': 400.0,
        'pass_yards': 230.0, 'rush_yards': 170.0,
        'turnovers': 1.5, 'first_downs': 20.0
    }

    spread_model = None
    spread_scaler = StandardScaler()
    spread_X, spread_y = [], []

    total_model = None
    total_scaler = StandardScaler()
    total_X, total_y = [], []

    def wavg(vals, wts):
        if not vals or not wts:
            return None
        return float(np.average(vals, weights=wts))

    def get_rest(tid, date):
        if tid not in last_game:
            return 7
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(last_game[tid][:10], '%Y-%m-%d')
            return min((curr - last).days, 21)  # CFB can have longer breaks
        except Exception:
            return 7

    def get_stats(tid, season):
        td = team_stats[tid][season]
        n = len(td['ppg'])
        if n == 0:
            prev = prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', league_avg['ppg']),
                'papg': prev.get('papg', league_avg['papg']),
                'yards': prev.get('yards', league_avg['yards']),
                'pass_yards': prev.get('pass_yards', league_avg['pass_yards']),
                'rush_yards': prev.get('rush_yards', league_avg['rush_yards']),
                'turnovers': prev.get('turnovers', league_avg['turnovers']),
                'first_downs': prev.get('first_downs', league_avg['first_downs']),
                'games': 0,
                'margins': [],
                'wins': [],
            }

        ppg = wavg(td['ppg'], td['wts'])
        papg = wavg(td['papg'], td['wts'])
        yards = wavg(td['yards'], td['yards_wts']) if td['yards'] else league_avg['yards']
        pass_yds = wavg(td['pass_yards'], td['pass_wts']) if td['pass_yards'] else league_avg['pass_yards']
        rush_yds = wavg(td['rush_yards'], td['rush_wts']) if td['rush_yards'] else league_avg['rush_yards']
        to = wavg(td['turnovers'], td['to_wts']) if td['turnovers'] else league_avg['turnovers']
        fd = wavg(td['first_downs'], td['fd_wts']) if td['first_downs'] else league_avg['first_downs']

        prev = prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)
        return {
            'ppg': blend * prev.get('ppg', league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', league_avg['papg']) + (1 - blend) * papg,
            'yards': yards,
            'pass_yards': pass_yds,
            'rush_yards': rush_yds,
            'turnovers': to,
            'first_downs': fd,
            'games': n,
            'margins': td['margins'],
            'wins': td['wins'],
        }

    def extract_spread_features(hid, aid, season, date, neutral=True):
        hs = get_stats(hid, season)
        aws = get_stats(aid, season)
        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None
        hr, ar = get_rest(hid, date), get_rest(aid, date)
        return np.array([
            hs['ppg'] - aws['ppg'],
            hs['papg'] - aws['papg'],
            hs['yards'] - aws['yards'],
            hs['pass_yards'] - aws['pass_yards'],
            hs['rush_yards'] - aws['rush_yards'],
            hs['turnovers'] - aws['turnovers'],
            hs['first_downs'] - aws['first_downs'],
            min(hr, 14) - min(ar, 14),
            0.0 if neutral else 1.0,  # Neutral site
            min(hs['games'] / 12.0, 1.0),
            min(aws['games'] / 12.0, 1.0),
        ])

    def extract_total_features(hid, aid, season):
        hs = get_stats(hid, season)
        aws = get_stats(aid, season)
        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None
        return np.array([
            hs['ppg'] + aws['ppg'],
            hs['papg'] + aws['papg'],
            hs['yards'] + aws['yards'],
            hs['turnovers'] + aws['turnovers'],
            min(hs['games'] / 12.0, 1.0),
            min(aws['games'] / 12.0, 1.0),
        ])

    def update(tid, season, date, pf, pa, yards=None, pass_yards=None,
               rush_yards=None, turnovers=None, first_downs=None):
        td = team_stats[tid][season]
        margin = pf - pa
        td['wts'] = [w * DECAY for w in td['wts']]
        td['ppg'].append(pf)
        td['papg'].append(pa)
        td['wts'].append(1.0)
        td['margins'].append(margin)
        td['wins'].append(1 if margin > 0 else 0)
        if pd.notna(yards):
            td['yards_wts'] = [w * DECAY for w in td['yards_wts']]
            td['yards'].append(yards)
            td['yards_wts'].append(1.0)
        if pd.notna(pass_yards):
            td['pass_wts'] = [w * DECAY for w in td['pass_wts']]
            td['pass_yards'].append(pass_yards)
            td['pass_wts'].append(1.0)
        if pd.notna(rush_yards):
            td['rush_wts'] = [w * DECAY for w in td['rush_wts']]
            td['rush_yards'].append(rush_yards)
            td['rush_wts'].append(1.0)
        if pd.notna(turnovers):
            td['to_wts'] = [w * DECAY for w in td['to_wts']]
            td['turnovers'].append(turnovers)
            td['to_wts'].append(1.0)
        if pd.notna(first_downs):
            td['fd_wts'] = [w * DECAY for w in td['fd_wts']]
            td['first_downs'].append(first_downs)
            td['fd_wts'].append(1.0)
        last_game[tid] = date

    def set_prev_season(season):
        nonlocal prev_ratings
        prev = season - 1
        for tid in team_stats:
            if prev in team_stats[tid]:
                td = team_stats[tid][prev]
                if td['ppg']:
                    prev_ratings[tid] = {
                        'ppg': np.mean(td['ppg']),
                        'papg': np.mean(td['papg']),
                        'yards': np.mean(td['yards']) if td['yards'] else 400.0,
                        'pass_yards': np.mean(td['pass_yards']) if td['pass_yards'] else 230.0,
                        'rush_yards': np.mean(td['rush_yards']) if td['rush_yards'] else 170.0,
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 1.5,
                        'first_downs': np.mean(td['first_downs']) if td['first_downs'] else 20.0,
                    }
        last_game.clear()

    def recent_form(margins, n=4):
        if len(margins) < n:
            return 0.0
        return float(np.mean(margins[-n:]))

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

    # Load games
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site,
               ht.name as home_team, at.name as away_team,
               hs.total_yards as home_yards, hs.passing_yards as home_pass_yards,
               hs.rushing_yards as home_rush_yards, hs.turnovers as home_to, hs.first_downs as home_fd,
               aws.total_yards as away_yards, aws.passing_yards as away_pass_yards,
               aws.rushing_yards as away_rush_yards, aws.turnovers as away_to, aws.first_downs as away_fd
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        WHERE g.completed = 1 AND g.home_score IS NOT NULL
        ORDER BY g.date
    ''', conn)

    teams = pd.read_sql_query('SELECT team_id, name FROM teams', conn)
    conn.close()

    # Process all games
    for season in sorted(games['season'].unique()):
        if season > games['season'].min():
            set_prev_season(season)

        for _, g in games[games['season'] == season].iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_spread = g['away_score'] - g['home_score']
            actual_total = g['home_score'] + g['away_score']
            neutral = g['neutral_site'] == 1

            sfeat = extract_spread_features(hid, aid, season, g['date'], neutral=neutral)
            if sfeat is not None:
                spread_X.append(sfeat)
                spread_y.append(actual_spread)

            tfeat = extract_total_features(hid, aid, season)
            if tfeat is not None:
                total_X.append(tfeat)
                total_y.append(actual_total)

            if len(spread_X) >= 100 and len(spread_X) % 100 == 0:
                X = np.array(spread_X)
                y = np.array(spread_y)
                spread_scaler.fit(X)
                spread_model = Ridge(alpha=1.0).fit(spread_scaler.transform(X), y)

            if len(total_X) >= 100 and len(total_X) % 100 == 0:
                X = np.array(total_X)
                y = np.array(total_y)
                total_scaler.fit(X)
                total_model = Ridge(alpha=1.0).fit(total_scaler.transform(X), y)

            update(hid, season, g['date'], g['home_score'], g['away_score'],
                   yards=g['home_yards'], pass_yards=g['home_pass_yards'],
                   rush_yards=g['home_rush_yards'], turnovers=g['home_to'],
                   first_downs=g['home_fd'])
            update(aid, season, g['date'], g['away_score'], g['home_score'],
                   yards=g['away_yards'], pass_yards=g['away_pass_yards'],
                   rush_yards=g['away_rush_yards'], turnovers=g['away_to'],
                   first_downs=g['away_fd'])

    # Final training
    X = np.array(spread_X)
    y = np.array(spread_y)
    spread_scaler.fit(X)
    spread_model = Ridge(alpha=1.0).fit(spread_scaler.transform(X), y)

    X = np.array(total_X)
    y = np.array(total_y)
    total_scaler.fit(X)
    total_model = Ridge(alpha=1.0).fit(total_scaler.transform(X), y)

    def find_team(name):
        name_lower = name.lower().strip()
        for _, t in teams.iterrows():
            t_lower = t['name'].lower()
            # Exact or partial match
            if name_lower == t_lower or name_lower in t_lower or t_lower in name_lower:
                return t['team_id'], t['name']
            # Handle common variations
            if 'ole miss' in name_lower and 'rebels' in t_lower:
                return t['team_id'], t['name']
            if 'miami' in name_lower and 'hurricanes' in t_lower:
                return t['team_id'], t['name']
        return None, None

    def predict_game(away_name, home_name, vegas_spread, vegas_total, game_date):
        aid, away_full = find_team(away_name)
        hid, home_full = find_team(home_name)

        if aid is None or hid is None:
            return None, f"Teams not found: {away_name} ({aid}), {home_name} ({hid})"

        hs = get_stats(hid, 2025)
        aws = get_stats(aid, 2025)

        sfeat = extract_spread_features(hid, aid, 2025, game_date, neutral=True)
        tfeat = extract_total_features(hid, aid, 2025)

        if sfeat is None or tfeat is None:
            return None, "Not enough data"

        pred_spread = spread_model.predict(spread_scaler.transform(sfeat.reshape(1, -1)))[0]
        pred_total = total_model.predict(total_scaler.transform(tfeat.reshape(1, -1)))[0]

        spread_edge = pred_spread - vegas_spread
        total_edge = pred_total - vegas_total

        return {
            'away': away_full,
            'home': home_full,
            'vegas_spread': vegas_spread,
            'model_spread': pred_spread,
            'spread_edge': spread_edge,
            'vegas_total': vegas_total,
            'model_total': pred_total,
            'total_edge': total_edge,
            'away_ppg': aws['ppg'],
            'away_papg': aws['papg'],
            'away_yards': aws['yards'],
            'home_ppg': hs['ppg'],
            'home_papg': hs['papg'],
            'home_yards': hs['yards'],
            'away_form': recent_form(aws['margins']),
            'home_form': recent_form(hs['margins']),
            'away_streak': streak(aws['wins']),
            'home_streak': streak(hs['wins']),
            'away_games': aws['games'],
            'home_games': hs['games'],
        }, None

    print('=' * 90)
    print('CFB PLAYOFF PREDICTIONS - 2025 Season')
    print('=' * 90)
    print()
    print('Note: CFB model was 1-2 (33%) on last year\'s final rounds - small sample caveat')
    print()

    # Semifinal games (neutral sites)
    # Spread convention: positive = away favored, negative = home favored
    # Miami -3.5 vs Ole Miss means Miami favored -> in neutral, we'll use "home" as the favored team
    # For neutral site, I'll put the favorite as "home" to match convention
    games_to_predict = [
        # Fiesta Bowl: Ole Miss vs Miami (Miami -3.5, O/U 51.5)
        ('Ole Miss', 'Miami', -3.5, 51.5, '2026-01-08', 'THU Fiesta Bowl'),
        # Peach Bowl: Oregon vs Indiana (Indiana -4, O/U ~51)
        ('Ducks', 'Hoosiers', -4.0, 51.0, '2026-01-09', 'FRI Peach Bowl'),
    ]

    for away, home, vegas_spread, vegas_total, date, label in games_to_predict:
        p, err = predict_game(away, home, vegas_spread, vegas_total, date)

        print(f"{label}: {away} vs {home}")
        print("-" * 60)

        if err:
            print(f"  Error: {err}")
            print()
            continue

        # Spread analysis
        spread_dir = "FADE VEGAS" if p['spread_edge'] > 0 else "WITH VEGAS"
        spread_conf = "HIGH" if abs(p['spread_edge']) >= 4 else ("MEDIUM" if abs(p['spread_edge']) >= 2 else "LOW")

        print(f"  SPREAD:")
        print(f"    Vegas: {home} {p['vegas_spread']:.1f}")
        print(f"    Model: {p['model_spread']:+.1f}")
        print(f"    Edge: {p['spread_edge']:+.1f} pts -> {spread_dir} ({spread_conf})")

        # Total analysis
        total_dir = "OVER" if p['total_edge'] > 0 else "UNDER"
        total_conf = "HIGH" if abs(p['total_edge']) >= 4 else ("MEDIUM" if abs(p['total_edge']) >= 2 else "LOW")

        print(f"  TOTAL:")
        print(f"    Vegas O/U: {p['vegas_total']:.1f}")
        print(f"    Model: {p['model_total']:.1f}")
        print(f"    Edge: {p['total_edge']:+.1f} pts -> {total_dir} ({total_conf})")

        print(f"  TEAM STATS:")
        print(f"    {p['away']}: {p['away_ppg']:.1f} PPG, {p['away_papg']:.1f} PAPG, {p['away_yards']:.0f} YPG")
        print(f"      Form: {p['away_form']:+.1f}, Streak: {p['away_streak']:+d}")
        print(f"    {p['home']}: {p['home_ppg']:.1f} PPG, {p['home_papg']:.1f} PAPG, {p['home_yards']:.0f} YPG")
        print(f"      Form: {p['home_form']:+.1f}, Streak: {p['home_streak']:+d}")
        print()

    print('=' * 90)
    print('SUMMARY')
    print('=' * 90)


if __name__ == '__main__':
    main()
