"""Generate totals predictions for NFL playoff games."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).parent / 'nfl_games.db'
DECAY = 0.96
PREV_HALF_LIFE = 4.0
MIN_GAMES = 2


def main():
    # Build model state
    team_stats = defaultdict(lambda: defaultdict(lambda: {
        'ppg': [], 'papg': [], 'wts': [],
        'yards': [], 'yards_wts': [],
    }))
    prev_ratings = {}

    total_model = None
    total_scaler = StandardScaler()
    total_X, total_y = [], []

    def wavg(vals, wts):
        if not vals or not wts:
            return None
        return float(np.average(vals, weights=wts))

    def get_stats(tid, season):
        td = team_stats[tid][season]
        n = len(td['ppg'])
        if n == 0:
            prev = prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', 22.0),
                'papg': prev.get('papg', 22.0),
                'yards': prev.get('yards', 330.0),
                'games': 0
            }

        ppg = wavg(td['ppg'], td['wts'])
        papg = wavg(td['papg'], td['wts'])
        yards = wavg(td['yards'], td['yards_wts']) if td['yards'] else 330.0

        prev = prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)
        return {
            'ppg': blend * prev.get('ppg', 22.0) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', 22.0) + (1 - blend) * papg,
            'yards': yards,
            'games': n
        }

    def extract_total_features(hid, aid, season):
        hs = get_stats(hid, season)
        aws = get_stats(aid, season)
        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None
        return np.array([
            hs['ppg'] + aws['ppg'],
            hs['papg'] + aws['papg'],
            hs['yards'] + aws['yards'],
            min(hs['games'] / 10, 1),
            min(aws['games'] / 10, 1)
        ])

    def update(tid, season, date, pf, pa, yards=None):
        td = team_stats[tid][season]
        td['wts'] = [w * DECAY for w in td['wts']]
        td['ppg'].append(pf)
        td['papg'].append(pa)
        td['wts'].append(1.0)
        if pd.notna(yards):
            td['yards_wts'] = [w * DECAY for w in td['yards_wts']]
            td['yards'].append(yards)
            td['yards_wts'].append(1.0)

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
                        'yards': np.mean(td['yards']) if td['yards'] else 330.0
                    }

    # Load games
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, ht.name as home_team, at.name as away_team,
               hs.total_yards as home_yards, aws.total_yards as away_yards
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
            actual_total = g['home_score'] + g['away_score']

            feat = extract_total_features(hid, aid, season)
            if feat is not None:
                total_X.append(feat)
                total_y.append(actual_total)

            if len(total_X) >= 50 and len(total_X) % 50 == 0:
                X = np.array(total_X)
                y = np.array(total_y)
                total_scaler.fit(X)
                total_model = Ridge(alpha=1.0).fit(total_scaler.transform(X), y)

            update(hid, season, g['date'], g['home_score'], g['away_score'], g['home_yards'])
            update(aid, season, g['date'], g['away_score'], g['home_score'], g['away_yards'])

    # Final training
    X = np.array(total_X)
    y = np.array(total_y)
    total_scaler.fit(X)
    total_model = Ridge(alpha=1.0).fit(total_scaler.transform(X), y)

    def find_team(name):
        name_lower = name.lower()
        for _, t in teams.iterrows():
            if name_lower in t['name'].lower() or t['name'].lower() in name_lower:
                return t['team_id']
        return None

    def predict_total(away, home, vegas_total):
        aid = find_team(away)
        hid = find_team(home)
        if aid is None or hid is None:
            return None

        hs = get_stats(hid, 2025)
        aws = get_stats(aid, 2025)

        feat = extract_total_features(hid, aid, 2025)
        if feat is None:
            return None

        pred = total_model.predict(total_scaler.transform(feat.reshape(1, -1)))[0]
        edge = pred - vegas_total

        return {
            'away': away, 'home': home,
            'vegas': vegas_total, 'model': pred, 'edge': edge,
            'away_ppg': aws['ppg'], 'away_papg': aws['papg'],
            'home_ppg': hs['ppg'], 'home_papg': hs['papg'],
            'combined_ppg': hs['ppg'] + aws['ppg'],
            'combined_papg': hs['papg'] + aws['papg'],
        }

    # This weekend's games
    games_to_predict = [
        ('Rams', 'Panthers', 46.5, 'SAT 4:30 PM'),
        ('Packers', 'Bears', 45.5, 'SAT 8:00 PM'),
        ('Bills', 'Jaguars', 51.5, 'SUN 1:00 PM'),
        ('49ers', 'Eagles', 45.5, 'SUN 4:30 PM'),
        ('Chargers', 'Patriots', 46.5, 'SUN 8:00 PM'),
        ('Texans', 'Steelers', 39.5, 'MON 8:15 PM'),
    ]

    print('=' * 90)
    print('NFL WILD CARD - TOTALS PREDICTIONS')
    print('=' * 90)
    print()
    print('Note: Model was 5-7 (41.7%) on playoff totals last year - use with caution!')
    print()

    results = []
    for away, home, vegas, time in games_to_predict:
        p = predict_total(away, home, vegas)
        if p:
            results.append((time, p))

    # Sort by edge
    results.sort(key=lambda x: abs(x[1]['edge']), reverse=True)

    for time, p in results:
        direction = 'OVER' if p['edge'] > 0 else 'UNDER'
        conf = 'HIGH' if abs(p['edge']) >= 4 else ('MEDIUM' if abs(p['edge']) >= 2 else 'LOW')
        print(f"{time}: {p['away']} @ {p['home']}")
        print(f"  Vegas O/U: {p['vegas']:.1f} | Model: {p['model']:.1f} | Edge: {p['edge']:+.1f} ({direction})")
        print(f"  {p['away']}: {p['away_ppg']:.1f} PPG, {p['away_papg']:.1f} PAPG")
        print(f"  {p['home']}: {p['home_ppg']:.1f} PPG, {p['home_papg']:.1f} PAPG")
        print(f"  Combined: {p['combined_ppg']:.1f} PPG scored, {p['combined_papg']:.1f} PPG allowed")
        print(f"  Confidence: {conf}")
        print()

    print('=' * 90)
    print('TOTALS LEANS (Edge >= 2.0 pts)')
    print('=' * 90)
    print()
    has_leans = False
    for time, p in results:
        direction = 'OVER' if p['edge'] > 0 else 'UNDER'
        if abs(p['edge']) >= 2:
            has_leans = True
            print(f"  {time}: {direction} {p['vegas']:.1f} ({p['away']}@{p['home']}) - Edge: {abs(p['edge']):.1f}")

    if not has_leans:
        print("  No significant edges found on totals.")
    print()


if __name__ == '__main__':
    main()
