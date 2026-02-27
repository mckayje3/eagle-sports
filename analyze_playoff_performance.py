"""Analyze model performance on NFL and CFB playoffs."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_DIR = Path(__file__).parent


def analyze_nfl_playoffs():
    """Analyze NFL 2024 playoff performance."""
    print("=" * 95)
    print("NFL 2024 PLAYOFFS - MODEL PERFORMANCE")
    print("=" * 95)

    # NFL Simple model constants (optimized)
    DECAY = 0.96
    PREV_HALF_LIFE = 4.0
    MIN_GAMES = 2

    team_stats = defaultdict(lambda: defaultdict(lambda: {
        'ppg': [], 'papg': [], 'wts': [],
        'yards': [], 'yards_wts': [],
        'pass_yards': [], 'pass_wts': [],
        'rush_yards': [], 'rush_wts': [],
        'turnovers': [], 'to_wts': [],
        'first_downs': [], 'fd_wts': [],
    }))
    prev_ratings = {}
    last_game = {}
    league_avg = {
        'ppg': 22.0, 'papg': 22.0, 'yards': 330.0,
        'pass_yards': 220.0, 'rush_yards': 110.0,
        'turnovers': 1.3, 'first_downs': 20.0
    }
    spread_model = None
    spread_scaler = StandardScaler()
    spread_X, spread_y = [], []

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
            return min((curr - last).days, 14)
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
                'games': 0
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
            'games': n
        }

    def extract_features(hid, aid, season, date):
        hs = get_stats(hid, season)
        aws = get_stats(aid, season)
        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None
        hr, ar = get_rest(hid, date), get_rest(aid, date)
        return np.array([
            hs['yards'] - aws['yards'],
            hs['pass_yards'] - aws['pass_yards'],
            hs['rush_yards'] - aws['rush_yards'],
            hs['turnovers'] - aws['turnovers'],
            hs['first_downs'] - aws['first_downs'],
            min(hr, 10) - min(ar, 10),
            1.0 if hr >= 13 else 0.0,
            1.0 if ar >= 13 else 0.0,
            min(hs['games'] / 10.0, 1.0),
        ])

    def update_team(tid, season, date, pf, pa, yards=None, pass_yards=None,
                    rush_yards=None, turnovers=None, first_downs=None):
        td = team_stats[tid][season]
        td['wts'] = [w * DECAY for w in td['wts']]
        td['ppg'].append(pf)
        td['papg'].append(pa)
        td['wts'].append(1.0)
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

    def set_previous_season(season):
        nonlocal prev_ratings, last_game
        prev = season - 1
        for tid in team_stats:
            if prev in team_stats[tid]:
                td = team_stats[tid][prev]
                if td['ppg']:
                    prev_ratings[tid] = {
                        'ppg': np.mean(td['ppg']),
                        'papg': np.mean(td['papg']),
                        'yards': np.mean(td['yards']) if td['yards'] else 330.0,
                        'pass_yards': np.mean(td['pass_yards']) if td['pass_yards'] else 220.0,
                        'rush_yards': np.mean(td['rush_yards']) if td['rush_yards'] else 110.0,
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 1.3,
                        'first_downs': np.mean(td['first_downs']) if td['first_downs'] else 20.0,
                    }
        last_game = {}

    # Load all games
    conn = sqlite3.connect(str(DB_DIR / 'nfl_games.db'))
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site,
               ht.name as home_team, at.name as away_team,
               hs.total_yards as home_yards, hs.passing_yards as home_pass_yards,
               hs.rushing_yards as home_rush_yards, hs.turnovers as home_to, hs.first_downs as home_fd,
               aws.total_yards as away_yards, aws.passing_yards as away_pass_yards,
               aws.rushing_yards as away_rush_yards, aws.turnovers as away_to, aws.first_downs as away_fd,
               o.latest_spread as vegas_spread
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score IS NOT NULL
        ORDER BY g.date
    ''', conn)
    conn.close()

    results = []
    seasons = sorted(games['season'].unique())

    for season in seasons:
        if season > seasons[0]:
            set_previous_season(season)

        season_games = games[games['season'] == season]
        for _, g in season_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_spread = g['away_score'] - g['home_score']

            # Get prediction before game
            feat = extract_features(hid, aid, season, g['date'])
            if feat is not None:
                spread_X.append(feat)
                spread_y.append(actual_spread)

            # Train periodically
            if len(spread_X) >= 50 and len(spread_X) % 50 == 0:
                X = np.array(spread_X)
                y = np.array(spread_y)
                spread_scaler.fit(X)
                spread_model = Ridge(alpha=1.0).fit(spread_scaler.transform(X), y)

            # Predict
            pred = None
            if feat is not None and spread_model is not None:
                X = spread_scaler.transform(feat.reshape(1, -1))
                pred = spread_model.predict(X)[0]

            # Store playoff results
            if season == 2024 and g['week'] >= 19:
                results.append({
                    'date': g['date'][:10],
                    'matchup': f"{g['away_team']} @ {g['home_team']}",
                    'score': f"{int(g['away_score'])}-{int(g['home_score'])}",
                    'actual_spread': actual_spread,
                    'vegas': g['vegas_spread'],
                    'model_pred': pred,
                    'week': g['week']
                })

            # Update
            update_team(hid, season, g['date'], g['home_score'], g['away_score'],
                        yards=g['home_yards'], pass_yards=g['home_pass_yards'],
                        rush_yards=g['home_rush_yards'], turnovers=g['home_to'],
                        first_downs=g['home_fd'])
            update_team(aid, season, g['date'], g['away_score'], g['home_score'],
                        yards=g['away_yards'], pass_yards=g['away_pass_yards'],
                        rush_yards=g['away_rush_yards'], turnovers=g['away_to'],
                        first_downs=g['away_fd'])

    # Print results
    wins, losses, pushes = 0, 0, 0
    print()
    for r in results:
        if r['model_pred'] is not None and pd.notna(r['vegas']):
            edge = r['model_pred'] - r['vegas']
            result = r['actual_spread'] - r['vegas']
            if abs(result) > 0.5:
                cover = edge * result > 0
                status = 'WIN' if cover else 'LOSS'
                if cover:
                    wins += 1
                else:
                    losses += 1
            else:
                status = 'PUSH'
                pushes += 1
            print(f"{r['date']} Wk{r['week']:>2}: {r['matchup']:<35} {r['score']:>7}")
            print(f"    Vegas: {r['vegas']:+.1f}, Model: {r['model_pred']:+.1f}, Edge: {edge:+.1f} -> {status}")
        else:
            print(f"{r['date']} Wk{r['week']:>2}: {r['matchup']:<35} {r['score']:>7} (no prediction)")

    total = wins + losses
    if total > 0:
        print()
        print(f"RECORD: {wins}-{losses} ({wins/total*100:.1f}% ATS)")
        roi = (wins - losses * 1.1) / total * 100
        print(f"ROI: {roi:+.1f}%")

    return results


def analyze_cfb_playoffs():
    """Analyze CFB 2024 playoff performance (last 2 rounds)."""
    print()
    print("=" * 95)
    print("CFB 2024 PLAYOFFS (Semifinals + Championship) - MODEL PERFORMANCE")
    print("=" * 95)

    # CFB Simple model constants (optimized)
    DECAY = 0.88
    PREV_HALF_LIFE = 5.0
    MIN_GAMES = 2

    team_stats = defaultdict(lambda: defaultdict(lambda: {
        'ppg': [], 'papg': [], 'wts': [],
        'yards': [], 'yards_wts': [],
        'pass_yards': [], 'pass_wts': [],
        'rush_yards': [], 'rush_wts': [],
        'turnovers': [], 'to_wts': [],
        'first_downs': [], 'fd_wts': [],
        'third_down_pct': [], 'td_wts': [],
    }))
    prev_ratings = {}
    last_game = {}
    league_avg = {
        'ppg': 28.0, 'papg': 28.0, 'yards': 400.0,
        'pass_yards': 230.0, 'rush_yards': 170.0,
        'turnovers': 1.5, 'first_downs': 20.0, 'third_down_pct': 40.0
    }
    spread_model = None
    spread_scaler = StandardScaler()
    spread_X, spread_y = [], []

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
            return min((curr - last).days, 14)
        except Exception:
            return 7

    def get_stats(tid, season):
        td = team_stats[tid][season]
        n = len(td['ppg'])
        if n == 0:
            prev = prev_ratings.get(tid, {})
            return {k: prev.get(k, league_avg.get(k, 0)) for k in league_avg} | {'games': 0}

        ppg = wavg(td['ppg'], td['wts'])
        papg = wavg(td['papg'], td['wts'])
        yards = wavg(td['yards'], td['yards_wts']) if td['yards'] else league_avg['yards']
        pass_yds = wavg(td['pass_yards'], td['pass_wts']) if td['pass_yards'] else league_avg['pass_yards']
        rush_yds = wavg(td['rush_yards'], td['rush_wts']) if td['rush_yards'] else league_avg['rush_yards']
        to = wavg(td['turnovers'], td['to_wts']) if td['turnovers'] else league_avg['turnovers']
        fd = wavg(td['first_downs'], td['fd_wts']) if td['first_downs'] else league_avg['first_downs']
        td_pct = wavg(td['third_down_pct'], td['td_wts']) if td['third_down_pct'] else league_avg['third_down_pct']

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
            'third_down_pct': td_pct,
            'games': n
        }

    def extract_features(hid, aid, season, date, neutral_site=False):
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
            hs['third_down_pct'] - aws['third_down_pct'],
            min(hr, 10) - min(ar, 10),
            0.0 if neutral_site else 1.0,
            min(hs['games'] / 10.0, 1.0),
            min(aws['games'] / 10.0, 1.0),
        ])

    def update_team(tid, season, date, pf, pa, yards=None, pass_yards=None,
                    rush_yards=None, turnovers=None, first_downs=None, third_down_pct=None):
        td = team_stats[tid][season]
        td['wts'] = [w * DECAY for w in td['wts']]
        td['ppg'].append(pf)
        td['papg'].append(pa)
        td['wts'].append(1.0)
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
        if pd.notna(third_down_pct):
            td['td_wts'] = [w * DECAY for w in td['td_wts']]
            td['third_down_pct'].append(third_down_pct)
            td['td_wts'].append(1.0)
        last_game[tid] = date

    def set_previous_season(season):
        nonlocal prev_ratings, last_game
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
                        'third_down_pct': np.mean(td['third_down_pct']) if td['third_down_pct'] else 40.0,
                    }
        last_game = {}

    # Load all games
    conn = sqlite3.connect(str(DB_DIR / 'cfb_games.db'))
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site, g.postseason_type,
               ht.name as home_team, at.name as away_team,
               hs.total_yards as home_yards, hs.passing_yards as home_pass_yards,
               hs.rushing_yards as home_rush_yards, hs.turnovers as home_to, hs.first_downs as home_fd,
               hs.third_down_conversions as home_3dc, hs.third_down_attempts as home_3da,
               aws.total_yards as away_yards, aws.passing_yards as away_pass_yards,
               aws.rushing_yards as away_rush_yards, aws.turnovers as away_to, aws.first_downs as away_fd,
               aws.third_down_conversions as away_3dc, aws.third_down_attempts as away_3da,
               o.latest_spread as vegas_spread
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score IS NOT NULL
        ORDER BY g.date
    ''', conn)
    conn.close()

    # Calculate 3rd down pct
    games['home_3d_pct'] = games.apply(
        lambda r: 100 * r['home_3dc'] / r['home_3da']
        if pd.notna(r['home_3dc']) and r['home_3da'] > 0 else None, axis=1
    )
    games['away_3d_pct'] = games.apply(
        lambda r: 100 * r['away_3dc'] / r['away_3da']
        if pd.notna(r['away_3dc']) and r['away_3da'] > 0 else None, axis=1
    )

    results = []
    seasons = sorted(games['season'].unique())

    for season in seasons:
        if season > seasons[0]:
            set_previous_season(season)

        season_games = games[games['season'] == season]
        for _, g in season_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_spread = g['away_score'] - g['home_score']
            neutral = g['neutral_site'] == 1

            # Get prediction before game
            feat = extract_features(hid, aid, season, g['date'], neutral_site=neutral)
            if feat is not None:
                spread_X.append(feat)
                spread_y.append(actual_spread)

            # Train periodically
            if len(spread_X) >= 100 and len(spread_X) % 100 == 0:
                X = np.array(spread_X)
                y = np.array(spread_y)
                spread_scaler.fit(X)
                spread_model = Ridge(alpha=1.0).fit(spread_scaler.transform(X), y)

            # Predict
            pred = None
            if feat is not None and spread_model is not None:
                X = spread_scaler.transform(feat.reshape(1, -1))
                pred = spread_model.predict(X)[0]

            # Store playoff results (Semifinals and Championship)
            if season == 2024 and g['postseason_type'] in ['Semifinals', 'National Championship']:
                results.append({
                    'date': g['date'][:10],
                    'round': g['postseason_type'],
                    'matchup': f"{g['away_team']} vs {g['home_team']}",
                    'score': f"{int(g['away_score'])}-{int(g['home_score'])}",
                    'actual_spread': actual_spread,
                    'vegas': g['vegas_spread'],
                    'model_pred': pred,
                })

            # Update
            update_team(hid, season, g['date'], g['home_score'], g['away_score'],
                        yards=g['home_yards'], pass_yards=g['home_pass_yards'],
                        rush_yards=g['home_rush_yards'], turnovers=g['home_to'],
                        first_downs=g['home_fd'], third_down_pct=g['home_3d_pct'])
            update_team(aid, season, g['date'], g['away_score'], g['home_score'],
                        yards=g['away_yards'], pass_yards=g['away_pass_yards'],
                        rush_yards=g['away_rush_yards'], turnovers=g['away_to'],
                        first_downs=g['away_fd'], third_down_pct=g['away_3d_pct'])

    # Print results
    wins, losses = 0, 0
    print()
    for r in results:
        if r['model_pred'] is not None and pd.notna(r['vegas']):
            edge = r['model_pred'] - r['vegas']
            result = r['actual_spread'] - r['vegas']
            if abs(result) > 0.5:
                cover = edge * result > 0
                status = 'WIN' if cover else 'LOSS'
                if cover:
                    wins += 1
                else:
                    losses += 1
            else:
                status = 'PUSH'
            print(f"{r['date']} ({r['round']})")
            print(f"  {r['matchup']}: {r['score']}")
            print(f"  Vegas: {r['vegas']:+.1f}, Model: {r['model_pred']:+.1f}, Edge: {edge:+.1f} -> {status}")
        else:
            print(f"{r['date']} ({r['round']}): {r['matchup']} {r['score']} (no prediction)")
        print()

    total = wins + losses
    if total > 0:
        print(f"RECORD: {wins}-{losses} ({wins/total*100:.1f}% ATS)")
        roi = (wins - losses * 1.1) / total * 100
        print(f"ROI: {roi:+.1f}%")

    return results


if __name__ == '__main__':
    analyze_nfl_playoffs()
    analyze_cfb_playoffs()
