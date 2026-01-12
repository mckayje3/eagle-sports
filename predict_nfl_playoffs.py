"""Generate predictions for upcoming NFL playoff games."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).parent / 'nfl_games.db'

# Current playoff games - update each round as matchups become known
# Format: (away_team, home_team, vegas_spread, vegas_total, game_date, time_slot)
# Convention: spread = away_score - home_score (positive = away favored)
PLAYOFF_GAMES = [
    # 2025 Season Wild Card Weekend - Jan 10-13, 2026
    # Lines as of Jan 11, 2026 (source: CBS Sports)
    ('Rams', 'Panthers', +10.0, 46.5, '2026-01-10', 'FRI 8:00 PM'),      # Rams (away) favored by 10
    ('Bills', 'Jaguars', +1.5, 52.5, '2026-01-11', 'SAT 1:00 PM'),       # Bills (away) favored by 1.5
    ('Packers', 'Bears', +1.5, 45.5, '2026-01-11', 'SAT 4:30 PM'),       # Packers (away) favored by 1.5
    ('49ers', 'Eagles', -4.5, 44.5, '2026-01-11', 'SAT 8:00 PM'),        # Eagles (home) favored by 4.5
    ('Chargers', 'Patriots', -3.5, 46.5, '2026-01-12', 'SUN 1:00 PM'),   # Patriots (home) favored by 3.5
    ('Texans', 'Steelers', +3.0, 39.5, '2026-01-13', 'MON 8:00 PM'),     # Texans (away) favored by 3
]

# NFL Simple model constants (optimized)
DECAY = 0.96
PREV_HALF_LIFE = 4.0
MIN_GAMES = 2


def build_model_state():
    """Build model state from all historical games."""
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

    def set_previous_season(season):
        nonlocal prev_ratings
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
        last_game.clear()

    # Load all completed games
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

    # Get team name to ID mapping
    teams = pd.read_sql_query('SELECT team_id, name, abbreviation FROM teams', conn)
    conn.close()

    team_lookup = {}
    for _, t in teams.iterrows():
        team_lookup[t['name'].lower()] = t['team_id']
        team_lookup[t['abbreviation'].lower()] = t['team_id']
        # Common variations
        if t['name'] == '49ers':
            team_lookup['san francisco 49ers'] = t['team_id']
            team_lookup['sf'] = t['team_id']
            team_lookup['san francisco'] = t['team_id']

    # Process all games
    seasons = sorted(games['season'].unique())
    for season in seasons:
        if season > seasons[0]:
            set_previous_season(season)

        season_games = games[games['season'] == season]
        for _, g in season_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_spread = g['away_score'] - g['home_score']

            feat = extract_features(hid, aid, season, g['date'])
            if feat is not None:
                spread_X.append(feat)
                spread_y.append(actual_spread)

            if len(spread_X) >= 50 and len(spread_X) % 50 == 0:
                X = np.array(spread_X)
                y = np.array(spread_y)
                spread_scaler.fit(X)
                spread_model = Ridge(alpha=1.0).fit(spread_scaler.transform(X), y)

            update_team(hid, season, g['date'], g['home_score'], g['away_score'],
                        yards=g['home_yards'], pass_yards=g['home_pass_yards'],
                        rush_yards=g['home_rush_yards'], turnovers=g['home_to'],
                        first_downs=g['home_fd'])
            update_team(aid, season, g['date'], g['away_score'], g['home_score'],
                        yards=g['away_yards'], pass_yards=g['away_pass_yards'],
                        rush_yards=g['away_rush_yards'], turnovers=g['away_to'],
                        first_downs=g['away_fd'])

    # Final training
    X = np.array(spread_X)
    y = np.array(spread_y)
    spread_scaler.fit(X)
    spread_model = Ridge(alpha=1.0).fit(spread_scaler.transform(X), y)

    return {
        'team_stats': team_stats,
        'prev_ratings': prev_ratings,
        'last_game': last_game,
        'spread_model': spread_model,
        'spread_scaler': spread_scaler,
        'get_stats': get_stats,
        'extract_features': extract_features,
        'get_rest': get_rest,
        'team_lookup': team_lookup,
        'teams': teams,
    }


def recent_form(margins, n=4):
    """Average margin over last n games."""
    if len(margins) < n:
        return 0.0
    return float(np.mean(margins[-n:]))


def streak(wins):
    """Current win/loss streak."""
    if not wins:
        return 0
    s, last = 0, wins[-1]
    for w in reversed(wins):
        if w == last:
            s += 1
        else:
            break
    return s if last == 1 else -s


def predict_game(state, away_team, home_team, vegas_spread, game_date='2026-01-11'):
    """Predict a single game."""
    # Find team IDs
    away_key = away_team.lower().replace(' ', '')
    home_key = home_team.lower().replace(' ', '')

    # Try various lookups
    aid = None
    hid = None

    for _, t in state['teams'].iterrows():
        name_lower = t['name'].lower()
        abbr_lower = t['abbreviation'].lower()
        if away_key in name_lower or name_lower in away_key or abbr_lower == away_key[:3]:
            aid = t['team_id']
        if home_key in name_lower or name_lower in home_key or abbr_lower == home_key[:3]:
            hid = t['team_id']

    if aid is None or hid is None:
        return None, f"Could not find teams: {away_team} ({aid}) vs {home_team} ({hid})"

    season = 2025
    get_stats = state['get_stats']
    extract_features = state['extract_features']

    hs = get_stats(hid, season)
    aws = get_stats(aid, season)

    feat = extract_features(hid, aid, season, game_date)
    if feat is None:
        return None, "Not enough games to predict"

    X = state['spread_scaler'].transform(feat.reshape(1, -1))
    pred_spread = state['spread_model'].predict(X)[0]

    edge = pred_spread - vegas_spread

    # Get recent form
    h_form = recent_form(hs['margins'])
    a_form = recent_form(aws['margins'])
    h_streak = streak(hs['wins'])
    a_streak = streak(aws['wins'])

    return {
        'away_team': away_team,
        'home_team': home_team,
        'vegas_spread': vegas_spread,
        'model_spread': pred_spread,
        'edge': edge,
        'home_ppg': hs['ppg'],
        'home_papg': hs['papg'],
        'away_ppg': aws['ppg'],
        'away_papg': aws['papg'],
        'home_yards': hs['yards'],
        'away_yards': aws['yards'],
        'home_form': h_form,
        'away_form': a_form,
        'home_streak': h_streak,
        'away_streak': a_streak,
        'home_games': hs['games'],
        'away_games': aws['games'],
    }, None


def main():
    print("=" * 90)
    print("NFL WILD CARD PREDICTIONS - 2025 Season Playoffs")
    print("=" * 90)
    print()

    # Build model state
    print("Building model from historical data...")
    state = build_model_state()
    print(f"Model trained on {len(state['team_stats'])} teams")
    print()

    # Use the PLAYOFF_GAMES constant
    games = PLAYOFF_GAMES

    results = []
    csv_rows = []
    for away, home, vegas, vegas_total, date, time in games:
        pred, err = predict_game(state, away, home, vegas, date)
        if err:
            print(f"Error: {err}")
            continue
        results.append((time, pred, vegas_total))

        # Build CSV row
        pred_total = pred['home_ppg'] + pred['away_ppg']
        csv_rows.append({
            'game_id': 0,
            'date': date,
            'time_slot': time,
            'week': 'WC',
            'home_team': home,
            'away_team': away,
            'pred_home_score': round((pred_total - pred['model_spread']) / 2, 1),
            'pred_away_score': round((pred_total + pred['model_spread']) / 2, 1),
            'pred_spread': round(pred['model_spread'], 1),
            'pred_total': round(pred_total, 1),
            'vegas_spread': vegas,
            'vegas_total': vegas_total,
            'edge': round(pred['edge'], 1),
            'is_playoff': True,
            'home_ppg': pred['home_ppg'],
            'away_ppg': pred['away_ppg'],
            'home_form': pred['home_form'],
            'away_form': pred['away_form'],
        })

    # Write CSV
    csv_df = pd.DataFrame(csv_rows)
    csv_df = csv_df.sort_values('edge', key=abs, ascending=False)
    csv_df.to_csv('nfl_playoff_predictions.csv', index=False)
    print(f"Saved {len(csv_rows)} predictions to nfl_playoff_predictions.csv")
    print()

    # Sort by absolute edge
    results.sort(key=lambda x: abs(x[1]['edge']), reverse=True)

    print("=" * 90)
    print("PREDICTIONS (Sorted by Edge Size)")
    print("=" * 90)
    print()

    for time, p, v_total in results:
        edge_dir = "FADE VEGAS" if p['edge'] > 0 else "WITH VEGAS"
        confidence = "HIGH" if abs(p['edge']) >= 4 else ("MEDIUM" if abs(p['edge']) >= 2 else "LOW")
        pred_total = p['home_ppg'] + p['away_ppg']
        total_edge = pred_total - v_total

        print(f"{time}: {p['away_team']} @ {p['home_team']}")
        print(f"  Spread: Vegas {p['vegas_spread']:+.1f} | Model {p['model_spread']:+.1f} | Edge {p['edge']:+.1f}")
        print(f"  Total:  Vegas {v_total:.1f} | Model {pred_total:.1f} | Edge {total_edge:+.1f} ({'OVER' if total_edge > 0 else 'UNDER'})")
        print(f"  Recommendation: {edge_dir} ({confidence} confidence)")
        print(f"  {p['away_team']}: {p['away_ppg']:.1f} PPG, {p['away_papg']:.1f} PAPG, {p['away_yards']:.0f} YPG, Form: {p['away_form']:+.1f}, Streak: {p['away_streak']:+d}")
        print(f"  {p['home_team']}: {p['home_ppg']:.1f} PPG, {p['home_papg']:.1f} PAPG, {p['home_yards']:.0f} YPG, Form: {p['home_form']:+.1f}, Streak: {p['home_streak']:+d}")
        print()

    # Easy wins summary
    print("=" * 90)
    print("POTENTIAL PLAYS (Edge >= 3.0 points)")
    print("=" * 90)
    print()

    plays = [(t, p, vt) for t, p, vt in results if abs(p['edge']) >= 3.0]
    if plays:
        for time, p, _ in plays:
            vegas = p['vegas_spread']
            # vegas > 0 means away team favored, vegas < 0 means home team favored
            if p['edge'] > 0:
                # Model says away team better -> bet AWAY
                if vegas > 0:
                    pick = f"{p['away_team']} -{vegas:.1f}"  # Away fav, lay points
                else:
                    pick = f"{p['away_team']} +{abs(vegas):.1f}"  # Away dog, take points
            else:
                # Model says home team better -> bet HOME
                if vegas > 0:
                    pick = f"{p['home_team']} +{vegas:.1f}"  # Home dog, take points
                else:
                    pick = f"{p['home_team']} {vegas:.1f}"  # Home fav, lay points
            print(f"  {time}: {pick} (Edge: {abs(p['edge']):.1f} pts)")
    else:
        print("  No high-confidence plays found.")

    print()


if __name__ == '__main__':
    main()
