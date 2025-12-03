"""
Update 2025 Season - Fetch current odds, backfill historical odds, generate predictions
"""
import json
import sqlite3
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def load_api_key():
    """Load API key from config"""
    with open('odds_api_config.json', 'r') as f:
        config = json.load(f)
        return config.get('api_key')


def fetch_current_odds(sport_key, api_key):
    """Fetch current odds for a sport"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'spreads,totals',
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }

    response = requests.get(url, params=params, timeout=30)
    remaining = response.headers.get('x-requests-remaining', 'Unknown')
    print(f"  API requests remaining: {remaining}")

    if response.status_code == 200:
        return response.json()
    else:
        print(f"  Error: {response.status_code}")
        return []


def fetch_historical_odds(sport_key, api_key, date_str):
    """Fetch historical odds for a specific date"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds-history"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'spreads,totals',
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'date': date_str
    }

    response = requests.get(url, params=params, timeout=30)
    remaining = response.headers.get('x-requests-remaining', 'Unknown')

    if response.status_code == 200:
        data = response.json()
        return data.get('data', []), remaining
    else:
        return [], remaining


class NFLGameMatcher:
    """Match Odds API games to NFL database"""

    def __init__(self, db_path='nfl_games.db'):
        self.db_path = db_path

    def match_game(self, odds_game):
        """Match an odds game to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        home_team = odds_game.get('home_team', '')
        away_team = odds_game.get('away_team', '')
        game_date = odds_game.get('commence_time', '')[:10]

        cursor.execute('''
            SELECT g.game_id
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE (ht.name LIKE ? OR ht.display_name LIKE ?)
            AND (at.name LIKE ? OR at.display_name LIKE ?)
            AND DATE(g.date) = DATE(?)
        ''', (f'%{home_team}%', f'%{home_team}%',
              f'%{away_team}%', f'%{away_team}%',
              game_date))

        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None


class CFBGameMatcher:
    """Match Odds API games to CFB database"""

    def __init__(self, db_path='cfb_games.db'):
        self.db_path = db_path

    def match_game(self, odds_game):
        """Match an odds game to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        home_team = odds_game.get('home_team', '')
        away_team = odds_game.get('away_team', '')
        game_date = odds_game.get('commence_time', '')[:10]

        cursor.execute('''
            SELECT g.game_id
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE (ht.name LIKE ? OR ht.display_name LIKE ?)
            AND (at.name LIKE ? OR at.display_name LIKE ?)
            AND DATE(g.date) = DATE(?)
        ''', (f'%{home_team}%', f'%{home_team}%',
              f'%{away_team}%', f'%{away_team}%',
              game_date))

        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None


def save_odds(db_path, game_id, spread_home, total, source='TheOddsAPI'):
    """Save odds to database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if exists
    cursor.execute('SELECT id FROM game_odds WHERE game_id = ? AND source = ?', (game_id, source))
    existing = cursor.fetchone()

    if existing:
        cursor.execute('''
            UPDATE game_odds SET
                closing_spread_home = ?,
                closing_total = ?,
                latest_line = ?,
                latest_total_line = ?,
                updated_at = ?
            WHERE game_id = ? AND source = ?
        ''', (spread_home, total, spread_home, total, datetime.now().isoformat(), game_id, source))
    else:
        cursor.execute('''
            INSERT INTO game_odds
            (game_id, source, closing_spread_home, closing_total, latest_line, latest_total_line, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (game_id, source, spread_home, total, spread_home, total, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def process_odds_games(games, matcher, db_path):
    """Process odds games and save to database"""
    matched = 0
    saved = 0

    for game in games:
        home = game.get('home_team')
        away = game.get('away_team')

        if not home or not away:
            continue

        game_id = matcher.match_game(game)

        if not game_id:
            continue

        matched += 1

        bookmakers = game.get('bookmakers', [])
        if not bookmakers:
            continue

        spreads_home = []
        totals = []

        for book in bookmakers:
            for market in book.get('markets', []):
                if market['key'] == 'spreads':
                    for outcome in market.get('outcomes', []):
                        if outcome['name'] == home:
                            spreads_home.append(outcome['point'])
                elif market['key'] == 'totals':
                    for outcome in market.get('outcomes', []):
                        if outcome['name'] == 'Over':
                            totals.append(outcome['point'])

        if spreads_home or totals:
            avg_spread = sum(spreads_home) / len(spreads_home) if spreads_home else None
            avg_total = sum(totals) / len(totals) if totals else None

            save_odds(db_path, game_id,
                     round(avg_spread, 1) if avg_spread else None,
                     round(avg_total, 1) if avg_total else None)
            saved += 1

    return matched, saved


def update_nfl_odds(api_key):
    """Update NFL 2025 odds"""
    print("\n" + "="*70)
    print("UPDATING NFL 2025 ODDS")
    print("="*70)

    matcher = NFLGameMatcher()

    # Fetch current odds
    print("\nFetching current NFL odds...")
    games = fetch_current_odds('americanfootball_nfl', api_key)
    print(f"  Retrieved {len(games)} games")

    matched, saved = process_odds_games(games, matcher, 'nfl_games.db')
    print(f"  Matched: {matched}, Saved: {saved}")

    # Backfill historical odds for 2025 season (Sep-Dec)
    print("\nBackfilling historical NFL 2025 odds...")
    dates = []

    # Generate dates for NFL season (Sep 4 - current)
    for month in [9, 10, 11, 12]:
        for day in [1, 5, 8, 12, 15, 19, 22, 26, 29]:
            try:
                date = datetime(2025, month, day)
                if date <= datetime.now():
                    dates.append(date)
            except:
                pass

    total_saved = 0
    for i, date in enumerate(dates[:15], 1):  # Limit to 15 dates to save API credits
        date_str = date.strftime('%Y-%m-%dT17:00:00Z')
        print(f"  [{i}] {date.strftime('%Y-%m-%d')}...", end=" ")

        games, remaining = fetch_historical_odds('americanfootball_nfl', api_key, date_str)

        if games:
            matched, saved = process_odds_games(games, matcher, 'nfl_games.db')
            print(f"Found {len(games)}, saved {saved}")
            total_saved += saved
        else:
            print("No games")

        time.sleep(0.5)

    print(f"\nTotal historical odds saved: {total_saved}")


def update_cfb_odds(api_key):
    """Update CFB 2025 odds"""
    print("\n" + "="*70)
    print("UPDATING CFB 2025 ODDS")
    print("="*70)

    matcher = CFBGameMatcher()

    # Fetch current odds
    print("\nFetching current CFB odds...")
    games = fetch_current_odds('americanfootball_ncaaf', api_key)
    print(f"  Retrieved {len(games)} games")

    matched, saved = process_odds_games(games, matcher, 'cfb_games.db')
    print(f"  Matched: {matched}, Saved: {saved}")

    # Backfill historical odds for 2025 season (Aug-Dec)
    print("\nBackfilling historical CFB 2025 odds...")
    dates = []

    for month in [8, 9, 10, 11, 12]:
        for day in [1, 7, 14, 21, 28]:
            try:
                date = datetime(2025, month, day)
                if date <= datetime.now():
                    dates.append(date)
            except:
                pass

    total_saved = 0
    for i, date in enumerate(dates[:15], 1):  # Limit to save API credits
        date_str = date.strftime('%Y-%m-%dT17:00:00Z')
        print(f"  [{i}] {date.strftime('%Y-%m-%d')}...", end=" ")

        games, remaining = fetch_historical_odds('americanfootball_ncaaf', api_key, date_str)

        if games:
            matched, saved = process_odds_games(games, matcher, 'cfb_games.db')
            print(f"Found {len(games)}, saved {saved}")
            total_saved += saved
        else:
            print("No games")

        time.sleep(0.5)

    print(f"\nTotal historical odds saved: {total_saved}")


def main():
    api_key = load_api_key()

    if not api_key:
        print("ERROR: No API key found in odds_api_config.json")
        return

    update_nfl_odds(api_key)
    update_cfb_odds(api_key)

    print("\n" + "="*70)
    print("ODDS UPDATE COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
