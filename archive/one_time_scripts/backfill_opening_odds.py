"""
Backfill Opening Odds for 2024 NFL Season
Fetches historical odds from The Odds API for the opening line dates
"""

import json
import sqlite3
from datetime import datetime, timedelta
import requests
import time
from game_matcher import GameMatcher

# Load API key
with open('odds_api_config.json', 'r') as f:
    config = json.load(f)
    API_KEY = config['api_key']

BASE_URL = "https://api.the-odds-api.com/v4/sports"
SPORT = "americanfootball_nfl"


def fetch_historical_odds(date_str):
    """
    Fetch historical odds for a specific date

    Args:
        date_str: ISO format date (e.g., '2024-09-03T12:00:00Z')

    Returns:
        List of games with odds
    """
    url = f"{BASE_URL}/{SPORT}/odds-history"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'spreads,totals',
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'date': date_str
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        remaining = response.headers.get('x-requests-remaining')
        print(f"  API requests remaining: {remaining}")

        data = response.json()
        games = data.get('data', [])
        return games

    except Exception as e:
        print(f"  Error fetching odds: {e}")
        return []


def get_consensus_spread(game_data):
    """Calculate consensus spread from bookmakers"""
    spreads = []
    totals = []
    home_team = game_data.get('home_team')

    for bookmaker in game_data.get('bookmakers', []):
        for market in bookmaker.get('markets', []):
            if market.get('key') == 'spreads':
                for outcome in market.get('outcomes', []):
                    if outcome.get('name') == home_team:
                        point = outcome.get('point')
                        if point is not None:
                            spreads.append(point)
            elif market.get('key') == 'totals':
                for outcome in market.get('outcomes', []):
                    point = outcome.get('point')
                    if point is not None:
                        totals.append(point)
                        break

    return {
        'spread': sum(spreads) / len(spreads) if spreads else None,
        'total': sum(totals) / len(totals) if totals else None,
        'num_books': len(game_data.get('bookmakers', []))
    }


def update_opening_line(conn, game_id, opening_spread, opening_total, timestamp):
    """Update the opening_line in game_odds table"""
    cursor = conn.cursor()

    # Check if row exists
    cursor.execute('SELECT opening_line FROM game_odds WHERE game_id = ?', (game_id,))
    row = cursor.fetchone()

    if row is None:
        # Insert new row
        cursor.execute('''
            INSERT INTO game_odds (game_id, source, opening_line, opening_total_line,
                                   opening_line_timestamp, latest_line, latest_total_line)
            VALUES (?, 'TheOddsAPI_Historical', ?, ?, ?, ?, ?)
        ''', (game_id, opening_spread, opening_total, timestamp, opening_spread, opening_total))
    else:
        # Update existing - only update opening if not already set differently
        cursor.execute('''
            UPDATE game_odds SET
                opening_line = ?,
                opening_total_line = ?,
                opening_line_timestamp = ?,
                line_movement = latest_line - ?
            WHERE game_id = ?
        ''', (opening_spread, opening_total, timestamp, opening_spread, game_id))

    conn.commit()


def main():
    print("=" * 80)
    print("BACKFILLING 2024 NFL OPENING ODDS")
    print("=" * 80)

    conn = sqlite3.connect('nfl_games.db')
    matcher = GameMatcher('nfl_games.db')

    # 2024 NFL weeks and their approximate opening dates
    # Opening lines typically come out Sunday/Monday after previous week
    week_opening_dates = [
        # (week, opening_date) - fetch odds from ~5-6 days before first game
        (1, '2024-09-01T12:00:00Z'),   # Week 1 - lines out early
        (2, '2024-09-09T12:00:00Z'),   # After Week 1 Monday game
        (3, '2024-09-16T12:00:00Z'),
        (4, '2024-09-23T12:00:00Z'),
        (5, '2024-09-30T12:00:00Z'),
        (6, '2024-10-07T12:00:00Z'),
        (7, '2024-10-14T12:00:00Z'),
        (8, '2024-10-21T12:00:00Z'),
        (9, '2024-10-28T12:00:00Z'),
        (10, '2024-11-04T12:00:00Z'),
        (11, '2024-11-11T12:00:00Z'),
        (12, '2024-11-18T12:00:00Z'),
        (13, '2024-11-25T12:00:00Z'),
        (14, '2024-12-02T12:00:00Z'),
        (15, '2024-12-09T12:00:00Z'),
        (16, '2024-12-16T12:00:00Z'),
        (17, '2024-12-23T12:00:00Z'),
        (18, '2024-12-30T12:00:00Z'),
    ]

    total_updated = 0
    total_matched = 0

    for week, opening_date in week_opening_dates:
        print(f"\n{'='*60}")
        print(f"Week {week} - Fetching opening odds from {opening_date[:10]}")
        print('='*60)

        games = fetch_historical_odds(opening_date)
        print(f"  Retrieved {len(games)} games from API")

        if not games:
            print("  No games found, skipping...")
            continue

        matched = 0
        updated = 0

        for game in games:
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            commence_time = game.get('commence_time')

            # Get consensus odds
            consensus = get_consensus_spread(game)

            if consensus['spread'] is None:
                continue

            # Match to our database
            game_id = matcher.match_odds_api_game({
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time
            })

            if game_id:
                matched += 1
                update_opening_line(
                    conn,
                    game_id,
                    consensus['spread'],
                    consensus['total'],
                    opening_date
                )
                updated += 1
                print(f"  + {away_team} @ {home_team}: opening spread {consensus['spread']:+.1f}")

        total_matched += matched
        total_updated += updated
        print(f"  Week {week}: Matched {matched}, Updated {updated}")

        # Brief pause between API calls
        time.sleep(1)

    conn.close()

    print(f"\n{'='*80}")
    print("BACKFILL COMPLETE")
    print('='*80)
    print(f"Total games matched: {total_matched}")
    print(f"Total opening lines updated: {total_updated}")


if __name__ == '__main__':
    main()
