"""
Non-interactive backfill of NFL odds (2023-2025)
"""
import json
import time
import sqlite3
from datetime import datetime
from odds_api_scraper import OddsAPIScraper

# NFL 2023 Season (17 weeks + playoffs)
NFL_2023_DATES = [
    ('2023-09-07T20:00:00Z', '2023 Week 1'),
    ('2023-09-14T20:00:00Z', '2023 Week 2'),
    ('2023-09-21T20:00:00Z', '2023 Week 3'),
    ('2023-09-28T20:00:00Z', '2023 Week 4'),
    ('2023-10-05T20:00:00Z', '2023 Week 5'),
    ('2023-10-12T20:00:00Z', '2023 Week 6'),
    ('2023-10-19T20:00:00Z', '2023 Week 7'),
    ('2023-10-26T20:00:00Z', '2023 Week 8'),
    ('2023-11-02T20:00:00Z', '2023 Week 9'),
    ('2023-11-09T20:00:00Z', '2023 Week 10'),
    ('2023-11-16T20:00:00Z', '2023 Week 11'),
    ('2023-11-23T20:00:00Z', '2023 Week 12'),
    ('2023-11-30T20:00:00Z', '2023 Week 13'),
    ('2023-12-07T20:00:00Z', '2023 Week 14'),
    ('2023-12-14T20:00:00Z', '2023 Week 15'),
    ('2023-12-21T20:00:00Z', '2023 Week 16'),
    ('2023-12-28T20:00:00Z', '2023 Week 17'),
    ('2024-01-06T20:00:00Z', '2023 Week 18'),
]

# NFL 2024 Season
NFL_2024_DATES = [
    ('2024-09-05T20:00:00Z', '2024 Week 1'),
    ('2024-09-12T20:00:00Z', '2024 Week 2'),
    ('2024-09-19T20:00:00Z', '2024 Week 3'),
    ('2024-09-26T20:00:00Z', '2024 Week 4'),
    ('2024-10-03T20:00:00Z', '2024 Week 5'),
    ('2024-10-10T20:00:00Z', '2024 Week 6'),
    ('2024-10-17T20:00:00Z', '2024 Week 7'),
    ('2024-10-24T20:00:00Z', '2024 Week 8'),
    ('2024-10-31T20:00:00Z', '2024 Week 9'),
    ('2024-11-07T20:00:00Z', '2024 Week 10'),
    ('2024-11-14T20:00:00Z', '2024 Week 11'),
    ('2024-11-21T20:00:00Z', '2024 Week 12'),
    ('2024-11-28T20:00:00Z', '2024 Week 13'),
    ('2024-12-05T20:00:00Z', '2024 Week 14'),
    ('2024-12-12T20:00:00Z', '2024 Week 15'),
    ('2024-12-19T20:00:00Z', '2024 Week 16'),
    ('2024-12-26T20:00:00Z', '2024 Week 17'),
    ('2025-01-02T20:00:00Z', '2024 Week 18'),
]

# NFL 2025 Season (up to current week 12)
NFL_2025_DATES = [
    ('2025-09-04T20:00:00Z', '2025 Week 1'),
    ('2025-09-11T20:00:00Z', '2025 Week 2'),
    ('2025-09-18T20:00:00Z', '2025 Week 3'),
    ('2025-09-25T20:00:00Z', '2025 Week 4'),
    ('2025-10-02T20:00:00Z', '2025 Week 5'),
    ('2025-10-09T20:00:00Z', '2025 Week 6'),
    ('2025-10-16T20:00:00Z', '2025 Week 7'),
    ('2025-10-23T20:00:00Z', '2025 Week 8'),
    ('2025-10-30T20:00:00Z', '2025 Week 9'),
    ('2025-11-06T20:00:00Z', '2025 Week 10'),
    ('2025-11-13T20:00:00Z', '2025 Week 11'),
    ('2025-11-20T20:00:00Z', '2025 Week 12'),
]

# NFL team name mapping (Odds API to ESPN)
NFL_TEAM_MAPPING = {
    'Arizona Cardinals': 'Cardinals',
    'Atlanta Falcons': 'Falcons',
    'Baltimore Ravens': 'Ravens',
    'Buffalo Bills': 'Bills',
    'Carolina Panthers': 'Panthers',
    'Chicago Bears': 'Bears',
    'Cincinnati Bengals': 'Bengals',
    'Cleveland Browns': 'Browns',
    'Dallas Cowboys': 'Cowboys',
    'Denver Broncos': 'Broncos',
    'Detroit Lions': 'Lions',
    'Green Bay Packers': 'Packers',
    'Houston Texans': 'Texans',
    'Indianapolis Colts': 'Colts',
    'Jacksonville Jaguars': 'Jaguars',
    'Kansas City Chiefs': 'Chiefs',
    'Las Vegas Raiders': 'Raiders',
    'Los Angeles Chargers': 'Chargers',
    'Los Angeles Rams': 'Rams',
    'Miami Dolphins': 'Dolphins',
    'Minnesota Vikings': 'Vikings',
    'New England Patriots': 'Patriots',
    'New Orleans Saints': 'Saints',
    'New York Giants': 'Giants',
    'New York Jets': 'Jets',
    'Philadelphia Eagles': 'Eagles',
    'Pittsburgh Steelers': 'Steelers',
    'San Francisco 49ers': '49ers',
    'Seattle Seahawks': 'Seahawks',
    'Tampa Bay Buccaneers': 'Buccaneers',
    'Tennessee Titans': 'Titans',
    'Washington Commanders': 'Commanders',
}


def match_nfl_game(db_conn, home_team: str, away_team: str, game_date: str):
    """Match an NFL game from Odds API to ESPN game_id"""
    cursor = db_conn.cursor()

    # Normalize team names
    home_name = NFL_TEAM_MAPPING.get(home_team, home_team)
    away_name = NFL_TEAM_MAPPING.get(away_team, away_team)

    # Find team IDs
    cursor.execute('SELECT team_id FROM teams WHERE name = ? OR display_name LIKE ?',
                   (home_name, f'%{home_name}%'))
    home_result = cursor.fetchone()

    cursor.execute('SELECT team_id FROM teams WHERE name = ? OR display_name LIKE ?',
                   (away_name, f'%{away_name}%'))
    away_result = cursor.fetchone()

    if not home_result or not away_result:
        return None

    home_id = home_result[0]
    away_id = away_result[0]

    # Parse date
    try:
        date_obj = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
        date_str = date_obj.strftime('%Y-%m-%d')

        # Look for game within 3 days
        cursor.execute('''
            SELECT game_id FROM games
            WHERE home_team_id = ? AND away_team_id = ?
            AND date BETWEEN ? AND ?
        ''', (home_id, away_id,
              f"{date_str}T00:00:00",
              f"{(date_obj.replace(day=date_obj.day+7)).strftime('%Y-%m-%d')}T23:59:59"))

        result = cursor.fetchone()
        if result:
            return result[0]
    except:
        pass

    return None


def backfill_nfl_odds():
    """Backfill NFL odds for 2023-2025"""
    print("\n" + "="*80)
    print("NFL ODDS BACKFILL (2023-2025)")
    print("="*80 + "\n")

    # Load API key
    try:
        with open('odds_api_config.json', 'r') as f:
            config = json.load(f)
            api_key = config.get('api_key')
    except FileNotFoundError:
        print("ERROR: odds_api_config.json not found!")
        return

    if not api_key:
        print("ERROR: No API key in config file")
        return

    # Initialize - override sport for NFL
    scraper = OddsAPIScraper(api_key=api_key)
    scraper.SPORT = 'americanfootball_nfl'  # Override default CFB sport
    conn = sqlite3.connect('nfl_games.db')

    # Check remaining requests
    remaining = scraper.get_remaining_requests()
    all_dates = NFL_2023_DATES + NFL_2024_DATES + NFL_2025_DATES
    print(f"Starting API requests remaining: {remaining}")
    print(f"Total weeks to fetch: {len(all_dates)}")
    print(f"Estimated cost: ~{len(all_dates) * 20} requests\n")

    total_games = 0
    total_matched = 0
    total_saved = 0

    for i, (date, description) in enumerate(all_dates, 1):
        print(f"\n[{i}/{len(all_dates)}] {description} ({date[:10]})")

        games = scraper.fetch_historical_odds(
            date=date,
            markets=['spreads', 'totals']
        )

        if not games:
            print(f"  No games found")
            continue

        print(f"  Retrieved {len(games)} games")
        total_games += len(games)

        matched = 0
        saved = 0

        for game in games:
            home = game.get('home_team')
            away = game.get('away_team')
            commence = game.get('commence_time')

            if not home or not away:
                continue

            game_id = match_nfl_game(conn, home, away, commence or date)

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

                try:
                    cursor = conn.cursor()
                    cursor.execute('SELECT id FROM game_odds WHERE game_id = ? AND source = ?',
                                   (game_id, 'TheOddsAPI'))
                    existing = cursor.fetchone()

                    if existing:
                        cursor.execute('''
                            UPDATE game_odds SET
                                closing_spread_home = ?,
                                closing_total = ?,
                                updated_at = datetime('now')
                            WHERE game_id = ? AND source = ?
                        ''', (
                            round(avg_spread, 1) if avg_spread else None,
                            round(avg_total, 1) if avg_total else None,
                            game_id,
                            'TheOddsAPI'
                        ))
                    else:
                        cursor.execute('''
                            INSERT INTO game_odds
                            (game_id, source, closing_spread_home, closing_total, timestamp)
                            VALUES (?, ?, ?, ?, datetime('now'))
                        ''', (
                            game_id,
                            'TheOddsAPI',
                            round(avg_spread, 1) if avg_spread else None,
                            round(avg_total, 1) if avg_total else None
                        ))
                    conn.commit()
                    saved += 1
                except Exception as e:
                    print(f"  Error saving game_id {game_id}: {e}")

        total_matched += matched
        total_saved += saved
        print(f"  Matched: {matched} | Saved: {saved}")

        if i < len(all_dates):
            time.sleep(2)

    conn.close()

    print("\n" + "="*80)
    print("NFL ODDS BACKFILL COMPLETE")
    print("="*80)
    print(f"Total games fetched: {total_games}")
    print(f"Total matched: {total_matched}")
    print(f"Total saved: {total_saved}")
    print(f"API requests remaining: {scraper.get_remaining_requests()}")

    return total_saved


if __name__ == '__main__':
    backfill_nfl_odds()
