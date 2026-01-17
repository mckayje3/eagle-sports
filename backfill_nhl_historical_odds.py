"""
NHL Historical Odds Backfill using The Odds API
Updates odds_and_predictions table with historical spread/total data for NHL games

Usage:
    python backfill_nhl_historical_odds.py 2023    # Backfill 2022-23 season
    python backfill_nhl_historical_odds.py 2024    # Backfill 2023-24 season
    python backfill_nhl_historical_odds.py 2025    # Backfill 2024-25 season
    python backfill_nhl_historical_odds.py         # Show current coverage

Requires: odds_api_config.json with {"api_key": "YOUR_KEY"}
Get API key from: https://the-odds-api.com (free tier: 500 requests/month)
"""
from __future__ import annotations

import json
import time
import sqlite3
import requests
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = 'nhl_games.db'

# NHL team name mappings: Odds API -> ESPN naming
TEAM_MAPPINGS = {
    # Common variations
    'Montreal Canadiens': 'Montréal Canadiens',
    'Vegas Golden Knights': 'Vegas Golden Knights',
    'St Louis Blues': 'St. Louis Blues',
    'St. Louis Blues': 'St. Louis Blues',
}


def load_api_key() -> str:
    """Load API key from config file"""
    try:
        with open('odds_api_config.json', 'r') as f:
            return json.load(f)['api_key']
    except FileNotFoundError:
        print("ERROR: odds_api_config.json not found!")
        print("Create this file with: {\"api_key\": \"YOUR_KEY\"}")
        print("Get API key from: https://the-odds-api.com")
        raise


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching"""
    return TEAM_MAPPINGS.get(name, name)


def fetch_historical_odds(api_key: str, date: str, markets: list | None = None):
    """
    Fetch historical NHL odds from The Odds API

    Args:
        api_key: The Odds API key
        date: ISO format date string (e.g., '2023-10-15T00:00:00Z')
        markets: List of markets to fetch (default: spreads, totals, h2h)

    Returns:
        (games_list, remaining_requests)
    """
    if markets is None:
        markets = ['spreads', 'totals', 'h2h']  # h2h = moneyline

    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds-history"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': ','.join(markets),
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'date': date
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        remaining = response.headers.get('x-requests-remaining', 'Unknown')

        if response.status_code != 200:
            logger.error(f"API error {response.status_code}: {response.text[:200]}")
            return [], remaining

        data = response.json()
        games = data.get('data', [])
        return games, remaining

    except requests.RequestException as e:
        logger.error(f"Request error: {e}")
        return [], None


def match_game_to_db(conn: sqlite3.Connection, home_team: str, away_team: str, game_date: str) -> int | None:
    """
    Match an Odds API game to our database

    Args:
        conn: SQLite connection
        home_team: Home team name from Odds API
        away_team: Away team name from Odds API
        game_date: Game date (YYYY-MM-DD format)

    Returns:
        game_id if matched, None otherwise
    """
    home_norm = normalize_team_name(home_team)
    away_norm = normalize_team_name(away_team)

    cursor = conn.cursor()

    # Try to match by teams and date (with some flexibility)
    cursor.execute('''
        SELECT g.game_id
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE (ht.name LIKE ? OR ht.display_name LIKE ?)
        AND (at.name LIKE ? OR at.display_name LIKE ?)
        AND DATE(g.game_date_eastern) = DATE(?)
    ''', (f'%{home_norm}%', f'%{home_norm}%',
          f'%{away_norm}%', f'%{away_norm}%',
          game_date))

    result = cursor.fetchone()
    return result[0] if result else None


def save_odds_to_db(conn: sqlite3.Connection, game_id: int, spread: float | None,
                    total: float | None, ml_home: int | None, ml_away: int | None,
                    timestamp: str) -> str:
    """
    Save odds to odds_and_predictions table

    Args:
        conn: SQLite connection
        game_id: ESPN game ID
        spread: Home team spread (negative = favored)
        total: Over/under total
        ml_home: Home team moneyline (American odds)
        ml_away: Away team moneyline (American odds)
        timestamp: Odds timestamp

    Returns:
        'inserted', 'updated', or 'skipped'
    """
    cursor = conn.cursor()

    # Check if record exists
    cursor.execute('''SELECT id, latest_spread, latest_total, latest_moneyline_home
                      FROM odds_and_predictions WHERE game_id = ?''', (game_id,))
    existing = cursor.fetchone()

    if existing:
        # Only update if we have new data and existing data is missing
        updates = []
        params = []

        if spread is not None and existing[1] is None:
            updates.append('latest_spread = ?')
            updates.append('opening_spread = ?')
            params.extend([spread, spread])
        if total is not None and existing[2] is None:
            updates.append('latest_total = ?')
            updates.append('opening_total = ?')
            params.extend([total, total])
        if ml_home is not None and existing[3] is None:
            updates.append('latest_moneyline_home = ?')
            updates.append('opening_moneyline_home = ?')
            params.extend([ml_home, ml_home])
        if ml_away is not None and existing[3] is None:  # Use same check as home
            updates.append('latest_moneyline_away = ?')
            updates.append('opening_moneyline_away = ?')
            params.extend([ml_away, ml_away])

        if updates:
            updates.append('odds_updated_at = ?')
            updates.append("source = 'TheOddsAPI'")
            params.append(timestamp)
            params.append(game_id)

            cursor.execute(f'''
                UPDATE odds_and_predictions
                SET {', '.join(updates)}
                WHERE game_id = ?
            ''', params)
            return 'updated'
        return 'skipped'
    else:
        # Insert new record
        cursor.execute('''
            INSERT INTO odds_and_predictions
            (game_id, source, opening_spread, latest_spread, opening_total, latest_total,
             opening_moneyline_home, latest_moneyline_home, opening_moneyline_away, latest_moneyline_away,
             odds_updated_at)
            VALUES (?, 'TheOddsAPI', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (game_id, spread, spread, total, total, ml_home, ml_home, ml_away, ml_away, timestamp))
        return 'inserted'


def backfill_nhl_season(season_year: int, interval_days: int = 3):
    """
    Backfill NHL odds for a season

    Args:
        season_year: The year the season STARTS (e.g., 2022 for 2022-23 season)
        interval_days: Days between API calls (3 = every 3 days to save API credits)
    """
    print(f"\n{'='*80}")
    print(f"NHL {season_year}-{str(season_year+1)[-2:]} HISTORICAL ODDS BACKFILL")
    print(f"{'='*80}\n")

    api_key = load_api_key()
    conn = sqlite3.connect(DB_PATH)

    # NHL season runs Oct-Jun
    start_date = datetime(season_year, 10, 10)  # After preseason
    end_date = datetime(season_year + 1, 6, 20)  # After Stanley Cup Finals

    # Generate sample dates
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=interval_days)

    print(f"Sampling {len(dates)} dates (every {interval_days} days)")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Estimated API cost: {len(dates) * 20} credits (10 per market x 2 markets)")
    print(f"NHL sport key: icehockey_nhl\n")

    stats = {'fetched': 0, 'matched': 0, 'inserted': 0, 'updated': 0, 'skipped': 0}
    remaining = None

    for i, date in enumerate(dates, 1):
        # Use noon UTC to capture most games for that day
        date_str = date.strftime('%Y-%m-%dT17:00:00Z')
        print(f"[{i}/{len(dates)}] {date.strftime('%Y-%m-%d')}...", end=" ", flush=True)

        games, remaining = fetch_historical_odds(api_key, date_str)

        if not games:
            print("No games")
            continue

        stats['fetched'] += len(games)
        matched = 0
        saved = 0

        for game in games:
            home = game.get('home_team')
            away = game.get('away_team')
            commence = game.get('commence_time', '')[:10]

            if not home or not away:
                continue

            game_id = match_game_to_db(conn, home, away, commence)

            if not game_id:
                continue

            matched += 1

            # Extract odds from bookmakers
            bookmakers = game.get('bookmakers', [])
            spreads_home = []
            totals = []
            ml_home_list = []
            ml_away_list = []

            for book in bookmakers:
                for market in book.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == home:
                                point = outcome.get('point')
                                if point is not None:
                                    spreads_home.append(point)
                    elif market['key'] == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == 'Over':
                                point = outcome.get('point')
                                if point is not None:
                                    totals.append(point)
                    elif market['key'] == 'h2h':  # Moneyline
                        for outcome in market.get('outcomes', []):
                            price = outcome.get('price')
                            if price is not None:
                                if outcome['name'] == home:
                                    ml_home_list.append(price)
                                elif outcome['name'] == away:
                                    ml_away_list.append(price)

            # Calculate consensus (average across bookmakers)
            avg_spread = round(sum(spreads_home) / len(spreads_home), 1) if spreads_home else None
            avg_total = round(sum(totals) / len(totals), 1) if totals else None
            avg_ml_home = round(sum(ml_home_list) / len(ml_home_list)) if ml_home_list else None
            avg_ml_away = round(sum(ml_away_list) / len(ml_away_list)) if ml_away_list else None

            if avg_spread is not None or avg_total is not None or avg_ml_home is not None:
                result = save_odds_to_db(conn, game_id, avg_spread, avg_total,
                                         avg_ml_home, avg_ml_away, date_str)
                stats[result] += 1
                if result in ('inserted', 'updated'):
                    saved += 1

        stats['matched'] += matched
        conn.commit()
        print(f"Found {len(games)}, matched {matched}, saved {saved}")

        # Rate limiting (Odds API has rate limits)
        time.sleep(1.5)

        # Progress update every 20 dates
        if i % 20 == 0:
            print(f"\n--- Progress: {i}/{len(dates)} | API remaining: {remaining} ---\n")

    conn.close()

    print(f"\n{'='*80}")
    print(f"BACKFILL COMPLETE - {season_year}-{str(season_year+1)[-2:]} Season")
    print(f"{'='*80}")
    print(f"Games fetched from API: {stats['fetched']}")
    print(f"Games matched to DB: {stats['matched']}")
    print(f"New odds inserted: {stats['inserted']}")
    print(f"Odds updated: {stats['updated']}")
    print(f"Skipped (already had odds): {stats['skipped']}")
    print(f"API requests remaining: {remaining}")

    return stats


def show_coverage():
    """Show current odds coverage for NHL"""
    conn = sqlite3.connect(DB_PATH)
    print("\nCurrent NHL odds coverage:")
    print("-" * 50)

    query = '''
        SELECT g.season,
               COUNT(*) as total_games,
               SUM(CASE WHEN g.completed = 1 THEN 1 ELSE 0 END) as completed,
               SUM(CASE WHEN o.latest_spread IS NOT NULL THEN 1 ELSE 0 END) as with_spread,
               SUM(CASE WHEN o.latest_total IS NOT NULL THEN 1 ELSE 0 END) as with_total
        FROM games g
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        GROUP BY g.season
        ORDER BY g.season
    '''

    cursor = conn.cursor()
    cursor.execute(query)

    print(f"{'Season':<10} {'Total':<8} {'Done':<8} {'Spread':<10} {'Total':<10}")
    print("-" * 50)

    for row in cursor.fetchall():
        season, total, completed, with_spread, with_total = row
        spread_pct = (with_spread / total * 100) if total > 0 else 0
        total_pct = (with_total / total * 100) if total > 0 else 0
        print(f"{season}-{str(season+1)[-2:]:<5} {total:<8} {completed:<8} "
              f"{with_spread:<4} ({spread_pct:>4.0f}%) {with_total:<4} ({total_pct:>4.0f}%)")

    conn.close()


def main():
    import sys

    if len(sys.argv) > 1:
        season = int(sys.argv[1])
        # season_year = starting year (e.g., 2022 for 2022-23 season)
        backfill_nhl_season(season)
    else:
        show_coverage()
        print("\nUsage: python backfill_nhl_historical_odds.py <season_year>")
        print("  season_year = starting year (e.g., 2022 for 2022-23 season)")
        print("\nTo backfill all seasons:")
        print("  python backfill_nhl_historical_odds.py 2022  # 2022-23")
        print("  python backfill_nhl_historical_odds.py 2023  # 2023-24")
        print("  python backfill_nhl_historical_odds.py 2024  # 2024-25")
        print("\nRequires odds_api_config.json with your API key")
        print("Get key from: https://the-odds-api.com")


if __name__ == '__main__':
    main()
