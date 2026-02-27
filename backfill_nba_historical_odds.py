"""
NBA Historical Odds Backfill using The Odds API
Updates odds_and_predictions table with historical spread/total data
"""
import json
import time
import sqlite3
import requests
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Team name mappings: Odds API -> ESPN
TEAM_MAPPINGS = {
    'Los Angeles Lakers': 'LA Lakers',
    'Los Angeles Clippers': 'LA Clippers',
    'LA Clippers': 'LA Clippers',
    'LA Lakers': 'LA Lakers',
}


def load_api_key():
    """Load API key from config file"""
    with open('odds_api_config.json', 'r') as f:
        return json.load(f)['api_key']


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching"""
    return TEAM_MAPPINGS.get(name, name)


def fetch_historical_odds(api_key: str, date: str, markets: list = None):
    """Fetch historical NBA odds from The Odds API"""
    if markets is None:
        markets = ['spreads', 'totals']

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds-history"
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

    except Exception as e:
        logger.error(f"Error fetching odds: {e}")
        return [], None


def match_game_to_db(conn, home_team: str, away_team: str, game_date: str) -> int | None:
    """Match an Odds API game to our database"""
    home_norm = normalize_team_name(home_team)
    away_norm = normalize_team_name(away_team)

    cursor = conn.cursor()

    # Try to match by teams and date
    cursor.execute('''
        SELECT g.game_id
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE (ht.name LIKE ? OR ht.display_name LIKE ?)
        AND (at.name LIKE ? OR at.display_name LIKE ?)
        AND DATE(g.date) = DATE(?)
    ''', (f'%{home_norm}%', f'%{home_norm}%',
          f'%{away_norm}%', f'%{away_norm}%',
          game_date))

    result = cursor.fetchone()
    return result[0] if result else None


def save_odds_to_db(conn, game_id: int, spread: float | None, total: float | None, timestamp: str):
    """Save odds to odds_and_predictions table"""
    cursor = conn.cursor()

    # Check if record exists
    cursor.execute('SELECT id, latest_spread, latest_total FROM odds_and_predictions WHERE game_id = ?', (game_id,))
    existing = cursor.fetchone()

    if existing:
        # Only update if we have new data
        updates = []
        params = []

        if spread is not None and existing[1] is None:
            updates.append('latest_spread = ?')
            params.append(spread)
        if total is not None and existing[2] is None:
            updates.append('latest_total = ?')
            params.append(total)

        if updates:
            updates.append('odds_updated_at = ?')
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
            INSERT INTO odds_and_predictions (game_id, source, latest_spread, latest_total, odds_updated_at)
            VALUES (?, 'TheOddsAPI', ?, ?, ?)
        ''', (game_id, spread, total, timestamp))
        return 'inserted'


def backfill_nba_season(season_year: int, interval_days: int = 3):
    """
    Backfill NBA odds for a season

    Args:
        season_year: The year the season ENDS (e.g., 2024 for 2023-24 season)
        interval_days: Days between API calls (3 = every 3 days)
    """
    print(f"\n{'='*80}")
    print(f"NBA {season_year-1}-{str(season_year)[-2:]} HISTORICAL ODDS BACKFILL")
    print(f"{'='*80}\n")

    api_key = load_api_key()
    conn = sqlite3.connect('nba_games.db')

    # NBA season runs Oct-Jun
    # Season 2024 = Oct 2023 to Jun 2024
    start_date = datetime(season_year - 1, 10, 20)  # Late October when season starts
    end_date = datetime(season_year, 6, 20)  # June when finals end

    # Generate sample dates
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=interval_days)

    print(f"Sampling {len(dates)} dates (every {interval_days} days)")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Estimated API cost: {len(dates) * 20} credits (10 per market x 2 markets)\n")

    stats = {'fetched': 0, 'matched': 0, 'inserted': 0, 'updated': 0, 'skipped': 0}
    remaining = None

    for i, date in enumerate(dates, 1):
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

            for book in bookmakers:
                for market in book.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == home:
                                spreads_home.append(outcome.get('point', 0))
                    elif market['key'] == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == 'Over':
                                totals.append(outcome.get('point', 0))

            # Calculate consensus
            avg_spread = round(sum(spreads_home) / len(spreads_home), 1) if spreads_home else None
            avg_total = round(sum(totals) / len(totals), 1) if totals else None

            if avg_spread is not None or avg_total is not None:
                result = save_odds_to_db(conn, game_id, avg_spread, avg_total, date_str)
                stats[result] += 1
                if result in ('inserted', 'updated'):
                    saved += 1

        stats['matched'] += matched
        conn.commit()
        print(f"Found {len(games)}, matched {matched}, saved {saved}")

        # Rate limiting
        time.sleep(1.5)

        # Progress update every 20 dates
        if i % 20 == 0:
            print(f"\n--- Progress: {i}/{len(dates)} | API remaining: {remaining} ---\n")

    conn.close()

    print(f"\n{'='*80}")
    print(f"BACKFILL COMPLETE - {season_year-1}-{str(season_year)[-2:]} Season")
    print(f"{'='*80}")
    print(f"Games fetched from API: {stats['fetched']}")
    print(f"Games matched to DB: {stats['matched']}")
    print(f"New odds inserted: {stats['inserted']}")
    print(f"Odds updated: {stats['updated']}")
    print(f"Skipped (already had odds): {stats['skipped']}")
    print(f"API requests remaining: {remaining}")

    return stats


def main():
    import sys

    if len(sys.argv) > 1:
        season = int(sys.argv[1])
        backfill_nba_season(season)
    else:
        # Check current coverage first
        conn = sqlite3.connect('nba_games.db')
        print("Current NBA odds coverage:")
        q = '''SELECT g.season, COUNT(*) as games,
               SUM(CASE WHEN o.latest_spread IS NOT NULL THEN 1 ELSE 0 END) as with_odds
               FROM games g
               LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
               WHERE g.completed = 1
               GROUP BY g.season ORDER BY g.season'''
        for row in conn.execute(q).fetchall():
            pct = (row[2]/row[1]*100) if row[1] > 0 else 0
            print(f"  {row[0]}: {row[2]}/{row[1]} ({pct:.0f}%)")
        conn.close()

        print("\nUsage: python backfill_nba_historical_odds.py <season_year>")
        print("  season_year = ending year (e.g., 2024 for 2023-24 season)")


if __name__ == '__main__':
    main()
