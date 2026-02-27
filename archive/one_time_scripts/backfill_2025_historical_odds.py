"""
Backfill 2025 historical odds using The Odds API
Strategy: Fetch heavy game days (20+ games) for maximum efficiency
Cost: 280 requests (14 dates × 20 requests per date)
"""
import json
import sqlite3
import time
from datetime import datetime
from odds_api_scraper import OddsAPIScraper
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_heavy_game_dates():
    """Get dates with 20+ games from 2025 season weeks 1-14"""
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT DATE(date) as game_date, COUNT(*) as num_games
        FROM games
        WHERE season = 2025 AND week BETWEEN 1 AND 14
        GROUP BY DATE(date)
        HAVING COUNT(*) >= 20
        ORDER BY game_date
    ''')

    dates = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()

    return dates


def match_team_to_espn(api_team_name, cursor):
    """
    Match The Odds API team name to ESPN team_id using school names

    Args:
        api_team_name: Team name from The Odds API (e.g., "Alabama Crimson Tide")
        cursor: SQLite cursor

    Returns:
        team_id or None
    """
    # The Odds API uses full names like "Alabama Crimson Tide"
    # We need to match against school_name (e.g., "Alabama")

    # Try exact match on display_name first
    cursor.execute('''
        SELECT team_id, school_name, display_name, name
        FROM teams
        WHERE display_name = ? OR school_name = ?
    ''', (api_team_name, api_team_name))

    result = cursor.fetchone()
    if result:
        logger.debug(f"Exact match: '{api_team_name}' -> {result[1]} (ID: {result[0]})")
        return result[0]

    # Try partial match - extract school name from full name
    # e.g., "Alabama Crimson Tide" -> look for "Alabama"
    cursor.execute('''
        SELECT team_id, school_name, display_name, name
        FROM teams
        WHERE ? LIKE school_name || '%' OR school_name LIKE ? || '%'
        ORDER BY LENGTH(school_name) DESC
        LIMIT 1
    ''', (api_team_name, api_team_name))

    result = cursor.fetchone()
    if result:
        logger.debug(f"Partial match: '{api_team_name}' -> {result[1]} (ID: {result[0]})")
        return result[0]

    # Try fuzzy match with individual words
    words = api_team_name.split()
    for word in words:
        if len(word) > 3:  # Skip short words like "at", "of"
            cursor.execute('''
                SELECT team_id, school_name, display_name
                FROM teams
                WHERE school_name LIKE ? OR display_name LIKE ?
                ORDER BY LENGTH(school_name) DESC
                LIMIT 1
            ''', (f'%{word}%', f'%{word}%'))

            result = cursor.fetchone()
            if result:
                logger.debug(f"Fuzzy match: '{api_team_name}' (word: {word}) -> {result[1]} (ID: {result[0]})")
                return result[0]

    logger.warning(f"Could not match team: '{api_team_name}'")
    return None


def find_game_id(home_team_id, away_team_id, game_date, cursor):
    """Find ESPN game_id for a matchup on a specific date"""
    cursor.execute('''
        SELECT game_id
        FROM games
        WHERE home_team_id = ? AND away_team_id = ?
        AND DATE(date) = ?
        AND season = 2025
    ''', (home_team_id, away_team_id, game_date))

    result = cursor.fetchone()
    return result[0] if result else None


def backfill_historical_odds():
    """Main function to backfill historical odds"""

    print("=" * 80)
    print("2025 HISTORICAL ODDS BACKFILL")
    print("Strategy: Heavy game days (20+ games per day)")
    print("=" * 80)

    # Load API key
    with open('odds_api_config.json', 'r') as f:
        config = json.load(f)

    scraper = OddsAPIScraper(api_key=config['api_key'])

    # Check remaining requests
    remaining_before = scraper.get_remaining_requests()
    print(f"\nRequests available: {remaining_before}")

    # Get heavy game dates
    game_dates = get_heavy_game_dates()
    print(f"\nFound {len(game_dates)} heavy game dates")
    print(f"Estimated cost: {len(game_dates) * 20} requests (spreads + totals)")
    print(f"Expected remaining: {remaining_before - (len(game_dates) * 20)} requests\n")

    # Confirm before proceeding
    response = input("Proceed with fetching? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return

    print("\n" + "=" * 80)
    print("FETCHING HISTORICAL ODDS")
    print("=" * 80 + "\n")

    # Connect to database
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    stats = {
        'dates_processed': 0,
        'games_found': 0,
        'games_matched': 0,
        'odds_saved': 0,
        'requests_used': 0
    }

    for date_str, num_games in game_dates:
        stats['dates_processed'] += 1

        print(f"\n[{stats['dates_processed']}/{len(game_dates)}] {date_str} ({num_games} games)")
        print("-" * 60)

        # Convert date to ISO format for API
        # The Odds API expects: YYYY-MM-DDTHH:MM:SSZ
        # Fetch odds for noon on that day
        iso_date = f"{date_str}T12:00:00Z"

        try:
            # Fetch historical odds (costs 20 requests: 10 for spreads, 10 for totals)
            games_data = scraper.fetch_historical_odds(
                date=iso_date,
                markets=['spreads', 'totals'],
                regions='us'
            )

            stats['requests_used'] += 20
            stats['games_found'] += len(games_data)

            print(f"  Retrieved {len(games_data)} games from API")

            # Process each game
            for game_data in games_data:
                try:
                    # Parse odds data
                    parsed = scraper.parse_odds_data(game_data)
                    consensus = scraper.get_consensus_odds(parsed)

                    home_team_name = consensus.get('home_team')
                    away_team_name = consensus.get('away_team')

                    # Match teams to ESPN IDs
                    home_id = match_team_to_espn(home_team_name, cursor)
                    away_id = match_team_to_espn(away_team_name, cursor)

                    if not home_id or not away_id:
                        logger.warning(f"  Could not match: {away_team_name} @ {home_team_name}")
                        continue

                    # Find the ESPN game_id
                    game_id = find_game_id(home_id, away_id, date_str, cursor)

                    if not game_id:
                        logger.warning(f"  Game not found: {away_team_name} @ {home_team_name}")
                        continue

                    stats['games_matched'] += 1

                    # Save odds to database
                    odds_data = {
                        'game_id': game_id,
                        'source': 'TheOddsAPI',
                        'opening_spread_home': consensus.get('spread_home'),
                        'opening_spread_away': consensus.get('spread_away'),
                        'opening_total': consensus.get('total'),
                        'timestamp': consensus.get('commence_time')
                    }

                    # Insert or update
                    cursor.execute('''
                        INSERT OR REPLACE INTO game_odds
                        (game_id, source, opening_spread_home, opening_spread_away,
                         opening_total, timestamp, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    ''', (
                        odds_data['game_id'],
                        odds_data['source'],
                        odds_data['opening_spread_home'],
                        odds_data['opening_spread_away'],
                        odds_data['opening_total'],
                        odds_data['timestamp']
                    ))

                    stats['odds_saved'] += 1

                    # Show progress
                    spread = consensus.get('spread_home', 'N/A')
                    total = consensus.get('total', 'N/A')
                    print(f"  ✓ {away_team_name} @ {home_team_name}: {spread}/{total}")

                except Exception as e:
                    logger.error(f"  Error processing game: {e}")
                    continue

            # Commit after each date
            conn.commit()

            # Rate limiting - be nice to the API
            time.sleep(2)

        except Exception as e:
            logger.error(f"Error fetching odds for {date_str}: {e}")
            continue

    conn.close()

    # Final summary
    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE!")
    print("=" * 80)
    print(f"\nDates processed: {stats['dates_processed']}")
    print(f"Games found in API: {stats['games_found']}")
    print(f"Games matched to ESPN: {stats['games_matched']}")
    print(f"Odds records saved: {stats['odds_saved']}")
    print(f"API requests used: {stats['requests_used']}")

    remaining_after = scraper.get_remaining_requests()
    print(f"\nRequests remaining: {remaining_after}")
    print(f"Requests used: {remaining_before - remaining_after}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    backfill_historical_odds()
