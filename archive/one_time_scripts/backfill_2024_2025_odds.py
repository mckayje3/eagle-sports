"""
Comprehensive Historical Odds Backfill for 2024 + 2025 Seasons
Fetches ALL game dates with spreads and totals
Estimated: ~2,720 requests for complete coverage
"""
import json
import sqlite3
import time
from datetime import datetime
from odds_api_scraper import OddsAPIScraper
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_all_game_dates(seasons):
    """Get ALL unique game dates from specified seasons"""
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    placeholders = ','.join('?' * len(seasons))
    cursor.execute(f'''
        SELECT season, DATE(date) as game_date, COUNT(*) as num_games
        FROM games
        WHERE season IN ({placeholders})
        GROUP BY season, DATE(date)
        ORDER BY season, game_date
    ''', seasons)

    dates = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
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
    # Try exact match on display_name first
    cursor.execute('''
        SELECT team_id, school_name, display_name
        FROM teams
        WHERE display_name = ? OR school_name = ?
    ''', (api_team_name, api_team_name))

    result = cursor.fetchone()
    if result:
        logger.debug(f"Exact match: '{api_team_name}' -> {result[1]} (ID: {result[0]})")
        return result[0]

    # Try partial match - The Odds API format is "School Mascot" (e.g., "Alabama Crimson Tide")
    # We want to match the school part
    cursor.execute('''
        SELECT team_id, school_name, display_name
        FROM teams
        WHERE ? LIKE '%' || school_name || '%'
        ORDER BY LENGTH(school_name) DESC
        LIMIT 1
    ''', (api_team_name,))

    result = cursor.fetchone()
    if result:
        logger.debug(f"Partial match: '{api_team_name}' -> {result[1]} (ID: {result[0]})")
        return result[0]

    # Try fuzzy match with individual words
    words = api_team_name.split()
    for word in words:
        if len(word) > 3:  # Skip short words
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


def find_game_id(home_team_id, away_team_id, game_date, season, cursor):
    """Find ESPN game_id for a matchup on a specific date"""
    cursor.execute('''
        SELECT game_id
        FROM games
        WHERE home_team_id = ? AND away_team_id = ?
        AND DATE(date) = ?
        AND season = ?
    ''', (home_team_id, away_team_id, game_date, season))

    result = cursor.fetchone()
    return result[0] if result else None


def backfill_historical_odds():
    """Main function to backfill 2024 + 2025 historical odds"""

    print("=" * 80)
    print("2024 + 2025 COMPLETE HISTORICAL ODDS BACKFILL")
    print("=" * 80)

    # Load API key
    with open('odds_api_config.json', 'r') as f:
        config = json.load(f)

    scraper = OddsAPIScraper(api_key=config['api_key'])

    # Check remaining requests
    remaining_before = scraper.get_remaining_requests()
    print(f"\nRequests available: {remaining_before:,}")

    # Get all game dates for 2024 and 2025
    game_dates = get_all_game_dates([2024, 2025])

    # Summary by season
    dates_2024 = [d for d in game_dates if d[0] == 2024]
    dates_2025 = [d for d in game_dates if d[0] == 2025]
    games_2024 = sum(d[2] for d in dates_2024)
    games_2025 = sum(d[2] for d in dates_2025)

    print(f"\n2024 Season: {len(dates_2024)} dates, {games_2024} games")
    print(f"2025 Season: {len(dates_2025)} dates, {games_2025} games")
    print(f"TOTAL: {len(game_dates)} dates, {games_2024 + games_2025} games")

    estimated_cost = len(game_dates) * 20  # 20 requests per date (spreads + totals)
    print(f"\nEstimated cost: {estimated_cost:,} requests")
    print(f"Expected remaining: {remaining_before - estimated_cost:,} requests")

    # Auto-proceed (confirmation bypassed for non-interactive execution)
    print("\n" + "=" * 80)
    print("Auto-proceeding with fetch...")
    # response = input("Proceed with fetching? (yes/no): ")
    # if response.lower() != 'yes':
    #     print("Cancelled.")
    #     return

    print("\n" + "=" * 80)
    print("FETCHING HISTORICAL ODDS")
    print("=" * 80 + "\n")

    # Connect to database
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    stats = {
        'dates_processed': 0,
        'games_api': 0,
        'games_matched': 0,
        'odds_saved': 0,
        'requests_used': 0,
        'by_season': {2024: 0, 2025: 0}
    }

    for season, date_str, num_games in game_dates:
        stats['dates_processed'] += 1

        print(f"\n[{stats['dates_processed']}/{len(game_dates)}] {season} - {date_str} ({num_games} games)")
        print("-" * 60)

        # Convert date to ISO format for API
        iso_date = f"{date_str}T12:00:00Z"

        try:
            # Fetch historical odds (20 requests: spreads + totals)
            games_data = scraper.fetch_historical_odds(
                date=iso_date,
                markets=['spreads', 'totals'],
                regions='us'
            )

            stats['requests_used'] += 20
            stats['games_api'] += len(games_data)

            if not games_data:
                print(f"  No games returned from API")
                time.sleep(1)
                continue

            print(f"  Retrieved {len(games_data)} games from API")

            # Process each game
            for game_data in games_data:
                try:
                    # Parse odds data
                    parsed = scraper.parse_odds_data(game_data)
                    consensus = scraper.get_consensus_odds(parsed)

                    home_team_name = consensus.get('home_team')
                    away_team_name = consensus.get('away_team')

                    if not home_team_name or not away_team_name:
                        continue

                    # Match teams to ESPN IDs
                    home_id = match_team_to_espn(home_team_name, cursor)
                    away_id = match_team_to_espn(away_team_name, cursor)

                    if not home_id or not away_id:
                        logger.warning(f"  ✗ No match: {away_team_name} @ {home_team_name}")
                        continue

                    # Find the ESPN game_id
                    game_id = find_game_id(home_id, away_id, date_str, season, cursor)

                    if not game_id:
                        logger.warning(f"  ✗ Game not in DB: {away_team_name} @ {home_team_name}")
                        continue

                    stats['games_matched'] += 1
                    stats['by_season'][season] += 1

                    # Save odds to database
                    spread_home = consensus.get('spread_home')
                    spread_away = consensus.get('spread_away')
                    total = consensus.get('total')

                    cursor.execute('''
                        INSERT OR REPLACE INTO game_odds
                        (game_id, source, opening_spread_home, opening_spread_away,
                         opening_total, timestamp, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    ''', (
                        game_id,
                        'TheOddsAPI',
                        spread_home,
                        spread_away,
                        total,
                        consensus.get('commence_time')
                    ))

                    stats['odds_saved'] += 1

                    # Show progress
                    spread_str = f"{spread_home:+.1f}" if spread_home else "N/A"
                    total_str = f"{total:.1f}" if total else "N/A"
                    print(f"  ✓ {away_team_name} @ {home_team_name}: {spread_str} / {total_str}")

                except Exception as e:
                    logger.error(f"  Error processing game: {e}")
                    continue

            # Commit after each date
            conn.commit()

            # Rate limiting
            time.sleep(1.5)

        except Exception as e:
            logger.error(f"Error fetching odds for {date_str}: {e}")
            time.sleep(2)
            continue

    conn.close()

    # Final summary
    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE!")
    print("=" * 80)
    print(f"\nDates processed: {stats['dates_processed']}")
    print(f"Games from API: {stats['games_api']}")
    print(f"Games matched: {stats['games_matched']}")
    print(f"  - 2024: {stats['by_season'][2024]}")
    print(f"  - 2025: {stats['by_season'][2025]}")
    print(f"Odds records saved: {stats['odds_saved']}")
    print(f"API requests used: {stats['requests_used']:,}")

    remaining_after = scraper.get_remaining_requests()
    print(f"\nRequests remaining: {remaining_after:,}")
    print(f"Requests consumed: {remaining_before - remaining_after:,}")

    print("\n" + "=" * 80)
    print("You now have comprehensive odds data for:")
    print("  - 2024 season spreads & totals")
    print("  - 2025 season spreads & totals")
    print("\nReady for model training with Vegas benchmarks!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    backfill_historical_odds()
