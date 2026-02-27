"""
Backfill NFL 2025 historical odds for missing games
Uses The Odds API historical endpoint
"""
import json
import sqlite3
import time
from datetime import datetime
from nfl_odds_api_scraper import NFLOddsAPIScraper
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_missing_game_dates():
    """Get unique dates with missing odds in NFL 2025"""
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT DISTINCT DATE(g.date) as game_date, COUNT(*) as num_missing
        FROM games g
        LEFT JOIN game_odds go ON g.game_id = go.game_id
        WHERE g.season = 2025 AND go.game_id IS NULL
        GROUP BY DATE(g.date)
        ORDER BY game_date
    ''')

    dates = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()

    return dates


def match_team_to_espn(api_team_name, cursor):
    """Match Odds API team name to ESPN team_id"""
    # NFL team name mappings (Odds API to ESPN format)
    team_mappings = {
        'Arizona Cardinals': 'Arizona Cardinals',
        'Atlanta Falcons': 'Atlanta Falcons',
        'Baltimore Ravens': 'Baltimore Ravens',
        'Buffalo Bills': 'Buffalo Bills',
        'Carolina Panthers': 'Carolina Panthers',
        'Chicago Bears': 'Chicago Bears',
        'Cincinnati Bengals': 'Cincinnati Bengals',
        'Cleveland Browns': 'Cleveland Browns',
        'Dallas Cowboys': 'Dallas Cowboys',
        'Denver Broncos': 'Denver Broncos',
        'Detroit Lions': 'Detroit Lions',
        'Green Bay Packers': 'Green Bay Packers',
        'Houston Texans': 'Houston Texans',
        'Indianapolis Colts': 'Indianapolis Colts',
        'Jacksonville Jaguars': 'Jacksonville Jaguars',
        'Kansas City Chiefs': 'Kansas City Chiefs',
        'Las Vegas Raiders': 'Las Vegas Raiders',
        'Los Angeles Chargers': 'Los Angeles Chargers',
        'Los Angeles Rams': 'Los Angeles Rams',
        'Miami Dolphins': 'Miami Dolphins',
        'Minnesota Vikings': 'Minnesota Vikings',
        'New England Patriots': 'New England Patriots',
        'New Orleans Saints': 'New Orleans Saints',
        'New York Giants': 'New York Giants',
        'New York Jets': 'New York Jets',
        'Philadelphia Eagles': 'Philadelphia Eagles',
        'Pittsburgh Steelers': 'Pittsburgh Steelers',
        'San Francisco 49ers': 'San Francisco 49ers',
        'Seattle Seahawks': 'Seattle Seahawks',
        'Tampa Bay Buccaneers': 'Tampa Bay Buccaneers',
        'Tennessee Titans': 'Tennessee Titans',
        'Washington Commanders': 'Washington Commanders',
    }

    # Map the name
    mapped_name = team_mappings.get(api_team_name, api_team_name)

    # Try exact match on display_name
    cursor.execute('''
        SELECT team_id, display_name
        FROM teams
        WHERE display_name = ?
    ''', (mapped_name,))

    result = cursor.fetchone()
    if result:
        return result[0]

    # Try partial match
    cursor.execute('''
        SELECT team_id, display_name
        FROM teams
        WHERE display_name LIKE ?
        ORDER BY LENGTH(display_name) DESC
        LIMIT 1
    ''', (f'%{api_team_name}%',))

    result = cursor.fetchone()
    if result:
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


def backfill_nfl_2025_odds():
    """Main function to backfill NFL 2025 historical odds"""

    print("=" * 80)
    print("NFL 2025 HISTORICAL ODDS BACKFILL")
    print("=" * 80)

    # Load API key
    with open('odds_api_config.json', 'r') as f:
        config = json.load(f)

    scraper = NFLOddsAPIScraper(api_key=config['api_key'])

    # Check remaining requests
    remaining_before = scraper.get_remaining_requests()
    print(f"\nRequests available: {remaining_before}")

    # Get dates with missing odds
    game_dates = get_missing_game_dates()
    print(f"\nFound {len(game_dates)} dates with missing odds")
    for date_str, num_missing in game_dates:
        print(f"  {date_str}: {num_missing} games missing")

    estimated_cost = len(game_dates) * 20  # 10 for spreads, 10 for totals
    print(f"\nEstimated cost: {estimated_cost} requests (spreads + totals)")
    print(f"Expected remaining: {remaining_before - estimated_cost} requests")

    print("\n" + "=" * 80)
    print("FETCHING HISTORICAL ODDS")
    print("=" * 80 + "\n")

    # Connect to database
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    stats = {
        'dates_processed': 0,
        'games_found': 0,
        'games_matched': 0,
        'odds_saved': 0,
        'requests_used': 0
    }

    for date_str, num_missing in game_dates:
        stats['dates_processed'] += 1

        print(f"\n[{stats['dates_processed']}/{len(game_dates)}] {date_str} ({num_missing} games missing)")
        print("-" * 60)

        # Convert date to ISO format for API
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
                        # Try swapped home/away (sometimes Odds API gets this wrong)
                        game_id = find_game_id(away_id, home_id, date_str, cursor)
                        if game_id:
                            logger.info(f"  Found game with swapped home/away")

                    if not game_id:
                        logger.warning(f"  Game not found: {away_team_name} @ {home_team_name}")
                        continue

                    stats['games_matched'] += 1

                    # Check if we already have odds for this game
                    cursor.execute('SELECT game_id FROM game_odds WHERE game_id = ?', (game_id,))
                    if cursor.fetchone():
                        logger.info(f"  Already have odds for game {game_id}, skipping")
                        continue

                    # Save odds to database
                    cursor.execute('''
                        INSERT OR REPLACE INTO game_odds
                        (game_id, source, opening_spread_home, opening_spread_away,
                         opening_total, timestamp, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    ''', (
                        game_id,
                        'TheOddsAPI',
                        consensus.get('spread_home'),
                        consensus.get('spread_away'),
                        consensus.get('total'),
                        consensus.get('commence_time')
                    ))

                    stats['odds_saved'] += 1

                    # Show progress
                    spread = consensus.get('spread_home', 'N/A')
                    total = consensus.get('total', 'N/A')
                    if spread:
                        spread = f"{spread:+.1f}"
                    if total:
                        total = f"{total:.1f}"
                    print(f"  + {away_team_name} @ {home_team_name}: spread={spread}, total={total}")

                except Exception as e:
                    logger.error(f"  Error processing game: {e}")
                    continue

            # Commit after each date
            conn.commit()

            # Rate limiting
            time.sleep(1)

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
    print(f"Actual requests used: {remaining_before - remaining_after}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    backfill_nfl_2025_odds()
