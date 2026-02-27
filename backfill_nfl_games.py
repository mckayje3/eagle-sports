"""
Backfill NFL games from ESPN for 2022-2023 seasons.
Uses date ranges to properly fetch historical data (ESPN API year param doesn't work for old seasons).
"""
import sys
import time
import os
from datetime import datetime, timedelta

# Change to project directory
os.chdir('C:/Users/jbeast/documents/coding/sports')
sys.path.insert(0, 'C:/Users/jbeast/documents/coding/sports')

from nfl_espn_scraper import NFLESPNScraper

def get_nfl_season_dates(season):
    """
    Get the start date and week dates for an NFL season.
    NFL regular season typically starts the Thursday after Labor Day.
    """
    # NFL season start dates (Thursday after Labor Day)
    season_starts = {
        2022: datetime(2022, 9, 8),   # 2022 season started Sep 8
        2023: datetime(2023, 9, 7),   # 2023 season started Sep 7
        2024: datetime(2024, 9, 5),   # 2024 season started Sep 5
    }
    return season_starts.get(season, datetime(season, 9, 7))

def scrape_by_dates(scraper, season, start_date, num_days=7, is_postseason=False):
    """Scrape games by date range"""
    end_date = start_date + timedelta(days=num_days - 1)
    date_range = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

    url = f"{scraper.BASE_URL}/scoreboard"
    params = {'dates': date_range, 'limit': 100}

    try:
        response = scraper.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching {date_range}: {e}")
        return 0

    events = data.get('events', [])
    if not events:
        return 0

    print(f"  Found {len(events)} games for {date_range}")

    for event in events:
        # Determine postseason type from event data
        season_type_data = event.get('season', {})
        season_type = season_type_data.get('type', 2)

        # Parse game data with correct season
        game_data = scraper.parse_game_data(event, season, season_type)

        # Save teams
        competitions = event.get('competitions', [{}])[0]
        for competitor in competitions.get('competitors', []):
            team_data = scraper.parse_team_data(competitor.get('team', {}))
            if team_data['team_id'] and team_data['name']:
                scraper.db.insert_or_update_team(team_data)

        # Save game
        scraper.db.insert_or_update_game(game_data)

        # Fetch detailed stats if game is completed
        if game_data['completed']:
            game_id = game_data['game_id']
            scraper.process_game_details(str(game_id), season)
            time.sleep(0.3)

    return len(events)

def backfill_season(scraper, season):
    """Backfill a complete season using date ranges"""
    print(f"\n{'='*60}")
    print(f"BACKFILLING {season} NFL SEASON (using dates)")
    print(f"{'='*60}")

    season_start = get_nfl_season_dates(season)
    total_games = 0

    # Regular season: 18 weeks
    print(f"\n[{season}] Regular Season...")
    for week in range(18):
        week_start = season_start + timedelta(days=week * 7)
        print(f"Week {week + 1}: {week_start.strftime('%Y-%m-%d')}")
        games = scrape_by_dates(scraper, season, week_start, num_days=7)
        total_games += games
        time.sleep(0.5)

    # Postseason starts about 22 weeks after season start
    # Wild Card: ~Jan 13-15, Divisional: ~Jan 20-21, Conference: ~Jan 28, Super Bowl: ~Feb 11
    print(f"\n[{season}] Postseason...")

    # For the postseason, we need to go to the next calendar year
    next_year = season + 1

    # Wild Card Weekend (mid-January)
    wild_card_start = datetime(next_year, 1, 13)
    print(f"Wild Card: {wild_card_start.strftime('%Y-%m-%d')}")
    games = scrape_by_dates(scraper, season, wild_card_start, num_days=4, is_postseason=True)
    total_games += games
    time.sleep(0.5)

    # Divisional Round
    divisional_start = datetime(next_year, 1, 20)
    print(f"Divisional: {divisional_start.strftime('%Y-%m-%d')}")
    games = scrape_by_dates(scraper, season, divisional_start, num_days=2, is_postseason=True)
    total_games += games
    time.sleep(0.5)

    # Conference Championships
    conference_start = datetime(next_year, 1, 27)
    print(f"Conference: {conference_start.strftime('%Y-%m-%d')}")
    games = scrape_by_dates(scraper, season, conference_start, num_days=2, is_postseason=True)
    total_games += games
    time.sleep(0.5)

    # Super Bowl (first or second Sunday in February)
    super_bowl_start = datetime(next_year, 2, 10)
    print(f"Super Bowl: {super_bowl_start.strftime('%Y-%m-%d')}")
    games = scrape_by_dates(scraper, season, super_bowl_start, num_days=3, is_postseason=True)
    total_games += games

    print(f"\n{season} season complete! Total games: {total_games}")
    return total_games

def main():
    scraper = NFLESPNScraper('nfl_games.db')

    # Initialize database connection
    scraper.db.connect()
    scraper.db.initialize_schema()

    cursor = scraper.db.conn.cursor()

    # Check current state
    cursor.execute("SELECT season, COUNT(*) FROM games GROUP BY season ORDER BY season")
    print("Current game counts:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} games")

    # Backfill 2022 season
    backfill_season(scraper, 2022)

    # Backfill 2023 season
    backfill_season(scraper, 2023)

    scraper.db.conn.commit()

    # Print summary
    cursor.execute("SELECT season, COUNT(*) FROM games GROUP BY season ORDER BY season")
    print("\n" + "="*60)
    print("FINAL GAME COUNTS BY SEASON:")
    print("="*60)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} games")

    # Print playoff game counts
    cursor.execute("SELECT season, COUNT(*) FROM games WHERE postseason_type IS NOT NULL GROUP BY season ORDER BY season")
    print("\nPlayoff games by season:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} playoff games")

    # Total with stats
    cursor.execute("SELECT COUNT(*) FROM team_game_stats")
    print(f"\nTotal team game stats records: {cursor.fetchone()[0]}")

    scraper.db.conn.close()
    print("\nBackfill complete!")

if __name__ == '__main__':
    main()
