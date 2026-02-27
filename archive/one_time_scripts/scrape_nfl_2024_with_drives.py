"""
Scrape 2024 NFL season with drive and weather data
"""
from nfl_espn_scraper import NFLESPNScraper
import time

def main():
    print("=" * 80)
    print("SCRAPING 2024 NFL SEASON - WITH DRIVE & WEATHER DATA")
    print("=" * 80)

    scraper = NFLESPNScraper('nfl_games.db')

    # Scrape all weeks of 2024 regular season
    start_week = 1
    end_week = 18  # NFL regular season is 18 weeks

    print(f"\nStarting scrape for 2024 season (weeks {start_week}-{end_week})")
    print("This will populate drive data for all completed games")
    print("=" * 80)

    scraper.scrape_season(season=2024, start_week=start_week, end_week=end_week, season_type=2)

    print("\n" + "=" * 80)
    print("2024 NFL SEASON SCRAPE COMPLETE!")
    print("=" * 80)

    # Check results
    import sqlite3
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM games WHERE season = 2024")
    total_games = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM games WHERE season = 2024 AND completed = 1")
    completed_games = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT game_id) FROM drives WHERE game_id IN (SELECT game_id FROM games WHERE season = 2024)")
    games_with_drives = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM drives WHERE game_id IN (SELECT game_id FROM games WHERE season = 2024)")
    total_drives = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM games WHERE season = 2024 AND is_dome IS NOT NULL")
    games_with_dome_data = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM games WHERE season = 2024 AND is_dome = 1")
    dome_games = cursor.fetchone()[0]

    print(f"\nRESULTS:")
    print(f"  Total games (2024): {total_games}")
    print(f"  Completed games: {completed_games}")
    print(f"  Games with drive data: {games_with_drives}")
    print(f"  Total drives recorded: {total_drives}")
    print(f"  Games with dome data: {games_with_dome_data}")
    print(f"  Games in domes: {dome_games}")

    if total_drives > 0:
        cursor.execute("SELECT AVG(yards) FROM drives WHERE game_id IN (SELECT game_id FROM games WHERE season = 2024)")
        avg_yards = cursor.fetchone()[0]
        print(f"  Average yards per drive: {avg_yards:.1f if avg_yards else 0}")

    conn.close()

if __name__ == '__main__':
    main()
