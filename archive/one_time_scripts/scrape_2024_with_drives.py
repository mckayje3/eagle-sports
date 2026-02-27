"""
Scrape 2024 CFB season with drive and weather data
"""
from espn_scraper import ESPNScraper
import time

def main():
    print("=" * 80)
    print("SCRAPING 2024 CFB SEASON - WITH DRIVE & WEATHER DATA")
    print("=" * 80)

    scraper = ESPNScraper('cfb_games.db')

    # Scrape all weeks of 2024 regular season
    start_week = 1
    end_week = 15

    print(f"\nStarting scrape for 2024 season (weeks {start_week}-{end_week})")
    print("This will populate drive data for all completed games")
    print("=" * 80)

    for week in range(start_week, end_week + 1):
        scraper.scrape_week(season=2024, week=week, season_type=2)
        time.sleep(1)  # Rate limiting

    print("\n" + "=" * 80)
    print("2024 CFB SEASON SCRAPE COMPLETE!")
    print("=" * 80)

    # Check results
    import sqlite3
    conn = sqlite3.connect('cfb_games.db')
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

    print(f"\nRESULTS:")
    print(f"  Total games (2024): {total_games}")
    print(f"  Completed games: {completed_games}")
    print(f"  Games with drive data: {games_with_drives}")
    print(f"  Total drives recorded: {total_drives}")
    print(f"  Games with dome data: {games_with_dome_data}")

    conn.close()

if __name__ == '__main__':
    main()
