"""
Scrape 2025 NFL season with drive data
"""
from nfl_espn_scraper import NFLESPNScraper
import sqlite3
import time

def main():
    print("=" * 80)
    print("SCRAPING 2025 NFL SEASON WITH DRIVE DATA")
    print("=" * 80)

    # Verify current database state
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM games WHERE season = 2025')
    existing_2025 = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM games WHERE season = 2024')
    existing_2024 = cursor.fetchone()[0]

    print(f"\nCurrent database state:")
    print(f"  2024 games: {existing_2024}")
    print(f"  2025 games: {existing_2025}")

    conn.close()

    # Initialize scraper
    scraper = NFLESPNScraper(db_path='nfl_games.db')

    # Scrape 2025 season (weeks 1-18)
    # NFL 2025 season started in September, currently in November (Week ~12)
    print(f"\nScraping 2025 NFL season (weeks 1-18)...")
    print("This will take several minutes...\n")

    for week in range(1, 19):
        print(f"\nScraping week {week}...")
        scraper.scrape_week(season=2025, week=week, season_type=2)
        time.sleep(1)  # Rate limiting

    # Verify results
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM games WHERE season = 2025')
    total_2025 = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM games WHERE season = 2025 AND completed = 1')
    completed_2025 = cursor.fetchone()[0]

    cursor.execute('SELECT MIN(week), MAX(week) FROM games WHERE season = 2025 AND completed = 1')
    week_range = cursor.fetchone()

    cursor.execute('''
        SELECT COUNT(DISTINCT game_id)
        FROM drives
        WHERE game_id IN (SELECT game_id FROM games WHERE season = 2025)
    ''')
    games_with_drives_2025 = cursor.fetchone()[0]

    cursor.execute('''
        SELECT COUNT(*)
        FROM drives
        WHERE game_id IN (SELECT game_id FROM games WHERE season = 2025)
    ''')
    total_drives_2025 = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM games WHERE season = 2024')
    total_2024 = cursor.fetchone()[0]

    print("\n" + "=" * 80)
    print("SCRAPING COMPLETE!")
    print("=" * 80)

    print(f"\nFinal database state:")
    print(f"  2024 games: {total_2024} (preserved)")
    print(f"  2025 games: {total_2025}")
    print(f"  2025 completed: {completed_2025}")
    print(f"  2025 week range: {week_range[0]}-{week_range[1]}")
    print(f"  2025 games with drives: {games_with_drives_2025}")
    print(f"  2025 total drives: {total_drives_2025}")

    if total_drives_2025 > 0:
        avg_drives = total_drives_2025 / max(games_with_drives_2025, 1)
        print(f"  Average drives per game: {avg_drives:.1f}")

    conn.close()

if __name__ == '__main__':
    main()
