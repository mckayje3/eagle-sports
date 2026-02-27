"""
Backfill drive data for NFL 2023 and 2024 seasons
"""
from nfl_espn_scraper import NFLESPNScraper
import sqlite3
import time

def main():
    print("=" * 80)
    print("BACKFILLING NFL 2023-2024 DRIVE DATA")
    print("=" * 80)

    # Check current state
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    # Only get games that don't already have drive data
    cursor.execute('''
        SELECT g.game_id, g.season, g.week
        FROM games g
        LEFT JOIN drives d ON g.game_id = d.game_id
        WHERE g.season IN (2023, 2024) AND g.completed = 1 AND d.game_id IS NULL
        ORDER BY g.season, g.week, g.game_id
    ''')
    games = cursor.fetchall()

    print(f"\nGames needing drive data: {len(games)}")
    conn.close()

    if len(games) == 0:
        print("No games need drive data backfill!")
        return

    # Initialize scraper and connect to database
    scraper = NFLESPNScraper('nfl_games.db')
    scraper.db.connect()

    print(f"\nProcessing {len(games)} games from 2023-2024...")

    for idx, (game_id, season, week) in enumerate(games):
        if (idx + 1) % 25 == 0:
            print(f"  Progress: {idx + 1}/{len(games)} games processed")

        try:
            # Fetch game details which includes drive data
            scraper.process_game_details(str(game_id), season)
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"  Error processing game {game_id}: {e}")
            continue

    scraper.db.close()

    # Verify results
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    for season in [2023, 2024]:
        cursor.execute('SELECT COUNT(DISTINCT game_id) FROM drives WHERE game_id IN (SELECT game_id FROM games WHERE season = ?)', (season,))
        games_with_drives = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM drives WHERE game_id IN (SELECT game_id FROM games WHERE season = ?)', (season,))
        total_drives = cursor.fetchone()[0]

        print(f"\nNFL {season} Results:")
        print(f"  Games with drives: {games_with_drives}")
        print(f"  Total drives: {total_drives}")
        if games_with_drives > 0:
            print(f"  Average drives per game: {total_drives / games_with_drives:.1f}")

    conn.close()

    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
