"""
Backfill drive data for CFB 2024 season
"""
from espn_scraper import ESPNScraper
import sqlite3
import time

def main():
    print("=" * 80)
    print("BACKFILLING CFB 2024 DRIVE DATA")
    print("=" * 80)

    # Check current state
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    # Only get games that don't already have drive data
    cursor.execute('''
        SELECT g.game_id, g.week
        FROM games g
        LEFT JOIN drives d ON g.game_id = d.game_id
        WHERE g.season = 2024 AND g.completed = 1 AND d.game_id IS NULL
        ORDER BY g.week, g.game_id
    ''')
    games = cursor.fetchall()
    games_needing_drives = len(games)

    print(f"\nGames needing drive data: {games_needing_drives}")

    conn.close()

    if games_needing_drives == 0:
        print("No games need drive data backfill!")
        return

    # Initialize scraper
    scraper = ESPNScraper('cfb_games.db')
    scraper.db.connect()

    print(f"\nProcessing {len(games)} games from 2024...")

    for idx, (game_id, week) in enumerate(games):
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(games)} games processed")

        try:
            # Fetch game details and process stats + drives
            game_details = scraper.fetch_game_details(str(game_id))
            if game_details:
                scraper.process_game_stats(game_id, game_details)
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"  Error processing game {game_id}: {e}")
            continue

    scraper.db.close()

    # Verify results
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(DISTINCT game_id) FROM drives WHERE game_id IN (SELECT game_id FROM games WHERE season = 2024)')
    games_with_drives = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM drives WHERE game_id IN (SELECT game_id FROM games WHERE season = 2024)')
    total_drives = cursor.fetchone()[0]

    conn.close()

    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE!")
    print("=" * 80)
    print(f"\nCFB 2024 Results:")
    print(f"  Games with drives: {games_with_drives}")
    print(f"  Total drives: {total_drives}")
    if games_with_drives > 0:
        print(f"  Average drives per game: {total_drives / games_with_drives:.1f}")

if __name__ == '__main__':
    main()
