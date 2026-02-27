"""
Backfill NFL box scores for games with missing stats
"""
import sqlite3
import time
from nfl_espn_scraper import NFLESPNScraper


def backfill_boxscores():
    """Backfill box score stats for all games with missing data"""
    print("=" * 80)
    print("NFL BOX SCORE BACKFILL")
    print("=" * 80)

    scraper = NFLESPNScraper('nfl_games.db')
    scraper.db.connect()

    # Find games with missing stats
    cursor = scraper.db.conn.cursor()

    # Get all completed games
    cursor.execute('''
        SELECT g.game_id, g.season, g.week
        FROM games g
        WHERE g.completed = 1
        ORDER BY g.season, g.week
    ''')
    all_games = cursor.fetchall()

    # Get games that have stats
    cursor.execute('''
        SELECT DISTINCT game_id FROM team_game_stats
        WHERE total_yards IS NOT NULL
    ''')
    games_with_stats = set(row[0] for row in cursor.fetchall())

    # Find games needing stats
    games_to_fetch = [(gid, s, w) for gid, s, w in all_games if gid not in games_with_stats]

    print(f"\nTotal completed games: {len(all_games)}")
    print(f"Games with stats: {len(games_with_stats)}")
    print(f"Games needing stats: {len(games_to_fetch)}")

    if not games_to_fetch:
        print("\nAll games have stats!")
        return

    print(f"\nFetching box scores for {len(games_to_fetch)} games...")

    success = 0
    failed = 0

    for i, (game_id, season, week) in enumerate(games_to_fetch, 1):
        print(f"[{i}/{len(games_to_fetch)}] Game {game_id} ({season} Week {week})...", end=" ")

        try:
            scraper.process_game_details(str(game_id), season)

            # Check if stats were added
            cursor.execute('''
                SELECT COUNT(*) FROM team_game_stats
                WHERE game_id = ? AND total_yards IS NOT NULL
            ''', (game_id,))
            count = cursor.fetchone()[0]

            if count > 0:
                print("OK")
                success += 1
            else:
                print("No stats available")
                failed += 1

        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

        # Rate limiting
        time.sleep(0.5)

        # Commit periodically
        if i % 20 == 0:
            scraper.db.conn.commit()

    scraper.db.conn.commit()
    scraper.db.close()

    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE")
    print("=" * 80)
    print(f"Success: {success}")
    print(f"Failed/No stats: {failed}")


if __name__ == '__main__':
    backfill_boxscores()
