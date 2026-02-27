"""
Fix NFL 2025 team_game_stats by re-scraping completed games
This will populate the missing points, passing_completions, etc.
"""
from nfl_espn_scraper import NFLESPNScraper
import sqlite3
import time

def main():
    print("=" * 80)
    print("FIXING NFL 2025 TEAM STATS")
    print("=" * 80)

    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    # Get all completed 2025 games
    cursor.execute('''
        SELECT game_id, week
        FROM games
        WHERE season = 2025 AND completed = 1
        ORDER BY week, game_id
    ''')
    games = cursor.fetchall()

    print(f"\nFound {len(games)} completed games to re-scrape")
    conn.close()

    # Initialize scraper
    scraper = NFLESPNScraper(db_path='nfl_games.db')

    # Ensure database connection is initialized
    if not scraper.db.conn:
        scraper.db.connect()
        scraper.db.initialize_schema()

    # Re-process each game to update team stats
    for idx, (game_id, week) in enumerate(games, 1):
        print(f"[{idx}/{len(games)}] Re-scraping game {game_id} (Week {week})...")
        try:
            scraper.process_game_details(str(game_id), 2025)
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Verify the fix
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT COUNT(*) as total, COUNT(points) as with_points
        FROM team_game_stats ts
        JOIN games g ON ts.game_id = g.game_id
        WHERE g.season = 2025
    ''')
    result = cursor.fetchone()

    print("\n" + "=" * 80)
    print("FIX COMPLETE!")
    print("=" * 80)
    print(f"Team game stats: {result[1]}/{result[0]} now have points data")
    print(f"Coverage: {100 * result[1] / max(result[0], 1):.1f}%")

    conn.close()

if __name__ == '__main__':
    main()
