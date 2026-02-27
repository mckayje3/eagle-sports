"""
Test drive data extraction from ESPN API
"""
from espn_scraper import ESPNScraper
import sqlite3

def test_drive_extraction():
    """Test drive extraction on a completed game"""

    print("=" * 80)
    print("TESTING DRIVE DATA EXTRACTION")
    print("=" * 80)

    # Initialize scraper
    scraper = ESPNScraper('cfb_games.db')

    # Test on a single completed game
    test_game_id = '401752910'  # Recent Week 13 game (Washington vs UCLA)

    print(f"\nFetching details for game {test_game_id}...")
    scraper.db.connect()

    game_details = scraper.fetch_game_details(test_game_id)

    if game_details:
        print(f"  [OK] Game details fetched")

        # Process drive data
        print(f"\nProcessing drive data...")
        scraper.process_drive_data(int(test_game_id), game_details)

        # Check database for drive data
        print("\n" + "-" * 80)
        print("CHECKING DATABASE FOR DRIVE DATA:")
        print("-" * 80)

        cursor = scraper.db.conn.cursor()

        # Count drives for this game
        cursor.execute("SELECT COUNT(*) FROM drives WHERE game_id = ?", (test_game_id,))
        drive_count = cursor.fetchone()[0]

        print(f"\n  Drives saved: {drive_count}")

        if drive_count > 0:
            # Show sample drives
            cursor.execute("""
                SELECT
                    drive_number,
                    team_id,
                    plays,
                    yards,
                    time_elapsed_display,
                    result,
                    is_score,
                    description
                FROM drives
                WHERE game_id = ?
                ORDER BY drive_number
                LIMIT 10
            """, (test_game_id,))

            drives = cursor.fetchall()

            print(f"\n  First 10 drives:")
            print("\n  #  | Team ID | Plays | Yards | Time  | Result | Score | Description")
            print("  " + "-" * 76)

            for drive in drives:
                num, team, plays, yards, time, result, score, desc = drive
                print(f"  {num:2} | {team:7} | {plays or 'N/A':5} | {yards or 'N/A':5} | {time or 'N/A':5} | {result or 'N/A':6} | {score:5} | {desc[:30] if desc else 'N/A':30}")

            # Show statistics
            print("\n" + "-" * 80)
            print("DRIVE STATISTICS:")
            print("-" * 80)

            # Scoring drives
            cursor.execute("SELECT COUNT(*) FROM drives WHERE game_id = ? AND is_score = 1", (test_game_id,))
            scoring_drives = cursor.fetchone()[0]

            # Average yards per drive
            cursor.execute("SELECT AVG(yards) FROM drives WHERE game_id = ? AND yards IS NOT NULL", (test_game_id,))
            avg_yards = cursor.fetchone()[0]

            # Drive results breakdown
            cursor.execute("""
                SELECT result, COUNT(*) as count
                FROM drives
                WHERE game_id = ?
                GROUP BY result
                ORDER BY count DESC
            """, (test_game_id,))
            results = cursor.fetchall()

            print(f"\n  Total drives: {drive_count}")
            print(f"  Scoring drives: {scoring_drives}")
            if avg_yards:
                print(f"  Average yards per drive: {avg_yards:.1f}")
            else:
                print(f"  Average yards per drive: 0.0")

            print(f"\n  Drive results:")
            for result, count in results:
                print(f"    - {result or 'Unknown'}: {count}")

            print("\n" + "=" * 80)
            print("[OK] Drive data extraction successful!")
            print("=" * 80)
        else:
            print("\n  [ERROR] No drives were saved to database")

        scraper.db.close()
    else:
        print("  [ERROR] Failed to fetch game details")

if __name__ == '__main__':
    test_drive_extraction()
