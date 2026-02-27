"""
Scrape 2025 college football season from ESPN
This includes all games through the current week
"""
from espn_scraper import ESPNScraper
import time

def scrape_2025_season():
    """Scrape the 2025 season (through current week)"""
    print("\n" + "="*80)
    print("SCRAPING 2025 COLLEGE FOOTBALL SEASON")
    print("="*80 + "\n")

    scraper = ESPNScraper(db_path='cfb_games.db')

    # Scrape regular season (weeks 1-15)
    # Note: Weeks without games will be skipped automatically
    print("Scraping 2025 Regular Season (Weeks 1-15)")
    print("This will take several minutes...\n")

    scraper.scrape_season(
        season=2025,
        start_week=1,
        end_week=15,
        season_type=2  # Regular season
    )

    print("\n" + "="*80)
    print("2025 SEASON SCRAPE COMPLETE!")
    print("="*80 + "\n")

    # Show summary
    import sqlite3
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM games WHERE season=2025')
    total_games = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM games WHERE season=2025 AND completed=1')
    completed_games = cursor.fetchone()[0]

    cursor.execute('''
        SELECT COUNT(DISTINCT g.game_id)
        FROM games g
        JOIN team_game_stats ts ON g.game_id = ts.game_id
        WHERE g.season=2025
    ''')
    games_with_stats = cursor.fetchone()[0]

    cursor.execute('SELECT MIN(week), MAX(week) FROM games WHERE season=2025')
    week_range = cursor.fetchone()

    cursor.execute('SELECT COUNT(*) FROM games WHERE season=2024 AND completed=1')
    games_2024 = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM games WHERE season IN (2024, 2025) AND completed=1')
    total_combined = cursor.fetchone()[0]

    conn.close()

    print(f"2025 Season:")
    print(f"  Total games: {total_games}")
    print(f"  Completed games: {completed_games}")
    print(f"  Games with detailed stats: {games_with_stats}")
    print(f"  Week range: {week_range[0]} to {week_range[1]}")

    print(f"\nCombined Training Data:")
    print(f"  2024 completed games: {games_2024}")
    print(f"  2025 completed games: {completed_games}")
    print(f"  Total completed games: {total_combined}")
    print(f"  Increase: +{games_2024} more games from 2024 season")

    print("\n" + "="*80 + "\n")

    print("Next steps:")
    print("1. Regenerate ML features: py ml_feature_extraction_v2.py")
    print("2. Retrain models: py train_score_predictor.py")
    print("3. Generate new predictions: py predict_scores.py\n")


if __name__ == '__main__':
    scrape_2025_season()
