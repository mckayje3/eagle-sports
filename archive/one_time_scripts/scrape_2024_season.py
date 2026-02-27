"""
Scrape complete 2024 college football season from ESPN
This includes all games, scores, and detailed statistics
"""
from espn_scraper import ESPNScraper
import time

def scrape_2024_season():
    """Scrape the entire 2024 season"""
    print("\n" + "="*80)
    print("SCRAPING 2024 COLLEGE FOOTBALL SEASON")
    print("="*80 + "\n")

    scraper = ESPNScraper(db_path='cfb_games.db')

    # Scrape regular season (weeks 1-15)
    print("Scraping 2024 Regular Season (Weeks 1-15)")
    print("This will take several minutes...\n")

    scraper.scrape_season(
        season=2024,
        start_week=1,
        end_week=15,
        season_type=2  # Regular season
    )

    print("\n" + "="*80)
    print("Scraping 2024 Postseason (Conference Championships & Bowls)")
    print("="*80 + "\n")

    # Scrape postseason (conference championships, bowls, playoff)
    scraper.scrape_season(
        season=2024,
        start_week=1,
        end_week=10,  # Postseason weeks
        season_type=3  # Postseason
    )

    print("\n" + "="*80)
    print("2024 SEASON SCRAPE COMPLETE!")
    print("="*80 + "\n")

    # Show summary
    import sqlite3
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM games WHERE season=2024')
    total_games = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM games WHERE season=2024 AND completed=1')
    completed_games = cursor.fetchone()[0]

    cursor.execute('''
        SELECT COUNT(DISTINCT g.game_id)
        FROM games g
        JOIN team_game_stats ts ON g.game_id = ts.game_id
        WHERE g.season=2024
    ''')
    games_with_stats = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(DISTINCT team_id) FROM teams')
    total_teams = cursor.fetchone()[0]

    conn.close()

    print(f"Total 2024 games: {total_games}")
    print(f"Completed games: {completed_games}")
    print(f"Games with detailed stats: {games_with_stats}")
    print(f"Total teams in database: {total_teams}")
    print("\n" + "="*80 + "\n")

    print("Next step: Scrape Vegas odds for 2024 season")
    print("See: backfill_historical_odds.py")
    print("Note: This requires The Odds API key and costs API requests\n")


if __name__ == '__main__':
    scrape_2024_season()
