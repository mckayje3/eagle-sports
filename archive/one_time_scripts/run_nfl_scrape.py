"""
Run NFL Data Collection for 2023-2025 (Non-Interactive)
Scrapes ESPN stats and The Odds API historical odds
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from nfl_espn_scraper import NFLESPNScraper
from nfl_odds_api_scraper import NFLOddsAPIScraper
from datetime import datetime, timedelta
import time


def get_nfl_week_dates(season: int, week: int) -> str:
    """Get approximate date for NFL season and week"""
    season_starts = {
        2023: datetime(2023, 9, 7),
        2024: datetime(2024, 9, 5),
        2025: datetime(2025, 9, 4),
    }

    start_date = season_starts.get(season, datetime(season, 9, 7))
    week_date = start_date + timedelta(weeks=week-1)
    week_date = week_date.replace(hour=12, minute=0, second=0, microsecond=0)

    return week_date.strftime('%Y-%m-%dT%H:%M:%SZ')


def scrape_espn():
    """Scrape ESPN data for 2023-2025"""
    print("\n" + "="*80)
    print("SCRAPING NFL DATA FROM ESPN (2023-2025)")
    print("="*80 + "\n")

    scraper = NFLESPNScraper(db_path='nfl_games.db')

    # 2023 Season
    print("\n📅 2023 NFL Season")
    print("-"*60)
    print("\n🏈 2023 Regular Season (Weeks 1-18)")
    scraper.scrape_season(season=2023, start_week=1, end_week=18, season_type=2)

    print("\n🏆 2023 Postseason")
    scraper.scrape_season(season=2023, start_week=1, end_week=4, season_type=3)

    # 2024 Season
    print("\n📅 2024 NFL Season")
    print("-"*60)
    print("\n🏈 2024 Regular Season (Weeks 1-18)")
    scraper.scrape_season(season=2024, start_week=1, end_week=18, season_type=2)

    print("\n🏆 2024 Postseason")
    scraper.scrape_season(season=2024, start_week=1, end_week=4, season_type=3)

    # 2025 Season
    print("\n📅 2025 NFL Season (In Progress)")
    print("-"*60)
    print("\n🏈 2025 Regular Season (Weeks 1-18)")
    scraper.scrape_season(season=2025, start_week=1, end_week=18, season_type=2)

    print("\n" + "="*80)
    print("ESPN SCRAPE COMPLETE!")
    print("="*80)

    # Summary
    import sqlite3
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    cursor.execute('SELECT season, COUNT(*) FROM games GROUP BY season ORDER BY season')
    print("\nGames by Season:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} games")

    cursor.execute('SELECT COUNT(DISTINCT team_id) FROM teams')
    print(f"\nTotal Teams: {cursor.fetchone()[0]}")

    cursor.execute('SELECT COUNT(*) FROM team_game_stats')
    print(f"Total Stats: {cursor.fetchone()[0]}")

    conn.close()


def scrape_odds(api_key: str, max_requests: int = 2000):
    """Scrape Odds API data"""
    print("\n" + "="*80)
    print("SCRAPING NFL ODDS FROM THE ODDS API (2023-2025)")
    print("="*80 + "\n")

    scraper = NFLOddsAPIScraper(api_key=api_key, db_path='nfl_games.db')

    # Check credits
    remaining = scraper.get_remaining_requests()
    if remaining:
        print(f"\n📊 Starting Credits: {remaining}")
        print(f"📊 Target Usage: {max_requests}")
        print(f"📊 Estimated Final: {remaining - max_requests}\n")

    requests_used = 0
    markets = ['spreads', 'totals']

    # 2023 Season - Sample every 2 weeks
    print("\n📅 2023 Season Odds")
    print("-"*60)

    for week in range(1, 19, 2):
        if requests_used >= max_requests:
            break

        date = get_nfl_week_dates(2023, week)
        print(f"\n🏈 2023 Week {week} ({date[:10]})")

        games = scraper.fetch_historical_odds(date, markets=markets)
        for game in games:
            parsed = scraper.parse_odds_data(game)
            consensus = scraper.get_consensus_odds(parsed)
            if consensus:
                scraper.save_odds_to_database(consensus)

        requests_used += 20
        time.sleep(2)

    # 2023 Playoffs
    for week in [19, 20, 21, 22]:
        if requests_used >= max_requests:
            break

        date = get_nfl_week_dates(2023, week)
        print(f"\n🏆 2023 Playoff Week {week-18} ({date[:10]})")

        games = scraper.fetch_historical_odds(date, markets=markets)
        for game in games:
            parsed = scraper.parse_odds_data(game)
            consensus = scraper.get_consensus_odds(parsed)
            if consensus:
                scraper.save_odds_to_database(consensus)

        requests_used += 20
        time.sleep(2)

    # 2024 Season
    print("\n📅 2024 Season Odds")
    print("-"*60)

    for week in range(1, 19, 2):
        if requests_used >= max_requests:
            break

        date = get_nfl_week_dates(2024, week)
        print(f"\n🏈 2024 Week {week} ({date[:10]})")

        games = scraper.fetch_historical_odds(date, markets=markets)
        for game in games:
            parsed = scraper.parse_odds_data(game)
            consensus = scraper.get_consensus_odds(parsed)
            if consensus:
                scraper.save_odds_to_database(consensus)

        requests_used += 20
        time.sleep(2)

    # 2024 Playoffs
    for week in [19, 20, 21, 22]:
        if requests_used >= max_requests:
            break

        date = get_nfl_week_dates(2024, week)
        if datetime.now() < datetime.fromisoformat(date.replace('Z', '+00:00')):
            continue

        print(f"\n🏆 2024 Playoff Week {week-18} ({date[:10]})")

        games = scraper.fetch_historical_odds(date, markets=markets)
        for game in games:
            parsed = scraper.parse_odds_data(game)
            consensus = scraper.get_consensus_odds(parsed)
            if consensus:
                scraper.save_odds_to_database(consensus)

        requests_used += 20
        time.sleep(2)

    # 2025 Season
    print("\n📅 2025 Season Odds")
    print("-"*60)

    for week in range(1, 13, 2):
        if requests_used >= max_requests:
            break

        date = get_nfl_week_dates(2025, week)
        if datetime.now() < datetime.fromisoformat(date.replace('Z', '+00:00')):
            break

        print(f"\n🏈 2025 Week {week} ({date[:10]})")

        games = scraper.fetch_historical_odds(date, markets=markets)
        for game in games:
            parsed = scraper.parse_odds_data(game)
            consensus = scraper.get_consensus_odds(parsed)
            if consensus:
                scraper.save_odds_to_database(consensus)

        requests_used += 20
        time.sleep(2)

    print("\n" + "="*80)
    print("ODDS SCRAPE COMPLETE!")
    print("="*80)

    # Final credits
    final = scraper.get_remaining_requests()
    if final:
        print(f"\n📊 Final Credits: {final}")
        print(f"📊 Used: ~{requests_used}")

    # Summary
    import sqlite3
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM game_odds')
    print(f"\nTotal Games with Odds: {cursor.fetchone()[0]}")
    conn.close()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("NFL DATA COLLECTION: 2023-2025 SEASONS")
    print("="*80)
    print("\nStarting automated scraping...")
    print("="*80 + "\n")

    # Step 1: ESPN
    scrape_espn()

    # Step 2: Odds API
    API_KEY = "c4d18443a6c8d011f87772fcd7c596d9"
    MAX_REQUESTS = 2000

    scrape_odds(API_KEY, MAX_REQUESTS)

    print("\n" + "="*80)
    print("NFL DATA COLLECTION COMPLETE!")
    print("="*80)
    print("\n✅ Database: nfl_games.db")
    print("✅ 3 seasons of games + odds")
    print("✅ Ready for Deep-Eagle training!")
    print("="*80 + "\n")
