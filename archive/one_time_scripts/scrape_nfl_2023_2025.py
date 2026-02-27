"""
Comprehensive NFL Data Collection for 2023-2025
Scrapes ESPN stats and The Odds API historical odds
"""
from nfl_espn_scraper import NFLESPNScraper
from nfl_odds_api_scraper import NFLOddsAPIScraper
from datetime import datetime, timedelta
import time


def scrape_nfl_espn_data():
    """
    Scrape NFL game data from ESPN for 2023-2025
    """
    print("\n" + "="*80)
    print("SCRAPING NFL DATA FROM ESPN (2023-2025)")
    print("="*80 + "\n")

    scraper = NFLESPNScraper(db_path='nfl_games.db')

    # === 2023 Season ===
    print("\n📅 2023 NFL Season")
    print("-"*60)

    # 2023 Regular Season (Weeks 1-18)
    print("\n🏈 2023 Regular Season (Weeks 1-18)")
    scraper.scrape_season(
        season=2023,
        start_week=1,
        end_week=18,
        season_type=2  # Regular season
    )

    # 2023 Postseason (Wild Card, Divisional, Conference, Super Bowl)
    print("\n🏆 2023 Postseason")
    scraper.scrape_season(
        season=2023,
        start_week=1,
        end_week=4,
        season_type=3  # Postseason
    )

    # === 2024 Season ===
    print("\n📅 2024 NFL Season")
    print("-"*60)

    # 2024 Regular Season (Weeks 1-18)
    print("\n🏈 2024 Regular Season (Weeks 1-18)")
    scraper.scrape_season(
        season=2024,
        start_week=1,
        end_week=18,
        season_type=2  # Regular season
    )

    # 2024 Postseason
    print("\n🏆 2024 Postseason")
    scraper.scrape_season(
        season=2024,
        start_week=1,
        end_week=4,
        season_type=3  # Postseason
    )

    # === 2025 Season (Current) ===
    print("\n📅 2025 NFL Season (In Progress)")
    print("-"*60)

    # 2025 Regular Season (up to current week)
    print("\n🏈 2025 Regular Season (Weeks 1-current)")
    scraper.scrape_season(
        season=2025,
        start_week=1,
        end_week=18,  # Will only scrape completed weeks
        season_type=2  # Regular season
    )

    print("\n" + "="*80)
    print("ESPN SCRAPE COMPLETE!")
    print("="*80 + "\n")

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
    print(f"Total Team Stats Records: {cursor.fetchone()[0]}")

    conn.close()


def get_nfl_week_dates(season: int, week: int) -> str:
    """
    Get approximate date for a given NFL season and week
    Returns ISO format date string for odds API

    Args:
        season: Year (e.g., 2023)
        week: Week number (1-18 for regular, 19-22 for postseason)

    Returns:
        ISO date string (e.g., '2023-09-10T12:00:00Z')
    """
    # NFL season typically starts first Thursday after Labor Day (early September)
    # Approximate start dates
    season_starts = {
        2023: datetime(2023, 9, 7),   # 2023 season started Sept 7
        2024: datetime(2024, 9, 5),   # 2024 season started Sept 5
        2025: datetime(2025, 9, 4),   # 2025 estimated
    }

    start_date = season_starts.get(season, datetime(season, 9, 7))

    # Each week is 7 days apart
    # Regular season: weeks 1-18
    # Postseason: weeks 19-22 (Wild Card, Divisional, Conference, Super Bowl)
    week_date = start_date + timedelta(weeks=week-1)

    # Set to Sunday noon (most games)
    week_date = week_date.replace(hour=12, minute=0, second=0, microsecond=0)

    return week_date.strftime('%Y-%m-%dT%H:%M:%SZ')


def scrape_nfl_odds_data(api_key: str, max_requests: int = 2000):
    """
    Scrape historical NFL odds from The Odds API

    Args:
        api_key: The Odds API key
        max_requests: Maximum number of requests to use (default: 2000)

    Strategy:
    - Each historical request costs ~10 credits
    - We'll sample key dates throughout each season
    - Focus on regular season + playoffs
    """
    print("\n" + "="*80)
    print("SCRAPING NFL ODDS FROM THE ODDS API (2023-2025)")
    print("="*80 + "\n")

    scraper = NFLOddsAPIScraper(api_key=api_key, db_path='nfl_games.db')

    # Check available credits
    remaining = scraper.get_remaining_requests()
    if remaining is None:
        print("⚠️  Could not check remaining requests")
    elif remaining < 100:
        print(f"⚠️  WARNING: Only {remaining} requests remaining!")
        print("Consider reducing the scope of the scrape")
        return

    print(f"\n📊 Starting Credits: ~{remaining}")
    print(f"📊 Target Usage: {max_requests} requests")
    print(f"📊 Estimated Final Balance: ~{remaining - max_requests if remaining else 'Unknown'}\n")

    requests_used = 0
    markets = ['spreads', 'totals']  # 2 markets = ~20 requests per date

    # === 2023 Season ===
    print("\n📅 2023 Season Odds")
    print("-"*60)

    # Sample every 2 weeks of regular season
    for week in range(1, 19, 2):  # Weeks 1, 3, 5, ..., 17
        if requests_used >= max_requests:
            print(f"Reached request limit ({max_requests})")
            break

        date = get_nfl_week_dates(2023, week)
        print(f"\n🏈 2023 Week {week} ({date[:10]})...")

        games = scraper.fetch_historical_odds(date, markets=markets)

        for game in games:
            parsed = scraper.parse_odds_data(game)
            consensus = scraper.get_consensus_odds(parsed)
            if consensus:
                scraper.save_odds_to_database(consensus)

        requests_used += 20  # Approximate
        time.sleep(2)  # Rate limiting

    # 2023 Playoffs
    for week in [19, 20, 21, 22]:  # Wild Card, Divisional, Conference, Super Bowl
        if requests_used >= max_requests:
            break

        date = get_nfl_week_dates(2023, week)
        print(f"\n🏆 2023 Playoff Week {week-18} ({date[:10]})...")

        games = scraper.fetch_historical_odds(date, markets=markets)

        for game in games:
            parsed = scraper.parse_odds_data(game)
            consensus = scraper.get_consensus_odds(parsed)
            if consensus:
                scraper.save_odds_to_database(consensus)

        requests_used += 20
        time.sleep(2)

    # === 2024 Season ===
    print("\n📅 2024 Season Odds")
    print("-"*60)

    # Sample every 2 weeks
    for week in range(1, 19, 2):
        if requests_used >= max_requests:
            break

        date = get_nfl_week_dates(2024, week)
        print(f"\n🏈 2024 Week {week} ({date[:10]})...")

        games = scraper.fetch_historical_odds(date, markets=markets)

        for game in games:
            parsed = scraper.parse_odds_data(game)
            consensus = scraper.get_consensus_odds(parsed)
            if consensus:
                scraper.save_odds_to_database(consensus)

        requests_used += 20
        time.sleep(2)

    # 2024 Playoffs (if completed)
    for week in [19, 20, 21, 22]:
        if requests_used >= max_requests:
            break

        date = get_nfl_week_dates(2024, week)
        if datetime.now() < datetime.fromisoformat(date.replace('Z', '+00:00')):
            print(f"\n⏭️  Skipping future week {week-18}")
            continue

        print(f"\n🏆 2024 Playoff Week {week-18} ({date[:10]})...")

        games = scraper.fetch_historical_odds(date, markets=markets)

        for game in games:
            parsed = scraper.parse_odds_data(game)
            consensus = scraper.get_consensus_odds(parsed)
            if consensus:
                scraper.save_odds_to_database(consensus)

        requests_used += 20
        time.sleep(2)

    # === 2025 Season (Current) ===
    print("\n📅 2025 Season Odds")
    print("-"*60)

    # Sample recent weeks only (odds API has current odds for free)
    for week in range(1, 13, 2):  # Up to week 12
        if requests_used >= max_requests:
            break

        date = get_nfl_week_dates(2025, week)
        if datetime.now() < datetime.fromisoformat(date.replace('Z', '+00:00')):
            print(f"\n⏭️  Skipping future week {week}")
            break

        print(f"\n🏈 2025 Week {week} ({date[:10]})...")

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
    print("="*80 + "\n")

    # Check final balance
    final_remaining = scraper.get_remaining_requests()
    if final_remaining:
        print(f"📊 Final Credits Remaining: ~{final_remaining}")
        print(f"📊 Estimated Used: ~{requests_used}")

    # Summary
    import sqlite3
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM game_odds')
    odds_count = cursor.fetchone()[0]
    print(f"\nTotal Games with Odds: {odds_count}")

    conn.close()


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("NFL DATA COLLECTION: 2023-2025 SEASONS")
    print("="*80)
    print("\nThis script will:")
    print("  1. Scrape all NFL games from ESPN (2023-2025)")
    print("  2. Scrape historical odds from The Odds API")
    print("\nEstimated Time:")
    print("  - ESPN: ~45-60 minutes (3 seasons)")
    print("  - Odds API: ~30-45 minutes (~1,000-2,000 requests)")
    print("  - Total: ~1.5-2 hours")
    print("\nPress Ctrl+C to cancel\n")
    print("="*80 + "\n")

    # Step 1: Scrape ESPN data
    input("Press Enter to start ESPN scrape...")
    scrape_nfl_espn_data()

    # Step 2: Scrape Odds data
    print("\n" + "="*80)
    print("STEP 2: ODDS DATA")
    print("="*80 + "\n")

    api_key = input("Enter your Odds API key (or press Enter to skip): ").strip()

    if api_key:
        max_requests = input("Max requests to use (default 2000, press Enter to accept): ").strip()
        max_requests = int(max_requests) if max_requests else 2000

        scrape_nfl_odds_data(api_key, max_requests=max_requests)
    else:
        print("\n⏭️  Skipping odds scrape (no API key provided)")

    print("\n" + "="*80)
    print("NFL DATA COLLECTION COMPLETE!")
    print("="*80)
    print("\nDatabase: nfl_games.db")
    print("\nYou now have:")
    print("  ✅ 3 seasons of NFL games (2023-2025)")
    print("  ✅ Detailed statistics for each game")
    print("  ✅ Historical odds data")
    print("\nReady for Deep-Eagle training!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
