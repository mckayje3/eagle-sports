"""
Backfill Vegas odds for the complete 2024 season
Uses The Odds API to fetch historical odds data
"""
import json
import time
from datetime import datetime
from odds_api_scraper import OddsAPIScraper
from game_matcher import GameMatcher
from database import FootballDatabase


# 2024 Season - All 14 weeks
DATES_2024 = [
    ('2024-08-31T17:00:00Z', '2024 Week 1 - Season Opener'),
    ('2024-09-07T17:00:00Z', '2024 Week 2'),
    ('2024-09-14T17:00:00Z', '2024 Week 3'),
    ('2024-09-21T17:00:00Z', '2024 Week 4'),
    ('2024-09-28T17:00:00Z', '2024 Week 5'),
    ('2024-10-05T17:00:00Z', '2024 Week 6'),
    ('2024-10-12T17:00:00Z', '2024 Week 7'),
    ('2024-10-19T17:00:00Z', '2024 Week 8'),
    ('2024-10-26T17:00:00Z', '2024 Week 9'),
    ('2024-11-02T17:00:00Z', '2024 Week 10'),
    ('2024-11-09T17:00:00Z', '2024 Week 11'),
    ('2024-11-16T17:00:00Z', '2024 Week 12'),
    ('2024-11-23T17:00:00Z', '2024 Week 13'),
    ('2024-11-30T17:00:00Z', '2024 Week 14 - Rivalry Week'),
]


def backfill_2024_odds(api_key):
    """
    Backfill Vegas odds for complete 2024 season

    Args:
        api_key: The Odds API key
    """
    print("\n" + "="*80)
    print("2024 SEASON VEGAS ODDS BACKFILL")
    print("="*80 + "\n")

    # Initialize
    scraper = OddsAPIScraper(api_key=api_key)
    matcher = GameMatcher()
    db = FootballDatabase()
    db.connect()

    # Check remaining requests
    remaining = scraper.get_remaining_requests()
    print(f"Starting API requests remaining: {remaining}")
    print(f"Each date costs ~20 requests (10 per market)")
    print(f"Total dates to fetch: {len(DATES_2024)}")
    print(f"Total cost: ~{len(DATES_2024) * 20} requests")
    print(f"Remaining after: ~{remaining - (len(DATES_2024) * 20)}\n")

    if remaining < len(DATES_2024) * 20:
        print("WARNING: Not enough API requests remaining!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    else:
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled")
            return

    # Track stats
    total_games = 0
    total_matched = 0
    total_saved = 0

    # Process each week
    for i, (date, description) in enumerate(DATES_2024, 1):
        print("\n" + "="*80)
        print(f"[{i}/{len(DATES_2024)}] {description}")
        print(f"Date: {date}")
        print("="*80 + "\n")

        # Fetch historical odds
        print("Fetching historical odds...")
        games = scraper.fetch_historical_odds(
            date=date,
            markets=['spreads', 'totals']
        )

        if not games:
            print(f"  [ERROR] No games found for {date}")
            continue

        print(f"  [OK] Retrieved {len(games)} games\n")
        total_games += len(games)

        # Match and save each game
        matched = 0
        saved = 0

        for j, game in enumerate(games, 1):
            home = game.get('home_team')
            away = game.get('away_team')

            if not home or not away:
                continue

            # Match to ESPN game_id
            game_id = matcher.match_odds_api_game(game)

            if not game_id:
                print(f"  [WARN] Could not match: {away} @ {home}")
                continue

            matched += 1

            # Parse odds from bookmakers
            bookmakers = game.get('bookmakers', [])
            if not bookmakers:
                continue

            # Calculate consensus
            spreads_home = []
            totals = []

            for book in bookmakers:
                for market in book.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == home:
                                spreads_home.append(outcome['point'])

                    elif market['key'] == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == 'Over':
                                totals.append(outcome['point'])

            # Save to game_odds table
            if spreads_home or totals:
                avg_spread = sum(spreads_home) / len(spreads_home) if spreads_home else None
                avg_total = sum(totals) / len(totals) if totals else None

                # Insert into game_odds
                try:
                    cursor = db.conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO game_odds
                        (game_id, sportsbook, spread, total, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        game_id,
                        'TheOddsAPI_Consensus',
                        round(avg_spread, 1) if avg_spread else None,
                        round(avg_total, 1) if avg_total else None,
                        date
                    ))
                    db.conn.commit()
                    saved += 1

                except Exception as e:
                    print(f"  [ERROR] Error saving game_id {game_id}: {e}")

        total_matched += matched
        total_saved += saved

        print(f"\nWeek Summary:")
        print(f"  Games fetched: {len(games)}")
        print(f"  Matched to ESPN: {matched}")
        print(f"  Saved to database: {saved}")

        # Check remaining requests
        remaining = scraper.get_remaining_requests()
        print(f"\nAPI requests remaining: {remaining}")

        # Brief pause between requests
        if i < len(DATES_2024):
            print("Waiting 2 seconds...")
            time.sleep(2)

    db.close()

    # Final summary
    print("\n" + "="*80)
    print("2024 ODDS BACKFILL COMPLETE")
    print("="*80)
    print(f"\nTotal dates processed: {len(DATES_2024)}")
    print(f"Total games fetched: {total_games}")
    print(f"Total matched to ESPN: {total_matched}")
    print(f"Total saved to database: {total_saved}")
    print(f"\nAPI requests remaining: {remaining}")
    print("="*80 + "\n")


if __name__ == '__main__':
    # Load API key
    try:
        with open('odds_api_config.json', 'r') as f:
            config = json.load(f)
            api_key = config.get('api_key')
    except FileNotFoundError:
        print("ERROR: odds_api_config.json not found!")
        print("Create this file with your API key first")
        exit(1)

    if not api_key:
        print("ERROR: No API key in config file")
        exit(1)

    # Run backfill
    backfill_2024_odds(api_key)
