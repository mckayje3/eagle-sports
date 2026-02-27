"""
Strategic backfill of historical odds data
Uses The Odds API to fetch key game dates from past seasons
"""
import json
import time
from datetime import datetime
from odds_api_scraper import OddsAPIScraper
from game_matcher import GameMatcher
from database import FootballDatabase


# Key dates to backfill (strategically chosen)
# Format: YYYY-MM-DDTHH:00:00Z (Saturday noon UTC)
HISTORICAL_DATES = [
    # 2024 Season (most recent completed season)
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

    # 2023 Season (key weeks)
    ('2023-09-02T17:00:00Z', '2023 Week 1 - Season Opener'),
    ('2023-10-07T17:00:00Z', '2023 Week 6'),
    ('2023-11-04T17:00:00Z', '2023 Week 10'),
    ('2023-11-25T17:00:00Z', '2023 Week 13 - Rivalry Week'),

    # 2022 Season (key weeks)
    ('2022-09-03T17:00:00Z', '2022 Week 1 - Season Opener'),
    ('2022-10-08T17:00:00Z', '2022 Week 6'),
    ('2022-11-26T17:00:00Z', '2022 Week 13 - Rivalry Week'),

    # 2021 Season (key weeks)
    ('2021-09-04T17:00:00Z', '2021 Week 1 - Season Opener'),
    ('2021-11-27T17:00:00Z', '2021 Week 13 - Rivalry Week'),
]


def backfill_historical_odds(api_key, max_dates=None, start_from=0):
    """
    Backfill historical odds data

    Args:
        api_key: The Odds API key
        max_dates: Maximum number of dates to fetch (None = all)
        start_from: Index to start from (for resuming)
    """
    print("\n" + "="*80)
    print("HISTORICAL ODDS BACKFILL")
    print("="*80 + "\n")

    # Initialize
    scraper = OddsAPIScraper(api_key=api_key)
    matcher = GameMatcher()
    db = FootballDatabase()
    db.connect()

    # Check remaining requests
    remaining = scraper.get_remaining_requests()
    print(f"Starting API requests remaining: {remaining}")
    print(f"Each date costs 20 requests (10 per market)")
    print(f"Can fetch approximately {remaining // 20} dates\n")

    # Determine which dates to fetch
    dates_to_fetch = HISTORICAL_DATES[start_from:]
    if max_dates:
        dates_to_fetch = dates_to_fetch[:max_dates]

    print(f"Planning to fetch {len(dates_to_fetch)} dates")
    print(f"Total cost: ~{len(dates_to_fetch) * 20} requests")
    print(f"Remaining after: ~{remaining - (len(dates_to_fetch) * 20)}\n")

    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled")
        return

    # Track stats
    total_games = 0
    total_matched = 0
    total_saved = 0

    # Process each date
    for i, (date, description) in enumerate(dates_to_fetch, start_from + 1):
        print("\n" + "="*80)
        print(f"[{i}/{len(HISTORICAL_DATES)}] {description}")
        print(f"Date: {date}")
        print("="*80 + "\n")

        # Fetch historical odds
        print("Fetching historical odds...")
        games = scraper.fetch_historical_odds(
            date=date,
            markets=['spreads', 'totals']
        )

        if not games:
            print(f"  X No games found for {date}")
            continue

        print(f"  OK Retrieved {len(games)} games\n")
        total_games += len(games)

        # Parse season from date
        date_obj = datetime.fromisoformat(date.replace('Z', '+00:00'))
        season = date_obj.year
        if date_obj.month < 8:
            season -= 1

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

            # Create odds data
            odds_data = {
                'game_id': game_id,
                'source': 'TheOddsAPI_Historical'
            }

            if spreads_home:
                avg_spread = sum(spreads_home) / len(spreads_home)
                odds_data['opening_spread_home'] = round(avg_spread, 1)
                odds_data['opening_spread_away'] = round(-avg_spread, 1)

            if totals:
                avg_total = sum(totals) / len(totals)
                odds_data['opening_total'] = round(avg_total, 1)

            # Save to database
            try:
                db.insert_or_update_odds(odds_data)
                saved += 1

                # Also save to odds_movement
                movement_data = {
                    'game_id': game_id,
                    'source': 'TheOddsAPI_Historical',
                    'spread_home': odds_data.get('opening_spread_home'),
                    'spread_away': odds_data.get('opening_spread_away'),
                    'total': odds_data.get('opening_total'),
                    'timestamp': date
                }
                db.insert_odds_movement(movement_data)

            except Exception as e:
                print(f"  X Error saving: {e}")

        total_matched += matched
        total_saved += saved

        print(f"\nDate Summary:")
        print(f"  Games fetched: {len(games)}")
        print(f"  Matched to ESPN: {matched}")
        print(f"  Saved to database: {saved}")

        # Check remaining requests
        remaining = scraper.get_remaining_requests()
        print(f"\nAPI requests remaining: {remaining}")

        # Brief pause between requests
        if i < len(dates_to_fetch):
            print("Waiting 2 seconds...")
            time.sleep(2)

    db.close()

    # Final summary
    print("\n" + "="*80)
    print("BACKFILL COMPLETE")
    print("="*80)
    print(f"\nTotal dates processed: {len(dates_to_fetch)}")
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
    # You can customize these parameters:
    # - max_dates: Limit how many dates to fetch
    # - start_from: Resume from a specific index
    backfill_historical_odds(api_key, max_dates=23)
