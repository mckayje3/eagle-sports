"""
Test script for The Odds API integration
Run this after getting your free API key
"""
import json
import sys
from odds_api_scraper import OddsAPIScraper
from datetime import datetime

def test_api_connection(api_key):
    """Test basic API connection and check remaining requests"""
    print("\n" + "="*80)
    print("TESTING THE ODDS API CONNECTION")
    print("="*80 + "\n")

    scraper = OddsAPIScraper(api_key=api_key)

    # Check remaining requests
    print("Checking API quota...")
    remaining = scraper.get_remaining_requests()

    if remaining is None:
        print("ERROR: Could not connect to The Odds API")
        print("   Check your API key and internet connection")
        return False

    print(f"SUCCESS: Connected successfully!")
    print(f"   Remaining requests: {remaining}/500")
    print()

    return True

def test_current_odds(api_key):
    """Test fetching current college football odds"""
    print("\n" + "="*80)
    print("FETCHING CURRENT NCAAF ODDS")
    print("="*80 + "\n")

    scraper = OddsAPIScraper(api_key=api_key)

    # Fetch current odds (costs 1 request per market)
    games = scraper.fetch_current_odds(markets=['spreads', 'totals'])

    if not games:
        print("ERROR: No games found")
        return

    print(f"SUCCESS: Retrieved {len(games)} upcoming games\n")

    # Display first 3 games
    print("Sample Games:")
    print("-" * 80)

    for i, game in enumerate(games[:3], 1):
        parsed = scraper.parse_odds_data(game)
        consensus = scraper.get_consensus_odds(parsed)

        print(f"\nGame {i}: {consensus['away_team']} @ {consensus['home_team']}")
        print(f"  Kickoff: {consensus['commence_time']}")
        print(f"  Spread:  Home {consensus.get('spread_home', 'N/A')}")
        print(f"  Total:   {consensus.get('total', 'N/A')}")
        print(f"  ML:      Home {consensus.get('moneyline_home', 'N/A')}, Away {consensus.get('moneyline_away', 'N/A')}")

    # Save full data to JSON
    with open('odds_api_current_games.json', 'w') as f:
        json.dump(games, f, indent=2)

    print("\n" + "-" * 80)
    print(f"SUCCESS: Saved full data to: odds_api_current_games.json")
    print(f"   Total games: {len(games)}")

def test_historical_odds(api_key, test_date='2024-09-07T12:00:00Z'):
    """
    Test fetching historical odds
    WARNING: This costs 10 requests per market!
    """
    print("\n" + "="*80)
    print("TESTING HISTORICAL ODDS (COSTS 10 REQUESTS)")
    print("="*80 + "\n")

    response = input("Fetch historical data? This will use 20 requests (10 per market). (y/N): ")

    if response.lower() != 'y':
        print("Skipping historical test")
        return

    scraper = OddsAPIScraper(api_key=api_key)

    # Fetch historical odds for opening weekend 2024
    print(f"\nFetching historical odds for: {test_date}")
    print("(Week 1 of 2024 season)")

    games = scraper.fetch_historical_odds(date=test_date, markets=['spreads', 'totals'])

    if not games:
        print("ERROR: No historical data found for this date")
        return

    print(f"\nSUCCESS: Retrieved {len(games)} games from {test_date}\n")

    # Display sample
    print("Sample Historical Games:")
    print("-" * 80)

    for i, game in enumerate(games[:3], 1):
        parsed = scraper.parse_odds_data(game)
        consensus = scraper.get_consensus_odds(parsed)

        print(f"\nGame {i}: {consensus['away_team']} @ {consensus['home_team']}")
        print(f"  Spread: Home {consensus.get('spread_home', 'N/A')}")
        print(f"  Total:  {consensus.get('total', 'N/A')}")

    # Save to JSON
    with open('odds_api_historical_games.json', 'w') as f:
        json.dump(games, f, indent=2)

    print("\n" + "-" * 80)
    print(f"SUCCESS: Saved historical data to: odds_api_historical_games.json")
    print(f"   Total games: {len(games)}")

def main():
    """Main test runner"""
    print("\n" + "="*80)
    print("THE ODDS API - TEST SCRIPT")
    print("="*80)

    # Check if API key is provided
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        # Try to load from config file
        try:
            with open('odds_api_config.json', 'r') as f:
                config = json.load(f)
                api_key = config.get('api_key')
        except FileNotFoundError:
            print("\nERROR: No API key found!")
            print("\nOptions:")
            print("  1. Run: py test_odds_api.py YOUR_API_KEY")
            print("  2. Create odds_api_config.json with your API key:")
            print('     {"api_key": "YOUR_KEY_HERE"}')
            print("\nGet your free API key at: https://the-odds-api.com/")
            return

    if not api_key:
        print("ERROR: API key is empty")
        return

    print(f"\nUsing API key: {api_key[:20]}...")

    # Run tests
    if not test_api_connection(api_key):
        return

    test_current_odds(api_key)

    # Optional: Test historical (costs more requests)
    test_historical_odds(api_key)

    # Final summary
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Review the JSON files generated")
    print("  2. Check data quality and coverage")
    print("  3. Decide if you want to backfill historical data")
    print("  4. Set up automated scraping for current games")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
