"""
Test script to see what The Odds API historical endpoint actually returns
"""
import json
from odds_api_scraper import OddsAPIScraper

# Load API key
with open('odds_api_config.json', 'r') as f:
    config = json.load(f)

scraper = OddsAPIScraper(api_key=config['api_key'])

print("Testing historical odds API response format...")
print("=" * 80)

# Try fetching one recent date
test_date = "2025-11-22T12:00:00Z"  # A known game day
print(f"\nFetching odds for {test_date}")

games_data = scraper.fetch_historical_odds(
    date=test_date,
    markets=['spreads'],
    regions='us'
)

print(f"Type of data: {type(games_data)}")

if isinstance(games_data, dict):
    print(f"\nDict keys: {list(games_data.keys())}")
    print(f"\nFull dict structure:")
    print(json.dumps(games_data, indent=2)[:2000])  # First 2000 chars
elif isinstance(games_data, list):
    print(f"\nGames returned: {len(games_data)}")
    if games_data:
        print(f"\nFirst item type: {type(games_data[0])}")
        print(f"\nFirst item content:")
        print(json.dumps(games_data[0], indent=2))

print("\n" + "=" * 80)
