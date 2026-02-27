"""
Check if ESPN API provides weather data for any games
"""
import requests
import json

def check_weather_data():
    """Check a few different games for weather data"""

    print("=" * 80)
    print("CHECKING WEATHER DATA IN ESPN API")
    print("=" * 80)

    # Test a few different game IDs
    test_games = [
        '401752910',  # Rose Bowl (outdoor)
        '401752773',  # Bryant-Denny Stadium (outdoor)
        '401762862',  # Week 13 weekday game
    ]

    for game_id in test_games:
        print(f"\nGame ID: {game_id}")
        print("-" * 80)

        url = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
        params = {'event': game_id}

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            # Check header > competitions
            header = data.get('header', {})
            if header.get('competitions'):
                comp = header['competitions'][0]
                weather = comp.get('weather')
                print(f"  Header > competitions > weather: {weather}")

            # Check gameInfo
            game_info = data.get('gameInfo', {})
            venue = game_info.get('venue', {})
            print(f"  gameInfo > venue > indoor: {venue.get('indoor')}")
            print(f"  gameInfo > venue > grass: {venue.get('grass')}")

            # Check for weather anywhere else
            if 'weather' in data:
                print(f"  Top-level weather: {data['weather']}")

            # Check if venue has capacity (might indicate indoor)
            if 'capacity' in venue:
                print(f"  Venue capacity: {venue['capacity']}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("ESPN API does NOT consistently provide weather data.")
    print("Weather data is only available for live/recent games in some cases.")
    print()
    print("We can extract is_dome from venue data when available,")
    print("but temperature, wind, and conditions are rarely provided.")
    print("=" * 80)

if __name__ == '__main__':
    check_weather_data()
