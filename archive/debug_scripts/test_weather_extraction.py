"""
Test weather data extraction from ESPN API
"""
from espn_scraper import ESPNScraper
import sqlite3

def test_weather_extraction():
    """Test weather extraction on a recent completed game"""

    print("=" * 80)
    print("TESTING WEATHER DATA EXTRACTION")
    print("=" * 80)

    # Initialize scraper
    scraper = ESPNScraper('cfb_games.db')

    # Test on a recent Week 13 game
    print("\nFetching game details from ESPN API...")
    game_details = scraper.fetch_game_details('401752910')  # Recent Week 13 game

    # Extract weather from the API response
    if game_details:
        competitions = game_details.get('gameInfo', {}).get('competitions', [{}])[0] if 'gameInfo' in game_details else game_details.get('competitions', [{}])[0]
        weather = competitions.get('weather', {})
        venue = competitions.get('venue', {})

        print("\n" + "-" * 80)
        print("WEATHER DATA FROM API:")
        print("-" * 80)
        print(f"Weather object: {weather}")
        print(f"Venue indoor: {venue.get('indoor')}")

        # Now scrape a week and check if weather is saved
        print("\n" + "-" * 80)
        print("SCRAPING WEEK 13 TO TEST WEATHER SAVING:")
        print("-" * 80)
        scraper.scrape_week(season=2025, week=13)

        # Check database for weather data
        print("\n" + "-" * 80)
        print("CHECKING DATABASE FOR WEATHER DATA:")
        print("-" * 80)

        conn = sqlite3.connect('cfb_games.db')
        cursor = conn.cursor()

        # Check completed games with weather data
        cursor.execute("""
            SELECT
                game_id,
                home_team_id,
                away_team_id,
                temperature,
                wind_speed,
                conditions,
                is_dome,
                venue_name
            FROM games
            WHERE week = 13 AND season = 2025 AND completed = 1
            LIMIT 10
        """)

        games = cursor.fetchall()

        if games:
            print(f"\nFound {len(games)} completed Week 13 games:")
            print("\nGame ID | Temp | Wind | Conditions | Dome | Venue")
            print("-" * 80)

            weather_count = 0
            for game in games:
                game_id, home, away, temp, wind, cond, dome, venue = game
                if temp is not None or wind is not None or cond is not None:
                    weather_count += 1
                    has_weather = "[YES]"
                else:
                    has_weather = "[NO]"

                print(f"{game_id} | {temp or 'N/A':>4} | {wind or 'N/A':>4} | {cond or 'N/A':>12} | {dome} | {venue[:30] if venue else 'N/A'} {has_weather}")

            print("\n" + "=" * 80)
            print(f"SUMMARY: {weather_count}/{len(games)} games have weather data")
            print("=" * 80)
        else:
            print("No completed games found for Week 13")

        conn.close()
    else:
        print("Failed to fetch game details")

if __name__ == '__main__':
    test_weather_extraction()
