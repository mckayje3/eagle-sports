"""
Debug script to see what ESPN actually returns for game stats
"""
from nfl_espn_scraper import NFLESPNScraper
import json

scraper = NFLESPNScraper()

# Fetch one completed game
game_id = '401772510'  # Week 1 game
details = scraper.fetch_game_details(game_id)

if 'boxscore' in details:
    teams = details['boxscore'].get('teams', [])

    for team_data in teams:
        team_name = team_data.get('team', {}).get('displayName', 'Unknown')
        stats = team_data.get('statistics', [])

        print(f"\n{'='*80}")
        print(f"TEAM: {team_name}")
        print('='*80)

        for stat in stats:
            name = stat.get('name', '')
            display_value = stat.get('displayValue', '')
            print(f"{name:40} = {display_value}")

        break  # Just show one team
else:
    print("No boxscore data found!")
