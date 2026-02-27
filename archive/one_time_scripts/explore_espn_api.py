"""
Explore what additional data ESPN's API provides
"""
import requests
import json

def explore_game_data(game_id='401752910'):  # Recent Week 13 game
    """Fetch a completed game and show all available data"""

    print("=" * 80)
    print("EXPLORING ESPN CFB API DATA")
    print("=" * 80)

    # Fetch game summary (what we currently use)
    url = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
    params = {'event': game_id}

    response = requests.get(url, params=params)
    data = response.json()

    print(f"\nGame ID: {game_id}")
    print(f"\nTop-level keys available:")
    for key in sorted(data.keys()):
        print(f"  - {key}")

    # Explore boxscore
    if 'boxscore' in data:
        print("\n" + "-" * 80)
        print("BOXSCORE DATA:")
        print("-" * 80)
        boxscore = data['boxscore']
        print(f"Boxscore keys: {list(boxscore.keys())}")

        if 'teams' in boxscore:
            print(f"\nNumber of teams: {len(boxscore['teams'])}")
            for team in boxscore['teams']:
                print(f"\n  Team: {team.get('team', {}).get('displayName')}")
                print(f"  Statistics available: {list(team.get('statistics', [{}])[0].keys()) if team.get('statistics') else 'None'}")

    # Explore plays (play-by-play)
    if 'plays' in data:
        print("\n" + "-" * 80)
        print("PLAY-BY-PLAY DATA:")
        print("-" * 80)
        plays = data['plays']
        print(f"Number of plays: {len(plays)}")
        if plays:
            sample_play = plays[0]
            print(f"\nSample play keys: {list(sample_play.keys())}")
            print(f"Sample play: {sample_play.get('text', 'N/A')}")

    # Explore drives
    if 'drives' in data:
        print("\n" + "-" * 80)
        print("DRIVE DATA:")
        print("-" * 80)
        drives = data['drives']
        print(f"Drives keys: {list(drives.keys())}")
        if 'previous' in drives:
            print(f"Number of drives: {len(drives['previous'])}")
            if drives['previous']:
                sample_drive = drives['previous'][0]
                print(f"\nSample drive keys: {list(sample_drive.keys())}")

    # Explore scoring plays
    if 'scoringPlays' in data:
        print("\n" + "-" * 80)
        print("SCORING PLAYS DATA:")
        print("-" * 80)
        scoring = data['scoringPlays']
        print(f"Number of scoring plays: {len(scoring)}")
        if scoring:
            sample = scoring[0]
            print(f"Sample scoring play keys: {list(sample.keys())}")

    # Explore standings
    if 'standings' in data:
        print("\n" + "-" * 80)
        print("STANDINGS DATA:")
        print("-" * 80)
        print("Standings data available: YES")

    # Explore leaders
    if 'leaders' in data:
        print("\n" + "-" * 80)
        print("GAME LEADERS DATA:")
        print("-" * 80)
        leaders = data['leaders']
        print(f"Number of leader categories: {len(leaders)}")
        for leader_cat in leaders[:3]:  # Show first 3
            print(f"\n  Category: {leader_cat.get('displayName')}")
            print(f"  Leaders: {len(leader_cat.get('leaders', []))} teams")

    # Explore player stats
    if 'boxscore' in data and 'players' in data['boxscore']:
        print("\n" + "-" * 80)
        print("PLAYER STATISTICS DATA:")
        print("-" * 80)
        players = data['boxscore']['players']
        print(f"Number of teams with player stats: {len(players)}")
        if players:
            team_players = players[0]
            print(f"\nTeam: {team_players.get('team', {}).get('displayName')}")
            print(f"Stat categories available:")
            for cat in team_players.get('statistics', []):
                print(f"  - {cat.get('name')}: {len(cat.get('athletes', []))} players")
                if cat.get('athletes'):
                    sample_player = cat['athletes'][0]
                    print(f"    Sample player keys: {list(sample_player.keys())}")

    # Explore weather
    if 'gameInfo' in data:
        print("\n" + "-" * 80)
        print("GAME INFO DATA:")
        print("-" * 80)
        game_info = data['gameInfo']
        print(f"Game info keys: {list(game_info.keys())}")
        if 'weather' in game_info:
            print(f"Weather data: {game_info['weather']}")

    # Explore predictor/win probability
    if 'predictor' in data or 'winprobability' in data:
        print("\n" + "-" * 80)
        print("WIN PROBABILITY DATA:")
        print("-" * 80)
        if 'predictor' in data:
            print("Predictor data available: YES")
        if 'winprobability' in data:
            print("Win probability data available: YES")

    # Save full response for inspection
    with open('espn_api_sample.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("\n" + "=" * 80)
    print("Full API response saved to: espn_api_sample.json")
    print("=" * 80)

    return data

if __name__ == '__main__':
    explore_game_data()
