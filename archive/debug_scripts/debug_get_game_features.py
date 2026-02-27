"""Debug get_game_features"""
from ml_feature_extraction_v2 import FeatureExtractorV2
import sqlite3
import pandas as pd

extractor = FeatureExtractorV2('cfb_games.db')

game_id = 401752773

print(f"Testing get_game_features for game_id: {game_id}")
print()

# First, check if the game exists
conn = sqlite3.connect('cfb_games.db')
game_check = pd.read_sql_query("""
    SELECT g.game_id, g.season, g.week, g.home_team_id, g.away_team_id,
           ht.name as home_team, at.name as away_team
    FROM games g
    JOIN teams ht ON g.home_team_id = ht.team_id
    JOIN teams at ON g.away_team_id = at.team_id
    WHERE g.game_id = ?
""", conn, params=[game_id])

if game_check.empty:
    print("ERROR: Game not found in database!")
else:
    print("Game found:")
    print(game_check)
    print()

    # Now call get_game_features
    features = extractor.get_game_features(game_id)

    print(f"Features returned: {len(features)} keys")
    print()
    print("Key features:")
    print(f"  season: {features.get('season', 'MISSING')}")
    print(f"  week: {features.get('week', 'MISSING')}")
    print(f"  home_games_played: {features.get('home_games_played', 'MISSING')}")
    print(f"  home_wins: {features.get('home_wins', 'MISSING')}")
    print(f"  home_points_scored_avg: {features.get('home_points_scored_avg', 'MISSING')}")

conn.close()
