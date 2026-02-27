"""Debug parameter types"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('cfb_games.db')

# Get game details like get_game_features does
game_id = 401752773

query = """
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team_id,
        g.away_team_id
    FROM games g
    WHERE g.game_id = ?
"""

game_df = pd.read_sql_query(query, conn, params=[game_id])
game = game_df.iloc[0]

print("Game details from database:")
print(f"  home_team_id: {game['home_team_id']} (type: {type(game['home_team_id'])})")
print(f"  season: {game['season']} (type: {type(game['season'])})")
print(f"  week: {game['week']} (type: {type(game['week'])})")
print()

# Now test the stats query with these exact values
team_id = game['home_team_id']
season = game['season']
up_to_week = game['week']

print(f"Calling get_team_season_stats with:")
print(f"  team_id: {team_id} (type: {type(team_id)})")
print(f"  season: {season} (type: {type(season)})")
print(f"  up_to_week: {up_to_week} (type: {type(up_to_week)})")
print()

stats_query = """
    SELECT
        g.game_id,
        g.week
    FROM games g
    WHERE (g.home_team_id = ? OR g.away_team_id = ?)
        AND g.season = ?
        AND g.completed = 1
        AND g.week < ?
    ORDER BY g.week
"""

params = [team_id, team_id, season, up_to_week]

games = pd.read_sql_query(stats_query, conn, params=params)

print(f"Found {len(games)} games")

if games.empty:
    # Try with explicit int conversion
    print()
    print("Trying with explicit int conversion...")
    params = [int(team_id), int(team_id), int(season), int(up_to_week)]
    games = pd.read_sql_query(stats_query, conn, params=params)
    print(f"Found {len(games)} games with int conversion")

conn.close()
