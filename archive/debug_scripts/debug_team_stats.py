"""Debug team stats extraction"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('cfb_games.db')

# Test parameters from Alabama Week 13 game
team_id = 333
season = 2025
up_to_week = 13

print(f"Getting stats for Team {team_id}, Season {season}, up to Week {up_to_week}")
print()

# The exact query from get_team_season_stats
query = """
    SELECT
        g.game_id,
        g.week,
        g.home_team_id,
        g.away_team_id,
        g.home_score,
        g.away_score
    FROM games g
    WHERE (g.home_team_id = ? OR g.away_team_id = ?)
        AND g.season = ?
        AND g.completed = 1
"""

params = [team_id, team_id, season]

if up_to_week:
    query += " AND g.week < ?"
    params.append(up_to_week)

query += " ORDER BY g.week"

print("SQL Query:")
print(query)
print()
print(f"Parameters: {params}")
print()

games_df = pd.read_sql_query(query, conn, params=params)

print(f"Found {len(games_df)} games")
print()

if not games_df.empty:
    print("Games:")
    print(games_df[['game_id', 'week', 'home_score', 'away_score']])
else:
    print("NO GAMES FOUND!")

conn.close()
