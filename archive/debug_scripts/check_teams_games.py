"""Check team names and missing games"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('cfb_games.db')

# Check team display names
print("=" * 80)
print("TEAM NAME FORMAT")
print("=" * 80)
query = """
SELECT team_id, name, display_name
FROM teams
WHERE name IN ('Ducks', 'Volunteers', 'Bulldogs', 'Panthers', 'Yellow Jackets', 'Gators')
ORDER BY name
"""
teams_df = pd.read_sql_query(query, conn)
print(teams_df.to_string(index=False))

# Check for Tennessee vs Florida
print("\n" + "=" * 80)
print("CHECKING FOR TENNESSEE vs FLORIDA")
print("=" * 80)
query = """
SELECT
    g.game_id,
    g.season,
    g.week,
    g.date,
    g.completed,
    home.name as home_team,
    home.display_name as home_school,
    away.name as away_team,
    away.display_name as away_school
FROM games g
JOIN teams home ON g.home_team_id = home.team_id
JOIN teams away ON g.away_team_id = away.team_id
WHERE (home.name = 'Volunteers' OR away.name = 'Volunteers')
  AND (home.name = 'Gators' OR away.name = 'Gators')
  AND g.season = 2025
"""
tn_fl = pd.read_sql_query(query, conn)
if tn_fl.empty:
    print("Game not found!")
else:
    print(tn_fl.to_string(index=False))

# Check for Georgia Tech vs Pittsburgh
print("\n" + "=" * 80)
print("CHECKING FOR GEORGIA TECH vs PITTSBURGH")
print("=" * 80)
query = """
SELECT
    g.game_id,
    g.season,
    g.week,
    g.date,
    g.completed,
    home.name as home_team,
    home.display_name as home_school,
    away.name as away_team,
    away.display_name as away_school
FROM games g
JOIN teams home ON g.home_team_id = home.team_id
JOIN teams away ON g.away_team_id = away.team_id
WHERE (home.name = 'Yellow Jackets' OR away.name = 'Yellow Jackets')
  AND (home.name = 'Panthers' OR away.name = 'Panthers')
  AND g.season = 2025
"""
gt_pitt = pd.read_sql_query(query, conn)
if gt_pitt.empty:
    print("Game not found!")
else:
    print(gt_pitt.to_string(index=False))

# Check all upcoming games for week 13
print("\n" + "=" * 80)
print("ALL UPCOMING WEEK 13 GAMES (limit 10)")
print("=" * 80)
query = """
SELECT
    g.game_id,
    g.week,
    g.date,
    g.completed,
    home.display_name as home_school,
    away.display_name as away_school
FROM games g
JOIN teams home ON g.home_team_id = home.team_id
JOIN teams away ON g.away_team_id = away.team_id
WHERE g.season = 2025
  AND g.week = 13
  AND g.completed = 0
ORDER BY g.date
LIMIT 10
"""
upcoming = pd.read_sql_query(query, conn)
print(upcoming.to_string(index=False))

conn.close()
