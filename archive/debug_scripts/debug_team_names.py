import sqlite3
import pandas as pd

print("Prediction team names:")
preds = pd.read_csv('enhanced_predictions_week_13.csv')
print(preds[['home_team', 'away_team']].head(20))

print("\n\nDatabase team names:")
conn = sqlite3.connect('cfb_games.db')
results = pd.read_sql_query("""
    SELECT ht.name as home_team, at.name as away_team
    FROM games g
    JOIN teams ht ON g.home_team_id = ht.team_id
    JOIN teams at ON g.away_team_id = at.team_id
    WHERE g.season = 2024 AND g.week = 13 AND g.completed = 1
    LIMIT 20
""", conn)
conn.close()

print(results)
