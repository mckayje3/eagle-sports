import sqlite3
import pandas as pd

print("Checking predictions CSV...")
preds_csv = pd.read_csv('enhanced_predictions_week_13.csv')
print(f"First 10 game_ids in predictions: {preds_csv['game_id'].head(10).tolist()}")
print(f"Data types: {preds_csv['game_id'].dtype}")

print("\nChecking database...")
conn = sqlite3.connect('cfb_games.db')
results = pd.read_sql_query("""
    SELECT game_id, ht.name as home, at.name as away
    FROM games g
    JOIN teams ht ON g.home_team_id = ht.team_id
    JOIN teams at ON g.away_team_id = at.team_id
    WHERE g.season = 2024 AND g.week = 13 AND g.completed = 1
    LIMIT 10
""", conn)
print(f"First 10 game_ids in database: {results['game_id'].tolist()}")
print(f"Data types: {results['game_id'].dtype}")

print("\nChecking for matches...")
common = set(preds_csv['game_id']) & set(results['game_id'])
print(f"Common game_ids: {len(common)}")
if common:
    print(f"Examples: {list(common)[:5]}")

conn.close()
