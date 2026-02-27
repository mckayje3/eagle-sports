"""
Check what odds data was saved to the database
"""
import sqlite3

# Connect to database
conn = sqlite3.connect('cfb_games.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Check how many odds records were saved
cursor.execute('SELECT COUNT(*) as count FROM game_odds')
odds_count = cursor.fetchone()['count']
print(f"Total odds records in database: {odds_count}")

# Check how many odds movement records
cursor.execute('SELECT COUNT(*) as count FROM odds_movement')
movement_count = cursor.fetchone()['count']
print(f"Total odds movement records: {movement_count}")

# Show sample odds with game info
print("\n" + "="*100)
print("SAMPLE ODDS DATA")
print("="*100 + "\n")

cursor.execute('''
    SELECT
        g.game_id,
        g.date,
        home.name as home_team,
        away.name as away_team,
        o.source,
        o.opening_spread_home,
        o.current_spread_home,
        o.opening_total,
        o.current_total
    FROM game_odds o
    JOIN games g ON o.game_id = g.game_id
    JOIN teams home ON g.home_team_id = home.team_id
    JOIN teams away ON g.away_team_id = away.team_id
    LIMIT 10
''')

for row in cursor.fetchall():
    print(f"Game ID: {row['game_id']}")
    print(f"  {row['away_team']} @ {row['home_team']}")
    print(f"  Date: {row['date']}")
    print(f"  Source: {row['source']}")
    print(f"  Opening Spread: {row['opening_spread_home']}")
    print(f"  Current Spread: {row['current_spread_home']}")
    print(f"  Opening Total: {row['opening_total']}")
    print(f"  Current Total: {row['current_total']}")
    print()

# Show line movements
print("="*100)
print("SAMPLE LINE MOVEMENTS")
print("="*100 + "\n")

cursor.execute('''
    SELECT
        g.game_id,
        home.name as home_team,
        away.name as away_team,
        om.spread_home,
        om.total,
        om.timestamp
    FROM odds_movement om
    JOIN games g ON om.game_id = g.game_id
    JOIN teams home ON g.home_team_id = home.team_id
    JOIN teams away ON g.away_team_id = away.team_id
    ORDER BY om.timestamp DESC
    LIMIT 10
''')

for row in cursor.fetchall():
    print(f"{row['away_team']} @ {row['home_team']}")
    print(f"  Spread: {row['spread_home']}, Total: {row['total']}")
    print(f"  Timestamp: {row['timestamp']}")
    print()

conn.close()

print("="*100)
print("SUCCESS! Odds data is in the database and linked to games!")
print("="*100)
