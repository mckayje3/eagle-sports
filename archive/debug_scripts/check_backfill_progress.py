"""
Check backfill progress
"""
import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

# Get games by week
cursor.execute('''
    SELECT season, week, COUNT(*) as games
    FROM games
    GROUP BY season, week
    ORDER BY season, week
''')

print("\nGames by week:")
print("-" * 40)
for row in cursor.fetchall():
    print(f"Season {row[0]}, Week {row[1]:2d}: {row[2]:3d} games")

# Get total
cursor.execute('SELECT COUNT(*) as total FROM games')
total = cursor.fetchone()[0]
print("-" * 40)
print(f"Total games: {total}")

# Get teams count
cursor.execute('SELECT COUNT(*) FROM teams')
teams = cursor.fetchone()[0]
print(f"Total teams: {teams}")

conn.close()
