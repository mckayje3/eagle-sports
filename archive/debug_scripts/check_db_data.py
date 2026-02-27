"""Quick check of database contents"""
import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

# Check completed games by season
cursor.execute('SELECT season, COUNT(*) as game_count FROM games WHERE completed = 1 GROUP BY season ORDER BY season')
print('Completed Games by Season:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} games')

# Check teams with stats by season
cursor.execute('''
    SELECT g.season, COUNT(DISTINCT tgs.team_id) as teams_with_stats
    FROM games g
    LEFT JOIN team_game_stats tgs ON g.game_id = tgs.game_id
    WHERE g.completed = 1
    GROUP BY g.season
    ORDER BY g.season
''')
print('\nTeams with Stats by Season:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} teams')

# Check sample of 2024 data
cursor.execute('''
    SELECT week, COUNT(*) as games
    FROM games
    WHERE season = 2024 AND completed = 1
    GROUP BY week
    ORDER BY week
''')
print('\n2024 Season by Week:')
total_2024 = 0
for row in cursor.fetchall():
    print(f'  Week {row[0]}: {row[1]} games')
    total_2024 += row[1]
print(f'  Total 2024: {total_2024} games')

conn.close()
