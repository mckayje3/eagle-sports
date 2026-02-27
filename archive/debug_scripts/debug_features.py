import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

print('Checking team_game_stats table...\n')

# Check a sample record
cursor.execute('SELECT * FROM team_game_stats LIMIT 1')
sample = cursor.fetchone()
print('Sample record:', sample)
print()

# Check team 8 specifically
cursor.execute('''
    SELECT tgs.points, tgs.total_yards, tgs.passing_yards, tgs.rushing_yards, g.week
    FROM team_game_stats tgs
    JOIN games g ON tgs.game_id = g.game_id
    WHERE tgs.team_id = 8 AND g.season = 2024 AND g.completed = 1
    LIMIT 5
''')
print('Team 8 (first 5 games):')
for row in cursor.fetchall():
    print(f'  Week {row[4]}: {row[0]} pts, {row[1]} yds, {row[2]} pass, {row[3]} rush')

conn.close()
