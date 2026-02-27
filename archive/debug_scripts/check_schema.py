import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

# Check team_game_stats schema
cursor.execute('PRAGMA table_info(team_game_stats)')
print('team_game_stats columns:')
for row in cursor.fetchall():
    print(f'  {row[1]} ({row[2]})')

cursor.execute('SELECT COUNT(*) FROM team_game_stats')
print(f'\nTotal team_game_stats records: {cursor.fetchone()[0]}')

# Check games table
cursor.execute('SELECT COUNT(*) FROM games WHERE completed = 1')
print(f'Completed games: {cursor.fetchone()[0]}')

conn.close()
