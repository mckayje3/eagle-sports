import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

cursor.execute('PRAGMA table_info(game_odds)')
print('game_odds columns:')
for row in cursor.fetchall():
    print(f'  {row[1]} ({row[2]})')

cursor.execute('SELECT * FROM game_odds LIMIT 1')
print('\nSample row:')
for row in cursor.fetchall():
    print(row)

conn.close()
