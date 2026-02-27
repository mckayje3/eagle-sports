import sqlite3

conn = sqlite3.connect('nfl_games.db')
cursor = conn.cursor()

cursor.execute('SELECT season, week, COUNT(*) as cnt FROM games GROUP BY season, week ORDER BY season, week')
print('Season | Week | Games')
print('-' * 25)
for row in cursor.fetchall():
    print(f' {row[0]}  |  {row[1]:2d}  |  {row[2]:2d}')

cursor.execute('SELECT COUNT(*) FROM team_game_stats')
stats_count = cursor.fetchone()[0]
print(f'\nTotal team stats records: {stats_count}')

conn.close()
