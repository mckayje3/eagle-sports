import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

# Check games by season
cursor.execute('SELECT season, COUNT(*) FROM games GROUP BY season ORDER BY season')
print('Games by Season:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} games')

# Check total games with odds
cursor.execute('SELECT COUNT(DISTINCT game_id) FROM game_odds')
odds = cursor.fetchone()[0]
print(f'\nTotal Games with Odds: {odds}')

# Check odds by season
cursor.execute('''
    SELECT g.season, COUNT(DISTINCT g.game_id)
    FROM games g
    JOIN game_odds o ON g.game_id = o.game_id
    GROUP BY g.season
    ORDER BY g.season
''')
print('\nOdds by Season:')
rows = cursor.fetchall()
if rows:
    for row in rows:
        print(f'  {row[0]}: {row[1]} games')
else:
    print('  No odds data found')

# Check teams
cursor.execute('SELECT COUNT(DISTINCT team_id) FROM teams')
teams = cursor.fetchone()[0]
print(f'\nTotal Teams: {teams}')

# Check team stats
cursor.execute('SELECT COUNT(*) FROM team_game_stats')
stats = cursor.fetchone()[0]
print(f'Total Team Stats Records: {stats}')

conn.close()
