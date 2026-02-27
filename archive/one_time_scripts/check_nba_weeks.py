import sqlite3

conn = sqlite3.connect('users.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT week, COUNT(*) as games,
           SUM(CASE WHEN vegas_spread != 0 AND vegas_spread IS NOT NULL THEN 1 ELSE 0 END) as with_spread,
           SUM(game_completed) as completed
    FROM prediction_cache
    WHERE sport = 'NBA' AND season = 2025
    GROUP BY week
    ORDER BY week
''')

print('NBA 2025 Predictions by Week:')
print('='*70)
print('Week  | Games | With Spread | Completed | Missing Spread')
print('-'*70)
for row in cursor.fetchall():
    missing = row[1] - row[2]
    print(f'{row[0]:5} | {row[1]:5} | {row[2]:11} | {row[3]:9} | {missing}')

conn.close()
