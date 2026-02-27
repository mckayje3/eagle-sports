"""Check which weeks have completed games"""
import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT week,
           COUNT(*) as total,
           SUM(CASE WHEN completed=1 THEN 1 ELSE 0 END) as completed
    FROM games
    GROUP BY week
    ORDER BY week
''')

print('Week | Total | Completed')
print('-----|-------|----------')
for row in cursor.fetchall():
    print(f'{row[0]:4} | {row[1]:5} | {row[2]:9}')

conn.close()
