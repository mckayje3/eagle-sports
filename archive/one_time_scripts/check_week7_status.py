"""Check Week 7 NBA completed games status"""
import sqlite3

conn = sqlite3.connect('users.db')
cursor = conn.cursor()

print('NBA 2025 Week 7 Completed Games Status:')
print('='*70)

cursor.execute('''
    SELECT home_team, away_team, game_date, vegas_spread, vegas_total, game_completed
    FROM prediction_cache
    WHERE sport = 'NBA' AND season = 2025 AND week = 7 AND game_completed = 1
    ORDER BY game_date
''')

for row in cursor.fetchall():
    has_spread = 'YES' if row[3] and row[3] != 0 else 'NO'
    has_total = 'YES' if row[4] and row[4] != 0 else 'NO'
    print(f'{row[1][:15]:15} @ {row[0][:15]:15} | {row[2][:10]} | Spread: {has_spread} | Total: {has_total}')

cursor.execute('''
    SELECT
        SUM(CASE WHEN game_completed = 1 THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN game_completed = 1 AND vegas_spread != 0 AND vegas_spread IS NOT NULL THEN 1 ELSE 0 END) as completed_with_odds,
        COUNT(*) as total,
        SUM(CASE WHEN vegas_spread != 0 AND vegas_spread IS NOT NULL THEN 1 ELSE 0 END) as total_with_odds
    FROM prediction_cache
    WHERE sport = 'NBA' AND season = 2025 AND week = 7
''')
row = cursor.fetchone()
print('')
print(f'Summary: {row[1]}/{row[0]} completed games have odds, {row[3]}/{row[2]} total games have odds')
conn.close()
