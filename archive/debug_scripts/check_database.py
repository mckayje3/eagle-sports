"""Check what's in the database"""
import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

print('=' * 60)
print('DATABASE CONTENTS SUMMARY')
print('=' * 60)

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print('\nTables in database:')
for t in tables:
    print(f'  - {t}')

print('\n' + '-' * 60)

# Check games table
print('\nGAMES TABLE:')
cursor.execute('SELECT season, COUNT(*) as games, MIN(week) as min_week, MAX(week) as max_week FROM games GROUP BY season ORDER BY season')
for row in cursor.fetchall():
    season, count, min_week, max_week = row
    cursor.execute('SELECT COUNT(*) FROM games WHERE season=? AND completed=1', (season,))
    completed = cursor.fetchone()[0]
    print(f'  {season}: {count} games total, {completed} completed (weeks {min_week}-{max_week})')

print('\n' + '-' * 60)

# Check team_game_stats
print('\nTEAM_GAME_STATS TABLE:')
cursor.execute('''
    SELECT g.season, COUNT(*) as stat_records
    FROM team_game_stats tgs
    JOIN games g ON tgs.game_id = g.game_id
    GROUP BY g.season
    ORDER BY g.season
''')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} stat records')

print('\n' + '-' * 60)

# Check odds tables
print('\nODDS/BETTING DATA:')
if 'game_odds' in tables:
    cursor.execute('SELECT COUNT(*) FROM game_odds')
    count = cursor.fetchone()[0]
    print(f'  game_odds: {count} records')

    # Check schema
    cursor.execute('PRAGMA table_info(game_odds)')
    cols = [row[1] for row in cursor.fetchall()]
    print(f'    Columns: {", ".join(cols)}')

    # Sample data
    cursor.execute('SELECT * FROM game_odds LIMIT 1')
    sample = cursor.fetchone()
    if sample:
        print(f'    Sample: {sample}')

if 'odds_movement' in tables:
    cursor.execute('SELECT COUNT(*) FROM odds_movement')
    count = cursor.fetchone()[0]
    print(f'  odds_movement: {count} records')

if 'game_odds' not in tables and 'odds_movement' not in tables:
    print('  No odds tables found')

print('\n' + '-' * 60)

# Check teams table
cursor.execute('SELECT COUNT(*) FROM teams')
team_count = cursor.fetchone()[0]
print(f'\nTEAMS TABLE: {team_count} teams')

print('\n' + '=' * 60)

conn.close()
