"""
Check is_dome field population in database
"""
import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

# Check how many games have is_dome data
cursor.execute('SELECT COUNT(*) FROM games WHERE is_dome IS NOT NULL')
total_with_dome = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM games WHERE is_dome = 1')
dome_games = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM games WHERE is_dome = 0')
outdoor_games = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM games')
total_games = cursor.fetchone()[0]

print("=" * 80)
print("IS_DOME FIELD STATUS")
print("=" * 80)
print(f"Total games: {total_games}")
print(f"Games with is_dome data: {total_with_dome}")
print(f"  - Indoor/Dome: {dome_games}")
print(f"  - Outdoor: {outdoor_games}")
print(f"Games without is_dome data: {total_games - total_with_dome}")

# Show venues marked as domes
print("\n" + "-" * 80)
print("VENUES MARKED AS DOMES (is_dome=1):")
print("-" * 80)
cursor.execute('''
    SELECT venue_name, COUNT(*) as game_count
    FROM games
    WHERE is_dome = 1
    GROUP BY venue_name
    ORDER BY game_count DESC
    LIMIT 20
''')

for venue, count in cursor.fetchall():
    print(f"  {venue}: {count} games")

# Show sample outdoor stadiums
print("\n" + "-" * 80)
print("SAMPLE OUTDOOR STADIUMS (is_dome=0):")
print("-" * 80)
cursor.execute('''
    SELECT venue_name, COUNT(*) as game_count
    FROM games
    WHERE is_dome = 0
    GROUP BY venue_name
    ORDER BY game_count DESC
    LIMIT 10
''')

for venue, count in cursor.fetchall():
    print(f"  {venue}: {count} games")

conn.close()
