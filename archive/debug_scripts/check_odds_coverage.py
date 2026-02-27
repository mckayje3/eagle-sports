"""Check odds coverage by source"""
import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

print("=" * 80)
print("ODDS COVERAGE ANALYSIS")
print("=" * 80)

# Count by source
print("\nOdds records by source:")
cursor.execute('''
    SELECT source, COUNT(*) as count
    FROM game_odds
    GROUP BY source
''')
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} records")

# Count games with odds vs without
cursor.execute('''
    SELECT
        COUNT(DISTINCT g.game_id) as total_games,
        COUNT(DISTINCT go.game_id) as games_with_odds
    FROM games g
    LEFT JOIN game_odds go ON g.game_id = go.game_id
    WHERE g.season IN (2024, 2025) AND g.completed = 1
''')
total, with_odds = cursor.fetchone()
print(f"\nCompleted games: {total}")
print(f"Games with odds: {with_odds}")
print(f"Coverage: {with_odds/total*100:.1f}%")

# Sample TheOddsAPI records
print("\nSample TheOddsAPI records:")
cursor.execute('''
    SELECT g.season, g.week, g.date,
           g.home_team_id, g.away_team_id,
           go.opening_spread_home, go.opening_total
    FROM game_odds go
    JOIN games g ON go.game_id = g.game_id
    WHERE go.source = 'TheOddsAPI'
      AND go.opening_spread_home IS NOT NULL
    ORDER BY g.date DESC
    LIMIT 10
''')
for row in cursor.fetchall():
    season, week, date, home, away, spread, total = row
    spread_str = f"{spread:+.1f}" if spread else "N/A"
    total_str = f"{total:.1f}" if total else "N/A"
    print(f"  {season} W{week} - {away} @ {home}: {spread_str} / {total_str}")

conn.close()
print("\n" + "=" * 80)
