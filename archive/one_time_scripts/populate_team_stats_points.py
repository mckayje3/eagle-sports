"""
Populate points field in team_game_stats from games table
The points are in the games table (home_score/away_score) but team_game_stats.points is NULL
"""
import sqlite3

conn = sqlite3.connect('nfl_games.db')
cursor = conn.cursor()

# Update home team points
cursor.execute('''
    UPDATE team_game_stats
    SET points = (
        SELECT home_score
        FROM games
        WHERE games.game_id = team_game_stats.game_id
        AND team_game_stats.team_id = games.home_team_id
    )
    WHERE EXISTS (
        SELECT 1 FROM games
        WHERE games.game_id = team_game_stats.game_id
        AND team_game_stats.team_id = games.home_team_id
        AND games.home_score IS NOT NULL
    )
''')

home_updated = cursor.rowcount
print(f"Updated {home_updated} home team records with points")

# Update away team points
cursor.execute('''
    UPDATE team_game_stats
    SET points = (
        SELECT away_score
        FROM games
        WHERE games.game_id = team_game_stats.game_id
        AND team_game_stats.team_id = games.away_team_id
    )
    WHERE EXISTS (
        SELECT 1 FROM games
        WHERE games.game_id = team_game_stats.game_id
        AND team_game_stats.team_id = games.away_team_id
        AND games.away_score IS NOT NULL
    )
''')

away_updated = cursor.rowcount
print(f"Updated {away_updated} away team records with points")

conn.commit()

# Verify
cursor.execute('''
    SELECT COUNT(*) as total, COUNT(points) as with_points
    FROM team_game_stats ts
    JOIN games g ON ts.game_id = g.game_id
    WHERE g.season = 2025
''')
result = cursor.fetchone()

print(f"\nVerification:")
print(f"  Total 2025 records: {result[0]}")
print(f"  With points: {result[1]}")
print(f"  Coverage: {100 * result[1] / result[0]:.1f}%")

conn.close()
print("\nDone!")
