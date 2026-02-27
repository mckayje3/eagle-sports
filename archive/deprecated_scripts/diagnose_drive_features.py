"""
Diagnostic script to understand why Deep Eagle drive features are missing
"""
import sqlite3
import pandas as pd

def test_drive_query():
    """Test the drive stats query to see what's happening"""
    conn = sqlite3.connect('nfl_games.db')

    # Get a sample team from Week 13
    sample_team = pd.read_sql_query('''
        SELECT DISTINCT home_team_id as team_id, ht.display_name as team_name
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        WHERE g.season = 2025 AND g.week = 13 AND g.completed = 0
        LIMIT 1
    ''', conn)

    if sample_team.empty:
        print("No Week 13 games found!")
        return

    team_id = sample_team.iloc[0]['team_id']
    team_name = sample_team.iloc[0]['team_name']

    print(f"\n{'='*80}")
    print(f"TESTING DRIVE STATS FOR: {team_name} (ID: {team_id})")
    print(f"Season: 2025, Current Week: 13 (should look at weeks < 13)")
    print('='*80)

    # Test 1: Do any drives exist for this team?
    total_drives = pd.read_sql_query('''
        SELECT COUNT(*) as count
        FROM drives d
        WHERE d.team_id = ?
    ''', conn, params=(team_id,))
    print(f"\n1. Total drives for this team (all time): {total_drives.iloc[0]['count']}")

    # Test 2: Drives in 2025 season
    drives_2025 = pd.read_sql_query('''
        SELECT COUNT(*) as count
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE d.team_id = ?
            AND g.season = 2025
    ''', conn, params=(team_id,))
    print(f"2. Drives in 2025 season: {drives_2025.iloc[0]['count']}")

    # Test 3: Drives before Week 13
    drives_before_13 = pd.read_sql_query('''
        SELECT COUNT(*) as count
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE d.team_id = ?
            AND g.season = 2025
            AND g.week < 13
    ''', conn, params=(team_id,))
    print(f"3. Drives before Week 13: {drives_before_13.iloc[0]['count']}")

    # Test 4: Completed games before Week 13
    drives_completed = pd.read_sql_query('''
        SELECT COUNT(*) as count
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE d.team_id = ?
            AND g.season = 2025
            AND g.week < 13
            AND g.completed = 1
    ''', conn, params=(team_id,))
    print(f"4. Drives in COMPLETED games before Week 13: {drives_completed.iloc[0]['count']}")

    # Test 5: Run the actual query from _get_drive_stats
    print(f"\n5. Running actual _get_drive_stats query:")
    query_off = '''
        SELECT
            COUNT(*) as total_drives,
            AVG(CASE
                WHEN d.result LIKE '%TD%' THEN 7
                WHEN d.result LIKE '%FG%' THEN 3
                WHEN d.result LIKE '%SAFETY%' THEN 2
                ELSE 0
            END) as ppd,
            AVG(d.yards) as ypd,
            AVG(d.plays) as plays_per_drive,
            AVG(d.time_elapsed_seconds) as seconds_per_drive,
            AVG(d.is_score) as scoring_pct,
            SUM(CASE WHEN d.start_yards_to_endzone <= 20 THEN d.is_score ELSE 0 END) * 1.0 /
                NULLIF(SUM(CASE WHEN d.start_yards_to_endzone <= 20 THEN 1 ELSE 0 END), 0) as redzone_pct,
            SUM(CASE WHEN d.plays <= 3 AND d.is_score = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as three_and_out_pct,
            SUM(CASE WHEN d.yards >= 20 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as explosive_drive_pct
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE d.team_id = ?
            AND g.season = ?
            AND g.week < ?
            AND g.completed = 1
    '''

    result = pd.read_sql_query(query_off, conn, params=(team_id, 2025, 13))

    if result.empty:
        print("  ❌ Query returned empty!")
    else:
        print("  ✓ Query returned results:")
        for col in result.columns:
            val = result.iloc[0][col]
            print(f"    {col}: {val}")

    # Test 6: Check if drives table has the required columns
    print(f"\n6. Checking drives table schema:")
    schema = pd.read_sql_query("PRAGMA table_info(drives)", conn)
    print(f"  Columns: {', '.join(schema['name'].tolist())}")

    # Test 7: Sample some actual drive data
    print(f"\n7. Sample drives for this team:")
    sample_drives = pd.read_sql_query('''
        SELECT d.*, g.week, g.season
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE d.team_id = ?
            AND g.season = 2025
            AND g.week < 13
        LIMIT 3
    ''', conn, params=(team_id,))

    if sample_drives.empty:
        print("  ❌ No sample drives found!")
    else:
        print(f"  ✓ Found {len(sample_drives)} sample drives")
        print(sample_drives[['week', 'plays', 'yards', 'result', 'is_score']].to_string(index=False))

    conn.close()

if __name__ == '__main__':
    test_drive_query()
