import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

tables = ['games', 'teams', 'team_game_stats', 'game_odds', 'odds_movement', 'drives']

for table in tables:
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
    result = cursor.fetchone()
    print(f"\n{'='*80}")
    print(f"{table.upper()} TABLE")
    print('='*80)
    if result:
        print(result[0])
    else:
        print("Not found")

conn.close()
