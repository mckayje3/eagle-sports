"""
Create drives table for storing drive-by-drive data
"""
import sqlite3

def create_drives_table(db_path):
    """Create drives table in the specified database"""

    print(f"\nCreating drives table in {db_path}...")
    print("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create drives table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS drives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            drive_number INTEGER,
            team_id INTEGER NOT NULL,
            start_period INTEGER,
            start_clock TEXT,
            start_yard_line INTEGER,
            start_yards_to_endzone INTEGER,
            end_period INTEGER,
            end_clock TEXT,
            end_yard_line INTEGER,
            end_yards_to_endzone INTEGER,
            plays INTEGER,
            yards INTEGER,
            time_elapsed_seconds INTEGER,
            time_elapsed_display TEXT,
            result TEXT,
            is_score INTEGER DEFAULT 0,
            description TEXT,
            FOREIGN KEY (game_id) REFERENCES games (game_id),
            FOREIGN KEY (team_id) REFERENCES teams (team_id)
        )
    ''')

    # Create indexes for better query performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_drives_game ON drives(game_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_drives_team ON drives(team_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_drives_result ON drives(result)')

    conn.commit()

    # Verify table was created
    cursor.execute("PRAGMA table_info(drives)")
    columns = cursor.fetchall()

    print(f"\n  [OK] drives table created successfully")
    print(f"\n  Columns ({len(columns)}):")
    for col in columns:
        print(f"    - {col[1]} ({col[2]})")

    # Check if any data exists
    cursor.execute("SELECT COUNT(*) FROM drives")
    count = cursor.fetchone()[0]
    print(f"\n  Current records: {count}")

    conn.close()
    print("\n" + "=" * 80)

def main():
    """Create drives table in both CFB and NFL databases"""

    print("=" * 80)
    print("CREATING DRIVES TABLES")
    print("=" * 80)

    databases = ['cfb_games.db', 'nfl_games.db']

    for db in databases:
        try:
            create_drives_table(db)
        except Exception as e:
            print(f"\n  [ERROR] Failed to create table in {db}: {e}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
