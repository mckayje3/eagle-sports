"""
Add weather-related columns to games table
"""
import sqlite3

def add_weather_columns():
    """Add weather columns to both CFB and NFL databases"""

    databases = ['cfb_games.db', 'nfl_games.db']

    for db_name in databases:
        print(f"\n{'='*60}")
        print(f"Updating {db_name}")
        print('='*60)

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Check existing columns
        cursor.execute("PRAGMA table_info(games);")
        existing_columns = [col[1] for col in cursor.fetchall()]

        # Columns to add
        new_columns = {
            'temperature': 'INTEGER',      # Temperature in Fahrenheit
            'wind_speed': 'INTEGER',       # Wind speed in MPH
            'conditions': 'TEXT',          # Weather description (Clear, Rain, Snow, etc.)
            'is_dome': 'INTEGER'           # 1=indoor/dome, 0=outdoor
        }

        added = []
        skipped = []

        for col_name, col_type in new_columns.items():
            if col_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE games ADD COLUMN {col_name} {col_type}")
                    added.append(col_name)
                    print(f"  Added: {col_name} ({col_type})")
                except sqlite3.OperationalError as e:
                    print(f"  Error adding {col_name}: {e}")
            else:
                skipped.append(col_name)
                print(f"  Skipped: {col_name} (already exists)")

        conn.commit()

        # Show updated schema
        cursor.execute("PRAGMA table_info(games);")
        columns = cursor.fetchall()

        print(f"\n  Weather columns in games table:")
        for col in columns:
            if col[1] in new_columns.keys():
                print(f"    - {col[1]} ({col[2]})")

        # Check if any games already have weather data
        cursor.execute("SELECT COUNT(*) FROM games WHERE temperature IS NOT NULL")
        with_weather = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM games")
        total_games = cursor.fetchone()[0]

        print(f"\n  Games with weather data: {with_weather}/{total_games}")

        conn.close()
        print(f"\n  [OK] {db_name} updated successfully")

    print("\n" + "="*60)
    print("Weather columns added to all databases!")
    print("="*60)

if __name__ == '__main__':
    add_weather_columns()
