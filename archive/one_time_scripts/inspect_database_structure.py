"""
Complete database structure inspection
Show all tables and columns across all databases
"""
import sqlite3
import pandas as pd

def inspect_database(db_path, db_name):
    """Inspect a single database and show all tables/columns"""
    print("\n" + "=" * 80)
    print(f"{db_name.upper()}")
    print("=" * 80)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()

        if not tables:
            print(f"No tables found in {db_name}")
            conn.close()
            return

        print(f"\nFound {len(tables)} tables\n")

        for table in tables:
            table_name = table[0]
            print("-" * 80)
            print(f"TABLE: {table_name}")
            print("-" * 80)

            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            print("\nColumns:")
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, pk = col
                pk_marker = " [PRIMARY KEY]" if pk else ""
                null_marker = " NOT NULL" if not_null else ""
                default_marker = f" DEFAULT {default_val}" if default_val else ""
                print(f"  {col_name:25} {col_type:15}{pk_marker}{null_marker}{default_marker}")

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"\nTotal rows: {count:,}")

            # Show sample data if rows exist
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                sample = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]

                print("\nSample data (first 3 rows):")
                df = pd.DataFrame(sample, columns=col_names)
                print(df.to_string())

            print("\n")

        conn.close()

    except Exception as e:
        print(f"Error inspecting {db_name}: {e}")

def main():
    databases = [
        ('cfb_games.db', 'CFB Games Database'),
        ('nfl_games.db', 'NFL Games Database'),
        ('users.db', 'Users/Predictions Database'),
        ('sports_predictions.db', 'Sports Predictions Database')
    ]

    print("=" * 80)
    print("DATABASE STRUCTURE INSPECTION")
    print("=" * 80)

    for db_path, db_name in databases:
        try:
            inspect_database(db_path, db_name)
        except FileNotFoundError:
            print(f"\n{db_name}: File not found - {db_path}")
        except Exception as e:
            print(f"\n{db_name}: Error - {e}")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
