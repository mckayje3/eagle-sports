"""
Update teams table schema:
1. Remove school_name column (now duplicative of name)
2. Add mascot_name column to store the mascot
"""
import sqlite3

def extract_mascot(display_name):
    """
    Extract mascot from display_name
    Examples:
        "Oregon Ducks" -> "Ducks"
        "Texas A&M Aggies" -> "Aggies"
        "USC Trojans" -> "Trojans"
    """
    if not display_name or display_name == "TBD":
        return None

    # List of all mascots
    mascots = [
        'Aggies', 'Aztecs', 'Badgers', 'Bearcats', 'Bearkats', 'Bears', 'Beavers',
        'Bengals', 'Bison', 'Black Knights', 'Blazers', 'Blue Devils', 'Blue Hens',
        'Blue Raiders', 'Bobcats', 'Boilermakers', 'Braves', 'Broncos', 'Bruins',
        'Buccaneers', 'Buckeyes', 'Buffaloes', 'Bulldogs', 'Bulls', 'Cardinals',
        'Catamounts', 'Cavaliers', 'Chanticleers', 'Chippewas', 'Colonels',
        'Colonials', 'Commodores', 'Cougars', 'Cowboys', 'Coyotes', 'Crimson Tide',
        'Crusaders', 'Cyclones', 'Demon Deacons', 'Demons', 'Ducks', 'Dukes',
        'Eagles', 'Falcons', 'Fighting Camels', 'Fighting Hawks', 'Fighting Illini',
        'Fighting Irish', 'Flames', 'Flashes', 'Friars', 'Gators', 'Golden Bears',
        'Golden Eagles', 'Golden Flashes', 'Golden Gophers', 'Golden Hurricane',
        'Golden Lions', 'Governors', 'Great Danes', 'Green Wave', 'Grizzlies',
        'Hardrockers', 'Hatters', 'Hawkeyes', 'Hawks', 'Hilltoppers', 'Hokies',
        'Hoosiers', 'Horned Frogs', 'Hornets', 'Huskies', 'Hurricanes', 'Jaguars',
        'Jackrabbits', 'Javelinas', 'Jayhawks', 'Jets', 'Judges', 'Keydets',
        'Knights', 'Leathernecks', 'Leopards', 'Lions', 'Lobos', 'Lumberjacks',
        'Mean Green', 'Midshipmen', 'Miners', 'Minutemen', 'Mocs', 'Monarchs',
        'Monks', 'Mountain Hawks', 'Mountaineers', 'Musketeers', 'Mustangs',
        'Nittany Lions', 'Nor\'easters', 'Orange', 'Ospreys', 'Owls',
        'Paladins', 'Panthers', 'Patriots', 'Penguins', 'Phoenix', 'Pioneers',
        'Pirates', 'Privateers', 'Purple Aces', 'Quakers', 'Racers',
        'Rainbow Warriors', 'Ragin\' Cajuns', 'Raiders', 'Rams', 'Rattlers',
        'Ravens', 'Razorbacks', 'Rebels', 'Red Flash', 'Red Raiders',
        'Red Wolves', 'Redbirds', 'Redhawks', 'RedHawks', 'Retrievers',
        'Roadrunners', 'Rockets', 'Roughriders', 'Runnin\' Bulldogs',
        'Running Rebels', 'Salukis', 'Scarlet Knights', 'Seawolves',
        'Seminoles', 'Sharks', 'Skyhawks', 'Sooners', 'Spartans', 'Spider',
        'Spiders', 'Stags', 'Sun Devils', 'Sycamores', 'Tar Heels', 'Terrapins',
        'Terriers', 'Texans', 'Thunderbirds', 'Thundering Herd', 'Tigers',
        'Titans', 'Tomahawks', 'Toreros', 'Trailblazers', 'Tribe', 'Tritons',
        'Trojans', 'Utes', 'Vandals', 'Vikings', 'Volunteers', 'Warhawks',
        'Warriors', 'Waves', 'Wildcats', 'Wolf Pack', 'Wolfpack', 'Wolverines',
        'Wranglers', 'Yellow Jackets', 'Zips', '49ers', 'Cardinal'
    ]

    # Try to find mascot at end of display name
    for mascot in mascots:
        if display_name.endswith(mascot):
            return mascot

    # If no match, try to get last word
    parts = display_name.strip().split()
    if len(parts) > 1:
        return parts[-1]

    return None

def update_schema():
    """Update the teams table schema"""
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    print("Updating teams table schema...")
    print("-" * 80)

    # Step 1: Check if school_name column exists and drop it
    cursor.execute("PRAGMA table_info(teams);")
    columns = cursor.fetchall()
    col_names = [col[1] for col in columns]

    if 'school_name' in col_names:
        print("\n1. Removing 'school_name' column (now duplicative)...")

        # SQLite doesn't support DROP COLUMN directly, so we need to recreate the table
        # First, get all data
        cursor.execute("SELECT team_id, name, abbreviation, display_name, logo_url, color, conference FROM teams")
        teams_data = cursor.fetchall()

        # Drop the old table
        cursor.execute("DROP TABLE teams")

        # Create new table without school_name
        cursor.execute('''
            CREATE TABLE teams (
                team_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                abbreviation TEXT,
                display_name TEXT,
                logo_url TEXT,
                color TEXT,
                conference TEXT
            )
        ''')

        # Restore data
        cursor.executemany('''
            INSERT INTO teams (team_id, name, abbreviation, display_name, logo_url, color, conference)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', teams_data)

        print("   [OK] Removed 'school_name' column")
    else:
        print("\n1. 'school_name' column does not exist - skipping removal")

    # Step 2: Add mascot_name column
    cursor.execute("PRAGMA table_info(teams);")
    columns = cursor.fetchall()
    col_names = [col[1] for col in columns]

    if 'mascot_name' not in col_names:
        print("\n2. Adding 'mascot_name' column...")
        cursor.execute("ALTER TABLE teams ADD COLUMN mascot_name TEXT")
        print("   [OK] Added 'mascot_name' column")
    else:
        print("\n2. 'mascot_name' column already exists")

    # Step 3: Populate mascot_name from display_name
    print("\n3. Populating mascot names...")
    cursor.execute("SELECT team_id, display_name FROM teams WHERE team_id > 0")
    teams = cursor.fetchall()

    updates = []
    for team_id, display_name in teams:
        mascot = extract_mascot(display_name)
        if mascot:
            updates.append((mascot, team_id))

    cursor.executemany("UPDATE teams SET mascot_name = ? WHERE team_id = ?", updates)

    print(f"   [OK] Updated {len(updates)} teams with mascot names")

    conn.commit()

    # Step 4: Show sample of updated data
    print("\n4. Sample data after update:")
    print("-" * 80)
    cursor.execute("""
        SELECT team_id, name, mascot_name, display_name
        FROM teams
        WHERE team_id > 0
        ORDER BY name
        LIMIT 10
    """)

    import pandas as pd
    df = pd.DataFrame(cursor.fetchall(), columns=['team_id', 'name', 'mascot_name', 'display_name'])
    print(df.to_string())

    # Verify final schema
    print("\n5. Final schema:")
    print("-" * 80)
    cursor.execute("PRAGMA table_info(teams);")
    columns = cursor.fetchall()

    print(f"{'#':<5} {'Column Name':<20} {'Type':<15} {'Not Null':<10} {'PK':<5}")
    print("-" * 60)
    for col in columns:
        col_id, col_name, col_type, not_null, default_val, pk = col
        print(f"{col_id:<5} {col_name:<20} {col_type:<15} {'YES' if not_null else 'NO':<10} {'YES' if pk else '':<5}")

    conn.close()

    print("\n" + "=" * 80)
    print("Schema update complete!")
    print("=" * 80)

if __name__ == '__main__':
    update_schema()
