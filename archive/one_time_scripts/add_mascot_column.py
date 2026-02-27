"""
Add mascot_name column to teams table (without dropping anything)
"""
import sqlite3

def extract_mascot(display_name):
    """Extract mascot from display_name"""
    if not display_name or display_name == "TBD":
        return None

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

    for mascot in mascots:
        if display_name.endswith(mascot):
            return mascot

    # Try last word as fallback
    parts = display_name.strip().split()
    if len(parts) > 1:
        return parts[-1]

    return None

def main():
    print("Adding mascot_name column to teams table...")

    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    # Check if column exists
    cursor.execute("PRAGMA table_info(teams);")
    columns = [col[1] for col in cursor.fetchall()]

    if 'mascot_name' in columns:
        print("mascot_name column already exists")
    else:
        cursor.execute("ALTER TABLE teams ADD COLUMN mascot_name TEXT")
        print("Added mascot_name column")

    # Update mascot names
    cursor.execute("SELECT team_id, display_name FROM teams WHERE team_id > 0")
    teams = cursor.fetchall()

    print(f"\nUpdating mascot names for {len(teams)} teams...")

    updates = []
    for team_id, display_name in teams:
        mascot = extract_mascot(display_name)
        if mascot:
            updates.append((mascot, team_id))

    if updates:
        cursor.executemany("UPDATE teams SET mascot_name = ? WHERE team_id = ?", updates)
        conn.commit()
        print(f"Updated {len(updates)} teams")

        # Show sample
        cursor.execute("""
            SELECT team_id, name, mascot_name, display_name
            FROM teams
            WHERE team_id > 0
            ORDER BY name
            LIMIT 10
        """)

        print("\nSample data:")
        print("-" * 80)
        import pandas as pd
        df = pd.DataFrame(cursor.fetchall(), columns=['team_id', 'name', 'mascot', 'display_name'])
        print(df.to_string())
    else:
        print("No teams to update")

    conn.close()
    print("\nDone!")

if __name__ == '__main__':
    main()
