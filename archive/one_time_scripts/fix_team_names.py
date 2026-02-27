"""
Update the 'name' field in teams table to use school name instead of mascot
This makes the name field unique and more useful for matching
"""
import sqlite3
import re

def extract_school_name(display_name):
    """
    Extract school name from display_name
    Examples:
        "Oregon Ducks" -> "Oregon"
        "Texas A&M Aggies" -> "Texas A&M"
        "USC Trojans" -> "USC"
    """
    if not display_name or display_name == "TBD":
        return display_name

    # Remove common mascot suffixes
    mascots = [
        'Aggies', 'Aztecs', 'Badgers', 'Bearcats', 'Bearkats', 'Bears', 'Beavers',
        'Bengals', 'Bison', 'Black Knights', 'Blazers', 'Blue Devils', 'Blue Hens',
        'Blue Raiders', 'Bobcats', 'Boilermakers', 'Broncos', 'Bruins', 'Buckeyes',
        'Buffaloes', 'Bulls', 'Bulldogs', 'Cardinals', 'Catamounts', 'Cavaliers',
        'Chanticleers', 'Chippewas', 'Colonels', 'Commodores', 'Cougars', 'Cowboys',
        'Crimson Tide', 'Crusaders', 'Cyclones', 'Demon Deacons', 'Ducks', 'Dukes',
        'Eagles', 'Falcons', 'Fighting Hawks', 'Fighting Illini', 'Fighting Irish',
        'Flames', 'Flashes', 'Friars', 'Gators', 'Golden Bears', 'Golden Eagles',
        'Golden Flashes', 'Golden Gophers', 'Golden Hurricane', 'Governors', 'Grizzlies',
        'Hardrockers', 'Hatters', 'Hawkeyes', 'Hilltoppers', 'Hokies', 'Hoosiers',
        'Horned Frogs', 'Hornets', 'Huskies', 'Hurricanes', 'Jaguars', 'Javelinas',
        'Jayhawks', 'Jets', 'Judges', 'Knights', 'Leopards', 'Lions', 'Lumberjacks',
        'Mean Green', 'Midshipmen', 'Miners', 'Minutemen', 'Monarchs', 'Monks',
        'Mountaineers', 'Musketeers', 'Mustangs', 'Nittany Lions', 'Nor\'easters',
        'Ospreys', 'Owls', 'Panthers', 'Patriots', 'Paladins', 'Phoenix', 'Pirates',
        'Pioneers', 'Privateers', 'Purple Aces', 'Racers', 'Rainbow Warriors',
        'Ragin\' Cajuns', 'Rams', 'Rattlers', 'Ravens', 'Razorbacks', 'Rebels',
        'Red Flash', 'Red Raiders', 'Red Wolves', 'Redbirds', 'Redhawks', 'RedHawks',
        'Retrievers', 'Rockets', 'Roadrunners', 'Roughriders', 'Running Rebels',
        'Salukis', 'Scarlet Knights', 'Seminoles', 'Seawolves', '49ers', 'Sooners',
        'Spartans', 'Spider', 'Stags', 'Sun Devils', 'Tar Heels', 'Terrapins',
        'Terriers', 'Thunderbirds', 'Thundering Herd', 'Tigers', 'Titans', 'Tomahawks',
        'Toreros', 'Tribe', 'Tritons', 'Trojans', 'Utes', 'Vandals', 'Vikings',
        'Volunteers', 'Warhawks', 'Warriors', 'Waves', 'Wildcats', 'Wolf Pack',
        'Wolfpack', 'Wolverines', 'Wranglers', 'Yellow Jackets', 'Zips'
    ]

    # Try to match and remove mascot from end
    for mascot in mascots:
        if display_name.endswith(mascot):
            school = display_name[:-len(mascot)].strip()
            if school:  # Make sure we have something left
                return school

    # If no match, just return the first word(s) before the last word
    parts = display_name.strip().split()
    if len(parts) > 1:
        return ' '.join(parts[:-1])

    return display_name

def update_team_names():
    """Update all team names in the database"""
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    # Get all teams
    cursor.execute("SELECT team_id, name, display_name FROM teams WHERE team_id > 0")
    teams = cursor.fetchall()

    print(f"Updating {len(teams)} teams...")
    print("\nChanges:")
    print("-" * 80)

    updates = []
    for team_id, old_name, display_name in teams:
        new_name = extract_school_name(display_name)

        if new_name != old_name:
            updates.append((new_name, team_id))
            print(f"{team_id:4d} | {old_name:20s} -> {new_name:20s} | ({display_name})")

    if updates:
        print(f"\n{len(updates)} teams will be updated.")

        # Update the database
        cursor.executemany("UPDATE teams SET name = ? WHERE team_id = ?", updates)
        conn.commit()

        print("\nDatabase updated successfully!")

        # Verify no duplicates
        cursor.execute("""
            SELECT name, COUNT(*) as count
            FROM teams
            WHERE team_id > 0
            GROUP BY name
            HAVING count > 1
        """)
        dupes = cursor.fetchall()

        if dupes:
            print(f"\nWarning: {len(dupes)} duplicate names still exist:")
            for name, count in dupes:
                print(f"  {name}: {count} occurrences")
                # Show which teams
                cursor.execute("SELECT team_id, display_name FROM teams WHERE name = ?", (name,))
                dupe_teams = cursor.fetchall()
                for tid, dname in dupe_teams:
                    print(f"    - {tid}: {dname}")
        else:
            print("\nAll team names are now unique!")
    else:
        print("\nNo updates needed - all names are already correct.")

    conn.close()

if __name__ == '__main__':
    update_team_names()
