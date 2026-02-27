"""
Update teams table to add school_name column (without mascot)
"""
import sqlite3

def extract_school_name(display_name, mascot):
    """Extract school name from display_name by removing mascot"""
    if not display_name or display_name == "TBD":
        return display_name

    # Remove the mascot from the end of display_name
    # e.g., "Auburn Tigers" -> "Auburn"
    # e.g., "South Alabama Jaguars" -> "South Alabama"
    if mascot and mascot in display_name:
        school_name = display_name.replace(mascot, "").strip()
        return school_name

    return display_name

def update_teams_table():
    """Add school_name column and populate it"""
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    # Add school_name column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE teams ADD COLUMN school_name TEXT")
        print("Added school_name column")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("school_name column already exists")
        else:
            raise

    # Get all teams
    cursor.execute("SELECT team_id, name, display_name FROM teams")
    teams = cursor.fetchall()

    print(f"\nUpdating {len(teams)} teams...")

    # Update each team with school name
    for team_id, mascot, display_name in teams:
        school_name = extract_school_name(display_name, mascot)
        cursor.execute("UPDATE teams SET school_name = ? WHERE team_id = ?",
                      (school_name, team_id))
        print(f"  {team_id}: {display_name} -> {school_name}")

    conn.commit()
    conn.close()
    print("\nDone! School names updated.")

if __name__ == "__main__":
    update_teams_table()
