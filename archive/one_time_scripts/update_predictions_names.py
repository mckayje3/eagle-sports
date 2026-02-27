"""
Update prediction_cache to use school names instead of mascots
"""
import sqlite3

def update_predictions():
    """Update team names in prediction_cache"""
    # Get team mappings from cfb_games.db
    cfb_conn = sqlite3.connect('cfb_games.db')
    cursor = cfb_conn.cursor()
    cursor.execute("SELECT team_id, name, school_name FROM teams")
    team_mapping = {row[1]: row[2] for row in cursor.fetchall()}  # mascot -> school_name
    cfb_conn.close()

    print(f"Loaded {len(team_mapping)} team mappings")

    # Update predictions in users.db
    users_conn = sqlite3.connect('users.db')
    cursor = users_conn.cursor()

    # Get all predictions
    cursor.execute("SELECT id, home_team, away_team FROM prediction_cache")
    predictions = cursor.fetchall()

    print(f"\nUpdating {len(predictions)} predictions...")

    updated = 0
    for pred_id, home_team, away_team in predictions:
        new_home = team_mapping.get(home_team, home_team)
        new_away = team_mapping.get(away_team, away_team)

        if new_home != home_team or new_away != away_team:
            cursor.execute("""
                UPDATE prediction_cache
                SET home_team = ?, away_team = ?
                WHERE id = ?
            """, (new_home, new_away, pred_id))
            print(f"  {pred_id}: {away_team} @ {home_team} -> {new_away} @ {new_home}")
            updated += 1

    users_conn.commit()
    users_conn.close()

    print(f"\nDone! Updated {updated} predictions with school names.")

if __name__ == "__main__":
    update_predictions()
