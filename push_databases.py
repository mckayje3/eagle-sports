"""
Push databases to GitHub after data updates.
Run this after scraping or backfilling data to sync with Streamlit Cloud.
"""
import subprocess
import sys
from datetime import datetime


def run_command(cmd, capture=True):
    """Run a shell command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if result.returncode != 0 and capture:
        print(f"Error: {result.stderr}")
    return result


def get_db_stats():
    """Get quick stats from each database"""
    import sqlite3
    stats = {}

    dbs = {
        'nfl_games.db': 'NFL',
        'cfb_games.db': 'CFB',
        'nba_games.db': 'NBA',
        'cbb_games.db': 'CBB'
    }

    for db_file, name in dbs.items():
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM games')
            games = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM games WHERE completed = 1')
            completed = cursor.fetchone()[0]
            stats[name] = f"{completed}/{games} games"
            conn.close()
        except Exception as e:
            stats[name] = f"Error: {e}"

    return stats


def main():
    print("=" * 60)
    print("PUSHING DATABASES TO GITHUB")
    print("=" * 60)

    # Get current stats
    print("\nDatabase Status:")
    stats = get_db_stats()
    for name, stat in stats.items():
        print(f"  {name}: {stat}")

    # Stage database files
    print("\nStaging databases...")
    run_command("git add nfl_games.db cfb_games.db nba_games.db cbb_games.db")

    # Check if there are changes
    result = run_command("git diff --cached --name-only")
    if not result.stdout.strip():
        print("\nNo database changes to commit.")
        return

    print(f"Changed files: {result.stdout.strip()}")

    # Create commit message with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    stats_summary = ", ".join([f"{k}: {v.split('/')[0]}" for k, v in stats.items()])
    commit_msg = f"Update databases ({timestamp})\n\n{stats_summary}"

    # Commit
    print("\nCommitting...")
    run_command(f'git commit -m "{commit_msg}"')

    # Push
    print("\nPushing to GitHub...")
    result = run_command("git push")

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("SUCCESS! Databases pushed to GitHub.")
        print("Streamlit Cloud will auto-deploy with new data.")
        print("=" * 60)
    else:
        print("\nPush failed. Check your connection or credentials.")


if __name__ == '__main__':
    main()
