"""
Analyze prediction performance compared to actual outcomes and Vegas lines
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def analyze_performance():
    """Analyze predictions vs actual outcomes for last week"""

    # Connect to databases
    cfb_conn = sqlite3.connect('cfb_games.db')
    nfl_conn = sqlite3.connect('nfl_games.db')
    users_conn = sqlite3.connect('users.db')

    # Check what tables exist in users.db
    print("=" * 80)
    print("Checking users.db structure...")
    print("=" * 80)
    users_cursor = users_conn.cursor()
    tables = users_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    print(f"Tables in users.db: {[t[0] for t in tables]}\n")

    # Check if predictions table exists
    if ('predictions',) in tables:
        # Get schema of predictions table
        schema = users_cursor.execute("PRAGMA table_info(predictions);").fetchall()
        print("Predictions table schema:")
        for col in schema:
            print(f"  {col[1]} ({col[2]})")
        print()

        # Check predictions data
        pred_query = """
            SELECT * FROM predictions
            ORDER BY prediction_date DESC
            LIMIT 5
        """
        predictions_sample = pd.read_sql_query(pred_query, users_conn)
        print(f"Sample predictions (latest 5):")
        print(predictions_sample.to_string())
        print()

    # Get CFB games from last week (Week 13, around Nov 23, 2024)
    print("\n" + "=" * 80)
    print("CFB GAMES ANALYSIS - Week 13")
    print("=" * 80)

    cfb_query = """
        SELECT
            g.game_id,
            g.week,
            g.date,
            g.completed,
            ht.name as home_team,
            at.name as away_team,
            g.home_score,
            g.away_score,
            CASE
                WHEN g.winner_team_id = g.home_team_id THEN ht.name
                WHEN g.winner_team_id = g.away_team_id THEN at.name
                ELSE NULL
            END as actual_winner,
            go.current_spread_home as vegas_spread,
            go.closing_spread_home as vegas_closing_spread
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN game_odds go ON g.game_id = go.game_id
        WHERE g.season = 2024
        AND g.week = 13
        AND g.completed = 1
        ORDER BY g.date DESC
    """

    cfb_games = pd.read_sql_query(cfb_query, cfb_conn)
    print(f"\nTotal completed CFB games in Week 13: {len(cfb_games)}")
    if len(cfb_games) > 0:
        print("\nSample of completed CFB games:")
        print(cfb_games.head(10).to_string())

    # Get NFL games from last week
    print("\n" + "=" * 80)
    print("NFL GAMES ANALYSIS - Recent Week")
    print("=" * 80)

    nfl_query = """
        SELECT
            g.game_id,
            g.week,
            g.date,
            g.completed,
            ht.name as home_team,
            at.name as away_team,
            g.home_score,
            g.away_score,
            CASE
                WHEN g.winner_team_id = g.home_team_id THEN ht.name
                WHEN g.winner_team_id = g.away_team_id THEN at.name
                ELSE NULL
            END as actual_winner,
            go.current_spread_home as vegas_spread,
            go.closing_spread_home as vegas_closing_spread
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN game_odds go ON g.game_id = go.game_id
        WHERE g.season = 2024
        AND g.completed = 1
        ORDER BY g.date DESC
        LIMIT 20
    """

    nfl_games = pd.read_sql_query(nfl_query, nfl_conn)
    print(f"\nTotal recent completed NFL games: {len(nfl_games)}")
    if len(nfl_games) > 0:
        print("\nSample of completed NFL games:")
        print(nfl_games.head(10).to_string())

    # Load predictions from CSV
    print("\n" + "=" * 80)
    print("PREDICTIONS FROM CSV FILES")
    print("=" * 80)

    try:
        enhanced_preds = pd.read_csv('enhanced_predictions_week_13.csv')
        print(f"\nEnhanced predictions Week 13: {len(enhanced_preds)} games")
        print("\nSample:")
        print(enhanced_preds.head(10).to_string())
    except Exception as e:
        print(f"Could not load enhanced predictions: {e}")

    # Close connections
    cfb_conn.close()
    nfl_conn.close()
    users_conn.close()

    print("\n" + "=" * 80)
    print("Analysis complete - ready to match predictions with outcomes")
    print("=" * 80)

if __name__ == '__main__':
    analyze_performance()
