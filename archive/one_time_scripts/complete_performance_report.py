"""
Complete Performance Report for Week 13 2025
CFB and NFL predictions vs actual outcomes
"""
import sqlite3
import pandas as pd
import numpy as np

def analyze_cfb():
    """Analyze CFB predictions"""
    print("=" * 80)
    print("CFB WEEK 13 2025 - PREDICTIONS PERFORMANCE")
    print("=" * 80)

    # Load enhanced predictions
    preds = pd.read_csv('enhanced_predictions_week_13.csv')
    print(f"\nTotal CFB predictions made: {len(preds)}")

    # Load completed games
    conn = sqlite3.connect('cfb_games.db')
    results = pd.read_sql_query("""
        SELECT
            g.game_id,
            g.date,
            ht.name as home_team,
            at.name as away_team,
            g.home_score,
            g.away_score,
            (g.home_score - g.away_score) as actual_margin,
            g.completed,
            CASE
                WHEN g.winner_team_id = g.home_team_id THEN ht.name
                WHEN g.winner_team_id = g.away_team_id THEN at.name
                ELSE 'Tie'
            END as actual_winner,
            go.current_spread_home as vegas_spread
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN game_odds go ON g.game_id = go.game_id
        WHERE g.season = 2025 AND g.week = 13
    """, conn)
    conn.close()

    print(f"Total games in Week 13: {len(results)}")
    print(f"Completed games: {results['completed'].sum()}")
    print(f"Remaining games: {len(results) - results['completed'].sum()}")

    # Match predictions to results
    merged = preds.merge(results, on='game_id', how='inner', suffixes=('_pred', '_db'))
    completed = merged[merged['completed'] == 1].copy()

    # Standardize column names
    if 'home_team_pred' in completed.columns:
        completed['home_team'] = completed['home_team_pred']
        completed['away_team'] = completed['away_team_pred']
    elif 'home_team_db' in completed.columns:
        completed['home_team'] = completed['home_team_db']
        completed['away_team'] = completed['away_team_db']

    print(f"\nGames with results available: {len(completed)}")

    if len(completed) == 0:
        print("\nNo completed games yet to analyze.")
        return None

    # Remove duplicates if any
    completed = completed.drop_duplicates(subset=['game_id'])
    print(f"After removing duplicates: {len(completed)}")

    # Calculate metrics
    completed['correct_winner'] = completed['predicted_winner'] == completed['actual_winner']
    completed['spread_error'] = abs(completed['predicted_spread'] - completed['actual_margin'])

    # Vegas comparison - find the correct column name
    vegas_col = None
    for col in completed.columns:
        if 'vegas' in col.lower() and 'spread' in col.lower():
            vegas_col = col
            break

    if vegas_col and vegas_col != 'vegas_spread':
        completed['vegas_spread'] = completed[vegas_col]
    elif 'vegas_spread' not in completed.columns:
        completed['vegas_spread'] = np.nan

    vegas_mask = ~completed['vegas_spread'].isna()
    completed['vegas_error'] = np.nan
    if vegas_mask.sum() > 0:
        completed.loc[vegas_mask, 'vegas_error'] = abs(
            completed.loc[vegas_mask, 'vegas_spread'] - completed.loc[vegas_mask, 'actual_margin']
        )
    completed['beat_vegas'] = completed['spread_error'] < completed['vegas_error']

    # Print results
    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)

    correct = completed['correct_winner'].sum()
    total = len(completed)
    accuracy = (correct / total) * 100

    print(f"\nWINNER PREDICTIONS:")
    print(f"  Correct: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Incorrect: {total-correct}/{total} ({100-accuracy:.1f}%)")

    print(f"\nSPREAD PREDICTIONS:")
    print(f"  Average error: {completed['spread_error'].mean():.2f} points")
    print(f"  Median error: {completed['spread_error'].median():.2f} points")

    # Vegas comparison
    vegas_games = completed[vegas_mask]
    if len(vegas_games) > 0:
        vegas_avg = vegas_games['vegas_error'].mean()
        model_avg = vegas_games['spread_error'].mean()
        better_count = vegas_games['beat_vegas'].sum()
        diff = vegas_avg - model_avg

        print(f"\nVS VEGAS LINES:")
        print(f"  Games with Vegas lines: {len(vegas_games)}/{total}")
        print(f"  Vegas avg error: {vegas_avg:.2f} pts")
        print(f"  Our avg error: {model_avg:.2f} pts")
        print(f"  Difference: {'+' if diff > 0 else ''}{diff:.2f} pts ({'BETTER' if diff > 0 else 'WORSE'})")
        print(f"  Beat Vegas: {better_count}/{len(vegas_games)} games ({better_count/len(vegas_games)*100:.1f}%)")

    # Game details
    print(f"\n" + "-" * 80)
    print("GAME DETAILS")
    print("-" * 80)

    for _, row in completed.iterrows():
        status = "[CORRECT]" if row['correct_winner'] else "[WRONG]"
        print(f"\n{status} {row['away_team']} @ {row['home_team']}")
        print(f"  Score: {row['away_score']}-{row['home_score']} (Winner: {row['actual_winner']})")
        print(f"  Predicted: {row['predicted_winner']} ({row['confidence']:.1%} confidence)")
        print(f"  Spread error: {row['spread_error']:.2f} pts (predicted {row['predicted_spread']:.1f}, actual {row['actual_margin']:.0f})")
        if not pd.isna(row['vegas_spread']):
            beat = "[BEAT VEGAS]" if row['beat_vegas'] else "[VEGAS CLOSER]"
            print(f"  Vegas: {row['vegas_spread']:.1f} spread, {row['vegas_error']:.2f} error {beat}")

    return completed

def analyze_nfl():
    """Analyze NFL predictions"""
    print("\n\n" + "=" * 80)
    print("NFL WEEK 12 2025 - PREDICTIONS STATUS")
    print("=" * 80)

    # Check for NFL predictions
    try:
        # Check upcoming predictions
        upcoming = pd.read_csv('upcoming_game_predictions.csv')
        print(f"\nFound {len(upcoming)} predictions in upcoming_game_predictions.csv")
        print("Note: These appear to be older/simpler predictions (all same confidence)")

        # Check NFL database
        conn = sqlite3.connect('nfl_games.db')
        nfl_games = pd.read_sql_query("""
            SELECT season, week, COUNT(*) as total, SUM(completed) as completed
            FROM games
            WHERE season = 2025 AND week IN (11, 12)
            GROUP BY season, week
            ORDER BY week DESC
        """, conn)
        conn.close()

        print("\nNFL games in database:")
        print(nfl_games.to_string())

        # Try to match
        conn = sqlite3.connect('nfl_games.db')
        recent = pd.read_sql_query("""
            SELECT g.game_id, g.week, ht.name as home, at.name as away,
                   g.home_score, g.away_score, g.completed
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.season = 2025 AND g.week = 12 AND g.completed = 1
        """, conn)
        conn.close()

        if len(recent) > 0:
            print(f"\n{len(recent)} NFL games completed in Week 12:")
            print(recent[['week', 'away', 'home', 'away_score', 'home_score']].to_string())
        else:
            print("\nNo completed NFL games found for Week 12")

    except FileNotFoundError:
        print("\nNo NFL predictions CSV found")
    except Exception as e:
        print(f"\nError checking NFL: {e}")

def main():
    cfb_results = analyze_cfb()
    analyze_nfl()

    if cfb_results is not None and len(cfb_results) > 0:
        # Save summary
        output = cfb_results[[
            'game_id', 'home_team', 'away_team',
            'predicted_winner', 'confidence', 'predicted_spread',
            'home_score', 'away_score', 'actual_winner', 'actual_margin',
            'correct_winner', 'spread_error',
            'vegas_spread', 'vegas_error', 'beat_vegas'
        ]].copy()

        output.to_csv('week_13_complete_report.csv', index=False)
        print("\n\n" + "=" * 80)
        print("Report saved to: week_13_complete_report.csv")
        print("=" * 80)

if __name__ == '__main__':
    main()
