"""
Week 13 2025 CFB Predictions Performance Report
Analyze predictions vs actual outcomes for completed games
"""
import sqlite3
import pandas as pd
import numpy as np

def main():
    print("="  * 80)
    print("WEEK 13 2025 CFB PREDICTIONS - PERFORMANCE REPORT")
    print("=" * 80)

    # Load predictions
    print("\nLoading predictions...")
    preds = pd.read_csv('enhanced_predictions_week_13.csv')
    print(f"  Loaded {len(preds)} predictions")

    # Load completed games from 2025 Week 13
    print("Loading completed 2025 Week 13 games...")
    conn = sqlite3.connect('cfb_games.db')

    query = """
        SELECT
            g.game_id,
            g.week,
            g.date,
            ht.name as home_team,
            at.name as away_team,
            g.home_score,
            g.away_score,
            (g.home_score - g.away_score) as actual_margin,
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
        WHERE g.season = 2025
        AND g.week = 13
        AND g.completed = 1
    """

    results = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Loaded {len(results)} completed games")

    if len(results) == 0:
        print("\nNo completed games found for 2025 Week 13 yet.")
        print("Games are scheduled for later this week.")
        return

    # Match by game_id
    print("\nMatching predictions to results...")
    merged = preds.merge(results, on='game_id', how='inner', suffixes=('_pred', '_actual'))
    print(f"  Matched {len(merged)} games")

    if len(merged) == 0:
        print("\nNo matches found. Predictions may be for games not yet completed.")
        print(f"\nCompleted games ({len(results)}):")
        print(results[['home_team', 'away_team', 'home_score', 'away_score']].to_string())
        return

    # Calculate metrics
    merged['correct_winner'] = merged['predicted_winner'] == merged['actual_winner']
    merged['spread_error'] = abs(merged['predicted_spread'] - merged['actual_margin'])

    # Vegas comparison - handle column naming
    vegas_col = None
    for col in merged.columns:
        if 'vegas' in col.lower() and 'spread' in col.lower():
            vegas_col = col
            break

    if vegas_col:
        vegas_mask = ~merged[vegas_col].isna()
        merged['vegas_error'] = np.nan
        merged.loc[vegas_mask, 'vegas_error'] = abs(
            merged.loc[vegas_mask, vegas_col] - merged.loc[vegas_mask, 'actual_margin']
        )
        merged['model_better'] = merged['spread_error'] < merged['vegas_error']
        # Rename for consistency
        merged['vegas_spread'] = merged[vegas_col]
    else:
        vegas_mask = pd.Series([False] * len(merged))
        merged['vegas_error'] = np.nan
        merged['vegas_spread'] = np.nan
        merged['model_better'] = False

    # GENERATE REPORT
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    total = len(merged)
    correct = merged['correct_winner'].sum()
    accuracy = (correct / total) * 100

    print(f"\nWINNER PREDICTIONS")
    print(f"  Games analyzed: {total}")
    print(f"  Correct: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Incorrect: {total-correct}/{total} ({100-accuracy:.1f}%)")

    # Spread accuracy
    avg_error = merged['spread_error'].mean()
    median_error = merged['spread_error'].median()

    print(f"\nSPREAD PREDICTIONS")
    print(f"  Average error: {avg_error:.2f} points")
    print(f"  Median error: {median_error:.2f} points")

    # Vegas comparison
    vegas_games = merged[vegas_mask]
    if len(vegas_games) > 0:
        vegas_avg = vegas_games['vegas_error'].mean()
        model_avg = vegas_games['spread_error'].mean()
        better_count = vegas_games['model_better'].sum()
        better_pct = (better_count / len(vegas_games)) * 100

        print(f"\nCOMPARISON TO VEGAS")
        print(f"  Games with Vegas lines: {len(vegas_games)}")
        print(f"  Vegas avg error: {vegas_avg:.2f} pts")
        print(f"  Our avg error: {model_avg:.2f} pts")

        diff = vegas_avg - model_avg
        if diff > 0:
            print(f"  Result: We were {diff:.2f} pts BETTER than Vegas")
        elif diff < 0:
            print(f"  Result: We were {abs(diff):.2f} pts WORSE than Vegas")
        else:
            print(f"  Result: TIED with Vegas")

        print(f"  More accurate than Vegas: {better_count}/{len(vegas_games)} games ({better_pct:.1f}%)")

    # Game details
    print(f"\n" + "=" * 80)
    print("GAME-BY-GAME RESULTS")
    print("=" * 80)

    for _, row in merged.iterrows():
        status = "CORRECT" if row['correct_winner'] else "WRONG"
        print(f"\n{row['away_team_pred']} @ {row['home_team_pred']}")
        print(f"  Predicted: {row['predicted_winner']} (confidence: {row['confidence']:.1%})")
        print(f"  Actual: {row['actual_winner']} won {row['away_score']}-{row['home_score']}")
        print(f"  Spread - Predicted: {row['predicted_spread']:.1f}, Actual: {row['actual_margin']:.0f}, Error: {row['spread_error']:.2f}")
        if not pd.isna(row['vegas_spread']):
            print(f"  Vegas spread: {row['vegas_spread']:.1f}, Vegas error: {row['vegas_error']:.2f}")
        print(f"  Status: [{status}]")

    # Save results
    output = merged[[
        'game_id', 'home_team_pred', 'away_team_pred',
        'predicted_winner', 'confidence', 'predicted_spread',
        'home_score', 'away_score', 'actual_winner', 'actual_margin',
        'correct_winner', 'spread_error',
        'vegas_spread', 'vegas_error', 'model_better'
    ]].copy()

    output.to_csv('week_13_2025_performance.csv', index=False)
    print(f"\n" + "=" * 80)
    print(f"Detailed results saved to: week_13_2025_performance.csv")
    print("=" * 80)

    # Check for NFL as well
    print("\n\nChecking NFL predictions...")
    check_nfl_predictions()

def check_nfl_predictions():
    """Check if there are NFL predictions to analyze"""
    try:
        # Check for NFL database
        conn = sqlite3.connect('nfl_games.db')
        nfl_query = """
            SELECT COUNT(*) as count
            FROM games
            WHERE completed = 1
            ORDER BY date DESC
            LIMIT 1
        """
        result = pd.read_sql_query(nfl_query, conn)
        conn.close()

        if result['count'][0] > 0:
            print("  NFL games found in database")
            print("  Note: No NFL predictions CSV found in current analysis")
        else:
            print("  No completed NFL games found")
    except:
        print("  NFL database not available")

if __name__ == '__main__':
    main()
