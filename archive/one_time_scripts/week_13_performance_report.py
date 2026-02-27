"""
Complete performance analysis for Week 13 predictions
Compares predictions to actual outcomes and Vegas lines
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def load_predictions():
    """Load predictions from CSV"""
    preds = pd.read_csv('enhanced_predictions_week_13.csv')
    print(f"Loaded {len(preds)} predictions")
    return preds

def load_actual_results():
    """Load actual game results from database"""
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
            CASE
                WHEN g.winner_team_id = g.home_team_id THEN ht.name
                WHEN g.winner_team_id = g.away_team_id THEN at.name
                ELSE 'Tie'
            END as actual_winner,
            (g.home_score - g.away_score) as actual_margin,
            go.current_spread_home as vegas_spread,
            go.closing_spread_home as vegas_closing_spread
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN game_odds go ON g.game_id = go.game_id
        WHERE g.season = 2024
        AND g.week = 13
        AND g.completed = 1
    """

    results = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(results)} completed games")
    return results

def match_predictions_to_results(predictions, results):
    """Match predictions with actual outcomes"""

    # First try to merge on game_id
    merged = predictions.merge(
        results,
        on='game_id',
        how='inner',
        suffixes=('_pred', '_actual')
    )

    print(f"\nMatched {len(merged)} games on game_id")

    # If no matches, try matching on team names
    if len(merged) == 0:
        print("No game_id matches found, trying team name matching...")
        merged = predictions.merge(
            results,
            left_on=['home_team', 'away_team'],
            right_on=['home_team', 'away_team'],
            how='inner',
            suffixes=('_pred', '_actual')
        )
        print(f"Matched {len(merged)} games on team names")

    if len(merged) == 0:
        print("\nERROR: No matches found!")
        print("\nSample prediction teams:")
        print(predictions[['home_team', 'away_team']].head())
        print("\nSample result teams:")
        print(results[['home_team', 'away_team']].head())
        return merged

    # Calculate metrics
    merged['prediction_correct'] = (
        merged['predicted_winner'] == merged['actual_winner']
    )

    # Calculate spread prediction error (predicted spread vs actual margin)
    merged['spread_error'] = abs(
        merged['predicted_spread'] - merged['actual_margin']
    )

    # Handle vegas_spread column name
    vegas_col = 'vegas_spread' if 'vegas_spread' in merged.columns else 'vegas_spread_pred'
    if vegas_col not in merged.columns:
        # Check for other possible column names
        possible_cols = [c for c in merged.columns if 'vegas' in c.lower() and 'spread' in c.lower()]
        if possible_cols:
            vegas_col = possible_cols[0]
        else:
            merged['vegas_spread_final'] = np.nan
            vegas_col = 'vegas_spread_final'

    # Calculate Vegas spread error (if available)
    merged['vegas_error'] = np.nan
    mask = ~merged[vegas_col].isna()
    if mask.sum() > 0:
        merged.loc[mask, 'vegas_error'] = abs(
            merged.loc[mask, vegas_col] - merged.loc[mask, 'actual_margin']
        )

    # Determine who was more accurate (model or Vegas)
    merged['model_more_accurate'] = merged['spread_error'] < merged['vegas_error']

    # Rename for consistency
    if vegas_col != 'vegas_spread_final':
        merged['vegas_spread_final'] = merged[vegas_col]

    return merged

def print_summary_report(analysis):
    """Print comprehensive summary report"""

    print("\n" + "=" * 80)
    print("WEEK 13 PREDICTION PERFORMANCE REPORT")
    print("=" * 80)

    total_games = len(analysis)
    correct_predictions = analysis['prediction_correct'].sum()
    accuracy = (correct_predictions / total_games) * 100

    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"   Total games analyzed: {total_games}")
    print(f"   Correct predictions: {correct_predictions}")
    print(f"   Accuracy: {accuracy:.1f}%")

    # Spread accuracy
    avg_spread_error = analysis['spread_error'].mean()
    median_spread_error = analysis['spread_error'].median()

    print(f"\n📏 SPREAD PREDICTION ACCURACY")
    print(f"   Average spread error: {avg_spread_error:.2f} points")
    print(f"   Median spread error: {median_spread_error:.2f} points")

    # Compare to Vegas
    games_with_vegas = analysis[~analysis['vegas_error'].isna()]
    if len(games_with_vegas) > 0:
        avg_vegas_error = games_with_vegas['vegas_error'].mean()
        median_vegas_error = games_with_vegas['vegas_error'].median()
        model_better_count = games_with_vegas['model_more_accurate'].sum()
        model_better_pct = (model_better_count / len(games_with_vegas)) * 100

        print(f"\n🎰 COMPARISON TO VEGAS LINES")
        print(f"   Games with Vegas lines: {len(games_with_vegas)}")
        print(f"   Vegas average error: {avg_vegas_error:.2f} points")
        print(f"   Vegas median error: {median_vegas_error:.2f} points")
        print(f"   Model average error: {games_with_vegas['spread_error'].mean():.2f} points")
        print(f"   Model median error: {games_with_vegas['spread_error'].median():.2f} points")
        print(f"\n   Model was more accurate: {model_better_count}/{len(games_with_vegas)} games ({model_better_pct:.1f}%)")

        # Difference in accuracy
        improvement = avg_vegas_error - games_with_vegas['spread_error'].mean()
        if improvement > 0:
            print(f"   ✅ Model was {improvement:.2f} points better on average")
        else:
            print(f"   ❌ Model was {abs(improvement):.2f} points worse on average")

    # Confidence analysis
    print(f"\n🎯 CONFIDENCE LEVEL ANALYSIS")
    high_conf = analysis[analysis['confidence'] >= 0.8]
    med_conf = analysis[(analysis['confidence'] >= 0.6) & (analysis['confidence'] < 0.8)]
    low_conf = analysis[analysis['confidence'] < 0.6]

    print(f"   High confidence (≥80%): {len(high_conf)} games, {(high_conf['prediction_correct'].sum()/len(high_conf)*100):.1f}% accurate")
    if len(med_conf) > 0:
        print(f"   Medium confidence (60-80%): {len(med_conf)} games, {(med_conf['prediction_correct'].sum()/len(med_conf)*100):.1f}% accurate")
    if len(low_conf) > 0:
        print(f"   Low confidence (<60%): {len(low_conf)} games, {(low_conf['prediction_correct'].sum()/len(low_conf)*100):.1f}% accurate")

    return analysis

def print_detailed_results(analysis):
    """Print detailed game-by-game results"""

    print("\n" + "=" * 80)
    print("DETAILED GAME RESULTS (Incorrect Predictions)")
    print("=" * 80)

    incorrect = analysis[~analysis['prediction_correct']].copy()
    incorrect = incorrect.sort_values('confidence', ascending=False)

    # Get correct column names
    home_col = 'home_team_actual' if 'home_team_actual' in incorrect.columns else 'home_team'
    away_col = 'away_team_actual' if 'away_team_actual' in incorrect.columns else 'away_team'
    vegas_col = 'vegas_spread_final' if 'vegas_spread_final' in incorrect.columns else 'vegas_spread'

    for idx, row in incorrect.iterrows():
        print(f"\n🏈 {row[away_col]} @ {row[home_col]}")
        print(f"   Predicted: {row['predicted_winner']} (confidence: {row['confidence']:.1%})")
        print(f"   Actual: {row['actual_winner']} ({row['away_score']}-{row['home_score']})")
        print(f"   Predicted spread: {row['predicted_spread']:.1f}")
        print(f"   Actual margin: {row['actual_margin']:.1f}")
        if vegas_col in row and not pd.isna(row[vegas_col]):
            print(f"   Vegas spread: {row[vegas_col]:.1f}")

    print("\n" + "=" * 80)
    print("BIGGEST UPSETS (High confidence incorrect predictions)")
    print("=" * 80)

    upsets = incorrect.nlargest(5, 'confidence')
    for idx, row in upsets.iterrows():
        print(f"\n{row[away_col]} @ {row[home_col]}")
        print(f"   We predicted: {row['predicted_winner']} ({row['confidence']:.1%} confident)")
        print(f"   Result: {row['actual_winner']} won {row['away_score']}-{row['home_score']}")

    print("\n" + "=" * 80)
    print("BEST PREDICTIONS (Closest spread predictions)")
    print("=" * 80)

    # Get correct column names for full analysis
    home_col_full = 'home_team_actual' if 'home_team_actual' in analysis.columns else 'home_team'
    away_col_full = 'away_team_actual' if 'away_team_actual' in analysis.columns else 'away_team'

    best = analysis.nsmallest(10, 'spread_error')
    for idx, row in best.iterrows():
        print(f"\n{row[away_col_full]} @ {row[home_col_full]}")
        print(f"   Predicted spread: {row['predicted_spread']:.1f}, Actual: {row['actual_margin']:.1f}")
        print(f"   Error: {row['spread_error']:.2f} points")
        if not pd.isna(row['vegas_error']):
            print(f"   Vegas error: {row['vegas_error']:.2f} points")

def save_detailed_report(analysis):
    """Save detailed results to CSV"""

    # Build column list dynamically based on what exists
    cols = ['game_id'] if 'game_id' in analysis.columns else []

    # Add team columns
    home_col = 'home_team_actual' if 'home_team_actual' in analysis.columns else 'home_team'
    away_col = 'away_team_actual' if 'away_team_actual' in analysis.columns else 'away_team'
    cols.extend([home_col, away_col])

    # Add prediction columns
    cols.extend(['predicted_winner', 'home_win_prob', 'confidence', 'predicted_spread'])

    # Add Vegas spread if it exists
    if 'vegas_spread_final' in analysis.columns:
        cols.append('vegas_spread_final')

    # Add result columns
    cols.extend(['home_score', 'away_score', 'actual_winner', 'actual_margin'])
    cols.extend(['prediction_correct', 'spread_error', 'vegas_error', 'model_more_accurate'])

    # Filter to only include columns that exist
    cols = [c for c in cols if c in analysis.columns]

    output = analysis[cols].copy()
    output.to_csv('week_13_performance_analysis.csv', index=False)
    print(f"\n💾 Detailed analysis saved to: week_13_performance_analysis.csv")

def main():
    print("Loading data...")
    predictions = load_predictions()
    results = load_actual_results()

    print("\nMatching predictions to results...")
    analysis = match_predictions_to_results(predictions, results)

    print_summary_report(analysis)
    print_detailed_results(analysis)
    save_detailed_report(analysis)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
