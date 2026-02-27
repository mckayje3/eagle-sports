"""
Analyze Week 13 predictions performance
Match by game_id and compare to Vegas lines
"""
import sqlite3
import pandas as pd
import numpy as np

def main():
    # Load predictions
    print("Loading predictions...")
    preds = pd.read_csv('enhanced_predictions_week_13.csv')
    print(f"  {len(preds)} predictions loaded")

    # Load completed games
    print("\nLoading completed games...")
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
        WHERE g.season = 2024
        AND g.week = 13
        AND g.completed = 1
    """

    results = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  {len(results)} completed games loaded")

    # Match by game_id
    print("\nMatching predictions to results...")
    merged = preds.merge(results, on='game_id', how='inner', suffixes=('_pred', '_result'))
    print(f"  {len(merged)} games matched")

    if len(merged) == 0:
        print("\n❌ No matches found. Game IDs may not align.")
        print("\nLet's check game IDs:")
        print(f"Prediction game_ids (first 5): {preds['game_id'].head().tolist()}")
        print(f"Result game_ids (first 5): {results['game_id'].head().tolist()}")
        return

    # Calculate accuracy metrics
    print("\nCalculating metrics...")

    # Winner prediction accuracy
    merged['correct_winner'] = merged['predicted_winner'] == merged['actual_winner']

    # Spread accuracy
    merged['spread_error'] = abs(merged['predicted_spread'] - merged['actual_margin'])

    # Vegas spread accuracy (where available)
    vegas_mask = ~merged['vegas_spread'].isna()
    merged['vegas_error'] = np.nan
    merged.loc[vegas_mask, 'vegas_error'] = abs(
        merged.loc[vegas_mask, 'vegas_spread'] - merged.loc[vegas_mask, 'actual_margin']
    )

    # Model vs Vegas
    merged['model_better_than_vegas'] = merged['spread_error'] < merged['vegas_error']

    # Print report
    print("\n" + "=" * 80)
    print("WEEK 13 PERFORMANCE REPORT")
    print("=" * 80)

    total = len(merged)
    correct = merged['correct_winner'].sum()
    accuracy = (correct / total) * 100

    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"   Games analyzed: {total}")
    print(f"   Correct winner predictions: {correct}/{total} ({accuracy:.1f}%)")

    # Spread analysis
    avg_error = merged['spread_error'].mean()
    median_error = merged['spread_error'].median()

    print(f"\n📏 SPREAD PREDICTION ACCURACY")
    print(f"   Average error: {avg_error:.2f} points")
    print(f"   Median error: {median_error:.2f} points")

    # Vegas comparison
    vegas_games = merged[vegas_mask]
    if len(vegas_games) > 0:
        vegas_avg = vegas_games['vegas_error'].mean()
        model_avg = vegas_games['spread_error'].mean()
        model_better = vegas_games['model_better_than_vegas'].sum()

        print(f"\n🎰 VS VEGAS LINES")
        print(f"   Games with Vegas lines: {len(vegas_games)}/{total}")
        print(f"   Vegas average error: {vegas_avg:.2f} points")
        print(f"   Our average error: {model_avg:.2f} points")

        diff = vegas_avg - model_avg
        if diff > 0:
            print(f"   ✅ We were {diff:.2f} points better on average")
        else:
            print(f"   ❌ We were {abs(diff):.2f} points worse on average")

        pct = (model_better / len(vegas_games)) * 100
        print(f"   Games where we were more accurate: {model_better}/{len(vegas_games)} ({pct:.1f}%)")

    # Confidence analysis
    print(f"\n🎯 BY CONFIDENCE LEVEL")
    high_conf = merged[merged['confidence'] >= 0.9]
    med_conf = merged[(merged['confidence'] >= 0.7) & (merged['confidence'] < 0.9)]
    low_conf = merged[merged['confidence'] < 0.7]

    if len(high_conf) > 0:
        acc = (high_conf['correct_winner'].sum() / len(high_conf)) * 100
        print(f"   High confidence (≥90%): {len(high_conf)} games, {acc:.1f}% accurate")

    if len(med_conf) > 0:
        acc = (med_conf['correct_winner'].sum() / len(med_conf)) * 100
        print(f"   Medium confidence (70-90%): {len(med_conf)} games, {acc:.1f}% accurate")

    if len(low_conf) > 0:
        acc = (low_conf['correct_winner'].sum() / len(low_conf)) * 100
        print(f"   Low confidence (<70%): {len(low_conf)} games, {acc:.1f}% accurate")

    # Biggest misses
    print(f"\n❌ BIGGEST MISSES")
    misses = merged[~merged['correct_winner']].nlargest(5, 'confidence')
    for _, row in misses.iterrows():
        print(f"\n   {row['away_team_pred']} @ {row['home_team_pred']}")
        print(f"   Predicted: {row['predicted_winner']} ({row['confidence']:.1%})")
        print(f"   Actual: {row['actual_winner']} won {row['away_score']}-{row['home_score']}")

    # Best predictions
    print(f"\n✅ MOST ACCURATE SPREADS")
    best = merged.nsmallest(5, 'spread_error')
    for _, row in best.iterrows():
        print(f"\n   {row['away_team_pred']} @ {row['home_team_pred']}")
        print(f"   Score: {row['away_score']}-{row['home_score']} (margin: {row['actual_margin']:.0f})")
        print(f"   Predicted spread: {row['predicted_spread']:.1f}, Error: {row['spread_error']:.2f}")

    # Save detailed results
    output = merged[[
        'game_id', 'home_team_pred', 'away_team_pred',
        'predicted_winner', 'confidence', 'predicted_spread',
        'home_score', 'away_score', 'actual_winner', 'actual_margin',
        'correct_winner', 'spread_error',
        'vegas_spread', 'vegas_error', 'model_better_than_vegas'
    ]].copy()

    output.to_csv('week_13_analysis.csv', index=False)
    print(f"\n💾 Detailed results saved to: week_13_analysis.csv")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
