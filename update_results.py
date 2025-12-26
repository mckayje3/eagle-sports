"""
Update predictions with actual game results and compare to Vegas

This script:
1. Fetches actual game results from the database
2. Updates predicted_scores.csv with actual outcomes
3. Calculates accuracy metrics
4. Compares model performance vs Vegas odds
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime


def get_completed_games(db_path='cfb_games.db', season=2025):
    """Get completed games with scores"""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.date,
            g.home_team_id,
            g.away_team_id,
            home.display_name as home_team,
            away.display_name as away_team,
            g.home_score,
            g.away_score,
            g.completed
        FROM games g
        JOIN teams home ON g.home_team_id = home.team_id
        JOIN teams away ON g.away_team_id = away.team_id
        WHERE g.season = ?
            AND g.completed = 1
            AND g.home_score IS NOT NULL
            AND g.away_score IS NOT NULL
        ORDER BY g.week, g.date
    """

    df = pd.read_sql_query(query, conn, params=[season])
    conn.close()

    return df


def get_vegas_odds(db_path='cfb_games.db', game_ids=None):
    """Get Vegas odds for games"""
    conn = sqlite3.connect(db_path)

    if game_ids is None or len(game_ids) == 0:
        return pd.DataFrame()

    # Convert to list if needed
    if isinstance(game_ids, (int, np.integer)):
        game_ids = [game_ids]

    placeholders = ','.join(['?' for _ in game_ids])
    query = f"""
        SELECT
            game_id,
            sportsbook,
            spread,
            total,
            timestamp
        FROM odds_and_predictions
        WHERE game_id IN ({placeholders})
        ORDER BY game_id, timestamp DESC
    """

    df = pd.read_sql_query(query, conn, params=list(game_ids))
    conn.close()

    # Get the most recent odds for each game
    if not df.empty:
        df = df.groupby('game_id').first().reset_index()

    return df


def update_predictions_with_results(
    predictions_file='predicted_scores.csv',
    output_file='predictions_with_results.csv',
    db_path='cfb_games.db'
):
    """Update predictions CSV with actual results"""

    print("="*80)
    print("UPDATING PREDICTIONS WITH ACTUAL RESULTS")
    print("="*80 + "\n")

    # Load predictions
    try:
        predictions_df = pd.read_csv(predictions_file)
        print(f"Loaded {len(predictions_df)} predictions from {predictions_file}")
    except FileNotFoundError:
        print(f"ERROR: {predictions_file} not found!")
        print("Run 'py predict_scores.py' first to generate predictions.")
        return

    # Get completed games
    completed_df = get_completed_games(db_path)
    print(f"Found {len(completed_df)} completed games in database")

    if completed_df.empty:
        print("\nNo completed games found yet. Check back after Saturday!")
        return

    # Get Vegas odds for completed games
    completed_game_ids = completed_df['game_id'].tolist()
    vegas_df = get_vegas_odds(db_path, completed_game_ids)
    print(f"Found Vegas odds for {len(vegas_df)} games")

    # Merge results with predictions
    predictions_df = predictions_df.merge(
        completed_df[['game_id', 'home_score', 'away_score', 'completed']],
        on='game_id',
        how='left',
        suffixes=('', '_actual')
    )

    # Rename to avoid confusion
    predictions_df.rename(columns={
        'home_score': 'actual_home_score',
        'away_score': 'actual_away_score'
    }, inplace=True)

    # Merge Vegas odds
    predictions_df = predictions_df.merge(
        vegas_df[['game_id', 'spread', 'total', 'sportsbook']],
        on='game_id',
        how='left',
        suffixes=('', '_vegas')
    )

    # Rename Vegas columns
    predictions_df.rename(columns={
        'spread': 'vegas_spread',
        'total': 'vegas_total',
        'sportsbook': 'vegas_sportsbook'
    }, inplace=True)

    # Calculate actual metrics for completed games
    mask = predictions_df['completed'] == 1

    if mask.sum() > 0:
        # Actual spread (positive = home team won by X)
        predictions_df.loc[mask, 'actual_spread'] = (
            predictions_df.loc[mask, 'actual_home_score'] -
            predictions_df.loc[mask, 'actual_away_score']
        )

        # Actual total
        predictions_df.loc[mask, 'actual_total'] = (
            predictions_df.loc[mask, 'actual_home_score'] +
            predictions_df.loc[mask, 'actual_away_score']
        )

        # Actual winner
        predictions_df.loc[mask, 'actual_winner'] = predictions_df.loc[mask].apply(
            lambda row: row['home_team'] if row['actual_home_score'] > row['actual_away_score']
            else row['away_team'], axis=1
        )

        # Prediction accuracy - Winner
        predictions_df.loc[mask, 'correct_winner'] = (
            predictions_df.loc[mask, 'predicted_winner'] ==
            predictions_df.loc[mask, 'actual_winner']
        ).astype(int)

        # Prediction errors
        predictions_df.loc[mask, 'spread_error'] = abs(
            predictions_df.loc[mask, 'predicted_spread'] -
            predictions_df.loc[mask, 'actual_spread']
        )

        predictions_df.loc[mask, 'total_error'] = abs(
            predictions_df.loc[mask, 'predicted_total'] -
            predictions_df.loc[mask, 'actual_total']
        )

        # Vegas accuracy (if available)
        vegas_mask = mask & predictions_df['vegas_spread'].notna()
        if vegas_mask.sum() > 0:
            # Vegas spread error
            predictions_df.loc[vegas_mask, 'vegas_spread_error'] = abs(
                predictions_df.loc[vegas_mask, 'vegas_spread'] -
                predictions_df.loc[vegas_mask, 'actual_spread']
            )

            # Vegas total error
            predictions_df.loc[vegas_mask, 'vegas_total_error'] = abs(
                predictions_df.loc[vegas_mask, 'vegas_total'] -
                predictions_df.loc[vegas_mask, 'actual_total']
            )

            # Vegas winner prediction (favorite based on spread)
            predictions_df.loc[vegas_mask, 'vegas_predicted_winner'] = predictions_df.loc[vegas_mask].apply(
                lambda row: row['home_team'] if row['vegas_spread'] > 0 else row['away_team'],
                axis=1
            )

            predictions_df.loc[vegas_mask, 'vegas_correct_winner'] = (
                predictions_df.loc[vegas_mask, 'vegas_predicted_winner'] ==
                predictions_df.loc[vegas_mask, 'actual_winner']
            ).astype(int)

    # Save updated predictions
    predictions_df.to_csv(output_file, index=False)
    print(f"\n✓ Updated predictions saved to {output_file}")

    # Print summary statistics
    completed_games = mask.sum()
    if completed_games > 0:
        print(f"\n" + "="*80)
        print(f"RESULTS SUMMARY ({completed_games} completed games)")
        print("="*80)

        # Model accuracy
        win_accuracy = predictions_df.loc[mask, 'correct_winner'].mean()
        avg_spread_error = predictions_df.loc[mask, 'spread_error'].mean()
        avg_total_error = predictions_df.loc[mask, 'total_error'].mean()

        print(f"\n🤖 MODEL PERFORMANCE:")
        print(f"  Win/Loss Accuracy:    {win_accuracy*100:.1f}%")
        print(f"  Avg Spread Error:     {avg_spread_error:.2f} points")
        print(f"  Avg Total Error:      {avg_total_error:.2f} points")

        # Vegas comparison
        vegas_games = (mask & predictions_df['vegas_spread'].notna()).sum()
        if vegas_games > 0:
            vegas_win_accuracy = predictions_df.loc[vegas_mask, 'vegas_correct_winner'].mean()
            vegas_spread_error = predictions_df.loc[vegas_mask, 'vegas_spread_error'].mean()
            vegas_total_error = predictions_df.loc[vegas_mask, 'vegas_total_error'].mean()

            print(f"\n🎰 VEGAS PERFORMANCE ({vegas_games} games with odds):")
            print(f"  Win/Loss Accuracy:    {vegas_win_accuracy*100:.1f}%")
            print(f"  Avg Spread Error:     {vegas_spread_error:.2f} points")
            print(f"  Avg Total Error:      {vegas_total_error:.2f} points")

            print(f"\n📊 MODEL vs VEGAS:")
            win_diff = win_accuracy - vegas_win_accuracy
            spread_diff = vegas_spread_error - avg_spread_error
            total_diff = vegas_total_error - avg_total_error

            print(f"  Win Accuracy:     {win_diff:+.1%} {'✓ Better' if win_diff > 0 else '✗ Worse'}")
            print(f"  Spread Error:     {spread_diff:+.2f} pts {'✓ Better' if spread_diff > 0 else '✗ Worse'}")
            print(f"  Total Error:      {total_diff:+.2f} pts {'✓ Better' if total_diff > 0 else '✗ Worse'}")

        # Show some example results
        print(f"\n" + "="*80)
        print("SAMPLE RESULTS")
        print("="*80)

        sample = predictions_df[mask].head(5)
        for _, game in sample.iterrows():
            print(f"\n{game['away_team']} @ {game['home_team']}")
            print(f"  Actual Score: {int(game['actual_away_score'])}-{int(game['actual_home_score'])}")
            print(f"  Predicted: {int(game['predicted_away_score'])}-{int(game['predicted_home_score'])}")
            print(f"  Winner: {game['actual_winner']} ({'✓' if game['correct_winner'] else '✗'})")
            print(f"  Spread: Predicted {game['predicted_spread']:+.1f}, Actual {game['actual_spread']:+.1f} (Error: {game['spread_error']:.1f})")
            print(f"  Total: Predicted {game['predicted_total']:.1f}, Actual {game['actual_total']:.1f} (Error: {game['total_error']:.1f})")

    print("\n" + "="*80)

    return predictions_df


def main():
    """Main function"""
    update_predictions_with_results(
        predictions_file='predicted_scores.csv',
        output_file='predictions_with_results.csv',
        db_path='cfb_games.db'
    )


if __name__ == '__main__':
    main()
