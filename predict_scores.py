"""
Predict Actual Game Scores for Upcoming College Football Games

Uses deep learning models to predict:
- Actual home team score
- Actual away team score
- Point differential (spread)
- Total points (over/under)
- Win probability
"""

import sqlite3
import pandas as pd
import numpy as np
from ml_feature_extraction_v2 import FeatureExtractorV2
from cfb_score_predictor import CFBScorePredictor
import os
from datetime import datetime


def get_upcoming_games(db_path='cfb_games.db', season=2025, limit=200):
    """Get list of upcoming games"""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.date,
            g.home_team_id,
            g.away_team_id,
            home.name as home_team,
            away.name as away_team,
            home.display_name as home_school,
            away.display_name as away_school,
            g.neutral_site
        FROM games g
        JOIN teams home ON g.home_team_id = home.team_id
        JOIN teams away ON g.away_team_id = away.team_id
        WHERE g.season = ?
            AND g.completed = 0
            AND g.date > datetime('now')
        ORDER BY g.week, g.date, g.game_id
        LIMIT ?
    """

    df = pd.read_sql_query(query, conn, params=[season, limit])
    conn.close()

    return df


def predict_game_scores(
    model_dir='models',
    db_path='cfb_games.db',
    season=2025,
    output_file='predicted_scores.csv',
    # Margins of error based on validation performance (MAE)
    spread_margin=12.4,  # Spread prediction MAE from training
    total_margin=11.4    # Total prediction MAE from training
):
    """Predict scores for upcoming games with confidence intervals"""
    print("\n" + "="*90)
    print("COLLEGE FOOTBALL SCORE PREDICTIONS")
    print("Deep Learning-Based Score, Spread, and Total Predictions")
    print("="*90 + "\n")

    # Check if models exist
    if not os.path.exists(f'{model_dir}/win_model.pt'):
        print(f"ERROR: Models not found in {model_dir}/")
        print("Please run 'py train_score_predictor.py' first to train the models.")
        return

    # Load predictor
    print("Loading prediction models...")
    predictor = CFBScorePredictor()
    try:
        predictor.load(model_dir)
        print("  [OK] All models loaded successfully\n")
    except (FileNotFoundError, OSError) as e:
        print(f"  [ERROR] Model file not found: {e}")
        return
    except (ValueError, RuntimeError) as e:
        print(f"  [ERROR] Error loading models: {e}")
        return

    # Get upcoming games
    print("Fetching upcoming games...")
    upcoming = get_upcoming_games(db_path=db_path, season=season)

    if upcoming.empty:
        print("\nNo upcoming games found!")
        print("All games for the season may be completed.")
        return

    print(f"Found {len(upcoming)} upcoming games\n")

    # Extract features and make predictions
    print("Extracting features and predicting scores...")
    print("-"*90 + "\n")

    extractor = FeatureExtractorV2(db_path=db_path)
    predictions_list = []

    for idx, game in upcoming.iterrows():
        try:
            # Get features for this game
            features = extractor.get_game_features(game['game_id'])

            if features is None:
                print(f"  [WARN] Could not extract features for {game['home_school']} vs {game['away_school']}")
                continue

            # Convert features dict to dataframe
            feature_df = pd.DataFrame([features])

            # Use only the features that the model was trained on
            available_features = [col for col in predictor.feature_columns if col in feature_df.columns]
            X = feature_df[available_features].values

            # Make predictions
            preds = predictor.predict_scores(X)

            # Extract predictions
            home_score = preds['home_score'][0]
            away_score = preds['away_score'][0]
            win_prob = preds['home_win_prob'][0]
            spread = preds['spread'][0]
            total = preds['total'][0]

            # Determine predicted winner
            if win_prob > 0.5:
                predicted_winner = game['home_school']
                confidence = win_prob
            else:
                predicted_winner = game['away_school']
                confidence = 1 - win_prob

            # Calculate confidence intervals (80% confidence ~= 1.28 std errors)
            spread_confidence_low = spread - spread_margin
            spread_confidence_high = spread + spread_margin
            total_confidence_low = max(0, total - total_margin)  # Can't be negative
            total_confidence_high = total + total_margin

            # Store prediction
            predictions_list.append({
                'game_id': game['game_id'],
                'week': game['week'],
                'date': game['date'],
                'away_team': game['away_school'],
                'home_team': game['home_school'],
                'predicted_away_score': away_score,
                'predicted_home_score': home_score,
                'predicted_winner': predicted_winner,
                'win_probability': confidence,
                'predicted_spread': spread,
                'predicted_total': total,
                'spread_margin_error': spread_margin,
                'total_margin_error': total_margin,
                'spread_confidence_low': spread_confidence_low,
                'spread_confidence_high': spread_confidence_high,
                'total_confidence_low': total_confidence_low,
                'total_confidence_high': total_confidence_high,
                'neutral_site': game['neutral_site']
            })

            # Print prediction
            neutral = " (Neutral)" if game['neutral_site'] else ""
            print(f"Week {game['week']} - {game['date']}{neutral}")
            print(f"  {game['away_school']} @ {game['home_school']}")
            print(f"  Predicted Score: {game['away_school']} {away_score}, {game['home_school']} {home_score}")
            print(f"  Winner: {predicted_winner} ({confidence*100:.1f}% confidence)")
            print(f"  Spread: {game['home_school']} {spread:+.1f} (±{spread_margin:.1f})")
            print(f"  Total: {total:.1f} (±{total_margin:.1f})")
            print()

        except (ValueError, KeyError) as e:
            print(f"  [ERROR] Error predicting {game['home_school']} vs {game['away_school']}: {e}")
            continue
        except RuntimeError as e:
            print(f"  [ERROR] Model error for {game['home_school']} vs {game['away_school']}: {e}")
            continue

    # Save predictions to CSV
    if predictions_list:
        predictions_df = pd.DataFrame(predictions_list)
        predictions_df.to_csv(output_file, index=False)

        print("-"*90)
        print(f"\n[SUCCESS] Predictions saved to {output_file}")
        print(f"  Total predictions: {len(predictions_list)}")

        # Summary statistics
        print("\nPrediction Summary:")
        print("-"*90)
        print(f"Average predicted total:     {predictions_df['predicted_total'].mean():.1f} points")
        print(f"Total range:                 {predictions_df['predicted_total'].min():.1f} - {predictions_df['predicted_total'].max():.1f}")
        print(f"Average predicted spread:    {abs(predictions_df['predicted_spread']).mean():.1f} points")
        print(f"Largest spread:              {predictions_df['predicted_spread'].abs().max():.1f} points")
        print(f"Close games (<7 pts):        {(abs(predictions_df['predicted_spread']) < 7).sum()} games")
        print(f"Blowouts (>21 pts):          {(abs(predictions_df['predicted_spread']) > 21).sum()} games")

        # Top games to watch
        print("\nTop Games to Watch (Closest Spreads):")
        print("-"*90)
        predictions_df['abs_spread'] = abs(predictions_df['predicted_spread'])
        closest_games = predictions_df.nsmallest(5, 'abs_spread')
        for _, game in closest_games.iterrows():
            print(f"  Week {game['week']}: {game['away_team']} @ {game['home_team']}")
            print(f"    Predicted: {game['predicted_away_score']}-{game['predicted_home_score']} (Spread: {game['predicted_spread']:+.1f})")

        # Biggest blowouts
        print("\nBiggest Predicted Blowouts:")
        print("-"*90)
        blowouts = predictions_df.nlargest(5, 'abs_spread')
        for _, game in blowouts.iterrows():
            favorite = game['home_team'] if game['predicted_spread'] > 0 else game['away_team']
            print(f"  Week {game['week']}: {game['away_team']} @ {game['home_team']}")
            print(f"    Predicted: {game['predicted_away_score']}-{game['predicted_home_score']} ({favorite} by {abs(game['predicted_spread']):.1f})")

    else:
        print("\n[WARN] No predictions were generated")

    print("\n" + "="*90 + "\n")


def main():
    """Main function"""
    predict_game_scores(
        model_dir='models',
        db_path='cfb_games.db',
        season=2025,
        output_file='predicted_scores.csv'
    )


if __name__ == '__main__':
    main()
