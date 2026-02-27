"""
Predict outcomes for upcoming college football games using PyTorch models

This script uses the PyTorch-based predictor and deep framework.
"""

import sqlite3
import pandas as pd
import numpy as np
from ml_feature_extraction_v2 import FeatureExtractorV2
from cfb_predictor_pytorch import CFBPredictorPyTorch
import os


def get_upcoming_games(db_path='cfb_games.db', season=2025):
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
            g.neutral_site
        FROM games g
        JOIN teams home ON g.home_team_id = home.team_id
        JOIN teams away ON g.away_team_id = away.team_id
        WHERE g.season = ?
            AND g.completed = 0
            AND g.date > datetime('now')
        ORDER BY g.date, g.game_id
        LIMIT 50
    """

    df = pd.read_sql_query(query, conn, params=[season])
    conn.close()

    return df


def predict_games(
    win_model_path='models/cfb_win_predictor_pytorch.pt',
    spread_model_path='models/cfb_spread_predictor_pytorch.pt',
    db_path='cfb_games.db'
):
    """Make predictions for upcoming games"""
    print("="*80)
    print("COLLEGE FOOTBALL GAME PREDICTIONS (PyTorch)")
    print("="*80 + "\n")

    # Check if models exist
    if not os.path.exists(win_model_path):
        print(f"ERROR: Win predictor model not found at {win_model_path}")
        print("Run 'py train_pytorch_model.py' to train the models first.")
        return

    if not os.path.exists(spread_model_path):
        print(f"ERROR: Spread predictor model not found at {spread_model_path}")
        print("Run 'py train_pytorch_model.py' to train the models first.")
        return

    # Load win predictor
    print("Loading win predictor...")
    win_predictor = CFBPredictorPyTorch(
        model_type='feedforward',
        task='classification'
    )
    try:
        win_predictor.load(win_model_path)
        print("  ✓ Win predictor loaded")
    except Exception as e:
        print(f"  ✗ Error loading win predictor: {e}")
        return

    # Load spread predictor
    print("Loading spread predictor...")
    spread_predictor = CFBPredictorPyTorch(
        model_type='feedforward',
        task='regression'
    )
    try:
        spread_predictor.load(spread_model_path)
        print("  ✓ Spread predictor loaded")
    except Exception as e:
        print(f"  ✗ Error loading spread predictor: {e}")
        return

    # Get upcoming games
    print("\nFetching upcoming games...")
    upcoming = get_upcoming_games(db_path=db_path)

    if upcoming.empty:
        print("\nNo upcoming games found!")
        print("All games for the season may be completed.")
        return

    print(f"Found {len(upcoming)} upcoming games\n")

    # Extract features for each game
    print("Extracting features and making predictions...")
    print("-"*80 + "\n")

    extractor = FeatureExtractorV2(db_path=db_path)
    predictions_list = []

    for idx, game in upcoming.iterrows():
        try:
            # Get features for this game
            features = extractor.get_game_features(
                game_id=game['game_id'],
                home_team_id=game['home_team_id'],
                away_team_id=game['away_team_id'],
                season=game['season'],
                week=game['week']
            )

            if features is None:
                print(f"  ⚠ Could not extract features for {game['home_team']} vs {game['away_team']}")
                continue

            # Convert features dict to dataframe with correct columns
            feature_df = pd.DataFrame([features])

            # Use only the features that the model was trained on
            available_features = [col for col in win_predictor.feature_columns if col in feature_df.columns]
            X = feature_df[available_features].values

            # Make predictions
            win_prob = win_predictor.predict(X)[0]
            spread = spread_predictor.predict(X)[0]

            # Determine predicted winner
            if win_prob > 0.5:
                predicted_winner = game['home_team']
                confidence = win_prob
            else:
                predicted_winner = game['away_team']
                confidence = 1 - win_prob

            # Store prediction
            predictions_list.append({
                'game_id': game['game_id'],
                'week': game['week'],
                'date': game['date'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'predicted_winner': predicted_winner,
                'home_win_prob': win_prob,
                'confidence': confidence,
                'predicted_spread': spread,
                'neutral_site': game['neutral_site']
            })

            # Print prediction
            print(f"Week {game['week']} - {game['date']}")
            print(f"  {game['away_team']} @ {game['home_team']}")
            print(f"  Predicted Winner: {predicted_winner} ({confidence*100:.1f}% confidence)")
            print(f"  Predicted Spread: {game['home_team']} {spread:+.1f}")
            print(f"  Home Win Probability: {win_prob*100:.1f}%")
            print()

        except Exception as e:
            print(f"  ✗ Error predicting {game['home_team']} vs {game['away_team']}: {e}")
            continue

    # Save predictions to CSV
    if predictions_list:
        predictions_df = pd.DataFrame(predictions_list)
        output_file = 'upcoming_game_predictions_pytorch.csv'
        predictions_df.to_csv(output_file, index=False)
        print("-"*80)
        print(f"\n✓ Predictions saved to {output_file}")
        print(f"  Total predictions: {len(predictions_list)}")
    else:
        print("\n⚠ No predictions were generated")

    print("\n" + "="*80)


def main():
    """Main function"""
    predict_games()


if __name__ == '__main__':
    main()
