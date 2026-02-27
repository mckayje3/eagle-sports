"""
Predict outcomes for upcoming college football games
Uses the trained model to make predictions
"""
import sqlite3
import pandas as pd
import numpy as np
from ml_feature_extraction import FeatureExtractor
from cfb_predictor import CFBPredictor


def get_upcoming_games(db_path='cfb_games.db', season=2025):
    """Get list of upcoming games"""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.date,
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


def predict_games():
    """Make predictions for upcoming games"""
    print("="*80)
    print("COLLEGE FOOTBALL GAME PREDICTIONS")
    print("="*80 + "\n")

    # Load model
    print("Loading trained model...")
    predictor = CFBPredictor()
    try:
        predictor.load_model('cfb_model.keras')
    except (FileNotFoundError, OSError) as e:
        print(f"ERROR: Model not found! ({e})")
        print("Run 'py cfb_predictor.py' to train the model first.")
        return

    # Also load the scaler from training data
    print("Loading feature scaler...")
    df_train = pd.read_csv('ml_features_2025.csv')
    feature_cols = [
        'week', 'neutral_site', 'h2h_games', 'h2h_win_pct',
        'home_recent_win_pct', 'away_recent_win_pct', 'recent_form_diff'
    ]
    X_train = df_train[feature_cols].values
    predictor.scaler.fit(X_train)  # Fit scaler on training data

    # Get upcoming games
    print("Fetching upcoming games...")
    upcoming = get_upcoming_games()

    if upcoming.empty:
        print("\nNo upcoming games found!")
        print("All games for the season may be completed.")
        return

    print(f"\nFound {len(upcoming)} upcoming games\n")

    # Extract features for each game
    extractor = FeatureExtractor()
    predictions_list = []

    print("Making predictions...")
    print("-"*80 + "\n")

    for idx, game in upcoming.iterrows():
        # Extract features
        features = extractor.get_game_features(game['game_id'])

        # Get just the feature values we need
        feature_values = [
            features.get('week', 0),
            features.get('neutral_site', 0),
            features.get('h2h_games', 0),
            features.get('h2h_win_pct', 0.5),
            features.get('home_recent_win_pct', 0),
            features.get('away_recent_win_pct', 0),
            features.get('recent_form_diff', 0)
        ]

        X = np.array([feature_values])

        # Make prediction
        win_prob = predictor.predict(X)[0][0]

        # Get betting line if available
        spread = features.get('spread', None)

        predictions_list.append({
            'game_id': game['game_id'],
            'week': game['week'],
            'date': game['date'],
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'neutral_site': game['neutral_site'],
            'home_win_prob': win_prob,
            'predicted_winner': game['home_team'] if win_prob > 0.5 else game['away_team'],
            'confidence': max(win_prob, 1 - win_prob),
            'spread': spread
        })

    # Convert to DataFrame and display
    pred_df = pd.DataFrame(predictions_list)

    # Sort by week and confidence
    pred_df = pred_df.sort_values(['week', 'confidence'], ascending=[True, False])

    print(f"{'Week':<6} {'Matchup':<50} {'Predicted Winner':<25} {'Confidence':<12} {'Spread':<10}")
    print("="*110)

    for _, pred in pred_df.iterrows():
        matchup = f"{pred['away_team']} @ {pred['home_team']}"
        if pred['neutral_site']:
            matchup += " (N)"

        confidence_str = f"{pred['confidence']:.1%}"

        # Color code confidence
        if pred['confidence'] >= 0.75:
            conf_label = "HIGH"
        elif pred['confidence'] >= 0.65:
            conf_label = "MEDIUM"
        else:
            conf_label = "LOW"

        spread_str = f"{pred['spread']:+.1f}" if pd.notna(pred['spread']) else "N/A"

        print(f"{pred['week']:<6.0f} {matchup:<50} {pred['predicted_winner']:<25} {confidence_str:<12} {spread_str:<10}")

    # Summary stats
    print("\n" + "="*110)
    print("\nPREDICTION SUMMARY:")
    print(f"  Total predictions: {len(pred_df)}")
    print(f"  High confidence (>75%): {len(pred_df[pred_df['confidence'] >= 0.75])}")
    print(f"  Medium confidence (65-75%): {len(pred_df[(pred_df['confidence'] >= 0.65) & (pred_df['confidence'] < 0.75)])}")
    print(f"  Low confidence (<65%): {len(pred_df[pred_df['confidence'] < 0.65])}")
    print(f"  Home team favored: {len(pred_df[pred_df['home_win_prob'] > 0.5])}")
    print(f"  Away team favored: {len(pred_df[pred_df['home_win_prob'] <= 0.5])}")

    # Save to CSV
    pred_df.to_csv('upcoming_game_predictions.csv', index=False)
    print(f"\nPredictions saved to: upcoming_game_predictions.csv")

    # Show confidence by week
    print("\n" + "="*110)
    print("\nCONFIDENCE BY WEEK:")
    for week in sorted(pred_df['week'].unique()):
        week_games = pred_df[pred_df['week'] == week]
        avg_conf = week_games['confidence'].mean()
        print(f"  Week {week:.0f}: {len(week_games)} games, avg confidence: {avg_conf:.1%}")

    print("\n" + "="*110)


if __name__ == '__main__':
    predict_games()
