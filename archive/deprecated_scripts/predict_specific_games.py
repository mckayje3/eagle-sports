"""
Predict specific games by team names
"""
import sqlite3
import pandas as pd
import numpy as np
from ml_feature_extraction import FeatureExtractor
from cfb_predictor import CFBPredictor


def find_game_by_teams(team1, team2, db_path='cfb_games.db'):
    """Find a game by partial team name match"""
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
            g.neutral_site,
            g.completed
        FROM games g
        JOIN teams home ON g.home_team_id = home.team_id
        JOIN teams away ON g.away_team_id = away.team_id
        WHERE g.season = 2025
            AND g.completed = 0
            AND (
                (home.name LIKE ? AND away.name LIKE ?)
                OR (home.name LIKE ? AND away.name LIKE ?)
                OR (home.display_name LIKE ? AND away.display_name LIKE ?)
                OR (home.display_name LIKE ? AND away.display_name LIKE ?)
            )
        LIMIT 1
    """

    # Add wildcards
    t1 = f'%{team1}%'
    t2 = f'%{team2}%'

    df = pd.read_sql_query(query, conn, params=[t1, t2, t2, t1, t1, t2, t2, t1])
    conn.close()

    if df.empty:
        return None
    return df.iloc[0]


def predict_specific_games(game_list):
    """
    Predict outcomes for specific games

    Args:
        game_list: List of tuples (team1, team2)
    """
    print("="*100)
    print("SPECIFIC GAME PREDICTIONS")
    print("="*100 + "\n")

    # Load model
    print("Loading trained model...")
    predictor = CFBPredictor()
    try:
        predictor.load_model('cfb_model.keras')
    except:
        print("ERROR: Model not found!")
        print("Run 'py cfb_predictor.py' to train the model first.")
        return

    # Load scaler
    print("Loading feature scaler...")
    df_train = pd.read_csv('ml_features_2025.csv')
    feature_cols = [
        'week', 'neutral_site', 'h2h_games', 'h2h_win_pct',
        'home_recent_win_pct', 'away_recent_win_pct', 'recent_form_diff'
    ]
    X_train = df_train[feature_cols].values
    predictor.scaler.fit(X_train)

    # Feature extractor
    extractor = FeatureExtractor()

    print("\nSearching for games and making predictions...\n")
    print("="*100)

    predictions = []

    for team1, team2 in game_list:
        print(f"\nSearching for: {team1} vs {team2}")

        # Find the game
        game = find_game_by_teams(team1, team2)

        if game is None:
            print(f"  X Game not found or already completed")
            continue

        print(f"  Found: {game['away_team']} @ {game['home_team']}")
        print(f"  Week {game['week']}, Date: {game['date']}")

        # Extract features
        try:
            features = extractor.get_game_features(int(game['game_id']))
        except Exception as e:
            print(f"  X Error extracting features: {e}")
            continue

        # Get feature values
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
        home_win_prob = predictor.predict(X)[0][0]
        away_win_prob = 1 - home_win_prob

        # Determine winner
        if home_win_prob > 0.5:
            predicted_winner = game['home_team']
            confidence = home_win_prob
        else:
            predicted_winner = game['away_team']
            confidence = away_win_prob

        # Get spread if available
        spread = features.get('spread', None)

        # Predicted spread (rough estimate based on probability)
        if home_win_prob > 0.5:
            # Home favored
            implied_spread = -(home_win_prob - 0.5) * 40  # Scale to typical spreads
        else:
            # Away favored
            implied_spread = (away_win_prob - 0.5) * 40

        predictions.append({
            'matchup': f"{game['away_team']} @ {game['home_team']}",
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'week': game['week'],
            'date': game['date'],
            'home_win_prob': home_win_prob,
            'away_win_prob': away_win_prob,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'spread': spread,
            'implied_spread': implied_spread,
            'neutral_site': game['neutral_site']
        })

        print(f"  Prediction: {predicted_winner} ({confidence:.1%} confidence)")
        if spread:
            print(f"  Vegas Spread: {spread:+.1f} (Home)")
        print(f"  Model Implied Spread: {implied_spread:+.1f} (Home)")

    # Display results
    if not predictions:
        print("\nNo games found!")
        return

    print("\n" + "="*100)
    print("PREDICTION SUMMARY")
    print("="*100 + "\n")

    # Sort by confidence
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values('confidence', ascending=False)

    print(f"{'Matchup':<45} {'Predicted Winner':<20} {'Confidence':<12} {'Vegas':<10} {'Model':<10}")
    print("="*100)

    for _, pred in pred_df.iterrows():
        matchup = pred['matchup']
        if pred['neutral_site']:
            matchup += " (N)"

        vegas = f"{pred['spread']:+.1f}" if pd.notna(pred['spread']) else "N/A"
        model_spread = f"{pred['implied_spread']:+.1f}"

        print(f"{matchup:<45} {pred['predicted_winner']:<20} {pred['confidence']:>11.1%} {vegas:>9} {model_spread:>9}")

    # Detailed breakdown
    print("\n" + "="*100)
    print("DETAILED PREDICTIONS")
    print("="*100 + "\n")

    for _, pred in pred_df.iterrows():
        print(f"\n{pred['matchup']}")
        print(f"  Week {pred['week']}, {pred['date']}")
        print(f"  {pred['home_team']}: {pred['home_win_prob']:.1%} win probability")
        print(f"  {pred['away_team']}: {pred['away_win_prob']:.1%} win probability")
        print(f"  PREDICTION: {pred['predicted_winner']} wins ({pred['confidence']:.1%} confidence)")

        if pd.notna(pred['spread']):
            print(f"  Vegas Line: {pred['home_team']} {pred['spread']:+.1f}")
            print(f"  Model Line: {pred['home_team']} {pred['implied_spread']:+.1f}")

            # Compare with spread
            spread_diff = pred['implied_spread'] - pred['spread']
            if abs(spread_diff) > 3:
                if spread_diff > 0:
                    print(f"  >> Model thinks {pred['home_team']} is STRONGER than Vegas suggests (by {spread_diff:.1f} points)")
                else:
                    print(f"  >> Model thinks {pred['home_team']} is WEAKER than Vegas suggests (by {abs(spread_diff):.1f} points)")

    # Save to CSV
    pred_df.to_csv('specific_game_predictions.csv', index=False)
    print("\n" + "="*100)
    print(f"\nPredictions saved to: specific_game_predictions.csv")
    print("="*100 + "\n")


if __name__ == '__main__':
    # Parse games from user request
    games = [
        ('Louisville', 'SMU'),
        ('Missouri State', 'Kennesaw'),
        ('USC', 'Oregon'),
        ('Jacksonville State', 'FIU'),
        ('East Carolina', 'UTSA'),
        ('Southern Miss', 'South Alabama'),
        ('TCU', 'Houston'),
        ('Pittsburgh', 'Georgia Tech'),
        ('Tennessee', 'Florida'),
        ('BYU', 'Cincinnati')
    ]

    predict_specific_games(games)
