"""
Enhanced Prediction System
Uses both enhanced win/loss classifier and spread predictor for better predictions
"""
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import sys

# Import our models
from cfb_predictor_v2 import CFBPredictorV2
from spread_predictor import SpreadPredictor
from ml_feature_extraction_v2 import FeatureExtractorV2


class EnhancedPredictionSystem:
    """Combined system using both enhanced models"""

    def __init__(self, db_path='cfb_games.db'):
        self.db_path = db_path
        self.win_loss_model = CFBPredictorV2()
        self.spread_model = SpreadPredictor()
        self.feature_extractor = FeatureExtractorV2(db_path)

        # Load trained models
        try:
            self.win_loss_model.load_model('cfb_model_v2.keras')
            print("Loaded enhanced win/loss model (v2)")
        except:
            print("WARNING: Could not load cfb_model_v2.keras")
            print("Run: py cfb_predictor_v2.py first to train the model")
            sys.exit(1)

        try:
            self.spread_model.load_model('spread_model.keras')
            print("Loaded spread prediction model")
        except:
            print("WARNING: Could not load spread_model.keras")
            print("Run: py spread_predictor.py first to train the model")
            sys.exit(1)

        # Load feature columns from training data
        try:
            train_data = pd.read_csv('ml_features_v2_2025.csv')
            exclude_cols = [
                'game_id', 'home_team_id', 'away_team_id',
                'home_win', 'point_differential',
                'home_team', 'away_team', 'season'
            ]
            self.win_loss_features = [col for col in train_data.columns if col not in exclude_cols]
            print(f"Using {len(self.win_loss_features)} features for win/loss prediction")
        except:
            print("ERROR: Could not load ml_features_v2_2025.csv")
            sys.exit(1)

    def predict_game(self, game_id):
        """
        Make comprehensive prediction for a single game

        Returns dict with:
        - home_win_prob
        - predicted_winner
        - confidence
        - predicted_spread
        - vegas_spread (if available)
        - spread_agreement
        """
        # Extract features
        features_dict = self.feature_extractor.get_game_features(game_id)

        # Check if we got valid features (dict with more than just game_id)
        if len(features_dict) <= 1:
            return None

        # Convert to DataFrame for model input
        features_df = pd.DataFrame([features_dict])

        # Get game details
        conn = sqlite3.connect(self.db_path)
        game_query = """
            SELECT
                g.week,
                ht.name as home_team,
                at.name as away_team
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.game_id = ?
        """
        game_info = pd.read_sql_query(game_query, conn, params=[game_id])

        # Get Vegas spread if available
        odds_query = """
            SELECT closing_spread_home, current_spread_home, opening_spread_home
            FROM game_odds
            WHERE game_id = ?
            ORDER BY id DESC LIMIT 1
        """
        odds = pd.read_sql_query(odds_query, conn, params=[game_id])

        vegas_spread = None
        if not odds.empty:
            row = odds.iloc[0]
            # Use closing spread if available, otherwise current, otherwise opening
            if pd.notna(row['closing_spread_home']):
                vegas_spread = row['closing_spread_home']
            elif pd.notna(row['current_spread_home']):
                vegas_spread = row['current_spread_home']
            elif pd.notna(row['opening_spread_home']):
                vegas_spread = row['opening_spread_home']

        conn.close()

        # Prepare features for models
        X_win_loss = features_df[self.win_loss_features].values
        X_spread = features_df[self.spread_model.feature_columns].values

        # Win/loss prediction
        home_win_prob = self.win_loss_model.predict_proba(X_win_loss)[0][0]

        # Spread prediction (positive = home team favored)
        predicted_spread = self.spread_model.predict(X_spread)[0][0]

        # Determine winner
        predicted_winner = game_info.iloc[0]['home_team'] if home_win_prob > 0.5 else game_info.iloc[0]['away_team']

        # Confidence (how far from 50/50)
        confidence = abs(home_win_prob - 0.5) * 2  # Scale to 0-1

        # Check agreement with Vegas
        spread_agreement = None
        if vegas_spread is not None:
            # Vegas spread is from home team perspective (negative = home favored)
            # Our spread is home - away (positive = home favored)
            # So we need to flip Vegas spread for comparison
            model_agrees = (predicted_spread > 0) == (-vegas_spread > 0)
            spread_diff = abs(predicted_spread - (-vegas_spread))
            spread_agreement = {
                'agrees': model_agrees,
                'difference': spread_diff
            }

        result = {
            'game_id': game_id,
            'week': game_info.iloc[0]['week'],
            'home_team': game_info.iloc[0]['home_team'],
            'away_team': game_info.iloc[0]['away_team'],
            'home_win_prob': home_win_prob,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'predicted_spread': predicted_spread,
            'vegas_spread': vegas_spread,
            'spread_agreement': spread_agreement
        }

        return result

    def predict_multiple_games(self, game_ids):
        """Predict multiple games"""
        predictions = []

        for game_id in game_ids:
            pred = self.predict_game(game_id)
            if pred:
                predictions.append(pred)

        return pd.DataFrame(predictions)

    def predict_week(self, week, season=2025):
        """Predict all games in a week"""
        conn = sqlite3.connect(self.db_path)

        # Get upcoming/incomplete games for the week
        query = """
            SELECT game_id
            FROM games
            WHERE week = ? AND season = ? AND completed = 0
            ORDER BY date
        """

        games = pd.read_sql_query(query, conn, params=[week, season])
        conn.close()

        if games.empty:
            print(f"No upcoming games found for Week {week}")
            return None

        print(f"Found {len(games)} games for Week {week}")

        return self.predict_multiple_games(games['game_id'].tolist())

    def format_prediction(self, pred):
        """Format prediction for display"""
        output = []
        output.append(f"\n{pred['away_team']} @ {pred['home_team']}")
        output.append(f"  Week {pred['week']}")
        output.append(f"")
        output.append(f"  Predicted Winner: {pred['predicted_winner']}")
        output.append(f"  Home Win Probability: {pred['home_win_prob']:.1%}")
        output.append(f"  Confidence: {pred['confidence']:.1%}")
        output.append(f"")
        output.append(f"  Model Spread: {pred['home_team']} {pred['predicted_spread']:+.1f}")

        if pred['vegas_spread'] is not None:
            output.append(f"  Vegas Spread: {pred['home_team']} {-pred['vegas_spread']:+.1f}")

            if pred['spread_agreement']:
                agrees = "YES" if pred['spread_agreement']['agrees'] else "NO"
                diff = pred['spread_agreement']['difference']
                output.append(f"  Model agrees with Vegas: {agrees} (diff: {diff:.1f})")

        return "\n".join(output)


def main():
    """Main prediction script"""
    print("="*80)
    print("ENHANCED PREDICTION SYSTEM")
    print("="*80)
    print()

    # Initialize system
    system = EnhancedPredictionSystem()
    print()

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'week':
            # Predict specific week
            week = int(sys.argv[2]) if len(sys.argv) > 2 else 13
            print(f"Predicting Week {week} games...")
            print()

            predictions = system.predict_week(week)

            if predictions is not None:
                print("="*80)
                print(f"WEEK {week} PREDICTIONS")
                print("="*80)

                for _, pred in predictions.iterrows():
                    print(system.format_prediction(pred))

                # Save to CSV
                output_file = f'enhanced_predictions_week_{week}.csv'
                predictions.to_csv(output_file, index=False)
                print()
                print("="*80)
                print(f"Predictions saved to {output_file}")

        elif sys.argv[1] == 'game':
            # Predict specific game
            game_id = int(sys.argv[2])
            print(f"Predicting game {game_id}...")
            print()

            pred = system.predict_game(game_id)

            if pred:
                print("="*80)
                print("GAME PREDICTION")
                print("="*80)
                print(system.format_prediction(pred))
                print("="*80)

    else:
        # Default: predict upcoming games
        print("Predicting all upcoming games...")
        print()

        conn = sqlite3.connect('cfb_games.db')
        upcoming = pd.read_sql_query(
            "SELECT DISTINCT week FROM games WHERE completed = 0 ORDER BY week LIMIT 1",
            conn
        )
        conn.close()

        if not upcoming.empty:
            week = upcoming.iloc[0]['week']
            predictions = system.predict_week(week)

            if predictions is not None:
                print("="*80)
                print(f"UPCOMING GAMES - WEEK {week}")
                print("="*80)

                for _, pred in predictions.iterrows():
                    print(system.format_prediction(pred))

                # Save to CSV
                output_file = 'enhanced_predictions_upcoming.csv'
                predictions.to_csv(output_file, index=False)
                print()
                print("="*80)
                print(f"Predictions saved to {output_file}")
                print("="*80)


if __name__ == '__main__':
    main()
