"""
Deep Eagle Prediction Script
Generates predictions for upcoming games using trained Deep Eagle models
"""

import torch
import pickle
import pandas as pd
import numpy as np
import sqlite3
import sys
from deep_eagle_feature_extractor import DeepEagleFeatureExtractor
from train_deep_eagle import DeepEagleModel


class DeepEaglePredictor:
    """Load trained Deep Eagle models and make predictions"""

    def __init__(self, model_path, scaler_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.sport = checkpoint['sport']
        self.season = checkpoint['season']
        self.feature_cols = checkpoint['feature_cols']

        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Build model
        input_dim = len(self.feature_cols)
        self.model = DeepEagleModel(input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Loaded {self.sport} {self.season} model")
        print(f"Features: {input_dim}")

    def predict_games(self, features_df):
        """
        Generate predictions for games

        Args:
            features_df: DataFrame with game features (must include self.feature_cols)

        Returns:
            DataFrame with predictions added
        """
        # Extract feature columns (exclude IDs and targets)
        exclude_cols = [
            'game_id', 'season', 'week', 'home_team_id', 'away_team_id',
            'home_score', 'away_score', 'point_spread', 'total_points', 'home_win'
        ]

        available_features = [col for col in features_df.columns if col not in exclude_cols]

        # Verify we have the right features
        missing_features = set(self.feature_cols) - set(available_features)
        if missing_features:
            print(f"WARNING: Missing {len(missing_features)} features: {list(missing_features)[:5]}...")
            # Add missing features as 0
            for feature in missing_features:
                features_df[feature] = 0

        # Now we should have all features
        X = features_df[self.feature_cols].values

        # Handle NaN and inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        # Add predictions to dataframe
        result_df = features_df.copy()
        result_df['pred_home_score'] = predictions[:, 0]
        result_df['pred_away_score'] = predictions[:, 1]
        result_df['pred_spread'] = predictions[:, 0] - predictions[:, 1]
        result_df['pred_total'] = predictions[:, 0] + predictions[:, 1]
        result_df['pred_home_win'] = (predictions[:, 0] > predictions[:, 1]).astype(int)
        result_df['pred_home_win_prob'] = 1 / (1 + np.exp(-(predictions[:, 0] - predictions[:, 1]) / 10))

        return result_df


def predict_upcoming_games(sport, season, db_path, model_path, scaler_path, min_week=None):
    """
    Generate predictions for upcoming games

    Args:
        sport: 'cfb' or 'nfl'
        season: year (e.g., 2025)
        db_path: path to database
        model_path: path to trained model
        scaler_path: path to scaler
        min_week: minimum week to predict (None = all upcoming)
    """
    print(f"\n{'='*80}")
    print(f"DEEP EAGLE PREDICTIONS - {sport.upper()} {season}")
    print('='*80)

    # Load predictor
    predictor = DeepEaglePredictor(model_path, scaler_path)

    # Extract features for upcoming games
    extractor = DeepEagleFeatureExtractor(db_path, sport=sport)

    # Get upcoming games
    conn = sqlite3.connect(db_path)
    query = '''
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.date,
            g.home_team_id,
            g.away_team_id,
            ht.display_name as home_team,
            at.display_name as away_team,
            g.neutral_site,
            g.conference_game
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.season = ? AND g.completed = 0
    '''

    if min_week:
        query += f' AND g.week >= {min_week}'

    query += ' ORDER BY g.week, g.date'

    upcoming_df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    print(f"\nFound {len(upcoming_df)} upcoming games")

    if len(upcoming_df) == 0:
        print("No upcoming games found!")
        return None

    # Extract features for these games
    print(f"\nExtracting features...")
    all_features = []

    for idx, game in upcoming_df.iterrows():
        try:
            # Create a game dict with dummy scores for feature extraction
            game_dict = {
                'game_id': game['game_id'],
                'season': game['season'],
                'week': game['week'],
                'home_team_id': game['home_team_id'],
                'away_team_id': game['away_team_id'],
                'neutral_site': game['neutral_site'],
                'conference_game': game['conference_game'],
                'home_score': 0,  # Dummy - not used for prediction
                'away_score': 0,  # Dummy - not used for prediction
                'temperature': 0,  # Use 0 to match training data distribution
                'wind_speed': 0,
                'is_dome': 0
            }

            # Get features from completed games + this game's context
            features = extractor._extract_game_features(pd.Series(game_dict))
            if features is not None:
                # Only remove the actual target variables (home/away score, derived metrics)
                # Keep all other features including stats
                for key in ['home_score', 'away_score', 'point_spread', 'total_points', 'home_win']:
                    features.pop(key, None)
                all_features.append(features)
        except Exception as e:
            print(f"  Error extracting features for game {game['game_id']}: {e}")
            continue

    if len(all_features) == 0:
        print("Could not extract features for any games!")
        return None

    features_df = pd.DataFrame(all_features)
    print(f"Extracted features for {len(features_df)} games")

    # Make predictions
    print("\nGenerating predictions...")
    predictions_df = predictor.predict_games(features_df)

    # Merge with game info (ensure we keep the week column from upcoming_df)
    predictions_df = predictions_df.merge(
        upcoming_df[['game_id', 'home_team', 'away_team', 'date', 'week']],
        on='game_id',
        how='left',
        suffixes=('_feat', '')  # Keep week from upcoming_df
    )

    # Drop duplicate week column if it exists
    if 'week_feat' in predictions_df.columns:
        predictions_df = predictions_df.drop(columns=['week_feat'])

    # Get odds if available - use COALESCE to get best available value from multiple sources
    conn = sqlite3.connect(db_path)
    odds_df = pd.read_sql_query('''
        SELECT
            game_id,
            COALESCE(MAX(current_spread_home), MAX(opening_spread_home)) as vegas_spread,
            COALESCE(MAX(current_total), MAX(opening_total)) as vegas_total
        FROM game_odds
        GROUP BY game_id
    ''', conn)
    conn.close()

    predictions_df = predictions_df.merge(odds_df, on='game_id', how='left')

    # Calculate edge vs Vegas
    # Model pred_spread: home_score - away_score (negative = away team wins)
    # Vegas spread_home: points to home team (positive = home is underdog, away favored)
    # To compare: convert Vegas to same convention (negate it)
    # Vegas spread in "home - away" terms = -vegas_spread
    # Edge = model_spread - (-vegas_spread) = model_spread + vegas_spread
    predictions_df['spread_edge'] = predictions_df['pred_spread'] + predictions_df['vegas_spread']
    predictions_df['total_edge'] = predictions_df['pred_total'] - predictions_df['vegas_total']

    # Sort by week and absolute edge (biggest disagreements first)
    predictions_df['abs_edge'] = predictions_df['spread_edge'].abs()
    predictions_df = predictions_df.sort_values(['week', 'abs_edge'], ascending=[True, False])

    return predictions_df


def display_predictions(predictions_df, top_n=None):
    """Display predictions in formatted table"""

    if predictions_df is None or len(predictions_df) == 0:
        print("No predictions to display")
        return

    if top_n:
        predictions_df = predictions_df.head(top_n)

    print(f"\n{'='*120}")
    print(f"PREDICTIONS")
    print('='*120)

    print(f"\n{'Week':<6} {'Matchup':<40} {'Pred Score':<15} {'Pred Spread':<12} {'Vegas':<10} {'Edge':<10} {'Confidence':<10}")
    print('-' * 120)

    for _, row in predictions_df.iterrows():
        week = int(row['week'])
        matchup = f"{row['away_team']} @ {row['home_team']}"
        pred_score = f"{row['pred_home_score']:.1f} - {row['pred_away_score']:.1f}"
        pred_spread = f"{row['pred_spread']:+.1f}"
        vegas_spread = f"{row['vegas_spread']:+.1f}" if pd.notna(row.get('vegas_spread')) else "N/A"
        edge = f"{row['spread_edge']:+.1f}" if pd.notna(row.get('spread_edge')) else "N/A"
        confidence = f"{row['pred_home_win_prob']:.1%}"

        print(f"{week:<6} {matchup:<40} {pred_score:<15} {pred_spread:<12} {vegas_spread:<10} {edge:<10} {confidence:<10}")

    print('='*120)


def main():
    if len(sys.argv) < 4:
        print("Usage: py predict_deep_eagle.py <sport> <season> <db_path> [min_week]")
        print("Example: py predict_deep_eagle.py nfl 2025 nfl_games.db 13")
        sys.exit(1)

    sport = sys.argv[1].lower()
    season = int(sys.argv[2])
    db_path = sys.argv[3]
    min_week = int(sys.argv[4]) if len(sys.argv) > 4 else None

    # Model paths
    model_path = f'models/deep_eagle_{sport}_{season}.pt'
    scaler_path = f'models/deep_eagle_{sport}_{season}_scaler.pkl'

    # Generate predictions
    predictions_df = predict_upcoming_games(
        sport, season, db_path, model_path, scaler_path, min_week
    )

    if predictions_df is not None:
        # Display all predictions
        display_predictions(predictions_df)

        # Save to CSV
        output_path = f'deep_eagle_predictions_{sport}_{season}.csv'
        predictions_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")

        # Show top picks (biggest edges)
        if pd.notna(predictions_df['spread_edge']).any():
            print(f"\n{'='*120}")
            print("TOP 10 PICKS (Biggest Edge vs Vegas)")
            print('='*120)
            top_picks = predictions_df.nlargest(10, 'spread_edge', keep='all')
            display_predictions(top_picks, top_n=10)


if __name__ == '__main__':
    main()
