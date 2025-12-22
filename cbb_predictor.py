"""
CBB Deep Eagle Predictor
Makes predictions for upcoming Men's College Basketball games using Deep Eagle model

SPREAD CONVENTION (Vegas standard):
    spread = away_score - home_score
    NEGATIVE spread (-7) = HOME team favored by 7
    POSITIVE spread (+7) = AWAY team favored by 7

See spread_utils.py for the authoritative definition.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import sqlite3
from datetime import datetime, timedelta
from spread_utils import validate_prediction_spread, get_predicted_winner


class DeepEagleModel(nn.Module):
    """Deep Eagle neural network for score prediction"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(DeepEagleModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        self.home_score_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.away_score_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        home_score = self.home_score_head(features)
        away_score = self.away_score_head(features)
        return torch.cat([home_score, away_score], dim=1)


class CBBPredictor:
    """Predict CBB game outcomes using Deep Eagle model"""

    def __init__(self, model_path='models/deep_eagle_cbb_2025.pt',
                 scaler_path='models/deep_eagle_cbb_2025_scaler.pkl',
                 db_path='cbb_games.db'):
        self.db_path = db_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self._load_model()

    def _load_model(self):
        """Load trained model and scaler"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.feature_cols = checkpoint.get('feature_cols', [])

            # Rebuild model with correct input dimension
            input_dim = len(self.feature_cols)
            self.model = DeepEagleModel(input_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            print(f"Loaded CBB Deep Eagle model from {self.model_path}")
            print(f"  Features: {len(self.feature_cols)}")

        except FileNotFoundError:
            print(f"Model not found at {self.model_path}")
            print("Train a model first: py train_cbb_model.py")
            self.model = None

    def get_upcoming_games(self, days=7):
        """Get upcoming CBB games from database"""
        from datetime import datetime, timedelta

        conn = sqlite3.connect(self.db_path)

        # CBB games are stored with ISO timestamps (e.g., 2025-12-18T01:00Z)
        # Generate date range strings that match the database format
        now = datetime.utcnow()
        start_date = (now - timedelta(hours=12)).strftime('%Y-%m-%d')
        end_date = (now + timedelta(days=days)).strftime('%Y-%m-%d')

        query = '''
            SELECT
                g.game_id,
                g.date,
                g.home_team_id,
                g.away_team_id,
                ht.name as home_team,
                at.name as away_team,
                g.neutral_site,
                g.conference_game
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.completed = 0
                AND g.date >= ?
                AND g.date <= ?
            ORDER BY g.date
        '''

        games = pd.read_sql_query(query, conn, params=(start_date, end_date + 'Z'))
        conn.close()

        return games

    def extract_features(self, game_row):
        """Extract features for a single game for prediction"""
        conn = sqlite3.connect(self.db_path)

        features = {}

        # Convert numpy types to native Python types for SQLite compatibility
        # numpy.int64 doesn't work properly as SQLite parameters
        home_team_id = int(game_row['home_team_id'])
        away_team_id = int(game_row['away_team_id'])
        game_id = int(game_row['game_id'])

        # Game context
        features['week_normalized'] = 0.5  # Mid-season estimate
        features['neutral_site'] = game_row.get('neutral_site', 0) or 0
        features['conference_game'] = game_row.get('conference_game', 0) or 0

        # Get historical stats for both teams
        home_stats = self._get_team_stats(conn, home_team_id)
        away_stats = self._get_team_stats(conn, away_team_id)

        for key, value in home_stats.items():
            features[f'home_hist_{key}'] = value
        for key, value in away_stats.items():
            features[f'away_hist_{key}'] = value

        # Get odds
        odds = self._get_odds(conn, game_id)
        for key, value in odds.items():
            features[f'odds_{key}'] = value

        # Calculate differentials
        features['ppg_differential'] = home_stats.get('ppg', 0) - away_stats.get('ppg', 0)
        features['papg_differential'] = home_stats.get('papg', 0) - away_stats.get('papg', 0)
        features['win_pct_differential'] = home_stats.get('win_pct', 0) - away_stats.get('win_pct', 0)
        features['fg_pct_differential'] = home_stats.get('fg_pct', 0) - away_stats.get('fg_pct', 0)
        features['three_pct_differential'] = home_stats.get('three_pct', 0) - away_stats.get('three_pct', 0)
        features['ft_pct_differential'] = home_stats.get('ft_pct', 0) - away_stats.get('ft_pct', 0)
        features['reb_differential'] = home_stats.get('rpg', 0) - away_stats.get('rpg', 0)
        features['oreb_differential'] = home_stats.get('oreb_pg', 0) - away_stats.get('oreb_pg', 0)
        features['to_differential'] = home_stats.get('to_pg', 0) - away_stats.get('to_pg', 0)
        features['ast_to_differential'] = home_stats.get('ast_to_ratio', 0) - away_stats.get('ast_to_ratio', 0)
        features['venue_ppg_differential'] = home_stats.get('home_ppg', 0) - away_stats.get('away_ppg', 0)
        features['venue_win_pct_differential'] = home_stats.get('home_win_pct', 0) - away_stats.get('away_win_pct', 0)
        features['combined_home_advantage'] = (
            home_stats.get('home_away_ppg_diff', 0) + away_stats.get('home_away_ppg_diff', 0)
        ) / 2
        features['home_sos'] = home_stats.get('opponent_ppg', 0)
        features['away_sos'] = away_stats.get('opponent_ppg', 0)

        conn.close()

        return features

    def _get_team_stats(self, conn, team_id):
        """Get team's season statistics"""
        cursor = conn.cursor()

        # Convert numpy types to native Python (SQLite doesn't handle numpy.int64)
        team_id = int(team_id)

        # Get current season
        cursor.execute('SELECT MAX(season) FROM games')
        season = cursor.fetchone()[0]

        # Get PPG and basic stats
        cursor.execute('''
            SELECT
                COUNT(*) as games_played,
                AVG(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as ppg,
                AVG(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_pct
            FROM games
            WHERE season = ? AND completed = 1
                AND (home_team_id = ? OR away_team_id = ?)
        ''', (team_id, team_id, team_id, season, team_id, team_id))

        row = cursor.fetchone()
        if not row or row[0] == 0:
            return self._empty_stats()

        stats = {
            'games_played': row[0],
            'ppg': row[1] or 0,
            'papg': row[2] or 0,
            'win_pct': row[3] or 0,
        }

        # Get home stats
        cursor.execute('''
            SELECT
                COUNT(*) as home_games,
                AVG(home_score) as home_ppg,
                AVG(away_score) as home_papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as home_win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND home_team_id = ?
        ''', (team_id, season, team_id))
        home_row = cursor.fetchone()

        # Get away stats
        cursor.execute('''
            SELECT
                COUNT(*) as away_games,
                AVG(away_score) as away_ppg,
                AVG(home_score) as away_papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as away_win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND away_team_id = ?
        ''', (team_id, season, team_id))
        away_row = cursor.fetchone()

        # Get box score stats
        cursor.execute('''
            SELECT
                AVG(ts.field_goal_pct) as fg_pct,
                AVG(ts.three_point_pct) as three_pct,
                AVG(ts.free_throw_pct) as ft_pct,
                AVG(ts.total_rebounds) as rpg,
                AVG(ts.offensive_rebounds) as oreb_pg,
                AVG(ts.defensive_rebounds) as dreb_pg,
                AVG(ts.assists) as apg,
                AVG(ts.turnovers) as to_pg,
                AVG(ts.steals) as spg,
                AVG(ts.blocks) as bpg
            FROM team_game_stats ts
            JOIN games g ON ts.game_id = g.game_id
            WHERE ts.team_id = ? AND g.season = ? AND g.completed = 1
        ''', (team_id, season))
        box_row = cursor.fetchone()

        stats['home_games'] = home_row[0] if home_row else 0
        stats['home_ppg'] = home_row[1] or 0 if home_row else 0
        stats['home_papg'] = home_row[2] or 0 if home_row else 0
        stats['home_win_pct'] = home_row[3] or 0 if home_row else 0

        stats['away_games'] = away_row[0] if away_row else 0
        stats['away_ppg'] = away_row[1] or 0 if away_row else 0
        stats['away_papg'] = away_row[2] or 0 if away_row else 0
        stats['away_win_pct'] = away_row[3] or 0 if away_row else 0

        stats['home_away_ppg_diff'] = stats['home_ppg'] - stats['away_ppg'] if stats['home_games'] > 0 and stats['away_games'] > 0 else 0

        if box_row:
            stats['fg_pct'] = box_row[0] or 45.0  # CBB average
            stats['three_pct'] = box_row[1] or 34.0
            stats['ft_pct'] = box_row[2] or 70.0
            stats['rpg'] = box_row[3] or 35.0
            stats['oreb_pg'] = box_row[4] or 10.0
            stats['dreb_pg'] = box_row[5] or 25.0
            stats['apg'] = box_row[6] or 13.0
            stats['to_pg'] = box_row[7] or 12.0
            stats['spg'] = box_row[8] or 6.0
            stats['bpg'] = box_row[9] or 3.0
            stats['ast_to_ratio'] = stats['apg'] / stats['to_pg'] if stats['to_pg'] > 0 else stats['apg']
            # Use training mean for true_shooting (~0.55)
            stats['true_shooting'] = 0.55
        else:
            # Use CBB averages when no data available
            stats['fg_pct'] = 45.0
            stats['three_pct'] = 34.0
            stats['ft_pct'] = 70.0
            stats['rpg'] = 35.0
            stats['oreb_pg'] = 10.0
            stats['dreb_pg'] = 25.0
            stats['apg'] = 13.0
            stats['to_pg'] = 12.0
            stats['spg'] = 6.0
            stats['bpg'] = 3.0
            stats['ast_to_ratio'] = 1.1
            stats['true_shooting'] = 0.55

        # Use training mean for opponent_ppg/SOS (~78)
        stats['opponent_ppg'] = 78.0

        return stats

    def _empty_stats(self):
        """Return empty stats dict with CBB averages as defaults"""
        return {
            'games_played': 0, 'ppg': 70.0, 'papg': 70.0, 'win_pct': 0.5,
            'fg_pct': 45.0, 'three_pct': 34.0, 'ft_pct': 70.0, 'true_shooting': 0.55,
            'rpg': 35.0, 'oreb_pg': 10.0, 'dreb_pg': 25.0,
            'apg': 13.0, 'to_pg': 12.0, 'ast_to_ratio': 1.1,
            'spg': 6.0, 'bpg': 3.0,
            'home_games': 0, 'home_ppg': 72.0, 'home_papg': 68.0, 'home_win_pct': 0.6,
            'away_games': 0, 'away_ppg': 68.0, 'away_papg': 72.0, 'away_win_pct': 0.4,
            'home_away_ppg_diff': 4.0,
            'opponent_ppg': 78.0
        }

    def _get_odds(self, conn, game_id):
        """Get odds for a game from odds_and_predictions table"""
        cursor = conn.cursor()

        # Convert numpy types to native Python (SQLite doesn't handle numpy.int64)
        game_id = int(game_id)

        cursor.execute('''
            SELECT opening_spread, latest_spread, opening_total, latest_total,
                   opening_moneyline_home, latest_moneyline_home,
                   opening_moneyline_away, latest_moneyline_away
            FROM odds_and_predictions WHERE game_id = ?
        ''', (game_id,))

        row = cursor.fetchone()

        # Use training data mean for CBB total (~143) when missing
        default_total = 143.0

        if not row:
            return {
                'opening_spread': 0, 'closing_spread': 0,
                'opening_total': default_total, 'closing_total': default_total,
                'opening_ml_home': 0, 'closing_ml_home': 0,
                'opening_ml_away': 0, 'closing_ml_away': 0,
                '_missing_odds': True
            }

        has_spread = row[0] is not None or row[1] is not None
        has_total = row[2] is not None or row[3] is not None

        return {
            'opening_spread': row[0] or 0,
            'closing_spread': row[1] or row[0] or 0,
            'opening_total': row[2] or default_total,
            'closing_total': row[3] or row[2] or default_total,
            'opening_ml_home': row[4] or 0,
            'closing_ml_home': row[5] or row[4] or 0,
            'opening_ml_away': row[6] or 0,
            'closing_ml_away': row[7] or row[6] or 0,
            '_missing_odds': not (has_spread and has_total)
        }

    def predict(self, games_df):
        """Generate predictions for games"""
        if self.model is None:
            print("No model loaded")
            return None

        predictions = []

        for idx, game in games_df.iterrows():
            try:
                features = self.extract_features(game)

                # Build feature vector in correct order
                feature_vector = []
                for col in self.feature_cols:
                    feature_vector.append(features.get(col, 0))

                feature_vector = np.array([feature_vector])
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
                feature_vector = self.scaler.transform(feature_vector)

                # Predict
                with torch.no_grad():
                    X = torch.FloatTensor(feature_vector).to(self.device)
                    pred = self.model(X).cpu().numpy()[0]

                home_score = max(0, pred[0])
                away_score = max(0, pred[1])
                # VEGAS CONVENTION: spread = away - home
                # Negative spread (-7) = HOME favored by 7
                # Positive spread (+7) = AWAY favored by 7
                spread = away_score - home_score
                total = home_score + away_score

                # Calculate confidence
                score_diff = abs(spread)
                confidence = min(0.95, 0.5 + score_diff / 30)

                # Validate spread convention before saving
                validate_prediction_spread(
                    round(spread, 1), round(home_score, 1), round(away_score, 1),
                    context=f"game_id={game['game_id']}"
                )

                predictions.append({
                    'game_id': game['game_id'],
                    'date': game['date'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'pred_home_score': round(home_score, 1),
                    'pred_away_score': round(away_score, 1),
                    'pred_spread': round(spread, 1),
                    'pred_total': round(total, 1),
                    'confidence': round(confidence, 3),
                    # Use spread_utils for consistent convention enforcement
                    'predicted_winner': get_predicted_winner(spread, game['home_team'], game['away_team'])
                })

            except Exception as e:
                print(f"Error predicting game {game['game_id']}: {e}")
                continue

        return pd.DataFrame(predictions)

    def predict_upcoming(self, days=7):
        """Get and predict upcoming games"""
        games = self.get_upcoming_games(days=days)

        if games.empty:
            print("No upcoming games found")
            return None

        print(f"Found {len(games)} upcoming games")
        return self.predict(games)

    def save_predictions(self, predictions_df, output_path='cbb_predictions.csv'):
        """Save predictions to CSV"""
        predictions_df.to_csv(output_path, index=False)
        print(f"Saved {len(predictions_df)} predictions to {output_path}")


if __name__ == '__main__':
    import sys

    predictor = CBBPredictor()

    if predictor.model is None:
        print("\nNo model available. Train a model first:")
        print("  1. Extract features: py cbb_feature_extractor.py 2024")
        print("  2. Train model: py train_deep_eagle_cbb.py 2025 cbb_2025_deep_eagle_features.csv")
        sys.exit(1)

    # Get upcoming games
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    games = predictor.get_upcoming_games(days=days)

    if games.empty:
        print(f"No upcoming games found in next {days} days")
        sys.exit(0)

    print(f"\nFound {len(games)} upcoming games")

    # Generate predictions
    predictions = predictor.predict(games)

    if predictions is not None and not predictions.empty:
        print(f"\n{'='*80}")
        print("CBB PREDICTIONS")
        print('='*80)
        for _, pred in predictions.iterrows():
            print(f"\n{pred['date']}: {pred['away_team']} @ {pred['home_team']}")
            print(f"  Predicted: {pred['pred_away_score']:.0f} - {pred['pred_home_score']:.0f}")
            print(f"  Spread: {pred['predicted_winner']} by {abs(pred['pred_spread']):.1f}")
            print(f"  Total: {pred['pred_total']:.1f}")
            print(f"  Confidence: {pred['confidence']:.1%}")

        # Save predictions
        predictor.save_predictions(predictions)
