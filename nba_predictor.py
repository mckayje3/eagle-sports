"""
NBA Deep Eagle Predictor
Makes predictions for upcoming NBA games using Deep Eagle model
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import sqlite3
from datetime import datetime, timedelta


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


class NBAPredictor:
    """Predict NBA game outcomes using Deep Eagle model"""

    def __init__(self, model_path='models/deep_eagle_nba_2025.pt',
                 scaler_path='models/deep_eagle_nba_2025_scaler.pkl',
                 db_path='nba_games.db'):
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
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.feature_cols = checkpoint.get('feature_cols', [])

            # Rebuild model with correct input dimension
            input_dim = len(self.feature_cols)
            self.model = DeepEagleModel(input_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            print(f"Loaded NBA Deep Eagle model from {self.model_path}")
            print(f"  Features: {len(self.feature_cols)}")

        except FileNotFoundError:
            print(f"Model not found at {self.model_path}")
            print("Train a model first: py train_deep_eagle.py nba 2025 ...")
            self.model = None

    def get_upcoming_games(self, days=7):
        """Get upcoming NBA games from database"""
        conn = sqlite3.connect(self.db_path)

        query = '''
            SELECT
                g.game_id,
                g.date,
                g.season,
                g.home_team_id,
                g.away_team_id,
                ht.display_name as home_team,
                at.display_name as away_team
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.completed = 0
                AND g.date >= date('now')
                AND g.date <= date('now', ?)
            ORDER BY g.date
        '''

        games = pd.read_sql_query(query, conn, params=(f'+{days} days',))
        conn.close()

        return games

    def extract_features(self, game_row):
        """Extract features for a single game for prediction"""
        conn = sqlite3.connect(self.db_path)

        features = {}

        # Get historical stats for both teams
        season = game_row.get('season', 2025)
        home_stats = self._get_team_stats(conn, game_row['home_team_id'], season)
        away_stats = self._get_team_stats(conn, game_row['away_team_id'], season)

        for key, value in home_stats.items():
            features[f'home_hist_{key}'] = value
        for key, value in away_stats.items():
            features[f'away_hist_{key}'] = value

        # Get odds
        odds = self._get_odds(conn, game_row['game_id'])
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
        features['ast_differential'] = home_stats.get('apg', 0) - away_stats.get('apg', 0)
        features['to_differential'] = home_stats.get('to_pg', 0) - away_stats.get('to_pg', 0)

        conn.close()
        return features

    def _get_team_stats(self, conn, team_id, season):
        """Get team's season statistics"""
        cursor = conn.cursor()

        # Get PPG and basic stats
        cursor.execute('''
            SELECT
                COUNT(*) as games_played,
                AVG(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as ppg,
                AVG(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as win_pct
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
            stats['fg_pct'] = box_row[0] or 0
            stats['three_pct'] = box_row[1] or 0
            stats['ft_pct'] = box_row[2] or 0
            stats['rpg'] = box_row[3] or 0
            stats['oreb_pg'] = box_row[4] or 0
            stats['dreb_pg'] = box_row[5] or 0
            stats['apg'] = box_row[6] or 0
            stats['to_pg'] = box_row[7] or 0
            stats['spg'] = box_row[8] or 0
            stats['bpg'] = box_row[9] or 0
            stats['ast_to_ratio'] = stats['apg'] / stats['to_pg'] if stats['to_pg'] > 0 else stats['apg']
        else:
            for key in ['fg_pct', 'three_pct', 'ft_pct', 'rpg', 'oreb_pg', 'dreb_pg',
                        'apg', 'to_pg', 'spg', 'bpg', 'ast_to_ratio']:
                stats[key] = 0

        return stats

    def _empty_stats(self):
        """Return empty stats dict"""
        return {
            'games_played': 0, 'ppg': 0, 'papg': 0, 'win_pct': 0,
            'fg_pct': 0, 'three_pct': 0, 'ft_pct': 0,
            'rpg': 0, 'oreb_pg': 0, 'dreb_pg': 0,
            'apg': 0, 'to_pg': 0, 'ast_to_ratio': 0,
            'spg': 0, 'bpg': 0,
            'home_games': 0, 'home_ppg': 0, 'home_papg': 0, 'home_win_pct': 0,
            'away_games': 0, 'away_ppg': 0, 'away_papg': 0, 'away_win_pct': 0,
            'home_away_ppg_diff': 0
        }

    def _get_odds(self, conn, game_id):
        """Get odds for a game"""
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COALESCE(closing_spread_home, opening_spread_home) as spread,
                COALESCE(closing_total, opening_total) as total,
                opening_moneyline_home,
                opening_moneyline_away
            FROM game_odds WHERE game_id = ?
            ORDER BY updated_at DESC LIMIT 1
        ''', (game_id,))

        row = cursor.fetchone()
        if not row:
            return {
                'spread': 0, 'total': 0,
                'ml_home': 0, 'ml_away': 0,
            }

        return {
            'spread': row[0] or 0,
            'total': row[1] or 0,
            'ml_home': row[2] or 0,
            'ml_away': row[3] or 0,
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
                spread = home_score - away_score
                total = home_score + away_score

                # Calculate confidence
                score_diff = abs(spread)
                confidence = min(0.95, 0.5 + score_diff / 25)

                # Get vegas odds for comparison
                conn = sqlite3.connect(self.db_path)
                odds = self._get_odds(conn, game['game_id'])
                conn.close()

                predictions.append({
                    'game_id': game['game_id'],
                    'date': game['date'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'pred_home_score': round(home_score, 1),
                    'pred_away_score': round(away_score, 1),
                    'pred_spread': round(spread, 1),
                    'pred_total': round(total, 1),
                    'vegas_spread': odds['spread'],
                    'vegas_total': odds['total'],
                    'confidence': round(confidence, 3),
                    'predicted_winner': game['home_team'] if spread > 0 else game['away_team']
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

    def save_predictions(self, predictions_df, output_path='nba_predictions.csv'):
        """Save predictions to CSV"""
        predictions_df.to_csv(output_path, index=False)
        print(f"Saved {len(predictions_df)} predictions to {output_path}")


if __name__ == '__main__':
    import sys

    predictor = NBAPredictor()

    if predictor.model is None:
        print("\nNo model available. Train a model first.")
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
        print("NBA PREDICTIONS")
        print('='*80)
        for _, pred in predictions.iterrows():
            print(f"\n{pred['date']}: {pred['away_team']} @ {pred['home_team']}")
            print(f"  Predicted: {pred['pred_away_score']:.0f} - {pred['pred_home_score']:.0f}")
            print(f"  Spread: {pred['predicted_winner']} by {abs(pred['pred_spread']):.1f}")
            print(f"  Total: {pred['pred_total']:.1f}")
            print(f"  Confidence: {pred['confidence']:.1%}")

        # Save predictions
        predictor.save_predictions(predictions)
