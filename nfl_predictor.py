"""
NFL Deep Eagle Predictor
Makes predictions for upcoming NFL games using Deep Eagle model
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


class NFLPredictor:
    """Predict NFL game outcomes using Deep Eagle model"""

    def __init__(self, model_path='models/deep_eagle_nfl_2025.pt',
                 scaler_path='models/deep_eagle_nfl_2025_scaler.pkl',
                 db_path='nfl_games.db'):
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

            print(f"Loaded NFL Deep Eagle model from {self.model_path}")
            print(f"  Features: {len(self.feature_cols)}")

        except FileNotFoundError:
            print(f"Model not found at {self.model_path}")
            print("Train a model first: py train_deep_eagle.py nfl 2025 ...")
            self.model = None

    def get_current_week(self):
        """Calculate current NFL week based on date"""
        today = datetime.now()

        # NFL 2025 season starts Sept 4, 2025
        if today.year >= 2025 and today.month >= 9:
            season_start = datetime(2025, 9, 4)
            season = 2025
        else:
            season_start = datetime(2024, 9, 5)
            season = 2024

        if today < season_start:
            return 1, season

        days_since_start = (today - season_start).days
        week = (days_since_start // 7) + 1
        week = min(week, 18)  # NFL regular season is 18 weeks

        return week, season

    def get_upcoming_games(self, week=None, days=None):
        """Get upcoming NFL games from database"""
        conn = sqlite3.connect(self.db_path)

        if week:
            query = '''
                SELECT
                    g.game_id,
                    g.date,
                    g.week,
                    g.season,
                    g.home_team_id,
                    g.away_team_id,
                    ht.display_name as home_team,
                    at.display_name as away_team,
                    g.neutral_site,
                    g.conference_game
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.team_id
                JOIN teams at ON g.away_team_id = at.team_id
                WHERE g.completed = 0 AND g.week >= ?
                ORDER BY g.week, g.date
            '''
            games = pd.read_sql_query(query, conn, params=(week,))
        elif days:
            query = '''
                SELECT
                    g.game_id,
                    g.date,
                    g.week,
                    g.season,
                    g.home_team_id,
                    g.away_team_id,
                    ht.display_name as home_team,
                    at.display_name as away_team,
                    g.neutral_site,
                    g.conference_game
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.team_id
                JOIN teams at ON g.away_team_id = at.team_id
                WHERE g.completed = 0
                    AND g.date >= date('now')
                    AND g.date <= date('now', ?)
                ORDER BY g.date
            '''
            games = pd.read_sql_query(query, conn, params=(f'+{days} days',))
        else:
            query = '''
                SELECT
                    g.game_id,
                    g.date,
                    g.week,
                    g.season,
                    g.home_team_id,
                    g.away_team_id,
                    ht.display_name as home_team,
                    at.display_name as away_team,
                    g.neutral_site,
                    g.conference_game
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.team_id
                JOIN teams at ON g.away_team_id = at.team_id
                WHERE g.completed = 0
                ORDER BY g.week, g.date
            '''
            games = pd.read_sql_query(query, conn)

        conn.close()
        return games

    def extract_features(self, game_row):
        """Extract features for a single game for prediction"""
        conn = sqlite3.connect(self.db_path)

        features = {}

        # Game context
        features['week_normalized'] = game_row.get('week', 10) / 18.0
        features['neutral_site'] = game_row.get('neutral_site', 0) or 0
        features['conference_game'] = game_row.get('conference_game', 0) or 0

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
        features['yard_differential'] = home_stats.get('total_yards_pg', 0) - away_stats.get('total_yards_pg', 0)
        features['turnover_differential'] = home_stats.get('turnover_margin', 0) - away_stats.get('turnover_margin', 0)
        features['third_down_differential'] = home_stats.get('third_down_pct', 0) - away_stats.get('third_down_pct', 0)
        features['redzone_differential'] = home_stats.get('redzone_pct', 0) - away_stats.get('redzone_pct', 0)

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
                AVG(ts.total_yards) as total_yards_pg,
                AVG(ts.passing_yards) as pass_yards_pg,
                AVG(ts.rushing_yards) as rush_yards_pg,
                AVG(ts.turnovers) as turnovers_pg,
                AVG(ts.fumbles_lost) as fumbles_pg,
                AVG(ts.interceptions_thrown) as ints_pg,
                AVG(ts.third_down_pct) as third_down_pct,
                AVG(ts.fourth_down_pct) as fourth_down_pct,
                AVG(ts.possession_time) as avg_possession,
                AVG(ts.sacks) as sacks_pg
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

        if box_row:
            stats['total_yards_pg'] = box_row[0] or 0
            stats['pass_yards_pg'] = box_row[1] or 0
            stats['rush_yards_pg'] = box_row[2] or 0
            stats['turnovers_pg'] = box_row[3] or 0
            stats['fumbles_pg'] = box_row[4] or 0
            stats['ints_pg'] = box_row[5] or 0
            stats['third_down_pct'] = box_row[6] or 0
            stats['fourth_down_pct'] = box_row[7] or 0
            stats['avg_possession'] = box_row[8] or 0
            stats['sacks_pg'] = box_row[9] or 0
            stats['turnover_margin'] = 0  # Would need opponent data
            stats['redzone_pct'] = 0  # Placeholder
        else:
            for key in ['total_yards_pg', 'pass_yards_pg', 'rush_yards_pg', 'turnovers_pg',
                        'fumbles_pg', 'ints_pg', 'third_down_pct', 'fourth_down_pct',
                        'avg_possession', 'sacks_pg', 'turnover_margin', 'redzone_pct']:
                stats[key] = 0

        return stats

    def _empty_stats(self):
        """Return empty stats dict"""
        return {
            'games_played': 0, 'ppg': 0, 'papg': 0, 'win_pct': 0,
            'home_games': 0, 'home_ppg': 0, 'home_papg': 0, 'home_win_pct': 0,
            'away_games': 0, 'away_ppg': 0, 'away_papg': 0, 'away_win_pct': 0,
            'total_yards_pg': 0, 'pass_yards_pg': 0, 'rush_yards_pg': 0,
            'turnovers_pg': 0, 'fumbles_pg': 0, 'ints_pg': 0,
            'third_down_pct': 0, 'fourth_down_pct': 0, 'avg_possession': 0,
            'sacks_pg': 0, 'turnover_margin': 0, 'redzone_pct': 0
        }

    def _get_odds(self, conn, game_id):
        """Get odds for a game"""
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COALESCE(current_spread_home, opening_spread_home) as spread,
                COALESCE(current_total, opening_total) as total,
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
                confidence = min(0.95, 0.5 + score_diff / 20)

                # Get vegas odds for comparison
                conn = sqlite3.connect(self.db_path)
                odds = self._get_odds(conn, game['game_id'])
                conn.close()

                predictions.append({
                    'game_id': game['game_id'],
                    'date': game['date'],
                    'week': game.get('week', 0),
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

    def predict_upcoming(self, week=None, days=7):
        """Get and predict upcoming games"""
        if week:
            games = self.get_upcoming_games(week=week)
        else:
            games = self.get_upcoming_games(days=days)

        if games.empty:
            print("No upcoming games found")
            return None

        print(f"Found {len(games)} upcoming games")
        return self.predict(games)

    def save_predictions(self, predictions_df, output_path='nfl_predictions.csv'):
        """Save predictions to CSV"""
        predictions_df.to_csv(output_path, index=False)
        print(f"Saved {len(predictions_df)} predictions to {output_path}")


if __name__ == '__main__':
    import sys

    predictor = NFLPredictor()

    if predictor.model is None:
        print("\nNo model available. Train a model first.")
        sys.exit(1)

    # Get upcoming games
    week = int(sys.argv[1]) if len(sys.argv) > 1 else None

    if week:
        games = predictor.get_upcoming_games(week=week)
    else:
        current_week, season = predictor.get_current_week()
        print(f"Current NFL Week: {current_week}, Season: {season}")
        games = predictor.get_upcoming_games(week=current_week)

    if games.empty:
        print("No upcoming games found")
        sys.exit(0)

    print(f"\nFound {len(games)} upcoming games")

    # Generate predictions
    predictions = predictor.predict(games)

    if predictions is not None and not predictions.empty:
        print(f"\n{'='*80}")
        print("NFL PREDICTIONS")
        print('='*80)
        for _, pred in predictions.iterrows():
            print(f"\nWeek {pred['week']}: {pred['away_team']} @ {pred['home_team']}")
            print(f"  Predicted: {pred['pred_away_score']:.0f} - {pred['pred_home_score']:.0f}")
            print(f"  Spread: {pred['predicted_winner']} by {abs(pred['pred_spread']):.1f}")
            print(f"  Total: {pred['pred_total']:.1f}")
            print(f"  Confidence: {pred['confidence']:.1%}")

        # Save predictions
        predictor.save_predictions(predictions)
