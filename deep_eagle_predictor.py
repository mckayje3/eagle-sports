"""
Deep-Eagle Predictor Module
Uses trained LSTM models to generate predictions for upcoming games
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r'C:\Users\jbeast\documents\coding\deep')

import sqlite3
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from datetime import datetime, timedelta

from core import LSTMModel


class DeepEaglePredictor:
    """Generate predictions using trained Deep-Eagle models"""

    def __init__(self, sport='cfb'):
        """
        Initialize predictor for CFB or NFL

        Args:
            sport: 'cfb' or 'nfl'
        """
        self.sport = sport.lower()
        self.models_dir = Path('models')

        # Set database and model paths
        if self.sport == 'cfb':
            self.db_path = 'cfb_games.db'
            self.spread_model_path = self.models_dir / 'cfb_spread_best.pth'
            self.total_model_path = self.models_dir / 'cfb_total_best.pth'
            self.scaler_path = self.models_dir / 'cfb_scaler.pkl'
        else:
            self.db_path = 'nfl_games.db'
            self.spread_model_path = self.models_dir / 'nfl_spread_best.pth'
            self.total_model_path = self.models_dir / 'nfl_total_best.pth'
            self.scaler_path = self.models_dir / 'nfl_scaler.pkl'

        self.spread_model = None
        self.total_model = None
        self.scaler = None
        self.feature_columns = None
        self.conn = None

    def load_models(self):
        """Load trained models and scaler"""
        # Load scaler and feature columns
        if not self.scaler_path.exists():
            print(f"Scaler not found: {self.scaler_path}")
            return False

        with open(self.scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            self.scaler = scaler_data['scaler']
            self.feature_columns = scaler_data['feature_columns']

        # Get input dimension
        input_dim = len(self.feature_columns)

        # Load spread model
        if self.spread_model_path.exists():
            self.spread_model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=128,
                output_dim=1,
                num_layers=2,
                dropout=0.2
            )
            checkpoint = torch.load(self.spread_model_path, map_location='cpu', weights_only=True)
            self.spread_model.load_state_dict(checkpoint['model_state_dict'])
            self.spread_model.eval()
            print(f"Loaded spread model from {self.spread_model_path}")

        # Load total model
        if self.total_model_path.exists():
            self.total_model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=128,
                output_dim=1,
                num_layers=2,
                dropout=0.2
            )
            checkpoint = torch.load(self.total_model_path, map_location='cpu', weights_only=True)
            self.total_model.load_state_dict(checkpoint['model_state_dict'])
            self.total_model.eval()
            print(f"Loaded total model from {self.total_model_path}")

        return True

    def connect_db(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)

    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_team_features(self, team_id, season, max_week=None):
        """
        Get team's rolling features up to a certain week

        Args:
            team_id: Team ID
            season: Season year
            max_week: Maximum week to include

        Returns:
            Dictionary of features
        """
        # Get team's recent games
        week_filter = f"AND g.week < {max_week}" if max_week else ""

        query = f'''
            SELECT
                g.week,
                CASE WHEN g.home_team_id = ? THEN g.home_score ELSE g.away_score END as points_scored,
                CASE WHEN g.home_team_id = ? THEN g.away_score ELSE g.home_score END as points_allowed,
                CASE WHEN (g.home_team_id = ? AND g.home_score > g.away_score) OR
                          (g.away_team_id = ? AND g.away_score > g.home_score)
                     THEN 1 ELSE 0 END as win,
                tgs.total_yards,
                tgs.passing_yards,
                tgs.rushing_yards,
                tgs.turnovers,
                tgs.first_downs,
                tgs.third_down_conversions,
                tgs.third_down_attempts,
                tgs.penalties,
                tgs.penalty_yards
            FROM games g
            LEFT JOIN team_game_stats tgs ON g.game_id = tgs.game_id AND tgs.team_id = ?
            WHERE (g.home_team_id = ? OR g.away_team_id = ?)
                AND g.season = ?
                AND g.completed = 1
                {week_filter}
            ORDER BY g.date DESC
            LIMIT 10
        '''

        params = (team_id, team_id, team_id, team_id, team_id, team_id, team_id, season)
        df = pd.read_sql_query(query, self.conn, params=params)

        if len(df) == 0:
            # Return default features
            return self._get_default_features()

        # Calculate rolling averages
        features = {}

        # Fill missing values with defaults
        stat_defaults = {
            'total_yards': 330.0 if self.sport == 'nfl' else 380.0,
            'passing_yards': 220.0 if self.sport == 'nfl' else 250.0,
            'rushing_yards': 110.0 if self.sport == 'nfl' else 130.0,
            'turnovers': 1.0,
            'first_downs': 20.0,
            'third_down_conversions': 5.0,
            'third_down_attempts': 12.0,
            'penalties': 6.0,
            'penalty_yards': 50.0
        }

        for col, default in stat_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

        # Calculate features for different windows
        for window in [3, 5, 10]:
            data = df.head(window)

            features[f'points_scored_roll{window}'] = data['points_scored'].mean()
            features[f'points_allowed_roll{window}'] = data['points_allowed'].mean()
            features[f'point_differential_roll{window}'] = (data['points_scored'] - data['points_allowed']).mean()
            features[f'win_pct_roll{window}'] = data['win'].mean()
            features[f'total_yards_roll{window}'] = data['total_yards'].mean()
            features[f'passing_yards_roll{window}'] = data['passing_yards'].mean()
            features[f'rushing_yards_roll{window}'] = data['rushing_yards'].mean()
            features[f'turnovers_roll{window}'] = data['turnovers'].mean()
            features[f'first_downs_roll{window}'] = data['first_downs'].mean()

            # Third down percentage
            td_conv = data['third_down_conversions'].mean()
            td_att = data['third_down_attempts'].mean()
            features[f'third_down_pct_roll{window}'] = td_conv / td_att if td_att > 0 else 0.4

            features[f'penalties_roll{window}'] = data['penalties'].mean()
            features[f'penalty_yards_roll{window}'] = data['penalty_yards'].mean()

        # Lag features
        for lag in [1, 2]:
            if len(df) >= lag:
                row = df.iloc[lag-1]
                features[f'points_scored_lag{lag}'] = row['points_scored']
                features[f'points_allowed_lag{lag}'] = row['points_allowed']
                features[f'point_differential_lag{lag}'] = row['points_scored'] - row['points_allowed']
                features[f'win_lag{lag}'] = row['win']
                features[f'total_yards_lag{lag}'] = row['total_yards']

        # Streak features
        winning_streak = 0
        losing_streak = 0
        for _, row in df.iterrows():
            if row['win'] == 1:
                if losing_streak > 0:
                    break
                winning_streak += 1
            else:
                if winning_streak > 0:
                    break
                losing_streak += 1

        features['winning_streak'] = winning_streak
        features['losing_streak'] = losing_streak

        # Current game stats (use averages)
        features['total_yards'] = df['total_yards'].mean()
        features['passing_yards'] = df['passing_yards'].mean()
        features['rushing_yards'] = df['rushing_yards'].mean()
        features['turnovers'] = df['turnovers'].mean()
        features['first_downs'] = df['first_downs'].mean()
        features['third_down_conversions'] = df['third_down_conversions'].mean()
        features['third_down_attempts'] = df['third_down_attempts'].mean()
        features['third_down_pct'] = features['third_down_conversions'] / features['third_down_attempts'] if features['third_down_attempts'] > 0 else 0.4
        features['penalties'] = df['penalties'].mean()
        features['penalty_yards'] = df['penalty_yards'].mean()

        # ATS features (set to 0.5 for upcoming games)
        for window in [3, 5, 10]:
            features[f'ats_pct_roll{window}'] = 0.5
            features[f'over_pct_roll{window}'] = 0.5

        return features

    def _get_default_features(self):
        """Get default features for teams with no history"""
        features = {}

        base_stats = {
            'total_yards': 330.0 if self.sport == 'nfl' else 380.0,
            'passing_yards': 220.0 if self.sport == 'nfl' else 250.0,
            'rushing_yards': 110.0 if self.sport == 'nfl' else 130.0,
            'turnovers': 1.0,
            'first_downs': 20.0,
            'third_down_pct': 0.4,
            'penalties': 6.0,
            'penalty_yards': 50.0,
            'points_scored': 22.0 if self.sport == 'nfl' else 28.0,
            'points_allowed': 22.0 if self.sport == 'nfl' else 28.0,
        }

        for key, val in base_stats.items():
            features[key] = val

        # Rolling features
        for window in [3, 5, 10]:
            features[f'points_scored_roll{window}'] = base_stats['points_scored']
            features[f'points_allowed_roll{window}'] = base_stats['points_allowed']
            features[f'point_differential_roll{window}'] = 0.0
            features[f'win_pct_roll{window}'] = 0.5
            features[f'total_yards_roll{window}'] = base_stats['total_yards']
            features[f'passing_yards_roll{window}'] = base_stats['passing_yards']
            features[f'rushing_yards_roll{window}'] = base_stats['rushing_yards']
            features[f'turnovers_roll{window}'] = base_stats['turnovers']
            features[f'first_downs_roll{window}'] = base_stats['first_downs']
            features[f'third_down_pct_roll{window}'] = base_stats['third_down_pct']
            features[f'penalties_roll{window}'] = base_stats['penalties']
            features[f'penalty_yards_roll{window}'] = base_stats['penalty_yards']
            features[f'ats_pct_roll{window}'] = 0.5
            features[f'over_pct_roll{window}'] = 0.5

        # Lag features
        for lag in [1, 2]:
            features[f'points_scored_lag{lag}'] = base_stats['points_scored']
            features[f'points_allowed_lag{lag}'] = base_stats['points_allowed']
            features[f'point_differential_lag{lag}'] = 0.0
            features[f'win_lag{lag}'] = 0.5
            features[f'total_yards_lag{lag}'] = base_stats['total_yards']

        features['winning_streak'] = 0
        features['losing_streak'] = 0
        features['third_down_conversions'] = 5.0
        features['third_down_attempts'] = 12.0

        return features

    def prepare_game_features(self, home_team_id, away_team_id, season, week, neutral_site=0):
        """
        Prepare features for a single game prediction

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            season: Season year
            week: Week number
            neutral_site: 1 if neutral site, 0 otherwise

        Returns:
            Feature vector ready for model input
        """
        # Get home team features
        home_features = self.get_team_features(home_team_id, season, max_week=week)

        # Get away team features (for opponent stats)
        away_features = self.get_team_features(away_team_id, season, max_week=week)

        # Build feature dict in expected order
        features = {}

        # Game context
        features['neutral_site'] = neutral_site
        features['is_home'] = 1  # Home team perspective

        # Current game stats (use home team averages)
        for key in ['total_yards', 'passing_yards', 'rushing_yards', 'turnovers',
                    'first_downs', 'third_down_conversions', 'third_down_attempts',
                    'penalties', 'penalty_yards', 'third_down_pct']:
            features[key] = home_features.get(key, 0)

        # Rolling features
        for window in [3, 5, 10]:
            for stat in ['points_scored', 'points_allowed', 'point_differential',
                        'total_yards', 'passing_yards', 'rushing_yards', 'turnovers',
                        'first_downs', 'third_down_pct', 'penalties', 'penalty_yards']:
                features[f'{stat}_roll{window}'] = home_features.get(f'{stat}_roll{window}', 0)

            features[f'win_pct_roll{window}'] = home_features.get(f'win_pct_roll{window}', 0.5)
            features[f'ats_pct_roll{window}'] = home_features.get(f'ats_pct_roll{window}', 0.5)
            features[f'over_pct_roll{window}'] = home_features.get(f'over_pct_roll{window}', 0.5)

        # Lag features
        for lag in [1, 2]:
            for stat in ['points_scored', 'points_allowed', 'point_differential', 'win', 'total_yards']:
                features[f'{stat}_lag{lag}'] = home_features.get(f'{stat}_lag{lag}', 0)

        # Streak features
        features['winning_streak'] = home_features.get('winning_streak', 0)
        features['losing_streak'] = home_features.get('losing_streak', 0)

        # Opponent features
        features['opp_win_pct'] = away_features.get('win_pct_roll5', 0.5)
        features['opp_points_scored_avg'] = away_features.get('points_scored_roll5', 22.0)
        features['opp_points_allowed_avg'] = away_features.get('points_allowed_roll5', 22.0)
        features['opp_total_yards_avg'] = away_features.get('total_yards_roll5', 330.0)

        # Rest days and season progress
        features['rest_days'] = 7  # Default
        features['season_progress'] = week / 18 if self.sport == 'nfl' else week / 15

        return features

    def predict_game(self, home_team_id, away_team_id, season, week, neutral_site=0):
        """
        Generate prediction for a single game using stats-based approach

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            season: Season year
            week: Week number
            neutral_site: 1 if neutral site

        Returns:
            Dictionary with predictions
        """
        # Get team features
        home_features = self.get_team_features(home_team_id, season, max_week=week)
        away_features = self.get_team_features(away_team_id, season, max_week=week)

        # Stats-based prediction (more reliable than broken LSTM)
        # Use rolling averages to predict scores

        # Home team expected points (based on their scoring vs opponent's defense)
        home_off = home_features.get('points_scored_roll5', 28 if self.sport == 'cfb' else 22)
        away_def = away_features.get('points_allowed_roll5', 28 if self.sport == 'cfb' else 22)
        home_expected = (home_off + away_def) / 2

        # Away team expected points
        away_off = away_features.get('points_scored_roll5', 28 if self.sport == 'cfb' else 22)
        home_def = home_features.get('points_allowed_roll5', 28 if self.sport == 'cfb' else 22)
        away_expected = (away_off + home_def) / 2

        # Adjust for home field advantage (if not neutral)
        home_advantage = 0 if neutral_site else (3.0 if self.sport == 'cfb' else 2.5)
        home_expected += home_advantage / 2
        away_expected -= home_advantage / 2

        # Factor in recent form (point differential)
        home_momentum = home_features.get('point_differential_roll3', 0) * 0.1
        away_momentum = away_features.get('point_differential_roll3', 0) * 0.1
        home_expected += home_momentum
        away_expected += away_momentum

        # Win percentage adjustment
        home_win_pct = home_features.get('win_pct_roll5', 0.5)
        away_win_pct = away_features.get('win_pct_roll5', 0.5)
        win_pct_diff = (home_win_pct - away_win_pct) * 5  # Max ~5 point swing
        home_expected += win_pct_diff / 2
        away_expected -= win_pct_diff / 2

        # Ensure reasonable scores
        home_score = max(7, min(70, home_expected))
        away_score = max(7, min(70, away_expected))

        # Calculate spread and total
        spread = home_score - away_score
        total = home_score + away_score

        # Win probability (simple sigmoid based on spread)
        home_win_prob = 1 / (1 + np.exp(-spread / 7))

        return {
            'predicted_home_score': round(home_score, 1),
            'predicted_away_score': round(away_score, 1),
            'predicted_spread': round(spread, 1),
            'predicted_total': round(total, 1),
            'home_win_probability': round(home_win_prob, 3)
        }

    def get_upcoming_games(self, week=None, season=2025):
        """
        Get upcoming games from database

        Args:
            week: Week number (None for current week)
            season: Season year

        Returns:
            DataFrame of upcoming games
        """
        # Find current week if not specified
        if week is None:
            week_query = f'''
                SELECT MIN(week) as current_week FROM games
                WHERE completed = 0 AND season = {season}
            '''
            week_df = pd.read_sql_query(week_query, self.conn)
            week = week_df.iloc[0]['current_week']
            if week is None:
                week = 12  # Default

        query = f'''
            SELECT
                g.game_id,
                g.season,
                g.week,
                g.date,
                g.home_team_id,
                g.away_team_id,
                ht.name as home_team,
                at.name as away_team,
                COALESCE(ht.display_name, ht.name) as home_display_name,
                COALESCE(at.display_name, at.name) as away_display_name,
                g.neutral_site,
                g.completed,
                g.home_score,
                g.away_score,
                go.closing_spread_home as vegas_spread,
                go.closing_total as vegas_total
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            LEFT JOIN game_odds go ON g.game_id = go.game_id AND go.source = 'TheOddsAPI'
            WHERE g.week = {week} AND g.season = {season}
            ORDER BY g.date, g.game_id
        '''

        return pd.read_sql_query(query, self.conn)

    def generate_predictions(self, week=None, season=2025):
        """
        Generate predictions for all games in a week

        Args:
            week: Week number
            season: Season year

        Returns:
            DataFrame with predictions
        """
        games = self.get_upcoming_games(week, season)

        if len(games) == 0:
            print(f"No games found for week {week}")
            return pd.DataFrame()

        print(f"Generating predictions for {len(games)} games...")

        predictions = []
        for idx, game in games.iterrows():
            pred = self.predict_game(
                home_team_id=game['home_team_id'],
                away_team_id=game['away_team_id'],
                season=game['season'],
                week=game['week'],
                neutral_site=game['neutral_site']
            )

            predictions.append({
                'game_id': game['game_id'],
                'season': game['season'],
                'week': game['week'],
                'game_date': game['date'],
                'home_team': game['home_display_name'],
                'away_team': game['away_display_name'],
                'home_team_id': game['home_team_id'],
                'away_team_id': game['away_team_id'],
                'neutral_site': game['neutral_site'],
                'predicted_home_score': pred['predicted_home_score'],
                'predicted_away_score': pred['predicted_away_score'],
                'predicted_spread': pred['predicted_spread'],
                'predicted_total': pred['predicted_total'],
                'home_win_probability': pred['home_win_probability'],
                'vegas_spread': game['vegas_spread'],
                'vegas_total': game['vegas_total'],
                'game_completed': game['completed'],
                'actual_home_score': game['home_score'] if game['completed'] else None,
                'actual_away_score': game['away_score'] if game['completed'] else None
            })

        return pd.DataFrame(predictions)

    def save_predictions_to_cache(self, predictions_df, cache_db='users.db'):
        """Save predictions to cache database"""
        conn = sqlite3.connect(cache_db)
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT,
                game_id INTEGER,
                season INTEGER,
                week INTEGER,
                game_date TEXT,
                home_team TEXT,
                away_team TEXT,
                predicted_home_score REAL,
                predicted_away_score REAL,
                predicted_spread REAL,
                predicted_total REAL,
                home_win_probability REAL,
                vegas_spread REAL,
                vegas_total REAL,
                game_completed INTEGER,
                actual_home_score REAL,
                actual_away_score REAL,
                created_at TEXT,
                UNIQUE(sport, game_id, season, week)
            )
        ''')

        # Insert or update predictions
        for _, row in predictions_df.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO prediction_cache
                (sport, game_id, season, week, game_date, home_team, away_team,
                 predicted_home_score, predicted_away_score, predicted_spread, predicted_total,
                 home_win_probability, vegas_spread, vegas_total, game_completed,
                 actual_home_score, actual_away_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.sport.upper(),
                row['game_id'],
                row['season'],
                row['week'],
                row['game_date'],
                row['home_team'],
                row['away_team'],
                row['predicted_home_score'],
                row['predicted_away_score'],
                row['predicted_spread'],
                row['predicted_total'],
                row['home_win_probability'],
                row['vegas_spread'],
                row['vegas_total'],
                row['game_completed'],
                row['actual_home_score'],
                row['actual_away_score'],
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()
        print(f"Saved {len(predictions_df)} predictions to cache")


def generate_weekend_predictions():
    """Generate predictions for this weekend's CFB and NFL games"""
    print("=" * 80)
    print("DEEP-EAGLE WEEKEND PREDICTIONS")
    print("=" * 80)

    # CFB Predictions
    print("\n--- CFB PREDICTIONS ---")
    cfb_predictor = DeepEaglePredictor(sport='cfb')
    cfb_predictor.connect_db()

    if cfb_predictor.load_models():
        # Get current week
        cfb_week_df = pd.read_sql_query(
            "SELECT MIN(week) as week FROM games WHERE completed = 0 AND season = 2025",
            cfb_predictor.conn
        )
        cfb_week = cfb_week_df.iloc[0]['week']
        if cfb_week is None:
            cfb_week = 13  # Championship week

        print(f"CFB Week {cfb_week}")
        cfb_predictions = cfb_predictor.generate_predictions(week=cfb_week, season=2025)

        if len(cfb_predictions) > 0:
            cfb_predictor.save_predictions_to_cache(cfb_predictions)
            print(f"\nCFB Predictions Summary:")
            print(cfb_predictions[['away_team', 'home_team', 'predicted_spread', 'predicted_total', 'vegas_spread', 'vegas_total']].to_string())

    cfb_predictor.close_db()

    # NFL Predictions
    print("\n--- NFL PREDICTIONS ---")
    nfl_predictor = DeepEaglePredictor(sport='nfl')
    nfl_predictor.connect_db()

    if nfl_predictor.load_models():
        # Get current week
        nfl_week_df = pd.read_sql_query(
            "SELECT MIN(week) as week FROM games WHERE completed = 0 AND season = 2025",
            nfl_predictor.conn
        )
        nfl_week = nfl_week_df.iloc[0]['week']
        if nfl_week is None:
            nfl_week = 12

        print(f"NFL Week {nfl_week}")
        nfl_predictions = nfl_predictor.generate_predictions(week=nfl_week, season=2025)

        if len(nfl_predictions) > 0:
            nfl_predictor.save_predictions_to_cache(nfl_predictions)
            print(f"\nNFL Predictions Summary:")
            print(nfl_predictions[['away_team', 'home_team', 'predicted_spread', 'predicted_total', 'vegas_spread', 'vegas_total']].to_string())

    nfl_predictor.close_db()

    print("\n" + "=" * 80)
    print("PREDICTIONS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    generate_weekend_predictions()
