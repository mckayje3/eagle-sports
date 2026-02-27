"""
Deep Learning Predictor for College Football Games
Uses trained TensorFlow/Keras models to predict game outcomes
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import sqlite3
from pathlib import Path

class DeepLearningPredictor:
    """
    Loads and uses trained deep learning models for game predictions
    Combines win probability and spread predictions to generate score predictions
    """

    def __init__(self, model_dir='.'):
        """Initialize predictor and load trained models"""
        self.model_dir = Path(model_dir)
        self.win_model = None
        self.spread_model = None
        self.win_scaler = None
        self.spread_scaler = None
        self.feature_columns = []
        self.loaded = False

        # Try to load models
        try:
            self._load_models()
            self.loaded = True
            print("[OK] Deep learning models loaded successfully!")
        except Exception as e:
            print(f"[WARN] Could not load deep learning models: {e}")
            print("  Falling back to statistical predictor")

    def _load_models(self):
        """Load trained models and scalers"""
        # Load win/loss model
        win_model_path = self.model_dir / 'cfb_model_v2.keras'
        win_scaler_path = self.model_dir / 'cfb_model_v2_scaler.pkl'

        if not win_model_path.exists():
            raise FileNotFoundError(f"Win model not found: {win_model_path}")

        self.win_model = keras.models.load_model(win_model_path)
        print(f"  Loaded win model: {win_model_path}")

        # Load win scaler
        with open(win_scaler_path, 'rb') as f:
            data = pickle.load(f)
            self.win_scaler = data['scaler']
            self.feature_columns = data['feature_columns']
        print(f"  Loaded win scaler with {len(self.feature_columns)} features")

        # Load spread model
        spread_model_path = self.model_dir / 'spread_model.keras'
        spread_scaler_path = self.model_dir / 'spread_model_scaler.pkl'

        if spread_model_path.exists():
            self.spread_model = keras.models.load_model(spread_model_path)
            print(f"  Loaded spread model: {spread_model_path}")

            with open(spread_scaler_path, 'rb') as f:
                spread_data = pickle.load(f)
                self.spread_scaler = spread_data['scaler']
            print(f"  Loaded spread scaler")
        else:
            print(f"  Spread model not found, will use win probability only")

    def get_team_stats(self, cursor, team_id, season, week):
        """Extract comprehensive stats for a team up to a given week"""

        # Get basic game results
        cursor.execute('''
            SELECT
                COUNT(*) as games,
                SUM(CASE WHEN (home_team_id = ? AND home_score > away_score) OR
                              (away_team_id = ? AND away_score > home_score)
                         THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN (home_team_id = ? AND home_score < away_score) OR
                              (away_team_id = ? AND away_score < home_score)
                         THEN 1 ELSE 0 END) as losses
            FROM games
            WHERE season = ? AND week < ? AND completed = 1
            AND (home_team_id = ? OR away_team_id = ?)
        ''', (team_id, team_id, team_id, team_id, season, week, team_id, team_id))

        games, wins, losses = cursor.fetchone()
        if not games or games == 0:
            return None

        # Get points from games table and detailed stats from team_game_stats
        cursor.execute('''
            SELECT
                SUM(CASE WHEN g.home_team_id = ? THEN g.home_score ELSE g.away_score END) as total_points_scored,
                SUM(CASE WHEN g.home_team_id = ? THEN g.away_score ELSE g.home_score END) as total_points_allowed
            FROM games g
            WHERE (g.home_team_id = ? OR g.away_team_id = ?)
            AND g.season = ? AND g.week < ? AND g.completed = 1
        ''', (team_id, team_id, team_id, team_id, season, week))

        points_stats = cursor.fetchone()
        total_scored = points_stats[0] or 0
        total_allowed = points_stats[1] or 0

        # Get detailed stats from team_game_stats
        cursor.execute('''
            SELECT
                AVG(tgs.total_yards) as avg_yards,
                AVG(tgs.passing_yards) as avg_pass_yards,
                AVG(tgs.rushing_yards) as avg_rush_yards,
                AVG(tgs.turnovers) as avg_turnovers,
                AVG(tgs.first_downs) as avg_first_downs,
                AVG(CAST(tgs.third_down_conversions AS FLOAT) / NULLIF(tgs.third_down_attempts, 0)) as third_down_pct,
                AVG(tgs.penalties) as avg_penalties,
                AVG(tgs.penalty_yards) as avg_penalty_yards
            FROM team_game_stats tgs
            JOIN games g ON tgs.game_id = g.game_id
            WHERE tgs.team_id = ? AND g.season = ? AND g.week < ? AND g.completed = 1
        ''', (team_id, season, week))

        stats = cursor.fetchone()

        return {
            'games_played': games,
            'wins': wins or 0,
            'losses': losses or 0,
            'points_scored_total': total_scored,
            'points_allowed_total': total_allowed,
            'points_scored_avg': total_scored / games if games > 0 else 0,
            'points_allowed_avg': total_allowed / games if games > 0 else 0,
            'point_differential_avg': (total_scored - total_allowed) / games if games > 0 else 0,
            'win_pct': wins / games if games > 0 else 0,
            'total_yards_avg': stats[0] or 0,
            'passing_yards_avg': stats[1] or 0,
            'rushing_yards_avg': stats[2] or 0,
            'turnovers_avg': stats[3] or 0,
            'first_downs_avg': stats[4] or 0,
            'third_down_pct': stats[5] or 0,
            'penalties_avg': stats[6] or 0,
            'penalty_yards_avg': stats[7] or 0
        }

    def get_team_features(self, home_team_id, away_team_id, season, week):
        """
        Extract features for a matchup from the database

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            season: Season year
            week: Week number

        Returns:
            Feature vector matching training data format
        """
        conn = sqlite3.connect('cfb_games.db')
        cursor = conn.cursor()

        # Get stats for both teams
        home_stats = self.get_team_stats(cursor, home_team_id, season, week)
        away_stats = self.get_team_stats(cursor, away_team_id, season, week)

        # If no stats available, use defaults
        if not home_stats:
            home_stats = {
                'games_played': 0, 'wins': 0, 'losses': 0,
                'points_scored_total': 0, 'points_allowed_total': 0,
                'points_scored_avg': 25, 'points_allowed_avg': 25,
                'point_differential_avg': 0, 'win_pct': 0.5,
                'total_yards_avg': 350, 'passing_yards_avg': 200, 'rushing_yards_avg': 150,
                'turnovers_avg': 1.5, 'first_downs_avg': 20, 'third_down_pct': 0.4,
                'penalties_avg': 6, 'penalty_yards_avg': 50
            }
        if not away_stats:
            away_stats = {
                'games_played': 0, 'wins': 0, 'losses': 0,
                'points_scored_total': 0, 'points_allowed_total': 0,
                'points_scored_avg': 25, 'points_allowed_avg': 25,
                'point_differential_avg': 0, 'win_pct': 0.5,
                'total_yards_avg': 350, 'passing_yards_avg': 200, 'rushing_yards_avg': 150,
                'turnovers_avg': 1.5, 'first_downs_avg': 20, 'third_down_pct': 0.4,
                'penalties_avg': 6, 'penalty_yards_avg': 50
            }

        # Get head-to-head stats
        cursor.execute('''
            SELECT COUNT(*),
                   SUM(CASE WHEN (home_team_id = ? AND home_score > away_score) OR
                                 (away_team_id = ? AND away_score > home_score)
                            THEN 1 ELSE 0 END)
            FROM games
            WHERE completed = 1
            AND ((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?))
        ''', (home_team_id, home_team_id, home_team_id, away_team_id, away_team_id, home_team_id))

        h2h = cursor.fetchone()
        h2h_games = h2h[0] or 0
        h2h_wins = h2h[1] or 0
        h2h_win_pct = h2h_wins / h2h_games if h2h_games > 0 else 0.5

        # Get recent form (last 3 games)
        for team_id, prefix in [(home_team_id, 'home'), (away_team_id, 'away')]:
            cursor.execute('''
                SELECT AVG(CASE WHEN (home_team_id = ? AND home_score > away_score) OR
                                     (away_team_id = ? AND away_score > home_score)
                                THEN 1.0 ELSE 0.0 END)
                FROM (
                    SELECT * FROM games
                    WHERE season = ? AND week < ? AND completed = 1
                    AND (home_team_id = ? OR away_team_id = ?)
                    ORDER BY week DESC LIMIT 3
                )
            ''', (team_id, team_id, season, week, team_id, team_id))

            recent = cursor.fetchone()[0]
            if prefix == 'home':
                home_recent = recent or 0.5
            else:
                away_recent = recent or 0.5

        conn.close()

        # Build feature dictionary matching training features
        features = {
            'week': week,
            'neutral_site': 0,
            'home_games_played': home_stats['games_played'],
            'home_wins': home_stats['wins'],
            'home_losses': home_stats['losses'],
            'home_points_scored_total': home_stats['points_scored_total'],
            'home_points_allowed_total': home_stats['points_allowed_total'],
            'home_points_scored_avg': home_stats['points_scored_avg'],
            'home_points_allowed_avg': home_stats['points_allowed_avg'],
            'home_point_differential_avg': home_stats['point_differential_avg'],
            'home_win_pct': home_stats['win_pct'],
            'home_total_yards_avg': home_stats['total_yards_avg'],
            'home_passing_yards_avg': home_stats['passing_yards_avg'],
            'home_rushing_yards_avg': home_stats['rushing_yards_avg'],
            'home_turnovers_avg': home_stats['turnovers_avg'],
            'home_first_downs_avg': home_stats['first_downs_avg'],
            'home_third_down_pct': home_stats['third_down_pct'],
            'home_penalties_avg': home_stats['penalties_avg'],
            'home_penalty_yards_avg': home_stats['penalty_yards_avg'],
            'away_games_played': away_stats['games_played'],
            'away_wins': away_stats['wins'],
            'away_losses': away_stats['losses'],
            'away_points_scored_total': away_stats['points_scored_total'],
            'away_points_allowed_total': away_stats['points_allowed_total'],
            'away_points_scored_avg': away_stats['points_scored_avg'],
            'away_points_allowed_avg': away_stats['points_allowed_avg'],
            'away_point_differential_avg': away_stats['point_differential_avg'],
            'away_win_pct': away_stats['win_pct'],
            'away_total_yards_avg': away_stats['total_yards_avg'],
            'away_passing_yards_avg': away_stats['passing_yards_avg'],
            'away_rushing_yards_avg': away_stats['rushing_yards_avg'],
            'away_turnovers_avg': away_stats['turnovers_avg'],
            'away_first_downs_avg': away_stats['first_downs_avg'],
            'away_third_down_pct': away_stats['third_down_pct'],
            'away_penalties_avg': away_stats['penalties_avg'],
            'away_penalty_yards_avg': away_stats['penalty_yards_avg'],
            'win_pct_diff': home_stats['win_pct'] - away_stats['win_pct'],
            'points_scored_diff': home_stats['points_scored_avg'] - away_stats['points_scored_avg'],
            'points_allowed_diff': home_stats['points_allowed_avg'] - away_stats['points_allowed_avg'],
            'point_differential_diff': home_stats['point_differential_avg'] - away_stats['point_differential_avg'],
            'yards_diff': home_stats['total_yards_avg'] - away_stats['total_yards_avg'],
            'passing_yards_diff': home_stats['passing_yards_avg'] - away_stats['passing_yards_avg'],
            'rushing_yards_diff': home_stats['rushing_yards_avg'] - away_stats['rushing_yards_avg'],
            'turnovers_diff': away_stats['turnovers_avg'] - home_stats['turnovers_avg'],  # Lower is better
            'h2h_games': h2h_games,
            'h2h_win_pct': h2h_win_pct,
            'home_recent_win_pct': home_recent,
            'away_recent_win_pct': away_recent,
            'recent_form_diff': home_recent - away_recent,
            'home_score': 0,  # Not used for prediction
            'away_score': 0   # Not used for prediction
        }

        # Build feature vector in correct order
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0.0))

        return np.array([feature_vector])

    def predict_game(self, home_team_id, away_team_id, season, week):
        """
        Predict game outcome with deep learning models

        Returns:
            Dictionary with predictions
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded")

        # Get features
        X = self.get_team_features(home_team_id, away_team_id, season, week)

        # Scale features
        X_scaled = self.win_scaler.transform(X)

        # Predict win probability
        win_prob = self.win_model.predict(X_scaled, verbose=0)[0][0]

        # Estimate spread from win probability
        # win_prob of 0.7 ≈ -7 point spread (home favored)
        # win_prob of 0.5 = 0 spread (even game)
        # win_prob of 0.3 ≈ +7 point spread (away favored)
        spread = -(win_prob - 0.5) * 28  # Maps probability to spread

        # Estimate total points based on team averages
        conn = sqlite3.connect('cfb_games.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT AVG(home_score + away_score)
            FROM games
            WHERE season = ? AND week < ? AND completed = 1
        ''', (season, week))

        result = cursor.fetchone()
        total = result[0] if result and result[0] else 55.0
        conn.close()

        # Calculate scores from spread and total
        home_score = (total + spread) / 2
        away_score = (total - spread) / 2

        return {
            'predicted_home_score': round(home_score, 1),
            'predicted_away_score': round(away_score, 1),
            'predicted_spread': round(spread, 1),
            'predicted_total': round(total, 1),
            'home_win_probability': round(win_prob, 3)
        }
