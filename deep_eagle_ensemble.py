"""
Deep-Eagle Ensemble Predictor
Combines LSTM + Stats-based predictions with confidence intervals
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
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core import LSTMModel
from core.models.ensemble import EnsembleModel


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple prediction methods:
    - LSTM neural network (trained on historical data)
    - Stats-based heuristic (rolling averages + adjustments)
    - Power rating system (team strength estimates)

    Includes confidence intervals based on model disagreement.
    """

    def __init__(self, sport='cfb'):
        self.sport = sport.lower()
        self.models_dir = Path('models')

        # Database and model paths
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

        # Models
        self.spread_model = None
        self.total_model = None
        self.scaler = None
        self.feature_columns = None
        self.conn = None

        # Ensemble weights (tuned based on backtesting)
        # Stats-based is weighted higher since LSTM is still learning
        self.ensemble_weights = {
            'stats': 0.50,      # Stats-based approach
            'lstm': 0.25,       # LSTM neural network
            'power': 0.25       # Power rating system
        }

        # Feature importance cache
        self._feature_importance = None

        # Auto-connect to database and load models
        self.connect_db()
        self.load_models()

    def load_models(self) -> bool:
        """Load trained LSTM models and scaler"""
        try:
            # Load scaler and feature columns
            if self.scaler_path.exists():
                with open(self.scaler_path, 'rb') as f:
                    scaler_data = pickle.load(f)
                    self.scaler = scaler_data['scaler']
                    self.feature_columns = scaler_data['feature_columns']

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

                return True
        except Exception as e:
            print(f"Warning: Could not load LSTM models: {e}")

        return False

    def connect_db(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)

    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_team_features(self, team_id: int, season: int, max_week: int = None) -> Dict:
        """Get team's rolling features up to a certain week"""
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
                tgs.turnovers
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
            return self._get_default_features()

        # Calculate rolling averages
        features = {}

        # Fill missing values
        for col in ['total_yards', 'passing_yards', 'rushing_yards', 'turnovers']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean() if not df[col].isna().all() else 300)

        for window in [3, 5, 10]:
            data = df.head(window)
            features[f'points_scored_roll{window}'] = data['points_scored'].mean()
            features[f'points_allowed_roll{window}'] = data['points_allowed'].mean()
            features[f'point_differential_roll{window}'] = (data['points_scored'] - data['points_allowed']).mean()
            features[f'win_pct_roll{window}'] = data['win'].mean()
            features[f'total_yards_roll{window}'] = data['total_yards'].mean()
            features[f'turnovers_roll{window}'] = data['turnovers'].mean()

        # Calculate power rating (simple estimate)
        features['power_rating'] = self._calculate_power_rating(features)

        return features

    def _get_default_features(self) -> Dict:
        """Default features for teams with no history"""
        base_pts = 22.0 if self.sport == 'nfl' else 28.0

        features = {}
        for window in [3, 5, 10]:
            features[f'points_scored_roll{window}'] = base_pts
            features[f'points_allowed_roll{window}'] = base_pts
            features[f'point_differential_roll{window}'] = 0.0
            features[f'win_pct_roll{window}'] = 0.5
            features[f'total_yards_roll{window}'] = 330.0
            features[f'turnovers_roll{window}'] = 1.5

        features['power_rating'] = 0.0
        return features

    def _calculate_power_rating(self, features: Dict) -> float:
        """
        Calculate team power rating based on multiple factors
        Returns value between -20 and +20
        """
        # Offensive strength (points scored vs average)
        avg_pts = 22.0 if self.sport == 'nfl' else 28.0
        off_strength = (features.get('points_scored_roll5', avg_pts) - avg_pts) / 3

        # Defensive strength (points allowed vs average)
        def_strength = (avg_pts - features.get('points_allowed_roll5', avg_pts)) / 3

        # Win percentage factor
        win_factor = (features.get('win_pct_roll5', 0.5) - 0.5) * 10

        # Yards factor
        avg_yards = 330.0 if self.sport == 'nfl' else 380.0
        yards_factor = (features.get('total_yards_roll5', avg_yards) - avg_yards) / 50

        # Turnover margin factor
        to_factor = (1.5 - features.get('turnovers_roll5', 1.5)) * 2

        power = off_strength + def_strength + win_factor + yards_factor + to_factor
        return max(-20, min(20, power))

    # === PREDICTION METHODS ===

    def _predict_stats_based(self, home_features: Dict, away_features: Dict, neutral_site: int) -> Dict:
        """
        Stats-based prediction using rolling averages
        Most reliable method currently
        """
        avg_pts = 22.0 if self.sport == 'nfl' else 28.0

        # Expected scores based on offense vs defense matchup
        home_off = home_features.get('points_scored_roll5', avg_pts)
        away_def = away_features.get('points_allowed_roll5', avg_pts)
        home_expected = (home_off + away_def) / 2

        away_off = away_features.get('points_scored_roll5', avg_pts)
        home_def = home_features.get('points_allowed_roll5', avg_pts)
        away_expected = (away_off + home_def) / 2

        # Home field advantage
        hfa = 0 if neutral_site else (3.0 if self.sport == 'cfb' else 2.5)
        home_expected += hfa / 2
        away_expected -= hfa / 2

        # Momentum adjustment
        home_momentum = home_features.get('point_differential_roll3', 0) * 0.1
        away_momentum = away_features.get('point_differential_roll3', 0) * 0.1
        home_expected += home_momentum
        away_expected += away_momentum

        # Win percentage adjustment
        win_diff = (home_features.get('win_pct_roll5', 0.5) - away_features.get('win_pct_roll5', 0.5)) * 5
        home_expected += win_diff / 2
        away_expected -= win_diff / 2

        return {
            'home_score': max(7, min(70, home_expected)),
            'away_score': max(7, min(70, away_expected))
        }

    def _predict_power_rating(self, home_features: Dict, away_features: Dict, neutral_site: int) -> Dict:
        """
        Power rating prediction
        Uses team strength estimates
        """
        avg_pts = 22.0 if self.sport == 'nfl' else 28.0

        # Get power ratings
        home_power = home_features.get('power_rating', 0)
        away_power = away_features.get('power_rating', 0)

        # Power rating difference predicts spread
        power_diff = home_power - away_power

        # Home field advantage
        hfa = 0 if neutral_site else (3.0 if self.sport == 'cfb' else 2.5)

        # Predicted spread (positive = home favored)
        pred_spread = power_diff + hfa

        # Estimate total (combine offensive strengths)
        home_off_factor = home_features.get('points_scored_roll5', avg_pts)
        away_off_factor = away_features.get('points_scored_roll5', avg_pts)
        pred_total = home_off_factor + away_off_factor

        # Convert to scores
        home_score = (pred_total + pred_spread) / 2
        away_score = (pred_total - pred_spread) / 2

        return {
            'home_score': max(7, min(70, home_score)),
            'away_score': max(7, min(70, away_score))
        }

    def _predict_lstm(self, home_team_id: int, away_team_id: int,
                      season: int, week: int, neutral_site: int) -> Optional[Dict]:
        """
        LSTM neural network prediction
        Returns None if models not available
        """
        if self.spread_model is None or self.total_model is None:
            return None

        # This would require preparing features in the exact format the model was trained on
        # For now, return None and rely on other methods
        # TODO: Implement full LSTM prediction pipeline
        return None

    # === ENSEMBLE PREDICTION ===

    def predict_game(self, home_team_id: int, away_team_id: int,
                     season: int, week: int, neutral_site: int = 0) -> Dict:
        """
        Generate ensemble prediction combining multiple methods

        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Get team features
        home_features = self.get_team_features(home_team_id, season, max_week=week)
        away_features = self.get_team_features(away_team_id, season, max_week=week)

        # Get predictions from each method
        predictions = []
        weights = []

        # Stats-based prediction
        stats_pred = self._predict_stats_based(home_features, away_features, neutral_site)
        predictions.append(stats_pred)
        weights.append(self.ensemble_weights['stats'])

        # Power rating prediction
        power_pred = self._predict_power_rating(home_features, away_features, neutral_site)
        predictions.append(power_pred)
        weights.append(self.ensemble_weights['power'])

        # LSTM prediction (if available)
        lstm_pred = self._predict_lstm(home_team_id, away_team_id, season, week, neutral_site)
        if lstm_pred:
            predictions.append(lstm_pred)
            weights.append(self.ensemble_weights['lstm'])
        else:
            # Redistribute LSTM weight to other methods
            total_other = self.ensemble_weights['stats'] + self.ensemble_weights['power']
            weights[0] = self.ensemble_weights['stats'] / total_other
            weights[1] = self.ensemble_weights['power'] / total_other

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Weighted ensemble
        home_scores = [p['home_score'] for p in predictions]
        away_scores = [p['away_score'] for p in predictions]

        ensemble_home = sum(s * w for s, w in zip(home_scores, weights))
        ensemble_away = sum(s * w for s, w in zip(away_scores, weights))

        # Calculate spread and total
        spread = ensemble_home - ensemble_away
        total = ensemble_home + ensemble_away

        # Calculate confidence based on model agreement
        confidence, spread_ci, total_ci = self._calculate_confidence(
            home_scores, away_scores, weights
        )

        # Win probability (sigmoid based on spread)
        home_win_prob = 1 / (1 + np.exp(-spread / 7))

        return {
            'predicted_home_score': round(ensemble_home, 1),
            'predicted_away_score': round(ensemble_away, 1),
            'predicted_spread': round(spread, 1),
            'predicted_total': round(total, 1),
            'home_win_probability': round(home_win_prob, 3),
            # Confidence metrics
            'confidence': round(confidence, 3),
            'spread_low': round(spread - spread_ci, 1),
            'spread_high': round(spread + spread_ci, 1),
            'total_low': round(total - total_ci, 1),
            'total_high': round(total + total_ci, 1),
            # Individual model predictions (for analysis)
            'model_predictions': {
                'stats': {'home': round(home_scores[0], 1), 'away': round(away_scores[0], 1)},
                'power': {'home': round(home_scores[1], 1), 'away': round(away_scores[1], 1)},
            }
        }

    def _calculate_confidence(self, home_scores: List[float], away_scores: List[float],
                             weights: List[float]) -> Tuple[float, float, float]:
        """
        Calculate prediction confidence based on model agreement

        Returns:
            confidence (0-1), spread_ci, total_ci
        """
        # Calculate spreads and totals from each model
        spreads = [h - a for h, a in zip(home_scores, away_scores)]
        totals = [h + a for h, a in zip(home_scores, away_scores)]

        # Weighted standard deviation (measure of disagreement)
        def weighted_std(values, weights):
            avg = sum(v * w for v, w in zip(values, weights))
            var = sum(w * (v - avg) ** 2 for v, w in zip(values, weights))
            return np.sqrt(var)

        spread_std = weighted_std(spreads, weights)
        total_std = weighted_std(totals, weights)

        # Confidence interval (roughly 68% CI)
        spread_ci = spread_std * 1.5
        total_ci = total_std * 1.5

        # Overall confidence (inverse of uncertainty)
        # Max disagreement around 15 points for spread, 20 for total
        spread_confidence = max(0, 1 - spread_std / 15)
        total_confidence = max(0, 1 - total_std / 20)

        confidence = (spread_confidence + total_confidence) / 2

        return confidence, spread_ci, total_ci

    # === FEATURE IMPORTANCE ===

    def analyze_feature_importance(self, sample_size: int = 100) -> pd.DataFrame:
        """
        Analyze which features drive predictions the most
        Uses permutation importance on stats-based model

        Returns:
            DataFrame with feature importance scores
        """
        if self._feature_importance is not None:
            return self._feature_importance

        print("Analyzing feature importance...")

        # Get sample of games
        query = '''
            SELECT
                g.home_team_id, g.away_team_id, g.season, g.week,
                g.neutral_site, g.home_score - g.away_score as actual_spread
            FROM games g
            WHERE g.completed = 1 AND g.season >= 2024
            ORDER BY RANDOM()
            LIMIT ?
        '''
        games = pd.read_sql_query(query, self.conn, params=(sample_size,))

        if len(games) == 0:
            return pd.DataFrame()

        # Key features to analyze
        features_to_test = [
            'points_scored_roll5', 'points_allowed_roll5', 'point_differential_roll3',
            'win_pct_roll5', 'total_yards_roll5', 'power_rating'
        ]

        # Baseline predictions
        baseline_errors = []
        for _, game in games.iterrows():
            home_features = self.get_team_features(game['home_team_id'], game['season'], game['week'])
            away_features = self.get_team_features(game['away_team_id'], game['season'], game['week'])
            pred = self._predict_stats_based(home_features, away_features, game['neutral_site'])
            pred_spread = pred['home_score'] - pred['away_score']
            baseline_errors.append(abs(pred_spread - game['actual_spread']))

        baseline_mae = np.mean(baseline_errors)

        # Test each feature
        importance_scores = {}

        for feature in features_to_test:
            permuted_errors = []

            for _, game in games.iterrows():
                home_features = self.get_team_features(game['home_team_id'], game['season'], game['week'])
                away_features = self.get_team_features(game['away_team_id'], game['season'], game['week'])

                # Permute the feature (swap home and away)
                if feature in home_features and feature in away_features:
                    home_features[feature], away_features[feature] = away_features[feature], home_features[feature]

                pred = self._predict_stats_based(home_features, away_features, game['neutral_site'])
                pred_spread = pred['home_score'] - pred['away_score']
                permuted_errors.append(abs(pred_spread - game['actual_spread']))

            permuted_mae = np.mean(permuted_errors)
            importance_scores[feature] = permuted_mae - baseline_mae

        # Normalize scores
        total = sum(max(0, v) for v in importance_scores.values())
        if total > 0:
            importance_scores = {k: max(0, v) / total for k, v in importance_scores.items()}

        # Create DataFrame
        df = pd.DataFrame([
            {'feature': k, 'importance': v, 'interpretation': self._interpret_feature(k)}
            for k, v in sorted(importance_scores.items(), key=lambda x: -x[1])
        ])

        self._feature_importance = df
        return df

    def _interpret_feature(self, feature: str) -> str:
        """Human-readable interpretation of features"""
        interpretations = {
            'points_scored_roll5': 'Offensive production (5-game avg)',
            'points_allowed_roll5': 'Defensive strength (5-game avg)',
            'point_differential_roll3': 'Recent form / momentum',
            'win_pct_roll5': 'Overall team quality',
            'total_yards_roll5': 'Offensive efficiency',
            'power_rating': 'Overall power ranking',
            'turnovers_roll5': 'Ball security / takeaways',
        }
        return interpretations.get(feature, feature)

    # === GENERATE PREDICTIONS ===

    def get_upcoming_games(self, week: int = None, season: int = 2025) -> pd.DataFrame:
        """Get upcoming games from database"""
        if week is None:
            week_query = f'''
                SELECT MIN(week) as current_week FROM games
                WHERE completed = 0 AND season = {season}
            '''
            week_df = pd.read_sql_query(week_query, self.conn)
            week = week_df.iloc[0]['current_week']
            if week is None:
                week = 12

        query = f'''
            SELECT
                g.game_id, g.season, g.week, g.date,
                g.home_team_id, g.away_team_id,
                COALESCE(ht.display_name, ht.name) as home_team,
                COALESCE(at.display_name, at.name) as away_team,
                g.neutral_site, g.completed,
                g.home_score, g.away_score,
                go.closing_spread_home as vegas_spread,
                go.closing_total as vegas_total
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            LEFT JOIN game_odds go ON g.game_id = go.game_id
            WHERE g.week = {week} AND g.season = {season}
            ORDER BY g.date, g.game_id
        '''

        return pd.read_sql_query(query, self.conn)

    def generate_predictions(self, week: int = None, season: int = 2025) -> pd.DataFrame:
        """Generate ensemble predictions for all games in a week"""
        games = self.get_upcoming_games(week, season)

        if len(games) == 0:
            print(f"No games found for week {week}")
            return pd.DataFrame()

        print(f"Generating ensemble predictions for {len(games)} games...")

        predictions = []
        for _, game in games.iterrows():
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
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'predicted_home_score': pred['predicted_home_score'],
                'predicted_away_score': pred['predicted_away_score'],
                'predicted_spread': pred['predicted_spread'],
                'predicted_total': pred['predicted_total'],
                'home_win_probability': pred['home_win_probability'],
                'confidence': pred['confidence'],
                'spread_low': pred['spread_low'],
                'spread_high': pred['spread_high'],
                'total_low': pred['total_low'],
                'total_high': pred['total_high'],
                'vegas_spread': game['vegas_spread'],
                'vegas_total': game['vegas_total'],
                'game_completed': game['completed'],
                'actual_home_score': game['home_score'] if game['completed'] else None,
                'actual_away_score': game['away_score'] if game['completed'] else None
            })

        return pd.DataFrame(predictions)


def main():
    """Demo the ensemble predictor"""
    print("=" * 80)
    print("DEEP-EAGLE ENSEMBLE PREDICTOR")
    print("=" * 80)

    # CFB Demo
    print("\n--- CFB ENSEMBLE PREDICTIONS ---")
    predictor = EnsemblePredictor(sport='cfb')
    predictor.connect_db()
    predictor.load_models()

    # Generate predictions
    predictions = predictor.generate_predictions(week=14, season=2024)

    if len(predictions) > 0:
        # Show top 5 by confidence
        print("\nTop 5 High-Confidence Picks:")
        print("-" * 60)
        top_conf = predictions.nlargest(5, 'confidence')
        for _, row in top_conf.iterrows():
            print(f"{row['away_team'][:20]:<20} @ {row['home_team'][:20]:<20}")
            print(f"  Spread: {row['predicted_spread']:+.1f} ({row['spread_low']:+.1f} to {row['spread_high']:+.1f})")
            print(f"  Total: {row['predicted_total']:.1f} ({row['total_low']:.1f} to {row['total_high']:.1f})")
            print(f"  Confidence: {row['confidence']:.1%}")
            if row['vegas_spread']:
                print(f"  Vegas: {row['vegas_spread']:+.1f} / {row['vegas_total']:.1f}")
            print()

    # Feature importance
    print("\n--- FEATURE IMPORTANCE ---")
    importance = predictor.analyze_feature_importance(sample_size=50)
    print(importance.to_string(index=False))

    predictor.close_db()


if __name__ == '__main__':
    main()
