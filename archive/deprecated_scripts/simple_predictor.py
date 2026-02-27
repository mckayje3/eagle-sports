"""
Simple Statistical Predictor
Uses team statistics to generate reasonable predictions
This is a fallback until the deep learning models are properly trained
"""
import sqlite3
import numpy as np
from typing import Dict, Tuple

class SimplePredictor:
    """Simple predictor using team statistics"""

    def __init__(self, db_path='cfb_games.db'):
        self.db_path = db_path

    def get_team_stats(self, team_id: int, season: int, up_to_week: int) -> Dict:
        """Get team statistics up to a given week"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get home games stats
        cursor.execute('''
            SELECT
                COUNT(*) as games,
                AVG(CASE WHEN home_score IS NOT NULL THEN home_score ELSE 0 END) as avg_scored,
                AVG(CASE WHEN away_score IS NOT NULL THEN away_score ELSE 0 END) as avg_allowed,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) as wins
            FROM games
            WHERE home_team_id = ?
              AND season = ?
              AND week < ?
              AND completed = 1
        ''', (team_id, team_id, season, up_to_week))

        home_stats = cursor.fetchone()

        # Get away games stats
        cursor.execute('''
            SELECT
                COUNT(*) as games,
                AVG(CASE WHEN away_score IS NOT NULL THEN away_score ELSE 0 END) as avg_scored,
                AVG(CASE WHEN home_score IS NOT NULL THEN home_score ELSE 0 END) as avg_allowed,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) as wins
            FROM games
            WHERE away_team_id = ?
              AND season = ?
              AND week < ?
              AND completed = 1
        ''', (team_id, team_id, season, up_to_week))

        away_stats = cursor.fetchone()

        conn.close()

        # Combine stats
        total_games = (home_stats[0] or 0) + (away_stats[0] or 0)

        if total_games == 0:
            # Default stats for teams with no history
            return {
                'games': 0,
                'avg_scored': 24.0,
                'avg_allowed': 24.0,
                'win_pct': 0.5
            }

        total_scored = ((home_stats[1] or 0) * (home_stats[0] or 0) +
                       (away_stats[1] or 0) * (away_stats[0] or 0))
        total_allowed = ((home_stats[2] or 0) * (home_stats[0] or 0) +
                        (away_stats[2] or 0) * (away_stats[0] or 0))
        total_wins = (home_stats[3] or 0) + (away_stats[3] or 0)

        return {
            'games': total_games,
            'avg_scored': total_scored / total_games if total_games > 0 else 24.0,
            'avg_allowed': total_allowed / total_games if total_games > 0 else 24.0,
            'win_pct': total_wins / total_games if total_games > 0 else 0.5
        }

    def predict_game(self, home_team_id: int, away_team_id: int,
                    season: int, week: int) -> Dict:
        """Predict a game outcome"""

        # Get team stats
        home_stats = self.get_team_stats(home_team_id, season, week)
        away_stats = self.get_team_stats(away_team_id, season, week)

        # Predict scores using team averages with home field advantage
        home_field_advantage = 3.0

        # Home team score: their avg offense vs opponent's avg defense + HFA
        predicted_home_score = (home_stats['avg_scored'] * 0.6 +
                               (50 - away_stats['avg_allowed']) * 0.4 +
                               home_field_advantage)

        # Away team score: their avg offense vs opponent's avg defense
        predicted_away_score = (away_stats['avg_scored'] * 0.6 +
                               (50 - home_stats['avg_allowed']) * 0.4)

        # Ensure reasonable ranges
        predicted_home_score = max(7, min(56, predicted_home_score))
        predicted_away_score = max(7, min(56, predicted_away_score))

        # Calculate spread and total
        predicted_spread = predicted_home_score - predicted_away_score
        predicted_total = predicted_home_score + predicted_away_score

        # Calculate win probability using logistic function
        # Based on point differential and win percentages
        score_diff = predicted_home_score - predicted_away_score
        win_pct_diff = home_stats['win_pct'] - away_stats['win_pct']

        # Logistic function: 1 / (1 + exp(-x))
        x = (score_diff * 0.1) + (win_pct_diff * 2)
        home_win_prob = 1 / (1 + np.exp(-x))

        # Ensure probability is between 15% and 85%
        home_win_prob = max(0.15, min(0.85, home_win_prob))

        return {
            'predicted_home_score': int(round(predicted_home_score)),
            'predicted_away_score': int(round(predicted_away_score)),
            'predicted_spread': round(predicted_spread, 1),
            'predicted_total': round(predicted_total, 1),
            'home_win_probability': round(home_win_prob, 3)
        }

if __name__ == '__main__':
    # Test the predictor
    predictor = SimplePredictor()

    # Test with a sample game
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT game_id, home_team_id, away_team_id, week
        FROM games
        WHERE season = 2024 AND week = 13
        LIMIT 1
    ''')

    game = cursor.fetchone()
    if game:
        game_id, home_id, away_id, week = game

        prediction = predictor.predict_game(home_id, away_id, 2024, week)

        print("Test Prediction:")
        print(f"  Home Score: {prediction['predicted_home_score']}")
        print(f"  Away Score: {prediction['predicted_away_score']}")
        print(f"  Spread: {prediction['predicted_spread']:+.1f}")
        print(f"  Total: {prediction['predicted_total']:.1f}")
        print(f"  Win Probability: {prediction['home_win_probability']:.1%}")

    conn.close()
