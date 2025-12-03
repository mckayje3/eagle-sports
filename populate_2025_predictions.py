"""
Populate prediction cache with Deep Eagle predictions for 2025 season
Supports both CFB and NFL, including backfilling completed weeks WITHOUT data leakage
"""
import sqlite3
import torch
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os


class DeepEagleModel(torch.nn.Module):
    """Deep Eagle neural network for score prediction"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(DeepEagleModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.feature_extractor = torch.nn.Sequential(*layers)

        self.home_score_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

        self.away_score_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        home_score = self.home_score_head(features)
        away_score = self.away_score_head(features)
        return torch.cat([home_score, away_score], dim=1)


class SimpleFeatureExtractor:
    """Simple feature extractor using season-to-date stats only (no future data)"""

    def __init__(self, db_path, sport='cfb'):
        self.db_path = db_path
        self.sport = sport
        self.conn = sqlite3.connect(db_path)

    def get_team_stats_before_week(self, team_id, season, week):
        """Get team's stats from games BEFORE this week (no data leakage)"""
        cursor = self.conn.cursor()

        # Get average stats from completed games before this week
        cursor.execute('''
            SELECT
                AVG(tgs.points) as avg_points,
                AVG(tgs.total_yards) as avg_yards,
                AVG(tgs.rushing_yards) as avg_rush_yards,
                AVG(tgs.passing_yards) as avg_pass_yards,
                AVG(tgs.turnovers) as avg_turnovers,
                COUNT(*) as games_played
            FROM team_game_stats tgs
            JOIN games g ON tgs.game_id = g.game_id
            WHERE tgs.team_id = ?
            AND g.season = ?
            AND g.week < ?
            AND g.completed = 1
        ''', (team_id, season, week))

        result = cursor.fetchone()

        if result and result[5] > 0:  # Has games
            return {
                'avg_points': result[0] or 0,
                'avg_yards': result[1] or 0,
                'avg_rush_yards': result[2] or 0,
                'avg_pass_yards': result[3] or 0,
                'avg_turnovers': result[4] or 0,
                'games_played': result[5]
            }

        # If no games this season, use previous season average
        cursor.execute('''
            SELECT
                AVG(tgs.points) as avg_points,
                AVG(tgs.total_yards) as avg_yards,
                AVG(tgs.rushing_yards) as avg_rush_yards,
                AVG(tgs.passing_yards) as avg_pass_yards,
                AVG(tgs.turnovers) as avg_turnovers
            FROM team_game_stats tgs
            JOIN games g ON tgs.game_id = g.game_id
            WHERE tgs.team_id = ?
            AND g.season = ?
            AND g.completed = 1
        ''', (team_id, season - 1))

        result = cursor.fetchone()

        if result and result[0] is not None:
            return {
                'avg_points': result[0] or 0,
                'avg_yards': result[1] or 0,
                'avg_rush_yards': result[2] or 0,
                'avg_pass_yards': result[3] or 0,
                'avg_turnovers': result[4] or 0,
                'games_played': 0
            }

        # Default stats
        default_pts = 25 if self.sport == 'cfb' else 22
        return {
            'avg_points': default_pts,
            'avg_yards': 350,
            'avg_rush_yards': 150,
            'avg_pass_yards': 200,
            'avg_turnovers': 1.5,
            'games_played': 0
        }

    def get_odds_data(self, game_id):
        """Get odds for a game"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT
                COALESCE(current_spread_home, opening_spread_home, closing_spread_home) as spread,
                COALESCE(current_total, opening_total, closing_total) as total
            FROM game_odds
            WHERE game_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
        ''', (game_id,))

        result = cursor.fetchone()
        if result:
            return {'spread': result[0], 'total': result[1]}
        return {'spread': None, 'total': None}

    def extract_game_features(self, game_row):
        """Extract features for a single game"""
        game_id = game_row['game_id']
        season = game_row['season']
        week = game_row['week']
        home_team_id = game_row['home_team_id']
        away_team_id = game_row['away_team_id']

        # Get stats for both teams (only using data from before this week)
        home_stats = self.get_team_stats_before_week(home_team_id, season, week)
        away_stats = self.get_team_stats_before_week(away_team_id, season, week)

        # Get odds
        odds = self.get_odds_data(game_id)

        # Build feature dict
        features = {
            'game_id': game_id,
            'season': season,
            'week': week,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,

            # Home team stats
            'home_avg_points': home_stats['avg_points'],
            'home_avg_yards': home_stats['avg_yards'],
            'home_avg_rush': home_stats['avg_rush_yards'],
            'home_avg_pass': home_stats['avg_pass_yards'],
            'home_avg_to': home_stats['avg_turnovers'],
            'home_games': home_stats['games_played'],

            # Away team stats
            'away_avg_points': away_stats['avg_points'],
            'away_avg_yards': away_stats['avg_yards'],
            'away_avg_rush': away_stats['avg_rush_yards'],
            'away_avg_pass': away_stats['avg_pass_yards'],
            'away_avg_to': away_stats['avg_turnovers'],
            'away_games': away_stats['games_played'],

            # Differentials
            'points_diff': home_stats['avg_points'] - away_stats['avg_points'],
            'yards_diff': home_stats['avg_yards'] - away_stats['avg_yards'],
            'rush_diff': home_stats['avg_rush_yards'] - away_stats['avg_rush_yards'],
            'pass_diff': home_stats['avg_pass_yards'] - away_stats['avg_pass_yards'],

            # Odds
            'vegas_spread': odds['spread'] if odds['spread'] else 0,
            'vegas_total': odds['total'] if odds['total'] else 45,

            # Contextual
            'is_neutral': game_row.get('neutral_site', 0),
            'week_num': week
        }

        return features

    def close(self):
        self.conn.close()


def generate_predictions_for_sport(sport, season=2025, weeks=None, backfill=False):
    """Generate predictions for a sport"""
    print(f"\n{'='*80}")
    print(f"GENERATING {sport.upper()} 2025 PREDICTIONS")
    print('='*80)

    db_path = f'{sport}_games.db'
    model_path = f'models/deep_eagle_{sport}_2025.pt'
    scaler_path = f'models/deep_eagle_{sport}_2025_scaler.pkl'

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return []

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    feature_cols = checkpoint['feature_cols']

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    input_dim = len(feature_cols)
    model = DeepEagleModel(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model with {input_dim} features")

    # Get games
    conn = sqlite3.connect(db_path)

    if backfill:
        # Get ALL games (completed and upcoming) - for backfilling we need completed games
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
                g.home_score,
                g.away_score,
                g.completed
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.season = ?
            AND g.home_team_id > 0 AND g.away_team_id > 0
        '''
        params = [season]
    else:
        # Only upcoming games
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
                g.home_score,
                g.away_score,
                g.completed
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.season = ? AND g.completed = 0
        '''
        params = [season]

    if weeks:
        week_list = ','.join(map(str, weeks))
        query += f' AND g.week IN ({week_list})'

    query += ' ORDER BY g.week, g.date'

    games_df = pd.read_sql_query(query, conn, params=params)
    print(f"Found {len(games_df)} games")

    if len(games_df) == 0:
        conn.close()
        return []

    # Extract features
    extractor = SimpleFeatureExtractor(db_path, sport)
    all_predictions = []

    for week in sorted(games_df['week'].unique()):
        week_games = games_df[games_df['week'] == week]
        print(f"\nWeek {week}: {len(week_games)} games")

        for _, game in week_games.iterrows():
            try:
                features = extractor.extract_game_features(game.to_dict())

                # Prepare feature vector (use model's expected feature order)
                feature_vector = []
                for col in feature_cols:
                    if col in features:
                        feature_vector.append(features[col])
                    else:
                        feature_vector.append(0)

                X = np.array([feature_vector], dtype=np.float32)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                X = scaler.transform(X)

                X_tensor = torch.FloatTensor(X).to(device)

                with torch.no_grad():
                    pred = model(X_tensor).cpu().numpy()[0]

                pred_home = float(pred[0])
                pred_away = float(pred[1])
                pred_spread = pred_home - pred_away
                pred_total = pred_home + pred_away

                # Win probability (simple logistic)
                win_prob = 1 / (1 + np.exp(-pred_spread / 7))

                # Confidence (based on spread magnitude and data quality)
                games_played = features['home_games'] + features['away_games']
                base_conf = 0.75 + min(games_played / 20, 0.15)
                spread_conf = min(abs(pred_spread) / 30, 0.10)
                confidence = base_conf + spread_conf

                # Get odds for comparison
                vegas_spread = features.get('vegas_spread', 0)
                vegas_total = features.get('vegas_total', 0)

                prediction = {
                    'game_id': game['game_id'],
                    'sport': sport.upper(),
                    'season': season,
                    'week': int(game['week']),
                    'game_date': game['date'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'predicted_home_score': round(pred_home, 1),
                    'predicted_away_score': round(pred_away, 1),
                    'predicted_spread': round(pred_spread, 1),
                    'predicted_total': round(pred_total, 1),
                    'home_win_probability': round(win_prob, 3),
                    'vegas_spread': vegas_spread,
                    'vegas_total': vegas_total,
                    'game_completed': int(game['completed']),
                    'actual_home_score': game['home_score'] if game['completed'] else None,
                    'actual_away_score': game['away_score'] if game['completed'] else None,
                    'confidence': round(confidence, 3),
                    'spread_low': round(pred_spread - 5, 1),
                    'spread_high': round(pred_spread + 5, 1),
                    'total_low': round(pred_total - 8, 1),
                    'total_high': round(pred_total + 8, 1)
                }

                all_predictions.append(prediction)

            except Exception as e:
                print(f"  Error on game {game['game_id']}: {e}")
                continue

    extractor.close()
    conn.close()

    print(f"\nGenerated {len(all_predictions)} predictions")
    return all_predictions


def save_predictions_to_cache(predictions):
    """Save predictions to users.db prediction_cache"""
    if not predictions:
        print("No predictions to save")
        return

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Group by sport and week for summary
    summary = {}

    for pred in predictions:
        sport = pred['sport']
        week = pred['week']
        key = f"{sport} Week {week}"
        summary[key] = summary.get(key, 0) + 1

        # Delete existing prediction for this game
        cursor.execute(
            'DELETE FROM prediction_cache WHERE game_id = ? AND sport = ?',
            (pred['game_id'], pred['sport'])
        )

        # Insert new prediction
        cursor.execute('''
            INSERT INTO prediction_cache (
                sport, game_id, season, week, game_date,
                home_team, away_team,
                predicted_home_score, predicted_away_score,
                predicted_spread, predicted_total,
                home_win_probability,
                vegas_spread, vegas_total,
                game_completed, actual_home_score, actual_away_score,
                created_at, confidence,
                spread_low, spread_high, total_low, total_high
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pred['sport'],
            pred['game_id'],
            pred['season'],
            pred['week'],
            pred['game_date'],
            pred['home_team'],
            pred['away_team'],
            pred['predicted_home_score'],
            pred['predicted_away_score'],
            pred['predicted_spread'],
            pred['predicted_total'],
            pred['home_win_probability'],
            pred['vegas_spread'],
            pred['vegas_total'],
            pred['game_completed'],
            pred['actual_home_score'],
            pred['actual_away_score'],
            datetime.now().isoformat(),
            pred['confidence'],
            pred['spread_low'],
            pred['spread_high'],
            pred['total_low'],
            pred['total_high']
        ))

    conn.commit()
    conn.close()

    print("\nSaved predictions to cache:")
    for key, count in sorted(summary.items()):
        print(f"  {key}: {count} games")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate 2025 predictions')
    parser.add_argument('--sport', choices=['nfl', 'cfb', 'both'], default='both')
    parser.add_argument('--weeks', type=str, help='Comma-separated list of weeks (e.g., 14,15)')
    parser.add_argument('--backfill', action='store_true', help='Include completed games')
    parser.add_argument('--all-weeks', action='store_true', help='Generate for all 2025 weeks')

    args = parser.parse_args()

    weeks = None
    if args.weeks:
        weeks = [int(w.strip()) for w in args.weeks.split(',')]
    elif args.all_weeks:
        # All possible weeks
        weeks = list(range(0, 18))

    all_predictions = []

    if args.sport in ['nfl', 'both']:
        predictions = generate_predictions_for_sport('nfl', 2025, weeks, args.backfill)
        all_predictions.extend(predictions)

    if args.sport in ['cfb', 'both']:
        predictions = generate_predictions_for_sport('cfb', 2025, weeks, args.backfill)
        all_predictions.extend(predictions)

    save_predictions_to_cache(all_predictions)

    print(f"\n{'='*80}")
    print("PREDICTION GENERATION COMPLETE")
    print(f"Total predictions: {len(all_predictions)}")
    print('='*80)


if __name__ == '__main__':
    main()
