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
    """Deep Eagle neural network for score prediction - supports both old and new architectures"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], head_hidden=32, use_old_names=False):
        super(DeepEagleModel, self).__init__()
        self.use_old_names = use_old_names

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        if use_old_names:
            self.features = nn.Sequential(*layers)
            self.home_head = nn.Sequential(
                nn.Linear(prev_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1)
            )
            self.away_head = nn.Sequential(
                nn.Linear(prev_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1)
            )
        else:
            self.feature_extractor = nn.Sequential(*layers)
            self.home_score_head = nn.Sequential(
                nn.Linear(prev_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1)
            )
            self.away_score_head = nn.Sequential(
                nn.Linear(prev_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1)
            )

    def forward(self, x):
        if self.use_old_names:
            features = self.features(x)
            home_score = self.home_head(features)
            away_score = self.away_head(features)
        else:
            features = self.feature_extractor(x)
            home_score = self.home_score_head(features)
            away_score = self.away_score_head(features)
        return torch.cat([home_score, away_score], dim=1)


def infer_model_architecture(state_dict):
    """Infer model architecture from state dict layer shapes"""
    use_old_names = any(key.startswith('features.') for key in state_dict.keys())
    prefix = 'features' if use_old_names else 'feature_extractor'
    head_prefix = 'home_head' if use_old_names else 'home_score_head'

    # Infer hidden dims from feature extractor layers
    hidden_dims = []
    layer_idx = 0
    while f'{prefix}.{layer_idx}.weight' in state_dict:
        weight = state_dict[f'{prefix}.{layer_idx}.weight']
        hidden_dims.append(weight.shape[0])
        layer_idx += 4  # Skip linear, batchnorm, relu, dropout

    # Infer input dim
    input_dim = state_dict[f'{prefix}.0.weight'].shape[1]

    # Infer head hidden dim
    head_hidden = state_dict[f'{head_prefix}.0.weight'].shape[0]

    return input_dim, hidden_dims, head_hidden, use_old_names


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
        import os

        # Try primary model, fall back to 2024 if not found
        model_paths = [
            (self.model_path, self.scaler_path),
            ('models/deep_eagle_nba_2024.pt', 'models/deep_eagle_nba_2024_scaler.pkl'),
            ('models/deep_eagle_nba_2023.pt', 'models/deep_eagle_nba_2023_scaler.pkl'),
        ]

        for model_path, scaler_path in model_paths:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    self.feature_cols = checkpoint.get('feature_cols', [])

                    # Infer architecture from saved state dict
                    state_dict = checkpoint['model_state_dict']
                    input_dim, hidden_dims, head_hidden, use_old_names = infer_model_architecture(state_dict)

                    # Rebuild model with correct architecture
                    self.model = DeepEagleModel(
                        input_dim, hidden_dims=hidden_dims, head_hidden=head_hidden, use_old_names=use_old_names
                    ).to(self.device)
                    self.model.load_state_dict(state_dict)
                    self.model.eval()

                    # Load scaler
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)

                    arch_type = "old" if use_old_names else "new"
                    print(f"Loaded NBA Deep Eagle model from {model_path}")
                    print(f"  Features: {input_dim}, Hidden: {hidden_dims}, Architecture: {arch_type}")
                    return

                except Exception as e:
                    print(f"Error loading {model_path}: {e}")
                    continue

        print("No NBA model found. Train a model first: py train_deep_eagle_nba.py 2025 nba_2025_deep_eagle_features.csv")
        self.model = None

    def get_upcoming_games(self, days=7):
        """Get upcoming NBA games from database"""
        from datetime import datetime, timedelta

        conn = sqlite3.connect(self.db_path)

        # NBA games are stored with ISO timestamps (e.g., 2025-12-18T01:00Z)
        # Generate date range strings that match the database format
        now = datetime.utcnow()
        start_date = (now - timedelta(hours=12)).strftime('%Y-%m-%d')
        end_date = (now + timedelta(days=days)).strftime('%Y-%m-%d')

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
                AND g.date >= ?
                AND g.date <= ?
            ORDER BY g.date
        '''

        games = pd.read_sql_query(query, conn, params=(start_date, end_date + 'Z'))
        conn.close()

        return games

    def extract_features(self, game_row):
        """Extract features for a single game for prediction - matches training feature set"""
        conn = sqlite3.connect(self.db_path)

        features = {}

        # Convert numpy types to native Python types for SQLite compatibility
        # numpy.int64 doesn't work properly as SQLite parameters
        season = int(game_row.get('season', 2025))
        home_team_id = int(game_row['home_team_id'])
        away_team_id = int(game_row['away_team_id'])
        game_id = int(game_row['game_id'])
        game_date = game_row['date']

        # Season progress features
        games_into_season = self._get_games_into_season(conn, home_team_id, season, game_date)
        features['games_into_season'] = games_into_season
        features['season_progress'] = min(1.0, games_into_season / 82)
        features['attendance'] = 0  # Not available for predictions

        # Get historical stats for both teams
        home_stats = self._get_team_stats(conn, home_team_id, season)
        away_stats = self._get_team_stats(conn, away_team_id, season)

        for key, value in home_stats.items():
            features[f'home_hist_{key}'] = value
        for key, value in away_stats.items():
            features[f'away_hist_{key}'] = value

        # Get recent form (last 10 games)
        home_recent = self._get_recent_form(conn, home_team_id, season, game_date)
        away_recent = self._get_recent_form(conn, away_team_id, season, game_date)

        for key, value in home_recent.items():
            features[f'home_recent_{key}'] = value
        for key, value in away_recent.items():
            features[f'away_recent_{key}'] = value

        # Rest days and back-to-back
        features['home_rest_days'] = self._get_rest_days(conn, home_team_id, game_date)
        features['away_rest_days'] = self._get_rest_days(conn, away_team_id, game_date)
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        features['home_b2b'] = 1 if features['home_rest_days'] == 0 else 0
        features['away_b2b'] = 1 if features['away_rest_days'] == 0 else 0

        # Get odds
        odds = self._get_odds(conn, game_id)
        for key, value in odds.items():
            features[f'odds_{key}'] = value

        # Calculate differentials - use correct column names
        features['ppg_differential'] = home_stats.get('ppg', 0) - away_stats.get('ppg', 0)
        features['papg_differential'] = home_stats.get('papg', 0) - away_stats.get('papg', 0)
        features['win_pct_differential'] = home_stats.get('win_pct', 0) - away_stats.get('win_pct', 0)
        features['fg_pct_differential'] = home_stats.get('fg_pct', 0) - away_stats.get('fg_pct', 0)
        features['three_pct_differential'] = home_stats.get('three_pct', 0) - away_stats.get('three_pct', 0)
        features['rebound_differential'] = home_stats.get('rebounds_pg', 0) - away_stats.get('rebounds_pg', 0)
        features['assist_differential'] = home_stats.get('assists_pg', 0) - away_stats.get('assists_pg', 0)
        features['turnover_differential'] = home_stats.get('turnovers_pg', 0) - away_stats.get('turnovers_pg', 0)

        # Recent form differentials
        features['recent_ppg_diff'] = home_recent.get('ppg', 0) - away_recent.get('ppg', 0)
        features['recent_win_pct_diff'] = home_recent.get('win_pct', 0) - away_recent.get('win_pct', 0)

        # Venue-adjusted differentials (key for home court advantage)
        features['venue_ppg_differential'] = home_stats.get('home_ppg', 0) - away_stats.get('away_ppg', 0)
        features['venue_win_pct_differential'] = home_stats.get('home_win_pct', 0) - away_stats.get('away_win_pct', 0)

        # Combined home court advantage
        features['combined_home_advantage'] = (
            home_stats.get('home_away_ppg_diff', 0) + away_stats.get('home_away_ppg_diff', 0)
        ) / 2

        # Stats reliability based on season progress
        games_played = home_stats.get('games_played', 0)
        features['stats_reliability'] = games_played / (games_played + 10)
        features['vegas_reliability'] = 10 / (games_played + 10)
        features['prev_season_weight'] = max(0, 1 - games_played / 15)

        # Previous season features (simplified for prediction)
        features['prev_season_ppg_diff'] = 0
        features['prev_season_win_pct_diff'] = 0

        # Weighted features
        features['weighted_ppg_diff'] = features['ppg_differential'] * features['stats_reliability']
        features['weighted_vegas_spread'] = features.get('odds_latest_spread', 0) * features['vegas_reliability']
        features['blended_ppg_diff'] = features['weighted_ppg_diff']

        conn.close()
        return features

    def _get_games_into_season(self, conn, team_id, season, current_date):
        """Get number of games team has played so far this season"""
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) as games
            FROM games
            WHERE season = ? AND completed = 1
            AND (home_team_id = ? OR away_team_id = ?)
            AND date < ?
        ''', (season, team_id, team_id, current_date))
        result = cursor.fetchone()
        return result[0] if result else 0

    def _get_recent_form(self, conn, team_id, season, current_date, n_games=10):
        """Get team's recent form (last N games)"""
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                AVG(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as ppg,
                AVG(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as win_pct,
                COUNT(*) as games
            FROM (
                SELECT * FROM games
                WHERE season = ? AND completed = 1
                    AND (home_team_id = ? OR away_team_id = ?)
                    AND date < ?
                ORDER BY date DESC
                LIMIT ?
            )
        ''', (team_id, team_id, team_id, season, team_id, team_id, current_date, n_games))

        row = cursor.fetchone()
        if not row or row[3] == 0:
            return {'games': 0, 'ppg': 0, 'papg': 0, 'win_pct': 0}

        return {
            'games': row[3],
            'ppg': row[0] or 0,
            'papg': row[1] or 0,
            'win_pct': row[2] or 0
        }

    def _get_rest_days(self, conn, team_id, current_date):
        """Get rest days since team's last game"""
        cursor = conn.cursor()

        cursor.execute('''
            SELECT date FROM games
            WHERE completed = 1
                AND (home_team_id = ? OR away_team_id = ?)
                AND date < ?
            ORDER BY date DESC
            LIMIT 1
        ''', (team_id, team_id, current_date))

        row = cursor.fetchone()
        if not row:
            return 3  # Default rest days if no previous game

        try:
            # Parse dates - handle both ISO and date formats
            last_date = row[0][:10] if 'T' in row[0] else row[0]
            curr_date = current_date[:10] if 'T' in str(current_date) else str(current_date)

            last = datetime.strptime(last_date, '%Y-%m-%d')
            curr = datetime.strptime(curr_date, '%Y-%m-%d')
            rest = (curr - last).days - 1  # Subtract 1 because game day doesn't count
            return max(0, rest)
        except Exception:
            return 1

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

        # Get box score stats - use column names matching training data
        cursor.execute('''
            SELECT
                AVG(ts.field_goal_pct) as fg_pct,
                AVG(ts.three_point_pct) as three_pct,
                AVG(ts.free_throw_pct) as ft_pct,
                AVG(ts.total_rebounds) as rebounds_pg,
                AVG(ts.offensive_rebounds) as off_rebounds_pg,
                AVG(ts.defensive_rebounds) as def_rebounds_pg,
                AVG(ts.assists) as assists_pg,
                AVG(ts.turnovers) as turnovers_pg,
                AVG(ts.steals) as steals_pg,
                AVG(ts.blocks) as blocks_pg,
                AVG(ts.points_in_paint) as paint_pg,
                AVG(ts.fast_break_points) as fastbreak_pg,
                AVG(ts.bench_points) as bench_pg
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
            stats['fg_pct'] = box_row[0] or 47.0  # Default NBA average
            stats['three_pct'] = box_row[1] or 36.0
            stats['ft_pct'] = box_row[2] or 77.0
            stats['rebounds_pg'] = box_row[3] or 44.0
            stats['off_rebounds_pg'] = box_row[4] or 10.0
            stats['def_rebounds_pg'] = box_row[5] or 34.0
            stats['assists_pg'] = box_row[6] or 25.0
            stats['turnovers_pg'] = box_row[7] or 14.0
            stats['steals_pg'] = box_row[8] or 7.5
            stats['blocks_pg'] = box_row[9] or 5.0
            # Use training data means for columns that may be NULL
            stats['paint_pg'] = box_row[10] or 48.0  # Training mean ~48
            stats['fastbreak_pg'] = box_row[11] or 13.0  # Training mean ~13
            stats['bench_pg'] = box_row[12] or 35.0  # Training mean ~35
        else:
            # Use NBA averages when no data available
            stats['fg_pct'] = 47.0
            stats['three_pct'] = 36.0
            stats['ft_pct'] = 77.0
            stats['rebounds_pg'] = 44.0
            stats['off_rebounds_pg'] = 10.0
            stats['def_rebounds_pg'] = 34.0
            stats['assists_pg'] = 25.0
            stats['turnovers_pg'] = 14.0
            stats['steals_pg'] = 7.5
            stats['blocks_pg'] = 5.0
            stats['paint_pg'] = 48.0
            stats['fastbreak_pg'] = 13.0
            stats['bench_pg'] = 35.0

        return stats

    def _empty_stats(self):
        """Return empty stats dict - column names match training data"""
        return {
            'games_played': 0, 'ppg': 0, 'papg': 0, 'win_pct': 0,
            'fg_pct': 0, 'three_pct': 0, 'ft_pct': 0,
            'rebounds_pg': 0, 'off_rebounds_pg': 0, 'def_rebounds_pg': 0,
            'assists_pg': 0, 'turnovers_pg': 0,
            'steals_pg': 0, 'blocks_pg': 0,
            'paint_pg': 0, 'fastbreak_pg': 0, 'bench_pg': 0,
            'home_games': 0, 'home_ppg': 0, 'home_papg': 0, 'home_win_pct': 0,
            'away_games': 0, 'away_ppg': 0, 'away_papg': 0, 'away_win_pct': 0,
            'home_away_ppg_diff': 0
        }

    def _get_odds(self, conn, game_id):
        """Get odds for a game from odds_and_predictions table - names match training data"""
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                opening_spread, latest_spread,
                opening_total, latest_total,
                spread_movement, total_movement
            FROM odds_and_predictions WHERE game_id = ?
        ''', (game_id,))

        row = cursor.fetchone()

        # Use training data means when odds are missing
        default_total = 223.0

        if not row:
            return {
                'opening_spread': 0, 'latest_spread': 0,
                'opening_total': default_total, 'latest_total': default_total,
                'spread_movement': 0, 'total_movement': 0,
                'spread_movement_abs': 0, 'total_movement_abs': 0,
                'spread_movement_significant': 0, 'total_movement_significant': 0,
                'spread_movement_sig_direction': 0, 'total_movement_sig_direction': 0,
            }

        opening_spread = row[0] or 0
        latest_spread = row[1] or row[0] or 0
        opening_total = row[2] or default_total
        latest_total = row[3] or row[2] or default_total

        # Calculate movement
        spread_movement = row[4] if row[4] is not None else (latest_spread - opening_spread)
        total_movement = row[5] if row[5] is not None else (latest_total - opening_total)

        # Threshold features: significant movement >= 2.0 points
        spread_significant = abs(spread_movement) >= 2.0
        total_significant = abs(total_movement) >= 2.0

        return {
            'opening_spread': opening_spread,
            'latest_spread': latest_spread,
            'opening_total': opening_total,
            'latest_total': latest_total,
            'spread_movement': spread_movement,
            'total_movement': total_movement,
            'spread_movement_abs': abs(spread_movement),
            'total_movement_abs': abs(total_movement),
            'spread_movement_significant': 1 if spread_significant else 0,
            'total_movement_significant': 1 if total_significant else 0,
            'spread_movement_sig_direction': spread_movement if spread_significant else 0,
            'total_movement_sig_direction': total_movement if total_significant else 0,
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
                odds = self._get_odds(conn, int(game['game_id']))
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
                    'vegas_spread': odds['latest_spread'],
                    'vegas_total': odds['latest_total'],
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
