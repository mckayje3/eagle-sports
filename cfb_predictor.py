"""
CFB Deep Eagle Predictor
Makes predictions for upcoming CFB games using Deep Eagle model

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
from datetime import datetime
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


class CFBPredictor:
    """Predict CFB game outcomes using Deep Eagle model

    CFB Deep Eagle walk-forward validation (2025 season):
    - 57.1% ATS at 5+ pt edges (profitable)
    - Better than Enhanced Ridge at 55.9%
    """

    # Post-prediction adjustment constants (based on edge analysis backtest)
    BIG_FAVORITE_THRESHOLD = 14.0   # CFB has larger spreads than NFL
    UNDERDOG_ADJUSTMENT = 1.5       # Points to add toward underdog

    def __init__(self, model_path='models/deep_eagle_cfb_2025.pt',
                 scaler_path='models/deep_eagle_cfb_2025_scaler.pkl',
                 db_path='cfb_games.db'):
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

            print(f"Loaded CFB Deep Eagle model from {self.model_path}")
            print(f"  Features: {len(self.feature_cols)}")

        except FileNotFoundError:
            print(f"Model not found at {self.model_path}")
            print("Train a model first or check the model path")
            self.model = None

    def get_current_week(self):
        """Calculate current CFB week based on date"""
        today = datetime.now()

        # CFB 2025 season starts late August
        if today.year >= 2025 and today.month >= 8:
            season_start = datetime(2025, 8, 28)
            season = 2025
        else:
            season_start = datetime(2024, 8, 29)
            season = 2024

        if today < season_start:
            return 1, season

        days_since_start = (today - season_start).days
        week = (days_since_start // 7) + 1
        week = min(week, 18)  # CFB regular season + bowls

        return week, season

    def get_upcoming_games(self, week=None, days=None, season=None):
        """Get upcoming CFB games from database"""
        conn = sqlite3.connect(self.db_path)

        if week and season:
            query = '''
                SELECT
                    g.game_id,
                    g.date,
                    g.week,
                    g.season,
                    g.home_team_id,
                    g.away_team_id,
                    ht.name as home_team,
                    at.name as away_team,
                    g.neutral_site,
                    g.conference_game,
                    g.temperature,
                    g.wind_speed,
                    g.is_dome
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.team_id
                JOIN teams at ON g.away_team_id = at.team_id
                WHERE g.completed = 0 AND g.week = ? AND g.season = ?
                ORDER BY g.date
            '''
            games = pd.read_sql_query(query, conn, params=(week, season))
        elif week:
            query = '''
                SELECT
                    g.game_id,
                    g.date,
                    g.week,
                    g.season,
                    g.home_team_id,
                    g.away_team_id,
                    ht.name as home_team,
                    at.name as away_team,
                    g.neutral_site,
                    g.conference_game,
                    g.temperature,
                    g.wind_speed,
                    g.is_dome
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
                    ht.name as home_team,
                    at.name as away_team,
                    g.neutral_site,
                    g.conference_game,
                    g.temperature,
                    g.wind_speed,
                    g.is_dome
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
                    ht.name as home_team,
                    at.name as away_team,
                    g.neutral_site,
                    g.conference_game,
                    g.temperature,
                    g.wind_speed,
                    g.is_dome
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
        """Extract features for a single game for prediction - matches training feature names"""
        conn = sqlite3.connect(self.db_path)

        features = {}
        warnings = []

        # Game context - CFB uses 18 weeks max
        features['week_normalized'] = game_row.get('week', 10) / 18.0
        features['neutral_site'] = game_row.get('neutral_site', 0) or 0
        features['conference_game'] = game_row.get('conference_game', 0) or 0

        # Weather features
        features['temperature'] = game_row.get('temperature', 65) or 65
        features['wind_speed'] = game_row.get('wind_speed', 5) or 5
        features['is_dome'] = game_row.get('is_dome', 0) or 0

        # Get historical stats for both teams
        season = game_row.get('season', 2025)
        week = game_row.get('week', 10)
        home_stats = self._get_team_stats(conn, game_row['home_team_id'], season, week)
        away_stats = self._get_team_stats(conn, game_row['away_team_id'], season, week)

        # Check for missing team stats
        if home_stats.get('_missing_stats'):
            warnings.append(f"No game data for {game_row['home_team']} - using defaults")
        if away_stats.get('_missing_stats'):
            warnings.append(f"No game data for {game_row['away_team']} - using defaults")

        # Map stats to expected feature names
        for key, value in home_stats.items():
            if not key.startswith('_'):
                features[f'home_hist_{key}'] = value
        for key, value in away_stats.items():
            if not key.startswith('_'):
                features[f'away_hist_{key}'] = value

        # Get odds with correct feature names
        odds = self._get_odds(conn, game_row['game_id'])

        # Check for missing odds
        if odds.get('_missing_odds'):
            warnings.append(f"No odds data for {game_row['away_team']} @ {game_row['home_team']} - using defaults")

        features['odds_opening_spread'] = odds.get('opening_spread', 0)
        features['odds_latest_spread'] = odds.get('latest_spread', 0)
        features['odds_opening_total'] = odds.get('opening_total', 55)
        features['odds_latest_total'] = odds.get('latest_total', 55)
        features['odds_opening_ml_home'] = odds.get('opening_ml_home', 0)
        features['odds_latest_ml_home'] = odds.get('latest_ml_home', 0)
        features['odds_opening_ml_away'] = odds.get('opening_ml_away', 0)
        features['odds_latest_ml_away'] = odds.get('latest_ml_away', 0)

        # Line movement features
        spread_movement = odds.get('spread_movement', 0)
        total_movement = odds.get('total_movement', 0)
        features['odds_spread_movement'] = spread_movement
        features['odds_total_movement'] = total_movement
        features['odds_spread_movement_abs'] = abs(spread_movement)
        features['odds_total_movement_abs'] = abs(total_movement)

        # Threshold features: CFB uses 4.0 pts for significant spread movement
        spread_significant = abs(spread_movement) >= 4.0
        total_significant = abs(total_movement) >= 3.0
        features['odds_spread_movement_significant'] = 1 if spread_significant else 0
        features['odds_total_movement_significant'] = 1 if total_significant else 0
        features['odds_spread_movement_sig_direction'] = spread_movement if spread_significant else 0
        features['odds_total_movement_sig_direction'] = total_movement if total_significant else 0

        # Store warnings in features for later retrieval
        features['_warnings'] = warnings

        # Drive features
        home_drive = self._get_drive_stats(conn, game_row['home_team_id'], season, week)
        away_drive = self._get_drive_stats(conn, game_row['away_team_id'], season, week)

        for key, value in home_drive.items():
            features[f'home_drive_{key}'] = value
        for key, value in away_drive.items():
            features[f'away_drive_{key}'] = value

        # Calculate differentials matching training feature names
        features['ppg_differential'] = away_stats.get('ppg', 0) - home_stats.get('ppg', 0)
        features['papg_differential'] = away_stats.get('papg', 0) - home_stats.get('papg', 0)
        features['win_pct_differential'] = away_stats.get('win_pct', 0) - home_stats.get('win_pct', 0)
        features['ppd_differential'] = away_drive.get('ppd', 2.5) - home_drive.get('ppd', 2.5)
        features['scoring_pct_differential'] = away_drive.get('scoring_pct', 0.35) - home_drive.get('scoring_pct', 0.35)

        # Venue-based differentials
        features['venue_ppg_differential'] = away_stats.get('away_ppg', 0) - home_stats.get('home_ppg', 0)
        features['venue_win_pct_differential'] = away_stats.get('away_win_pct', 0) - home_stats.get('home_win_pct', 0)

        # Combined home advantage
        features['combined_home_advantage'] = (
            home_stats.get('home_away_ppg_diff', 0) + away_stats.get('home_away_ppg_diff', 0)
        ) / 2

        conn.close()
        return features

    def _get_team_stats(self, conn, team_id, season, current_week):
        """Get team's season statistics - feature names match training data"""
        cursor = conn.cursor()

        # Convert numpy types to native Python
        team_id = int(team_id)
        season = int(season)
        current_week = int(current_week)

        # Get PPG and basic stats
        cursor.execute('''
            SELECT
                COUNT(*) as games_played,
                AVG(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as ppg,
                AVG(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND week < ?
                AND (home_team_id = ? OR away_team_id = ?)
        ''', (team_id, team_id, team_id, season, current_week, team_id, team_id))

        row = cursor.fetchone()
        if not row or row[0] == 0:
            return self._empty_stats()

        stats = {
            'games_played': row[0],
            'ppg': row[1] or 0,
            'papg': row[2] or 0,
            'win_pct': row[3] or 0,
        }

        # Get box score stats for ypg and turnover_pg
        cursor.execute('''
            SELECT
                AVG(ts.total_yards) as ypg,
                AVG(ts.turnovers) as turnover_pg
            FROM team_game_stats ts
            JOIN games g ON ts.game_id = g.game_id
            WHERE ts.team_id = ? AND g.season = ? AND g.week < ? AND g.completed = 1
        ''', (team_id, season, current_week))
        box_row = cursor.fetchone()

        stats['ypg'] = box_row[0] or 400 if box_row else 400  # CFB avg higher than NFL
        stats['turnover_pg'] = box_row[1] or 1.5 if box_row else 1.5

        # Get home stats
        cursor.execute('''
            SELECT
                COUNT(*) as home_games,
                AVG(home_score) as home_ppg,
                AVG(away_score) as home_papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as home_win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND week < ? AND home_team_id = ?
        ''', (team_id, season, current_week, team_id))
        home_row = cursor.fetchone()

        # Get away stats
        cursor.execute('''
            SELECT
                COUNT(*) as away_games,
                AVG(away_score) as away_ppg,
                AVG(home_score) as away_papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as away_win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND week < ? AND away_team_id = ?
        ''', (team_id, season, current_week, team_id))
        away_row = cursor.fetchone()

        stats['home_games'] = home_row[0] if home_row else 0
        stats['home_ppg'] = home_row[1] or 0 if home_row else 0
        stats['home_papg'] = home_row[2] or 0 if home_row else 0
        stats['home_win_pct'] = home_row[3] or 0 if home_row else 0

        stats['away_games'] = away_row[0] if away_row else 0
        stats['away_ppg'] = away_row[1] or 0 if away_row else 0
        stats['away_papg'] = away_row[2] or 0 if away_row else 0
        stats['away_win_pct'] = away_row[3] or 0 if away_row else 0

        # Home/away PPG differential
        stats['home_away_ppg_diff'] = stats['home_ppg'] - stats['away_ppg']

        return stats

    def _empty_stats(self):
        """Return empty stats dict with CFB average defaults"""
        return {
            'games_played': 0, 'ppg': 28, 'papg': 28, 'win_pct': 0.5,
            'ypg': 400, 'turnover_pg': 1.5,
            'home_games': 0, 'home_ppg': 30, 'home_papg': 26, 'home_win_pct': 0.6,
            'away_games': 0, 'away_ppg': 26, 'away_papg': 30, 'away_win_pct': 0.4,
            'home_away_ppg_diff': 4.0,
            '_missing_stats': True
        }

    def _get_odds(self, conn, game_id):
        """Get odds for a game - returns keys matching training features"""
        cursor = conn.cursor()

        game_id = int(game_id)

        cursor.execute('''
            SELECT
                opening_spread,
                latest_spread,
                opening_total,
                latest_total,
                opening_moneyline_home,
                latest_moneyline_home,
                opening_moneyline_away,
                latest_moneyline_away,
                spread_movement,
                total_movement
            FROM odds_and_predictions WHERE game_id = ?
            ORDER BY odds_updated_at DESC LIMIT 1
        ''', (game_id,))

        row = cursor.fetchone()
        if not row:
            return {
                'opening_spread': 0, 'latest_spread': 0,
                'opening_total': 55, 'latest_total': 55,
                'opening_ml_home': -110, 'latest_ml_home': -110,
                'opening_ml_away': -110, 'latest_ml_away': -110,
                'spread_movement': 0, 'total_movement': 0,
                '_missing_odds': True
            }

        # Check if we have actual spread/total data
        has_spread = row[0] is not None or row[1] is not None
        has_total = row[2] is not None or row[3] is not None

        # Get spread values
        opening_spread = row[0] if row[0] is not None else row[1]
        latest_spread = row[1] if row[1] is not None else row[0]
        opening_spread = opening_spread or 0
        latest_spread = latest_spread or 0

        # Get total values
        opening_total = row[2] if row[2] is not None else row[3]
        latest_total = row[3] if row[3] is not None else row[2]
        opening_total = opening_total or 55
        latest_total = latest_total or 55

        # Calculate movement
        if row[8] is not None:
            spread_movement = row[8]
        elif row[0] is not None and row[1] is not None:
            spread_movement = latest_spread - opening_spread
        else:
            spread_movement = 0

        if row[9] is not None:
            total_movement = row[9]
        elif row[2] is not None and row[3] is not None:
            total_movement = latest_total - opening_total
        else:
            total_movement = 0

        return {
            'opening_spread': opening_spread,
            'latest_spread': latest_spread,
            'opening_total': opening_total,
            'latest_total': latest_total,
            'opening_ml_home': row[4] or -110,
            'latest_ml_home': row[5] or row[4] or -110,
            'opening_ml_away': row[6] or -110,
            'latest_ml_away': row[7] or row[6] or -110,
            'spread_movement': spread_movement,
            'total_movement': total_movement,
            '_missing_odds': not (has_spread and has_total)
        }

    def _get_drive_stats(self, conn, team_id, season, current_week):
        """Get team's drive statistics"""
        cursor = conn.cursor()

        team_id = int(team_id)
        season = int(season)
        current_week = int(current_week)

        # Query drive stats
        cursor.execute('''
            SELECT
                COUNT(*) as total_drives,
                AVG(CASE
                    WHEN d.result LIKE '%TD%' THEN 7
                    WHEN d.result LIKE '%FG%' THEN 3
                    WHEN d.result LIKE '%SAFETY%' THEN 2
                    ELSE 0
                END) as ppd,
                AVG(d.yards) as ypd,
                AVG(d.plays) as plays_per_drive,
                AVG(d.time_elapsed_seconds) as seconds_per_drive,
                AVG(d.is_score) as scoring_pct
            FROM drives d
            JOIN games g ON d.game_id = g.game_id
            WHERE d.team_id = ? AND g.season = ? AND g.week < ? AND g.completed = 1
        ''', (team_id, season, current_week))

        row = cursor.fetchone()

        if not row or row[0] == 0 or row[0] is None:
            return self._empty_drive_stats()

        return {
            'total_drives': row[0] or 0,
            'ppd': row[1] or 2.5,
            'ypd': row[2] or 30,
            'plays_per_drive': row[3] or 6.0,
            'seconds_per_drive': row[4] or 150,
            'scoring_pct': row[5] or 0.35,
            'redzone_pct': 0.55,  # Default
            'three_and_out_pct': 0.18,  # Default
            'explosive_drive_pct': 0.18,  # Default
            'def_ppd': 2.5,  # Default
            'def_ypd': 30,  # Default
            'def_scoring_pct': 0.35,
            'def_three_and_out_forced': 0.18,
        }

    def _empty_drive_stats(self):
        """Return empty drive stats with CFB average defaults"""
        return {
            'total_drives': 0,
            'ppd': 2.5, 'ypd': 30, 'plays_per_drive': 6.0, 'seconds_per_drive': 150,
            'scoring_pct': 0.35, 'redzone_pct': 0.55, 'three_and_out_pct': 0.18,
            'explosive_drive_pct': 0.18, 'def_ppd': 2.5, 'def_ypd': 30,
            'def_scoring_pct': 0.35, 'def_three_and_out_forced': 0.18
        }

    def _apply_spread_adjustments(
        self,
        model_spread: float,
        vegas_spread: float | None,
    ) -> tuple[float, list[str]]:
        """
        Apply post-prediction adjustments to the model spread.

        CFB-specific adjustments based on edge analysis backtest.
        """
        adjustments = []
        adjusted_spread = model_spread

        # BIG UNDERDOG ADJUSTMENT
        # CFB spreads are larger than NFL, use 14 pt threshold
        if vegas_spread is not None:
            if vegas_spread < -self.BIG_FAVORITE_THRESHOLD:
                # Home is big favorite, away is big underdog
                adjusted_spread += self.UNDERDOG_ADJUSTMENT
                adjustments.append(f"big_underdog_away:+{self.UNDERDOG_ADJUSTMENT}")
            elif vegas_spread > self.BIG_FAVORITE_THRESHOLD:
                # Away is big favorite, home is big underdog
                adjusted_spread -= self.UNDERDOG_ADJUSTMENT
                adjustments.append(f"big_underdog_home:-{self.UNDERDOG_ADJUSTMENT}")

        return adjusted_spread, adjustments

    def predict(self, games_df, mc_passes=100):
        """Generate predictions for games using MC Dropout for uncertainty estimation"""
        if self.model is None:
            print("No model loaded")
            return None

        predictions = []
        all_warnings = []

        for idx, game in games_df.iterrows():
            try:
                features = self.extract_features(game)

                # Collect any warnings
                game_warnings = features.pop('_warnings', [])
                all_warnings.extend(game_warnings)

                # Build feature vector in correct order
                feature_vector = []
                for col in self.feature_cols:
                    feature_vector.append(features.get(col, 0))

                feature_vector = np.array([feature_vector])
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
                feature_vector = self.scaler.transform(feature_vector)

                X = torch.FloatTensor(feature_vector).to(self.device)

                # MC Dropout: run multiple passes with dropout enabled
                self.model.eval()
                for module in self.model.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()

                mc_predictions = []
                with torch.no_grad():
                    for _ in range(mc_passes):
                        pred = self.model(X).cpu().numpy()[0]
                        mc_predictions.append(pred)

                # Reset all to eval mode
                self.model.eval()

                mc_predictions = np.array(mc_predictions)
                home_scores = np.maximum(0, mc_predictions[:, 0])
                away_scores = np.maximum(0, mc_predictions[:, 1])
                # VEGAS CONVENTION: spread = away - home
                spreads = away_scores - home_scores
                totals = home_scores + away_scores

                # Calculate means
                home_score = np.mean(home_scores)
                away_score = np.mean(away_scores)
                spread = np.mean(spreads)
                total = np.mean(totals)

                # Calculate MOE (standard deviations)
                home_moe = np.std(home_scores)
                away_moe = np.std(away_scores)
                spread_moe = np.std(spreads)
                total_moe = np.std(totals)

                # Home win probability from MC passes
                home_wins = np.sum(spreads < 0)
                home_win_prob = home_wins / mc_passes

                # Confidence based on spread MOE
                confidence = max(0.5, min(0.95, 0.95 - (spread_moe / 10) * 0.45))

                # Get vegas odds for comparison
                conn = sqlite3.connect(self.db_path)
                odds = self._get_odds(conn, game['game_id'])
                conn.close()

                # Apply post-prediction adjustments
                raw_spread = spread
                adjusted_spread, adjustment_notes = self._apply_spread_adjustments(
                    model_spread=spread,
                    vegas_spread=odds['latest_spread'],
                )
                spread = adjusted_spread

                # Validate spread convention
                validate_prediction_spread(
                    round(raw_spread, 1), round(home_score, 1), round(away_score, 1),
                    context=f"game_id={game['game_id']}"
                )

                predictions.append({
                    'game_id': game['game_id'],
                    'date': game['date'],
                    'week': game.get('week', 0),
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'pred_home_score': round(home_score, 1),
                    'pred_away_score': round(away_score, 1),
                    'pred_spread': round(spread, 1),
                    'pred_spread_base': round(raw_spread, 1),
                    'pred_total': round(total, 1),
                    'pred_home_MOE': round(home_moe, 2),
                    'pred_away_MOE': round(away_moe, 2),
                    'pred_spread_MOE': round(spread_moe, 2),
                    'pred_total_MOE': round(total_moe, 2),
                    'adjustment_notes': '; '.join(adjustment_notes) if adjustment_notes else '',
                    'vegas_spread': odds['latest_spread'],
                    'vegas_total': odds['latest_total'],
                    'confidence': round(confidence, 3),
                    'pred_home_win_prob': round(home_win_prob, 3),
                    'predicted_winner': get_predicted_winner(spread, game['home_team'], game['away_team'])
                })

            except Exception as e:
                print(f"Error predicting game {game['game_id']}: {e}")
                continue

        # Display warnings if any
        if all_warnings:
            print(f"\n{'='*60}")
            print(f"DATA WARNINGS ({len(all_warnings)} issues)")
            print('='*60)
            for warning in all_warnings:
                print(f"  * {warning}")
            print("Note: Predictions using default values may be less accurate")
            print('='*60)

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

    def save_predictions(self, predictions_df, output_path='cfb_predictions.csv'):
        """Save predictions to CSV"""
        predictions_df.to_csv(output_path, index=False)
        print(f"Saved {len(predictions_df)} predictions to {output_path}")


if __name__ == '__main__':
    import sys

    predictor = CFBPredictor()

    if predictor.model is None:
        print("\nNo model available. Train a model first.")
        sys.exit(1)

    # Get upcoming games
    week = int(sys.argv[1]) if len(sys.argv) > 1 else None

    if week:
        games = predictor.get_upcoming_games(week=week)
    else:
        current_week, season = predictor.get_current_week()
        print(f"Current CFB Week: {current_week}, Season: {season}")
        games = predictor.get_upcoming_games(week=current_week)

    if games.empty:
        print("No upcoming games found")
        sys.exit(0)

    print(f"\nFound {len(games)} upcoming games")

    # Generate predictions
    predictions = predictor.predict(games)

    if predictions is not None and not predictions.empty:
        print(f"\n{'='*80}")
        print("CFB PREDICTIONS")
        print('='*80)
        for _, pred in predictions.iterrows():
            print(f"\nWeek {pred['week']}: {pred['away_team']} @ {pred['home_team']}")
            print(f"  Predicted: {pred['pred_away_score']:.0f} - {pred['pred_home_score']:.0f}")
            print(f"  Spread: {pred['predicted_winner']} by {abs(pred['pred_spread']):.1f} (MOE: ±{pred['pred_spread_MOE']:.1f})")
            print(f"  Total: {pred['pred_total']:.1f} (MOE: ±{pred['pred_total_MOE']:.1f})")
            if pred['vegas_spread'] is not None:
                print(f"  Vegas: Spread {pred['vegas_spread']:+.1f}, Total {pred['vegas_total']:.1f}")
            print(f"  Home Win Prob: {pred['pred_home_win_prob']:.0%} | Confidence: {pred['confidence']:.1%}")

        # Save predictions
        predictor.save_predictions(predictions)
