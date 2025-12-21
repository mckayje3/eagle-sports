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
            print("Train a model first: py train_deep_eagle_nfl.py 2025 nfl_2025_deep_eagle_features.csv")
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
        """Extract features for a single game for prediction - matches training feature names"""
        conn = sqlite3.connect(self.db_path)

        features = {}
        warnings = []

        # Game context
        features['week_normalized'] = game_row.get('week', 10) / 18.0
        features['neutral_site'] = game_row.get('neutral_site', 0) or 0
        features['conference_game'] = game_row.get('conference_game', 0) or 0

        # Weather features (get from game if available, else defaults)
        features['temperature'] = game_row.get('temperature', 65) or 65
        features['wind_speed'] = game_row.get('wind_speed', 5) or 5
        features['is_dome'] = game_row.get('is_dome', 0) or 0

        # Get historical stats for both teams
        season = game_row.get('season', 2025)
        home_stats = self._get_team_stats(conn, game_row['home_team_id'], season)
        away_stats = self._get_team_stats(conn, game_row['away_team_id'], season)

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
        features['odds_opening_total'] = odds.get('opening_total', 45)
        features['odds_latest_total'] = odds.get('latest_total', 45)
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
        # Threshold features: significant movement
        # NFL uses 2.0 pts (~35% of avg spread)
        spread_significant = abs(spread_movement) >= 2.0
        total_significant = abs(total_movement) >= 2.0
        features['odds_spread_movement_significant'] = 1 if spread_significant else 0
        features['odds_total_movement_significant'] = 1 if total_significant else 0
        # Direction ONLY when significant (0 for small moves = model ignores)
        features['odds_spread_movement_sig_direction'] = spread_movement if spread_significant else 0
        features['odds_total_movement_sig_direction'] = total_movement if total_significant else 0

        # Store warnings in features for later retrieval
        features['_warnings'] = warnings

        # Drive features - use season averages as defaults since per-game drive data is complex
        # NFL average: ~12 drives per game, ~2.0 ppd, ~30 ypd
        home_drive = self._get_drive_stats(conn, game_row['home_team_id'], season)
        away_drive = self._get_drive_stats(conn, game_row['away_team_id'], season)

        for key, value in home_drive.items():
            features[f'home_drive_{key}'] = value
        for key, value in away_drive.items():
            features[f'away_drive_{key}'] = value

        # NFL-specific features: day of week, primetime, rest, recent form
        # These must match deep_eagle_feature_extractor.py lines 187-214
        game_date = game_row.get('date', '')
        features['day_of_week'] = self._get_day_of_week(game_date)
        features['is_primetime'] = 1 if features['day_of_week'] in [0, 3] else 0  # Thursday/Monday

        # Rest days for each team
        home_rest = self._get_rest_days(conn, game_row['home_team_id'], season, game_row.get('week', 10), game_date)
        away_rest = self._get_rest_days(conn, game_row['away_team_id'], season, game_row.get('week', 10), game_date)
        features['home_rest_days'] = home_rest['rest_days']
        features['away_rest_days'] = away_rest['rest_days']
        features['home_coming_off_bye'] = home_rest['coming_off_bye']
        features['away_coming_off_bye'] = away_rest['coming_off_bye']
        features['rest_advantage'] = home_rest['rest_days'] - away_rest['rest_days']

        # Recent form (last 4 games)
        home_form = self._get_recent_form(conn, game_row['home_team_id'], season, game_row.get('week', 10))
        away_form = self._get_recent_form(conn, game_row['away_team_id'], season, game_row.get('week', 10))
        features['home_recent_win_pct'] = home_form['win_pct']
        features['home_recent_ppg'] = home_form['ppg']
        features['home_recent_papg'] = home_form['papg']
        features['home_recent_margin'] = home_form['margin']
        features['away_recent_win_pct'] = away_form['win_pct']
        features['away_recent_ppg'] = away_form['ppg']
        features['away_recent_papg'] = away_form['papg']
        features['away_recent_margin'] = away_form['margin']
        features['recent_form_differential'] = home_form['margin'] - away_form['margin']

        # Calculate differentials matching training feature names
        features['ppg_differential'] = home_stats.get('ppg', 0) - away_stats.get('ppg', 0)
        features['papg_differential'] = home_stats.get('papg', 0) - away_stats.get('papg', 0)
        features['win_pct_differential'] = home_stats.get('win_pct', 0) - away_stats.get('win_pct', 0)
        features['ppd_differential'] = home_drive.get('ppd', 2.0) - away_drive.get('ppd', 2.0)
        features['scoring_pct_differential'] = home_drive.get('scoring_pct', 0.35) - away_drive.get('scoring_pct', 0.35)

        # Venue-based differentials
        features['venue_ppg_differential'] = home_stats.get('home_ppg', 0) - away_stats.get('away_ppg', 0)
        features['venue_win_pct_differential'] = home_stats.get('home_win_pct', 0) - away_stats.get('away_win_pct', 0)

        # Combined home advantage - MUST match training formula in deep_eagle_feature_extractor.py
        # Formula: average of both teams' home/away PPG differentials
        features['combined_home_advantage'] = (
            home_stats.get('home_away_ppg_diff', 0) + away_stats.get('home_away_ppg_diff', 0)
        ) / 2

        conn.close()
        return features

    def _get_team_stats(self, conn, team_id, season):
        """Get team's season statistics - feature names match training data"""
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

        # Get box score stats for ypg and turnover_pg
        cursor.execute('''
            SELECT
                AVG(ts.total_yards) as ypg,
                AVG(ts.turnovers) as turnover_pg
            FROM team_game_stats ts
            JOIN games g ON ts.game_id = g.game_id
            WHERE ts.team_id = ? AND g.season = ? AND g.completed = 1
        ''', (team_id, season))
        box_row = cursor.fetchone()

        stats['ypg'] = box_row[0] or 300 if box_row else 300  # NFL avg ~330 ypg
        stats['turnover_pg'] = box_row[1] or 1.0 if box_row else 1.0  # NFL avg ~1.0

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

        stats['home_games'] = home_row[0] if home_row else 0
        stats['home_ppg'] = home_row[1] or 0 if home_row else 0
        stats['home_papg'] = home_row[2] or 0 if home_row else 0
        stats['home_win_pct'] = home_row[3] or 0 if home_row else 0

        stats['away_games'] = away_row[0] if away_row else 0
        stats['away_ppg'] = away_row[1] or 0 if away_row else 0
        stats['away_papg'] = away_row[2] or 0 if away_row else 0
        stats['away_win_pct'] = away_row[3] or 0 if away_row else 0

        # Home/away PPG differential (for venue features)
        stats['home_away_ppg_diff'] = stats['home_ppg'] - stats['away_ppg']

        return stats

    def _empty_stats(self):
        """Return empty stats dict with NFL average defaults"""
        return {
            'games_played': 0, 'ppg': 22, 'papg': 22, 'win_pct': 0.5,
            'ypg': 330, 'turnover_pg': 1.0,
            'home_games': 0, 'home_ppg': 23, 'home_papg': 21, 'home_win_pct': 0.55,
            'away_games': 0, 'away_ppg': 21, 'away_papg': 23, 'away_win_pct': 0.45,
            'home_away_ppg_diff': 2.0,
            '_missing_stats': True
        }

    def _get_odds(self, conn, game_id):
        """Get odds for a game - returns keys matching training features"""
        cursor = conn.cursor()

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
            ORDER BY updated_at DESC LIMIT 1
        ''', (game_id,))

        row = cursor.fetchone()
        if not row:
            return {
                'opening_spread': 0, 'latest_spread': 0,
                'opening_total': 45, 'latest_total': 45,
                'opening_ml_home': -110, 'latest_ml_home': -110,
                'opening_ml_away': -110, 'latest_ml_away': -110,
                'spread_movement': 0, 'total_movement': 0,
                'spread_movement_abs': 0, 'total_movement_abs': 0,
                '_missing_odds': True
            }

        # Check if we have actual spread/total data
        has_spread = row[0] is not None or row[1] is not None
        has_total = row[2] is not None or row[3] is not None

        # Calculate movement from opening to latest
        opening_spread = row[0] or 0
        latest_spread = row[1] or row[0] or 0
        opening_total = row[2] or 45
        latest_total = row[3] or row[2] or 45

        spread_movement = row[8] if row[8] is not None else (latest_spread - opening_spread)
        total_movement = row[9] if row[9] is not None else (latest_total - opening_total)

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
            'spread_movement_abs': abs(spread_movement),
            'total_movement_abs': abs(total_movement),
            '_missing_odds': not (has_spread and has_total)
        }

    def _get_drive_stats(self, conn, team_id, season):
        """Get team's drive statistics - uses scoring data to estimate drive efficiency"""
        cursor = conn.cursor()

        # Estimate drive stats from game data
        # NFL avg: ~12 drives/game, ~2.0 PPD, ~30 YPD, ~35% scoring, ~20% 3-and-out
        cursor.execute('''
            SELECT
                COUNT(*) as games,
                AVG(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as ppg,
                AVG(ts.total_yards) as ypg
            FROM games g
            LEFT JOIN team_game_stats ts ON g.game_id = ts.game_id AND ts.team_id = ?
            WHERE g.season = ? AND g.completed = 1
                AND (g.home_team_id = ? OR g.away_team_id = ?)
        ''', (team_id, team_id, season, team_id, team_id))

        row = cursor.fetchone()
        games = row[0] if row else 0
        ppg = row[1] or 22 if row else 22
        ypg = row[2] or 330 if row else 330

        # Estimate drive stats from PPG and YPG
        # Assume ~12 drives per game
        drives_per_game = 12
        ppd = ppg / drives_per_game
        ypd = ypg / drives_per_game
        scoring_pct = ppd / 7 * 0.5  # Rough estimate based on PPD

        return {
            'total_drives': drives_per_game * games if games else 0,
            'ppd': ppd,
            'ypd': ypd,
            'plays_per_drive': 6.0,  # NFL average
            'seconds_per_drive': 150,  # ~2.5 minutes
            'scoring_pct': min(0.5, scoring_pct),
            'redzone_pct': 0.55,  # NFL average
            'three_and_out_pct': 0.20,  # NFL average
            'explosive_drive_pct': 0.15,  # NFL average
            'def_ppd': 2.0,  # Default defensive PPD
            'def_ypd': 28,  # Default defensive YPD
            'def_scoring_pct': 0.35,
            'def_three_and_out_forced': 0.22,
        }

    def _get_day_of_week(self, date_str):
        """
        Convert game date to day-of-week code for NFL
        Must match deep_eagle_feature_extractor.py

        Returns:
            int: 0=Thursday, 1=Saturday, 2=Sunday, 3=Monday, 4=Other
        """
        try:
            if 'T' in str(date_str):
                dt = datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
            else:
                dt = datetime.strptime(str(date_str)[:10], '%Y-%m-%d')

            weekday = dt.weekday()  # 0=Monday, 6=Sunday
            day_map = {
                3: 0,  # Thursday
                5: 1,  # Saturday
                6: 2,  # Sunday
                0: 3,  # Monday
            }
            return day_map.get(weekday, 4)  # 4 for other days
        except:
            return 2  # Default to Sunday

    def _get_rest_days(self, conn, team_id, season, current_week, current_date):
        """
        Get rest days since last game for a team
        Must match deep_eagle_feature_extractor.py

        Returns:
            dict with 'rest_days' and 'coming_off_bye'
        """
        cursor = conn.cursor()

        cursor.execute('''
            SELECT date, week
            FROM games
            WHERE season = ? AND week < ?
                AND (home_team_id = ? OR away_team_id = ?)
                AND completed = 1
            ORDER BY week DESC, date DESC
            LIMIT 1
        ''', (season, current_week, team_id, team_id))

        row = cursor.fetchone()

        if not row:
            # First game of season or no prior games
            return {'rest_days': 10, 'coming_off_bye': 0}

        last_date = row[0]
        last_week = row[1]

        try:
            if 'T' in str(current_date):
                curr_dt = datetime.fromisoformat(str(current_date).replace('Z', '+00:00'))
            else:
                curr_dt = datetime.strptime(str(current_date)[:10], '%Y-%m-%d')

            if 'T' in str(last_date):
                last_dt = datetime.fromisoformat(str(last_date).replace('Z', '+00:00'))
            else:
                last_dt = datetime.strptime(str(last_date)[:10], '%Y-%m-%d')

            rest_days = (curr_dt - last_dt).days
        except:
            rest_days = 7  # Default to 1 week

        # Coming off bye: skipped a week (gap > 1 week between games)
        coming_off_bye = 1 if (current_week - last_week) > 1 else 0

        return {
            'rest_days': min(rest_days, 14),  # Cap at 14 days
            'coming_off_bye': coming_off_bye
        }

    def _get_recent_form(self, conn, team_id, season, current_week, num_games=4):
        """
        Get team's recent form (last N games)
        Must match deep_eagle_feature_extractor.py

        Returns:
            dict with win_pct, ppg, papg, margin for recent games
        """
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                CASE WHEN home_team_id = ? THEN home_score ELSE away_score END as points_for,
                CASE WHEN home_team_id = ? THEN away_score ELSE home_score END as points_against,
                CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END as won
            FROM games
            WHERE season = ? AND week < ?
                AND (home_team_id = ? OR away_team_id = ?)
                AND completed = 1
            ORDER BY week DESC
            LIMIT ?
        ''', (team_id, team_id, team_id, season, current_week, team_id, team_id, num_games))

        rows = cursor.fetchall()

        if not rows:
            return {'win_pct': 0.5, 'ppg': 21, 'papg': 21, 'margin': 0}

        wins = sum(1 for r in rows if r[2] == 1)
        total_scored = sum(r[0] or 0 for r in rows)
        total_allowed = sum(r[1] or 0 for r in rows)
        games = len(rows)

        ppg = total_scored / games if games > 0 else 21
        papg = total_allowed / games if games > 0 else 21
        win_pct = wins / games if games > 0 else 0.5

        return {
            'win_pct': win_pct,
            'ppg': ppg,
            'papg': papg,
            'margin': ppg - papg
        }

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
                # Keep model in eval mode (for BatchNorm) but enable Dropout layers
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
                spreads = home_scores - away_scores
                totals = home_scores + away_scores

                # Calculate means
                home_score = np.mean(home_scores)
                away_score = np.mean(away_scores)
                spread = np.mean(spreads)
                total = np.mean(totals)

                # Calculate MOE (standard deviations) directly from passes
                home_moe = np.std(home_scores)
                away_moe = np.std(away_scores)
                spread_moe = np.std(spreads)
                total_moe = np.std(totals)

                # Calculate home win probability from MC passes
                home_wins = np.sum(spreads > 0)  # Home wins when spread is positive (home - away > 0)
                home_win_prob = home_wins / mc_passes

                # Confidence based on spread MOE (lower MOE = higher confidence)
                # Scale: MOE of 0 -> 95% confidence, MOE of 10+ -> 50% confidence
                confidence = max(0.5, min(0.95, 0.95 - (spread_moe / 10) * 0.45))

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
                    'pred_home_MOE': round(home_moe, 2),
                    'pred_away_MOE': round(away_moe, 2),
                    'pred_spread_MOE': round(spread_moe, 2),
                    'pred_total_MOE': round(total_moe, 2),
                    'vegas_spread': odds['latest_spread'],
                    'vegas_total': odds['latest_total'],
                    'confidence': round(confidence, 3),
                    'pred_home_win_prob': round(home_win_prob, 3),
                    'predicted_winner': game['home_team'] if spread > 0 else game['away_team']
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
            print(f"  Spread: {pred['predicted_winner']} by {abs(pred['pred_spread']):.1f} (MOE: ±{pred['pred_spread_MOE']:.1f})")
            print(f"  Total: {pred['pred_total']:.1f} (MOE: ±{pred['pred_total_MOE']:.1f})")
            if pred['vegas_spread'] is not None:
                print(f"  Vegas: Spread {pred['vegas_spread']:+.1f}, Total {pred['vegas_total']:.1f}")
            print(f"  Home Win Prob: {pred['pred_home_win_prob']:.0%} | Confidence: {pred['confidence']:.1%}")

        # Save predictions
        predictor.save_predictions(predictions)
