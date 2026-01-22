"""
NBA Predictor - Supports Ridge V2, Enhanced Ridge, and Deep Eagle models

PREFERRED MODEL: Ridge V2 (pure model - no Vegas blend, SRS, reduced HCA)
FALLBACK 1: Enhanced Ridge (dynamic HCA, injuries, momentum)
FALLBACK 2: Deep Eagle neural net (legacy, known to overfit)

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
import os
from datetime import datetime, timedelta
from spread_utils import validate_prediction_spread, get_predicted_winner


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
    """Predict NBA game outcomes using Ridge V2 (preferred), Enhanced Ridge, or Deep Eagle.

    Model priority:
    1. Ridge V2 (models/nba_ridge_v2.pkl) - Pure model, SRS, reduced HCA, no Vegas blend
    2. Enhanced Ridge (models/nba_ridge_enhanced.pkl) - Dynamic HCA, injuries, momentum
    3. Deep Eagle neural net - Legacy model, known to overfit

    Ridge V2 is a pure model without Vegas blending - designed as baseline for
    Deep Eagle edge classifier. Walk-forward: 56.2% overall, 61.2% at 7+ pt edges.

    CONFIDENCE SCORING (based on pattern analysis):
    - 2+ stars: 64.3% ATS (2025-2026 combined)
    - Filters out road favorite picks (35% ATS historically)
    - Prefers home picks, especially home favorites
    - Close games (Vegas < 4 pts) are best
    """

    # Legacy adjustment constants (used only with Deep Eagle model)
    BIG_FAVORITE_THRESHOLD = 10.0
    UNDERDOG_ADJUSTMENT = 1.5
    STRUGGLING_HOME_MARGIN = -3.0
    STRUGGLING_HOME_ADJUSTMENT = 1.5
    MIDDLE_SPREAD_EDGE_MIN = 2.0
    MIDDLE_SPREAD_EDGE_MAX = 6.0
    MIDDLE_TOTAL_EDGE_MIN = 4.0
    MIDDLE_TOTAL_EDGE_MAX = 6.0

    # Total fade constants (from walk-forward analysis)
    # When model predicts UNDER by 7+ pts, fade to OVER (59% win rate)
    FADE_UNDER_THRESHOLD = 7.0
    FADE_UNDER_ADJUSTMENT = 14.0  # Flip by adding 2x the edge

    @staticmethod
    def calculate_spread_confidence(
        edge: float,
        vegas_spread: float | None,
        min_edge: float = 5.0,
    ) -> tuple[int, bool]:
        """
        Calculate confidence stars for a spread pick based on pattern analysis.

        Rules (based on 2025-2026 backtest, 64.3% ATS at 2+ stars):
        1. Must have min_edge (default 5 pts) to qualify
        2. FADE road favorite picks (35% ATS -> 65% when faded)
        3. +1 for home picks
        4. +1 for home favorites (stacks with #3)
        5. +1 for close games (Vegas < 4 pts)
        6. -1 for blowout spreads (Vegas 10+ pts)
        7. +1 for big edges (7+ pts)

        Args:
            edge: Model spread - Vegas spread (negative = pick home)
            vegas_spread: Vegas spread (negative = home favored)
            min_edge: Minimum edge to qualify (default 5.0)

        Returns:
            Tuple of (confidence_stars, should_fade).
            should_fade=True means bet OPPOSITE of model's pick (bet home dog).
        """
        if vegas_spread is None:
            return 1, False  # No Vegas line to compare

        abs_edge = abs(edge)

        # Must have minimum edge
        if abs_edge < min_edge:
            return 0, False

        # Classify the pick
        pick_home = edge < 0
        vegas_home_fav = vegas_spread < 0

        # FADE road favorite picks (35% ATS -> 65% when faded)
        # Road fav = picking away team when away is Vegas favorite
        if not pick_home and not vegas_home_fav:  # Pick away, away is favored
            # Fade this pick - bet the home dog instead (64.7% ATS)
            return 2, True  # 2 stars for fade play

        # Start building confidence for normal picks
        score = 0

        # +1 for home picks (62.5% vs 55.4%)
        if pick_home:
            score += 1

        # +1 for home favorites (67.2% ATS)
        if pick_home and vegas_home_fav:
            score += 1

        # +1 for close games (Vegas spread < 4 pts) - 64.1% ATS
        if abs(vegas_spread) < 4:
            score += 1

        # -1 for blowout spreads (Vegas 10+ pts) - 53.8% ATS
        if abs(vegas_spread) >= 10:
            score -= 1

        # +1 for big edges (7+ pts)
        if abs_edge >= 7:
            score += 1

        # Minimum 1 star if we passed the edge threshold
        return max(score, 1), False

    @staticmethod
    def calculate_total_confidence(
        total_edge: float,
        vegas_total: float | None,
    ) -> tuple[int, bool]:
        """
        Calculate confidence stars for a total pick based on walk-forward analysis.

        Rules (based on 2025-2026 backtest):
        1. Most totals are ~50% - NOT profitable
        2. EXCEPTION: Fade UNDER 7+ = 59% win rate (79-55 record)
           - When model predicts UNDER by 7+ pts, bet OVER instead

        Args:
            total_edge: Model total - Vegas total (negative = pick UNDER)
            vegas_total: Vegas total line

        Returns:
            Tuple of (confidence_stars, should_fade).
            should_fade=True means bet OVER when model says UNDER.
        """
        if vegas_total is None or vegas_total <= 0:
            return 1, False  # No Vegas line to compare

        # Check for fade UNDER 7+ scenario
        # total_edge < -7 means model predicts UNDER by 7+ pts
        if total_edge <= -7.0:
            # Fade this to OVER (59% win rate in backtest)
            return 2, True  # 2 stars for fade play

        # All other totals are ~50% - not profitable
        return 1, False

    def __init__(self, model_path='models/deep_eagle_nba_2025.pt',
                 scaler_path='models/deep_eagle_nba_2025_scaler.pkl',
                 db_path='nba_games.db',
                 use_enhanced=True):
        self.db_path = db_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.ridge_v2_model = None  # Ridge V2 model (preferred)
        self.enhanced_model = None  # Enhanced Ridge model (fallback)
        self.scaler = None
        self.feature_cols = None
        self.using_ridge_v2 = False  # Track if using V2
        self.using_enhanced = False  # Track if using enhanced
        self._load_model(use_enhanced=use_enhanced)

    def _load_model(self, use_enhanced=True):
        """Load trained model and scaler. Prefers Ridge V2, then Enhanced Ridge, then Deep Eagle."""

        # Try Ridge V2 model first (preferred - pure model, no Vegas blend)
        if use_enhanced:
            try:
                from nba_ridge_v2 import NBARidgeV2
                v2_path = 'models/nba_ridge_v2.pkl'

                if os.path.exists(v2_path):
                    self.ridge_v2_model = NBARidgeV2.load()
                    self.using_ridge_v2 = True
                    self.using_enhanced = True  # For compatibility
                    print(f"Loaded NBA Ridge V2 model (pure - no Vegas blend)")
                    print(f"  Features: SRS, reduced HCA (1.5), road fav penalty")
                    print(f"  Walk-forward: 56.2% overall, 61.2% at 7+ pt edges")
                    return
                else:
                    print(f"Ridge V2 not found at {v2_path}, trying Enhanced Ridge...")

            except ImportError as e:
                print(f"Could not import NBARidgeV2: {e}")
            except Exception as e:
                print(f"Error loading Ridge V2: {e}")

        # Try Enhanced Ridge model (fallback 1)
        if use_enhanced:
            try:
                from nba_enhanced_ridge import NBAEnhancedModel
                enhanced_path = 'models/nba_ridge_enhanced.pkl'

                if os.path.exists(enhanced_path):
                    self.enhanced_model = NBAEnhancedModel.load()
                    self.using_enhanced = True
                    print(f"Loaded NBA Enhanced Ridge model")
                    print(f"  Features: dynamic HCA, injuries, momentum, streaks")
                    print(f"  Walk-forward: 52.3% overall, 59.4% at 2+ pt edges")
                    return
                else:
                    print(f"Enhanced model not found at {enhanced_path}")
                    print(f"  Train with: python nba_enhanced_ridge.py")

            except ImportError as e:
                print(f"Could not import NBAEnhancedModel: {e}")
            except Exception as e:
                print(f"Error loading Enhanced model: {e}")

        # Fall back to Deep Eagle neural net
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

                    self.using_enhanced = False
                    arch_type = "old" if use_old_names else "new"
                    print(f"Loaded NBA Deep Eagle model from {model_path}")
                    print(f"  Features: {input_dim}, Hidden: {hidden_dims}, Architecture: {arch_type}")
                    print(f"  WARNING: Deep Eagle model known to overfit.")
                    return

                except Exception as e:
                    print(f"Error loading {model_path}: {e}")
                    continue

        print("No NBA model found. Train: python nba_enhanced_ridge.py")
        self.model = None
        self.enhanced_model = None

    def get_upcoming_games(self, days=7):
        """Get upcoming NBA games from database"""
        from datetime import datetime, timedelta
        import pytz

        conn = sqlite3.connect(self.db_path)

        # Use Eastern time for date filtering since game_date_eastern is in ET
        eastern = pytz.timezone('US/Eastern')
        now_eastern = datetime.now(eastern)
        today = now_eastern.strftime('%Y-%m-%d')
        end_date = (now_eastern + timedelta(days=days)).strftime('%Y-%m-%d')

        # Use game_date_eastern for accurate date filtering
        # This ensures evening games (stored as next day UTC) are correctly included
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
                AND g.game_date_eastern >= ?
                AND g.game_date_eastern <= ?
            ORDER BY g.game_date_eastern, g.date
        '''

        games = pd.read_sql_query(query, conn, params=(today, end_date))
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

        # Convert numpy types to native Python (SQLite doesn't handle numpy.int64)
        team_id = int(team_id)
        season = int(season)

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

        # Convert numpy types to native Python (SQLite doesn't handle numpy.int64)
        game_id = int(game_id)

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

        # Get spread values - if opening is missing, use latest as opening (no movement)
        opening_spread = row[0] if row[0] is not None else row[1]
        latest_spread = row[1] if row[1] is not None else row[0]
        opening_spread = opening_spread or 0
        latest_spread = latest_spread or 0

        # Get total values - if opening is missing, use latest as opening (no movement)
        opening_total = row[2] if row[2] is not None else row[3]
        latest_total = row[3] if row[3] is not None else row[2]
        opening_total = opening_total or default_total
        latest_total = latest_total or default_total

        # Calculate movement - only if we have both opening and latest values
        if row[4] is not None:
            spread_movement = row[4]
        elif row[0] is not None and row[1] is not None:
            spread_movement = latest_spread - opening_spread
        else:
            spread_movement = 0  # No movement if we don't have opening

        if row[5] is not None:
            total_movement = row[5]
        elif row[2] is not None and row[3] is not None:
            total_movement = latest_total - opening_total
        else:
            total_movement = 0  # No movement if we don't have opening

        # Threshold features: significant movement
        # NBA uses 2.0 pts (~29% of avg spread)
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

    def _get_recent_margin(self, conn, team_id: int, current_date: str, n_games: int = 5) -> float:
        """Get team's average margin over last N games"""
        cursor = conn.cursor()
        cursor.execute('''
            SELECT AVG(
                CASE WHEN home_team_id = ? THEN home_score - away_score
                     ELSE away_score - home_score END
            ) as avg_margin
            FROM (
                SELECT home_team_id, home_score, away_score
                FROM games
                WHERE completed = 1
                    AND (home_team_id = ? OR away_team_id = ?)
                    AND date < ?
                ORDER BY date DESC
                LIMIT ?
            )
        ''', (team_id, team_id, team_id, current_date, n_games))
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else 0.0

    def _apply_spread_adjustments(
        self,
        model_spread: float,
        vegas_spread: float | None,
        home_team_id: int,
        away_team_id: int,
        game_date: str,
        conn: sqlite3.Connection
    ) -> tuple[float, list[str]]:
        """
        Apply post-prediction adjustments to the model spread.

        Adjustments applied:
        1. Big underdog: +1.5 pts toward teams that are 10+ point underdogs
        2. Struggling home: +1.5 pts toward road favorite when home team is struggling

        Returns:
            (adjusted_spread, list_of_adjustments_applied)
        """
        adjustments = []
        adjusted_spread = model_spread

        # 1. BIG UNDERDOG ADJUSTMENT
        # If Vegas has a team as 10+ point underdog, add points toward the underdog
        # vegas_spread > 10 means away is big favorite (home is big dog)
        # vegas_spread < -10 means home is big favorite (away is big dog)
        if vegas_spread is not None:
            if vegas_spread < -self.BIG_FAVORITE_THRESHOLD:
                # Home is big favorite, away is big underdog - add toward away
                adjusted_spread += self.UNDERDOG_ADJUSTMENT
                adjustments.append(f"big_underdog_away:+{self.UNDERDOG_ADJUSTMENT}")
            elif vegas_spread > self.BIG_FAVORITE_THRESHOLD:
                # Away is big favorite, home is big underdog - add toward home
                adjusted_spread -= self.UNDERDOG_ADJUSTMENT
                adjustments.append(f"big_underdog_home:-{self.UNDERDOG_ADJUSTMENT}")

        # 2. ROAD FAVORITE VS STRUGGLING HOME TEAM ADJUSTMENT
        # If road team is favored (vegas_spread > 0) and home team is struggling,
        # adjust toward the road team because model tends to overvalue struggling home teams
        if vegas_spread is not None and vegas_spread > 0:  # Road team is favored
            home_recent_margin = self._get_recent_margin(conn, home_team_id, game_date, n_games=5)
            if home_recent_margin < self.STRUGGLING_HOME_MARGIN:
                adjusted_spread += self.STRUGGLING_HOME_ADJUSTMENT
                adjustments.append(f"struggling_home_vs_road_fav:+{self.STRUGGLING_HOME_ADJUSTMENT}")

        # 3. FADE MIDDLE-EDGE FAVORITES
        # Backtest shows model is 35% ATS when it has 2-6pt edge toward the Vegas favorite
        # When this happens, flip the edge to bet the underdog instead (65% ATS when faded)
        if vegas_spread is not None:
            spread_edge = adjusted_spread - vegas_spread
            abs_edge = abs(spread_edge)

            if self.MIDDLE_SPREAD_EDGE_MIN <= abs_edge < self.MIDDLE_SPREAD_EDGE_MAX:
                # Determine if we're betting the favorite
                # vegas_spread < 0 means home is favorite
                # spread_edge < 0 means betting home
                betting_home = spread_edge < 0
                home_is_favorite = vegas_spread < 0
                betting_favorite = (betting_home and home_is_favorite) or (not betting_home and not home_is_favorite)

                if betting_favorite:
                    # Flip the edge by subtracting 2 * edge from adjusted_spread
                    fade_adjustment = -2 * spread_edge
                    adjusted_spread = adjusted_spread + fade_adjustment
                    adjustments.append(f"fade_middle_fav:{fade_adjustment:+.1f}")

        return adjusted_spread, adjustments

    def _apply_total_adjustments(
        self,
        pred_total: float,
        vegas_total: float | None
    ) -> tuple[float, list[str], bool]:
        """
        Apply post-prediction adjustments to the predicted total.

        Adjustments applied:
        1. Fade UNDER 7+: When model predicts UNDER by 7+ pts, flip to OVER (59% win rate)
        2. Fade middle-edge: When model has 4-6pt edge, flip direction

        Returns:
            (adjusted_total, list_of_adjustments_applied, should_fade_total)
        """
        adjustments = []
        adjusted_total = pred_total
        should_fade_total = False

        if vegas_total is not None and vegas_total > 0:
            total_edge = pred_total - vegas_total
            abs_edge = abs(total_edge)

            # FADE UNDER 7+ (highest priority - 59% win rate)
            # When model predicts UNDER by 7+ pts, flip to OVER
            if total_edge <= -self.FADE_UNDER_THRESHOLD:
                # Flip to OVER by adding 2x the edge magnitude
                fade_adjustment = self.FADE_UNDER_ADJUSTMENT
                adjusted_total = pred_total + fade_adjustment
                adjustments.append(f"fade_under_7:{fade_adjustment:+.1f}")
                should_fade_total = True

            # FADE MIDDLE-EDGE TOTALS (only if not already faded)
            # Backtest: 4-6pt OVER edges hit 34% (fade to 66%), UNDER edges hit 44% (fade to 56%)
            elif self.MIDDLE_TOTAL_EDGE_MIN <= abs_edge < self.MIDDLE_TOTAL_EDGE_MAX:
                # Flip the edge by subtracting 2 * edge
                fade_adjustment = -2 * total_edge
                adjusted_total = pred_total + fade_adjustment
                if total_edge > 0:
                    adjustments.append(f"fade_middle_over:{fade_adjustment:+.1f}")
                else:
                    adjustments.append(f"fade_middle_under:{fade_adjustment:+.1f}")

        return adjusted_total, adjustments, should_fade_total

    def predict(self, games_df):
        """Generate predictions for games using Ridge V2, Enhanced Ridge, or Deep Eagle model."""
        if self.ridge_v2_model is None and self.enhanced_model is None and self.model is None:
            print("No model loaded")
            return None

        predictions = []
        conn = sqlite3.connect(self.db_path)  # Open connection once for all games

        try:
            for idx, game in games_df.iterrows():
                try:
                    # Use Ridge V2 if available (preferred - pure model)
                    if self.using_ridge_v2 and self.ridge_v2_model is not None:
                        season = int(game.get('season', 2026))
                        home_id = int(game['home_team_id'])
                        away_id = int(game['away_team_id'])
                        game_date = game['date']
                        game_id = int(game['game_id'])

                        # Get Vegas odds for comparison (not blending)
                        odds = self._get_odds(conn, game_id)
                        vegas_spread = odds.get('latest_spread')
                        vegas_total = odds.get('latest_total', 220)

                        # Use Ridge V2's predict method (returns dict)
                        result = self.ridge_v2_model.predict(
                            home_id=home_id,
                            away_id=away_id,
                            season=season,
                            game_date=game_date,
                            game_id=game_id,
                            vegas_spread=vegas_spread,
                            vegas_total=vegas_total
                        )

                        # Extract values from result dict
                        raw_spread = result['predicted_spread']
                        raw_total = result['predicted_total']
                        home_score = result['home_score']
                        away_score = result['away_score']
                        total = raw_total

                    # Use Enhanced Ridge if V2 not available
                    elif self.using_enhanced and self.enhanced_model is not None:
                        season = int(game.get('season', 2026))
                        home_id = int(game['home_team_id'])
                        away_id = int(game['away_team_id'])
                        game_date = game['date']
                        game_id = int(game['game_id'])

                        # Get Vegas odds for the prediction
                        odds = self._get_odds(conn, game_id)
                        vegas_spread = odds.get('latest_spread')
                        vegas_total = odds.get('latest_total', 220)

                        # Use enhanced model's predict method (returns dict)
                        result = self.enhanced_model.predict(
                            home_id=home_id,
                            away_id=away_id,
                            season=season,
                            game_date=game_date,
                            game_id=game_id,
                            vegas_spread=vegas_spread,
                            vegas_total=vegas_total
                        )

                        # Extract values from result dict
                        raw_spread = result['predicted_spread']
                        raw_total = result['predicted_total']
                        home_score = result['home_score']
                        away_score = result['away_score']
                        total = raw_total

                    else:
                        # Fall back to Deep Eagle neural net
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
                        raw_spread = away_score - home_score
                        total = home_score + away_score

                    # Get vegas odds for comparison (may already have from enhanced model)
                    if not self.using_enhanced:
                        odds = self._get_odds(conn, int(game['game_id']))

                    # For Ridge V2 / Enhanced model: Apply total adjustments (fade UNDER 7+)
                    # For Deep Eagle: Apply legacy post-prediction adjustments for both spread and total
                    should_fade_total = False
                    if self.using_ridge_v2:
                        # Ridge V2: No spread adjustments (built in), but apply total fade
                        spread = raw_spread
                        raw_total = total
                        adjusted_total, total_adj_notes, should_fade_total = self._apply_total_adjustments(
                            pred_total=total,
                            vegas_total=odds['latest_total']
                        )
                        total = adjusted_total
                        adjustment_notes = ['ridge_v2_pure'] + total_adj_notes
                    elif self.using_enhanced:
                        # Enhanced model: No spread adjustments, but apply total fade
                        spread = raw_spread
                        raw_total = total
                        adjusted_total, total_adj_notes, should_fade_total = self._apply_total_adjustments(
                            pred_total=total,
                            vegas_total=odds['latest_total']
                        )
                        total = adjusted_total
                        adjustment_notes = ['enhanced_model'] + total_adj_notes
                    else:
                        # Apply post-prediction spread adjustments (Deep Eagle only)
                        adjusted_spread, spread_adj_notes = self._apply_spread_adjustments(
                            model_spread=raw_spread,
                            vegas_spread=odds['latest_spread'],
                            home_team_id=int(game['home_team_id']),
                            away_team_id=int(game['away_team_id']),
                            game_date=game['date'],
                            conn=conn
                        )

                        # Apply post-prediction total adjustments
                        raw_total = total
                        adjusted_total, total_adj_notes, should_fade_total = self._apply_total_adjustments(
                            pred_total=total,
                            vegas_total=odds['latest_total']
                        )

                        adjustment_notes = spread_adj_notes + total_adj_notes
                        spread = adjusted_spread
                        total = adjusted_total

                    # Calculate spread edge and confidence stars
                    spread_edge = spread - odds['latest_spread'] if odds['latest_spread'] is not None else 0
                    spread_stars, should_fade = self.calculate_spread_confidence(spread_edge, odds['latest_spread'])

                    # Calculate total edge and confidence stars
                    # Use raw_total for edge calculation (before adjustments)
                    total_edge = raw_total - odds['latest_total'] if odds['latest_total'] is not None else 0
                    total_stars, _ = self.calculate_total_confidence(total_edge, odds['latest_total'])
                    # Override total_stars if we're fading (already calculated in _apply_total_adjustments)
                    if should_fade_total:
                        total_stars = 2  # Fade plays are 2 stars

                    # Determine total pick (accounting for fade)
                    if should_fade_total:
                        total_pick = 'OVER'  # Fade UNDER = bet OVER
                        total_fade_note = 'FADE_UNDER_7'
                    else:
                        total_pick = 'OVER' if total_edge > 0 else 'UNDER'
                        total_fade_note = ''

                    # Determine the actual spread pick (accounting for fade)
                    if should_fade:
                        # Fade = bet opposite of model. Model picked away (road fav), so bet home
                        actual_pick_team = game['home_team']
                        actual_pick_spread = odds['latest_spread']  # Take the home dog spread
                        fade_note = 'FADE_ROAD_FAV'
                    else:
                        actual_pick_team = get_predicted_winner(spread, game['home_team'], game['away_team'])
                        actual_pick_spread = spread
                        fade_note = ''

                    # Legacy confidence score (for backwards compatibility)
                    score_diff = abs(spread)
                    confidence = min(0.95, 0.5 + score_diff / 25)

                    # Validate spread convention before saving (use raw spread, not adjusted)
                    validate_prediction_spread(
                        round(raw_spread, 1), round(home_score, 1), round(away_score, 1),
                        context=f"game_id={game['game_id']}"
                    )

                    predictions.append({
                        'game_id': game['game_id'],
                        'date': game['date'],
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'pred_home_score': round(home_score, 1),
                        'pred_away_score': round(away_score, 1),
                        'pred_spread': round(spread, 1),
                        'pred_spread_base': round(raw_spread, 1),  # Raw model spread before adjustments
                        'pred_total': round(total, 1),
                        'pred_total_base': round(raw_total, 1),  # Raw model total before adjustments
                        'adjustment_notes': '; '.join(adjustment_notes) if adjustment_notes else '',
                        'vegas_spread': odds['latest_spread'],
                        'vegas_total': odds['latest_total'],
                        # Spread pick fields
                        'spread_edge': round(spread_edge, 1),  # Edge vs Vegas
                        'spread_stars': spread_stars,  # Confidence stars (0-4)
                        'should_fade': should_fade,  # True = bet opposite of model
                        'fade_note': fade_note,  # Explanation if fading spread
                        'predicted_winner': actual_pick_team,  # Actual spread pick (accounts for fade)
                        # Total pick fields
                        'total_edge': round(total_edge, 1),  # Edge vs Vegas (before adjustments)
                        'total_stars': total_stars,  # Confidence stars for total (1-2)
                        'total_pick': total_pick,  # OVER or UNDER (accounts for fade)
                        'should_fade_total': should_fade_total,  # True = fade UNDER to OVER
                        'total_fade_note': total_fade_note,  # Explanation if fading total
                        # Legacy field
                        'confidence': round(confidence, 3),
                    })

                except Exception as e:
                    print(f"Error predicting game {game['game_id']}: {e}")
                    continue
        finally:
            conn.close()  # Always close connection

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

    if predictor.ridge_v2_model is None and predictor.enhanced_model is None and predictor.model is None:
        print("\nNo model available. Train with: python nba_ridge_v2.py")
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
        print("Spreads: 2+ stars = 64% ATS | Totals: Fade UNDER 7+ = 59% O/U")
        print("FADE plays = bet opposite of model")
        print('='*80)
        for _, pred in predictions.iterrows():
            spread_stars = '*' * pred.get('spread_stars', 0) if pred.get('spread_stars', 0) > 0 else '(skip)'
            total_stars = '*' * pred.get('total_stars', 1)
            spread_edge = pred.get('spread_edge', 0)
            total_edge = pred.get('total_edge', 0)
            should_fade = pred.get('should_fade', False)
            should_fade_total = pred.get('should_fade_total', False)

            print(f"\n{pred['date']}: {pred['away_team']} @ {pred['home_team']}")
            print(f"  Predicted: {pred['pred_away_score']:.0f} - {pred['pred_home_score']:.0f}")

            # Spread pick
            if should_fade:
                print(f"  ** SPREAD FADE ** Bet: {pred['home_team']} +{pred['vegas_spread']:.1f} (home dog)")
                print(f"  Model picked road fav but we FADE (65% ATS when fading)")
            else:
                print(f"  Spread: {pred['predicted_winner']} by {abs(pred['pred_spread']):.1f}")
            print(f"  Vegas: {pred['vegas_spread']}, Edge: {spread_edge:+.1f}, Stars: {spread_stars}")

            # Total pick
            if should_fade_total:
                print(f"  ** TOTAL FADE ** Bet OVER {pred['vegas_total']:.1f} (model said UNDER by {abs(total_edge):.1f})")
                print(f"  Fade UNDER 7+ = 59% win rate")
            else:
                total_pick = pred.get('total_pick', 'OVER' if total_edge > 0 else 'UNDER')
                print(f"  Total: {pred['pred_total']:.1f} ({total_pick})")
            print(f"  Vegas Total: {pred['vegas_total']}, Edge: {total_edge:+.1f}, Stars: {total_stars}")

        # Save predictions
        predictor.save_predictions(predictions)
