"""
NBA Simple Model - Ridge Regression with Vegas Blend

Clean baseline model with:
- Flat universal HCA = 1.8 (updated annually based on league data)
- No injury/player data - purely team-level stats
- Ridge on differential features with optimal decay 0.97

Grid search results (2026-01-04):
- Flat HCA (1.8): MAE=11.175, Winner=64.22%
- Best model MAE: 11.97 (Vegas: 10.55)

For dynamic per-team HCA and injury features, see nba_enhanced_ridge.py
"""
from __future__ import annotations

import sqlite3
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'nba_games.db'
MODEL_DIR = Path(__file__).parent / 'models'


class NBASimpleModel:
    """
    Simple NBA prediction model using Ridge regression on differential features.

    Key design choices based on empirical analysis:
    1. Use differentials (home - away) rather than raw stats
    2. Ridge regression prevents overfitting with small sample sizes
    3. Blend with Vegas odds for optimal performance
    4. Exponential decay weighting for recent games
    """

    # Optimal blend weights from analysis
    SPREAD_MODEL_WEIGHT = 0.90  # 90% model, 10% Vegas
    TOTAL_MODEL_WEIGHT = 0.40   # 40% model, 60% Vegas (Vegas is better for totals)

    # Model hyperparameters (tuned 2026-01-04)
    DECAY = 0.97  # Exponential decay for weighting recent games
    PREV_HALF_LIFE = 6.0  # Games until prev season influence halved
    HCA = 1.8  # Flat universal HCA (update annually)
    B2B_PENALTY = 1.0  # Back-to-back penalty

    def __init__(self):
        self.spread_model: Ridge | None = None
        self.total_model: Ridge | None = None
        self.spread_scaler: StandardScaler | None = None
        self.total_scaler: StandardScaler | None = None

        # Team state tracking for incremental predictions
        # Each stat has its own weight list to handle missing box score data
        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'ppg_wts': [],
            'papg': [], 'papg_wts': [],
            'fg_pct': [], 'fg_wts': [],
            'three_pct': [], 'three_wts': [],
            'ft_pct': [], 'ft_wts': [],
            'rebounds': [], 'reb_wts': [],
            'assists': [], 'ast_wts': [],
            'steals': [], 'stl_wts': [],
            'blocks': [], 'blk_wts': [],
            'turnovers': [], 'tov_wts': [],
            'game_count': 0  # Track total games for decay
        }))
        self.prev_ratings: dict = {}
        self.last_game: dict = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}

    def _weighted_avg(self, values: list, weights: list) -> float | None:
        """Calculate weighted average with paired decay weights."""
        if not values or not weights or len(values) != len(weights):
            return None
        return float(np.average(values, weights=weights))

    def _get_team_stats(self, team_id: int, season: int) -> dict:
        """Get current season stats with previous season blending."""
        td = self.team_games[team_id][season]
        games_played = td['game_count']

        # Get weighted averages using per-stat weights
        ppg = self._weighted_avg(td['ppg'], td['ppg_wts'])
        papg = self._weighted_avg(td['papg'], td['papg_wts'])
        fg_pct = self._weighted_avg(td['fg_pct'], td['fg_wts'])
        three_pct = self._weighted_avg(td['three_pct'], td['three_wts'])
        ft_pct = self._weighted_avg(td['ft_pct'], td['ft_wts'])
        rebounds = self._weighted_avg(td['rebounds'], td['reb_wts'])
        assists = self._weighted_avg(td['assists'], td['ast_wts'])
        steals = self._weighted_avg(td['steals'], td['stl_wts'])
        blocks = self._weighted_avg(td['blocks'], td['blk_wts'])
        turnovers = self._weighted_avg(td['turnovers'], td['tov_wts'])

        # Get previous season fallback
        prev = self.prev_ratings.get(team_id, {})
        prev_ppg = prev.get('ppg', self.league_avg['ppg'])
        prev_papg = prev.get('papg', self.league_avg['papg'])
        prev_fg = prev.get('fg_pct', 46.0)
        prev_three = prev.get('three_pct', 36.0)
        prev_ft = prev.get('ft_pct', 78.0)
        prev_reb = prev.get('rebounds', 44.0)
        prev_ast = prev.get('assists', 25.0)
        prev_stl = prev.get('steals', 7.5)
        prev_blk = prev.get('blocks', 5.0)
        prev_tov = prev.get('turnovers', 14.0)

        # Blend with previous season (half-life decay)
        if ppg is None:
            return {
                'ppg': prev_ppg, 'papg': prev_papg, 'games': 0,
                'fg_pct': prev_fg, 'three_pct': prev_three, 'ft_pct': prev_ft,
                'rebounds': prev_reb, 'assists': prev_ast,
                'steals': prev_stl, 'blocks': prev_blk, 'turnovers': prev_tov
            }

        blend = 0.5 ** (games_played / self.PREV_HALF_LIFE)

        return {
            'ppg': blend * prev_ppg + (1 - blend) * ppg,
            'papg': blend * prev_papg + (1 - blend) * papg,
            'fg_pct': blend * prev_fg + (1 - blend) * (fg_pct or prev_fg),
            'three_pct': blend * prev_three + (1 - blend) * (three_pct or prev_three),
            'ft_pct': blend * prev_ft + (1 - blend) * (ft_pct or prev_ft),
            'rebounds': blend * prev_reb + (1 - blend) * (rebounds or prev_reb),
            'assists': blend * prev_ast + (1 - blend) * (assists or prev_ast),
            'steals': blend * prev_stl + (1 - blend) * (steals or prev_stl),
            'blocks': blend * prev_blk + (1 - blend) * (blocks or prev_blk),
            'turnovers': blend * prev_tov + (1 - blend) * (turnovers or prev_tov),
            'games': games_played
        }

    def _get_rest_days(self, team_id: int, game_date: str) -> int:
        """Get days since team's last game."""
        if team_id not in self.last_game:
            return 3  # Default for first game

        curr = datetime.strptime(game_date[:10], '%Y-%m-%d')
        last = datetime.strptime(self.last_game[team_id][:10], '%Y-%m-%d')
        return max(0, min((curr - last).days - 1, 5))

    def extract_features(self, home_id: int, away_id: int, season: int,
                         game_date: str) -> np.ndarray:
        """
        Extract differential features for a game.

        Returns 12-dimensional feature vector (reduced from 18):
        - PPG differential
        - PAPG differential
        - FG% differential
        - Rebounds differential
        - Turnovers differential
        - Rest days differential
        - Home B2B indicator
        - Away B2B indicator
        - Net rating differential
        - Scoring margin differential
        - Home reliability
        - Away reliability

        Removed weak signals: 3P%, FT%, Assists, Steals, Blocks, HCA constant
        """
        home_stats = self._get_team_stats(home_id, season)
        away_stats = self._get_team_stats(away_id, season)

        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)

        home_net = home_stats['ppg'] - home_stats['papg']
        away_net = away_stats['ppg'] - away_stats['papg']

        features = np.array([
            home_stats['ppg'] - away_stats['ppg'],           # 0: PPG diff
            home_stats['papg'] - away_stats['papg'],         # 1: PAPG diff
            home_stats['fg_pct'] - away_stats['fg_pct'],     # 2: FG% diff
            home_stats['rebounds'] - away_stats['rebounds'], # 3: Reb diff
            home_stats['turnovers'] - away_stats['turnovers'],  # 4: TOV diff
            home_rest - away_rest,                           # 5: Rest diff
            1.0 if home_rest == 0 else 0.0,                  # 6: Home B2B
            1.0 if away_rest == 0 else 0.0,                  # 7: Away B2B
            home_net - away_net,                             # 8: Net rating diff
            (home_stats['ppg'] - home_stats['papg']) -
            (away_stats['ppg'] - away_stats['papg']),        # 9: Margin diff
            min(home_stats['games'] / 20.0, 1.0),            # 10: Home reliability
            min(away_stats['games'] / 20.0, 1.0),            # 11: Away reliability
        ])

        return features

    def extract_total_features(self, home_id: int, away_id: int, season: int,
                               game_date: str) -> np.ndarray:
        """
        Extract features for total points prediction.

        Returns 6-dimensional feature vector (simplified from 10).
        Analysis showed minimal features work as well as complex ones,
        and Vegas is actually better than model for totals.
        """
        home_stats = self._get_team_stats(home_id, season)
        away_stats = self._get_team_stats(away_id, season)

        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)

        features = np.array([
            home_stats['ppg'] + away_stats['ppg'],             # 0: Combined PPG
            home_stats['papg'] + away_stats['papg'],           # 1: Combined PAPG
            1.0 if home_rest == 0 else 0.0,                    # 2: Home B2B
            1.0 if away_rest == 0 else 0.0,                    # 3: Away B2B
            home_stats['rebounds'] + away_stats['rebounds'],   # 4: Combined rebounds
            home_stats['turnovers'] + away_stats['turnovers'], # 5: Combined turnovers
        ])

        return features

    def update_team(self, team_id: int, season: int, game_date: str,
                    points_for: int, points_against: int,
                    fg_pct: float = None, three_pct: float = None,
                    ft_pct: float = None, rebounds: float = None,
                    assists: float = None, steals: float = None,
                    blocks: float = None, turnovers: float = None):
        """Update team state after a game."""
        td = self.team_games[team_id][season]

        # Apply decay to all per-stat weights
        for wts_key in ['ppg_wts', 'papg_wts', 'fg_wts', 'three_wts', 'ft_wts',
                        'reb_wts', 'ast_wts', 'stl_wts', 'blk_wts', 'tov_wts']:
            td[wts_key] = [w * self.DECAY for w in td[wts_key]]

        # Add PPG/PAPG (always available)
        td['ppg'].append(points_for)
        td['ppg_wts'].append(1.0)
        td['papg'].append(points_against)
        td['papg_wts'].append(1.0)
        td['game_count'] += 1

        # Add box score stats with weights (only if available and not NaN)
        if pd.notna(fg_pct):
            td['fg_pct'].append(fg_pct)
            td['fg_wts'].append(1.0)
        if pd.notna(three_pct):
            td['three_pct'].append(three_pct)
            td['three_wts'].append(1.0)
        if pd.notna(ft_pct):
            td['ft_pct'].append(ft_pct)
            td['ft_wts'].append(1.0)
        if pd.notna(rebounds):
            td['rebounds'].append(rebounds)
            td['reb_wts'].append(1.0)
        if pd.notna(assists):
            td['assists'].append(assists)
            td['ast_wts'].append(1.0)
        if pd.notna(steals):
            td['steals'].append(steals)
            td['stl_wts'].append(1.0)
        if pd.notna(blocks):
            td['blocks'].append(blocks)
            td['blk_wts'].append(1.0)
        if pd.notna(turnovers):
            td['turnovers'].append(turnovers)
            td['tov_wts'].append(1.0)

        self.last_game[team_id] = game_date

    def set_previous_season(self, season: int):
        """Set previous season ratings for blending."""
        prev = season - 1
        for team_id in self.team_games:
            if prev in self.team_games[team_id]:
                td = self.team_games[team_id][prev]
                if td['ppg']:
                    self.prev_ratings[team_id] = {
                        'ppg': np.mean(td['ppg']),
                        'papg': np.mean(td['papg']),
                        'fg_pct': np.mean(td['fg_pct']) if td['fg_pct'] else 46.0,
                        'three_pct': np.mean(td['three_pct']) if td['three_pct'] else 36.0,
                        'ft_pct': np.mean(td['ft_pct']) if td['ft_pct'] else 78.0,
                        'rebounds': np.mean(td['rebounds']) if td['rebounds'] else 44.0,
                        'assists': np.mean(td['assists']) if td['assists'] else 25.0,
                        'steals': np.mean(td['steals']) if td['steals'] else 7.5,
                        'blocks': np.mean(td['blocks']) if td['blocks'] else 5.0,
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 14.0,
                    }
        self.last_game.clear()

    def train(self, db_path: Path = DB_PATH):
        """
        Train Ridge regression models on historical data.

        Uses walk-forward approach to avoid data leakage.
        """
        log.info("=" * 60)
        log.info("TRAINING NBA SIMPLE MODEL")
        log.info("=" * 60)

        conn = sqlite3.connect(str(db_path))

        # Get all games with box scores and odds
        games = pd.read_sql_query('''
            SELECT
                g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
                g.home_score, g.away_score,
                hs.field_goal_pct as home_fg, hs.three_point_pct as home_three,
                hs.free_throw_pct as home_ft, hs.total_rebounds as home_reb,
                hs.assists as home_ast, hs.steals as home_stl,
                hs.blocks as home_blk, hs.turnovers as home_tov,
                aws.field_goal_pct as away_fg, aws.three_point_pct as away_three,
                aws.free_throw_pct as away_ft, aws.total_rebounds as away_reb,
                aws.assists as away_ast, aws.steals as away_stl,
                aws.blocks as away_blk, aws.turnovers as away_tov,
                o.latest_spread as vegas_spread, o.latest_total as vegas_total
            FROM games g
            LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id
                AND g.home_team_id = hs.team_id
            LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id
                AND g.away_team_id = aws.team_id
            LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.home_score > 0 AND g.completed = 1
            ORDER BY g.date
        ''', conn)
        conn.close()

        log.info(f"Total games: {len(games)}")
        log.info(f"Games with Vegas odds: {games['vegas_spread'].notna().sum()}")

        # Process games chronologically
        X_spread, y_spread = [], []
        X_total, y_total = [], []
        vegas_spreads, vegas_totals = [], []
        game_seasons = []  # Track season for each row

        seasons = sorted(games['season'].unique())

        for season in seasons:
            if season > seasons[0]:
                self.set_previous_season(season)
                prev_games = games[games['season'] == season - 1]
                if len(prev_games) > 0:
                    self.league_avg['ppg'] = prev_games['home_score'].mean()
                    self.league_avg['papg'] = prev_games['away_score'].mean()

            season_games = games[games['season'] == season]

            for _, g in season_games.iterrows():
                # Extract features BEFORE updating (no data leakage)
                features = self.extract_features(
                    g['home_team_id'], g['away_team_id'],
                    season, g['date']
                )
                total_features = self.extract_total_features(
                    g['home_team_id'], g['away_team_id'],
                    season, g['date']
                )

                # Actual outcomes (spread = away - home, positive means away won)
                actual_spread = g['away_score'] - g['home_score']
                actual_total = g['home_score'] + g['away_score']

                # Store training data
                X_spread.append(features)
                y_spread.append(actual_spread)
                X_total.append(total_features)
                y_total.append(actual_total)
                game_seasons.append(season)

                # Store actual Vegas lines (NaN if missing - we'll handle in evaluation)
                vegas_spreads.append(g['vegas_spread'] if pd.notna(g['vegas_spread']) else np.nan)
                vegas_totals.append(g['vegas_total'] if pd.notna(g['vegas_total']) else np.nan)

                # Update team states
                self.update_team(
                    g['home_team_id'], season, g['date'],
                    g['home_score'], g['away_score'],
                    g['home_fg'], g['home_three'], g['home_ft'],
                    g['home_reb'], g['home_ast'], g['home_stl'],
                    g['home_blk'], g['home_tov']
                )
                self.update_team(
                    g['away_team_id'], season, g['date'],
                    g['away_score'], g['home_score'],
                    g['away_fg'], g['away_three'], g['away_ft'],
                    g['away_reb'], g['away_ast'], g['away_stl'],
                    g['away_blk'], g['away_tov']
                )

        X_spread = np.array(X_spread)
        y_spread = np.array(y_spread)
        X_total = np.array(X_total)
        y_total = np.array(y_total)
        vegas_spreads = np.array(vegas_spreads)
        vegas_totals = np.array(vegas_totals)
        game_seasons = np.array(game_seasons)

        # Handle NaN values
        nan_mask = np.isnan(X_spread).any(axis=1) | np.isnan(y_spread)
        log.info(f"Dropping {nan_mask.sum()} rows with NaN values")
        X_spread = X_spread[~nan_mask]
        y_spread = y_spread[~nan_mask]
        X_total = X_total[~nan_mask]
        y_total = y_total[~nan_mask]
        vegas_spreads = vegas_spreads[~nan_mask]
        vegas_totals = vegas_totals[~nan_mask]
        game_seasons = game_seasons[~nan_mask]

        # Also handle any remaining NaN in totals
        total_nan_mask = np.isnan(X_total).any(axis=1) | np.isnan(y_total)
        if total_nan_mask.sum() > 0:
            log.info(f"Dropping {total_nan_mask.sum()} additional rows with NaN in totals")
            X_spread = X_spread[~total_nan_mask]
            y_spread = y_spread[~total_nan_mask]
            X_total = X_total[~total_nan_mask]
            y_total = y_total[~total_nan_mask]
            vegas_spreads = vegas_spreads[~total_nan_mask]
            vegas_totals = vegas_totals[~total_nan_mask]
            game_seasons = game_seasons[~total_nan_mask]

        # Split: use second-to-last season as test (last may be incomplete)
        # If we have at least 3 seasons, use second-to-last; otherwise use last
        if len(seasons) >= 3:
            test_season = seasons[-2]  # Use 2025 (complete) not 2026 (ongoing)
            train_mask = game_seasons < test_season
            test_mask = game_seasons == test_season
        else:
            test_season = seasons[-1]
            train_mask = game_seasons != test_season
            test_mask = game_seasons == test_season

        X_train_s, X_test_s = X_spread[train_mask], X_spread[test_mask]
        y_train_s, y_test_s = y_spread[train_mask], y_spread[test_mask]
        X_train_t, X_test_t = X_total[train_mask], X_total[test_mask]
        y_train_t, y_test_t = y_total[train_mask], y_total[test_mask]
        vegas_test_s = vegas_spreads[test_mask]
        vegas_test_t = vegas_totals[test_mask]

        log.info(f"Train: {len(X_train_s)}, Test: {len(X_test_s)} (season {test_season})")

        # Train spread model
        self.spread_scaler = StandardScaler()
        X_train_s_scaled = self.spread_scaler.fit_transform(X_train_s)
        X_test_s_scaled = self.spread_scaler.transform(X_test_s)

        self.spread_model = Ridge(alpha=1.0)
        self.spread_model.fit(X_train_s_scaled, y_train_s)

        # Train total model
        self.total_scaler = StandardScaler()
        X_train_t_scaled = self.total_scaler.fit_transform(X_train_t)
        X_test_t_scaled = self.total_scaler.transform(X_test_t)

        self.total_model = Ridge(alpha=1.0)
        self.total_model.fit(X_train_t_scaled, y_train_t)

        # Evaluate
        log.info("\n" + "=" * 60)
        log.info("TEST SET RESULTS")
        log.info("=" * 60)

        # Identify games with actual Vegas lines (not NaN)
        has_vegas_spread = ~np.isnan(vegas_test_s)
        has_vegas_total = ~np.isnan(vegas_test_t)
        n_with_vegas = has_vegas_spread.sum()

        log.info(f"\nTest games: {len(y_test_s)} total, {n_with_vegas} with Vegas lines")

        # Spread predictions (model predicts all games)
        model_spread = self.spread_model.predict(X_test_s_scaled)

        # For blending, use model-only when Vegas missing
        vegas_for_blend_s = np.where(has_vegas_spread, vegas_test_s, model_spread)
        blended_spread = (self.SPREAD_MODEL_WEIGHT * model_spread +
                          (1 - self.SPREAD_MODEL_WEIGHT) * vegas_for_blend_s)

        # Model MAE on all games
        model_spread_mae = np.abs(model_spread - y_test_s).mean()
        # Vegas MAE only on games WITH Vegas lines
        vegas_spread_mae = np.abs(vegas_test_s[has_vegas_spread] - y_test_s[has_vegas_spread]).mean()
        # Blend MAE on all games
        blend_spread_mae = np.abs(blended_spread - y_test_s).mean()

        model_winner_acc = ((model_spread < 0) == (y_test_s < 0)).mean()
        vegas_winner_acc = ((vegas_test_s[has_vegas_spread] < 0) == (y_test_s[has_vegas_spread] < 0)).mean()
        blend_winner_acc = ((blended_spread < 0) == (y_test_s < 0)).mean()

        log.info("\nSPREAD PREDICTION:")
        log.info(f"{'Model':<25} {'MAE':<10} {'Winner Acc':<12} {'Games':<8}")
        log.info("-" * 60)
        log.info(f"{'Ridge (12 features)':<25} {model_spread_mae:.2f}      {model_winner_acc*100:.1f}%        {len(y_test_s)}")
        log.info(f"{'Vegas':<25} {vegas_spread_mae:.2f}      {vegas_winner_acc*100:.1f}%        {n_with_vegas}")
        log.info(f"{'Blended (90/10)':<25} {blend_spread_mae:.2f}      {blend_winner_acc*100:.1f}%        {len(y_test_s)}")

        # Total predictions
        model_total = self.total_model.predict(X_test_t_scaled)

        # For blending, use model-only when Vegas missing
        vegas_for_blend_t = np.where(has_vegas_total, vegas_test_t, model_total)
        blended_total = (self.TOTAL_MODEL_WEIGHT * model_total +
                         (1 - self.TOTAL_MODEL_WEIGHT) * vegas_for_blend_t)

        # Model MAE on all games
        model_total_mae = np.abs(model_total - y_test_t).mean()
        # Vegas MAE only on games WITH Vegas lines
        vegas_total_mae = np.abs(vegas_test_t[has_vegas_total] - y_test_t[has_vegas_total]).mean()
        # Blend MAE on all games
        blend_total_mae = np.abs(blended_total - y_test_t).mean()

        log.info("\nTOTAL PREDICTION:")
        log.info(f"{'Model':<25} {'MAE':<10} {'Games':<8}")
        log.info("-" * 45)
        log.info(f"{'Ridge (6 features)':<25} {model_total_mae:.2f}      {len(y_test_t)}")
        log.info(f"{'Vegas':<25} {vegas_total_mae:.2f}      {has_vegas_total.sum()}")
        log.info(f"{'Blended (40/60)':<25} {blend_total_mae:.2f}      {len(y_test_t)}")

        log.info("\n" + "=" * 60)
        log.info("MODEL COEFFICIENTS")
        log.info("=" * 60)

        feature_names = [
            'PPG diff', 'PAPG diff', 'FG% diff', 'Reb diff', 'TOV diff',
            'Rest diff', 'Home B2B', 'Away B2B', 'Net rating diff',
            'Margin diff', 'Home reliability', 'Away reliability'
        ]

        log.info("\nSpread model coefficients:")
        coefs = list(zip(feature_names, self.spread_model.coef_))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, coef in coefs[:10]:
            log.info(f"  {name:<20} {coef:+.3f}")

        return {
            'spread_mae': blend_spread_mae,
            'total_mae': blend_total_mae,
            'spread_mae_vs_vegas': vegas_spread_mae - blend_spread_mae,
            'total_mae_vs_vegas': vegas_total_mae - blend_total_mae,
        }

    def predict(self, home_id: int, away_id: int, season: int, game_date: str,
                vegas_spread: float = None, vegas_total: float = None) -> dict:
        """
        Make predictions for a game.

        Args:
            home_id: Home team ID
            away_id: Away team ID
            season: Season year
            game_date: Game date string
            vegas_spread: Vegas spread (away - home convention, positive = away favored)
            vegas_total: Vegas total points

        Returns:
            dict with predicted_spread, predicted_total, home_score, away_score
        """
        if self.spread_model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Extract features
        spread_features = self.extract_features(home_id, away_id, season, game_date)
        total_features = self.extract_total_features(home_id, away_id, season, game_date)

        # Scale and predict
        spread_scaled = self.spread_scaler.transform(spread_features.reshape(1, -1))
        total_scaled = self.total_scaler.transform(total_features.reshape(1, -1))

        model_spread = self.spread_model.predict(spread_scaled)[0]
        model_total = self.total_model.predict(total_scaled)[0]

        # Blend with Vegas if available
        if vegas_spread is not None:
            final_spread = (self.SPREAD_MODEL_WEIGHT * model_spread +
                           (1 - self.SPREAD_MODEL_WEIGHT) * vegas_spread)
        else:
            final_spread = model_spread

        if vegas_total is not None:
            final_total = (self.TOTAL_MODEL_WEIGHT * model_total +
                          (1 - self.TOTAL_MODEL_WEIGHT) * vegas_total)
        else:
            final_total = model_total

        # Calculate scores from spread and total
        # spread = away - home, so home = (total - spread) / 2
        home_score = (final_total - final_spread) / 2
        away_score = (final_total + final_spread) / 2

        return {
            'predicted_spread': final_spread,
            'predicted_total': final_total,
            'home_score': home_score,
            'away_score': away_score,
            'model_spread': model_spread,
            'model_total': model_total,
        }

    def save(self, path: Path = None):
        """Save trained model to disk."""
        if path is None:
            path = MODEL_DIR / 'nba_simple_model.pkl'

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert nested defaultdicts to regular dicts for pickling
        team_games_dict = {}
        for team_id, seasons in self.team_games.items():
            team_games_dict[team_id] = {}
            for season, stats in seasons.items():
                team_games_dict[team_id][season] = dict(stats)

        model_data = {
            'spread_model': self.spread_model,
            'total_model': self.total_model,
            'spread_scaler': self.spread_scaler,
            'total_scaler': self.total_scaler,
            'team_games': team_games_dict,
            'prev_ratings': self.prev_ratings,
            'last_game': self.last_game,
            'league_avg': self.league_avg,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        log.info(f"\nModel saved to: {path}")

    @classmethod
    def load(cls, path: Path = None) -> 'NBASimpleModel':
        """Load trained model from disk."""
        if path is None:
            path = MODEL_DIR / 'nba_simple_model.pkl'

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        model = cls()
        model.spread_model = model_data['spread_model']
        model.total_model = model_data['total_model']
        model.spread_scaler = model_data['spread_scaler']
        model.total_scaler = model_data['total_scaler']
        model.team_games = defaultdict(
            lambda: defaultdict(lambda: {
                'ppg': [], 'papg': [], 'wts': [],
                'fg_pct': [], 'three_pct': [], 'ft_pct': [],
                'rebounds': [], 'assists': [], 'steals': [], 'blocks': [], 'turnovers': []
            }),
            model_data['team_games']
        )
        model.prev_ratings = model_data['prev_ratings']
        model.last_game = model_data['last_game']
        model.league_avg = model_data['league_avg']

        return model


def main():
    """Train and save the model."""
    model = NBASimpleModel()
    results = model.train()
    model.save()

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"Spread MAE: {results['spread_mae']:.2f} (beats Vegas by {results['spread_mae_vs_vegas']:.2f})")
    log.info(f"Total MAE:  {results['total_mae']:.2f} (beats Vegas by {results['total_mae_vs_vegas']:.2f})")


if __name__ == '__main__':
    main()
