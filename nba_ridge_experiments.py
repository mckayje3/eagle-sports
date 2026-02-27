"""
NBA Ridge Model Experimentation Framework

Uses 2024 season (complete data) to systematically test:
- Feature engineering variations
- Hyperparameter tuning (alpha, decay, blend weights)
- Walk-forward validation

Goal: Dial in the Ridge model as much as possible.
"""
from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'nba_games.db'


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    alpha: float = 1.0
    decay: float = 0.93
    prev_half_life: float = 6.0
    hca: float = 2.0
    b2b_penalty: float = 1.0
    spread_vegas_weight: float = 0.10  # Weight for Vegas in blend
    total_vegas_weight: float = 0.40
    recent_games: int = 10
    use_opponent_adjusted: bool = False
    use_pace_adjusted: bool = False
    use_four_factors: bool = False
    feature_set: str = 'standard'  # standard, extended, minimal


@dataclass
class ExperimentResults:
    """Results from a single experiment."""
    config: ExperimentConfig
    spread_mae: float
    spread_mae_vs_vegas: float
    total_mae: float
    total_mae_vs_vegas: float
    winner_accuracy: float
    vegas_winner_accuracy: float
    ats_record: tuple  # (wins, losses, pushes)
    ou_record: tuple
    n_games: int

    def __str__(self):
        return (
            f"{self.config.name}:\n"
            f"  Spread MAE: {self.spread_mae:.3f} (Vegas: {self.spread_mae + self.spread_mae_vs_vegas:.3f}, diff: {self.spread_mae_vs_vegas:+.3f})\n"
            f"  Winner: {self.winner_accuracy*100:.1f}% (Vegas: {self.vegas_winner_accuracy*100:.1f}%)\n"
            f"  ATS: {self.ats_record[0]}-{self.ats_record[1]}-{self.ats_record[2]}\n"
            f"  Total MAE: {self.total_mae:.3f} (Vegas: {self.total_mae + self.total_mae_vs_vegas:.3f})\n"
            f"  O/U: {self.ou_record[0]}-{self.ou_record[1]}-{self.ou_record[2]}"
        )


class NBAExperimentModel:
    """Experimental NBA model for testing different configurations."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.spread_model: Ridge | None = None
        self.total_model: Ridge | None = None
        self.spread_scaler = StandardScaler()
        self.total_scaler = StandardScaler()

        # Team state
        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'fg_pct': [], 'three_pct': [], 'ft_pct': [],
            'rebounds': [], 'off_reb': [], 'def_reb': [],
            'assists': [], 'steals': [], 'blocks': [], 'turnovers': [],
            'paint': [], 'fastbreak': [],
            'fga': [], 'fgm': [], 'tpa': [], 'tpm': [], 'fta': [], 'ftm': [],
            # Opponent stats (for opponent-adjusted)
            'opp_ppg': [], 'opp_fg_pct': [], 'opp_three_pct': [],
        }))
        self.prev_ratings: dict = {}
        self.last_game: dict = {}
        self.league_avg = {'ppg': 115.0, 'pace': 100.0}

    def reset_state(self):
        """Reset team state for new experiment."""
        self.team_games.clear()
        self.prev_ratings.clear()
        self.last_game.clear()

    def _weighted_avg(self, values: list, weights: list) -> float | None:
        if not values or not weights:
            return None
        # Align lengths
        n = min(len(values), len(weights))
        return float(np.average(values[-n:], weights=weights[-n:]))

    def _get_team_stats(self, team_id: int, season: int) -> dict:
        """Get team stats with configurable blending."""
        td = self.team_games[team_id][season]
        games = len(td['ppg'])

        def wavg(key):
            if not td[key]:
                return None
            return self._weighted_avg(td[key], td['wts'])

        # Current season stats
        ppg = wavg('ppg')
        papg = wavg('papg')
        fg_pct = wavg('fg_pct')
        three_pct = wavg('three_pct')
        ft_pct = wavg('ft_pct')
        rebounds = wavg('rebounds')
        off_reb = wavg('off_reb')
        def_reb = wavg('def_reb')
        assists = wavg('assists')
        steals = wavg('steals')
        blocks = wavg('blocks')
        turnovers = wavg('turnovers')
        paint = wavg('paint')
        fastbreak = wavg('fastbreak')

        # Four factors components
        fga = wavg('fga')
        fgm = wavg('fgm')
        tpa = wavg('tpa')
        tpm = wavg('tpm')
        fta = wavg('fta')

        # Opponent-adjusted stats
        opp_ppg = wavg('opp_ppg')
        opp_fg_pct = wavg('opp_fg_pct')

        # Previous season fallback
        prev = self.prev_ratings.get(team_id, {})
        defaults = {
            'ppg': self.league_avg['ppg'], 'papg': self.league_avg['ppg'],
            'fg_pct': 47.0, 'three_pct': 36.0, 'ft_pct': 78.0,
            'rebounds': 44.0, 'off_reb': 10.0, 'def_reb': 34.0,
            'assists': 25.0, 'steals': 7.5, 'blocks': 5.0, 'turnovers': 14.0,
            'paint': 48.0, 'fastbreak': 13.0,
            'fga': 88.0, 'fgm': 41.0, 'tpa': 35.0, 'tpm': 13.0, 'fta': 22.0,
            'efg': 0.54, 'tov_rate': 0.13, 'orb_rate': 0.25, 'ft_rate': 0.25,
        }

        if ppg is None:
            return {k: prev.get(k, v) for k, v in defaults.items()} | {'games': 0}

        # Blend with previous season
        blend = 0.5 ** (games / self.config.prev_half_life)

        def blended(curr, key):
            if curr is None:
                return prev.get(key, defaults[key])
            return blend * prev.get(key, defaults[key]) + (1 - blend) * curr

        stats = {
            'ppg': blended(ppg, 'ppg'),
            'papg': blended(papg, 'papg'),
            'fg_pct': blended(fg_pct, 'fg_pct'),
            'three_pct': blended(three_pct, 'three_pct'),
            'ft_pct': blended(ft_pct, 'ft_pct'),
            'rebounds': blended(rebounds, 'rebounds'),
            'off_reb': blended(off_reb, 'off_reb'),
            'def_reb': blended(def_reb, 'def_reb'),
            'assists': blended(assists, 'assists'),
            'steals': blended(steals, 'steals'),
            'blocks': blended(blocks, 'blocks'),
            'turnovers': blended(turnovers, 'turnovers'),
            'paint': blended(paint, 'paint'),
            'fastbreak': blended(fastbreak, 'fastbreak'),
            'games': games,
        }

        # Four factors (if enabled)
        if self.config.use_four_factors:
            fga_val = blended(fga, 'fga') or 88
            tpm_val = blended(tpm, 'tpm') or 13
            fgm_val = blended(fgm, 'fgm') or 41
            tov_val = blended(turnovers, 'turnovers') or 14
            orb_val = blended(off_reb, 'off_reb') or 10
            drb_val = blended(def_reb, 'def_reb') or 34
            fta_val = blended(fta, 'fta') or 22

            # Effective FG% = (FGM + 0.5 * 3PM) / FGA
            stats['efg'] = (fgm_val + 0.5 * tpm_val) / fga_val if fga_val > 0 else 0.54
            # Turnover rate = TOV / (FGA + 0.44 * FTA + TOV)
            poss_proxy = fga_val + 0.44 * fta_val + tov_val
            stats['tov_rate'] = tov_val / poss_proxy if poss_proxy > 0 else 0.13
            # Offensive rebound rate = ORB / (ORB + Opp DRB) - approximate with just ORB
            stats['orb_rate'] = orb_val / (orb_val + drb_val) if (orb_val + drb_val) > 0 else 0.25
            # Free throw rate = FTA / FGA
            stats['ft_rate'] = fta_val / fga_val if fga_val > 0 else 0.25

        # Opponent-adjusted (if enabled)
        if self.config.use_opponent_adjusted and opp_ppg is not None:
            # Defensive rating relative to league average
            stats['def_rating'] = opp_ppg - self.league_avg['ppg']
        else:
            stats['def_rating'] = 0

        return stats

    def _get_rest_days(self, team_id: int, game_date: str) -> int:
        if team_id not in self.last_game:
            return 3
        from datetime import datetime
        try:
            curr = datetime.strptime(game_date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[team_id][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 1

    def _get_recent_form(self, team_id: int, season: int, n: int = None) -> dict:
        """Get recent form over last N games."""
        if n is None:
            n = self.config.recent_games
        td = self.team_games[team_id][season]

        if len(td['ppg']) < n:
            return {'ppg': None, 'papg': None, 'margin': None}

        recent_ppg = np.mean(td['ppg'][-n:])
        recent_papg = np.mean(td['papg'][-n:])

        return {
            'ppg': recent_ppg,
            'papg': recent_papg,
            'margin': recent_ppg - recent_papg,
        }

    def extract_spread_features(self, home_id: int, away_id: int,
                                 season: int, game_date: str) -> np.ndarray:
        """Extract features for spread prediction based on config."""
        home = self._get_team_stats(home_id, season)
        away = self._get_team_stats(away_id, season)

        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)

        home_recent = self._get_recent_form(home_id, season)
        away_recent = self._get_recent_form(away_id, season)

        # Base features (always included)
        features = [
            home['ppg'] - away['ppg'],                    # PPG differential
            home['papg'] - away['papg'],                  # Defensive differential
            (home['ppg'] - home['papg']) - (away['ppg'] - away['papg']),  # Net rating diff
            home_rest - away_rest,                        # Rest advantage
            1.0 if home_rest == 0 else 0.0,              # Home B2B
            1.0 if away_rest == 0 else 0.0,              # Away B2B
            min(home['games'] / 20.0, 1.0),              # Home reliability
            min(away['games'] / 20.0, 1.0),              # Away reliability
            1.0,                                          # HCA constant
        ]

        if self.config.feature_set in ('standard', 'extended'):
            # Standard shooting/box features
            features.extend([
                home['fg_pct'] - away['fg_pct'],
                home['three_pct'] - away['three_pct'],
                home['ft_pct'] - away['ft_pct'],
                home['rebounds'] - away['rebounds'],
                home['assists'] - away['assists'],
                home['steals'] - away['steals'],
                home['blocks'] - away['blocks'],
                home['turnovers'] - away['turnovers'],
            ])

        if self.config.feature_set == 'extended':
            # Extended features
            features.extend([
                home['paint'] - away['paint'],
                home['fastbreak'] - away['fastbreak'],
                home['off_reb'] - away['off_reb'],
                home['def_reb'] - away['def_reb'],
            ])

            # Recent form
            if home_recent['margin'] is not None and away_recent['margin'] is not None:
                features.append(home_recent['margin'] - away_recent['margin'])
            else:
                features.append(0.0)

        if self.config.use_four_factors:
            features.extend([
                home.get('efg', 0.54) - away.get('efg', 0.54),
                home.get('tov_rate', 0.13) - away.get('tov_rate', 0.13),
                home.get('orb_rate', 0.25) - away.get('orb_rate', 0.25),
                home.get('ft_rate', 0.25) - away.get('ft_rate', 0.25),
            ])

        if self.config.use_opponent_adjusted:
            features.append(home.get('def_rating', 0) - away.get('def_rating', 0))

        return np.array(features)

    def extract_total_features(self, home_id: int, away_id: int,
                                season: int, game_date: str) -> np.ndarray:
        """Extract features for total prediction."""
        home = self._get_team_stats(home_id, season)
        away = self._get_team_stats(away_id, season)

        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)

        features = [
            home['ppg'] + away['ppg'],                    # Combined scoring
            home['papg'] + away['papg'],                  # Combined defense (points allowed)
            (home['ppg'] + home['papg']) / 2,            # Home pace proxy
            (away['ppg'] + away['papg']) / 2,            # Away pace proxy
            home['fg_pct'] + away['fg_pct'],             # Combined FG%
            home['three_pct'] + away['three_pct'],       # Combined 3P%
            1.0 if home_rest == 0 else 0.0,              # Home B2B
            1.0 if away_rest == 0 else 0.0,              # Away B2B
            min(home['games'] / 20.0, 1.0),
            min(away['games'] / 20.0, 1.0),
        ]

        if self.config.feature_set == 'extended':
            features.extend([
                home['paint'] + away['paint'],
                home['fastbreak'] + away['fastbreak'],
            ])

        return np.array(features)

    def update_team(self, team_id: int, season: int, game_date: str,
                    pts_for: int, pts_against: int, box: dict):
        """Update team state after a game."""
        td = self.team_games[team_id][season]

        # Apply decay
        td['wts'] = [w * self.config.decay for w in td['wts']]

        # Add new game
        td['ppg'].append(pts_for)
        td['papg'].append(pts_against)
        td['wts'].append(1.0)

        # Box score stats
        for key in ['fg_pct', 'three_pct', 'ft_pct', 'rebounds', 'off_reb', 'def_reb',
                    'assists', 'steals', 'blocks', 'turnovers', 'paint', 'fastbreak',
                    'fga', 'fgm', 'tpa', 'tpm', 'fta']:
            if key in box and box[key] is not None:
                td[key].append(box[key])

        # Opponent stats for opponent-adjusted
        td['opp_ppg'].append(pts_against)

        self.last_game[team_id] = game_date

    def set_previous_season(self, current_season: int, games_df: pd.DataFrame):
        """Calculate previous season ratings."""
        prev_season = current_season - 1
        prev_games = games_df[games_df['season'] == prev_season]

        if len(prev_games) == 0:
            return

        # Calculate league average
        self.league_avg['ppg'] = prev_games['home_score'].mean()

        # Calculate team ratings
        for team_id in set(prev_games['home_team_id']) | set(prev_games['away_team_id']):
            team_home = prev_games[prev_games['home_team_id'] == team_id]
            team_away = prev_games[prev_games['away_team_id'] == team_id]

            pts = list(team_home['home_score']) + list(team_away['away_score'])
            pts_against = list(team_home['away_score']) + list(team_away['home_score'])

            if pts:
                self.prev_ratings[team_id] = {
                    'ppg': np.mean(pts),
                    'papg': np.mean(pts_against),
                }

        self.last_game.clear()


def load_2024_data() -> pd.DataFrame:
    """Load 2024 season data with all needed columns."""
    conn = sqlite3.connect(str(DB_PATH))

    games = pd.read_sql_query('''
        SELECT
            g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
            g.home_score, g.away_score,
            hs.field_goal_pct as home_fg, hs.three_point_pct as home_three,
            hs.free_throw_pct as home_ft, hs.total_rebounds as home_reb,
            hs.offensive_rebounds as home_oreb, hs.defensive_rebounds as home_dreb,
            hs.assists as home_ast, hs.steals as home_stl,
            hs.blocks as home_blk, hs.turnovers as home_tov,
            hs.points_in_paint as home_paint, hs.fast_break_points as home_fb,
            hs.field_goals_attempted as home_fga, hs.field_goals_made as home_fgm,
            hs.three_pointers_attempted as home_tpa, hs.three_pointers_made as home_tpm,
            hs.free_throws_attempted as home_fta,
            aws.field_goal_pct as away_fg, aws.three_point_pct as away_three,
            aws.free_throw_pct as away_ft, aws.total_rebounds as away_reb,
            aws.offensive_rebounds as away_oreb, aws.defensive_rebounds as away_dreb,
            aws.assists as away_ast, aws.steals as away_stl,
            aws.blocks as away_blk, aws.turnovers as away_tov,
            aws.points_in_paint as away_paint, aws.fast_break_points as away_fb,
            aws.field_goals_attempted as away_fga, aws.field_goals_made as away_fgm,
            aws.three_pointers_attempted as away_tpa, aws.three_pointers_made as away_tpm,
            aws.free_throws_attempted as away_fta,
            o.opening_spread, o.latest_spread as vegas_spread,
            o.opening_total, o.latest_total as vegas_total
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
    ''', conn)
    conn.close()

    return games


def run_experiment(config: ExperimentConfig, games: pd.DataFrame,
                   test_season: int = 2024) -> ExperimentResults:
    """Run a single experiment with walk-forward validation."""
    model = NBAExperimentModel(config)

    # Set up previous season ratings
    model.set_previous_season(test_season, games)

    # Process training seasons (before test season)
    train_games = games[games['season'] < test_season].copy()
    for season in sorted(train_games['season'].unique()):
        if season > train_games['season'].min():
            model.set_previous_season(season, games)

        season_games = train_games[train_games['season'] == season]
        for _, g in season_games.iterrows():
            home_box = {
                'fg_pct': g['home_fg'], 'three_pct': g['home_three'], 'ft_pct': g['home_ft'],
                'rebounds': g['home_reb'], 'off_reb': g['home_oreb'], 'def_reb': g['home_dreb'],
                'assists': g['home_ast'], 'steals': g['home_stl'], 'blocks': g['home_blk'],
                'turnovers': g['home_tov'], 'paint': g['home_paint'], 'fastbreak': g['home_fb'],
                'fga': g['home_fga'], 'fgm': g['home_fgm'], 'tpa': g['home_tpa'],
                'tpm': g['home_tpm'], 'fta': g['home_fta'],
            }
            away_box = {
                'fg_pct': g['away_fg'], 'three_pct': g['away_three'], 'ft_pct': g['away_ft'],
                'rebounds': g['away_reb'], 'off_reb': g['away_oreb'], 'def_reb': g['away_dreb'],
                'assists': g['away_ast'], 'steals': g['away_stl'], 'blocks': g['away_blk'],
                'turnovers': g['away_tov'], 'paint': g['away_paint'], 'fastbreak': g['away_fb'],
                'fga': g['away_fga'], 'fgm': g['away_fgm'], 'tpa': g['away_tpa'],
                'tpm': g['away_tpm'], 'fta': g['away_fta'],
            }
            model.update_team(g['home_team_id'], season, g['date'], g['home_score'], g['away_score'], home_box)
            model.update_team(g['away_team_id'], season, g['date'], g['away_score'], g['home_score'], away_box)

    # Walk-forward on test season
    model.set_previous_season(test_season, games)
    test_games = games[games['season'] == test_season].copy()

    X_spread, y_spread = [], []
    X_total, y_total = [], []
    vegas_spreads, vegas_totals = [], []

    for _, g in test_games.iterrows():
        # Extract features BEFORE updating
        spread_feat = model.extract_spread_features(g['home_team_id'], g['away_team_id'], test_season, g['date'])
        total_feat = model.extract_total_features(g['home_team_id'], g['away_team_id'], test_season, g['date'])

        actual_spread = g['away_score'] - g['home_score']
        actual_total = g['home_score'] + g['away_score']

        X_spread.append(spread_feat)
        y_spread.append(actual_spread)
        X_total.append(total_feat)
        y_total.append(actual_total)
        vegas_spreads.append(g['vegas_spread'] if pd.notna(g['vegas_spread']) else 0)
        vegas_totals.append(g['vegas_total'] if pd.notna(g['vegas_total']) else 220)

        # Update team state
        home_box = {
            'fg_pct': g['home_fg'], 'three_pct': g['home_three'], 'ft_pct': g['home_ft'],
            'rebounds': g['home_reb'], 'off_reb': g['home_oreb'], 'def_reb': g['home_dreb'],
            'assists': g['home_ast'], 'steals': g['home_stl'], 'blocks': g['home_blk'],
            'turnovers': g['home_tov'], 'paint': g['home_paint'], 'fastbreak': g['home_fb'],
            'fga': g['home_fga'], 'fgm': g['home_fgm'], 'tpa': g['home_tpa'],
            'tpm': g['home_tpm'], 'fta': g['home_fta'],
        }
        away_box = {
            'fg_pct': g['away_fg'], 'three_pct': g['away_three'], 'ft_pct': g['away_ft'],
            'rebounds': g['away_reb'], 'off_reb': g['away_oreb'], 'def_reb': g['away_dreb'],
            'assists': g['away_ast'], 'steals': g['away_stl'], 'blocks': g['away_blk'],
            'turnovers': g['away_tov'], 'paint': g['away_paint'], 'fastbreak': g['away_fb'],
            'fga': g['away_fga'], 'fgm': g['away_fgm'], 'tpa': g['away_tpa'],
            'tpm': g['away_tpm'], 'fta': g['away_fta'],
        }
        model.update_team(g['home_team_id'], test_season, g['date'], g['home_score'], g['away_score'], home_box)
        model.update_team(g['away_team_id'], test_season, g['date'], g['away_score'], g['home_score'], away_box)

    X_spread = np.array(X_spread)
    y_spread = np.array(y_spread)
    X_total = np.array(X_total)
    y_total = np.array(y_total)
    vegas_spreads = np.array(vegas_spreads)
    vegas_totals = np.array(vegas_totals)

    # Handle NaN
    valid_mask = ~(np.isnan(X_spread).any(axis=1) | np.isnan(y_spread))
    X_spread = X_spread[valid_mask]
    y_spread = y_spread[valid_mask]
    X_total = X_total[valid_mask]
    y_total = y_total[valid_mask]
    vegas_spreads = vegas_spreads[valid_mask]
    vegas_totals = vegas_totals[valid_mask]

    # Train on first 80%, test on last 20% (within-season validation)
    split_idx = int(len(X_spread) * 0.8)

    X_train_s, X_test_s = X_spread[:split_idx], X_spread[split_idx:]
    y_train_s, y_test_s = y_spread[:split_idx], y_spread[split_idx:]
    X_train_t, X_test_t = X_total[:split_idx], X_total[split_idx:]
    y_train_t, y_test_t = y_total[:split_idx], y_total[split_idx:]
    vegas_test_s = vegas_spreads[split_idx:]
    vegas_test_t = vegas_totals[split_idx:]

    # Scale and train
    X_train_s_scaled = model.spread_scaler.fit_transform(X_train_s)
    X_test_s_scaled = model.spread_scaler.transform(X_test_s)
    X_train_t_scaled = model.total_scaler.fit_transform(X_train_t)
    X_test_t_scaled = model.total_scaler.transform(X_test_t)

    model.spread_model = Ridge(alpha=config.alpha)
    model.spread_model.fit(X_train_s_scaled, y_train_s)

    model.total_model = Ridge(alpha=config.alpha)
    model.total_model.fit(X_train_t_scaled, y_train_t)

    # Predict
    model_spread = model.spread_model.predict(X_test_s_scaled)
    model_total = model.total_model.predict(X_test_t_scaled)

    # Blend with Vegas
    blended_spread = ((1 - config.spread_vegas_weight) * model_spread +
                      config.spread_vegas_weight * vegas_test_s)
    blended_total = ((1 - config.total_vegas_weight) * model_total +
                     config.total_vegas_weight * vegas_test_t)

    # Calculate metrics
    spread_mae = np.abs(blended_spread - y_test_s).mean()
    vegas_spread_mae = np.abs(vegas_test_s - y_test_s).mean()
    total_mae = np.abs(blended_total - y_test_t).mean()
    vegas_total_mae = np.abs(vegas_test_t - y_test_t).mean()

    # Winner accuracy
    model_winner = (blended_spread < 0) == (y_test_s < 0)
    vegas_winner = (vegas_test_s < 0) == (y_test_s < 0)

    # ATS record (did model beat the spread?)
    # Model says bet home if model_spread < vegas_spread (model thinks home will do better)
    ats_wins = 0
    ats_losses = 0
    ats_pushes = 0
    for i in range(len(y_test_s)):
        if vegas_test_s[i] == 0:
            continue
        model_edge = vegas_test_s[i] - blended_spread[i]  # Positive = model likes home
        actual_vs_spread = vegas_test_s[i] - y_test_s[i]  # Positive = home covered

        if abs(actual_vs_spread) < 0.5:
            ats_pushes += 1
        elif (model_edge > 0 and actual_vs_spread > 0) or (model_edge < 0 and actual_vs_spread < 0):
            ats_wins += 1
        else:
            ats_losses += 1

    # O/U record
    ou_wins = 0
    ou_losses = 0
    ou_pushes = 0
    for i in range(len(y_test_t)):
        if vegas_test_t[i] == 0:
            continue
        model_edge = blended_total[i] - vegas_test_t[i]  # Positive = model says over
        actual_vs_total = y_test_t[i] - vegas_test_t[i]  # Positive = went over

        if abs(actual_vs_total) < 0.5:
            ou_pushes += 1
        elif (model_edge > 0 and actual_vs_total > 0) or (model_edge < 0 and actual_vs_total < 0):
            ou_wins += 1
        else:
            ou_losses += 1

    return ExperimentResults(
        config=config,
        spread_mae=spread_mae,
        spread_mae_vs_vegas=vegas_spread_mae - spread_mae,
        total_mae=total_mae,
        total_mae_vs_vegas=vegas_total_mae - total_mae,
        winner_accuracy=model_winner.mean(),
        vegas_winner_accuracy=vegas_winner.mean(),
        ats_record=(ats_wins, ats_losses, ats_pushes),
        ou_record=(ou_wins, ou_losses, ou_pushes),
        n_games=len(y_test_s),
    )


def run_baseline_experiments():
    """Run baseline experiments to establish benchmarks."""
    log.info("=" * 70)
    log.info("NBA RIDGE MODEL EXPERIMENTS - 2024 SEASON")
    log.info("=" * 70)

    games = load_2024_data()
    log.info(f"Loaded {len(games)} games")
    log.info(f"2024 games: {len(games[games['season'] == 2024])}")
    log.info(f"Games with Vegas spread: {games['vegas_spread'].notna().sum()}")

    # Define experiments
    experiments = [
        # Baseline
        ExperimentConfig(name="Baseline (standard)", feature_set='standard'),

        # Feature set variations
        ExperimentConfig(name="Minimal features", feature_set='minimal'),
        ExperimentConfig(name="Extended features", feature_set='extended'),
        ExperimentConfig(name="Four Factors", feature_set='standard', use_four_factors=True),
        ExperimentConfig(name="Extended + Four Factors", feature_set='extended', use_four_factors=True),

        # Alpha variations
        ExperimentConfig(name="Alpha=0.1", alpha=0.1),
        ExperimentConfig(name="Alpha=0.5", alpha=0.5),
        ExperimentConfig(name="Alpha=2.0", alpha=2.0),
        ExperimentConfig(name="Alpha=5.0", alpha=5.0),
        ExperimentConfig(name="Alpha=10.0", alpha=10.0),

        # Decay variations
        ExperimentConfig(name="Decay=0.90", decay=0.90),
        ExperimentConfig(name="Decay=0.95", decay=0.95),
        ExperimentConfig(name="Decay=0.97", decay=0.97),
        ExperimentConfig(name="No decay", decay=1.0),

        # Vegas blend variations
        ExperimentConfig(name="No Vegas blend", spread_vegas_weight=0.0, total_vegas_weight=0.0),
        ExperimentConfig(name="20% Vegas spread", spread_vegas_weight=0.20),
        ExperimentConfig(name="30% Vegas spread", spread_vegas_weight=0.30),
        ExperimentConfig(name="50% Vegas each", spread_vegas_weight=0.50, total_vegas_weight=0.50),

        # Recent games window
        ExperimentConfig(name="Recent=5 games", recent_games=5),
        ExperimentConfig(name="Recent=15 games", recent_games=15),
        ExperimentConfig(name="Recent=20 games", recent_games=20),

        # Previous season blend
        ExperimentConfig(name="Prev half-life=3", prev_half_life=3.0),
        ExperimentConfig(name="Prev half-life=10", prev_half_life=10.0),
        ExperimentConfig(name="Prev half-life=15", prev_half_life=15.0),
    ]

    results = []
    for exp in experiments:
        log.info(f"\nRunning: {exp.name}...")
        try:
            result = run_experiment(exp, games)
            results.append(result)
            log.info(f"  Spread MAE: {result.spread_mae:.3f}, ATS: {result.ats_record[0]}-{result.ats_record[1]}")
        except Exception as e:
            log.error(f"  Failed: {e}")

    # Sort by spread MAE
    results.sort(key=lambda r: r.spread_mae)

    log.info("\n" + "=" * 70)
    log.info("RESULTS RANKED BY SPREAD MAE")
    log.info("=" * 70)

    log.info(f"\n{'Experiment':<30} {'MAE':>8} {'vs Vegas':>10} {'Winner%':>9} {'ATS':>12}")
    log.info("-" * 70)

    for r in results:
        ats_str = f"{r.ats_record[0]}-{r.ats_record[1]}-{r.ats_record[2]}"
        log.info(f"{r.config.name:<30} {r.spread_mae:>8.3f} {r.spread_mae_vs_vegas:>+10.3f} {r.winner_accuracy*100:>8.1f}% {ats_str:>12}")

    # Best configurations
    log.info("\n" + "=" * 70)
    log.info("TOP 5 CONFIGURATIONS")
    log.info("=" * 70)

    for i, r in enumerate(results[:5], 1):
        log.info(f"\n#{i}: {r.config.name}")
        log.info(f"    Alpha: {r.config.alpha}, Decay: {r.config.decay}")
        log.info(f"    Vegas weights: spread={r.config.spread_vegas_weight}, total={r.config.total_vegas_weight}")
        log.info(f"    Feature set: {r.config.feature_set}, Four factors: {r.config.use_four_factors}")
        log.info(str(r))

    return results


if __name__ == '__main__':
    run_baseline_experiments()
