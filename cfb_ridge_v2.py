"""
CFB Ridge Model V2 - Pure Model with SRS Ratings

Ported from NFL Ridge V2 (56.9% ATS at 5+ edge).

Key features:
1. SRS-style opponent-adjusted ratings (critical for 245-team CFB)
2. Real drive data from drives table
3. Dynamic per-team HCA (shrunk toward 3.5)
4. Dampened recent form and momentum
5. Walk-forward validation built in
6. Conference game indicator

CFB-specific adaptations from NFL Ridge V2:
- Lower decay (0.90 vs 0.96) - more games/season, faster adaptation
- Longer prev-season half-life (5.0 vs 4.0) - more roster continuity
- HCA base 3.5 (vs 2.5 NFL) - CFB home field advantage is larger
- Pass yards feature kept (more passing variance in CFB)
- Conference game indicator added (CFB-specific)
- 14-week season normalization (vs 17 NFL)
- Post-bye threshold 14+ days (vs 13 NFL)
- Larger HCA clip range [-3, 10] (vs [-1, 8] NFL)
- SRS recalc every 50 games (vs 30 NFL - more games total)

SPREAD CONVENTION (Vegas standard):
    spread = away_score - home_score
    NEGATIVE spread = HOME team favored
    POSITIVE spread = AWAY team favored
"""
from __future__ import annotations

import logging
import pickle
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'cfb_games.db'
MODEL_DIR = Path(__file__).parent / 'models'

# CFB-tuned constants
DECAY = 0.90               # Lower than NFL (0.96) - more games, faster decay
PREV_HALF_LIFE = 5.0       # Games until 50% current season weight
MIN_GAMES = 2              # Minimum games for prediction eligibility
SRS_BLEND_HALF_LIFE = 8.0  # Games for SRS blend with prev season
SRS_RECALC_INTERVAL = 50   # Recalculate SRS every N games (more games in CFB)
BASE_HCA = 3.5             # CFB home field advantage (~3.5 pts)
HCA_SHRINK = 0.5           # Shrinkage toward base HCA
HCA_HALF_LIFE = 15.0       # Games for HCA blend
FORM_WEIGHT = 0.35         # Dampen recent form influence (more volatile in CFB)
FORM_WINDOW = 4            # Last N games for form
MOMENTUM_WINDOW = 4        # Last N games for momentum


class CFBRidgeV2:
    """
    CFB Ridge Model V2 - Pure predictions with SRS ratings.

    Key features:
    1. SRS-style opponent-adjusted ratings (crucial for 245-team CFB)
    2. Real drive data (scoring %, YPD)
    3. Dynamic per-team HCA (shrunk toward 3.5)
    4. Dampened form/momentum
    5. Pass yards + conference game features
    """

    def __init__(self):
        self.spread_model: Ridge | None = None
        self.total_model: Ridge | None = None
        self.spread_scaler: StandardScaler | None = None
        self.total_scaler: StandardScaler | None = None

        # Score-based model (predicts individual team scores, derives spread/total)
        self.score_model: Ridge | None = None
        self.score_scaler: StandardScaler | None = None

        # Team state tracking
        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'ppg_wts': [],
            'papg': [], 'papg_wts': [],
            'yards': [], 'yards_wts': [],
            'pass_yards': [], 'pass_wts': [],
            'rush_yards': [], 'rush_wts': [],
            'turnovers': [], 'to_wts': [],
            'first_downs': [], 'fd_wts': [],
            # Drive efficiency (from real drive data)
            'scoring_pct': [], 'sp_wts': [],
            'ypd': [], 'ypd_wts': [],
            # Game-level tracking
            'margins': [],
            'wins': [],
            'opponents': [],  # For SRS calculation
            'game_count': 0,
        }))

        self.prev_ratings: dict = {}
        self.last_game: dict = {}
        self.league_avg = {
            'ppg': 28.0, 'papg': 28.0, 'yards': 400.0,
            'pass_yards': 230.0, 'rush_yards': 170.0,
            'turnovers': 1.5, 'first_downs': 20.0,
            'scoring_pct': 0.38, 'ypd': 32.0,
        }

        # SRS ratings (opponent-adjusted)
        self.team_srs: dict = defaultdict(lambda: defaultdict(float))
        self.prev_srs: dict = {}

        # Dynamic HCA tracking
        self.team_hca_data: dict = defaultdict(lambda: defaultdict(lambda: {
            'home_margins': [], 'away_margins': []
        }))
        self.prev_hca: dict = {}

    # -- Helper Methods -------------------------------------------------------

    def _weighted_avg(self, values: list, weights: list) -> float | None:
        if not values or not weights or len(values) != len(weights):
            return None
        return float(np.average(values, weights=weights))

    def _calculate_srs(self, season: int, iterations: int = 20) -> dict[int, float]:
        """
        Calculate Simple Rating System (SRS) for all teams.

        SRS = Average Margin + Strength of Schedule
        where SOS = average opponent SRS. Iteratively solved.
        CFB uses 20 iterations (vs NFL's 10) due to 245 teams.
        """
        teams_with_games = [
            tid for tid in self.team_games
            if season in self.team_games[tid]
            and self.team_games[tid][season]['game_count'] > 0
        ]

        if not teams_with_games:
            return {}

        # Initialize SRS with average margin
        avg_margins = {}
        srs = {}
        for tid in teams_with_games:
            td = self.team_games[tid][season]
            if td['margins']:
                avg_margins[tid] = np.mean(td['margins'])
            else:
                avg_margins[tid] = 0.0
            srs[tid] = avg_margins[tid]

        # Iterate to convergence
        for _ in range(iterations):
            new_srs = {}
            for tid in teams_with_games:
                td = self.team_games[tid][season]
                if td['opponents']:
                    opp_srs = [srs.get(opp, 0) for opp in td['opponents']]
                    sos = np.mean(opp_srs)
                    new_srs[tid] = avg_margins[tid] + sos
                else:
                    new_srs[tid] = avg_margins[tid]
            srs = new_srs

        return srs

    def _get_team_srs(self, team_id: int, season: int) -> float:
        """Get team's SRS rating, blending with previous season early on."""
        current_srs = self.team_srs.get(season, {}).get(team_id, 0)
        prev_srs = self.prev_srs.get(team_id, 0)

        td = self.team_games[team_id][season]
        games = td['game_count']

        if games == 0:
            return prev_srs

        blend = 0.5 ** (games / SRS_BLEND_HALF_LIFE)
        return blend * prev_srs + (1 - blend) * current_srs

    def _get_team_stats(self, team_id: int, season: int) -> dict:
        """Get decay-weighted team stats, blended with previous season."""
        td = self.team_games[team_id][season]
        games_played = td['game_count']

        ppg = self._weighted_avg(td['ppg'], td['ppg_wts'])
        papg = self._weighted_avg(td['papg'], td['papg_wts'])
        yards = self._weighted_avg(td['yards'], td['yards_wts'])
        pass_yds = self._weighted_avg(td['pass_yards'], td['pass_wts'])
        rush_yds = self._weighted_avg(td['rush_yards'], td['rush_wts'])
        turnovers = self._weighted_avg(td['turnovers'], td['to_wts'])
        first_downs = self._weighted_avg(td['first_downs'], td['fd_wts'])
        scoring_pct = self._weighted_avg(td['scoring_pct'], td['sp_wts'])
        ypd = self._weighted_avg(td['ypd'], td['ypd_wts'])

        prev = self.prev_ratings.get(team_id, {})

        if ppg is None:
            return {
                'ppg': prev.get('ppg', self.league_avg['ppg']),
                'papg': prev.get('papg', self.league_avg['papg']),
                'yards': prev.get('yards', self.league_avg['yards']),
                'pass_yards': prev.get('pass_yards', self.league_avg['pass_yards']),
                'rush_yards': prev.get('rush_yards', self.league_avg['rush_yards']),
                'turnovers': prev.get('turnovers', self.league_avg['turnovers']),
                'first_downs': prev.get('first_downs', self.league_avg['first_downs']),
                'scoring_pct': prev.get('scoring_pct', self.league_avg['scoring_pct']),
                'ypd': prev.get('ypd', self.league_avg['ypd']),
                'games': 0,
                'margins': [],
                'wins': [],
            }

        blend = 0.5 ** (games_played / PREV_HALF_LIFE)

        def _blend(current, key, default=None):
            if default is None:
                default = self.league_avg.get(key, 0)
            p = prev.get(key, default)
            if current is None:
                return p
            return blend * p + (1 - blend) * current

        return {
            'ppg': _blend(ppg, 'ppg'),
            'papg': _blend(papg, 'papg'),
            'yards': _blend(yards, 'yards'),
            'pass_yards': _blend(pass_yds, 'pass_yards'),
            'rush_yards': _blend(rush_yds, 'rush_yards'),
            'turnovers': _blend(turnovers, 'turnovers'),
            'first_downs': _blend(first_downs, 'first_downs'),
            'scoring_pct': _blend(scoring_pct, 'scoring_pct'),
            'ypd': _blend(ypd, 'ypd'),
            'games': games_played,
            'margins': td['margins'],
            'wins': td['wins'],
        }

    def _get_rest_days(self, team_id: int, game_date: str) -> int:
        """Get rest days since last game."""
        if team_id not in self.last_game:
            return 7  # Default week rest
        try:
            curr = datetime.strptime(game_date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[team_id][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days, 21))  # CFB can have longer breaks
        except (ValueError, TypeError):
            return 7

    def _get_dynamic_hca(self, home_id: int, season: int) -> float:
        """Calculate dynamic per-team HCA with shrinkage."""
        hd = self.team_hca_data[home_id][season]
        n_home = len(hd['home_margins'])
        n_away = len(hd['away_margins'])
        total_games = n_home + n_away

        if total_games == 0:
            return self.prev_hca.get(home_id, BASE_HCA)

        if n_home > 0 and n_away > 0:
            raw_hca = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
            raw_hca = max(-3, min(raw_hca, 10))  # CFB can have larger HCA
        else:
            raw_hca = BASE_HCA

        # Shrink toward league average
        shrunk_hca = BASE_HCA + HCA_SHRINK * (raw_hca - BASE_HCA)

        # Blend with previous season
        prev = self.prev_hca.get(home_id, BASE_HCA)
        blend_factor = 0.5 ** (total_games / HCA_HALF_LIFE)

        return blend_factor * prev + (1 - blend_factor) * shrunk_hca

    def _recent_form(self, margins: list, n: int = FORM_WINDOW) -> float:
        """Average margin over last n games, dampened."""
        if len(margins) < n:
            return 0.0
        return float(np.mean(margins[-n:])) * FORM_WEIGHT

    def _momentum(self, margins: list, n: int = MOMENTUM_WINDOW) -> float:
        """Trend in margins (second half vs first half of window)."""
        if len(margins) < n:
            return 0.0
        recent = margins[-n:]
        first_half = np.mean(recent[:n // 2])
        second_half = np.mean(recent[n // 2:])
        return (second_half - first_half) * FORM_WEIGHT

    def _streak(self, wins: list) -> int:
        """Current win/loss streak (positive=wins, negative=losses)."""
        if not wins:
            return 0
        s, last = 0, wins[-1]
        for w in reversed(wins):
            if w == last:
                s += 1
            else:
                break
        return s if last == 1 else -s

    # -- Feature Extraction ---------------------------------------------------

    def extract_spread_features(self, home_id: int, away_id: int, season: int,
                                game_date: str, week: int = 7,
                                neutral_site: bool = False,
                                is_conf_game: bool = False) -> np.ndarray | None:
        """
        Extract 20 spread features.

        Returns None if either team has fewer than MIN_GAMES.
        """
        hs = self._get_team_stats(home_id, season)
        aws = self._get_team_stats(away_id, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest_days(home_id, game_date)
        ar = self._get_rest_days(away_id, game_date)

        # SRS differential
        home_srs = self._get_team_srs(home_id, season)
        away_srs = self._get_team_srs(away_id, season)

        # Dynamic HCA (0 for neutral site)
        hca = 0.0 if neutral_site else self._get_dynamic_hca(home_id, season)

        # Post-bye indicators (14+ days rest in CFB)
        h_post_bye = 1.0 if hr >= 14 else 0.0
        a_post_bye = 1.0 if ar >= 14 else 0.0

        return np.array([
            # Core stat differentials
            hs['ppg'] - aws['ppg'],                          #  0: PPG diff
            hs['papg'] - aws['papg'],                        #  1: PAPG diff
            home_srs - away_srs,                             #  2: SRS diff (KEY)

            # Yards-based
            hs['yards'] - aws['yards'],                      #  3: Total yards diff
            hs['pass_yards'] - aws['pass_yards'],            #  4: Pass yards diff (CFB-specific)
            hs['rush_yards'] - aws['rush_yards'],            #  5: Rush yards diff

            # Efficiency metrics
            hs['turnovers'] - aws['turnovers'],              #  6: Turnover diff
            hs['first_downs'] - aws['first_downs'],          #  7: First downs diff

            # Drive efficiency (from REAL drive data)
            hs['scoring_pct'] - aws['scoring_pct'],          #  8: Scoring % diff
            hs['ypd'] - aws['ypd'],                          #  9: Yards per drive diff

            # Form and momentum (dampened)
            self._recent_form(hs['margins']) - self._recent_form(aws['margins']),  # 10
            self._momentum(hs['margins']) - self._momentum(aws['margins']),        # 11
            self._streak(hs['wins']) - self._streak(aws['wins']),                  # 12

            # Rest and bye
            min(hr, 14) - min(ar, 14),                       # 13: Rest diff
            h_post_bye - a_post_bye,                         # 14: Post-bye diff

            # Context
            hca,                                             # 15: Dynamic HCA
            1.0 if is_conf_game else 0.0,                    # 16: Conference game (CFB-specific)
            min(week / 14.0, 1.0),                           # 17: Season progress (CFB 14 weeks)

            # Reliability
            min(hs['games'] / 10.0, 1.0),                   # 18: Home reliability
            min(aws['games'] / 10.0, 1.0),                  # 19: Away reliability
        ])

    def extract_total_features(self, home_id: int, away_id: int, season: int,
                               game_date: str, week: int = 7) -> np.ndarray | None:
        """
        Extract 12 total features (same structure as NFL).

        Returns None if either team has fewer than MIN_GAMES.
        """
        hs = self._get_team_stats(home_id, season)
        aws = self._get_team_stats(away_id, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest_days(home_id, game_date)
        ar = self._get_rest_days(away_id, game_date)

        return np.array([
            hs['ppg'] + aws['ppg'],                          #  0: Combined PPG
            hs['papg'] + aws['papg'],                        #  1: Combined PAPG
            hs['yards'] + aws['yards'],                      #  2: Combined yards
            hs['turnovers'] + aws['turnovers'],              #  3: Combined turnovers
            hs['scoring_pct'] + aws['scoring_pct'],          #  4: Combined scoring %
            hs['ypd'] + aws['ypd'],                          #  5: Combined YPD
            abs(self._recent_form(hs['margins'])) + abs(self._recent_form(aws['margins'])),  # 6: Form volatility
            1.0 if hr >= 14 else 0.0,                        #  7: Home post-bye
            1.0 if ar >= 14 else 0.0,                        #  8: Away post-bye
            min(week / 14.0, 1.0),                           #  9: Season progress
            min(hs['games'] / 10.0, 1.0),                   # 10: Home reliability
            min(aws['games'] / 10.0, 1.0),                  # 11: Away reliability
        ])

    def extract_score_features(self, team_id: int, opp_id: int, season: int,
                               game_date: str, week: int = 7,
                               is_home: bool = True,
                               neutral: bool = False) -> np.ndarray | None:
        """
        Extract 20 features for predicting one team's score.

        Each game produces 2 rows (home→home_score, away→away_score).
        This doubles the training data and lets spread/total emerge from score predictions.
        """
        ts = self._get_team_stats(team_id, season)
        ops = self._get_team_stats(opp_id, season)

        if ts['games'] < MIN_GAMES or ops['games'] < MIN_GAMES:
            return None

        team_srs = self._get_team_srs(team_id, season)
        opp_srs = self._get_team_srs(opp_id, season)
        rest = self._get_rest_days(team_id, game_date)

        # HCA: positive for home team, 0 for away/neutral
        if neutral:
            hca_val = 0.0
        elif is_home:
            hca_val = self._get_dynamic_hca(team_id, season)
        else:
            hca_val = 0.0

        return np.array([
            # Team offensive profile
            ts['ppg'],                                           #  0
            ts['yards'], ts['pass_yards'], ts['rush_yards'],    #  1-3
            ts['turnovers'],                                     #  4
            ts['first_downs'],                                   #  5
            ts['scoring_pct'], ts['ypd'],                        #  6-7
            # Opponent defensive profile
            ops['papg'],                                         #  8
            ops['turnovers'],                                    #  9
            ops['scoring_pct'],                                  # 10
            ops['ypd'],                                          # 11
            # Strength ratings
            team_srs, opp_srs,                                   # 12-13
            # Form
            self._recent_form(ts['margins']),                    # 14
            self._momentum(ts['margins']),                       # 15
            self._streak(ts['wins']),                            # 16
            # Context
            min(rest, 14),                                       # 17
            hca_val,                                             # 18
            min(week / 14.0, 1.0),                               # 19
        ])

    # -- State Updates --------------------------------------------------------

    def update_team(self, team_id: int, opponent_id: int, season: int,
                    game_date: str, points_for: int, points_against: int,
                    is_home: bool, yards: float = None, pass_yards: float = None,
                    rush_yards: float = None, turnovers: float = None,
                    first_downs: float = None,
                    scoring_pct: float = None, ypd: float = None):
        """Update team state after a game."""
        td = self.team_games[team_id][season]

        # Decay all weights
        for wts_key in ['ppg_wts', 'papg_wts', 'yards_wts', 'pass_wts',
                         'rush_wts', 'to_wts', 'fd_wts', 'sp_wts', 'ypd_wts']:
            td[wts_key] = [w * DECAY for w in td[wts_key]]

        # Core stats
        td['ppg'].append(points_for)
        td['ppg_wts'].append(1.0)
        td['papg'].append(points_against)
        td['papg_wts'].append(1.0)
        td['game_count'] += 1
        td['opponents'].append(opponent_id)

        margin = points_for - points_against
        td['margins'].append(margin)
        td['wins'].append(1 if margin > 0 else 0)

        # Box score stats
        if pd.notna(yards):
            td['yards'].append(yards)
            td['yards_wts'].append(1.0)
        if pd.notna(pass_yards):
            td['pass_yards'].append(pass_yards)
            td['pass_wts'].append(1.0)
        if pd.notna(rush_yards):
            td['rush_yards'].append(rush_yards)
            td['rush_wts'].append(1.0)
        if pd.notna(turnovers):
            td['turnovers'].append(turnovers)
            td['to_wts'].append(1.0)
        if pd.notna(first_downs):
            td['first_downs'].append(first_downs)
            td['fd_wts'].append(1.0)

        # Drive efficiency
        if pd.notna(scoring_pct):
            td['scoring_pct'].append(scoring_pct)
            td['sp_wts'].append(1.0)
        if pd.notna(ypd):
            td['ypd'].append(ypd)
            td['ypd_wts'].append(1.0)

        # HCA data
        hd = self.team_hca_data[team_id][season]
        if is_home:
            hd['home_margins'].append(margin)
        else:
            hd['away_margins'].append(-margin)

        self.last_game[team_id] = game_date

    def set_previous_season(self, season: int):
        """Set previous season ratings, HCA, and SRS."""
        prev = season - 1

        for team_id in self.team_games:
            if prev in self.team_games[team_id]:
                td = self.team_games[team_id][prev]
                if td['ppg']:
                    self.prev_ratings[team_id] = {
                        'ppg': np.mean(td['ppg']),
                        'papg': np.mean(td['papg']),
                        'yards': np.mean(td['yards']) if td['yards'] else self.league_avg['yards'],
                        'pass_yards': np.mean(td['pass_yards']) if td['pass_yards'] else self.league_avg['pass_yards'],
                        'rush_yards': np.mean(td['rush_yards']) if td['rush_yards'] else self.league_avg['rush_yards'],
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else self.league_avg['turnovers'],
                        'first_downs': np.mean(td['first_downs']) if td['first_downs'] else self.league_avg['first_downs'],
                        'scoring_pct': np.mean(td['scoring_pct']) if td['scoring_pct'] else self.league_avg['scoring_pct'],
                        'ypd': np.mean(td['ypd']) if td['ypd'] else self.league_avg['ypd'],
                    }

        # Previous season HCA
        for team_id in self.team_hca_data:
            if prev in self.team_hca_data[team_id]:
                hd = self.team_hca_data[team_id][prev]
                if hd['home_margins'] and hd['away_margins']:
                    raw_hca = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
                    raw_hca = max(-3, min(raw_hca, 10))
                    self.prev_hca[team_id] = BASE_HCA + HCA_SHRINK * (raw_hca - BASE_HCA)

        # Previous season SRS
        self.prev_srs = self._calculate_srs(prev)

        self.last_game.clear()

    # -- Prediction -----------------------------------------------------------

    def predict(self, home_id: int, away_id: int, season: int,
                game_date: str, week: int = 7,
                neutral_site: bool = False, is_conf_game: bool = False,
                vegas_spread: float = None, vegas_total: float = None) -> dict:
        """Make pure model predictions. Uses score model (primary) or spread/total (fallback)."""
        if self.score_model is None and self.spread_model is None:
            raise ValueError("Model not trained -- call train() first")

        # Recalculate SRS for current predictions
        current_srs = self._calculate_srs(season)
        if current_srs:
            self.team_srs[season] = current_srs

        home_score = None
        away_score = None
        spread = None
        total = None

        # Primary: score-based model (predicts individual team scores)
        if self.score_model is not None and self.score_scaler is not None:
            home_feat = self.extract_score_features(
                home_id, away_id, season, game_date, week,
                is_home=not neutral_site, neutral=neutral_site
            )
            away_feat = self.extract_score_features(
                away_id, home_id, season, game_date, week,
                is_home=False, neutral=neutral_site
            )
            if home_feat is not None and away_feat is not None:
                home_scaled = self.score_scaler.transform(home_feat.reshape(1, -1))
                away_scaled = self.score_scaler.transform(away_feat.reshape(1, -1))
                home_score = max(0, float(self.score_model.predict(home_scaled)[0]))
                away_score = max(0, float(self.score_model.predict(away_scaled)[0]))
                spread = away_score - home_score
                total = home_score + away_score

        # Fallback: spread/total models (backward compat)
        if spread is None and self.spread_model is not None:
            spread_features = self.extract_spread_features(
                home_id, away_id, season, game_date, week,
                neutral_site=neutral_site, is_conf_game=is_conf_game
            )
            if spread_features is not None:
                spread_scaled = self.spread_scaler.transform(spread_features.reshape(1, -1))
                spread = float(self.spread_model.predict(spread_scaled)[0])

        if total is None and self.total_model is not None:
            total_features = self.extract_total_features(
                home_id, away_id, season, game_date, week
            )
            if total_features is not None:
                total_scaled = self.total_scaler.transform(total_features.reshape(1, -1))
                total = float(self.total_model.predict(total_scaled)[0])

        # Derive scores from spread+total if score model didn't produce them
        if home_score is None and spread is not None and total is not None:
            home_score = (total - spread) / 2
            away_score = (total + spread) / 2

        # SRS diff and HCA for diagnostics
        srs_diff = None
        hca = None
        home_srs = self._get_team_srs(home_id, season)
        away_srs = self._get_team_srs(away_id, season)
        srs_diff = home_srs - away_srs
        hca = 0.0 if neutral_site else self._get_dynamic_hca(home_id, season)

        return {
            'predicted_spread': spread,
            'predicted_total': total,
            'home_score': home_score,
            'away_score': away_score,
            'srs_diff': srs_diff,
            'dynamic_hca': hca,
        }

    # -- Training -------------------------------------------------------------

    def train(self, db_path: Path = DB_PATH):
        """Train Ridge V2 model with walk-forward validation."""
        log.info("=" * 70)
        log.info("CFB RIDGE V2 TRAINING")
        log.info("SRS ratings + real drive data + walk-forward validation")
        log.info("=" * 70)

        conn = sqlite3.connect(str(db_path))

        # Load games with box scores (regular season only)
        games = pd.read_sql_query('''
            SELECT g.game_id, g.season, g.week, g.date, g.game_date_eastern,
                   g.home_team_id, g.away_team_id,
                   g.home_score, g.away_score, g.neutral_site, g.conference_game,
                   hs.total_yards as home_yards, hs.passing_yards as home_pass_yards,
                   hs.rushing_yards as home_rush_yards, hs.turnovers as home_to,
                   hs.first_downs as home_fd,
                   aws.total_yards as away_yards, aws.passing_yards as away_pass_yards,
                   aws.rushing_yards as away_rush_yards, aws.turnovers as away_to,
                   aws.first_downs as away_fd,
                   o.latest_spread as vegas_spread, o.latest_total as vegas_total
            FROM games g
            LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id
                AND g.home_team_id = hs.team_id
            LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id
                AND g.away_team_id = aws.team_id
            LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.completed = 1 AND g.home_score IS NOT NULL
              AND g.postseason_type IS NULL
            ORDER BY g.date, g.week
        ''', conn)

        # Load real drive data, deduplicated
        drives = pd.read_sql_query('''
            SELECT DISTINCT d.game_id, d.team_id, d.yards, d.is_score
            FROM drives d
            JOIN games g ON d.game_id = g.game_id
            WHERE g.completed = 1
        ''', conn)
        conn.close()

        # Calculate per-team-per-game drive efficiency
        if not drives.empty:
            drive_stats = drives.groupby(['game_id', 'team_id']).agg(
                total_yards=('yards', 'sum'),
                num_drives=('yards', 'count'),
                scores=('is_score', 'sum'),
            ).reset_index()
            drive_stats['ypd'] = drive_stats['total_yards'] / drive_stats['num_drives'].replace(0, 1)
            drive_stats['scoring_pct'] = drive_stats['scores'] / drive_stats['num_drives'].replace(0, 1)

            # Merge home drives
            home_drives = drive_stats.rename(columns={
                'ypd': 'home_ypd', 'scoring_pct': 'home_scoring_pct'
            })
            games = games.merge(
                home_drives[['game_id', 'team_id', 'home_ypd', 'home_scoring_pct']],
                left_on=['game_id', 'home_team_id'],
                right_on=['game_id', 'team_id'],
                how='left'
            ).drop(columns=['team_id'], errors='ignore')

            # Merge away drives
            away_drives = drive_stats.rename(columns={
                'ypd': 'away_ypd', 'scoring_pct': 'away_scoring_pct'
            })
            games = games.merge(
                away_drives[['game_id', 'team_id', 'away_ypd', 'away_scoring_pct']],
                left_on=['game_id', 'away_team_id'],
                right_on=['game_id', 'team_id'],
                how='left'
            ).drop(columns=['team_id'], errors='ignore')

        log.info(f"Total games: {len(games)}")
        log.info(f"Games with drive data: {games.get('home_ypd', pd.Series()).notna().sum()}")
        log.info(f"Games with Vegas spread: {games['vegas_spread'].notna().sum()}")
        log.info(f"Games with Vegas total: {games['vegas_total'].notna().sum()}")

        # -- Walk-forward training --------------------------------------------
        X_spread, y_spread = [], []
        X_total, y_total = [], []
        # Score-based model data (2 rows per game: home→home_score, away→away_score)
        X_score, y_score = [], []
        score_game_idx = []  # Maps score rows back to game index
        score_game_counter = 0
        score_seasons = []  # Season per game (not per row)
        score_vegas_s, score_vegas_t = [], []
        score_actual_spread, score_actual_total = [], []
        vegas_spreads_list = []
        vegas_totals_list = []
        total_seasons = []
        game_seasons = []
        game_weeks = []
        game_neutral = []
        game_home_fav = []
        game_conf = []

        seasons = sorted(games['season'].unique())
        log.info(f"Seasons: {seasons}")

        for season in seasons:
            if season > seasons[0]:
                self.set_previous_season(season)
                prev_games = games[games['season'] == season - 1]
                if len(prev_games) > 0:
                    self.league_avg['ppg'] = prev_games['home_score'].mean()
                    self.league_avg['papg'] = prev_games['away_score'].mean()

            season_games = games[games['season'] == season]
            games_processed = 0

            for _, g in season_games.iterrows():
                hid = g['home_team_id']
                aid = g['away_team_id']
                actual_spread = g['away_score'] - g['home_score']
                actual_total = g['home_score'] + g['away_score']
                neutral = g['neutral_site'] == 1
                is_conf = g['conference_game'] == 1
                week = g['week']

                # Recalculate SRS periodically
                if games_processed > 0 and games_processed % SRS_RECALC_INTERVAL == 0:
                    self.team_srs[season] = self._calculate_srs(season)

                # Extract features (only if teams have enough games)
                home_games = self.team_games[hid][season]['game_count']
                away_games = self.team_games[aid][season]['game_count']

                if home_games >= MIN_GAMES and away_games >= MIN_GAMES:
                    spread_feat = self.extract_spread_features(
                        hid, aid, season, g['date'], week,
                        neutral_site=neutral, is_conf_game=is_conf
                    )
                    total_feat = self.extract_total_features(
                        hid, aid, season, g['date'], week
                    )

                    if spread_feat is not None:
                        X_spread.append(spread_feat)
                        y_spread.append(actual_spread)
                        game_seasons.append(season)
                        game_weeks.append(week)
                        game_neutral.append(neutral)
                        game_conf.append(is_conf)
                        vegas_spreads_list.append(
                            g['vegas_spread'] if pd.notna(g['vegas_spread']) else np.nan
                        )
                        game_home_fav.append(
                            1 if pd.notna(g['vegas_spread']) and g['vegas_spread'] < 0 else 0
                        )

                    if total_feat is not None:
                        X_total.append(total_feat)
                        y_total.append(actual_total)
                        vegas_totals_list.append(
                            g['vegas_total'] if pd.notna(g['vegas_total']) else np.nan
                        )
                        total_seasons.append(season)

                    # Score-based features (2 rows per game)
                    home_score_feat = self.extract_score_features(
                        hid, aid, season, g['date'], week,
                        is_home=not neutral, neutral=neutral
                    )
                    away_score_feat = self.extract_score_features(
                        aid, hid, season, g['date'], week,
                        is_home=False, neutral=neutral
                    )
                    if home_score_feat is not None and away_score_feat is not None:
                        vs = g['vegas_spread'] if pd.notna(g['vegas_spread']) else np.nan
                        vt = g['vegas_total'] if pd.notna(g['vegas_total']) else np.nan
                        # Home row
                        X_score.append(home_score_feat)
                        y_score.append(g['home_score'])
                        score_game_idx.append(score_game_counter)
                        score_seasons.append(season)
                        score_vegas_s.append(vs)
                        score_vegas_t.append(vt)
                        score_actual_spread.append(actual_spread)
                        score_actual_total.append(actual_total)
                        # Away row
                        X_score.append(away_score_feat)
                        y_score.append(g['away_score'])
                        score_game_idx.append(score_game_counter)
                        score_seasons.append(season)
                        score_vegas_s.append(vs)
                        score_vegas_t.append(vt)
                        score_actual_spread.append(actual_spread)
                        score_actual_total.append(actual_total)
                        score_game_counter += 1

                # Update team states
                self.update_team(
                    hid, aid, season, g['date'],
                    g['home_score'], g['away_score'], is_home=not neutral,
                    yards=g['home_yards'], pass_yards=g['home_pass_yards'],
                    rush_yards=g['home_rush_yards'], turnovers=g['home_to'],
                    first_downs=g['home_fd'],
                    scoring_pct=g.get('home_scoring_pct'), ypd=g.get('home_ypd')
                )
                self.update_team(
                    aid, hid, season, g['date'],
                    g['away_score'], g['home_score'], is_home=False,
                    yards=g['away_yards'], pass_yards=g['away_pass_yards'],
                    rush_yards=g['away_rush_yards'], turnovers=g['away_to'],
                    first_downs=g['away_fd'],
                    scoring_pct=g.get('away_scoring_pct'), ypd=g.get('away_ypd')
                )
                games_processed += 1

            # Final SRS for season
            self.team_srs[season] = self._calculate_srs(season)

        # Convert arrays
        X_spread = np.array(X_spread)
        y_spread = np.array(y_spread)
        vegas_spreads_arr = np.array(vegas_spreads_list)
        game_seasons = np.array(game_seasons)
        game_weeks = np.array(game_weeks)
        game_neutral = np.array(game_neutral)
        game_home_fav = np.array(game_home_fav)
        game_conf = np.array(game_conf)

        X_total = np.array(X_total)
        y_total = np.array(y_total)
        vegas_totals_arr = np.array(vegas_totals_list)
        total_seasons_arr = np.array(total_seasons)

        # Drop NaN rows
        spread_nan = np.isnan(X_spread).any(axis=1) | np.isnan(y_spread)
        log.info(f"Dropping {spread_nan.sum()} spread rows with NaN features")
        X_spread = X_spread[~spread_nan]
        y_spread = y_spread[~spread_nan]
        vegas_spreads_arr = vegas_spreads_arr[~spread_nan]
        game_seasons = game_seasons[~spread_nan]
        game_weeks = game_weeks[~spread_nan]
        game_neutral = game_neutral[~spread_nan]
        game_home_fav = game_home_fav[~spread_nan]
        game_conf = game_conf[~spread_nan]

        total_nan = np.isnan(X_total).any(axis=1) | np.isnan(y_total)
        log.info(f"Dropping {total_nan.sum()} total rows with NaN features")
        X_total = X_total[~total_nan]
        y_total = y_total[~total_nan]
        vegas_totals_arr = vegas_totals_arr[~total_nan]
        total_seasons_arr = total_seasons_arr[~total_nan]

        # Convert score arrays
        X_score = np.array(X_score)
        y_score = np.array(y_score)
        score_game_idx = np.array(score_game_idx)
        score_seasons = np.array(score_seasons)
        score_vegas_s = np.array(score_vegas_s)
        score_vegas_t = np.array(score_vegas_t)
        score_actual_spread = np.array(score_actual_spread)
        score_actual_total = np.array(score_actual_total)

        # Drop NaN score rows
        if len(X_score) > 0:
            score_nan = np.isnan(X_score).any(axis=1) | np.isnan(y_score)
            log.info(f"Dropping {score_nan.sum()} score rows with NaN features")
            X_score = X_score[~score_nan]
            y_score = y_score[~score_nan]
            score_game_idx = score_game_idx[~score_nan]
            score_seasons = score_seasons[~score_nan]
            score_vegas_s = score_vegas_s[~score_nan]
            score_vegas_t = score_vegas_t[~score_nan]
            score_actual_spread = score_actual_spread[~score_nan]
            score_actual_total = score_actual_total[~score_nan]

        log.info(f"\nSpread samples: {len(X_spread)}, Total samples: {len(X_total)}")
        log.info(f"Score samples: {len(X_score)} (from {score_game_counter} games)")

        # -- Walk-forward: train on first 2 seasons, test on last 2 ----------
        test_seasons = sorted(set(game_seasons))[-2:]

        # Spread split
        train_mask = ~np.isin(game_seasons, test_seasons)
        test_mask = np.isin(game_seasons, test_seasons)

        X_train_s = X_spread[train_mask]
        y_train_s = y_spread[train_mask]
        X_test_s = X_spread[test_mask]
        y_test_s = y_spread[test_mask]

        vegas_test_s = vegas_spreads_arr[test_mask]
        test_weeks = game_weeks[test_mask]
        test_seasons_arr = game_seasons[test_mask]
        test_neutral = game_neutral[test_mask]
        test_home_fav = game_home_fav[test_mask]
        test_conf = game_conf[test_mask]

        # Total split
        total_test_seasons = sorted(set(total_seasons_arr))[-2:]
        total_train_mask = ~np.isin(total_seasons_arr, total_test_seasons)
        total_test_mask = np.isin(total_seasons_arr, total_test_seasons)

        X_train_t = X_total[total_train_mask]
        y_train_t = y_total[total_train_mask]
        X_test_t = X_total[total_test_mask]
        y_test_t = y_total[total_test_mask]

        vegas_test_t = vegas_totals_arr[total_test_mask]
        total_test_seasons_arr = total_seasons_arr[total_test_mask]

        log.info(f"Spread -- Train: {len(X_train_s)}, Test: {len(X_test_s)} (seasons {test_seasons})")
        log.info(f"Total  -- Train: {len(X_train_t)}, Test: {len(X_test_t)} (seasons {total_test_seasons})")

        # -- Train models -----------------------------------------------------
        self.spread_scaler = StandardScaler()
        X_train_s_scaled = self.spread_scaler.fit_transform(X_train_s)
        X_test_s_scaled = self.spread_scaler.transform(X_test_s)

        self.spread_model = Ridge(alpha=1.0)
        self.spread_model.fit(X_train_s_scaled, y_train_s)

        self.total_scaler = StandardScaler()
        X_train_t_scaled = self.total_scaler.fit_transform(X_train_t)
        X_test_t_scaled = self.total_scaler.transform(X_test_t)

        self.total_model = Ridge(alpha=1.0)
        self.total_model.fit(X_train_t_scaled, y_train_t)

        # -- Train score model ------------------------------------------------
        # All score arrays are per-row (2 per game). Use row-level season mask.
        score_test_seasons_list = sorted(set(score_seasons))[-2:]
        score_train_rows = ~np.isin(score_seasons, score_test_seasons_list)
        score_test_rows = np.isin(score_seasons, score_test_seasons_list)

        self.score_scaler = StandardScaler()
        X_train_sc = self.score_scaler.fit_transform(X_score[score_train_rows])
        X_test_sc = self.score_scaler.transform(X_score[score_test_rows])

        self.score_model = Ridge(alpha=1.0)
        self.score_model.fit(X_train_sc, y_score[score_train_rows])

        log.info(f"\nScore -- Train: {score_train_rows.sum()} rows, "
                 f"Test: {score_test_rows.sum()} rows (seasons {score_test_seasons_list})")

        # Map score predictions back to game-level spread/total
        score_preds = self.score_model.predict(X_test_sc)
        test_game_idx_arr = score_game_idx[score_test_rows]
        unique_test_games = np.unique(test_game_idx_arr)
        n_test_games = len(unique_test_games)

        score_spread_pred = np.zeros(n_test_games)
        score_total_pred = np.zeros(n_test_games)
        game_actual_s = np.zeros(n_test_games)
        game_actual_t = np.zeros(n_test_games)
        game_vegas_s = np.full(n_test_games, np.nan)
        game_vegas_t = np.full(n_test_games, np.nan)

        # Per-row test arrays for indexing
        test_actual_s = score_actual_spread[score_test_rows]
        test_actual_t = score_actual_total[score_test_rows]
        test_vs = score_vegas_s[score_test_rows]
        test_vt = score_vegas_t[score_test_rows]

        for i, gidx in enumerate(unique_test_games):
            row_indices = np.where(test_game_idx_arr == gidx)[0]
            if len(row_indices) == 2:
                home_pred = max(0, score_preds[row_indices[0]])
                away_pred = max(0, score_preds[row_indices[1]])
                score_spread_pred[i] = away_pred - home_pred
                score_total_pred[i] = home_pred + away_pred
            # Actuals/vegas are same for both rows of a game — take first
            game_actual_s[i] = test_actual_s[row_indices[0]]
            game_actual_t[i] = test_actual_t[row_indices[0]]
            game_vegas_s[i] = test_vs[row_indices[0]]
            game_vegas_t[i] = test_vt[row_indices[0]]

        # Score model evaluation (game-level)
        score_actual_s = game_actual_s
        score_actual_t = game_actual_t
        score_vs = game_vegas_s
        score_vt = game_vegas_t

        score_spread_mae = np.abs(score_spread_pred - score_actual_s).mean()
        score_total_mae = np.abs(score_total_pred - score_actual_t).mean()

        log.info("\n" + "=" * 70)
        log.info("SCORE MODEL RESULTS")
        log.info("=" * 70)
        log.info(f"Score-based Spread MAE: {score_spread_mae:.2f}")
        log.info(f"Score-based Total MAE:  {score_total_mae:.2f}")

        has_score_vs = ~np.isnan(score_vs)
        has_score_vt = ~np.isnan(score_vt)
        if has_score_vs.sum() > 0:
            vegas_s_mae = np.abs(score_vs[has_score_vs] - score_actual_s[has_score_vs]).mean()
            log.info(f"Vegas Spread MAE:       {vegas_s_mae:.2f}")
        if has_score_vt.sum() > 0:
            vegas_t_mae = np.abs(score_vt[has_score_vt] - score_actual_t[has_score_vt]).mean()
            log.info(f"Vegas Total MAE:        {vegas_t_mae:.2f}")

        # Score model coefficients
        score_feat_names = [
            'team_ppg', 'team_yards', 'team_pass_yds', 'team_rush_yds',
            'team_turnovers', 'team_first_downs',
            'team_scoring_pct', 'team_ypd',
            'opp_papg', 'opp_turnovers', 'opp_scoring_pct', 'opp_ypd',
            'team_srs', 'opp_srs',
            'team_form', 'team_momentum', 'team_streak',
            'team_rest', 'is_home_hca', 'season_progress',
        ]
        log.info("\nScore Model Coefficients:")
        coefs = list(zip(score_feat_names, self.score_model.coef_))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, coef in coefs:
            log.info(f"  {name:<22} {coef:+.4f}")
        log.info(f"  Intercept: {self.score_model.intercept_:+.4f}")

        # -- Evaluate spread/total models (existing) -------------------------
        model_spread = self.spread_model.predict(X_test_s_scaled)
        model_total = self.total_model.predict(X_test_t_scaled)

        self._print_evaluation(
            model_spread, y_test_s, vegas_test_s,
            model_total, y_test_t, vegas_test_t,
            test_weeks, test_seasons_arr, test_neutral, test_home_fav,
            test_conf, total_test_seasons_arr, X_test_s
        )

        return {
            'spread_mae': score_spread_mae,
            'vegas_spread_mae': np.nanmean(np.abs(score_vs[has_score_vs] - score_actual_s[has_score_vs])) if has_score_vs.sum() > 0 else np.nan,
            'total_mae': score_total_mae,
            'vegas_total_mae': np.nanmean(np.abs(score_vt[has_score_vt] - score_actual_t[has_score_vt])) if has_score_vt.sum() > 0 else np.nan,
            'diff_spread_mae': np.abs(model_spread - y_test_s).mean(),
            'diff_total_mae': np.abs(model_total - y_test_t).mean(),
        }

    def _print_evaluation(self, model_spread, y_test_s, vegas_test_s,
                          model_total, y_test_t, vegas_test_t,
                          test_weeks, test_seasons, test_neutral, test_home_fav,
                          test_conf, total_test_seasons, X_test_s,
                          db_path: Path = DB_PATH):
        """Print comprehensive backtest results."""
        has_vegas_s = ~np.isnan(vegas_test_s)
        has_vegas_t = ~np.isnan(vegas_test_t)

        # -- MAE Comparison ---------------------------------------------------
        log.info("\n" + "=" * 70)
        log.info("SPREAD RESULTS")
        log.info("=" * 70)

        model_mae = np.abs(model_spread - y_test_s).mean()
        vegas_mae = np.abs(vegas_test_s[has_vegas_s] - y_test_s[has_vegas_s]).mean()

        log.info(f"\n{'Model':<30} {'MAE':<10}")
        log.info("-" * 45)
        log.info(f"{'Ridge V2':<30} {model_mae:.2f}")
        log.info(f"{'Vegas':<30} {vegas_mae:.2f}")
        log.info(f"{'Gap':<30} {model_mae - vegas_mae:+.2f}")

        # -- ATS Analysis (spreads) -------------------------------------------
        log.info("\n" + "=" * 70)
        log.info("ATS ANALYSIS (vs Vegas Spread)")
        log.info("=" * 70)

        # Only analyze games with Vegas lines
        ms = model_spread[has_vegas_s]
        vs = vegas_test_s[has_vegas_s]
        ys = y_test_s[has_vegas_s]
        weeks_v = test_weeks[has_vegas_s]
        seasons_v = test_seasons[has_vegas_s]
        neutral_v = test_neutral[has_vegas_s]
        home_fav_v = test_home_fav[has_vegas_s]
        conf_v = test_conf[has_vegas_s]

        def calc_ats(pred, vegas, actual, mask=None):
            if mask is None:
                mask = np.ones(len(pred), dtype=bool)
            edge = pred[mask] - vegas[mask]
            result = actual[mask] - vegas[mask]
            push = np.abs(result) < 0.5
            wins = ((edge > 0) & (result > 0)) | ((edge < 0) & (result < 0))
            wins = wins & ~push
            valid = ~push
            return int(wins.sum()), int(valid.sum())

        log.info(f"\n{'Threshold':<20} {'Record':<15} {'Win %':<10} {'ROI':<10} {'P-value':<10} {'Games'}")
        log.info("-" * 75)

        for thresh in [0, 3, 5, 7]:
            edge = ms - vs
            mask = np.abs(edge) >= thresh
            wins, total = calc_ats(ms, vs, ys, mask)
            if total > 0:
                pct = wins / total * 100
                roi = (wins * 0.91 - (total - wins)) / total * 100
                # Binomial test p-value
                p_val = stats.binomtest(wins, total, 0.5, alternative='greater').pvalue if wins > total / 2 else 1.0
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                log.info(f"{thresh}+ pt edge{'':<10} {wins}-{total - wins:<10} {pct:.1f}%{'':<5} {roi:+.1f}%{'':<5} {p_val:.3f}{sig:<5} {total}")

        # -- By Pick Type -----------------------------------------------------
        log.info("\n--- BY PICK TYPE ---")
        edge = ms - vs

        home_pick = edge < 0
        if home_pick.sum() > 0:
            w, t = calc_ats(ms, vs, ys, home_pick)
            log.info(f"Home picks:       {w}-{t - w} ({w / t * 100:.1f}%) n={t}")

        away_pick = edge > 0
        if away_pick.sum() > 0:
            w, t = calc_ats(ms, vs, ys, away_pick)
            log.info(f"Away picks:       {w}-{t - w} ({w / t * 100:.1f}%) n={t}")

        road_fav_pick = (edge > 0) & (home_fav_v == 0)
        if road_fav_pick.sum() > 0:
            w, t = calc_ats(ms, vs, ys, road_fav_pick)
            log.info(f"Road fav picks:   {w}-{t - w} ({w / t * 100:.1f}%) n={t}")

        home_dog_pick = (edge < 0) & (home_fav_v == 0)
        if home_dog_pick.sum() > 0:
            w, t = calc_ats(ms, vs, ys, home_dog_pick)
            log.info(f"Home dog picks:   {w}-{t - w} ({w / t * 100:.1f}%) n={t}")

        home_fav_pick = (edge < 0) & (home_fav_v == 1)
        if home_fav_pick.sum() > 0:
            w, t = calc_ats(ms, vs, ys, home_fav_pick)
            log.info(f"Home fav picks:   {w}-{t - w} ({w / t * 100:.1f}%) n={t}")

        # -- Conference vs Non-Conference -------------------------------------
        log.info("\n--- CONFERENCE vs NON-CONFERENCE ---")
        for thresh in [0, 3, 5]:
            for conf_val, label in [(True, "Conf"), (False, "Non-Conf")]:
                mask = (conf_v == conf_val) & (np.abs(edge) >= thresh)
                if mask.sum() > 0:
                    w, t = calc_ats(ms, vs, ys, mask)
                    if t > 0:
                        log.info(f"  {label} {thresh}+ edge: {w}-{t - w} ({w / t * 100:.1f}%) n={t}")

        # -- By Week Range ----------------------------------------------------
        log.info("\n--- BY WEEK RANGE ---")
        for start, end in [(1, 4), (5, 8), (9, 14)]:
            mask = (weeks_v >= start) & (weeks_v <= end) & (np.abs(edge) >= 3)
            if mask.sum() > 0:
                w, t = calc_ats(ms, vs, ys, mask)
                if t > 0:
                    log.info(f"Weeks {start}-{end} (3+ edge): {w}-{t - w} ({w / t * 100:.1f}%) n={t}")

        # -- By Season (CRITICAL for honest evaluation) -----------------------
        log.info("\n--- BY SEASON (CRITICAL) ---")
        for season in sorted(set(seasons_v)):
            for thresh in [0, 3, 5, 7]:
                mask = (seasons_v == season) & (np.abs(edge) >= thresh)
                if mask.sum() > 0:
                    w, t = calc_ats(ms, vs, ys, mask)
                    if t > 0:
                        s_mae = np.abs(ms[seasons_v == season] - ys[seasons_v == season]).mean()
                        v_mae = np.abs(vs[seasons_v == season] - ys[seasons_v == season]).mean()
                        log.info(f"  {season} {thresh}+ edge: {w}-{t - w} ({w / t * 100:.1f}%) n={t}"
                                 f"  MAE: {s_mae:.2f} (Vegas: {v_mae:.2f})")

        # -- Totals Analysis --------------------------------------------------
        log.info("\n" + "=" * 70)
        log.info("TOTALS RESULTS")
        log.info("=" * 70)

        mt = model_total[has_vegas_t]
        vt = vegas_test_t[has_vegas_t]
        yt = y_test_t[has_vegas_t]
        total_seasons_v = total_test_seasons[has_vegas_t]

        total_mae = np.abs(mt - yt).mean()
        vegas_total_mae = np.abs(vt - yt).mean()

        log.info(f"\n{'Model':<30} {'MAE':<10}")
        log.info("-" * 45)
        log.info(f"{'Ridge V2 Total':<30} {total_mae:.2f}")
        log.info(f"{'Vegas Total':<30} {vegas_total_mae:.2f}")

        # O/U analysis
        log.info("\n--- OVER/UNDER ANALYSIS ---")
        total_edge = mt - vt

        for thresh in [0, 3, 5, 7]:
            over_mask = total_edge >= thresh
            if over_mask.sum() > 0:
                over_correct = (yt[over_mask] > vt[over_mask]).sum()
                over_total = over_mask.sum()
                log.info(f"OVER  {thresh}+ edge: {over_correct}-{over_total - over_correct} "
                         f"({over_correct / over_total * 100:.1f}%) n={over_total}")

            under_mask = total_edge <= -thresh
            if under_mask.sum() > 0:
                under_correct = (yt[under_mask] < vt[under_mask]).sum()
                under_total = under_mask.sum()
                log.info(f"UNDER {thresh}+ edge: {under_correct}-{under_total - under_correct} "
                         f"({under_correct / under_total * 100:.1f}%) n={under_total}")

        # O/U by season
        log.info("\n--- O/U BY SEASON ---")
        for season in sorted(set(total_seasons_v)):
            s_mask = total_seasons_v == season
            if s_mask.sum() > 0:
                s_edge = mt[s_mask] - vt[s_mask]
                for thresh in [0, 3, 5]:
                    over_m = s_edge >= thresh
                    if over_m.sum() > 0:
                        correct = (yt[s_mask][over_m] > vt[s_mask][over_m]).sum()
                        n = over_m.sum()
                        log.info(f"  {season} OVER {thresh}+: {correct}-{n - correct} ({correct / n * 100:.1f}%) n={n}")
                    under_m = s_edge <= -thresh
                    if under_m.sum() > 0:
                        correct = (yt[s_mask][under_m] < vt[s_mask][under_m]).sum()
                        n = under_m.sum()
                        log.info(f"  {season} UNDER {thresh}+: {correct}-{n - correct} ({correct / n * 100:.1f}%) n={n}")

        # Bias
        model_bias = np.mean(mt - yt)
        log.info(f"\nModel total bias: {model_bias:+.2f} pts (positive = OVER bias)")

        # -- Coefficients -----------------------------------------------------
        log.info("\n" + "=" * 70)
        log.info("SPREAD MODEL COEFFICIENTS")
        log.info("=" * 70)

        spread_feature_names = [
            'PPG diff', 'PAPG diff', 'SRS diff',
            'Yards diff', 'Pass yards diff', 'Rush yards diff',
            'Turnover diff', 'First downs diff',
            'Scoring % diff', 'YPD diff',
            'Recent form diff', 'Momentum diff', 'Streak diff',
            'Rest diff', 'Post-bye diff',
            'Dynamic HCA', 'Conference game', 'Season progress',
            'Home reliability', 'Away reliability',
        ]

        coefs = list(zip(spread_feature_names, self.spread_model.coef_))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, coef in coefs:
            log.info(f"  {name:<22} {coef:+.4f}")

        log.info(f"\n  Intercept: {self.spread_model.intercept_:+.4f}")

        # -- Top/Bottom SRS Teams ---------------------------------------------
        log.info("\n" + "=" * 70)
        log.info("SRS RATINGS (Last Season)")
        log.info("=" * 70)

        last_season = max(s for s in self.team_srs if self.team_srs[s])
        if last_season:
            srs = self.team_srs[last_season]
            if srs:
                conn = sqlite3.connect(str(db_path))
                teams_df = pd.read_sql_query(
                    "SELECT team_id, display_name FROM teams", conn
                )
                conn.close()
                team_names = dict(zip(teams_df['team_id'], teams_df['display_name']))

                sorted_srs = sorted(srs.items(), key=lambda x: x[1], reverse=True)
                log.info(f"\nTop 15 (Season {last_season}):")
                for tid, rating in sorted_srs[:15]:
                    name = team_names.get(tid, f"Team {tid}")
                    log.info(f"  {name:<25} {rating:+.1f}")
                log.info(f"\nBottom 10:")
                for tid, rating in sorted_srs[-10:]:
                    name = team_names.get(tid, f"Team {tid}")
                    log.info(f"  {name:<25} {rating:+.1f}")

    # -- Save/Load ------------------------------------------------------------

    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODEL_DIR / 'cfb_ridge_v2.pkl'

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert defaultdicts to regular dicts for pickling
        team_games_dict = {}
        for team_id, seasons in self.team_games.items():
            team_games_dict[team_id] = {}
            for season, team_stats in seasons.items():
                team_games_dict[team_id][season] = dict(team_stats)

        hca_data_dict = {}
        for team_id, seasons in self.team_hca_data.items():
            hca_data_dict[team_id] = {}
            for season, data in seasons.items():
                hca_data_dict[team_id][season] = dict(data)

        srs_dict = {}
        for season, ratings in self.team_srs.items():
            if isinstance(ratings, dict):
                srs_dict[season] = dict(ratings)

        model_data = {
            'spread_model': self.spread_model,
            'total_model': self.total_model,
            'spread_scaler': self.spread_scaler,
            'total_scaler': self.total_scaler,
            'score_model': self.score_model,
            'score_scaler': self.score_scaler,
            'team_games': team_games_dict,
            'team_hca_data': hca_data_dict,
            'team_srs': srs_dict,
            'prev_ratings': self.prev_ratings,
            'prev_hca': self.prev_hca,
            'prev_srs': self.prev_srs,
            'last_game': self.last_game,
            'league_avg': self.league_avg,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        log.info(f"\nModel saved to: {path}")

    @classmethod
    def load(cls, path: Path = None) -> CFBRidgeV2:
        """Load model from disk."""
        if path is None:
            path = MODEL_DIR / 'cfb_ridge_v2.pkl'

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        model = cls()
        model.spread_model = model_data['spread_model']
        model.total_model = model_data['total_model']
        model.spread_scaler = model_data['spread_scaler']
        model.total_scaler = model_data['total_scaler']
        model.score_model = model_data.get('score_model')
        model.score_scaler = model_data.get('score_scaler')

        # Restore team state with proper defaultdict factories
        model.team_games = defaultdict(
            lambda: defaultdict(lambda: {
                'ppg': [], 'ppg_wts': [], 'papg': [], 'papg_wts': [],
                'yards': [], 'yards_wts': [], 'pass_yards': [], 'pass_wts': [],
                'rush_yards': [], 'rush_wts': [], 'turnovers': [], 'to_wts': [],
                'first_downs': [], 'fd_wts': [],
                'scoring_pct': [], 'sp_wts': [], 'ypd': [], 'ypd_wts': [],
                'margins': [], 'wins': [], 'opponents': [], 'game_count': 0,
            }),
            model_data['team_games']
        )
        model.team_hca_data = defaultdict(
            lambda: defaultdict(lambda: {'home_margins': [], 'away_margins': []}),
            model_data.get('team_hca_data', {})
        )
        model.team_srs = defaultdict(
            lambda: defaultdict(float),
            model_data.get('team_srs', {})
        )
        model.prev_ratings = model_data['prev_ratings']
        model.prev_hca = model_data.get('prev_hca', {})
        model.prev_srs = model_data.get('prev_srs', {})
        model.last_game = model_data['last_game']
        model.league_avg = model_data['league_avg']

        return model


def main():
    """Train and save the CFB Ridge V2 model."""
    model = CFBRidgeV2()
    results = model.train()
    model.save()

    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"Score Model Spread MAE:  {results['spread_mae']:.2f}")
    log.info(f"Score Model Total MAE:   {results['total_mae']:.2f}")
    log.info(f"Diff Model Spread MAE:   {results['diff_spread_mae']:.2f}")
    log.info(f"Diff Model Total MAE:    {results['diff_total_mae']:.2f}")
    vegas_s = results.get('vegas_spread_mae')
    vegas_t = results.get('vegas_total_mae')
    if vegas_s is not None and not np.isnan(vegas_s):
        log.info(f"Vegas Spread MAE:        {vegas_s:.2f}")
    if vegas_t is not None and not np.isnan(vegas_t):
        log.info(f"Vegas Total MAE:         {vegas_t:.2f}")


if __name__ == '__main__':
    main()
