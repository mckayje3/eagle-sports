"""
NBA Ridge Model V2 - Pure Model (No Vegas Blend)

Key changes from V1:
1. NO Vegas spread blending - pure model predictions
2. SRS-style opponent-adjusted ratings
3. Reduced HCA (1.5 base instead of 2.2)
4. Road favorite penalty
5. Reduced recent form influence
6. Removed collinear Net Rating feature

Architecture: Ridge creates baseline, Deep Eagle calibrates edges.
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


class NBARidgeV2:
    """
    NBA Ridge Model V2 - Pure predictions without Vegas blend.

    Key features:
    1. SRS-style opponent-adjusted ratings
    2. Reduced home court advantage (1.5 pts base)
    3. Road favorite penalty (-1.5 pts)
    4. Dampened recent form (30% weight)
    """

    # NO Vegas blending - pure model
    SPREAD_MODEL_WEIGHT = 1.0
    TOTAL_MODEL_WEIGHT = 1.0

    # Reduced HCA parameters
    DECAY = 0.93
    PREV_HALF_LIFE = 6.0
    HCA_HALF_LIFE = 40.0  # Slower transition (was 30)
    HCA_SHRINK = 0.7  # More shrinkage toward base (was 0.5)
    BASE_HCA = 1.5  # Reduced from 2.2
    B2B_PENALTY = 1.5  # Increased from 1.0

    # Road favorite penalty
    ROAD_FAV_PENALTY = 1.5  # Points to add when model picks road fav

    # Dampened form
    FORM_WEIGHT = 0.3  # Reduce recent form influence

    # Star player thresholds
    STAR_IMPORTANCE_THRESHOLD = 0.35
    STAR_INJURY_FACTOR = 0.05

    NON_INJURY_REASONS = frozenset([
        "COACH'S DECISION", "NOT WITH TEAM", "REST",
        "G LEAGUE - TWO-WAY", "G LEAGUE", "PERSONAL"
    ])

    def __init__(self):
        self.spread_model: Ridge | None = None
        self.total_model: Ridge | None = None
        self.spread_scaler: StandardScaler | None = None
        self.total_scaler: StandardScaler | None = None

        # Team state tracking
        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'ppg_wts': [],
            'papg': [], 'papg_wts': [],
            'fg_pct': [], 'fg_wts': [],
            'rebounds': [], 'reb_wts': [],
            'turnovers': [], 'tov_wts': [],
            'margins': [],
            'wins': [],
            'opponents': [],  # Track opponents for SRS
            'game_count': 0,
        }))
        self.prev_ratings: dict = {}
        self.last_game: dict = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}

        # SRS ratings (opponent-adjusted)
        self.team_srs: dict = defaultdict(lambda: defaultdict(float))
        self.prev_srs: dict = {}

        # Dynamic HCA tracking
        self.team_hca_data: dict = defaultdict(lambda: defaultdict(lambda: {
            'home_margins': [], 'away_margins': []
        }))
        self.prev_hca: dict = {}

        # Player rankings
        self.player_ppg: dict = {}
        self.player_importance: dict = {}
        self.team_stars: dict = {}

    def _weighted_avg(self, values: list, weights: list) -> float | None:
        if not values or not weights or len(values) != len(weights):
            return None
        return float(np.average(values, weights=weights))

    def _calculate_srs(self, season: int, iterations: int = 10) -> dict[int, float]:
        """
        Calculate Simple Rating System (SRS) for all teams.

        SRS = Average Margin + Strength of Schedule
        where SOS = average opponent SRS

        Iteratively solved until convergence.
        """
        # Get all teams with games this season
        teams_with_games = [
            tid for tid in self.team_games
            if season in self.team_games[tid] and self.team_games[tid][season]['game_count'] > 0
        ]

        if not teams_with_games:
            return {}

        # Initialize SRS with average margin
        srs = {}
        avg_margins = {}
        for tid in teams_with_games:
            td = self.team_games[tid][season]
            if td['margins']:
                avg_margins[tid] = np.mean(td['margins'])
                srs[tid] = avg_margins[tid]
            else:
                avg_margins[tid] = 0.0
                srs[tid] = 0.0

        # Iterate to solve SRS
        for _ in range(iterations):
            new_srs = {}
            for tid in teams_with_games:
                td = self.team_games[tid][season]
                if td['opponents']:
                    # SOS = average opponent SRS
                    opp_srs = [srs.get(opp, 0) for opp in td['opponents']]
                    sos = np.mean(opp_srs) if opp_srs else 0
                    new_srs[tid] = avg_margins[tid] + sos
                else:
                    new_srs[tid] = avg_margins[tid]
            srs = new_srs

        return srs

    def _get_team_srs(self, team_id: int, season: int) -> float:
        """Get team's SRS rating, blending with previous season early on."""
        current_srs = self.team_srs[team_id].get(season, 0)
        prev_srs = self.prev_srs.get(team_id, 0)

        td = self.team_games[team_id][season]
        games = td['game_count'] if season in self.team_games[team_id] else 0

        if games == 0:
            return prev_srs

        # Blend with previous season (half-life of 15 games)
        blend = 0.5 ** (games / 15.0)
        return blend * prev_srs + (1 - blend) * current_srs

    def _get_team_stats(self, team_id: int, season: int) -> dict:
        td = self.team_games[team_id][season]
        games_played = td['game_count']

        ppg = self._weighted_avg(td['ppg'], td['ppg_wts'])
        papg = self._weighted_avg(td['papg'], td['papg_wts'])
        fg_pct = self._weighted_avg(td['fg_pct'], td['fg_wts'])
        rebounds = self._weighted_avg(td['rebounds'], td['reb_wts'])
        turnovers = self._weighted_avg(td['turnovers'], td['tov_wts'])

        prev = self.prev_ratings.get(team_id, {})
        prev_ppg = prev.get('ppg', self.league_avg['ppg'])
        prev_papg = prev.get('papg', self.league_avg['papg'])
        prev_fg = prev.get('fg_pct', 46.0)
        prev_reb = prev.get('rebounds', 44.0)
        prev_tov = prev.get('turnovers', 14.0)

        if ppg is None:
            return {
                'ppg': prev_ppg, 'papg': prev_papg, 'games': 0,
                'fg_pct': prev_fg, 'rebounds': prev_reb, 'turnovers': prev_tov,
                'margins': [], 'wins': []
            }

        blend = 0.5 ** (games_played / self.PREV_HALF_LIFE)

        return {
            'ppg': blend * prev_ppg + (1 - blend) * ppg,
            'papg': blend * prev_papg + (1 - blend) * papg,
            'fg_pct': blend * prev_fg + (1 - blend) * (fg_pct or prev_fg),
            'rebounds': blend * prev_reb + (1 - blend) * (rebounds or prev_reb),
            'turnovers': blend * prev_tov + (1 - blend) * (turnovers or prev_tov),
            'games': games_played,
            'margins': td['margins'],
            'wins': td['wins'],
        }

    def _get_rest_days(self, team_id: int, game_date: str) -> int:
        if team_id not in self.last_game:
            return 3
        curr = datetime.strptime(game_date[:10], '%Y-%m-%d')
        last = datetime.strptime(self.last_game[team_id][:10], '%Y-%m-%d')
        return max(0, min((curr - last).days - 1, 5))

    def _get_dynamic_hca(self, home_id: int, season: int) -> float:
        """Calculate dynamic HCA with more conservative parameters."""
        hd = self.team_hca_data[home_id][season]
        n_home = len(hd['home_margins'])
        n_away = len(hd['away_margins'])
        total_games = n_home + n_away

        if total_games == 0:
            return self.prev_hca.get(home_id, self.BASE_HCA)

        if n_home > 0 and n_away > 0:
            avg_home_margin = np.mean(hd['home_margins'])
            avg_away_margin = np.mean(hd['away_margins'])
            raw_hca = (avg_home_margin - avg_away_margin) / 2
            raw_hca = max(-1, min(raw_hca, 5))  # Tighter clamp
        else:
            raw_hca = self.BASE_HCA

        # More aggressive shrinkage
        shrunk_hca = self.BASE_HCA + self.HCA_SHRINK * (raw_hca - self.BASE_HCA)

        prev = self.prev_hca.get(home_id, self.BASE_HCA)
        blend = 0.5 ** (total_games / self.HCA_HALF_LIFE)

        return blend * prev + (1 - blend) * shrunk_hca

    def _get_recent_form(self, margins: list, n: int = 5) -> float:
        """Get dampened recent form."""
        if len(margins) < n:
            return 0.0
        raw_form = float(np.mean(margins[-n:]))
        # Apply dampening weight
        return raw_form * self.FORM_WEIGHT

    def _get_momentum(self, margins: list, n: int = 6) -> float:
        if len(margins) < n:
            return 0.0
        recent = margins[-n:]
        first_half = np.mean(recent[:n//2])
        second_half = np.mean(recent[n//2:])
        # Also dampen momentum
        return (second_half - first_half) * self.FORM_WEIGHT

    def _get_streak(self, wins: list) -> int:
        if not wins:
            return 0
        streak = 0
        last_result = wins[-1]
        for w in reversed(wins):
            if w == last_result:
                streak += 1
            else:
                break
        return streak if last_result == 1 else -streak

    def load_player_rankings(self, db_path: Path = DB_PATH, min_games: int = 15):
        """Load player importance rankings."""
        conn = sqlite3.connect(str(db_path))

        df = pd.read_sql_query(f'''
            SELECT
                pgs.player_id,
                pgs.team_id,
                AVG(pgs.minutes) as avg_minutes,
                AVG(pgs.points) as avg_points,
                AVG(pgs.plus_minus) as avg_pm,
                AVG(CAST(pgs.starter AS REAL)) as starter_rate,
                COUNT(*) as games
            FROM player_game_stats pgs
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.did_not_play = 0
              AND pgs.minutes > 0
              AND g.completed = 1
            GROUP BY pgs.player_id, pgs.team_id
            HAVING COUNT(*) >= {min_games}
        ''', conn)
        conn.close()

        if df.empty:
            log.warning("No player stats found")
            return

        team_totals = df.groupby('team_id').agg({
            'avg_minutes': 'sum',
            'avg_points': 'sum'
        }).rename(columns={'avg_minutes': 'team_minutes', 'avg_points': 'team_points'})

        df = df.merge(team_totals, on='team_id')

        df['minutes_share'] = df['avg_minutes'] / (df['team_minutes'] / 5)
        df['points_share'] = df['avg_points'] / (df['team_points'] / 5)

        team_pm = df.groupby('team_id')['avg_pm'].agg(['mean', 'std'])
        df = df.merge(team_pm, on='team_id', suffixes=('', '_team'))
        df['pm_zscore'] = (df['avg_pm'] - df['mean']) / df['std'].replace(0, 1)
        df['pm_normalized'] = ((df['pm_zscore'] + 3) / 6).clip(0, 1)

        df['importance'] = (
            0.40 * df['minutes_share'].clip(0, 1) +
            0.30 * df['points_share'].clip(0, 1) +
            0.15 * df['pm_normalized'] +
            0.15 * df['starter_rate']
        )

        self.player_ppg = dict(zip(df['player_id'], df['avg_points']))
        self.player_importance = dict(zip(df['player_id'], df['importance']))

        df['team_rank'] = df.groupby('team_id')['importance'].rank(
            ascending=False, method='first'
        )

        self.team_stars = {}
        for team_id in df['team_id'].unique():
            team_df = df[df['team_id'] == team_id].nsmallest(3, 'team_rank')
            self.team_stars[team_id] = [
                (row['player_id'], row['avg_points'], row['importance'])
                for _, row in team_df.iterrows()
            ]

        log.info(f"Loaded {len(self.player_ppg)} player rankings")

    def get_injury_adjustment(self, team_id: int, game_id: int,
                              db_path: Path = DB_PATH) -> float:
        conn = sqlite3.connect(str(db_path))

        dnp_players = pd.read_sql_query('''
            SELECT player_id, dnp_reason
            FROM player_game_stats
            WHERE game_id = ? AND team_id = ? AND did_not_play = 1
        ''', conn, params=(int(game_id), int(team_id)))
        conn.close()

        if dnp_players.empty:
            return 0.0

        injury_dnps = dnp_players[
            ~dnp_players['dnp_reason'].fillna('UNKNOWN').isin(self.NON_INJURY_REASONS)
        ]

        if injury_dnps.empty:
            return 0.0

        missing_star_ppg = 0.0
        for pid in injury_dnps['player_id']:
            ppg = self.player_ppg.get(pid, 0)
            importance = self.player_importance.get(pid, 0)
            if importance >= self.STAR_IMPORTANCE_THRESHOLD:
                missing_star_ppg += ppg

        return -missing_star_ppg * self.STAR_INJURY_FACTOR

    def extract_features(self, home_id: int, away_id: int, season: int,
                         game_date: str, game_id: int = None,
                         db_path: Path = DB_PATH) -> np.ndarray:
        """
        Extract feature vector (18 features).

        Changes from V1:
        - Added SRS differential
        - Removed collinear Net Rating
        - Dampened form features
        """
        home_stats = self._get_team_stats(home_id, season)
        away_stats = self._get_team_stats(away_id, season)

        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)

        # Core differentials
        ppg_diff = home_stats['ppg'] - away_stats['ppg']
        papg_diff = home_stats['papg'] - away_stats['papg']

        # SRS differential (opponent-adjusted)
        home_srs = self._get_team_srs(home_id, season)
        away_srs = self._get_team_srs(away_id, season)
        srs_diff = home_srs - away_srs

        # Dampened form features
        recent_diff = self._get_recent_form(home_stats['margins'], 5) - \
                      self._get_recent_form(away_stats['margins'], 5)
        momentum_diff = self._get_momentum(home_stats['margins'], 6) - \
                        self._get_momentum(away_stats['margins'], 6)

        # Streaks (not dampened - binary indicator)
        streak_diff = self._get_streak(home_stats['wins']) - \
                      self._get_streak(away_stats['wins'])

        # Dynamic HCA (reduced base)
        hca = self._get_dynamic_hca(home_id, season)

        # Injury adjustment
        injury_adj = 0.0
        if game_id is not None and self.player_ppg:
            home_inj = self.get_injury_adjustment(home_id, game_id, db_path)
            away_inj = self.get_injury_adjustment(away_id, game_id, db_path)
            injury_adj = home_inj - away_inj

        features = np.array([
            ppg_diff,                                    # 0: PPG diff
            papg_diff,                                   # 1: PAPG diff
            srs_diff,                                    # 2: SRS diff (opponent-adjusted)
            recent_diff,                                 # 3: Dampened recent form
            momentum_diff,                               # 4: Dampened momentum
            streak_diff,                                 # 5: Streak diff
            home_rest - away_rest,                       # 6: Rest diff
            1.0 if home_rest == 0 else 0.0,              # 7: Home B2B
            1.0 if away_rest == 0 else 0.0,              # 8: Away B2B
            hca,                                         # 9: Dynamic HCA (reduced)
            min(home_stats['games'] / 30.0, 1.0),        # 10: Home reliability
            min(away_stats['games'] / 30.0, 1.0),        # 11: Away reliability
            injury_adj,                                  # 12: Star injury adj
            (home_stats['games'] + away_stats['games']) / 164.0,  # 13: Season progress
            home_stats['fg_pct'] - away_stats['fg_pct'],          # 14: FG% diff
            home_stats['rebounds'] - away_stats['rebounds'],      # 15: Reb diff
            home_stats['turnovers'] - away_stats['turnovers'],    # 16: TOV diff
            1.0 if srs_diff < -3 else 0.0,               # 17: Road fav indicator
        ])

        return features

    def extract_total_features(self, home_id: int, away_id: int, season: int,
                               game_date: str, game_id: int = None,
                               db_path: Path = DB_PATH) -> np.ndarray:
        """Extract features for total prediction."""
        home_stats = self._get_team_stats(home_id, season)
        away_stats = self._get_team_stats(away_id, season)

        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)

        home_recent = abs(self._get_recent_form(home_stats['margins'], 5))
        away_recent = abs(self._get_recent_form(away_stats['margins'], 5))

        home_momentum = self._get_momentum(home_stats['margins'], 6)
        away_momentum = self._get_momentum(away_stats['margins'], 6)

        injury_total_adj = 0.0
        if game_id is not None and self.player_ppg:
            home_inj = self.get_injury_adjustment(home_id, game_id, db_path)
            away_inj = self.get_injury_adjustment(away_id, game_id, db_path)
            injury_total_adj = home_inj + away_inj

        features = np.array([
            home_stats['ppg'] + away_stats['ppg'],
            home_stats['papg'] + away_stats['papg'],
            (home_stats['ppg'] + home_stats['papg']) / 2,
            (away_stats['ppg'] + away_stats['papg']) / 2,
            1.0 if home_rest == 0 else 0.0,
            1.0 if away_rest == 0 else 0.0,
            min(home_stats['games'] / 30.0, 1.0),
            min(away_stats['games'] / 30.0, 1.0),
            home_stats['fg_pct'] + away_stats['fg_pct'],
            home_stats['rebounds'] + away_stats['rebounds'],
            home_stats['turnovers'] + away_stats['turnovers'],
            home_recent + away_recent,
            abs(home_momentum) + abs(away_momentum),
            (home_stats['games'] + away_stats['games']) / 164.0,
            injury_total_adj,
        ])

        return features

    def update_team(self, team_id: int, opponent_id: int, season: int,
                    game_date: str, points_for: int, points_against: int,
                    is_home: bool, fg_pct: float = None, rebounds: float = None,
                    turnovers: float = None):
        """Update team state after a game."""
        td = self.team_games[team_id][season]

        for wts_key in ['ppg_wts', 'papg_wts', 'fg_wts', 'reb_wts', 'tov_wts']:
            td[wts_key] = [w * self.DECAY for w in td[wts_key]]

        td['ppg'].append(points_for)
        td['ppg_wts'].append(1.0)
        td['papg'].append(points_against)
        td['papg_wts'].append(1.0)
        td['game_count'] += 1
        td['opponents'].append(opponent_id)  # Track for SRS

        margin = points_for - points_against
        td['margins'].append(margin)
        td['wins'].append(1 if margin > 0 else 0)

        if pd.notna(fg_pct):
            td['fg_pct'].append(fg_pct)
            td['fg_wts'].append(1.0)
        if pd.notna(rebounds):
            td['rebounds'].append(rebounds)
            td['reb_wts'].append(1.0)
        if pd.notna(turnovers):
            td['turnovers'].append(turnovers)
            td['tov_wts'].append(1.0)

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
                        'fg_pct': np.mean(td['fg_pct']) if td['fg_pct'] else 46.0,
                        'rebounds': np.mean(td['rebounds']) if td['rebounds'] else 44.0,
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 14.0,
                    }

        for team_id in self.team_hca_data:
            if prev in self.team_hca_data[team_id]:
                hd = self.team_hca_data[team_id][prev]
                if hd['home_margins'] and hd['away_margins']:
                    raw_hca = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
                    raw_hca = max(-1, min(raw_hca, 5))
                    self.prev_hca[team_id] = self.BASE_HCA + self.HCA_SHRINK * (raw_hca - self.BASE_HCA)

        # Calculate and store previous season SRS
        prev_srs = self._calculate_srs(prev)
        self.prev_srs = prev_srs

        self.last_game.clear()

    def train(self, db_path: Path = DB_PATH):
        """Train Ridge model V2."""
        log.info("=" * 60)
        log.info("TRAINING NBA RIDGE V2 (Pure Model - No Vegas)")
        log.info("=" * 60)

        self.load_player_rankings(db_path)

        conn = sqlite3.connect(str(db_path))
        games = pd.read_sql_query('''
            SELECT
                g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
                g.home_score, g.away_score,
                hs.field_goal_pct as home_fg, hs.total_rebounds as home_reb,
                hs.turnovers as home_tov,
                aws.field_goal_pct as away_fg, aws.total_rebounds as away_reb,
                aws.turnovers as away_tov,
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

        X_spread, y_spread = [], []
        X_total, y_total = [], []
        vegas_spreads = []
        game_seasons = []

        seasons = sorted(games['season'].unique())

        for season in seasons:
            if season > seasons[0]:
                self.set_previous_season(season)
                prev_games = games[games['season'] == season - 1]
                if len(prev_games) > 0:
                    self.league_avg['ppg'] = prev_games['home_score'].mean()
                    self.league_avg['papg'] = prev_games['away_score'].mean()

            season_games = games[games['season'] == season]

            # Recalculate SRS periodically during season
            games_processed = 0

            for _, g in season_games.iterrows():
                home_games = len(self.team_games[g['home_team_id']][season]['ppg'])
                away_games = len(self.team_games[g['away_team_id']][season]['ppg'])

                # Recalculate SRS every 50 games
                if games_processed % 50 == 0 and games_processed > 0:
                    self.team_srs[season] = self._calculate_srs(season)

                if home_games >= 10 and away_games >= 10:
                    features = self.extract_features(
                        g['home_team_id'], g['away_team_id'],
                        season, g['date'], g['game_id']
                    )
                    total_features = self.extract_total_features(
                        g['home_team_id'], g['away_team_id'],
                        season, g['date'], g['game_id']
                    )

                    actual_spread = g['away_score'] - g['home_score']
                    actual_total = g['home_score'] + g['away_score']

                    X_spread.append(features)
                    y_spread.append(actual_spread)
                    X_total.append(total_features)
                    y_total.append(actual_total)
                    game_seasons.append(season)
                    vegas_spreads.append(g['vegas_spread'] if pd.notna(g['vegas_spread']) else 0)

                # Update team states
                self.update_team(
                    g['home_team_id'], g['away_team_id'], season, g['date'],
                    g['home_score'], g['away_score'], is_home=True,
                    fg_pct=g['home_fg'], rebounds=g['home_reb'], turnovers=g['home_tov']
                )
                self.update_team(
                    g['away_team_id'], g['home_team_id'], season, g['date'],
                    g['away_score'], g['home_score'], is_home=False,
                    fg_pct=g['away_fg'], rebounds=g['away_reb'], turnovers=g['away_tov']
                )
                games_processed += 1

            # Final SRS calculation for season
            self.team_srs[season] = self._calculate_srs(season)

        X_spread = np.array(X_spread)
        y_spread = np.array(y_spread)
        X_total = np.array(X_total)
        y_total = np.array(y_total)
        vegas_spreads = np.array(vegas_spreads)
        game_seasons = np.array(game_seasons)

        nan_mask = np.isnan(X_spread).any(axis=1) | np.isnan(y_spread)
        log.info(f"Dropping {nan_mask.sum()} rows with NaN")
        X_spread = X_spread[~nan_mask]
        y_spread = y_spread[~nan_mask]
        X_total = X_total[~nan_mask]
        y_total = y_total[~nan_mask]
        vegas_spreads = vegas_spreads[~nan_mask]
        game_seasons = game_seasons[~nan_mask]

        # Walk-forward: train on all but last season, test on last
        test_season = seasons[-1]
        train_mask = game_seasons < test_season
        test_mask = game_seasons == test_season

        X_train_s, X_test_s = X_spread[train_mask], X_spread[test_mask]
        y_train_s, y_test_s = y_spread[train_mask], y_spread[test_mask]
        X_train_t, X_test_t = X_total[train_mask], X_total[test_mask]
        y_train_t, y_test_t = y_total[train_mask], y_total[test_mask]
        vegas_test = vegas_spreads[test_mask]

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
        log.info("TEST SET RESULTS (2026 Season)")
        log.info("=" * 60)

        model_spread = self.spread_model.predict(X_test_s_scaled)

        # Apply road favorite penalty post-prediction
        road_fav_mask = X_test_s[:, 17] == 1.0  # Road fav indicator
        model_spread_adj = model_spread.copy()
        model_spread_adj[road_fav_mask] += self.ROAD_FAV_PENALTY

        model_mae = np.abs(model_spread - y_test_s).mean()
        model_adj_mae = np.abs(model_spread_adj - y_test_s).mean()
        vegas_mae = np.abs(vegas_test - y_test_s).mean()

        log.info(f"\n{'Model':<25} {'MAE':<10}")
        log.info("-" * 40)
        log.info(f"{'Ridge V2 (raw)':<25} {model_mae:.2f}")
        log.info(f"{'Ridge V2 (road fav adj)':<25} {model_adj_mae:.2f}")
        log.info(f"{'Vegas':<25} {vegas_mae:.2f}")

        # ATS Analysis
        log.info("\n" + "=" * 60)
        log.info("ATS ANALYSIS (vs Vegas)")
        log.info("=" * 60)

        def calc_ats(pred, vegas, actual, thresh=0):
            edge = pred - vegas
            mask = np.abs(edge) >= thresh
            wins = ((edge[mask] > 0) & (actual[mask] > vegas[mask])) | \
                   ((edge[mask] < 0) & (actual[mask] < vegas[mask]))
            return wins.sum(), mask.sum()

        for thresh in [0, 3, 5, 7]:
            wins, total = calc_ats(model_spread_adj, vegas_test, y_test_s, thresh)
            if total > 0:
                pct = wins / total * 100
                roi = (wins * 0.91 - (total - wins)) / total * 100
                log.info(f"{thresh}+ pt edge: {wins}-{total-wins} ({pct:.1f}%) ROI: {roi:+.1f}% n={total}")

        # By pick type
        log.info("\n--- BY PICK TYPE ---")
        edge = model_spread_adj - vegas_test
        home_fav = vegas_test < 0

        # Road dog picks (edge > 0, home is fav)
        road_dog = (edge > 0) & home_fav
        if road_dog.sum() > 0:
            rd_wins = ((edge[road_dog] > 0) & (y_test_s[road_dog] > vegas_test[road_dog])).sum()
            log.info(f"Road Dog: {rd_wins}-{road_dog.sum()-rd_wins} ({rd_wins/road_dog.sum()*100:.1f}%)")

        # Home dog picks (edge < 0, home is dog)
        home_dog = (edge < 0) & ~home_fav
        if home_dog.sum() > 0:
            hd_wins = ((edge[home_dog] < 0) & (y_test_s[home_dog] < vegas_test[home_dog])).sum()
            log.info(f"Home Dog: {hd_wins}-{home_dog.sum()-hd_wins} ({hd_wins/home_dog.sum()*100:.1f}%)")

        # Coefficients
        log.info("\n" + "=" * 60)
        log.info("MODEL COEFFICIENTS")
        log.info("=" * 60)

        feature_names = [
            'PPG diff', 'PAPG diff', 'SRS diff', 'Recent form (damp)',
            'Momentum (damp)', 'Streak diff', 'Rest diff', 'Home B2B', 'Away B2B',
            'Dynamic HCA', 'Home reliability', 'Away reliability',
            'Injury adj', 'Season progress', 'FG% diff', 'Reb diff', 'TOV diff',
            'Road fav indicator'
        ]

        coefs = list(zip(feature_names, self.spread_model.coef_))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, coef in coefs:
            log.info(f"  {name:<20} {coef:+.3f}")

        return {
            'spread_mae': model_adj_mae,
            'spread_mae_vegas': vegas_mae,
        }

    def predict(self, home_id: int, away_id: int, season: int, game_date: str,
                game_id: int = None, vegas_spread: float = None,
                vegas_total: float = None) -> dict:
        """Make predictions (pure model, no Vegas blend)."""
        if self.spread_model is None:
            raise ValueError("Model not trained")

        # Recalculate SRS for current predictions
        self.team_srs[season] = self._calculate_srs(season)

        spread_features = self.extract_features(
            home_id, away_id, season, game_date, game_id
        )
        total_features = self.extract_total_features(
            home_id, away_id, season, game_date, game_id
        )

        spread_scaled = self.spread_scaler.transform(spread_features.reshape(1, -1))
        total_scaled = self.total_scaler.transform(total_features.reshape(1, -1))

        model_spread = self.spread_model.predict(spread_scaled)[0]
        model_total = self.total_model.predict(total_scaled)[0]

        # Apply road favorite penalty
        is_road_fav = spread_features[17] == 1.0
        if is_road_fav:
            model_spread += self.ROAD_FAV_PENALTY

        # Pure model - no Vegas blend
        final_spread = model_spread
        final_total = model_total

        home_score = (final_total - final_spread) / 2
        away_score = (final_total + final_spread) / 2

        return {
            'predicted_spread': final_spread,
            'predicted_total': final_total,
            'home_score': home_score,
            'away_score': away_score,
            'model_spread': model_spread,
            'model_total': model_total,
            'dynamic_hca': spread_features[9],
            'srs_diff': spread_features[2],
            'is_road_fav': is_road_fav,
        }

    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODEL_DIR / 'nba_ridge_v2.pkl'

        path.parent.mkdir(parents=True, exist_ok=True)

        team_games_dict = {}
        for team_id, seasons in self.team_games.items():
            team_games_dict[team_id] = {}
            for season, stats in seasons.items():
                team_games_dict[team_id][season] = dict(stats)

        hca_data_dict = {}
        for team_id, seasons in self.team_hca_data.items():
            hca_data_dict[team_id] = {}
            for season, data in seasons.items():
                hca_data_dict[team_id][season] = dict(data)

        srs_dict = {}
        for season, ratings in self.team_srs.items():
            srs_dict[season] = dict(ratings)

        model_data = {
            'spread_model': self.spread_model,
            'total_model': self.total_model,
            'spread_scaler': self.spread_scaler,
            'total_scaler': self.total_scaler,
            'team_games': team_games_dict,
            'team_hca_data': hca_data_dict,
            'team_srs': srs_dict,
            'prev_ratings': self.prev_ratings,
            'prev_hca': self.prev_hca,
            'prev_srs': self.prev_srs,
            'last_game': self.last_game,
            'league_avg': self.league_avg,
            'player_ppg': self.player_ppg,
            'player_importance': self.player_importance,
            'team_stars': self.team_stars,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        log.info(f"\nModel saved to: {path}")

    @classmethod
    def load(cls, path: Path = None) -> 'NBARidgeV2':
        """Load model from disk."""
        if path is None:
            path = MODEL_DIR / 'nba_ridge_v2.pkl'

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        model = cls()
        model.spread_model = model_data['spread_model']
        model.total_model = model_data['total_model']
        model.spread_scaler = model_data['spread_scaler']
        model.total_scaler = model_data['total_scaler']
        model.team_games = defaultdict(
            lambda: defaultdict(lambda: {
                'ppg': [], 'papg': [], 'ppg_wts': [], 'papg_wts': [],
                'fg_pct': [], 'fg_wts': [], 'rebounds': [], 'reb_wts': [],
                'turnovers': [], 'tov_wts': [], 'margins': [], 'wins': [],
                'opponents': [], 'game_count': 0
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
        model.player_ppg = model_data.get('player_ppg', {})
        model.player_importance = model_data.get('player_importance', {})
        model.team_stars = model_data.get('team_stars', {})

        return model


def main():
    """Train and save the V2 model."""
    model = NBARidgeV2()
    results = model.train()
    model.save()

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"Ridge V2 MAE: {results['spread_mae']:.2f}")
    log.info(f"Vegas MAE: {results['spread_mae_vegas']:.2f}")


if __name__ == '__main__':
    main()
