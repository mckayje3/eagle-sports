"""
NBA Enhanced Ridge Model

Full-featured model with:
- Dynamic per-team HCA (builds within season, with decay)
- Star player injury adjustments
- Recent form and momentum features

Best MAE parameters from grid search (2026-01-04):
- Decay: 0.93
- HCA Half-Life: 30 games
- HCA Shrinkage: 0.5
- Base HCA: 2.2

For clean baseline without injuries, see nba_simple_model.py
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


class NBAEnhancedModel:
    """
    Enhanced NBA prediction model with dynamic HCA and injury features.

    Key differences from simple model:
    1. Dynamic per-team HCA that builds within season
    2. Star player injury adjustments
    3. Recent form (last 5-10 games)
    4. Win/loss momentum
    """

    # Optimal blend weights
    SPREAD_MODEL_WEIGHT = 0.90
    TOTAL_MODEL_WEIGHT = 0.60

    # Dynamic HCA parameters (from grid search - best MAE)
    DECAY = 0.93  # PPG decay
    PREV_HALF_LIFE = 6.0
    HCA_HALF_LIFE = 30.0  # Games until HCA fully transitions to current season
    HCA_SHRINK = 0.5  # Shrinkage toward base
    BASE_HCA = 2.2  # Starting HCA estimate
    B2B_PENALTY = 1.0

    # Star player thresholds (DNP-based injury system)
    STAR_IMPORTANCE_THRESHOLD = 0.35  # Top ~3 players per team
    STAR_INJURY_FACTOR = 0.05  # Points to adjust per star PPG out

    # Non-injury DNP reasons to exclude
    NON_INJURY_REASONS = frozenset([
        "COACH'S DECISION", "NOT WITH TEAM", "REST",
        "G LEAGUE - TWO-WAY", "G LEAGUE", "PERSONAL"
    ])

    def __init__(self):
        self.spread_model: Ridge | None = None
        self.total_model: Ridge | None = None
        self.spread_scaler: StandardScaler | None = None
        self.total_scaler: StandardScaler | None = None

        # Team state tracking (per-stat weights to handle missing box scores)
        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'ppg_wts': [],
            'papg': [], 'papg_wts': [],
            'fg_pct': [], 'fg_wts': [],
            'rebounds': [], 'reb_wts': [],
            'turnovers': [], 'tov_wts': [],
            'margins': [],  # For momentum
            'wins': [],  # For streak
            'game_count': 0,
        }))
        self.prev_ratings: dict = {}
        self.last_game: dict = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}

        # Dynamic HCA tracking
        self.team_hca_data: dict = defaultdict(lambda: defaultdict(lambda: {
            'home_margins': [], 'away_margins': []
        }))
        self.prev_hca: dict = {}  # Previous season HCA values

        # Player rankings (DNP-based injury system)
        self.player_ppg: dict = {}  # player_id -> avg PPG
        self.player_importance: dict = {}  # player_id -> importance score
        self.team_stars: dict = {}  # team_id -> list of (player_id, ppg, importance)

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
        rebounds = self._weighted_avg(td['rebounds'], td['reb_wts'])
        turnovers = self._weighted_avg(td['turnovers'], td['tov_wts'])

        prev = self.prev_ratings.get(team_id, {})
        prev_ppg = prev.get('ppg', self.league_avg['ppg'])
        prev_papg = prev.get('papg', self.league_avg['papg'])
        prev_fg = prev.get('fg_pct', 46.0)
        prev_reb = prev.get('rebounds', 44.0)
        prev_tov = prev.get('turnovers', 14.0)

        if ppg is None:
            return {'ppg': prev_ppg, 'papg': prev_papg, 'games': 0,
                    'fg_pct': prev_fg, 'rebounds': prev_reb, 'turnovers': prev_tov,
                    'margins': [], 'wins': []}

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
        """
        Calculate dynamic per-team HCA.

        Blends previous season HCA with current season observations,
        using half-life decay to weight toward current data.
        """
        hd = self.team_hca_data[home_id][season]
        n_home = len(hd['home_margins'])
        n_away = len(hd['away_margins'])
        total_games = n_home + n_away

        # No current season data - use previous season or base
        if total_games == 0:
            return self.prev_hca.get(home_id, self.BASE_HCA)

        # Calculate current season raw HCA
        if n_home > 0 and n_away > 0:
            avg_home_margin = np.mean(hd['home_margins'])
            avg_away_margin = np.mean(hd['away_margins'])
            raw_hca = (avg_home_margin - avg_away_margin) / 2
            # Clamp to reasonable range
            raw_hca = max(-2, min(raw_hca, 8))
        else:
            raw_hca = self.BASE_HCA

        # Apply shrinkage toward base
        shrunk_hca = self.BASE_HCA + self.HCA_SHRINK * (raw_hca - self.BASE_HCA)

        # Blend with previous season using half-life
        prev = self.prev_hca.get(home_id, self.BASE_HCA)
        blend = 0.5 ** (total_games / self.HCA_HALF_LIFE)

        return blend * prev + (1 - blend) * shrunk_hca

    def _get_recent_form(self, margins: list, n: int = 5) -> float:
        """Get average margin over last n games."""
        if len(margins) < n:
            return 0.0
        return float(np.mean(margins[-n:]))

    def _get_momentum(self, margins: list, n: int = 6) -> float:
        """Get momentum (trend) - comparing second half vs first half of last n games."""
        if len(margins) < n:
            return 0.0
        recent = margins[-n:]
        first_half = np.mean(recent[:n//2])
        second_half = np.mean(recent[n//2:])
        return second_half - first_half

    def _get_streak(self, wins: list) -> int:
        """Get current win/loss streak. Positive = winning, negative = losing."""
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
        """
        Load player importance rankings using DNP-based system.

        Importance = weighted combination of:
        - Minutes share (40%)
        - Points share (30%)
        - Plus/minus impact (15%)
        - Starter status (15%)
        """
        conn = sqlite3.connect(str(db_path))

        # Get player stats with team context
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
            log.warning("No player stats found for importance scoring")
            return

        # Calculate team totals for normalization
        team_totals = df.groupby('team_id').agg({
            'avg_minutes': 'sum',
            'avg_points': 'sum'
        }).rename(columns={'avg_minutes': 'team_minutes', 'avg_points': 'team_points'})

        df = df.merge(team_totals, on='team_id')

        # Calculate shares
        df['minutes_share'] = df['avg_minutes'] / (df['team_minutes'] / 5)
        df['points_share'] = df['avg_points'] / (df['team_points'] / 5)

        # Normalize plus/minus within team (z-score)
        team_pm = df.groupby('team_id')['avg_pm'].agg(['mean', 'std'])
        df = df.merge(team_pm, on='team_id', suffixes=('', '_team'))
        df['pm_zscore'] = (df['avg_pm'] - df['mean']) / df['std'].replace(0, 1)
        df['pm_normalized'] = ((df['pm_zscore'] + 3) / 6).clip(0, 1)

        # Calculate importance score
        df['importance'] = (
            0.40 * df['minutes_share'].clip(0, 1) +
            0.30 * df['points_share'].clip(0, 1) +
            0.15 * df['pm_normalized'] +
            0.15 * df['starter_rate']
        )

        # Store player PPG and importance
        self.player_ppg = dict(zip(df['player_id'], df['avg_points']))
        self.player_importance = dict(zip(df['player_id'], df['importance']))

        # Build team star rankings
        df['team_rank'] = df.groupby('team_id')['importance'].rank(
            ascending=False, method='first'
        )

        # Store top 3 stars per team
        self.team_stars = {}
        for team_id in df['team_id'].unique():
            team_df = df[df['team_id'] == team_id].nsmallest(3, 'team_rank')
            self.team_stars[team_id] = [
                (row['player_id'], row['avg_points'], row['importance'])
                for _, row in team_df.iterrows()
            ]

        log.info(f"Loaded {len(self.player_ppg)} player rankings, "
                 f"{len(self.team_stars)} teams with stars")

    def get_injury_adjustment(self, team_id: int, game_id: int,
                              db_path: Path = DB_PATH) -> float:
        """
        Calculate spread adjustment for injured star players using DNP data.

        Uses player_game_stats.did_not_play as injury proxy (available for all seasons).
        Filters out non-injury DNPs (COACH'S DECISION, REST, etc.).

        Returns points to ADD to team's expected margin
        (negative if star is out = team scores less).
        """
        conn = sqlite3.connect(str(db_path))

        # Get players who DNP'd for this game on this team (excluding non-injuries)
        dnp_players = pd.read_sql_query('''
            SELECT player_id, dnp_reason
            FROM player_game_stats
            WHERE game_id = ? AND team_id = ? AND did_not_play = 1
        ''', conn, params=(int(game_id), int(team_id)))
        conn.close()

        if dnp_players.empty:
            return 0.0

        # Filter out non-injury DNPs
        injury_dnps = dnp_players[
            ~dnp_players['dnp_reason'].fillna('UNKNOWN').isin(self.NON_INJURY_REASONS)
        ]

        if injury_dnps.empty:
            return 0.0

        # Sum up PPG of missing star players
        missing_star_ppg = 0.0
        for pid in injury_dnps['player_id']:
            ppg = self.player_ppg.get(pid, 0)
            importance = self.player_importance.get(pid, 0)

            # Only count stars (high importance players)
            if importance >= self.STAR_IMPORTANCE_THRESHOLD:
                missing_star_ppg += ppg

        # Convert to spread adjustment using factor
        # Based on testing: factor=0.05 optimizes MAE
        return -missing_star_ppg * self.STAR_INJURY_FACTOR

    def extract_features(self, home_id: int, away_id: int, season: int,
                         game_date: str, game_id: int = None,
                         db_path: Path = DB_PATH) -> np.ndarray:
        """
        Extract enhanced feature vector (20 features).

        Includes:
        - Core stats differentials
        - Dynamic per-team HCA
        - Recent form and momentum
        - Streak indicators
        - Rest/B2B
        - Injury adjustments (if game_id provided)
        """
        home_stats = self._get_team_stats(home_id, season)
        away_stats = self._get_team_stats(away_id, season)

        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)

        # Core differentials
        ppg_diff = home_stats['ppg'] - away_stats['ppg']
        papg_diff = home_stats['papg'] - away_stats['papg']
        net_diff = (home_stats['ppg'] - home_stats['papg']) - \
                   (away_stats['ppg'] - away_stats['papg'])

        # Recent form (last 5 games)
        home_recent = self._get_recent_form(home_stats['margins'], 5)
        away_recent = self._get_recent_form(away_stats['margins'], 5)
        recent_diff = home_recent - away_recent

        # Momentum (trending up or down)
        home_momentum = self._get_momentum(home_stats['margins'], 6)
        away_momentum = self._get_momentum(away_stats['margins'], 6)
        momentum_diff = home_momentum - away_momentum

        # Streaks
        home_streak = self._get_streak(home_stats['wins'])
        away_streak = self._get_streak(away_stats['wins'])
        streak_diff = home_streak - away_streak

        # Dynamic HCA
        hca = self._get_dynamic_hca(home_id, season)

        # Injury adjustment
        injury_adj = 0.0
        if game_id is not None and self.player_ppg:
            home_inj = self.get_injury_adjustment(home_id, game_id, db_path)
            away_inj = self.get_injury_adjustment(away_id, game_id, db_path)
            injury_adj = home_inj - away_inj  # Positive = home loses more stars

        features = np.array([
            ppg_diff,                                    # 0: PPG diff
            papg_diff,                                   # 1: PAPG diff
            net_diff,                                    # 2: Net rating diff
            recent_diff,                                 # 3: Recent form diff (L5)
            momentum_diff,                               # 4: Momentum diff
            streak_diff,                                 # 5: Streak diff
            home_rest - away_rest,                       # 6: Rest diff
            1.0 if home_rest == 0 else 0.0,              # 7: Home B2B
            1.0 if away_rest == 0 else 0.0,              # 8: Away B2B
            hca,                                         # 9: Dynamic per-team HCA
            min(home_stats['games'] / 30.0, 1.0),        # 10: Home reliability
            min(away_stats['games'] / 30.0, 1.0),        # 11: Away reliability
            injury_adj,                                  # 12: Star injury adjustment
            (home_stats['games'] + away_stats['games']) / 164.0,  # 13: Season progress
            home_stats['fg_pct'] - away_stats['fg_pct'],          # 14: FG% diff
            home_stats['rebounds'] - away_stats['rebounds'],      # 15: Reb diff
            home_stats['turnovers'] - away_stats['turnovers'],    # 16: TOV diff
        ])

        return features

    def extract_total_features(self, home_id: int, away_id: int, season: int,
                               game_date: str, game_id: int = None,
                               db_path: Path = DB_PATH) -> np.ndarray:
        """
        Extract enhanced features for total points prediction (15 features).

        Mirrors spread model with total-appropriate versions of:
        - Core stats (PPG, PAPG, pace)
        - Recent form and momentum
        - Rest/B2B effects
        - Box scores (FG%, Reb, TOV)
        - Injury adjustments
        """
        home_stats = self._get_team_stats(home_id, season)
        away_stats = self._get_team_stats(away_id, season)

        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)

        # Recent form for totals - combined scoring trend
        home_recent_total = abs(self._get_recent_form(home_stats['margins'], 5))
        away_recent_total = abs(self._get_recent_form(away_stats['margins'], 5))

        # Momentum for totals - are games getting higher/lower scoring?
        home_momentum = self._get_momentum(home_stats['margins'], 6)
        away_momentum = self._get_momentum(away_stats['margins'], 6)

        # Injury adjustment for totals - stars out may affect total scoring
        injury_total_adj = 0.0
        if game_id is not None and self.player_ppg:
            home_inj = self.get_injury_adjustment(home_id, game_id, db_path)
            away_inj = self.get_injury_adjustment(away_id, game_id, db_path)
            # For totals, stars out = lower combined scoring
            injury_total_adj = home_inj + away_inj  # Both negative = lower total

        features = np.array([
            home_stats['ppg'] + away_stats['ppg'],           # 0: Combined PPG
            home_stats['papg'] + away_stats['papg'],         # 1: Combined PAPG
            (home_stats['ppg'] + home_stats['papg']) / 2,    # 2: Home pace proxy
            (away_stats['ppg'] + away_stats['papg']) / 2,    # 3: Away pace proxy
            1.0 if home_rest == 0 else 0.0,                  # 4: Home B2B
            1.0 if away_rest == 0 else 0.0,                  # 5: Away B2B
            min(home_stats['games'] / 30.0, 1.0),            # 6: Home reliability
            min(away_stats['games'] / 30.0, 1.0),            # 7: Away reliability
            home_stats['fg_pct'] + away_stats['fg_pct'],     # 8: Combined FG%
            home_stats['rebounds'] + away_stats['rebounds'], # 9: Combined rebounds
            home_stats['turnovers'] + away_stats['turnovers'],  # 10: Combined TOV
            home_recent_total + away_recent_total,           # 11: Recent game intensity
            abs(home_momentum) + abs(away_momentum),         # 12: Combined momentum (abs)
            (home_stats['games'] + away_stats['games']) / 164.0,  # 13: Season progress
            injury_total_adj,                                # 14: Star injury total adj
        ])

        return features

    def update_team(self, team_id: int, season: int, game_date: str,
                    points_for: int, points_against: int, is_home: bool,
                    fg_pct: float = None, rebounds: float = None,
                    turnovers: float = None):
        """Update team state after a game."""
        td = self.team_games[team_id][season]

        # Apply decay to all per-stat weights
        for wts_key in ['ppg_wts', 'papg_wts', 'fg_wts', 'reb_wts', 'tov_wts']:
            td[wts_key] = [w * self.DECAY for w in td[wts_key]]

        # Add PPG/PAPG (always available)
        td['ppg'].append(points_for)
        td['ppg_wts'].append(1.0)
        td['papg'].append(points_against)
        td['papg_wts'].append(1.0)
        td['game_count'] += 1

        margin = points_for - points_against
        td['margins'].append(margin)
        td['wins'].append(1 if margin > 0 else 0)

        # Add box score stats with weights (only if available and not NaN)
        if pd.notna(fg_pct):
            td['fg_pct'].append(fg_pct)
            td['fg_wts'].append(1.0)
        if pd.notna(rebounds):
            td['rebounds'].append(rebounds)
            td['reb_wts'].append(1.0)
        if pd.notna(turnovers):
            td['turnovers'].append(turnovers)
            td['tov_wts'].append(1.0)

        # Update HCA data
        hd = self.team_hca_data[team_id][season]
        if is_home:
            hd['home_margins'].append(margin)
        else:
            hd['away_margins'].append(-margin)  # Store as if home

        self.last_game[team_id] = game_date

    def set_previous_season(self, season: int):
        """Set previous season ratings and HCA for blending."""
        prev = season - 1

        # Team ratings (including box score stats)
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

        # Team HCA
        for team_id in self.team_hca_data:
            if prev in self.team_hca_data[team_id]:
                hd = self.team_hca_data[team_id][prev]
                if hd['home_margins'] and hd['away_margins']:
                    raw_hca = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
                    raw_hca = max(-2, min(raw_hca, 8))
                    self.prev_hca[team_id] = self.BASE_HCA + self.HCA_SHRINK * (raw_hca - self.BASE_HCA)

        self.last_game.clear()

    def train(self, db_path: Path = DB_PATH):
        """Train enhanced Ridge model."""
        log.info("=" * 60)
        log.info("TRAINING NBA ENHANCED MODEL")
        log.info("=" * 60)

        # Load player rankings first
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
        vegas_spreads, vegas_totals = [], []
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

            for _, g in season_games.iterrows():
                # Skip if not enough games
                home_games = len(self.team_games[g['home_team_id']][season]['ppg'])
                away_games = len(self.team_games[g['away_team_id']][season]['ppg'])

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
                    vegas_totals.append(g['vegas_total'] if pd.notna(g['vegas_total']) else 220)

                # Update team states with box score data
                self.update_team(
                    g['home_team_id'], season, g['date'],
                    g['home_score'], g['away_score'], is_home=True,
                    fg_pct=g['home_fg'], rebounds=g['home_reb'], turnovers=g['home_tov']
                )
                self.update_team(
                    g['away_team_id'], season, g['date'],
                    g['away_score'], g['home_score'], is_home=False,
                    fg_pct=g['away_fg'], rebounds=g['away_reb'], turnovers=g['away_tov']
                )

        X_spread = np.array(X_spread)
        y_spread = np.array(y_spread)
        X_total = np.array(X_total)
        y_total = np.array(y_total)
        vegas_spreads = np.array(vegas_spreads)
        vegas_totals = np.array(vegas_totals)
        game_seasons = np.array(game_seasons)

        # Handle NaN
        nan_mask = np.isnan(X_spread).any(axis=1) | np.isnan(y_spread)
        log.info(f"Dropping {nan_mask.sum()} rows with NaN")
        X_spread = X_spread[~nan_mask]
        y_spread = y_spread[~nan_mask]
        X_total = X_total[~nan_mask]
        y_total = y_total[~nan_mask]
        vegas_spreads = vegas_spreads[~nan_mask]
        vegas_totals = vegas_totals[~nan_mask]
        game_seasons = game_seasons[~nan_mask]

        # Split
        if len(seasons) >= 3:
            test_season = seasons[-2]
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

        model_spread = self.spread_model.predict(X_test_s_scaled)
        blended_spread = (self.SPREAD_MODEL_WEIGHT * model_spread +
                          (1 - self.SPREAD_MODEL_WEIGHT) * vegas_test_s)

        model_mae = np.abs(model_spread - y_test_s).mean()
        vegas_mae = np.abs(vegas_test_s - y_test_s).mean()
        blend_mae = np.abs(blended_spread - y_test_s).mean()

        model_acc = ((model_spread < 0) == (y_test_s < 0)).mean()
        vegas_acc = ((vegas_test_s < 0) == (y_test_s < 0)).mean()
        blend_acc = ((blended_spread < 0) == (y_test_s < 0)).mean()

        log.info("\nSPREAD PREDICTION:")
        log.info(f"{'Model':<25} {'MAE':<10} {'Winner Acc':<12}")
        log.info("-" * 50)
        log.info(f"{'Enhanced Ridge':<25} {model_mae:.2f}      {model_acc*100:.1f}%")
        log.info(f"{'Vegas':<25} {vegas_mae:.2f}      {vegas_acc*100:.1f}%")
        log.info(f"{'Blended (90/10)':<25} {blend_mae:.2f}      {blend_acc*100:.1f}%")

        # Coefficients
        log.info("\n" + "=" * 60)
        log.info("MODEL COEFFICIENTS")
        log.info("=" * 60)

        feature_names = [
            'PPG diff', 'PAPG diff', 'Net rating', 'Recent form (L5)',
            'Momentum', 'Streak diff', 'Rest diff', 'Home B2B', 'Away B2B',
            'Dynamic HCA', 'Home reliability', 'Away reliability',
            'Injury adj', 'Season progress', 'FG% diff', 'Reb diff', 'TOV diff'
        ]

        coefs = list(zip(feature_names, self.spread_model.coef_))
        coefs.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, coef in coefs:
            log.info(f"  {name:<20} {coef:+.3f}")

        return {
            'spread_mae': blend_mae,
            'spread_mae_model': model_mae,
            'spread_mae_vegas': vegas_mae,
            'winner_acc': blend_acc,
        }

    def predict(self, home_id: int, away_id: int, season: int, game_date: str,
                game_id: int = None, vegas_spread: float = None,
                vegas_total: float = None) -> dict:
        """Make predictions for a game."""
        if self.spread_model is None:
            raise ValueError("Model not trained. Call train() first.")

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

        home_score = (final_total - final_spread) / 2
        away_score = (final_total + final_spread) / 2

        # Get injury adjustment for reporting
        injury_adj = spread_features[12] if len(spread_features) > 12 else 0.0

        return {
            'predicted_spread': final_spread,
            'predicted_total': final_total,
            'home_score': home_score,
            'away_score': away_score,
            'model_spread': model_spread,
            'model_total': model_total,
            'dynamic_hca': spread_features[9],
            'injury_adjustment': injury_adj,
        }

    def save(self, path: Path = None):
        """Save trained model to disk."""
        if path is None:
            path = MODEL_DIR / 'nba_ridge_enhanced.pkl'

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert nested defaultdicts
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

        model_data = {
            'spread_model': self.spread_model,
            'total_model': self.total_model,
            'spread_scaler': self.spread_scaler,
            'total_scaler': self.total_scaler,
            'team_games': team_games_dict,
            'team_hca_data': hca_data_dict,
            'prev_ratings': self.prev_ratings,
            'prev_hca': self.prev_hca,
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
    def load(cls, path: Path = None) -> 'NBAEnhancedModel':
        """Load trained model from disk."""
        if path is None:
            path = MODEL_DIR / 'nba_ridge_enhanced.pkl'

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        model = cls()
        model.spread_model = model_data['spread_model']
        model.total_model = model_data['total_model']
        model.spread_scaler = model_data['spread_scaler']
        model.total_scaler = model_data['total_scaler']
        model.team_games = defaultdict(
            lambda: defaultdict(lambda: {'ppg': [], 'papg': [], 'wts': [],
                                          'margins': [], 'wins': []}),
            model_data['team_games']
        )
        model.team_hca_data = defaultdict(
            lambda: defaultdict(lambda: {'home_margins': [], 'away_margins': []}),
            model_data.get('team_hca_data', {})
        )
        model.prev_ratings = model_data['prev_ratings']
        model.prev_hca = model_data.get('prev_hca', {})
        model.last_game = model_data['last_game']
        model.league_avg = model_data['league_avg']
        model.player_ppg = model_data.get('player_ppg', {})
        model.player_importance = model_data.get('player_importance', {})
        model.team_stars = model_data.get('team_stars', {})

        return model


def main():
    """Train and save the enhanced model."""
    model = NBAEnhancedModel()
    results = model.train()
    model.save()

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"Enhanced Model MAE: {results['spread_mae_model']:.2f}")
    log.info(f"Blended MAE: {results['spread_mae']:.2f}")
    log.info(f"Vegas MAE: {results['spread_mae_vegas']:.2f}")
    log.info(f"Winner Accuracy: {results['winner_acc']*100:.1f}%")


if __name__ == '__main__':
    main()
