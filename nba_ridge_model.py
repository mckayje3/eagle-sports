"""
NBA Enhanced Ridge Model - Production Version

Full-featured Ridge regression model with:
- Recent form (last 5 games)
- Momentum/trend (last 6 games)
- Win/loss streaks
- Per-team HCA (scaled & shrunk toward league mean)
- Both spread and total predictions
- Injury impact features (optional, enabled by default)

Injury features track player importance (based on rolling minutes/points)
and calculate impact when rotation players are out with injuries.

Star Injury Adjustment (Post-hoc):
- Identifies star players (15+ PPG) who are out due to injury
- Applies post-hoc adjustment: (home_star_ppg_lost - away_star_ppg_lost) * 0.05
- Testing showed: +0.25 MAE improvement on injury games, 55% ATS (up from 51.2%)
- Factor of 0.05 optimizes MAE; higher factors (0.10-0.15) improve ATS but hurt MAE

Based on comparison analysis showing this outperforms simpler models,
especially for totals (much lower bias: +0.83 vs +2.56).
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


class NBARidgeModel:
    """
    Enhanced Ridge regression model for NBA predictions.

    Features (16 for spread, 12 for total):
    - PPG/PAPG differentials (weighted by decay)
    - Recent form (last 5 games)
    - Momentum (trend over last 6 games)
    - Win/loss streaks
    - Rest days and back-to-back indicators
    - Per-team HCA (scaled & shrunk)
    - Season progress / reliability weights
    """

    DECAY = 0.97
    MIN_GAMES = 10
    HCA_SCALE = 0.36
    HCA_SHRINK = 0.50
    HCA_DEFAULT = 1.8

    # Star injury post-hoc adjustment
    # Testing showed 0.05 factor gives +0.25 MAE improvement, 55% ATS on injury games
    STAR_INJURY_FACTOR = 0.05
    STAR_PPG_THRESHOLD = 15.0  # Player is a "star" if averaging 15+ PPG

    def __init__(self, use_injuries: bool = True):
        self.spread_model: Ridge | None = None
        self.total_model: Ridge | None = None
        self.spread_scaler: StandardScaler | None = None
        self.total_scaler: StandardScaler | None = None
        self.use_injuries = use_injuries

        # Team state tracking
        self.team_stats: dict = defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'margins': [], 'wins': []
        })
        self.prev_ratings: dict = {}
        self.last_game: dict = {}
        self.league_hca: float = self.HCA_DEFAULT
        self.team_hca: dict = {}

        # Player importance tracking (for injury features)
        # Maps (team_id, player_id) -> rolling stats
        self.player_stats: dict = defaultdict(lambda: {
            'minutes': [], 'points': [], 'games': 0
        })

    def reset(self):
        """Reset all team state."""
        self.team_stats.clear()
        self.player_stats.clear()
        self.prev_ratings.clear()
        self.last_game.clear()
        self.team_hca.clear()

    def _wavg(self, vals: list, wts: list) -> float | None:
        """Weighted average with decay weights."""
        if not vals or not wts:
            return None
        n = min(len(vals), len(wts))
        return float(np.average(vals[-n:], weights=wts[-n:]))

    def _get_rest(self, tid: int, date: str) -> int:
        """Get rest days since team's last game."""
        if tid not in self.last_game:
            return 3
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 2

    def get_team_hca(self, tid: int) -> float:
        """Get per-team HCA, falling back to league average."""
        return self.team_hca.get(tid, self.league_hca)

    def get_player_importance(self, tid: int) -> dict[int, float]:
        """
        Get importance scores for all players on a team.
        Importance is based on rolling average minutes and points contribution.

        Returns:
            Dict mapping player_id to importance score (0-1 scale)
        """
        team_players = {}
        total_minutes = 0
        total_points = 0

        # Collect stats for players on this team
        for (team_id, player_id), stats in self.player_stats.items():
            if team_id == tid and stats['games'] >= 5:
                avg_min = np.mean(stats['minutes'][-20:]) if stats['minutes'] else 0
                avg_pts = np.mean(stats['points'][-20:]) if stats['points'] else 0
                if avg_min > 0:
                    team_players[player_id] = {'minutes': avg_min, 'points': avg_pts}
                    total_minutes += avg_min
                    total_points += avg_pts

        if not team_players or total_minutes == 0:
            return {}

        # Calculate importance as weighted combination of minutes and points share
        importance = {}
        for pid, stats in team_players.items():
            min_share = stats['minutes'] / total_minutes
            pts_share = stats['points'] / total_points if total_points > 0 else 0
            importance[pid] = min_share * 0.6 + pts_share * 0.4

        return importance

    def get_star_ppg(self, tid: int) -> dict[int, float]:
        """
        Get PPG for star players (15+ PPG) on a team.

        Returns:
            Dict mapping player_id to PPG for players averaging >= STAR_PPG_THRESHOLD
        """
        stars = {}
        for (team_id, player_id), stats in self.player_stats.items():
            if team_id == tid and stats['games'] >= 5:
                avg_ppg = np.mean(stats['points'][-20:]) if stats['points'] else 0
                if avg_ppg >= self.STAR_PPG_THRESHOLD:
                    stars[player_id] = avg_ppg
        return stars

    def get_star_injury_adjustment(self, hid: int, aid: int, dnp_players: dict | None = None) -> float:
        """
        Calculate post-hoc spread adjustment for star injuries.

        Formula: (home_star_ppg_lost - away_star_ppg_lost) * STAR_INJURY_FACTOR

        A positive adjustment means the spread should be higher (away team favored more)
        because home team lost more star PPG.

        Args:
            hid: Home team ID
            aid: Away team ID
            dnp_players: Dict mapping team_id to list of injured player_ids

        Returns:
            Spread adjustment value to add to base prediction
        """
        if dnp_players is None:
            return 0.0

        home_stars = self.get_star_ppg(hid)
        away_stars = self.get_star_ppg(aid)

        home_ppg_lost = sum(home_stars.get(pid, 0) for pid in dnp_players.get(hid, []))
        away_ppg_lost = sum(away_stars.get(pid, 0) for pid in dnp_players.get(aid, []))

        return (home_ppg_lost - away_ppg_lost) * self.STAR_INJURY_FACTOR

    def get_injury_features(self, hid: int, aid: int, dnp_players: dict | None = None) -> tuple[float, float]:
        """
        Calculate injury impact features for a game.

        Args:
            hid: Home team ID
            aid: Away team ID
            dnp_players: Dict mapping team_id to list of injured player_ids (injury DNPs only)

        Returns:
            Tuple of (home_importance_lost, away_importance_lost)
        """
        if not self.use_injuries or dnp_players is None:
            return (0.0, 0.0)

        home_importance = self.get_player_importance(hid)
        away_importance = self.get_player_importance(aid)

        home_lost = sum(home_importance.get(pid, 0) for pid in dnp_players.get(hid, []))
        away_lost = sum(away_importance.get(pid, 0) for pid in dnp_players.get(aid, []))

        return (home_lost, away_lost)

    def update_player_stats(self, game_id: int, player_stats_df: pd.DataFrame):
        """
        Update player stats from a game's player box score.

        Args:
            game_id: Game ID (not used, for reference)
            player_stats_df: DataFrame with player_id, team_id, minutes, points, did_not_play
        """
        for _, row in player_stats_df.iterrows():
            if row.get('did_not_play', 0) == 1:
                continue  # Skip DNP players

            minutes = row.get('minutes', 0)
            points = row.get('points', 0)
            if minutes <= 0:
                continue

            key = (row['team_id'], row['player_id'])
            ps = self.player_stats[key]
            ps['minutes'].append(minutes)
            ps['points'].append(points)
            ps['games'] += 1

            # Keep only last 30 games of data
            if len(ps['minutes']) > 30:
                ps['minutes'] = ps['minutes'][-30:]
                ps['points'] = ps['points'][-30:]

    def calculate_team_hca(self, games_df: pd.DataFrame, season: int):
        """Calculate per-team HCA from a season's data."""
        sg = games_df[games_df['season'] == season]
        raw_hca = {}

        for tid in sg['home_team_id'].unique():
            home_games = sg[sg['home_team_id'] == tid]
            away_games = sg[sg['away_team_id'] == tid]

            if len(home_games) >= 10 and len(away_games) >= 10:
                home_margin = (home_games['home_score'] - home_games['away_score']).mean()
                away_margin = (away_games['away_score'] - away_games['home_score']).mean()
                raw_hca[tid] = home_margin - away_margin

        if not raw_hca:
            return

        league_mean = np.mean(list(raw_hca.values()))
        self.league_hca = self.HCA_SCALE * league_mean

        for tid, raw in raw_hca.items():
            shrunk = league_mean + self.HCA_SHRINK * (raw - league_mean)
            self.team_hca[tid] = self.HCA_SCALE * shrunk

    def extract_spread_features(self, hid: int, aid: int, date: str,
                                 dnp_players: dict | None = None) -> np.ndarray | None:
        """
        Extract features for spread prediction.

        Features (16 base + 2 injury = 18 total when injuries enabled):
        - 1-6: PPG/PAPG differentials (season and recent)
        - 7-8: Momentum and streak diffs
        - 9-11: Rest factors
        - 12-16: HCA, reliability, pace
        - 17-18: Injury impact (if enabled)
        """
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]

        if not hs['ppg'] or not aws['ppg']:
            return None
        if len(hs['ppg']) < self.MIN_GAMES or len(aws['ppg']) < self.MIN_GAMES:
            return None

        # Weighted season stats
        h_ppg = self._wavg(hs['ppg'], hs['wts'])
        h_papg = self._wavg(hs['papg'], hs['wts'])
        a_ppg = self._wavg(aws['ppg'], aws['wts'])
        a_papg = self._wavg(aws['papg'], aws['wts'])

        # Recent form (last 5 games, unweighted)
        h_recent_ppg = np.mean(hs['ppg'][-5:]) if len(hs['ppg']) >= 5 else h_ppg
        h_recent_papg = np.mean(hs['papg'][-5:]) if len(hs['papg']) >= 5 else h_papg
        a_recent_ppg = np.mean(aws['ppg'][-5:]) if len(aws['ppg']) >= 5 else a_ppg
        a_recent_papg = np.mean(aws['papg'][-5:]) if len(aws['papg']) >= 5 else a_papg

        # Momentum (trend over last 6 games)
        def get_trend(margins, n=6):
            if len(margins) < n:
                return 0
            recent = margins[-n:]
            return np.mean(recent[n//2:]) - np.mean(recent[:n//2])

        h_trend = get_trend(hs['margins'])
        a_trend = get_trend(aws['margins'])

        # Win streak
        def get_streak(wins):
            if not wins:
                return 0
            streak = 0
            last = wins[-1]
            for w in reversed(wins):
                if w == last:
                    streak += 1
                else:
                    break
            return streak if last == 1 else -streak

        h_streak = get_streak(hs['wins'])
        a_streak = get_streak(aws['wins'])

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        games_played = (len(hs['ppg']) + len(aws['ppg'])) / 2

        base_features = [
            h_ppg - a_ppg,                                          # 1: PPG diff
            h_papg - a_papg,                                        # 2: PAPG diff
            (h_ppg - h_papg) - (a_ppg - a_papg),                   # 3: Net rating diff
            h_recent_ppg - a_recent_ppg,                           # 4: Recent PPG diff
            h_recent_papg - a_recent_papg,                         # 5: Recent PAPG diff
            (h_recent_ppg - h_recent_papg) - (a_recent_ppg - a_recent_papg),  # 6: Recent net
            h_trend - a_trend,                                      # 7: Momentum diff
            h_streak - a_streak,                                    # 8: Streak diff
            hr - ar,                                                # 9: Rest diff
            1 if hr == 0 else 0,                                   # 10: Home B2B
            1 if ar == 0 else 0,                                   # 11: Away B2B
            self.get_team_hca(hid),                                # 12: Per-team HCA
            min(len(hs['ppg']) / 30, 1),                           # 13: Home reliability
            min(len(aws['ppg']) / 30, 1),                          # 14: Away reliability
            min(games_played / 82, 1),                             # 15: Season progress
            (h_ppg + a_ppg) / 2 - 115,                             # 16: Pace adjustment
        ]

        # Add injury features if enabled
        if self.use_injuries:
            home_lost, away_lost = self.get_injury_features(hid, aid, dnp_players)
            base_features.extend([
                home_lost,                                          # 17: Home importance lost
                away_lost,                                          # 18: Away importance lost
            ])

        return np.array(base_features)

    def extract_total_features(self, hid: int, aid: int, date: str,
                               dnp_players: dict | None = None) -> np.ndarray | None:
        """
        Extract features for total prediction.

        Features (12 base + 2 injury = 14 total when injuries enabled).
        """
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]

        if not hs['ppg'] or not aws['ppg']:
            return None
        if len(hs['ppg']) < self.MIN_GAMES or len(aws['ppg']) < self.MIN_GAMES:
            return None

        # Weighted season stats
        h_ppg = self._wavg(hs['ppg'], hs['wts'])
        h_papg = self._wavg(hs['papg'], hs['wts'])
        a_ppg = self._wavg(aws['ppg'], aws['wts'])
        a_papg = self._wavg(aws['papg'], aws['wts'])

        # Recent form
        h_recent_ppg = np.mean(hs['ppg'][-5:]) if len(hs['ppg']) >= 5 else h_ppg
        h_recent_papg = np.mean(hs['papg'][-5:]) if len(hs['papg']) >= 5 else h_papg
        a_recent_ppg = np.mean(aws['ppg'][-5:]) if len(aws['ppg']) >= 5 else a_ppg
        a_recent_papg = np.mean(aws['papg'][-5:]) if len(aws['papg']) >= 5 else a_papg

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        base_features = [
            h_ppg + a_ppg,                                          # 1: Combined PPG
            h_papg + a_papg,                                        # 2: Combined PAPG
            (h_ppg + h_papg) / 2,                                   # 3: Home pace proxy
            (a_ppg + a_papg) / 2,                                   # 4: Away pace proxy
            h_recent_ppg + a_recent_ppg,                           # 5: Recent combined PPG
            h_recent_papg + a_recent_papg,                         # 6: Recent combined PAPG
            1 if hr == 0 else 0,                                   # 7: Home B2B
            1 if ar == 0 else 0,                                   # 8: Away B2B
            min(len(hs['ppg']) / 30, 1),                           # 9: Home reliability
            min(len(aws['ppg']) / 30, 1),                          # 10: Away reliability
            min((len(hs['ppg']) + len(aws['ppg'])) / 164, 1),      # 11: Season progress
            (h_ppg + h_papg + a_ppg + a_papg) / 4 - 115,           # 12: Combined pace adjustment
        ]

        # Add injury features if enabled (affects total via scoring impact)
        if self.use_injuries:
            home_lost, away_lost = self.get_injury_features(hid, aid, dnp_players)
            base_features.extend([
                home_lost + away_lost,                              # 13: Combined importance lost
                home_lost - away_lost,                              # 14: Importance lost differential
            ])

        return np.array(base_features)

    def update_team(self, tid: int, pts_for: int, pts_against: int, date: str, won: bool):
        """Update team state after a game."""
        ts = self.team_stats[tid]
        ts['wts'] = [w * self.DECAY for w in ts['wts']]
        ts['ppg'].append(pts_for)
        ts['papg'].append(pts_against)
        ts['wts'].append(1.0)
        ts['margins'].append(pts_for - pts_against)
        ts['wins'].append(1 if won else 0)
        self.last_game[tid] = date

    def set_prev_season(self, season: int, games_df: pd.DataFrame):
        """Set previous season ratings and calculate per-team HCA."""
        prev = season - 1
        prev_games = games_df[games_df['season'] == prev]
        if len(prev_games) == 0:
            return

        for tid in set(prev_games['home_team_id']) | set(prev_games['away_team_id']):
            home = prev_games[prev_games['home_team_id'] == tid]
            away = prev_games[prev_games['away_team_id'] == tid]
            pts = list(home['home_score']) + list(away['away_score'])
            pts_ag = list(home['away_score']) + list(away['home_score'])
            if pts:
                self.prev_ratings[tid] = {'ppg': np.mean(pts), 'papg': np.mean(pts_ag)}

        # Calculate per-team HCA from previous season
        self.calculate_team_hca(games_df, prev)
        self.last_game.clear()

    def train(self, db_path: Path = DB_PATH):
        """Train Ridge regression models on historical data."""
        log.info("=" * 60)
        log.info("TRAINING NBA ENHANCED RIDGE MODEL")
        log.info(f"Injury features: {'ENABLED' if self.use_injuries else 'DISABLED'}")
        log.info("=" * 60)

        conn = sqlite3.connect(str(db_path))
        games = pd.read_sql_query('''
            SELECT
                g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
                g.home_score, g.away_score,
                o.latest_spread as vegas_spread, o.latest_total as vegas_total
            FROM games g
            LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.home_score > 0 AND g.completed = 1
            ORDER BY g.date
        ''', conn)

        # Load player stats if using injuries
        player_game_stats = None
        injury_dnps = None
        if self.use_injuries:
            log.info("Loading player stats for injury features...")
            player_game_stats = pd.read_sql_query('''
                SELECT game_id, player_id, team_id, minutes, points, did_not_play, dnp_reason
                FROM player_game_stats
            ''', conn)

            # Pre-compute injury DNPs per game (excluding coach's decision, rest, etc.)
            injury_dnps = player_game_stats[
                (player_game_stats['did_not_play'] == 1) &
                (~player_game_stats['dnp_reason'].isin(["COACH'S DECISION", "NOT WITH TEAM", "REST"]))
            ].groupby('game_id').apply(
                lambda x: {tid: list(x[x['team_id'] == tid]['player_id']) for tid in x['team_id'].unique()}
            ).to_dict()
            log.info(f"Found {len(injury_dnps)} games with injury DNPs")

        conn.close()

        log.info(f"Total games: {len(games)}")

        # Process games chronologically
        X_spread, y_spread = [], []
        X_total, y_total = [], []

        seasons = sorted(games['season'].unique())

        for season in seasons:
            if season > seasons[0]:
                self.set_prev_season(season, games)

            season_games = games[games['season'] == season]

            for _, g in season_games.iterrows():
                game_id = g['game_id']

                # Get injury DNPs for this game (if using injuries)
                dnp_players = injury_dnps.get(game_id) if injury_dnps else None

                # Extract features BEFORE updating
                spread_feat = self.extract_spread_features(
                    g['home_team_id'], g['away_team_id'], g['date'], dnp_players
                )
                total_feat = self.extract_total_features(
                    g['home_team_id'], g['away_team_id'], g['date'], dnp_players
                )

                actual_spread = g['away_score'] - g['home_score']
                actual_total = g['home_score'] + g['away_score']

                if spread_feat is not None:
                    X_spread.append(spread_feat)
                    y_spread.append(actual_spread)

                if total_feat is not None:
                    X_total.append(total_feat)
                    y_total.append(actual_total)

                # Update team state
                home_won = g['home_score'] > g['away_score']
                self.update_team(g['home_team_id'], g['home_score'], g['away_score'], g['date'], home_won)
                self.update_team(g['away_team_id'], g['away_score'], g['home_score'], g['date'], not home_won)

                # Update player stats (if using injuries)
                if self.use_injuries and player_game_stats is not None:
                    game_player_stats = player_game_stats[player_game_stats['game_id'] == game_id]
                    if not game_player_stats.empty:
                        self.update_player_stats(game_id, game_player_stats)

        X_spread = np.array(X_spread)
        y_spread = np.array(y_spread)
        X_total = np.array(X_total)
        y_total = np.array(y_total)

        log.info(f"Training samples: {len(X_spread)} spread, {len(X_total)} total")

        # Train spread model
        self.spread_scaler = StandardScaler()
        X_spread_scaled = self.spread_scaler.fit_transform(X_spread)
        self.spread_model = Ridge(alpha=0.1)
        self.spread_model.fit(X_spread_scaled, y_spread)

        # Train total model
        self.total_scaler = StandardScaler()
        X_total_scaled = self.total_scaler.fit_transform(X_total)
        self.total_model = Ridge(alpha=0.1)
        self.total_model.fit(X_total_scaled, y_total)

        # Report coefficients
        log.info("\nSpread model coefficients (top 8):")
        spread_names = ['PPG', 'PAPG', 'Net', 'RecPPG', 'RecPAPG', 'RecNet',
                        'Momentum', 'Streak', 'Rest', 'HomeB2B', 'AwayB2B',
                        'HCA', 'HomeRel', 'AwayRel', 'SeasonProg', 'Pace']
        coefs = sorted(zip(spread_names, self.spread_model.coef_),
                       key=lambda x: abs(x[1]), reverse=True)
        for name, coef in coefs[:8]:
            log.info(f"  {name:<12} {coef:+.4f}")

        log.info("\nTotal model coefficients (top 6):")
        total_names = ['CombPPG', 'CombPAPG', 'HomePace', 'AwayPace',
                       'RecCombPPG', 'RecCombPAPG', 'HomeB2B', 'AwayB2B',
                       'HomeRel', 'AwayRel', 'SeasonProg', 'CombPace']
        coefs = sorted(zip(total_names, self.total_model.coef_),
                       key=lambda x: abs(x[1]), reverse=True)
        for name, coef in coefs[:6]:
            log.info(f"  {name:<12} {coef:+.4f}")

        return {'n_spread': len(X_spread), 'n_total': len(X_total)}

    def predict(self, home_id: int, away_id: int, date: str,
                vegas_spread: float = None, vegas_total: float = None,
                dnp_players: dict | None = None) -> dict | None:
        """
        Make predictions for a game.

        Args:
            home_id: Home team ID
            away_id: Away team ID
            date: Game date string
            vegas_spread: Vegas spread line (optional)
            vegas_total: Vegas total line (optional)
            dnp_players: Dict mapping team_id to list of injured player_ids (optional)

        Returns dict with predicted_spread, predicted_total, home_score, away_score,
        star_injury_adjustment, or None if insufficient data.
        """
        if self.spread_model is None:
            raise ValueError("Model not trained. Call train() first.")

        spread_feat = self.extract_spread_features(home_id, away_id, date, dnp_players)
        total_feat = self.extract_total_features(home_id, away_id, date, dnp_players)

        if spread_feat is None or total_feat is None:
            return None

        # Predict base values
        spread_scaled = self.spread_scaler.transform(spread_feat.reshape(1, -1))
        total_scaled = self.total_scaler.transform(total_feat.reshape(1, -1))

        pred_spread_base = self.spread_model.predict(spread_scaled)[0]
        pred_total = self.total_model.predict(total_scaled)[0]

        # Apply post-hoc star injury adjustment to spread
        # This is separate from learned injury features - testing showed 0.05 factor
        # gives +0.25 MAE improvement and 55% ATS on games with star injuries
        star_adjustment = self.get_star_injury_adjustment(home_id, away_id, dnp_players)
        pred_spread = pred_spread_base + star_adjustment

        # Calculate scores from spread and total
        # spread = away - home, so home = (total - spread) / 2
        home_score = (pred_total - pred_spread) / 2
        away_score = (pred_total + pred_spread) / 2

        return {
            'predicted_spread': pred_spread,
            'predicted_spread_base': pred_spread_base,
            'star_injury_adjustment': star_adjustment,
            'predicted_total': pred_total,
            'home_score': home_score,
            'away_score': away_score,
            'vegas_spread': vegas_spread,
            'vegas_total': vegas_total,
        }

    def save(self, path: Path = None):
        """Save trained model to disk."""
        if path is None:
            path = MODEL_DIR / 'nba_ridge_enhanced.pkl'

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert defaultdict to regular dict for pickling
        team_stats_dict = {k: dict(v) for k, v in self.team_stats.items()}
        player_stats_dict = {k: dict(v) for k, v in self.player_stats.items()}

        model_data = {
            'spread_model': self.spread_model,
            'total_model': self.total_model,
            'spread_scaler': self.spread_scaler,
            'total_scaler': self.total_scaler,
            'team_stats': team_stats_dict,
            'player_stats': player_stats_dict,
            'prev_ratings': self.prev_ratings,
            'last_game': self.last_game,
            'league_hca': self.league_hca,
            'team_hca': self.team_hca,
            'use_injuries': self.use_injuries,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        log.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: Path = None) -> 'NBARidgeModel':
        """Load trained model from disk."""
        if path is None:
            path = MODEL_DIR / 'nba_ridge_enhanced.pkl'

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(use_injuries=model_data.get('use_injuries', True))
        model.spread_model = model_data['spread_model']
        model.total_model = model_data['total_model']
        model.spread_scaler = model_data['spread_scaler']
        model.total_scaler = model_data['total_scaler']
        model.team_stats = defaultdict(
            lambda: {'ppg': [], 'papg': [], 'wts': [], 'margins': [], 'wins': []},
            model_data['team_stats']
        )
        model.player_stats = defaultdict(
            lambda: {'minutes': [], 'points': [], 'games': 0},
            model_data.get('player_stats', {})
        )
        model.prev_ratings = model_data['prev_ratings']
        model.last_game = model_data['last_game']
        model.league_hca = model_data.get('league_hca', cls.HCA_DEFAULT)
        model.team_hca = model_data.get('team_hca', {})

        return model


class NBARidgePredictor:
    """
    Wrapper class for making predictions on upcoming games.
    Matches the interface expected by update_predictions_nba.py.
    """

    def __init__(self, model_path: Path = None, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.model_path = model_path or MODEL_DIR / 'nba_ridge_enhanced.pkl'
        self.model: NBARidgeModel | None = None
        self._load_model()

    def _load_model(self):
        """Load the trained model."""
        try:
            self.model = NBARidgeModel.load(self.model_path)
            log.info(f"Loaded NBA Ridge model from {self.model_path}")
        except FileNotFoundError:
            log.warning(f"Model not found at {self.model_path}. Train first with: py nba_ridge_model.py")
            self.model = None

    def get_injury_dnps(self, game_ids: list[int]) -> dict[int, dict]:
        """
        Get injury DNPs for upcoming games.

        Uses the most recent game's DNPs for each team as a proxy for expected
        injuries in upcoming games (assumes injured players remain out).

        Returns:
            Dict mapping game_id to dict of {team_id: [player_ids]}
        """
        conn = sqlite3.connect(str(self.db_path))

        # Get the most recent completed games for each team
        # to determine current injury status
        query = '''
            WITH team_last_game AS (
                SELECT
                    team_id,
                    MAX(g.game_id) as last_game_id
                FROM (
                    SELECT home_team_id as team_id, game_id, date
                    FROM games WHERE completed = 1
                    UNION ALL
                    SELECT away_team_id as team_id, game_id, date
                    FROM games WHERE completed = 1
                ) g
                GROUP BY team_id
            ),
            recent_injuries AS (
                SELECT
                    pgs.team_id,
                    pgs.player_id,
                    pgs.dnp_reason
                FROM player_game_stats pgs
                JOIN team_last_game tlg ON pgs.game_id = tlg.last_game_id
                                       AND pgs.team_id = tlg.team_id
                WHERE pgs.did_not_play = 1
                  AND pgs.dnp_reason NOT IN ("COACH'S DECISION", "NOT WITH TEAM", "REST")
            )
            SELECT team_id, player_id, dnp_reason FROM recent_injuries
        '''

        injuries = pd.read_sql_query(query, conn)
        conn.close()

        # Build dict of injuries per team
        team_injuries = {}
        for _, row in injuries.iterrows():
            tid = row['team_id']
            if tid not in team_injuries:
                team_injuries[tid] = []
            team_injuries[tid].append(row['player_id'])

        return team_injuries

    def get_upcoming_games(self, days: int = 7) -> pd.DataFrame:
        """Get upcoming NBA games from database."""
        import pytz

        conn = sqlite3.connect(str(self.db_path))

        eastern = pytz.timezone('US/Eastern')
        now_eastern = datetime.now(eastern)
        today = now_eastern.strftime('%Y-%m-%d')
        end_date = (now_eastern + timedelta(days=days)).strftime('%Y-%m-%d')

        query = '''
            SELECT
                g.game_id, g.date, g.season,
                g.home_team_id, g.away_team_id,
                ht.display_name as home_team,
                at.display_name as away_team,
                o.latest_spread as vegas_spread,
                o.latest_total as vegas_total
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.completed = 0
                AND g.game_date_eastern >= ?
                AND g.game_date_eastern <= ?
            ORDER BY g.game_date_eastern, g.date
        '''

        games = pd.read_sql_query(query, conn, params=(today, end_date))
        conn.close()

        return games

    def predict(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for games."""
        if self.model is None:
            log.error("No model loaded")
            return pd.DataFrame()

        # Get current injury status for all teams
        team_injuries = self.get_injury_dnps([])
        if team_injuries:
            log.info(f"Loaded injury data for {len(team_injuries)} teams")

        predictions = []

        for _, game in games_df.iterrows():
            date = game['date']
            home_id = int(game['home_team_id'])
            away_id = int(game['away_team_id'])
            vegas_spread = game.get('vegas_spread')
            vegas_total = game.get('vegas_total')

            # Build dnp_players dict for this game from current injuries
            dnp_players = {}
            if home_id in team_injuries:
                dnp_players[home_id] = team_injuries[home_id]
            if away_id in team_injuries:
                dnp_players[away_id] = team_injuries[away_id]

            pred = self.model.predict(home_id, away_id, date, vegas_spread, vegas_total, dnp_players)

            if pred is None:
                # Fallback for teams with insufficient data
                log.warning(f"Insufficient data for {game['away_team']} @ {game['home_team']}")
                continue

            # Calculate confidence based on spread magnitude
            spread_diff = abs(pred['predicted_spread'])
            confidence = min(0.95, 0.5 + spread_diff / 25)

            # Determine predicted winner
            if pred['predicted_spread'] < 0:
                predicted_winner = game['home_team']
            else:
                predicted_winner = game['away_team']

            predictions.append({
                'game_id': game['game_id'],
                'date': date,
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'pred_home_score': round(pred['home_score'], 1),
                'pred_away_score': round(pred['away_score'], 1),
                'pred_spread': round(pred['predicted_spread'], 1),
                'pred_spread_base': round(pred.get('predicted_spread_base', pred['predicted_spread']), 1),
                'star_injury_adj': round(pred.get('star_injury_adjustment', 0), 2),
                'pred_total': round(pred['predicted_total'], 1),
                'vegas_spread': vegas_spread if pd.notna(vegas_spread) else None,
                'vegas_total': vegas_total if pd.notna(vegas_total) else None,
                'confidence': round(confidence, 3),
                'predicted_winner': predicted_winner,
            })

        return pd.DataFrame(predictions)

    def predict_upcoming(self, days: int = 7) -> pd.DataFrame:
        """Get and predict upcoming games."""
        games = self.get_upcoming_games(days=days)

        if games.empty:
            log.info("No upcoming games found")
            return pd.DataFrame()

        log.info(f"Found {len(games)} upcoming games")
        return self.predict(games)

    def save_predictions(self, predictions_df: pd.DataFrame, output_path: str = 'nba_current_predictions.csv'):
        """Save predictions to CSV."""
        predictions_df.to_csv(output_path, index=False)
        log.info(f"Saved {len(predictions_df)} predictions to {output_path}")


# Need timedelta for get_upcoming_games
from datetime import timedelta


def main():
    """Train and save the model, then generate predictions."""
    log.info("Training NBA Enhanced Ridge Model...")

    model = NBARidgeModel()
    model.train()
    model.save()

    log.info("\n" + "=" * 60)
    log.info("GENERATING PREDICTIONS")
    log.info("=" * 60)

    predictor = NBARidgePredictor()
    predictions = predictor.predict_upcoming(days=7)

    if not predictions.empty:
        predictor.save_predictions(predictions)

        log.info(f"\nSample predictions:")
        for _, p in predictions.head(5).iterrows():
            log.info(f"  {p['away_team']} @ {p['home_team']}: "
                    f"{p['pred_away_score']:.0f}-{p['pred_home_score']:.0f} "
                    f"(spread: {p['pred_spread']:+.1f}, total: {p['pred_total']:.0f})")


if __name__ == '__main__':
    main()
