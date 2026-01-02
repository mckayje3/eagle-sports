"""
Betting Tracker System
Generates betting recommendations based on Vegas deviation strategies,
tracks placed bets, and calculates performance metrics.

NBA Edge Analysis (2026-01-02):
- Spreads: Only AWAY bias >= 5 is profitable (53-57% win rate)
- Totals: Fade the UNDER signal (bet OVER when model < Vegas-5) = 53.8% win rate
- HOME bias and regular UNDER bets are NOT profitable

Edge Classifier Integration (2026-01-02):
- Neural network meta-model predicts when to bet WITH vs FADE model
- Spread: BET_WITH @ 0.65 confidence = 53.8% win rate, +2.8% ROI
- Totals: FADE @ 0.70 confidence = 56.3% win rate, +7.5% ROI
"""
from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import sqlite3

logger = logging.getLogger(__name__)


class BettingTracker:
    """Core betting tracker logic"""

    # Edge classifier thresholds (from 2025 season backtest)
    CLASSIFIER_THRESHOLDS = {
        'NBA': {
            'spread': {'action': 'BET_WITH', 'threshold': 0.65, 'win_rate': 0.538},
            'total': {'action': 'FADE', 'threshold': 0.70, 'win_rate': 0.563},
        }
    }

    # Meta-feature columns used by the classifier
    CLASSIFIER_FEATURES = [
        'spread_edge', 'spread_edge_abs', 'spread_edge_positive',
        'total_edge', 'total_edge_abs', 'total_edge_positive',
        'model_spread', 'model_total', 'vegas_spread', 'vegas_total',
        'recent_spread_acc', 'recent_total_acc',
        'home_games', 'away_games', 'combined_games',
        'home_rest', 'away_rest', 'rest_diff', 'home_b2b', 'away_b2b',
        'season_progress', 'early_season', 'mid_season', 'late_season',
        'vegas_spread_abs', 'big_favorite', 'close_game',
        'vegas_total_high', 'vegas_total_low',
    ]

    # Deviation strategy thresholds based on historical analysis
    # Updated 2026-01-02 based on edge analysis
    SPREAD_STRATEGIES = {
        'CFB': [
            {'min_dev': 7, 'dir': 'HOME', 'win_rate': 0.788, 'base_units': 3},
            {'min_dev': 5, 'dir': 'HOME', 'win_rate': 0.733, 'base_units': 2},
            {'min_dev': 3, 'dir': 'HOME', 'win_rate': 0.700, 'base_units': 1},
        ],
        'NBA': [
            # Only AWAY bias is profitable (53-57% win rate)
            # HOME bias is NOT profitable - removed
            {'min_dev': 6, 'dir': 'AWAY', 'win_rate': 0.570, 'base_units': 2},
            {'min_dev': 5, 'dir': 'AWAY', 'win_rate': 0.534, 'base_units': 1},
        ],
        'CBB': [
            {'min_dev': 10, 'dir': 'HOME', 'win_rate': 0.750, 'base_units': 3},
            {'min_dev': 7, 'dir': 'HOME', 'win_rate': 0.616, 'base_units': 2},
        ],
        'NFL': [
            {'min_dev': 5, 'dir': 'AWAY', 'win_rate': 0.571, 'base_units': 1},
        ]
    }

    # Total strategies: when to bet OVER or UNDER
    # NBA: Fade the under signal (bet OVER when model < Vegas)
    TOTAL_STRATEGIES = {
        'NBA': [
            # When model says UNDER (model < vegas), fade it and bet OVER
            # This is a "fade" strategy - betting opposite of model signal
            {'min_dev': 7, 'dir': 'FADE_UNDER', 'win_rate': 0.537, 'base_units': 2},
            {'min_dev': 5, 'dir': 'FADE_UNDER', 'win_rate': 0.538, 'base_units': 1},
        ],
    }

    # Legacy alias for backwards compatibility
    STRATEGIES = SPREAD_STRATEGIES

    # Database paths for each sport
    SPORT_DBS = {
        'NFL': 'nfl_games.db',
        'CFB': 'cfb_games.db',
        'NBA': 'nba_games.db',
        'CBB': 'cbb_games.db'
    }

    def __init__(self, db_path: str = 'users.db'):
        self.db_path = db_path
        self._classifier = None
        self._classifier_scaler = None
        self._team_stats = None  # Lazy loaded for classifier

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def _get_sport_conn(self, sport: str) -> sqlite3.Connection:
        """Get connection to sport-specific database"""
        return sqlite3.connect(self.SPORT_DBS[sport])

    def _load_classifier(self, sport: str = 'NBA') -> bool:
        """Load the edge classifier model if available."""
        if self._classifier is not None:
            return True

        try:
            import torch
            from nba_edge_classifier import EdgeClassifier

            model_path = Path(__file__).parent / 'models' / 'nba_edge_classifier.pt'
            if not model_path.exists():
                logger.warning(f"Edge classifier not found at {model_path}")
                return False

            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Create and load model
            n_features = len(self.CLASSIFIER_FEATURES)
            self._classifier = EdgeClassifier(input_dim=n_features, hidden_dims=[64, 32])
            self._classifier.load_state_dict(checkpoint['model_state_dict'])
            self._classifier.eval()

            self._classifier_scaler = checkpoint['scaler']
            logger.info("Edge classifier loaded successfully")
            return True

        except Exception as e:
            logger.warning(f"Could not load edge classifier: {e}")
            return False

    def _load_team_stats(self, sport: str = 'NBA'):
        """Load team stats for meta-feature generation."""
        if self._team_stats is not None:
            return

        conn = self._get_sport_conn(sport)

        # Get completed games to build team stats
        games = pd.read_sql_query('''
            SELECT g.game_id, g.date, g.home_team_id, g.away_team_id,
                   g.home_score, g.away_score
            FROM games g
            WHERE g.completed = 1 AND g.home_score > 0
            ORDER BY g.date
        ''', conn)
        conn.close()

        # Build team stats with decay
        DECAY = 0.97
        self._team_stats = defaultdict(lambda: {'ppg': [], 'papg': [], 'wts': []})
        self._last_game = {}

        for _, g in games.iterrows():
            # Update home team
            hid = g['home_team_id']
            hs = self._team_stats[hid]
            hs['wts'] = [w * DECAY for w in hs['wts']]
            hs['ppg'].append(g['home_score'])
            hs['papg'].append(g['away_score'])
            hs['wts'].append(1.0)
            self._last_game[hid] = g['date']

            # Update away team
            aid = g['away_team_id']
            aws = self._team_stats[aid]
            aws['wts'] = [w * DECAY for w in aws['wts']]
            aws['ppg'].append(g['away_score'])
            aws['papg'].append(g['home_score'])
            aws['wts'].append(1.0)
            self._last_game[aid] = g['date']

        logger.info(f"Loaded stats for {len(self._team_stats)} teams")

    def _get_rest_days(self, team_id: int, game_date: str) -> int:
        """Get rest days for a team before a game."""
        if team_id not in self._last_game:
            return 3
        try:
            curr = datetime.strptime(game_date[:10], '%Y-%m-%d')
            last = datetime.strptime(self._last_game[team_id][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 2

    def _generate_meta_features(self, row: pd.Series, sport: str = 'NBA') -> Optional[np.ndarray]:
        """
        Generate meta-features for the edge classifier.

        Args:
            row: Game row with model predictions and Vegas lines
            sport: Sport type

        Returns:
            Feature array or None if cannot generate
        """
        if self._team_stats is None:
            self._load_team_stats(sport)

        # Get team IDs (need to look up from team names)
        conn = self._get_sport_conn(sport)
        cursor = conn.cursor()

        cursor.execute("SELECT team_id FROM teams WHERE display_name = ?", (row['home_team'],))
        home_result = cursor.fetchone()
        cursor.execute("SELECT team_id FROM teams WHERE display_name = ?", (row['away_team'],))
        away_result = cursor.fetchone()
        conn.close()

        if not home_result or not away_result:
            return None

        hid, aid = home_result[0], away_result[0]

        # Get model predictions and Vegas lines
        model_spread = row['predicted_away_score'] - row['predicted_home_score']
        model_total = row['predicted_home_score'] + row['predicted_away_score']
        vegas_spread = row['vegas_spread']
        vegas_total = row.get('vegas_total', model_total)  # Fallback if missing

        if vegas_total is None or pd.isna(vegas_total):
            vegas_total = model_total

        # Calculate edges
        spread_edge = model_spread - vegas_spread
        total_edge = model_total - vegas_total

        # Team stats
        hs = self._team_stats.get(hid, {'ppg': [], 'papg': [], 'wts': []})
        aws = self._team_stats.get(aid, {'ppg': [], 'papg': [], 'wts': []})
        home_games = len(hs['ppg'])
        away_games = len(aws['ppg'])

        # Rest days
        game_date = row.get('game_date', row.get('date', ''))
        home_rest = self._get_rest_days(hid, str(game_date))
        away_rest = self._get_rest_days(aid, str(game_date))

        # Season progress (estimate ~1230 games per season)
        season_progress = 0.5  # Default mid-season

        # Build feature vector
        features = {
            'spread_edge': spread_edge,
            'spread_edge_abs': abs(spread_edge),
            'spread_edge_positive': 1 if spread_edge > 0 else 0,
            'total_edge': total_edge,
            'total_edge_abs': abs(total_edge),
            'total_edge_positive': 1 if total_edge > 0 else 0,
            'model_spread': model_spread,
            'model_total': model_total,
            'vegas_spread': vegas_spread,
            'vegas_total': vegas_total,
            'recent_spread_acc': 0.5,  # Default - would need tracking
            'recent_total_acc': 0.5,
            'home_games': min(home_games / 30, 1),
            'away_games': min(away_games / 30, 1),
            'combined_games': min((home_games + away_games) / 60, 1),
            'home_rest': home_rest,
            'away_rest': away_rest,
            'rest_diff': home_rest - away_rest,
            'home_b2b': 1 if home_rest == 0 else 0,
            'away_b2b': 1 if away_rest == 0 else 0,
            'season_progress': season_progress,
            'early_season': 1 if season_progress < 0.25 else 0,
            'mid_season': 1 if 0.25 <= season_progress < 0.65 else 0,
            'late_season': 1 if season_progress >= 0.65 else 0,
            'vegas_spread_abs': abs(vegas_spread),
            'big_favorite': 1 if abs(vegas_spread) > 8 else 0,
            'close_game': 1 if abs(vegas_spread) < 3 else 0,
            'vegas_total_high': 1 if vegas_total > 230 else 0,
            'vegas_total_low': 1 if vegas_total < 215 else 0,
        }

        return np.array([features[col] for col in self.CLASSIFIER_FEATURES])

    def evaluate_with_classifier(self, row: pd.Series, sport: str = 'NBA') -> tuple[Optional[dict], Optional[dict]]:
        """
        Evaluate a game using the edge classifier.

        Returns:
            Tuple of (spread_recommendation, total_recommendation) or (None, None)
        """
        if sport not in self.CLASSIFIER_THRESHOLDS:
            return None, None

        if not self._load_classifier(sport):
            return None, None

        import torch

        features = self._generate_meta_features(row, sport)
        if features is None:
            return None, None

        # Scale and predict
        features_scaled = self._classifier_scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled)

        with torch.no_grad():
            spread_logits, total_logits = self._classifier(features_tensor)
            spread_probs = torch.softmax(spread_logits, dim=1).numpy()[0]
            total_probs = torch.softmax(total_logits, dim=1).numpy()[0]

        thresholds = self.CLASSIFIER_THRESHOLDS[sport]
        spread_rec, total_rec = None, None

        # Check spread recommendation
        # probs: [P(pass), P(bet_with), P(fade)]
        spread_cfg = thresholds['spread']
        if spread_cfg['action'] == 'BET_WITH' and spread_probs[1] >= spread_cfg['threshold']:
            model_spread = row['predicted_away_score'] - row['predicted_home_score']
            vegas_spread = row['vegas_spread']
            # BET_WITH means bet the direction model suggests
            # Positive spread_edge = model likes away more than Vegas
            spread_edge = model_spread - vegas_spread
            direction = 'AWAY' if spread_edge > 0 else 'HOME'

            spread_rec = {
                'bet_type': 'SPREAD',
                'recommended_side': direction,
                'classifier_action': 'BET_WITH',
                'classifier_confidence': float(spread_probs[1]),
                'expected_win_rate': spread_cfg['win_rate'],
                'recommended_units': 1,
                'reason': f"Classifier BET_WITH @ {spread_probs[1]:.0%} conf ({spread_cfg['win_rate']*100:.1f}% hist)"
            }
        elif spread_cfg['action'] == 'FADE' and spread_probs[2] >= spread_cfg['threshold']:
            model_spread = row['predicted_away_score'] - row['predicted_home_score']
            vegas_spread = row['vegas_spread']
            spread_edge = model_spread - vegas_spread
            # FADE means bet opposite of model
            direction = 'HOME' if spread_edge > 0 else 'AWAY'

            spread_rec = {
                'bet_type': 'SPREAD',
                'recommended_side': direction,
                'classifier_action': 'FADE',
                'classifier_confidence': float(spread_probs[2]),
                'expected_win_rate': spread_cfg['win_rate'],
                'recommended_units': 1,
                'reason': f"Classifier FADE @ {spread_probs[2]:.0%} conf ({spread_cfg['win_rate']*100:.1f}% hist)"
            }

        # Check total recommendation
        total_cfg = thresholds['total']
        if total_cfg['action'] == 'FADE' and total_probs[2] >= total_cfg['threshold']:
            model_total = row['predicted_home_score'] + row['predicted_away_score']
            vegas_total = row.get('vegas_total')
            if vegas_total and not pd.isna(vegas_total):
                total_edge = model_total - vegas_total
                # FADE: if model says UNDER (negative edge), bet OVER
                # FADE: if model says OVER (positive edge), bet UNDER
                direction = 'OVER' if total_edge < 0 else 'UNDER'

                total_rec = {
                    'bet_type': 'TOTAL',
                    'recommended_side': direction,
                    'classifier_action': 'FADE',
                    'classifier_confidence': float(total_probs[2]),
                    'expected_win_rate': total_cfg['win_rate'],
                    'recommended_units': 2 if total_probs[2] >= 0.75 else 1,
                    'reason': f"Classifier FADE @ {total_probs[2]:.0%} conf ({total_cfg['win_rate']*100:.1f}% hist)"
                }
        elif total_cfg['action'] == 'BET_WITH' and total_probs[1] >= total_cfg['threshold']:
            model_total = row['predicted_home_score'] + row['predicted_away_score']
            vegas_total = row.get('vegas_total')
            if vegas_total and not pd.isna(vegas_total):
                total_edge = model_total - vegas_total
                direction = 'OVER' if total_edge > 0 else 'UNDER'

                total_rec = {
                    'bet_type': 'TOTAL',
                    'recommended_side': direction,
                    'classifier_action': 'BET_WITH',
                    'classifier_confidence': float(total_probs[1]),
                    'expected_win_rate': total_cfg['win_rate'],
                    'recommended_units': 1,
                    'reason': f"Classifier BET_WITH @ {total_probs[1]:.0%} conf ({total_cfg['win_rate']*100:.1f}% hist)"
                }

        return spread_rec, total_rec

    def calculate_confidence_intervals(self, spread: float, moe: float) -> dict:
        """
        Calculate 68% and 95% confidence intervals from MOE.

        Args:
            spread: Model predicted spread
            moe: Margin of error from Monte Carlo dropout

        Returns:
            Dict with ci_68_low, ci_68_high, ci_95_low, ci_95_high
        """
        if moe is None or pd.isna(moe):
            moe = 3.0  # Default MOE if not available

        return {
            'ci_68_low': spread - moe,
            'ci_68_high': spread + moe,
            'ci_95_low': spread - 2 * moe,
            'ci_95_high': spread + 2 * moe
        }

    def calculate_units(self, confidence: float, expected_win_rate: float) -> float:
        """
        Calculate recommended units based on confidence and expected win rate.

        Returns 1-3 units:
        - 1 unit: confidence < 0.6 OR expected_wr < 0.60
        - 2 units: confidence 0.6-0.75 AND expected_wr 0.60-0.70
        - 3 units: confidence > 0.75 AND expected_wr > 0.70
        """
        if confidence is None or pd.isna(confidence):
            confidence = 0.5

        if confidence < 0.6 or expected_win_rate < 0.60:
            return 1.0
        elif confidence > 0.75 and expected_win_rate > 0.70:
            return 3.0
        else:
            return 2.0

    def evaluate_spread(self, sport: str, model_spread: float, vegas_spread: float,
                        confidence: float = None) -> Optional[Dict]:
        """
        Evaluate if a game qualifies for spread betting recommendation.

        Using Vegas convention (negative = home favored):
        Deviation = model_spread - vegas_spread
        - Negative deviation = model favors HOME more than Vegas (model more negative)
        - Positive deviation = model favors AWAY more than Vegas (model less negative)

        Returns:
            Dict with recommendation details or None if no recommendation
        """
        deviation = model_spread - vegas_spread
        abs_deviation = abs(deviation)
        # Negative deviation means model is more negative = favors home more
        direction = 'HOME' if deviation < 0 else 'AWAY'

        strategies = self.SPREAD_STRATEGIES.get(sport, [])

        for strategy in strategies:
            if abs_deviation >= strategy['min_dev'] and direction == strategy['dir']:
                units = self.calculate_units(confidence, strategy['win_rate'])

                return {
                    'bet_type': 'SPREAD',
                    'recommended_side': direction,
                    'deviation': deviation,
                    'abs_deviation': abs_deviation,
                    'expected_win_rate': strategy['win_rate'],
                    'recommended_units': units,
                    'reason': f"{sport} {direction} spread bias {abs_deviation:.1f} pts ({strategy['win_rate']*100:.0f}% historical)"
                }

        return None

    def evaluate_total(self, sport: str, model_total: float, vegas_total: float,
                       confidence: float = None) -> Optional[Dict]:
        """
        Evaluate if a game qualifies for total (O/U) betting recommendation.

        For NBA, we fade the UNDER signal:
        - When model < Vegas (model says UNDER), bet OVER instead

        Returns:
            Dict with recommendation details or None if no recommendation
        """
        deviation = model_total - vegas_total
        abs_deviation = abs(deviation)

        strategies = self.TOTAL_STRATEGIES.get(sport, [])

        for strategy in strategies:
            if strategy['dir'] == 'FADE_UNDER':
                # Model < Vegas = model says under, but we fade it = bet OVER
                if deviation <= -strategy['min_dev']:
                    units = self.calculate_units(confidence, strategy['win_rate'])

                    return {
                        'bet_type': 'TOTAL',
                        'recommended_side': 'OVER',
                        'deviation': deviation,
                        'abs_deviation': abs_deviation,
                        'expected_win_rate': strategy['win_rate'],
                        'recommended_units': units,
                        'reason': f"{sport} FADE UNDER: model {abs_deviation:.1f} below Vegas, bet OVER ({strategy['win_rate']*100:.0f}% historical)"
                    }

        return None

    # Legacy method name for backwards compatibility
    def evaluate_game(self, sport: str, model_spread: float, vegas_spread: float,
                      confidence: float = None) -> Optional[Dict]:
        """Legacy wrapper - only evaluates spread."""
        return self.evaluate_spread(sport, model_spread, vegas_spread, confidence)

    def generate_recommendations(self, sport: str, week: int = None,
                                  season: int = None) -> pd.DataFrame:
        """
        Generate betting recommendations for upcoming games.
        Includes both spread and total recommendations.

        Args:
            sport: NFL, CFB, NBA, or CBB
            week: Week number (optional, defaults to upcoming games)
            season: Season year (optional)

        Returns:
            DataFrame with recommendations
        """
        sport_conn = self._get_sport_conn(sport)

        # Sport-specific query differences:
        # - NFL/CFB have 'week' column, NBA/CBB don't
        # - NFL doesn't have 'confidence' or 'predicted_spread_MOE' in odds
        # - NBA/CBB have these columns

        if sport in ['NFL', 'CFB']:
            # Football sports have week column
            week_col = "g.week"
            if sport == 'NFL':
                # NFL doesn't have confidence/MOE columns
                conf_col = "NULL as confidence"
                moe_col = "NULL as spread_moe"
            else:
                conf_col = "o.confidence"
                moe_col = "o.predicted_spread_MOE as spread_moe"
        else:
            # Basketball sports don't have week
            week_col = "NULL as week"
            conf_col = "o.confidence"
            moe_col = "o.predicted_spread_MOE as spread_moe"

        query = f"""
            SELECT
                g.game_id,
                g.date as game_date,
                {week_col},
                g.season,
                ht.display_name as home_team,
                at.display_name as away_team,
                o.predicted_home_score,
                o.predicted_away_score,
                o.latest_spread as vegas_spread,
                o.latest_total as vegas_total,
                {conf_col},
                {moe_col}
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.completed = 0
            AND o.predicted_home_score IS NOT NULL
            AND o.latest_spread IS NOT NULL
        """

        params = []
        if week is not None and sport in ['NFL', 'CFB']:
            query += " AND g.week = ?"
            params.append(week)
        if season is not None:
            query += " AND g.season = ?"
            params.append(season)

        query += " ORDER BY g.date"

        df = pd.read_sql_query(query, sport_conn, params=params if params else None)
        sport_conn.close()

        if df.empty:
            return pd.DataFrame()

        # Calculate model spread and evaluate each game
        recommendations = []

        for _, row in df.iterrows():
            # Use Vegas convention: negative = home favored
            model_spread = row['predicted_away_score'] - row['predicted_home_score']
            model_total = row['predicted_home_score'] + row['predicted_away_score']
            vegas_spread = row['vegas_spread']
            vegas_total = row.get('vegas_total')
            confidence = row.get('confidence')
            moe = row.get('spread_moe')

            # Evaluate SPREAD recommendation
            spread_rec = self.evaluate_spread(sport, model_spread, vegas_spread, confidence)

            if spread_rec:
                ci = self.calculate_confidence_intervals(model_spread, moe)

                recommendations.append({
                    'game_id': row['game_id'],
                    'sport': sport,
                    'bet_type': 'SPREAD',
                    'game_date': row['game_date'],
                    'week': row['week'],
                    'season': row['season'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'model_spread': model_spread,
                    'vegas_spread': vegas_spread,
                    'model_total': model_total,
                    'vegas_total': vegas_total,
                    'spread_deviation': spread_rec['deviation'],
                    'deviation_direction': spread_rec['recommended_side'] + '_BIAS',
                    'spread_moe': moe,
                    'ci_68_low': ci['ci_68_low'],
                    'ci_68_high': ci['ci_68_high'],
                    'ci_95_low': ci['ci_95_low'],
                    'ci_95_high': ci['ci_95_high'],
                    'model_confidence': confidence,
                    'recommended_side': spread_rec['recommended_side'],
                    'recommended_units': spread_rec['recommended_units'],
                    'expected_win_rate': spread_rec['expected_win_rate'],
                    'reason': spread_rec['reason']
                })

            # Evaluate TOTAL recommendation (if Vegas total available)
            if vegas_total and pd.notna(vegas_total):
                total_rec = self.evaluate_total(sport, model_total, vegas_total, confidence)

                if total_rec:
                    recommendations.append({
                        'game_id': row['game_id'],
                        'sport': sport,
                        'bet_type': 'TOTAL',
                        'game_date': row['game_date'],
                        'week': row['week'],
                        'season': row['season'],
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'model_spread': model_spread,
                        'vegas_spread': vegas_spread,
                        'model_total': model_total,
                        'vegas_total': vegas_total,
                        'spread_deviation': total_rec['deviation'],  # Actually total deviation
                        'deviation_direction': total_rec['recommended_side'],
                        'spread_moe': moe,
                        'ci_68_low': None,
                        'ci_68_high': None,
                        'ci_95_low': None,
                        'ci_95_high': None,
                        'model_confidence': confidence,
                        'recommended_side': total_rec['recommended_side'],
                        'recommended_units': total_rec['recommended_units'],
                        'expected_win_rate': total_rec['expected_win_rate'],
                        'reason': total_rec['reason']
                    })

            # Evaluate CLASSIFIER recommendations (NBA only for now)
            if sport in self.CLASSIFIER_THRESHOLDS:
                clf_spread_rec, clf_total_rec = self.evaluate_with_classifier(row, sport)

                # Add classifier spread recommendation if not already covered by deviation strategy
                if clf_spread_rec and not spread_rec:
                    clf_spread_edge = model_spread - vegas_spread
                    recommendations.append({
                        'game_id': row['game_id'],
                        'sport': sport,
                        'bet_type': 'SPREAD_CLF',
                        'game_date': row['game_date'],
                        'week': row['week'],
                        'season': row['season'],
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'model_spread': model_spread,
                        'vegas_spread': vegas_spread,
                        'model_total': model_total,
                        'vegas_total': vegas_total,
                        'spread_deviation': clf_spread_edge,
                        'deviation_direction': f"CLF_{clf_spread_rec['classifier_action']}",
                        'spread_moe': moe,
                        'ci_68_low': None,
                        'ci_68_high': None,
                        'ci_95_low': None,
                        'ci_95_high': None,
                        'model_confidence': clf_spread_rec['classifier_confidence'],
                        'recommended_side': clf_spread_rec['recommended_side'],
                        'recommended_units': clf_spread_rec['recommended_units'],
                        'expected_win_rate': clf_spread_rec['expected_win_rate'],
                        'reason': clf_spread_rec['reason']
                    })

                # Add classifier total recommendation if not already covered
                if clf_total_rec and not total_rec:
                    clf_total_edge = model_total - (vegas_total or model_total)
                    recommendations.append({
                        'game_id': row['game_id'],
                        'sport': sport,
                        'bet_type': 'TOTAL_CLF',
                        'game_date': row['game_date'],
                        'week': row['week'],
                        'season': row['season'],
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'model_spread': model_spread,
                        'vegas_spread': vegas_spread,
                        'model_total': model_total,
                        'vegas_total': vegas_total,
                        'spread_deviation': clf_total_edge,
                        'deviation_direction': f"CLF_{clf_total_rec['classifier_action']}",
                        'spread_moe': moe,
                        'ci_68_low': None,
                        'ci_68_high': None,
                        'ci_95_low': None,
                        'ci_95_high': None,
                        'model_confidence': clf_total_rec['classifier_confidence'],
                        'recommended_side': clf_total_rec['recommended_side'],
                        'recommended_units': clf_total_rec['recommended_units'],
                        'expected_win_rate': clf_total_rec['expected_win_rate'],
                        'reason': clf_total_rec['reason']
                    })

        return pd.DataFrame(recommendations)

    def save_recommendations(self, recommendations: pd.DataFrame) -> int:
        """
        Save recommendations to database.

        Returns:
            Number of new recommendations saved
        """
        if recommendations.empty:
            return 0

        conn = self._get_conn()
        cursor = conn.cursor()

        saved = 0
        for _, rec in recommendations.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO betting_recommendations (
                        game_id, sport, game_date, week, season,
                        home_team, away_team, model_spread, vegas_spread,
                        spread_deviation, deviation_direction,
                        spread_moe, ci_68_low, ci_68_high, ci_95_low, ci_95_high,
                        model_confidence, recommended_side, recommended_units,
                        expected_win_rate, reason, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """, (
                    rec['game_id'], rec['sport'], rec['game_date'], rec['week'],
                    rec['season'], rec['home_team'], rec['away_team'],
                    rec['model_spread'], rec['vegas_spread'], rec['spread_deviation'],
                    rec['deviation_direction'], rec['spread_moe'],
                    rec['ci_68_low'], rec['ci_68_high'], rec['ci_95_low'], rec['ci_95_high'],
                    rec['model_confidence'], rec['recommended_side'],
                    rec['recommended_units'], rec['expected_win_rate'],
                    rec['reason'], datetime.now().isoformat()
                ))
                saved += 1
            except Exception as e:
                print(f"Error saving recommendation for game {rec['game_id']}: {e}")

        conn.commit()
        conn.close()
        return saved

    def get_recommendations(self, sport: str = None, status: str = 'pending') -> pd.DataFrame:
        """Get current recommendations from database"""
        conn = self._get_conn()

        query = "SELECT * FROM betting_recommendations WHERE 1=1"
        params = []

        if sport:
            query += " AND sport = ?"
            params.append(sport)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY game_date, sport"

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()
        return df

    def log_bet(self, recommendation_id: int = None, game_id: int = None,
                sport: str = None, bet_side: str = None, line: float = None,
                units: float = None, odds: int = -110,
                deviation: float = None, confidence: float = None) -> int:
        """
        Log a placed bet.

        Can be linked to a recommendation or created manually.

        Returns:
            Bet ID
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # If linked to recommendation, get details
        if recommendation_id:
            cursor.execute("""
                SELECT game_id, sport, vegas_spread, recommended_units,
                       spread_deviation, model_confidence
                FROM betting_recommendations WHERE id = ?
            """, (recommendation_id,))
            rec = cursor.fetchone()
            if rec:
                game_id = rec[0]
                sport = rec[1]
                line = line or rec[2]
                units = units or rec[3]
                deviation = rec[4]
                confidence = rec[5]

        cursor.execute("""
            INSERT INTO placed_bets (
                recommendation_id, game_id, sport, bet_side,
                line_at_bet, odds, units, deviation_at_bet, confidence_at_bet
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (recommendation_id, game_id, sport, bet_side,
              line, odds, units, deviation, confidence))

        bet_id = cursor.lastrowid

        # Update recommendation status if linked
        if recommendation_id:
            cursor.execute("""
                UPDATE betting_recommendations SET status = 'locked'
                WHERE id = ?
            """, (recommendation_id,))

        conn.commit()
        conn.close()
        return bet_id

    def update_results(self) -> Dict:
        """
        Update bet outcomes from completed games.

        Returns:
            Dict with counts of updated bets
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # Get pending bets
        cursor.execute("""
            SELECT id, game_id, sport, bet_side, line_at_bet, odds, units
            FROM placed_bets WHERE outcome = 'PENDING'
        """)
        pending_bets = cursor.fetchall()

        updated = {'wins': 0, 'losses': 0, 'pushes': 0}

        for bet in pending_bets:
            bet_id, game_id, sport, bet_side, line, odds, units = bet

            # Get actual result from sport database
            sport_conn = self._get_sport_conn(sport)
            result = pd.read_sql_query("""
                SELECT home_score, away_score, completed
                FROM games WHERE game_id = ?
            """, sport_conn, params=[game_id])
            sport_conn.close()

            if result.empty or not result.iloc[0]['completed']:
                continue

            home_score = result.iloc[0]['home_score']
            away_score = result.iloc[0]['away_score']
            actual_margin = home_score - away_score

            # Determine outcome
            # Spread is from home perspective (negative = home favored)
            # If betting HOME, home needs to win by more than -spread
            # If betting AWAY, away needs to cover (home wins by less than spread)

            if bet_side == 'HOME':
                # Home covering means: actual_margin > -line (since line is home spread)
                cover_margin = actual_margin + line  # How much home beat the spread by
            else:  # AWAY
                # Away covering means: actual_margin < -line
                cover_margin = -line - actual_margin  # How much away beat the spread by

            if abs(cover_margin) < 0.5:  # Push (within 0.5 points)
                outcome = 'PUSH'
                profit = 0.0
                updated['pushes'] += 1
            elif cover_margin > 0:
                outcome = 'WIN'
                # Calculate profit based on odds
                if odds < 0:
                    profit = units * (100 / abs(odds))
                else:
                    profit = units * (odds / 100)
                updated['wins'] += 1
            else:
                outcome = 'LOSS'
                profit = -units
                updated['losses'] += 1

            # Update bet record
            cursor.execute("""
                UPDATE placed_bets
                SET outcome = ?, actual_margin = ?, profit_units = ?
                WHERE id = ?
            """, (outcome, actual_margin, profit, bet_id))

            # Update recommendation status
            cursor.execute("""
                UPDATE betting_recommendations
                SET status = 'completed'
                WHERE game_id = ? AND sport = ?
            """, (game_id, sport))

        conn.commit()
        conn.close()
        return updated

    def get_placed_bets(self, sport: str = None, outcome: str = None) -> pd.DataFrame:
        """Get placed bets with optional filters"""
        conn = self._get_conn()

        query = "SELECT * FROM placed_bets WHERE 1=1"
        params = []

        if sport:
            query += " AND sport = ?"
            params.append(sport)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)

        query += " ORDER BY placed_at DESC"

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()
        return df

    def calculate_roi(self, wins: int, losses: int, pushes: int = 0,
                      odds: int = -110) -> float:
        """
        Calculate ROI for flat betting.

        At -110 odds:
        - Win: profit = 100/110 = 0.909 units
        - Loss: profit = -1.0 units
        - Push: profit = 0 units
        """
        if odds < 0:
            profit_per_win = 100 / abs(odds)
        else:
            profit_per_win = odds / 100

        total_wagered = wins + losses  # pushes not counted
        if total_wagered == 0:
            return 0.0

        total_profit = (wins * profit_per_win) - losses
        roi = (total_profit / total_wagered) * 100
        return roi

    def get_performance(self, sport: str = None,
                        deviation_min: float = None,
                        direction: str = None) -> Dict:
        """
        Get performance metrics.

        Args:
            sport: Filter by sport
            deviation_min: Minimum deviation threshold
            direction: 'HOME' or 'AWAY' bias filter

        Returns:
            Dict with performance metrics
        """
        conn = self._get_conn()

        query = """
            SELECT
                sport,
                outcome,
                units,
                profit_units,
                deviation_at_bet,
                bet_side
            FROM placed_bets
            WHERE outcome != 'PENDING'
        """
        params = []

        if sport:
            query += " AND sport = ?"
            params.append(sport)
        if deviation_min:
            query += " AND ABS(deviation_at_bet) >= ?"
            params.append(deviation_min)
        if direction:
            query += " AND bet_side = ?"
            params.append(direction)

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()

        if df.empty:
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'win_rate': 0.0,
                'roi': 0.0,
                'total_profit': 0.0,
                'total_wagered': 0.0
            }

        wins = len(df[df['outcome'] == 'WIN'])
        losses = len(df[df['outcome'] == 'LOSS'])
        pushes = len(df[df['outcome'] == 'PUSH'])

        total_profit = df['profit_units'].sum()
        total_wagered = df[df['outcome'] != 'PUSH']['units'].sum()

        return {
            'total_bets': len(df),
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            'roi': self.calculate_roi(wins, losses, pushes),
            'total_profit': total_profit,
            'total_wagered': total_wagered
        }

    def get_performance_by_sport(self) -> pd.DataFrame:
        """Get performance breakdown by sport"""
        conn = self._get_conn()

        query = """
            SELECT
                sport,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN outcome = 'PUSH' THEN 1 ELSE 0 END) as pushes,
                SUM(profit_units) as total_profit,
                COUNT(*) as total_bets
            FROM placed_bets
            WHERE outcome != 'PENDING'
            GROUP BY sport
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if not df.empty:
            df['win_rate'] = df['wins'] / (df['wins'] + df['losses'])
            df['roi'] = df.apply(
                lambda r: self.calculate_roi(r['wins'], r['losses'], r['pushes']),
                axis=1
            )

        return df

    def get_performance_by_deviation(self, buckets: List[Tuple] = None) -> pd.DataFrame:
        """
        Get performance breakdown by deviation bucket.

        Args:
            buckets: List of (min, max) tuples for buckets
                     Default: [(3, 5), (5, 7), (7, 10), (10, float('inf'))]
        """
        if buckets is None:
            buckets = [(3, 5), (5, 7), (7, 10), (10, 999)]

        conn = self._get_conn()

        results = []
        for min_dev, max_dev in buckets:
            query = """
                SELECT
                    bet_side,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    SUM(profit_units) as profit,
                    COUNT(*) as total_bets
                FROM placed_bets
                WHERE outcome != 'PENDING'
                AND ABS(deviation_at_bet) >= ? AND ABS(deviation_at_bet) < ?
                GROUP BY bet_side
            """

            df = pd.read_sql_query(query, conn, params=[min_dev, max_dev])

            for _, row in df.iterrows():
                bucket_label = f"{min_dev}-{max_dev}" if max_dev < 999 else f"{min_dev}+"
                results.append({
                    'bucket': bucket_label,
                    'direction': row['bet_side'],
                    'wins': row['wins'],
                    'losses': row['losses'],
                    'total_bets': row['total_bets'],
                    'profit': row['profit'],
                    'win_rate': row['wins'] / (row['wins'] + row['losses']) if (row['wins'] + row['losses']) > 0 else 0
                })

        conn.close()
        return pd.DataFrame(results)

    def get_settings(self) -> Dict:
        """Get betting settings"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT unit_size, default_odds, min_confidence FROM betting_settings WHERE id = 1")
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'unit_size': row[0],
                'default_odds': row[1],
                'min_confidence': row[2]
            }
        return {'unit_size': 100.0, 'default_odds': -110, 'min_confidence': 0.5}

    def update_settings(self, unit_size: float = None, default_odds: int = None,
                        min_confidence: float = None) -> None:
        """Update betting settings"""
        conn = self._get_conn()
        cursor = conn.cursor()

        updates = []
        params = []

        if unit_size is not None:
            updates.append("unit_size = ?")
            params.append(unit_size)
        if default_odds is not None:
            updates.append("default_odds = ?")
            params.append(default_odds)
        if min_confidence is not None:
            updates.append("min_confidence = ?")
            params.append(min_confidence)

        if updates:
            query = f"UPDATE betting_settings SET {', '.join(updates)} WHERE id = 1"
            cursor.execute(query, params)
            conn.commit()

        conn.close()

    def generate_and_save_all(self) -> Dict:
        """
        Generate and save recommendations for all sports.

        Returns:
            Dict with counts by sport
        """
        results = {}

        for sport in self.SPORT_DBS.keys():
            try:
                recs = self.generate_recommendations(sport)
                saved = self.save_recommendations(recs)
                results[sport] = {'generated': len(recs), 'saved': saved}
            except Exception as e:
                results[sport] = {'error': str(e)}

        return results


# Command-line interface
if __name__ == '__main__':
    import sys

    tracker = BettingTracker()

    if len(sys.argv) < 2:
        print("Usage: python betting_tracker.py <command>")
        print("Commands: generate, update, performance, recommendations")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'generate':
        # Generate recommendations for all sports
        results = tracker.generate_and_save_all()
        for sport, data in results.items():
            if 'error' in data:
                print(f"{sport}: ERROR - {data['error']}")
            else:
                print(f"{sport}: Generated {data['generated']}, Saved {data['saved']}")

    elif command == 'update':
        # Update results for completed games
        results = tracker.update_results()
        print(f"Updated: {results['wins']} wins, {results['losses']} losses, {results['pushes']} pushes")

    elif command == 'performance':
        # Show overall performance
        perf = tracker.get_performance()
        print(f"\nOverall Performance:")
        print(f"  Total Bets: {perf['total_bets']}")
        print(f"  Record: {perf['wins']}-{perf['losses']}-{perf['pushes']}")
        print(f"  Win Rate: {perf['win_rate']:.1%}")
        print(f"  ROI: {perf['roi']:+.1f}%")
        print(f"  Total Profit: {perf['total_profit']:+.2f} units")

        # By sport
        by_sport = tracker.get_performance_by_sport()
        if not by_sport.empty:
            print(f"\nBy Sport:")
            for _, row in by_sport.iterrows():
                print(f"  {row['sport']}: {row['wins']}-{row['losses']} ({row['win_rate']:.1%}), ROI: {row['roi']:+.1f}%")

    elif command == 'recommendations':
        # Show current recommendations
        sport_filter = sys.argv[2] if len(sys.argv) > 2 else None
        recs = tracker.get_recommendations(sport=sport_filter)

        if recs.empty:
            print("No pending recommendations")
        else:
            print(f"\nPending Recommendations ({len(recs)}):")
            for _, rec in recs.iterrows():
                print(f"\n  {rec['away_team']} @ {rec['home_team']} ({rec['sport']} Week {rec['week']})")
                print(f"    Model: {rec['model_spread']:+.1f}, Vegas: {rec['vegas_spread']:+.1f}")
                print(f"    Deviation: {rec['spread_deviation']:+.1f} ({rec['deviation_direction']})")
                print(f"    Recommended: {rec['recommended_side']} ({rec['recommended_units']:.0f} units)")
                print(f"    Expected WR: {rec['expected_win_rate']:.1%}")

    else:
        print(f"Unknown command: {command}")
