"""
NBA Segmented Models - Separate models for early/mid/late season

Key insight: Our model performs differently across the season:
- Early: Less team data, model struggles
- Mid: Best performance, closest to Vegas
- Late: Fatigue/tanking effects, model struggles

This module trains separate ridge models for each segment to preserve
mid-season edge while improving early/late performance.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'nba_games.db'
MODEL_DIR = Path(__file__).parent / 'models'

# Season segment boundaries (by games played in season)
SEGMENTS = {
    'early': (0, 300),      # First ~300 games (~25%)
    'mid': (300, 800),      # Games 300-800 (~40%)
    'late': (800, 9999),    # Games 800+ (~35%)
}


class SegmentedNBAModel:
    """Manages separate models for each season segment."""

    DECAY = 0.97
    MIN_GAMES = 10

    def __init__(self):
        self.models = {}  # segment -> {'spread': Ridge, 'total': Ridge}
        self.scalers = {}  # segment -> {'spread': Scaler, 'total': Scaler}
        self.team_stats = None
        self.last_game = {}

    def _reset_team_stats(self):
        """Reset team stats for walk-forward training."""
        self.team_stats = defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'home_ppg': [], 'home_papg': [],
            'away_ppg': [], 'away_papg': [],
            'margins': [], 'wins': []
        })
        self.last_game = {}

    def _wavg(self, vals, wts):
        """Weighted average with decay weights."""
        if not vals or not wts:
            return None
        n = min(len(vals), len(wts))
        return np.average(vals[-n:], weights=wts[-n:])

    def _get_rest(self, tid, date):
        """Get rest days for a team."""
        if tid not in self.last_game:
            return 3
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 2

    def _extract_spread_features(self, hid, aid, date, segment='mid'):
        """Extract spread prediction features."""
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]

        if len(hs['ppg']) < self.MIN_GAMES or len(aws['ppg']) < self.MIN_GAMES:
            return None

        h_ppg = self._wavg(hs['ppg'], hs['wts'])
        h_papg = self._wavg(hs['papg'], hs['wts'])
        a_ppg = self._wavg(aws['ppg'], aws['wts'])
        a_papg = self._wavg(aws['papg'], aws['wts'])

        # Recent form (last 5 games)
        h_recent = np.mean(hs['ppg'][-5:]) if len(hs['ppg']) >= 5 else h_ppg
        a_recent = np.mean(aws['ppg'][-5:]) if len(aws['ppg']) >= 5 else a_ppg

        # Momentum/trend
        def get_trend(margins):
            if len(margins) < 6:
                return 0
            return np.mean(margins[-3:]) - np.mean(margins[-6:-3])

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
            return streak if last else -streak

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        # Base features (all segments)
        features = [
            h_ppg - a_ppg,                              # 0: ppg diff
            h_papg - a_papg,                            # 1: defensive diff
            (h_ppg - h_papg) - (a_ppg - a_papg),        # 2: net rating diff
            h_recent - a_recent,                        # 3: recent form diff
            get_trend(hs['margins']) - get_trend(aws['margins']),  # 4: momentum
            get_streak(hs['wins']) - get_streak(aws['wins']),      # 5: streak
            hr - ar,                                    # 6: rest diff
            1 if hr == 0 else 0,                        # 7: home b2b
            1 if ar == 0 else 0,                        # 8: away b2b
            2.0,                                        # 9: HCA baseline
            min(len(hs['ppg']) / 30, 1),               # 10: home sample
            min(len(aws['ppg']) / 30, 1),              # 11: away sample
        ]

        # Segment-specific features
        if segment == 'early':
            # Early season: weight sample size more, less trust in trends
            features.extend([
                min(len(hs['ppg']) / 15, 1),           # 12: stricter sample check
                min(len(aws['ppg']) / 15, 1),          # 13
            ])
        elif segment == 'mid':
            # Mid season: full feature set, trends matter
            h_home_ppg = np.mean(hs['home_ppg'][-10:]) if len(hs['home_ppg']) >= 5 else h_ppg
            a_away_ppg = np.mean(aws['away_ppg'][-10:]) if len(aws['away_ppg']) >= 5 else a_ppg
            features.extend([
                h_home_ppg - h_ppg,                    # 12: home boost
                a_away_ppg - a_ppg,                    # 13: away adjustment
                get_trend(hs['margins']),             # 14: home momentum alone
                get_trend(aws['margins']),            # 15: away momentum alone
            ])
        else:  # late
            # Late season: add fatigue/rest emphasis
            features.extend([
                hr,                                    # 12: raw home rest
                ar,                                    # 13: raw away rest
                1 if hr >= 2 and ar == 0 else 0,      # 14: rest advantage
                1 if ar >= 2 and hr == 0 else 0,      # 15: rest disadvantage
            ])

        return np.array(features)

    def _extract_total_features(self, hid, aid, date, segment='mid'):
        """Extract total prediction features."""
        hs = self.team_stats[hid]
        aws = self.team_stats[aid]

        if len(hs['ppg']) < self.MIN_GAMES or len(aws['ppg']) < self.MIN_GAMES:
            return None

        h_ppg = self._wavg(hs['ppg'], hs['wts'])
        h_papg = self._wavg(hs['papg'], hs['wts'])
        a_ppg = self._wavg(aws['ppg'], aws['wts'])
        a_papg = self._wavg(aws['papg'], aws['wts'])

        h_recent = np.mean(hs['ppg'][-5:]) if len(hs['ppg']) >= 5 else h_ppg
        a_recent = np.mean(aws['ppg'][-5:]) if len(aws['ppg']) >= 5 else a_ppg

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        # Base features
        features = [
            h_ppg + a_ppg,                             # 0: combined offense
            h_papg + a_papg,                           # 1: combined defense
            (h_ppg + a_papg) / 2,                      # 2: expected home pts
            (a_ppg + h_papg) / 2,                      # 3: expected away pts
            h_recent + a_recent,                       # 4: recent scoring
            1 if hr == 0 else 0,                       # 5: home b2b
            1 if ar == 0 else 0,                       # 6: away b2b
            min(len(hs['ppg']) / 30, 1),              # 7: home sample
            min(len(aws['ppg']) / 30, 1),             # 8: away sample
        ]

        # Segment-specific
        if segment == 'early':
            features.extend([
                min(len(hs['ppg']) / 15, 1),
                min(len(aws['ppg']) / 15, 1),
            ])
        elif segment == 'mid':
            h_home_ppg = np.mean(hs['home_ppg'][-10:]) if len(hs['home_ppg']) >= 5 else h_ppg
            a_away_ppg = np.mean(aws['away_ppg'][-10:]) if len(aws['away_ppg']) >= 5 else a_ppg
            features.extend([
                h_home_ppg + a_away_ppg,               # venue-adjusted total
                abs(h_ppg - a_ppg),                    # mismatch indicator
            ])
        else:  # late
            features.extend([
                hr + ar,                               # combined rest
                1 if hr == 0 or ar == 0 else 0,       # any b2b
            ])

        return np.array(features)

    def _update_team(self, tid, pf, pa, date, is_home, won):
        """Update team stats after a game."""
        ts = self.team_stats[tid]
        ts['wts'] = [w * self.DECAY for w in ts['wts']]
        ts['ppg'].append(pf)
        ts['papg'].append(pa)
        ts['wts'].append(1.0)
        ts['margins'].append(pf - pa)
        ts['wins'].append(1 if won else 0)

        if is_home:
            ts['home_ppg'].append(pf)
            ts['home_papg'].append(pa)
        else:
            ts['away_ppg'].append(pf)
            ts['away_papg'].append(pa)

        self.last_game[tid] = date

    def _get_segment(self, season_game_num):
        """Determine which segment a game falls into."""
        for seg_name, (low, high) in SEGMENTS.items():
            if low <= season_game_num < high:
                return seg_name
        return 'late'

    def train_and_evaluate(self):
        """Train segmented models with walk-forward validation."""
        logger.info("=" * 70)
        logger.info("TRAINING SEGMENTED NBA MODELS")
        logger.info("=" * 70)

        conn = sqlite3.connect(str(DB_PATH))
        games = pd.read_sql_query('''
            SELECT g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
                   g.home_score, g.away_score,
                   o.latest_spread as vegas_spread, o.latest_total as vegas_total
            FROM games g
            LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.completed = 1 AND g.home_score > 0
            ORDER BY g.date
        ''', conn)
        conn.close()

        games = games[games['vegas_spread'].notna() & games['vegas_total'].notna()].copy()
        games['actual_spread'] = games['away_score'] - games['home_score']
        games['actual_total'] = games['home_score'] + games['away_score']

        logger.info(f"Total games with Vegas lines: {len(games)}")

        # Track season game numbers
        season_game_counts = defaultdict(int)
        games['season_game_num'] = 0
        for idx, row in games.iterrows():
            season_game_counts[row['season']] += 1
            games.at[idx, 'season_game_num'] = season_game_counts[row['season']]

        games['segment'] = games['season_game_num'].apply(self._get_segment)

        # Walk-forward training
        self._reset_team_stats()

        # Training data by segment
        train_data = {seg: {'X_spread': [], 'y_spread': [], 'X_total': [], 'y_total': []}
                      for seg in SEGMENTS.keys()}

        # Results tracking
        results = {seg: [] for seg in SEGMENTS.keys()}
        results['unified'] = []  # For comparison with single model

        # Unified model training data
        unified_X_spread, unified_y_spread = [], []
        unified_X_total, unified_y_total = [], []

        for _, g in games.iterrows():
            segment = g['segment']

            # Extract features for this segment
            spread_feat = self._extract_spread_features(
                g['home_team_id'], g['away_team_id'], g['date'], segment
            )
            total_feat = self._extract_total_features(
                g['home_team_id'], g['away_team_id'], g['date'], segment
            )

            # Also extract mid-season features for unified comparison
            unified_spread_feat = self._extract_spread_features(
                g['home_team_id'], g['away_team_id'], g['date'], 'mid'
            )
            unified_total_feat = self._extract_total_features(
                g['home_team_id'], g['away_team_id'], g['date'], 'mid'
            )

            # Make predictions if we have enough data
            seg_data = train_data[segment]
            min_samples = 50

            if spread_feat is not None and len(seg_data['X_spread']) >= min_samples:
                # Train segment model
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(np.array(seg_data['X_spread']))
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_scaled, np.array(seg_data['y_spread']))

                pred_spread = ridge.predict(scaler.transform(spread_feat.reshape(1, -1)))[0]

                # Total prediction
                t_scaler = StandardScaler()
                X_t_scaled = t_scaler.fit_transform(np.array(seg_data['X_total']))
                t_ridge = Ridge(alpha=1.0)
                t_ridge.fit(X_t_scaled, np.array(seg_data['y_total']))

                pred_total = t_ridge.predict(t_scaler.transform(total_feat.reshape(1, -1)))[0]

                results[segment].append({
                    'pred_spread': pred_spread,
                    'pred_total': pred_total,
                    'actual_spread': g['actual_spread'],
                    'actual_total': g['actual_total'],
                    'vegas_spread': g['vegas_spread'],
                    'vegas_total': g['vegas_total'],
                    'season': g['season'],
                })

            # Unified model prediction
            if unified_spread_feat is not None and len(unified_X_spread) >= 100:
                u_scaler = StandardScaler()
                X_scaled = u_scaler.fit_transform(np.array(unified_X_spread))
                u_ridge = Ridge(alpha=1.0)
                u_ridge.fit(X_scaled, np.array(unified_y_spread))

                u_pred_spread = u_ridge.predict(u_scaler.transform(unified_spread_feat.reshape(1, -1)))[0]

                ut_scaler = StandardScaler()
                X_t_scaled = ut_scaler.fit_transform(np.array(unified_X_total))
                ut_ridge = Ridge(alpha=1.0)
                ut_ridge.fit(X_t_scaled, np.array(unified_y_total))

                u_pred_total = ut_ridge.predict(ut_scaler.transform(unified_total_feat.reshape(1, -1)))[0]

                results['unified'].append({
                    'pred_spread': u_pred_spread,
                    'pred_total': u_pred_total,
                    'actual_spread': g['actual_spread'],
                    'actual_total': g['actual_total'],
                    'vegas_spread': g['vegas_spread'],
                    'vegas_total': g['vegas_total'],
                    'season': g['season'],
                    'segment': segment,
                })

            # Add to training data
            if spread_feat is not None:
                seg_data['X_spread'].append(spread_feat)
                seg_data['y_spread'].append(g['actual_spread'])
                seg_data['X_total'].append(total_feat)
                seg_data['y_total'].append(g['actual_total'])

            if unified_spread_feat is not None:
                unified_X_spread.append(unified_spread_feat)
                unified_y_spread.append(g['actual_spread'])
                unified_X_total.append(unified_total_feat)
                unified_y_total.append(g['actual_total'])

            # Update team stats
            won = g['home_score'] > g['away_score']
            self._update_team(g['home_team_id'], g['home_score'], g['away_score'],
                            g['date'], True, won)
            self._update_team(g['away_team_id'], g['away_score'], g['home_score'],
                            g['date'], False, not won)

        # Analyze results
        self._analyze_results(results)

        # Train final models on all data
        self._train_final_models(train_data, unified_X_spread, unified_y_spread,
                                  unified_X_total, unified_y_total)

        return results

    def _analyze_results(self, results):
        """Analyze and compare segmented vs unified model performance."""
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS COMPARISON")
        logger.info("=" * 70)

        # Per-segment analysis
        logger.info("\n--- SEGMENTED MODELS (each segment trained separately) ---")
        for segment in SEGMENTS.keys():
            if not results[segment]:
                continue

            df = pd.DataFrame(results[segment])
            spread_mae = np.abs(df['pred_spread'] - df['actual_spread']).mean()
            vegas_spread_mae = np.abs(df['vegas_spread'] - df['actual_spread']).mean()
            total_mae = np.abs(df['pred_total'] - df['actual_total']).mean()
            vegas_total_mae = np.abs(df['vegas_total'] - df['actual_total']).mean()
            total_bias = (df['pred_total'] - df['actual_total']).mean()

            # ATS analysis
            df['edge'] = df['pred_spread'] - df['vegas_spread']
            df['result'] = df['actual_spread'] - df['vegas_spread']

            ats_wins = 0
            ats_losses = 0
            for _, r in df.iterrows():
                if abs(r['edge']) >= 3:  # Only count significant edges
                    if r['edge'] > 0 and r['result'] > 0.5:
                        ats_wins += 1
                    elif r['edge'] > 0 and r['result'] < -0.5:
                        ats_losses += 1
                    elif r['edge'] < 0 and r['result'] < -0.5:
                        ats_wins += 1
                    elif r['edge'] < 0 and r['result'] > 0.5:
                        ats_losses += 1

            ats_pct = ats_wins / (ats_wins + ats_losses) * 100 if (ats_wins + ats_losses) > 0 else 0

            logger.info(f"\n{segment.upper()} SEASON ({len(df)} games):")
            logger.info(f"  Spread MAE: Model {spread_mae:.2f} vs Vegas {vegas_spread_mae:.2f} "
                       f"({'BETTER' if spread_mae < vegas_spread_mae else 'worse'})")
            logger.info(f"  Total MAE:  Model {total_mae:.2f} vs Vegas {vegas_total_mae:.2f} "
                       f"({'BETTER' if total_mae < vegas_total_mae else 'worse'})")
            logger.info(f"  Total Bias: {total_bias:+.2f}")
            logger.info(f"  ATS (>=3pt edge): {ats_wins}-{ats_losses} ({ats_pct:.1f}%)")

        # Unified model by segment
        logger.info("\n--- UNIFIED MODEL (same model for all segments) ---")
        if results['unified']:
            df = pd.DataFrame(results['unified'])

            for segment in SEGMENTS.keys():
                seg_df = df[df['segment'] == segment]
                if len(seg_df) == 0:
                    continue

                spread_mae = np.abs(seg_df['pred_spread'] - seg_df['actual_spread']).mean()
                vegas_spread_mae = np.abs(seg_df['vegas_spread'] - seg_df['actual_spread']).mean()
                total_mae = np.abs(seg_df['pred_total'] - seg_df['actual_total']).mean()
                vegas_total_mae = np.abs(seg_df['vegas_total'] - seg_df['actual_total']).mean()

                logger.info(f"\n{segment.upper()} ({len(seg_df)} games):")
                logger.info(f"  Spread MAE: Model {spread_mae:.2f} vs Vegas {vegas_spread_mae:.2f}")
                logger.info(f"  Total MAE:  Model {total_mae:.2f} vs Vegas {vegas_total_mae:.2f}")

        # Overall comparison
        logger.info("\n--- OVERALL COMPARISON ---")

        # Combine segmented results
        all_segmented = []
        for segment in SEGMENTS.keys():
            all_segmented.extend(results[segment])

        if all_segmented:
            seg_df = pd.DataFrame(all_segmented)
            seg_spread_mae = np.abs(seg_df['pred_spread'] - seg_df['actual_spread']).mean()
            seg_total_mae = np.abs(seg_df['pred_total'] - seg_df['actual_total']).mean()
            seg_total_bias = (seg_df['pred_total'] - seg_df['actual_total']).mean()

            logger.info(f"\nSegmented Models Combined:")
            logger.info(f"  Spread MAE: {seg_spread_mae:.2f}")
            logger.info(f"  Total MAE:  {seg_total_mae:.2f}")
            logger.info(f"  Total Bias: {seg_total_bias:+.2f}")

        if results['unified']:
            uni_df = pd.DataFrame(results['unified'])
            uni_spread_mae = np.abs(uni_df['pred_spread'] - uni_df['actual_spread']).mean()
            uni_total_mae = np.abs(uni_df['pred_total'] - uni_df['actual_total']).mean()
            uni_total_bias = (uni_df['pred_total'] - uni_df['actual_total']).mean()

            logger.info(f"\nUnified Model:")
            logger.info(f"  Spread MAE: {uni_spread_mae:.2f}")
            logger.info(f"  Total MAE:  {uni_total_mae:.2f}")
            logger.info(f"  Total Bias: {uni_total_bias:+.2f}")

            if all_segmented:
                spread_diff = seg_spread_mae - uni_spread_mae
                total_diff = seg_total_mae - uni_total_mae
                logger.info(f"\nSegmented vs Unified:")
                logger.info(f"  Spread: {spread_diff:+.3f} ({'SEGMENTED BETTER' if spread_diff < 0 else 'unified better'})")
                logger.info(f"  Total:  {total_diff:+.3f} ({'SEGMENTED BETTER' if total_diff < 0 else 'unified better'})")

    def _train_final_models(self, train_data, unified_X_spread, unified_y_spread,
                            unified_X_total, unified_y_total):
        """Train and save final production models."""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING FINAL MODELS")
        logger.info("=" * 70)

        for segment in SEGMENTS.keys():
            seg_data = train_data[segment]
            if len(seg_data['X_spread']) < 100:
                logger.warning(f"Skipping {segment} - not enough data ({len(seg_data['X_spread'])} samples)")
                continue

            # Spread model
            spread_scaler = StandardScaler()
            X_spread = spread_scaler.fit_transform(np.array(seg_data['X_spread']))
            spread_model = Ridge(alpha=1.0)
            spread_model.fit(X_spread, np.array(seg_data['y_spread']))

            # Total model
            total_scaler = StandardScaler()
            X_total = total_scaler.fit_transform(np.array(seg_data['X_total']))
            total_model = Ridge(alpha=1.0)
            total_model.fit(X_total, np.array(seg_data['y_total']))

            self.models[segment] = {'spread': spread_model, 'total': total_model}
            self.scalers[segment] = {'spread': spread_scaler, 'total': total_scaler}

            logger.info(f"Trained {segment} model: {len(seg_data['X_spread'])} samples")

        # Save models
        save_path = MODEL_DIR / 'nba_segmented_models.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'segments': SEGMENTS,
            }, f)
        logger.info(f"\nModels saved to {save_path}")

    def load(self):
        """Load trained models from disk."""
        load_path = MODEL_DIR / 'nba_segmented_models.pkl'
        if not load_path.exists():
            logger.warning(f"No saved models found at {load_path}")
            return False

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        self.models = data['models']
        self.scalers = data['scalers']
        logger.info(f"Loaded segmented models: {list(self.models.keys())}")
        return True

    def predict(self, home_team_id, away_team_id, date, season_game_num):
        """Make prediction using appropriate segment model."""
        segment = self._get_segment(season_game_num)

        if segment not in self.models:
            # Fall back to mid if segment not available
            segment = 'mid' if 'mid' in self.models else list(self.models.keys())[0]

        spread_feat = self._extract_spread_features(home_team_id, away_team_id, date, segment)
        total_feat = self._extract_total_features(home_team_id, away_team_id, date, segment)

        if spread_feat is None or total_feat is None:
            return None, None

        spread_scaled = self.scalers[segment]['spread'].transform(spread_feat.reshape(1, -1))
        total_scaled = self.scalers[segment]['total'].transform(total_feat.reshape(1, -1))

        pred_spread = self.models[segment]['spread'].predict(spread_scaled)[0]
        pred_total = self.models[segment]['total'].predict(total_scaled)[0]

        return pred_spread, pred_total


def main():
    """Train and evaluate segmented models."""
    model = SegmentedNBAModel()
    results = model.train_and_evaluate()

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
