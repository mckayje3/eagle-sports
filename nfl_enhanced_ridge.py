"""
NFL Enhanced Ridge Model - Extended features for better predictions.

Adds to Simple Model:
- Dynamic per-team home field advantage
- Recent form and momentum (last 4 games)
- Drive efficiency metrics (PPD, YPD)
- Weather/dome factors
- Primetime game indicators
- Week of season (early vs late)

Features:
- Spread: 18 features
- Total: 14 features
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).parent / 'nfl_games.db'
MODEL_DIR = Path(__file__).parent / 'models'

# Constants - Optimized from feature analysis
DECAY = 0.96  # Higher decay works better - recent games more predictive in NFL
PREV_HALF_LIFE = 4.0
MIN_GAMES = 2  # Lower threshold provides better coverage


class NFLEnhancedModel:
    """Enhanced Ridge model for NFL with form, HCA, and drive metrics."""

    def __init__(self):
        self.team_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'yards': [], 'yards_wts': [],
            'turnovers': [], 'to_wts': [],
            'third_down_pct': [], 'td_wts': [],
            'margins': [], 'wins': [],
            # Drive efficiency
            'ppd': [], 'ppd_wts': [],
            'ypd': [], 'ypd_wts': [],
            'scoring_pct': [], 'sp_wts': [],
        }))

        self.team_hca_data = defaultdict(lambda: defaultdict(lambda: {
            'home_margins': [], 'away_margins': []
        }))

        self.prev_ratings = {}
        self.prev_hca = {}
        self.last_game = {}
        self.league_avg = {
            'ppg': 22.0, 'papg': 22.0, 'yards': 330.0, 'turnovers': 1.3,
            'ppd': 1.9, 'ypd': 28.0, 'scoring_pct': 0.35
        }

        self.spread_model = None
        self.total_model = None
        self.spread_scaler = StandardScaler()
        self.total_scaler = StandardScaler()

        self.spread_X, self.spread_y = [], []
        self.total_X, self.total_y = [], []

    def _wavg(self, vals, wts):
        if not vals or not wts or len(vals) != len(wts):
            return None
        return float(np.average(vals, weights=wts))

    def _get_rest(self, tid, date):
        if tid not in self.last_game:
            return 7
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return min((curr - last).days, 14)
        except Exception:
            return 7

    def _get_stats(self, tid, season):
        td = self.team_stats[tid][season]
        n = len(td['ppg'])

        if n == 0:
            prev = self.prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', self.league_avg['ppg']),
                'papg': prev.get('papg', self.league_avg['papg']),
                'yards': prev.get('yards', self.league_avg['yards']),
                'turnovers': prev.get('turnovers', self.league_avg['turnovers']),
                'third_down_pct': prev.get('third_down_pct', 40.0),
                'ppd': prev.get('ppd', self.league_avg['ppd']),
                'ypd': prev.get('ypd', self.league_avg['ypd']),
                'scoring_pct': prev.get('scoring_pct', self.league_avg['scoring_pct']),
                'games': 0,
                'margins': [],
                'wins': [],
            }

        ppg = self._wavg(td['ppg'], td['wts'])
        papg = self._wavg(td['papg'], td['wts'])
        yards = self._wavg(td['yards'], td['yards_wts']) if td['yards'] else None
        to = self._wavg(td['turnovers'], td['to_wts']) if td['turnovers'] else None
        td_pct = self._wavg(td['third_down_pct'], td['td_wts']) if td['third_down_pct'] else None
        ppd = self._wavg(td['ppd'], td['ppd_wts']) if td['ppd'] else None
        ypd = self._wavg(td['ypd'], td['ypd_wts']) if td['ypd'] else None
        sp = self._wavg(td['scoring_pct'], td['sp_wts']) if td['scoring_pct'] else None

        prev = self.prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', self.league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', self.league_avg['papg']) + (1 - blend) * papg,
            'yards': yards if yards else prev.get('yards', self.league_avg['yards']),
            'turnovers': to if to else prev.get('turnovers', self.league_avg['turnovers']),
            'third_down_pct': td_pct if td_pct else prev.get('third_down_pct', 40.0),
            'ppd': ppd if ppd else prev.get('ppd', self.league_avg['ppd']),
            'ypd': ypd if ypd else prev.get('ypd', self.league_avg['ypd']),
            'scoring_pct': sp if sp else prev.get('scoring_pct', self.league_avg['scoring_pct']),
            'games': n,
            'margins': td['margins'],
            'wins': td['wins'],
        }

    def _get_dynamic_hca(self, home_id, season):
        """Calculate dynamic per-team HCA."""
        hd = self.team_hca_data[home_id][season]
        n_home = len(hd['home_margins'])
        n_away = len(hd['away_margins'])
        total = n_home + n_away

        if total == 0:
            return self.prev_hca.get(home_id, 2.5)  # NFL HCA default ~2.5

        if n_home > 0 and n_away > 0:
            raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
            raw = max(-3, min(raw, 8))  # Clip to reasonable range
        else:
            raw = 2.5

        # Shrink toward league average
        shrunk = 2.5 + 0.5 * (raw - 2.5)
        prev = self.prev_hca.get(home_id, 2.5)
        blend = 0.5 ** (total / 10.0)  # NFL: faster adaptation due to fewer games

        return blend * prev + (1 - blend) * shrunk

    def _recent_form(self, margins, n=4):
        """Average margin over last n games."""
        if len(margins) < n:
            return 0.0
        return float(np.mean(margins[-n:]))

    def _momentum(self, margins, n=4):
        """Trend in margins (recent vs earlier)."""
        if len(margins) < n:
            return 0.0
        recent = margins[-n:]
        return np.mean(recent[n//2:]) - np.mean(recent[:n//2])

    def _streak(self, wins):
        """Current win/loss streak."""
        if not wins:
            return 0
        s, last = 0, wins[-1]
        for w in reversed(wins):
            if w == last:
                s += 1
            else:
                break
        return s if last == 1 else -s

    def extract_spread_features(self, hid, aid, season, date, week,
                                 is_dome=False, is_primetime=False, neutral_site=False):
        """Extract spread features - optimized based on feature analysis.

        Key findings:
        - Yards-based features (ypd, scoring_pct) outperform points-based
        - Form/momentum features add value
        - Dynamic HCA helps
        """
        hs = self._get_stats(hid, season)
        aws = self._get_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        # Dynamic HCA (0 for neutral site)
        hca = 0.0 if neutral_site else self._get_dynamic_hca(hid, season)

        # Bye week advantage
        h_post_bye = 1.0 if hr >= 13 else 0.0
        a_post_bye = 1.0 if ar >= 13 else 0.0

        return np.array([
            # Yards differentials (best performing in analysis)
            hs['yards'] - aws['yards'],
            hs['ypd'] - aws['ypd'],  # Key: yards per drive

            # Drive efficiency (also showed value)
            hs['scoring_pct'] - aws['scoring_pct'],
            hs['ppd'] - aws['ppd'],

            # Turnovers (supporting feature)
            hs['turnovers'] - aws['turnovers'],

            # Form and momentum
            self._recent_form(hs['margins']) - self._recent_form(aws['margins']),
            self._momentum(hs['margins']) - self._momentum(aws['margins']),
            self._streak(hs['wins']) - self._streak(aws['wins']),

            # Rest and bye
            min(hr, 10) - min(ar, 10),
            h_post_bye - a_post_bye,

            # HCA
            hca,

            # Context
            1.0 if is_dome else 0.0,
            1.0 if is_primetime else 0.0,
            min(week / 17.0, 1.0),  # Season progress

            # Reliability
            min(hs['games'] / 10.0, 1.0),
            min(aws['games'] / 10.0, 1.0),
        ])

    def extract_total_features(self, hid, aid, season, date, week, is_dome=False):
        """Extract 14 total features."""
        hs = self._get_stats(hid, season)
        aws = self._get_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            hs['ppg'] + aws['ppg'],
            hs['papg'] + aws['papg'],
            hs['yards'] + aws['yards'],
            hs['turnovers'] + aws['turnovers'],
            # Drive efficiency (offensive combined)
            hs['ppd'] + aws['ppd'],
            hs['scoring_pct'] + aws['scoring_pct'],
            # Form volatility
            abs(self._recent_form(hs['margins'])) + abs(self._recent_form(aws['margins'])),
            # Rest
            1.0 if hr >= 13 else 0.0,
            1.0 if ar >= 13 else 0.0,
            # Context
            1.0 if is_dome else 0.0,
            min(week / 17.0, 1.0),
            # Reliability
            min(hs['games'] / 10.0, 1.0),
            min(aws['games'] / 10.0, 1.0),
            (hs['games'] + aws['games']) / 34.0,
        ])

    def predict(self, hid, aid, season, date, week, **kwargs):
        """Get predictions."""
        spread_feat = self.extract_spread_features(hid, aid, season, date, week, **kwargs)
        total_feat = self.extract_total_features(
            hid, aid, season, date, week,
            is_dome=kwargs.get('is_dome', False)
        )

        spread = None
        total = None

        if spread_feat is not None and self.spread_model is not None:
            X = self.spread_scaler.transform(spread_feat.reshape(1, -1))
            spread = self.spread_model.predict(X)[0]

        if total_feat is not None and self.total_model is not None:
            X = self.total_scaler.transform(total_feat.reshape(1, -1))
            total = self.total_model.predict(X)[0]

        return {
            'spread': spread,
            'total': total,
            'spread_feat': spread_feat,
            'total_feat': total_feat,
        }

    def update(self, tid, season, date, pf, pa, is_home, yards=None, turnovers=None,
               third_down_pct=None, ppd=None, ypd=None, scoring_pct=None):
        """Update team stats."""
        margin = pf - pa
        td = self.team_stats[tid][season]

        # Core stats with decay
        td['wts'] = [w * DECAY for w in td['wts']]
        td['ppg'].append(pf)
        td['papg'].append(pa)
        td['wts'].append(1.0)
        td['margins'].append(margin)
        td['wins'].append(1 if margin > 0 else 0)

        if pd.notna(yards):
            td['yards_wts'] = [w * DECAY for w in td['yards_wts']]
            td['yards'].append(yards)
            td['yards_wts'].append(1.0)

        if pd.notna(turnovers):
            td['to_wts'] = [w * DECAY for w in td['to_wts']]
            td['turnovers'].append(turnovers)
            td['to_wts'].append(1.0)

        if pd.notna(third_down_pct):
            td['td_wts'] = [w * DECAY for w in td['td_wts']]
            td['third_down_pct'].append(third_down_pct)
            td['td_wts'].append(1.0)

        # Drive efficiency
        if pd.notna(ppd):
            td['ppd_wts'] = [w * DECAY for w in td['ppd_wts']]
            td['ppd'].append(ppd)
            td['ppd_wts'].append(1.0)

        if pd.notna(ypd):
            td['ypd_wts'] = [w * DECAY for w in td['ypd_wts']]
            td['ypd'].append(ypd)
            td['ypd_wts'].append(1.0)

        if pd.notna(scoring_pct):
            td['sp_wts'] = [w * DECAY for w in td['sp_wts']]
            td['scoring_pct'].append(scoring_pct)
            td['sp_wts'].append(1.0)

        # HCA data
        hd = self.team_hca_data[tid][season]
        if is_home:
            hd['home_margins'].append(margin)
        else:
            hd['away_margins'].append(-margin)

        self.last_game[tid] = date

    def set_previous_season(self, season):
        """Set previous season ratings for blending."""
        prev = season - 1

        for tid in self.team_stats:
            if prev in self.team_stats[tid]:
                td = self.team_stats[tid][prev]
                if td['ppg']:
                    self.prev_ratings[tid] = {
                        'ppg': np.mean(td['ppg']),
                        'papg': np.mean(td['papg']),
                        'yards': np.mean(td['yards']) if td['yards'] else 330.0,
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 1.3,
                        'third_down_pct': np.mean(td['third_down_pct']) if td['third_down_pct'] else 40.0,
                        'ppd': np.mean(td['ppd']) if td['ppd'] else 1.9,
                        'ypd': np.mean(td['ypd']) if td['ypd'] else 28.0,
                        'scoring_pct': np.mean(td['scoring_pct']) if td['scoring_pct'] else 0.35,
                    }

        for tid in self.team_hca_data:
            if prev in self.team_hca_data[tid]:
                hd = self.team_hca_data[tid][prev]
                if hd['home_margins'] and hd['away_margins']:
                    raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
                    raw = max(-3, min(raw, 8))
                    self.prev_hca[tid] = 2.5 + 0.5 * (raw - 2.5)

        self.last_game.clear()

    def train_models(self):
        """Train Ridge models."""
        if len(self.spread_X) >= 50:
            X = np.array(self.spread_X)
            y = np.array(self.spread_y)
            self.spread_scaler.fit(X)
            X_s = self.spread_scaler.transform(X)
            self.spread_model = Ridge(alpha=1.0).fit(X_s, y)

        if len(self.total_X) >= 50:
            X = np.array(self.total_X)
            y = np.array(self.total_y)
            self.total_scaler.fit(X)
            X_s = self.total_scaler.transform(X)
            self.total_model = Ridge(alpha=1.0).fit(X_s, y)


def train_and_evaluate():
    """Train model and evaluate."""
    print("=" * 70)
    print("NFL ENHANCED MODEL TRAINING")
    print("=" * 70)

    conn = sqlite3.connect(str(DB_PATH))

    # Load games with drive stats
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site, g.is_dome,
               hs.total_yards as home_yards, hs.turnovers as home_to,
               hs.third_down_conversions as home_3dc, hs.third_down_attempts as home_3da,
               aws.total_yards as away_yards, aws.turnovers as away_to,
               aws.third_down_conversions as away_3dc, aws.third_down_attempts as away_3da,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id
            AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id
            AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date, g.week
    ''', conn)

    # Load drive efficiency per game
    drives = pd.read_sql_query('''
        SELECT d.game_id, g.home_team_id, g.away_team_id,
               d.team_id, d.yards, d.is_score
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE g.completed = 1
    ''', conn)
    conn.close()

    # Calculate drive efficiency per team per game
    if not drives.empty:
        drive_stats = drives.groupby(['game_id', 'team_id']).agg({
            'yards': ['sum', 'count'],
            'is_score': 'sum'
        }).reset_index()
        drive_stats.columns = ['game_id', 'team_id', 'total_yards', 'num_drives', 'scores']
        drive_stats['ppd'] = drive_stats['scores'] * 7 / drive_stats['num_drives'].replace(0, 1)  # Approx
        drive_stats['ypd'] = drive_stats['total_yards'] / drive_stats['num_drives'].replace(0, 1)
        drive_stats['scoring_pct'] = drive_stats['scores'] / drive_stats['num_drives'].replace(0, 1)

        # Merge with games
        home_drives = drive_stats.copy()
        home_drives = home_drives.rename(columns={
            'ppd': 'home_ppd', 'ypd': 'home_ypd', 'scoring_pct': 'home_scoring_pct'
        })
        games = games.merge(
            home_drives[['game_id', 'team_id', 'home_ppd', 'home_ypd', 'home_scoring_pct']],
            left_on=['game_id', 'home_team_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        ).drop(columns=['team_id'], errors='ignore')

        away_drives = drive_stats.copy()
        away_drives = away_drives.rename(columns={
            'ppd': 'away_ppd', 'ypd': 'away_ypd', 'scoring_pct': 'away_scoring_pct'
        })
        games = games.merge(
            away_drives[['game_id', 'team_id', 'away_ppd', 'away_ypd', 'away_scoring_pct']],
            left_on=['game_id', 'away_team_id'],
            right_on=['game_id', 'team_id'],
            how='left'
        ).drop(columns=['team_id'], errors='ignore')

    # Calculate third down percentage
    games['home_3d_pct'] = games.apply(
        lambda r: 100 * r['home_3dc'] / r['home_3da']
        if pd.notna(r['home_3dc']) and r['home_3da'] > 0 else None,
        axis=1
    )
    games['away_3d_pct'] = games.apply(
        lambda r: 100 * r['away_3dc'] / r['away_3da']
        if pd.notna(r['away_3dc']) and r['away_3da'] > 0 else None,
        axis=1
    )

    # Detect primetime (Thursday, Monday, or Sunday night)
    # Extract just the date part (first 10 chars) to avoid timezone issues
    games['day_of_week'] = games['date'].str[:10].apply(
        lambda x: pd.to_datetime(x).dayofweek if pd.notna(x) else 6
    )
    games['is_primetime'] = games['day_of_week'].isin([0, 3, 6])  # Mon, Thu, or check time

    print(f"Total games: {len(games)}")
    print(f"Games with drive data: {games['home_ppd'].notna().sum()}")

    model = NFLEnhancedModel()
    results = []
    seasons = sorted(games['season'].unique())

    for season in seasons:
        if season > seasons[0]:
            model.set_previous_season(season)

        season_games = games[games['season'] == season].copy()
        print(f"\nSeason {season}: {len(season_games)} games")

        for _, g in season_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_spread = g['away_score'] - g['home_score']
            actual_total = g['home_score'] + g['away_score']
            vegas_spread = g['vegas_spread']
            vegas_total = g['vegas_total']
            neutral = g['neutral_site'] == 1
            is_dome = g.get('is_dome', 0) == 1
            is_primetime = g.get('is_primetime', False)

            # Get prediction
            preds = model.predict(
                hid, aid, season, g['date'], g['week'],
                is_dome=is_dome, is_primetime=is_primetime, neutral_site=neutral
            )

            # Accumulate training data
            if preds['spread_feat'] is not None:
                model.spread_X.append(preds['spread_feat'])
                model.spread_y.append(actual_spread)

            if preds['total_feat'] is not None:
                model.total_X.append(preds['total_feat'])
                model.total_y.append(actual_total)

            # Retrain periodically
            if len(model.spread_X) % 50 == 0:
                model.train_models()

            # Record result
            if preds['spread'] is not None and pd.notna(vegas_spread):
                spread_err = abs(preds['spread'] - actual_spread)
                vegas_err = abs(vegas_spread - actual_spread)

                edge = preds['spread'] - vegas_spread
                result = actual_spread - vegas_spread
                push = abs(result) < 0.5
                model_ats = bool(edge * result > 0) if not push else None

                results.append({
                    'season': season,
                    'week': g['week'],
                    'spread_err': spread_err,
                    'vegas_err': vegas_err,
                    'model_ats': model_ats,
                    'total_err': abs(preds['total'] - actual_total) if preds['total'] else None,
                    'vegas_total_err': abs(vegas_total - actual_total) if pd.notna(vegas_total) else None,
                })

            # Update state
            model.update(
                hid, season, g['date'], g['home_score'], g['away_score'],
                is_home=not neutral,
                yards=g['home_yards'], turnovers=g['home_to'],
                third_down_pct=g['home_3d_pct'],
                ppd=g.get('home_ppd'), ypd=g.get('home_ypd'),
                scoring_pct=g.get('home_scoring_pct')
            )
            model.update(
                aid, season, g['date'], g['away_score'], g['home_score'],
                is_home=False,
                yards=g['away_yards'], turnovers=g['away_to'],
                third_down_pct=g['away_3d_pct'],
                ppd=g.get('away_ppd'), ypd=g.get('away_ypd'),
                scoring_pct=g.get('away_scoring_pct')
            )

    # Evaluate
    df = pd.DataFrame(results)
    print(f"\nTotal results: {len(df)}")

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    for season in df['season'].unique():
        mask = df['season'] == season
        spread_mae = df.loc[mask, 'spread_err'].mean()
        vegas_mae = df.loc[mask, 'vegas_err'].mean()
        ats_mask = df.loc[mask, 'model_ats'].dropna()
        ats_wins = ats_mask.mean() * 100 if len(ats_mask) > 0 else 0

        print(f"\n{season}:")
        print(f"  Spread MAE: {spread_mae:.2f} (Vegas: {vegas_mae:.2f})")
        print(f"  ATS: {ats_wins:.1f}% ({int(ats_mask.sum())}/{len(ats_mask)})")

    # Overall
    overall_ats = df['model_ats'].dropna()
    print(f"\nOverall:")
    print(f"  Spread MAE: {df['spread_err'].mean():.2f} (Vegas: {df['vegas_err'].mean():.2f})")
    print(f"  ATS: {overall_ats.mean() * 100:.1f}% ({int(overall_ats.sum())}/{len(overall_ats)})")

    # By week range
    print("\nBy Week Range:")
    for start, end in [(1, 4), (5, 9), (10, 14), (15, 18)]:
        mask = (df['week'] >= start) & (df['week'] <= end)
        if mask.sum() > 0:
            ats_mask = df.loc[mask, 'model_ats'].dropna()
            ats_pct = ats_mask.mean() * 100 if len(ats_mask) > 0 else 0
            print(f"  Weeks {start}-{end}: ATS {ats_pct:.1f}% ({len(ats_mask)} games)")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / 'nfl_enhanced_model.pkl', 'wb') as f:
        pickle.dump({
            'spread_model': model.spread_model,
            'total_model': model.total_model,
            'spread_scaler': model.spread_scaler,
            'total_scaler': model.total_scaler,
            'prev_ratings': dict(model.prev_ratings),
            'prev_hca': dict(model.prev_hca),
        }, f)

    print(f"\nModel saved to {MODEL_DIR / 'nfl_enhanced_model.pkl'}")

    return model, df


if __name__ == '__main__':
    train_and_evaluate()
