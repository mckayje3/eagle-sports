"""
CFB Enhanced Ridge Model - Extended features for better predictions.

Adds to Simple Model:
- Dynamic per-team home field advantage
- Recent form and momentum (last 4 games)
- Drive efficiency metrics (PPD, YPD)
- Conference game indicator
- Week of season (early vs late)

CFB-specific considerations:
- 12+ games per team (more data than NFL)
- Conference games more predictable
- Larger talent gaps between teams
- DECAY = 0.88 (lower than NFL due to more games)

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

DB_PATH = Path(__file__).parent / 'cfb_games.db'
MODEL_DIR = Path(__file__).parent / 'models'

# Constants - Optimized from CFB feature analysis
DECAY = 0.88  # Lower decay for CFB - more games provide stability
PREV_HALF_LIFE = 5.0
MIN_GAMES = 2


class CFBEnhancedModel:
    """Enhanced Ridge model for CFB with form, HCA, and drive metrics."""

    def __init__(self):
        self.team_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'yards': [], 'yards_wts': [],
            'pass_yards': [], 'pass_wts': [],
            'rush_yards': [], 'rush_wts': [],
            'turnovers': [], 'to_wts': [],
            'first_downs': [], 'fd_wts': [],
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
            'ppg': 28.0, 'papg': 28.0, 'yards': 400.0,
            'pass_yards': 230.0, 'rush_yards': 170.0,
            'turnovers': 1.5, 'first_downs': 20.0,
            'ppd': 2.2, 'ypd': 32.0, 'scoring_pct': 0.38
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
                'pass_yards': prev.get('pass_yards', self.league_avg['pass_yards']),
                'rush_yards': prev.get('rush_yards', self.league_avg['rush_yards']),
                'turnovers': prev.get('turnovers', self.league_avg['turnovers']),
                'first_downs': prev.get('first_downs', self.league_avg['first_downs']),
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
        pass_yds = self._wavg(td['pass_yards'], td['pass_wts']) if td['pass_yards'] else None
        rush_yds = self._wavg(td['rush_yards'], td['rush_wts']) if td['rush_yards'] else None
        to = self._wavg(td['turnovers'], td['to_wts']) if td['turnovers'] else None
        fd = self._wavg(td['first_downs'], td['fd_wts']) if td['first_downs'] else None
        ppd = self._wavg(td['ppd'], td['ppd_wts']) if td['ppd'] else None
        ypd = self._wavg(td['ypd'], td['ypd_wts']) if td['ypd'] else None
        sp = self._wavg(td['scoring_pct'], td['sp_wts']) if td['scoring_pct'] else None

        prev = self.prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', self.league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', self.league_avg['papg']) + (1 - blend) * papg,
            'yards': yards if yards else prev.get('yards', self.league_avg['yards']),
            'pass_yards': pass_yds if pass_yds else prev.get('pass_yards', self.league_avg['pass_yards']),
            'rush_yards': rush_yds if rush_yds else prev.get('rush_yards', self.league_avg['rush_yards']),
            'turnovers': to if to else prev.get('turnovers', self.league_avg['turnovers']),
            'first_downs': fd if fd else prev.get('first_downs', self.league_avg['first_downs']),
            'ppd': ppd if ppd else prev.get('ppd', self.league_avg['ppd']),
            'ypd': ypd if ypd else prev.get('ypd', self.league_avg['ypd']),
            'scoring_pct': sp if sp else prev.get('scoring_pct', self.league_avg['scoring_pct']),
            'games': n,
            'margins': td['margins'],
            'wins': td['wins'],
        }

    def _get_dynamic_hca(self, home_id, season):
        """Calculate dynamic per-team HCA for CFB."""
        hd = self.team_hca_data[home_id][season]
        n_home = len(hd['home_margins'])
        n_away = len(hd['away_margins'])
        total = n_home + n_away

        if total == 0:
            return self.prev_hca.get(home_id, 3.5)  # CFB HCA default ~3.5

        if n_home > 0 and n_away > 0:
            raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
            raw = max(-3, min(raw, 10))  # CFB can have larger HCA
        else:
            raw = 3.5

        # Shrink toward league average
        shrunk = 3.5 + 0.5 * (raw - 3.5)
        prev = self.prev_hca.get(home_id, 3.5)
        blend = 0.5 ** (total / 8.0)  # CFB: slightly slower adaptation

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
                                 is_conf_game=False, neutral_site=False):
        """Extract spread features - optimized based on CFB feature analysis.

        Key CFB findings:
        - Kitchen Sink combo works best (55% ATS)
        - Conference games more predictable
        - Lower decay (0.88) optimal
        - Drive efficiency adds value
        """
        hs = self._get_stats(hid, season)
        aws = self._get_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        # Dynamic HCA (0 for neutral site)
        hca = 0.0 if neutral_site else self._get_dynamic_hca(hid, season)

        return np.array([
            # Core differentials (from Kitchen Sink)
            hs['ppg'] - aws['ppg'],
            hs['papg'] - aws['papg'],
            hs['yards'] - aws['yards'],
            hs['pass_yards'] - aws['pass_yards'],
            hs['rush_yards'] - aws['rush_yards'],
            hs['turnovers'] - aws['turnovers'],
            hs['first_downs'] - aws['first_downs'],

            # Drive efficiency (adds value in CFB)
            hs['ypd'] - aws['ypd'],
            hs['scoring_pct'] - aws['scoring_pct'],

            # Form and momentum
            self._recent_form(hs['margins']) - self._recent_form(aws['margins']),
            self._momentum(hs['margins']) - self._momentum(aws['margins']),
            self._streak(hs['wins']) - self._streak(aws['wins']),

            # Rest
            min(hr, 10) - min(ar, 10),

            # HCA
            hca,

            # Context
            1.0 if is_conf_game else 0.0,
            min(week / 14.0, 1.0),  # Season progress (CFB weeks 1-14)

            # Reliability
            min(hs['games'] / 10.0, 1.0),
            min(aws['games'] / 10.0, 1.0),
        ])

    def extract_total_features(self, hid, aid, season, date, week):
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
            hs['ypd'] + aws['ypd'],
            hs['scoring_pct'] + aws['scoring_pct'],
            # Form volatility
            abs(self._recent_form(hs['margins'])) + abs(self._recent_form(aws['margins'])),
            # Rest difference
            min(hr, 10) - min(ar, 10),
            # Context
            min(week / 14.0, 1.0),
            # Reliability
            min(hs['games'] / 10.0, 1.0),
            min(aws['games'] / 10.0, 1.0),
            (hs['games'] + aws['games']) / 24.0,  # CFB: max ~12 games each
            1.0,  # Placeholder for balance
        ])

    def predict(self, hid, aid, season, date, week, **kwargs):
        """Get predictions."""
        spread_feat = self.extract_spread_features(hid, aid, season, date, week, **kwargs)
        total_feat = self.extract_total_features(hid, aid, season, date, week)

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

    def update(self, tid, season, date, pf, pa, is_home, yards=None, pass_yards=None,
               rush_yards=None, turnovers=None, first_downs=None,
               ppd=None, ypd=None, scoring_pct=None):
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

        if pd.notna(pass_yards):
            td['pass_wts'] = [w * DECAY for w in td['pass_wts']]
            td['pass_yards'].append(pass_yards)
            td['pass_wts'].append(1.0)

        if pd.notna(rush_yards):
            td['rush_wts'] = [w * DECAY for w in td['rush_wts']]
            td['rush_yards'].append(rush_yards)
            td['rush_wts'].append(1.0)

        if pd.notna(turnovers):
            td['to_wts'] = [w * DECAY for w in td['to_wts']]
            td['turnovers'].append(turnovers)
            td['to_wts'].append(1.0)

        if pd.notna(first_downs):
            td['fd_wts'] = [w * DECAY for w in td['fd_wts']]
            td['first_downs'].append(first_downs)
            td['fd_wts'].append(1.0)

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
                        'yards': np.mean(td['yards']) if td['yards'] else 400.0,
                        'pass_yards': np.mean(td['pass_yards']) if td['pass_yards'] else 230.0,
                        'rush_yards': np.mean(td['rush_yards']) if td['rush_yards'] else 170.0,
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 1.5,
                        'first_downs': np.mean(td['first_downs']) if td['first_downs'] else 20.0,
                        'ppd': np.mean(td['ppd']) if td['ppd'] else 2.2,
                        'ypd': np.mean(td['ypd']) if td['ypd'] else 32.0,
                        'scoring_pct': np.mean(td['scoring_pct']) if td['scoring_pct'] else 0.38,
                    }

        for tid in self.team_hca_data:
            if prev in self.team_hca_data[tid]:
                hd = self.team_hca_data[tid][prev]
                if hd['home_margins'] and hd['away_margins']:
                    raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
                    raw = max(-3, min(raw, 10))
                    self.prev_hca[tid] = 3.5 + 0.5 * (raw - 3.5)

        self.last_game.clear()

    def train_models(self):
        """Train Ridge models."""
        if len(self.spread_X) >= 100:  # CFB has more games
            X = np.array(self.spread_X)
            y = np.array(self.spread_y)
            self.spread_scaler.fit(X)
            X_s = self.spread_scaler.transform(X)
            self.spread_model = Ridge(alpha=1.0).fit(X_s, y)

        if len(self.total_X) >= 100:
            X = np.array(self.total_X)
            y = np.array(self.total_y)
            self.total_scaler.fit(X)
            X_s = self.total_scaler.transform(X)
            self.total_model = Ridge(alpha=1.0).fit(X_s, y)


def train_and_evaluate():
    """Train model and evaluate."""
    print("=" * 70)
    print("CFB ENHANCED MODEL TRAINING")
    print("=" * 70)

    conn = sqlite3.connect(str(DB_PATH))

    # Load games with stats
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
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
        drive_stats['ppd'] = drive_stats['scores'] * 7 / drive_stats['num_drives'].replace(0, 1)
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

    print(f"Total games: {len(games)}")
    print(f"Games with drive data: {games['home_ppd'].notna().sum() if 'home_ppd' in games.columns else 0}")

    model = CFBEnhancedModel()
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
            is_conf = g['conference_game'] == 1

            # Get prediction
            preds = model.predict(
                hid, aid, season, g['date'], g['week'],
                is_conf_game=is_conf, neutral_site=neutral
            )

            # Accumulate training data
            if preds['spread_feat'] is not None:
                model.spread_X.append(preds['spread_feat'])
                model.spread_y.append(actual_spread)

            if preds['total_feat'] is not None:
                model.total_X.append(preds['total_feat'])
                model.total_y.append(actual_total)

            # Retrain periodically
            if len(model.spread_X) % 100 == 0:
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
                    'conf_game': g['conference_game'],
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
                yards=g['home_yards'], pass_yards=g['home_pass_yards'],
                rush_yards=g['home_rush_yards'], turnovers=g['home_to'],
                first_downs=g['home_fd'],
                ppd=g.get('home_ppd'), ypd=g.get('home_ypd'),
                scoring_pct=g.get('home_scoring_pct')
            )
            model.update(
                aid, season, g['date'], g['away_score'], g['home_score'],
                is_home=False,
                yards=g['away_yards'], pass_yards=g['away_pass_yards'],
                rush_yards=g['away_rush_yards'], turnovers=g['away_to'],
                first_downs=g['away_fd'],
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

    # Conference games
    print("\nConference vs Non-Conference:")
    for conf in [0, 1]:
        mask = df['conf_game'] == conf
        if mask.sum() > 0:
            ats_mask = df.loc[mask, 'model_ats'].dropna()
            ats_pct = ats_mask.mean() * 100 if len(ats_mask) > 0 else 0
            label = "Conference" if conf == 1 else "Non-Conference"
            print(f"  {label}: ATS {ats_pct:.1f}% ({len(ats_mask)} games)")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / 'cfb_enhanced_model.pkl', 'wb') as f:
        pickle.dump({
            'spread_model': model.spread_model,
            'total_model': model.total_model,
            'spread_scaler': model.spread_scaler,
            'total_scaler': model.total_scaler,
            'prev_ratings': dict(model.prev_ratings),
            'prev_hca': dict(model.prev_hca),
        }, f)

    print(f"\nModel saved to {MODEL_DIR / 'cfb_enhanced_model.pkl'}")

    return model, df


if __name__ == '__main__':
    train_and_evaluate()
