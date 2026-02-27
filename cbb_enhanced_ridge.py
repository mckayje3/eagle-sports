"""
CBB Enhanced Ridge Model - Extended features for college basketball.

Adds to simple model:
- Recent form (last 5 games margin)
- Momentum (trend in margins)
- Dynamic per-team HCA
- Conference game indicator
- Season progress

Features (17 for spread, 15 for total)
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

DB_PATH = Path(__file__).parent / 'cbb_games.db'
MODEL_DIR = Path(__file__).parent / 'models'

# Constants
DECAY = 0.92  # Slightly more aggressive than simple
PREV_HALF_LIFE = 6.0
MIN_GAMES = 5  # Lower threshold for CBB (teams play ~30 games)


class CBBEnhancedModel:
    """Enhanced Ridge model for CBB with form and momentum."""

    def __init__(self):
        self.team_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'fg_pct': [], 'fg_wts': [],
            'rebounds': [], 'reb_wts': [],
            'turnovers': [], 'tov_wts': [],
            'margins': [], 'wins': [],
        }))

        # Dynamic HCA tracking
        self.team_hca_data = defaultdict(lambda: defaultdict(lambda: {
            'home_margins': [], 'away_margins': []
        }))

        self.prev_ratings = {}
        self.prev_hca = {}
        self.last_game = {}
        self.league_avg = {'ppg': 72.0, 'papg': 72.0}

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
            return 3
        try:
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 2

    def _get_stats(self, tid, season):
        td = self.team_stats[tid][season]
        n = len(td['ppg'])

        if n == 0:
            prev = self.prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', self.league_avg['ppg']),
                'papg': prev.get('papg', self.league_avg['papg']),
                'fg_pct': prev.get('fg_pct', 45.0),
                'rebounds': prev.get('rebounds', 35.0),
                'turnovers': prev.get('turnovers', 13.0),
                'games': 0,
                'margins': [],
                'wins': [],
            }

        ppg = self._wavg(td['ppg'], td['wts'])
        papg = self._wavg(td['papg'], td['wts'])
        fg = self._wavg(td['fg_pct'], td['fg_wts']) if td['fg_pct'] else None
        reb = self._wavg(td['rebounds'], td['reb_wts']) if td['rebounds'] else None
        tov = self._wavg(td['turnovers'], td['tov_wts']) if td['turnovers'] else None

        prev = self.prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', self.league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', self.league_avg['papg']) + (1 - blend) * papg,
            'fg_pct': fg if fg else prev.get('fg_pct', 45.0),
            'rebounds': reb if reb else prev.get('rebounds', 35.0),
            'turnovers': tov if tov else prev.get('turnovers', 13.0),
            'games': n,
            'margins': td['margins'],
            'wins': td['wins'],
        }

    def _get_dynamic_hca(self, home_id, season):
        """Get dynamic per-team HCA."""
        hd = self.team_hca_data[home_id][season]
        n_home = len(hd['home_margins'])
        n_away = len(hd['away_margins'])
        total = n_home + n_away

        # CBB default HCA ~3.5 points
        default_hca = 3.5

        if total == 0:
            return self.prev_hca.get(home_id, default_hca)

        if n_home > 0 and n_away > 0:
            raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
            raw = max(-2, min(raw, 10))  # CBB can have larger HCA
        else:
            raw = default_hca

        # Shrink toward mean
        shrunk = default_hca + 0.5 * (raw - default_hca)
        prev = self.prev_hca.get(home_id, default_hca)
        blend = 0.5 ** (total / 20.0)  # Faster blend for CBB

        return blend * prev + (1 - blend) * shrunk

    def _recent_form(self, margins, n=5):
        if len(margins) < n:
            return 0.0
        return float(np.mean(margins[-n:]))

    def _momentum(self, margins, n=6):
        if len(margins) < n:
            return 0.0
        recent = margins[-n:]
        return np.mean(recent[n//2:]) - np.mean(recent[:n//2])

    def _streak(self, wins):
        if not wins:
            return 0
        s, last = 0, wins[-1]
        for w in reversed(wins):
            if w == last:
                s += 1
            else:
                break
        return s if last == 1 else -s

    def extract_spread_features(self, hid, aid, season, date,
                                 neutral_site=False, conference_game=False,
                                 season_games_played=0):
        """Extract 17 spread features."""
        hs = self._get_stats(hid, season)
        aws = self._get_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        net = (hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg'])
        hca = 0.0 if neutral_site else self._get_dynamic_hca(hid, season)

        return np.array([
            hs['ppg'] - aws['ppg'],
            hs['papg'] - aws['papg'],
            net,
            self._recent_form(hs['margins']) - self._recent_form(aws['margins']),
            self._momentum(hs['margins']) - self._momentum(aws['margins']),
            self._streak(hs['wins']) - self._streak(aws['wins']),
            hr - ar,
            1.0 if hr == 0 else 0.0,
            1.0 if ar == 0 else 0.0,
            hca,
            min(hs['games'] / 20.0, 1.0),
            min(aws['games'] / 20.0, 1.0),
            1.0 if neutral_site else 0.0,
            1.0 if conference_game else 0.0,
            min(season_games_played / 5000, 1.0),  # CBB ~5000 D1 games
            hs['fg_pct'] - aws['fg_pct'],
            hs['turnovers'] - aws['turnovers'],
        ])

    def extract_total_features(self, hid, aid, season, date, season_games_played=0):
        """Extract 15 total features."""
        hs = self._get_stats(hid, season)
        aws = self._get_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        home_recent = abs(self._recent_form(hs['margins']))
        away_recent = abs(self._recent_form(aws['margins']))
        home_mom = abs(self._momentum(hs['margins']))
        away_mom = abs(self._momentum(aws['margins']))

        return np.array([
            hs['ppg'] + aws['ppg'],
            hs['papg'] + aws['papg'],
            (hs['ppg'] + hs['papg']) / 2,  # home pace proxy
            (aws['ppg'] + aws['papg']) / 2,  # away pace proxy
            1.0 if hr == 0 else 0.0,
            1.0 if ar == 0 else 0.0,
            min(hs['games'] / 20.0, 1.0),
            min(aws['games'] / 20.0, 1.0),
            hs['fg_pct'] + aws['fg_pct'],
            hs['rebounds'] + aws['rebounds'],
            hs['turnovers'] + aws['turnovers'],
            home_recent + away_recent,
            home_mom + away_mom,
            min(season_games_played / 5000, 1.0),
            0.0,  # placeholder for future injury adjustment
        ])

    def predict(self, hid, aid, season, date,
                neutral_site=False, conference_game=False, season_games_played=0):
        """Get predictions."""
        spread_feat = self.extract_spread_features(
            hid, aid, season, date, neutral_site, conference_game, season_games_played
        )
        total_feat = self.extract_total_features(hid, aid, season, date, season_games_played)

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

    def update(self, tid, season, date, pf, pa, is_home, fg=None, reb=None, tov=None):
        """Update team stats."""
        margin = pf - pa

        td = self.team_stats[tid][season]
        td['wts'] = [w * DECAY for w in td['wts']]
        td['ppg'].append(pf)
        td['papg'].append(pa)
        td['wts'].append(1.0)
        td['margins'].append(margin)
        td['wins'].append(1 if margin > 0 else 0)

        if pd.notna(fg):
            td['fg_wts'] = [w * DECAY for w in td['fg_wts']]
            td['fg_pct'].append(fg)
            td['fg_wts'].append(1.0)

        if pd.notna(reb):
            td['reb_wts'] = [w * DECAY for w in td['reb_wts']]
            td['rebounds'].append(reb)
            td['reb_wts'].append(1.0)

        if pd.notna(tov):
            td['tov_wts'] = [w * DECAY for w in td['tov_wts']]
            td['turnovers'].append(tov)
            td['tov_wts'].append(1.0)

        # HCA tracking
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
                        'fg_pct': np.mean(td['fg_pct']) if td['fg_pct'] else 45.0,
                        'rebounds': np.mean(td['rebounds']) if td['rebounds'] else 35.0,
                        'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 13.0,
                    }

        # HCA from previous season
        for tid in self.team_hca_data:
            if prev in self.team_hca_data[tid]:
                hd = self.team_hca_data[tid][prev]
                if hd['home_margins'] and hd['away_margins']:
                    raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
                    raw = max(-2, min(raw, 10))
                    self.prev_hca[tid] = 3.5 + 0.5 * (raw - 3.5)

        self.last_game.clear()

    def train_models(self):
        """Train Ridge models on accumulated data."""
        if len(self.spread_X) >= 100:
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

    @classmethod
    def load(cls, path: Path = None) -> 'CBBEnhancedModel':
        """Load a trained model from disk."""
        if path is None:
            path = MODEL_DIR / 'cbb_enhanced_model.pkl'

        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls()
        model.spread_model = data['spread_model']
        model.total_model = data['total_model']
        model.spread_scaler = data['spread_scaler']
        model.total_scaler = data['total_scaler']
        model.prev_ratings = data.get('prev_ratings', {})
        model.prev_hca = data.get('prev_hca', {})

        # Restore team stats (nested defaultdicts)
        model.team_stats = defaultdict(
            lambda: defaultdict(lambda: {
                'ppg': [], 'papg': [], 'wts': [],
                'fg_pct': [], 'fg_wts': [],
                'rebounds': [], 'reb_wts': [],
                'turnovers': [], 'tov_wts': [],
                'margins': [], 'wins': [],
            }),
            {k: defaultdict(lambda: {
                'ppg': [], 'papg': [], 'wts': [],
                'fg_pct': [], 'fg_wts': [],
                'rebounds': [], 'reb_wts': [],
                'turnovers': [], 'tov_wts': [],
                'margins': [], 'wins': [],
            }, v) for k, v in data.get('team_stats', {}).items()}
        )
        model.team_hca_data = defaultdict(
            lambda: defaultdict(lambda: {'home_margins': [], 'away_margins': []}),
            {k: defaultdict(lambda: {'home_margins': [], 'away_margins': []}, v)
             for k, v in data.get('team_hca_data', {}).items()}
        )
        model.last_game = data.get('last_game', {})
        return model


def train_and_evaluate():
    """Train model and evaluate on held-out season."""
    print("=" * 70)
    print("CBB ENHANCED MODEL TRAINING")
    print("=" * 70)

    conn = sqlite3.connect(str(DB_PATH))

    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site, g.conference_game,
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
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
    ''', conn)
    conn.close()

    print(f"Total games: {len(games)}")

    model = CBBEnhancedModel()
    results = []
    seasons = sorted(games['season'].unique())
    season_game_counts = {}

    for season in seasons:
        if season > seasons[0]:
            model.set_previous_season(season)

        season_game_counts[season] = 0
        season_games = games[games['season'] == season].copy()
        print(f"\nSeason {season}: {len(season_games)} games")

        for _, g in season_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_spread = g['away_score'] - g['home_score']
            actual_total = g['home_score'] + g['away_score']
            vegas_spread = g['vegas_spread']
            vegas_total = g['vegas_total']
            neutral = g['neutral_site'] == 1
            conf = g['conference_game'] == 1

            # Get prediction
            preds = model.predict(hid, aid, season, g['date'],
                                  neutral, conf, season_game_counts[season])

            # Accumulate training data
            if preds['spread_feat'] is not None:
                model.spread_X.append(preds['spread_feat'])
                model.spread_y.append(actual_spread)

            if preds['total_feat'] is not None:
                model.total_X.append(preds['total_feat'])
                model.total_y.append(actual_total)

            # Retrain periodically
            if len(model.spread_X) % 200 == 0:
                model.train_models()

            # Record result if we have prediction and Vegas line
            if preds['spread'] is not None and pd.notna(vegas_spread):
                spread_err = abs(preds['spread'] - actual_spread)
                vegas_err = abs(vegas_spread - actual_spread)

                edge = preds['spread'] - vegas_spread
                result = actual_spread - vegas_spread
                push = abs(result) < 0.5
                model_ats = bool(edge * result > 0) if not push else None

                results.append({
                    'season': season,
                    'spread_err': spread_err,
                    'vegas_err': vegas_err,
                    'model_ats': model_ats,
                    'total_err': abs(preds['total'] - actual_total) if preds['total'] else None,
                    'vegas_total_err': abs(vegas_total - actual_total) if pd.notna(vegas_total) else None,
                    'neutral': neutral,
                    'conference': conf,
                })

            # Update state
            model.update(hid, season, g['date'], g['home_score'], g['away_score'],
                        is_home=True, fg=g['home_fg'], reb=g['home_reb'], tov=g['home_tov'])
            model.update(aid, season, g['date'], g['away_score'], g['home_score'],
                        is_home=False, fg=g['away_fg'], reb=g['away_reb'], tov=g['away_tov'])

            season_game_counts[season] += 1

    # Evaluate
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    for season in df['season'].unique():
        mask = df['season'] == season
        spread_mae = df.loc[mask, 'spread_err'].mean()
        vegas_mae = df.loc[mask, 'vegas_err'].mean()
        ats_wins = df.loc[mask, 'model_ats'].dropna().mean() * 100

        print(f"\n{season}:")
        print(f"  Spread MAE: {spread_mae:.2f} (Vegas: {vegas_mae:.2f})")
        print(f"  ATS: {ats_wins:.1f}%")

    # Overall
    print(f"\nOverall:")
    print(f"  Spread MAE: {df['spread_err'].mean():.2f} (Vegas: {df['vegas_err'].mean():.2f})")
    print(f"  ATS: {df['model_ats'].dropna().mean() * 100:.1f}%")

    # By game type
    print("\nBy Game Type:")
    for label, mask in [('Neutral', df['neutral']), ('Home/Away', ~df['neutral'])]:
        if mask.sum() > 100:
            ats = df.loc[mask, 'model_ats'].dropna().mean() * 100
            mae = df.loc[mask, 'spread_err'].mean()
            print(f"  {label}: ATS {ats:.1f}%, MAE {mae:.2f}")

    for label, mask in [('Conference', df['conference']), ('Non-conference', ~df['conference'])]:
        if mask.sum() > 100:
            ats = df.loc[mask, 'model_ats'].dropna().mean() * 100
            mae = df.loc[mask, 'spread_err'].mean()
            print(f"  {label}: ATS {ats:.1f}%, MAE {mae:.2f}")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / 'cbb_enhanced_model.pkl', 'wb') as f:
        pickle.dump({
            'spread_model': model.spread_model,
            'total_model': model.total_model,
            'spread_scaler': model.spread_scaler,
            'total_scaler': model.total_scaler,
            'prev_ratings': dict(model.prev_ratings),
            'prev_hca': dict(model.prev_hca),
            # Include team stats for predictions after loading
            'team_stats': {k: dict(v) for k, v in model.team_stats.items()},
            'team_hca_data': {k: dict(v) for k, v in model.team_hca_data.items()},
            'last_game': dict(model.last_game),
        }, f)

    print(f"\nModel saved to {MODEL_DIR / 'cbb_enhanced_model.pkl'}")

    return model, df


if __name__ == '__main__':
    train_and_evaluate()
