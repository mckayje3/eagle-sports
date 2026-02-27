"""
CBB Simple Model - Clean Ridge regression baseline for college basketball.

Similar to NBA simple model but adapted for CBB:
- More teams (~360 D1 vs 30 NBA)
- Fewer games per team per season
- Neutral site games common
- Conference play matters

Features (12):
- PPG diff, PAPG diff, net rating
- FG%, rebounds, turnovers
- Rest/B2B indicators
- Home/away reliability
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
DECAY = 0.95  # Slightly lower than NBA due to shorter seasons
PREV_HALF_LIFE = 6.0  # Games until 50% current season weight
MIN_GAMES = 5  # Lower threshold for CBB (teams play ~30 games)


class CBBSimpleModel:
    """Simple Ridge model for CBB spread and total predictions."""

    def __init__(self):
        self.team_stats = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': [],
            'fg_pct': [], 'fg_wts': [],
            'rebounds': [], 'reb_wts': [],
            'turnovers': [], 'tov_wts': [],
        }))

        self.prev_ratings = {}
        self.last_game = {}
        self.league_avg = {'ppg': 72.0, 'papg': 72.0}  # CBB average ~72 ppg

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
        }

    def extract_spread_features(self, hid, aid, season, date, neutral_site=False):
        """Extract 12 spread features."""
        hs = self._get_stats(hid, season)
        aws = self._get_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)
        net = (hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg'])

        # Reduce HCA for neutral site games
        neutral_adj = 1.0 if neutral_site else 0.0

        return np.array([
            hs['ppg'] - aws['ppg'],
            hs['papg'] - aws['papg'],
            hs['fg_pct'] - aws['fg_pct'],
            hs['rebounds'] - aws['rebounds'],
            hs['turnovers'] - aws['turnovers'],
            hr - ar,
            1.0 if hr == 0 else 0.0,
            1.0 if ar == 0 else 0.0,
            net,
            net,  # duplicate for compatibility
            min(hs['games'] / 15.0, 1.0),  # CBB: max 30+ games
            min(aws['games'] / 15.0, 1.0),
        ])

    def extract_total_features(self, hid, aid, season, date):
        """Extract 6 total features."""
        hs = self._get_stats(hid, season)
        aws = self._get_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = self._get_rest(hid, date)
        ar = self._get_rest(aid, date)

        return np.array([
            hs['ppg'] + aws['ppg'],
            hs['papg'] + aws['papg'],
            1.0 if hr == 0 else 0.0,
            1.0 if ar == 0 else 0.0,
            hs['rebounds'] + aws['rebounds'],
            hs['turnovers'] + aws['turnovers'],
        ])

    def predict(self, hid, aid, season, date, neutral_site=False):
        """Get predictions."""
        spread_feat = self.extract_spread_features(hid, aid, season, date, neutral_site)
        total_feat = self.extract_total_features(hid, aid, season, date)

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

    def update(self, tid, season, date, pf, pa, fg=None, reb=None, tov=None):
        """Update team stats."""
        td = self.team_stats[tid][season]
        td['wts'] = [w * DECAY for w in td['wts']]
        td['ppg'].append(pf)
        td['papg'].append(pa)
        td['wts'].append(1.0)

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


def train_and_evaluate():
    """Train model and evaluate on held-out season."""
    print("=" * 70)
    print("CBB SIMPLE MODEL TRAINING")
    print("=" * 70)

    conn = sqlite3.connect(str(DB_PATH))

    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site,
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

    model = CBBSimpleModel()
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

            # Get prediction
            preds = model.predict(hid, aid, season, g['date'], neutral)

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
                })

            # Update state
            model.update(hid, season, g['date'], g['home_score'], g['away_score'],
                        fg=g['home_fg'], reb=g['home_reb'], tov=g['home_tov'])
            model.update(aid, season, g['date'], g['away_score'], g['home_score'],
                        fg=g['away_fg'], reb=g['away_reb'], tov=g['away_tov'])

    # Evaluate
    df = pd.DataFrame(results)
    print(f"\nTotal results: {len(df)}")
    print(f"model_ats value counts: {df['model_ats'].value_counts(dropna=False).to_dict()}")

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

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / 'cbb_simple_model.pkl', 'wb') as f:
        pickle.dump({
            'spread_model': model.spread_model,
            'total_model': model.total_model,
            'spread_scaler': model.spread_scaler,
            'total_scaler': model.total_scaler,
            'prev_ratings': dict(model.prev_ratings),
        }, f)

    print(f"\nModel saved to {MODEL_DIR / 'cbb_simple_model.pkl'}")

    return model, df


if __name__ == '__main__':
    train_and_evaluate()
