"""
NBA Optimal Blend Analysis

Find the optimal blend weights between our statistical model and Vegas.
Simple linear approach - no neural networks needed.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / 'nba_games.db'


class StatModel:
    def __init__(self):
        self.team_games = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'wts': []
        }))
        self.prev_ratings = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}
        self.last_game = {}
        self.decay = 0.93
        self.prev_hl = 6.0

    def _wavg(self, v, w):
        return float(np.average(v, weights=w)) if v else None

    def predict(self, hid, aid, season, gdate):
        def get_rating(tid):
            td = self.team_games[tid][season]
            gp = len(td['ppg'])
            ppg = self._wavg(td['ppg'], td['wts'])
            papg = self._wavg(td['papg'], td['wts'])
            prev_off = self.prev_ratings.get(tid, {}).get('off', self.league_avg['ppg'])
            prev_def = self.prev_ratings.get(tid, {}).get('def', self.league_avg['papg'])
            if ppg is None:
                return prev_off, prev_def
            blend = 0.5 ** (gp / self.prev_hl)
            return blend * prev_off + (1 - blend) * ppg, blend * prev_def + (1 - blend) * papg

        ho, hd = get_rating(hid)
        ao, ad = get_rating(aid)

        # Rest
        hr, ar = 3, 3
        if hid in self.last_game:
            curr = datetime.strptime(gdate[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[hid][:10], '%Y-%m-%d')
            hr = max(0, min((curr - last).days - 1, 5))
        if aid in self.last_game:
            curr = datetime.strptime(gdate[:10], '%Y-%m-%d')
            last = datetime.strptime(self.last_game[aid][:10], '%Y-%m-%d')
            ar = max(0, min((curr - last).days - 1, 5))

        hca = 2.0
        adj = 0.0
        if hr == 0:
            adj -= 1.0
        if ar == 0:
            adj += 1.0

        stat_spread = (ao + hd) / 2 - hca / 2 - adj / 2 - ((ho + ad) / 2 + hca / 2 + adj / 2)
        stat_total = (ho + ad) / 2 + (ao + hd) / 2

        return stat_spread, stat_total

    def update(self, hid, aid, hs, aws, season, gdate):
        for tid in [hid, aid]:
            self.team_games[tid][season]['wts'] = [
                w * self.decay for w in self.team_games[tid][season]['wts']
            ]
        self.team_games[hid][season]['ppg'].append(hs)
        self.team_games[hid][season]['papg'].append(aws)
        self.team_games[hid][season]['wts'].append(1.0)
        self.team_games[aid][season]['ppg'].append(aws)
        self.team_games[aid][season]['papg'].append(hs)
        self.team_games[aid][season]['wts'].append(1.0)
        self.last_game[hid] = gdate
        self.last_game[aid] = gdate

    def set_prev(self, season):
        ps = season - 1
        for tid in self.team_games:
            if ps in self.team_games[tid] and self.team_games[tid][ps]['ppg']:
                self.prev_ratings[tid] = {
                    'off': np.mean(self.team_games[tid][ps]['ppg']),
                    'def': np.mean(self.team_games[tid][ps]['papg'])
                }
        self.last_game.clear()

    def set_avg(self, ppg, papg):
        self.league_avg = {'ppg': ppg, 'papg': papg}


def main():
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.game_date_eastern, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.home_score > 0 AND o.latest_spread IS NOT NULL
        ORDER BY g.game_date_eastern
    ''', conn)
    conn.close()

    print('=' * 60)
    print('OPTIMAL BLEND WEIGHT ANALYSIS')
    print('=' * 60)
    print(f'Games with Vegas: {len(games)}')

    # Generate predictions
    model = StatModel()
    results = []

    for season in sorted(games.season.unique()):
        if season > games.season.min():
            model.set_prev(season)
            pg = games[games.season == season - 1]
            if len(pg) > 0:
                model.set_avg(pg.home_score.mean(), pg.away_score.mean())

        for _, g in games[games.season == season].iterrows():
            stat_spread, stat_total = model.predict(
                g.home_team_id, g.away_team_id, season, g.game_date_eastern
            )
            results.append({
                'season': season,
                'stat_spread': stat_spread,
                'stat_total': stat_total,
                'vegas_spread': g.vegas_spread,
                'vegas_total': g.vegas_total if pd.notna(g.vegas_total) else stat_total,
                'actual_spread': g.away_score - g.home_score,
                'actual_total': g.home_score + g.away_score
            })
            model.update(
                g.home_team_id, g.away_team_id,
                g.home_score, g.away_score, season, g.game_date_eastern
            )

    df = pd.DataFrame(results)

    # Split: train on all but last season
    seasons = sorted(df.season.unique())
    train = df[df.season != seasons[-1]]
    test = df[df.season == seasons[-1]]

    print(f'Train: {len(train)}, Test: {len(test)} (season {seasons[-1]})')

    # Try different blend weights
    print(f"\n{'Blend (Stat)':<15} {'Winner Acc':<12} {'Spread MAE'}")
    print('-' * 40)

    best_acc = 0
    best_weight = 0.5

    for w in np.arange(0, 1.05, 0.1):
        blend = w * test['stat_spread'] + (1 - w) * test['vegas_spread']
        acc = ((blend < 0) == (test['actual_spread'] < 0)).mean()
        mae = np.abs(blend - test['actual_spread']).mean()
        marker = ' <--' if acc > best_acc else ''
        if acc > best_acc:
            best_acc = acc
            best_weight = w
        print(f'{w:.1f}             {acc*100:.1f}%        {mae:.2f}{marker}')

    print(f'\nBest weight for statistical model: {best_weight:.1f}')

    # Linear regression
    print('\n=== LINEAR REGRESSION ===')
    X_train = train[['stat_spread', 'vegas_spread']].values
    y_train = train['actual_spread'].values
    X_test = test[['stat_spread', 'vegas_spread']].values
    y_test = test['actual_spread'].values

    lr = Ridge(alpha=1.0)
    lr.fit(X_train, y_train)
    print(f'Spread coefficients: stat={lr.coef_[0]:.3f}, vegas={lr.coef_[1]:.3f}')
    print(f'Intercept: {lr.intercept_:.3f}')

    lr_pred = lr.predict(X_test)
    lr_acc = ((lr_pred < 0) == (y_test < 0)).mean()
    lr_mae = np.abs(lr_pred - y_test).mean()

    # Total regression
    X_train_t = train[['stat_total', 'vegas_total']].values
    y_train_t = train['actual_total'].values
    X_test_t = test[['stat_total', 'vegas_total']].values
    y_test_t = test['actual_total'].values

    lr_t = Ridge(alpha=1.0)
    lr_t.fit(X_train_t, y_train_t)
    print(f'Total coefficients: stat={lr_t.coef_[0]:.3f}, vegas={lr_t.coef_[1]:.3f}')

    lr_t_pred = lr_t.predict(X_test_t)
    lr_t_mae = np.abs(lr_t_pred - y_test_t).mean()

    # Final comparison
    print('\n' + '=' * 60)
    print('FINAL COMPARISON (Test Set)')
    print('=' * 60)
    print(f"{'Model':<25} {'Winner Acc':<12} {'Spread MAE':<12} {'Total MAE'}")
    print('-' * 60)

    stat_acc = ((test['stat_spread'] < 0) == (test['actual_spread'] < 0)).mean()
    vegas_acc = ((test['vegas_spread'] < 0) == (test['actual_spread'] < 0)).mean()
    stat_mae = np.abs(test['stat_spread'] - test['actual_spread']).mean()
    vegas_mae = np.abs(test['vegas_spread'] - test['actual_spread']).mean()
    vegas_t_mae = np.abs(test['vegas_total'] - test['actual_total']).mean()
    stat_t_mae = np.abs(test['stat_total'] - test['actual_total']).mean()

    blend_50 = (test['stat_spread'] + test['vegas_spread']) / 2
    blend_50_acc = ((blend_50 < 0) == (test['actual_spread'] < 0)).mean()
    blend_50_mae = np.abs(blend_50 - test['actual_spread']).mean()

    best_blend = best_weight * test['stat_spread'] + (1 - best_weight) * test['vegas_spread']
    best_blend_mae = np.abs(best_blend - test['actual_spread']).mean()

    print(f"{'Statistical Model':<25} {stat_acc*100:.1f}%        {stat_mae:.2f}         {stat_t_mae:.2f}")
    print(f"{'Vegas':<25} {vegas_acc*100:.1f}%        {vegas_mae:.2f}         {vegas_t_mae:.2f}")
    print(f"{'50/50 Blend':<25} {blend_50_acc*100:.1f}%        {blend_50_mae:.2f}")
    print(f"{'Optimal Blend':<25} {best_acc*100:.1f}%        {best_blend_mae:.2f}")
    print(f"{'Linear Regression':<25} {lr_acc*100:.1f}%        {lr_mae:.2f}         {lr_t_mae:.2f}")


if __name__ == '__main__':
    main()
