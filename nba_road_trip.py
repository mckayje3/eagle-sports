"""
NBA Road Trip Effect Analysis

Test if consecutive away games affect team performance.
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / 'nba_games.db'


def analyze_road_trips():
    conn = sqlite3.connect(str(DB_PATH))

    games = pd.read_sql_query('''
        SELECT game_id, season, game_date_eastern, home_team_id, away_team_id,
               home_score, away_score, away_score - home_score as spread
        FROM games WHERE home_score > 0
        ORDER BY game_date_eastern, game_id
    ''', conn)
    conn.close()

    print('=' * 60)
    print('ROAD TRIP EFFECT ANALYSIS')
    print('=' * 60)

    # Track consecutive away games for each team
    team_away_streak = {}
    road_trip_data = []

    for _, g in games.iterrows():
        hid, aid = g['home_team_id'], g['away_team_id']

        # Home team: get their prior away streak, then reset
        home_prior_streak = team_away_streak.get(hid, 0)
        team_away_streak[hid] = 0

        # Away team: increment streak
        away_streak = team_away_streak.get(aid, 0) + 1
        team_away_streak[aid] = away_streak

        road_trip_data.append({
            'game_id': g['game_id'],
            'spread': g['spread'],
            'home_prior_streak': home_prior_streak,
            'away_road_game_num': away_streak
        })

    df = pd.DataFrame(road_trip_data)

    print('\n=== AWAY TEAM PERFORMANCE BY ROAD GAME NUMBER ===')
    print(f"{'Road Game #':<12} {'Avg Spread':<12} {'Away Win%':<12} {'Games'}")
    print('-' * 48)

    for num in range(1, 7):
        subset = df[df['away_road_game_num'] == num]
        if len(subset) > 50:
            avg_spread = subset['spread'].mean()  # Positive = away wins
            away_win_pct = (subset['spread'] > 0).mean()
            print(f'{num:<12} {avg_spread:<12.1f} {away_win_pct*100:<11.1f}% {len(subset)}')

    print('\n=== HOME TEAM - RETURNING FROM ROAD TRIP ===')
    print(f"{'Prior Away':<12} {'Avg Margin':<12} {'Home Win%':<12} {'Games'}")
    print('-' * 48)

    for streak in range(6):
        subset = df[df['home_prior_streak'] == streak]
        if len(subset) > 50:
            avg_margin = -subset['spread'].mean()  # Positive = home wins
            home_win_pct = (subset['spread'] < 0).mean()
            print(f'{streak:<12} {avg_margin:<12.1f} {home_win_pct*100:<11.1f}% {len(subset)}')

    # Correlation
    print('\n=== CORRELATION ===')
    corr = df['away_road_game_num'].corr(df['spread'])
    print(f'Away road game # vs spread: {corr:.3f}')

    return games


class RoadTripModel:
    def __init__(self, decay=0.93, prev_hl=6.0, hca=2.0, b2b=1.0, road_penalty=0.0):
        self.decay = decay
        self.prev_hl = prev_hl
        self.hca = hca
        self.b2b = b2b
        self.road_penalty = road_penalty
        self.team_games = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'weights': []
        }))
        self.prev_ratings = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}
        self.last_game = {}
        self.away_streak = {}

    def _wavg(self, vals, wts):
        return float(np.average(vals, weights=wts)) if vals else None

    def _get_rating(self, tid, season):
        td = self.team_games[tid][season]
        gp = len(td['ppg'])
        ppg = self._wavg(td['ppg'], td['weights'])
        papg = self._wavg(td['papg'], td['weights'])
        prev_off = self.prev_ratings.get(tid, {}).get('off', self.league_avg['ppg'])
        prev_def = self.prev_ratings.get(tid, {}).get('def', self.league_avg['papg'])
        if ppg is None:
            return prev_off, prev_def
        blend = 0.5 ** (gp / self.prev_hl)
        return blend * prev_off + (1 - blend) * ppg, blend * prev_def + (1 - blend) * papg

    def _rest(self, tid, gdate):
        if tid not in self.last_game:
            return 3
        curr = datetime.strptime(gdate[:10], '%Y-%m-%d')
        last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
        return max(0, min((curr - last).days - 1, 5))

    def predict(self, hid, aid, season, gdate):
        ho, hd = self._get_rating(hid, season)
        ao, ad = self._get_rating(aid, season)
        ph = (ho + ad) / 2 + self.hca / 2
        pa = (ao + hd) / 2 - self.hca / 2

        hr, ar = self._rest(hid, gdate), self._rest(aid, gdate)
        adj = 0.0
        if hr == 0:
            adj -= self.b2b
        if ar == 0:
            adj += self.b2b

        # Road trip penalty (beyond game 1)
        away_streak = self.away_streak.get(aid, 0) + 1
        if away_streak > 1:
            adj += (away_streak - 1) * self.road_penalty

        return ph + adj / 2, pa - adj / 2

    def update(self, hid, aid, hs, aws, season, gdate):
        for tid in [hid, aid]:
            self.team_games[tid][season]['weights'] = [
                w * self.decay for w in self.team_games[tid][season]['weights']
            ]

        self.team_games[hid][season]['ppg'].append(hs)
        self.team_games[hid][season]['papg'].append(aws)
        self.team_games[hid][season]['weights'].append(1.0)

        self.team_games[aid][season]['ppg'].append(aws)
        self.team_games[aid][season]['papg'].append(hs)
        self.team_games[aid][season]['weights'].append(1.0)

        self.last_game[hid] = gdate
        self.last_game[aid] = gdate

        self.away_streak[hid] = 0
        self.away_streak[aid] = self.away_streak.get(aid, 0) + 1

    def set_prev(self, season):
        ps = season - 1
        for tid in self.team_games:
            if ps in self.team_games[tid] and self.team_games[tid][ps]['ppg']:
                self.prev_ratings[tid] = {
                    'off': np.mean(self.team_games[tid][ps]['ppg']),
                    'def': np.mean(self.team_games[tid][ps]['papg'])
                }
        self.last_game.clear()
        self.away_streak.clear()

    def set_avg(self, ppg, papg):
        self.league_avg = {'ppg': ppg, 'papg': papg}


def run_test(games, road_penalty):
    model = RoadTripModel(road_penalty=road_penalty)
    preds = []

    for season in sorted(games.season.unique()):
        if season > games.season.min():
            model.set_prev(season)
            pg = games[games.season == season - 1]
            if len(pg) > 0:
                model.set_avg(pg['home_score'].mean(), pg['away_score'].mean())

        for _, g in games[games.season == season].iterrows():
            ph, pa = model.predict(
                g.home_team_id, g.away_team_id, season, g.game_date_eastern
            )
            preds.append({
                'pred_spread': pa - ph,
                'actual_spread': g.spread
            })
            model.update(
                g.home_team_id, g.away_team_id,
                g.home_score, g.away_score, season, g.game_date_eastern
            )

    df = pd.DataFrame(preds)
    df['pred_hw'] = df['pred_spread'] < 0
    df['actual_hw'] = df['actual_spread'] < 0
    return (df['pred_hw'] == df['actual_hw']).mean()


def main():
    games = analyze_road_trips()

    print('\n' + '=' * 60)
    print('TESTING ROAD TRIP ADJUSTMENT')
    print('=' * 60)

    baseline = run_test(games, 0.0)
    print(f'\nBaseline (no road trip adj): {baseline*100:.2f}%')

    print(f"\n{'Road Penalty':<14} {'Accuracy':<12} {'vs Baseline'}")
    print('-' * 40)

    results = []
    for penalty in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]:
        acc = run_test(games, penalty)
        diff = acc - baseline
        marker = ' <--' if acc > baseline else ''
        print(f'{penalty:<14.2f} {acc*100:<11.2f}% ({diff*100:+.2f}%){marker}')
        results.append((penalty, acc))

    best_penalty, best_acc = max(results, key=lambda x: x[1])
    print(f'\nBest road_penalty: {best_penalty}')
    print(f'Best accuracy: {best_acc*100:.2f}%')
    print(f'Improvement: {(best_acc - baseline)*100:+.2f}%')


if __name__ == '__main__':
    main()
