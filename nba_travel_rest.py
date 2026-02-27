"""Test travel distance and extended rest factors."""
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

DB_PATH = Path(__file__).parent / 'nba_games.db'


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3959
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))


class TravelRestModel:
    def __init__(self, decay=0.93, prev_hl=6.0, hca=2.0,
                 b2b_penalty=1.0, rest_bonus=0.0, travel_factor=0.0):
        self.decay = decay
        self.prev_hl = prev_hl
        self.hca = hca
        self.b2b_penalty = b2b_penalty
        self.rest_bonus = rest_bonus
        self.travel_factor = travel_factor

        self.team_games = defaultdict(lambda: defaultdict(lambda: {'ppg': [], 'papg': [], 'weights': []}))
        self.prev_ratings = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}
        self.last_game = {}
        self.last_loc = {}
        self.team_loc = {}

    def set_team_locations(self, locs):
        self.team_loc = locs

    def _wavg(self, vals, wts):
        return float(np.average(vals, weights=wts)) if vals else None

    def _get_rating(self, tid, season, gp):
        td = self.team_games[tid][season]
        ppg = self._wavg(td['ppg'], td['weights'])
        papg = self._wavg(td['papg'], td['weights'])
        prev_off = self.prev_ratings.get(tid, {}).get('off', self.league_avg['ppg'])
        prev_def = self.prev_ratings.get(tid, {}).get('def', self.league_avg['papg'])
        if ppg is None:
            return prev_off, prev_def
        blend = 0.5 ** (gp / self.prev_hl)
        return blend * prev_off + (1-blend) * ppg, blend * prev_def + (1-blend) * papg

    def _rest(self, tid, gdate):
        if tid not in self.last_game:
            return 3
        curr = datetime.strptime(gdate[:10], '%Y-%m-%d')
        last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
        return max(0, min((curr - last).days - 1, 5))

    def _travel(self, tid, dest_loc):
        if tid not in self.last_loc or not dest_loc or not self.last_loc[tid]:
            return 0
        prev = self.last_loc[tid]
        return haversine_miles(prev[0], prev[1], dest_loc[0], dest_loc[1])

    def predict(self, hid, aid, season, gdate):
        ho, hd = self._get_rating(hid, season, len(self.team_games[hid][season]['ppg']))
        ao, ad = self._get_rating(aid, season, len(self.team_games[aid][season]['ppg']))

        ph = (ho + ad)/2 + self.hca/2
        pa = (ao + hd)/2 - self.hca/2

        hr, ar = self._rest(hid, gdate), self._rest(aid, gdate)

        adj = 0.0
        # B2B penalty
        if hr == 0:
            adj -= self.b2b_penalty
        if ar == 0:
            adj += self.b2b_penalty

        # Extended rest bonus (beyond 1 day)
        if hr > 1:
            adj += (hr - 1) * self.rest_bonus
        if ar > 1:
            adj -= (ar - 1) * self.rest_bonus

        # Travel penalty for away team
        if self.travel_factor > 0:
            home_loc = self.team_loc.get(hid)
            away_travel = self._travel(aid, home_loc) / 1000
            adj += away_travel * self.travel_factor

        return ph + adj/2, pa - adj/2

    def update(self, hid, aid, hs, aws, season, gdate):
        for tid in [hid, aid]:
            self.team_games[tid][season]['weights'] = [w*self.decay for w in self.team_games[tid][season]['weights']]
        self.team_games[hid][season]['ppg'].append(hs)
        self.team_games[hid][season]['papg'].append(aws)
        self.team_games[hid][season]['weights'].append(1.0)
        self.team_games[aid][season]['ppg'].append(aws)
        self.team_games[aid][season]['papg'].append(hs)
        self.team_games[aid][season]['weights'].append(1.0)

        self.last_game[hid] = gdate
        self.last_game[aid] = gdate

        home_loc = self.team_loc.get(hid)
        if home_loc:
            self.last_loc[hid] = home_loc
            self.last_loc[aid] = home_loc

    def set_prev(self, season):
        ps = season - 1
        for tid in self.team_games:
            if ps in self.team_games[tid] and self.team_games[tid][ps]['ppg']:
                self.prev_ratings[tid] = {
                    'off': np.mean(self.team_games[tid][ps]['ppg']),
                    'def': np.mean(self.team_games[tid][ps]['papg'])
                }
        self.last_game.clear()
        self.last_loc.clear()

    def set_avg(self, ppg, papg):
        self.league_avg = {'ppg': ppg, 'papg': papg}


def run_test(games, team_loc, b2b=1.0, rest_bonus=0.0, travel_factor=0.0):
    model = TravelRestModel(b2b_penalty=b2b, rest_bonus=rest_bonus, travel_factor=travel_factor)
    model.set_team_locations(team_loc)
    preds = []
    for season in sorted(games.season.unique()):
        if season > games.season.min():
            model.set_prev(season)
            pg = games[games.season == season - 1]
            if len(pg) > 0:
                model.set_avg(pg['home_score'].mean(), pg['away_score'].mean())
        for _, g in games[games.season == season].iterrows():
            ph, pa = model.predict(g.home_team_id, g.away_team_id, season, g.game_date_eastern)
            preds.append({'pred_spread': pa - ph, 'actual_spread': g.spread})
            model.update(g.home_team_id, g.away_team_id, g.home_score, g.away_score, season, g.game_date_eastern)
    df = pd.DataFrame(preds)
    df['pred_hw'] = df['pred_spread'] < 0
    df['actual_hw'] = df['actual_spread'] < 0
    return (df['pred_hw'] == df['actual_hw']).mean()


def main():
    conn = sqlite3.connect(str(DB_PATH))
    teams = pd.read_sql_query('SELECT team_id, latitude, longitude FROM teams', conn)
    games = pd.read_sql_query('''
        SELECT game_id, season, game_date_eastern, home_team_id, away_team_id,
               home_score, away_score, away_score - home_score as spread
        FROM games WHERE home_score > 0 ORDER BY game_date_eastern, game_id
    ''', conn)
    conn.close()

    team_loc = {row['team_id']: (row['latitude'], row['longitude'])
                for _, row in teams.iterrows() if pd.notna(row['latitude'])}

    print('=' * 60)
    print('TRAVEL & EXTENDED REST MODEL TEST')
    print('=' * 60)

    # Baseline (B2B only)
    baseline = run_test(games, team_loc, b2b=1.0, rest_bonus=0.0, travel_factor=0.0)
    print(f'\nBaseline (B2B only): {baseline*100:.2f}%')

    # Test rest bonus
    print(f"\n{'Rest Bonus':<15} {'Winner Acc':<12} {'vs Baseline'}")
    print('-' * 45)

    for bonus in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        acc = run_test(games, team_loc, b2b=1.0, rest_bonus=bonus, travel_factor=0.0)
        diff = acc - baseline
        marker = ' <--' if acc > baseline else ''
        print(f'{bonus:<15.2f} {acc*100:<11.2f}% ({diff*100:+.2f}%){marker}')

    # Test travel factor
    print(f"\n{'Travel Factor':<15} {'Winner Acc':<12} {'vs Baseline'}")
    print('-' * 45)

    for factor in [0.0, 0.1, 0.25, 0.5, 1.0]:
        acc = run_test(games, team_loc, b2b=1.0, rest_bonus=0.0, travel_factor=factor)
        diff = acc - baseline
        marker = ' <--' if acc > baseline else ''
        print(f'{factor:<15.2f} {acc*100:<11.2f}% ({diff*100:+.2f}%){marker}')

    # Grid search best combination
    print('\nGrid Search (Rest Bonus x Travel Factor):')
    print('-' * 50)

    best_acc = baseline
    best_config = (0, 0)
    results = []

    for rest in [0.0, 0.25, 0.5, 0.75]:
        for travel in [0.0, 0.1, 0.25]:
            acc = run_test(games, team_loc, b2b=1.0, rest_bonus=rest, travel_factor=travel)
            results.append((rest, travel, acc))
            if acc > best_acc:
                best_acc = acc
                best_config = (rest, travel)

    # Show top 5
    results.sort(key=lambda x: x[2], reverse=True)
    print(f"{'Rest':<8} {'Travel':<10} {'Accuracy':<12} {'vs Baseline'}")
    for rest, travel, acc in results[:5]:
        diff = acc - baseline
        print(f'{rest:<8.2f} {travel:<10.2f} {acc*100:<11.2f}% ({diff*100:+.2f}%)')

    print(f'\nBest: rest_bonus={best_config[0]}, travel_factor={best_config[1]}')
    print(f'Best accuracy: {best_acc*100:.2f}%')
    print(f'Baseline: {baseline*100:.2f}%')
    print(f'Improvement: {(best_acc - baseline)*100:+.2f}%')


if __name__ == '__main__':
    main()
