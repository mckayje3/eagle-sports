"""
Grid search for optimal HCA and decay parameters.

Tests combinations of:
- PPG decay (0.93, 0.95, 0.97, 0.99)
- HCA half-life (4, 6, 8, 10, 15, 20)
- HCA shrinkage (0.3, 0.4, 0.5, 0.6, 0.7)
- Base HCA (1.5, 1.8, 2.0, 2.2)
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / 'nba_games.db'


class TunableModel:
    def __init__(
        self,
        decay: float = 0.97,
        prev_half_life: float = 6.0,
        hca_half_life: float = 6.0,
        hca_shrink: float = 0.5,
        base_hca: float = 2.0,
        b2b: float = 1.0
    ):
        self.decay = decay
        self.prev_half_life = prev_half_life
        self.hca_half_life = hca_half_life
        self.hca_shrink = hca_shrink
        self.base_hca = base_hca
        self.b2b = b2b

        self.team_games = defaultdict(lambda: defaultdict(lambda: {'ppg': [], 'papg': [], 'wts': []}))
        self.team_hca = defaultdict(lambda: defaultdict(lambda: {'home': [], 'away': []}))
        self.prev_ratings: dict = {}
        self.prev_hca: dict = {}
        self.last_game: dict = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}

    def _wavg(self, vals: list, wts: list) -> float | None:
        if not vals:
            return None
        return float(np.average(vals, weights=wts))

    def _get_rating(self, tid: int, season: int) -> tuple[float, float]:
        td = self.team_games[tid][season]
        games_played = len(td['ppg'])

        curr_ppg = self._wavg(td['ppg'], td['wts'])
        curr_papg = self._wavg(td['papg'], td['wts'])

        prev_ppg = self.prev_ratings.get(tid, {}).get('ppg', self.league_avg['ppg'])
        prev_papg = self.prev_ratings.get(tid, {}).get('papg', self.league_avg['papg'])

        if curr_ppg is None:
            return prev_ppg, prev_papg

        blend = 0.5 ** (games_played / self.prev_half_life)
        return blend * prev_ppg + (1-blend) * curr_ppg, blend * prev_papg + (1-blend) * curr_papg

    def _get_hca(self, home_id: int, season: int) -> float:
        hd = self.team_hca[home_id][season]
        n_home = len(hd['home'])
        n_away = len(hd['away'])
        total = n_home + n_away

        if total == 0:
            return self.prev_hca.get(home_id, self.base_hca)

        if n_home > 0 and n_away > 0:
            avg_home = np.mean(hd['home'])
            avg_away = np.mean(hd['away'])
            raw_hca = (avg_home - avg_away) / 2

            # Apply shrinkage toward base
            curr_hca = self.base_hca + self.hca_shrink * (raw_hca - self.base_hca)
            curr_hca = max(-1, min(curr_hca, 6))  # Clamp
        else:
            curr_hca = self.base_hca

        prev = self.prev_hca.get(home_id, self.base_hca)
        blend = 0.5 ** (total / self.hca_half_life)
        return blend * prev + (1 - blend) * curr_hca

    def _get_rest(self, tid: int, gdate: str) -> int:
        if tid not in self.last_game:
            return 3
        curr = datetime.strptime(gdate[:10], '%Y-%m-%d')
        last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
        return max(0, min((curr - last).days - 1, 5))

    def predict(self, home_id: int, away_id: int, season: int, gdate: str) -> tuple[float, float]:
        h_off, h_def = self._get_rating(home_id, season)
        a_off, a_def = self._get_rating(away_id, season)
        hca = self._get_hca(home_id, season)

        pred_h = (h_off + a_def) / 2 + hca / 2
        pred_a = (a_off + h_def) / 2 - hca / 2

        h_rest = self._get_rest(home_id, gdate)
        a_rest = self._get_rest(away_id, gdate)
        adj = 0
        if h_rest == 0:
            adj -= self.b2b
        if a_rest == 0:
            adj += self.b2b

        return pred_h + adj/2, pred_a - adj/2

    def update(self, home_id: int, away_id: int, h_score: int, a_score: int,
               season: int, gdate: str):
        for tid in [home_id, away_id]:
            self.team_games[tid][season]['wts'] = [
                w * self.decay for w in self.team_games[tid][season]['wts']
            ]

        self.team_games[home_id][season]['ppg'].append(h_score)
        self.team_games[home_id][season]['papg'].append(a_score)
        self.team_games[home_id][season]['wts'].append(1.0)

        self.team_games[away_id][season]['ppg'].append(a_score)
        self.team_games[away_id][season]['papg'].append(h_score)
        self.team_games[away_id][season]['wts'].append(1.0)

        margin = h_score - a_score
        self.team_hca[home_id][season]['home'].append(margin)
        self.team_hca[away_id][season]['away'].append(-margin)

        self.last_game[home_id] = gdate
        self.last_game[away_id] = gdate

    def set_prev_season(self, season: int):
        prev = season - 1
        for tid in self.team_games:
            if prev in self.team_games[tid] and self.team_games[tid][prev]['ppg']:
                self.prev_ratings[tid] = {
                    'ppg': np.mean(self.team_games[tid][prev]['ppg']),
                    'papg': np.mean(self.team_games[tid][prev]['papg'])
                }

        for tid in self.team_hca:
            if prev in self.team_hca[tid]:
                hd = self.team_hca[tid][prev]
                if hd['home'] and hd['away']:
                    raw = (np.mean(hd['home']) - np.mean(hd['away'])) / 2
                    self.prev_hca[tid] = self.base_hca + self.hca_shrink * (raw - self.base_hca)

        self.last_game.clear()


def evaluate_params(games_df: pd.DataFrame, **params) -> dict:
    model = TunableModel(**params)

    errors = []
    correct = 0
    total = 0
    seasons = sorted(games_df.season.unique())

    for season in seasons:
        if season > seasons[0]:
            model.set_prev_season(season)
            pg = games_df[games_df.season == season - 1]
            if len(pg) > 0:
                model.league_avg = {'ppg': pg.home_score.mean(), 'papg': pg.away_score.mean()}

        for _, g in games_df[games_df.season == season].iterrows():
            ph, pa = model.predict(g.home_team_id, g.away_team_id, season, g.game_date)
            pred_spread = pa - ph
            actual_spread = g.away_score - g.home_score

            errors.append(abs(pred_spread - actual_spread))
            if (pred_spread < 0) == (actual_spread < 0):
                correct += 1
            total += 1

            model.update(g.home_team_id, g.away_team_id, g.home_score, g.away_score,
                        season, g.game_date)

    return {
        'spread_mae': np.mean(errors),
        'winner_acc': correct / total,
        **params
    }


def main():
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT game_id, season, date as game_date, home_team_id, away_team_id,
               home_score, away_score
        FROM games
        WHERE home_score > 0 AND completed = 1
        ORDER BY date
    ''', conn)
    conn.close()

    print(f'Loaded {len(games)} games')
    print('='*80)
    print('GRID SEARCH: Decay x HCA Half-Life x Shrinkage x Base HCA')
    print('='*80)

    # Parameter grid
    decays = [0.93, 0.95, 0.97, 0.99]
    hca_half_lives = [4, 6, 8, 10, 15, 20, 30]
    shrinkages = [0.2, 0.3, 0.4, 0.5, 0.6]
    base_hcas = [1.5, 1.8, 2.0, 2.2]

    # Also test flat HCA baseline
    print('\nBaseline (Flat HCA=1.8, decay=0.97):')
    baseline = evaluate_params(games, decay=0.97, hca_half_life=1000, hca_shrink=0.0, base_hca=1.8)
    print(f"  MAE: {baseline['spread_mae']:.4f}, Winner: {baseline['winner_acc']*100:.2f}%")

    results = []
    total_combos = len(decays) * len(hca_half_lives) * len(shrinkages) * len(base_hcas)
    print(f'\nTesting {total_combos} parameter combinations...\n')

    count = 0
    best_mae = baseline['spread_mae']
    best_acc = baseline['winner_acc']

    for decay, hca_hl, shrink, base in product(decays, hca_half_lives, shrinkages, base_hcas):
        count += 1
        if count % 50 == 0:
            print(f'  Progress: {count}/{total_combos} ({count/total_combos*100:.0f}%)')

        result = evaluate_params(
            games,
            decay=decay,
            hca_half_life=hca_hl,
            hca_shrink=shrink,
            base_hca=base
        )
        results.append(result)

        # Track best
        if result['spread_mae'] < best_mae:
            best_mae = result['spread_mae']
        if result['winner_acc'] > best_acc:
            best_acc = result['winner_acc']

    df = pd.DataFrame(results)

    # Sort by MAE
    df_mae = df.sort_values('spread_mae')

    print('\n' + '='*80)
    print('TOP 15 BY SPREAD MAE')
    print('='*80)
    print(f"{'Rank':<5} {'Decay':<7} {'HCA_HL':<8} {'Shrink':<8} {'Base':<6} {'MAE':<10} {'Win%':<8} {'vs Base'}")
    print('-'*75)

    for i, row in df_mae.head(15).iterrows():
        rank = df_mae.index.get_loc(i) + 1
        mae_diff = row['spread_mae'] - baseline['spread_mae']
        print(f"{rank:<5} {row['decay']:<7} {row['hca_half_life']:<8} {row['hca_shrink']:<8.1f} "
              f"{row['base_hca']:<6} {row['spread_mae']:<10.4f} {row['winner_acc']*100:<8.2f} {mae_diff:+.4f}")

    # Sort by winner accuracy
    df_acc = df.sort_values('winner_acc', ascending=False)

    print('\n' + '='*80)
    print('TOP 15 BY WINNER ACCURACY')
    print('='*80)
    print(f"{'Rank':<5} {'Decay':<7} {'HCA_HL':<8} {'Shrink':<8} {'Base':<6} {'MAE':<10} {'Win%':<8} {'vs Base'}")
    print('-'*75)

    for i, row in df_acc.head(15).iterrows():
        rank = df_acc.index.get_loc(i) + 1
        acc_diff = (row['winner_acc'] - baseline['winner_acc']) * 100
        print(f"{rank:<5} {row['decay']:<7} {row['hca_half_life']:<8} {row['hca_shrink']:<8.1f} "
              f"{row['base_hca']:<6} {row['spread_mae']:<10.4f} {row['winner_acc']*100:<8.2f} {acc_diff:+.2f}%")

    # Combined ranking (normalize both metrics, sum)
    df['mae_norm'] = (df['spread_mae'] - df['spread_mae'].min()) / (df['spread_mae'].max() - df['spread_mae'].min())
    df['acc_norm'] = (df['winner_acc'].max() - df['winner_acc']) / (df['winner_acc'].max() - df['winner_acc'].min())
    df['combined'] = df['mae_norm'] + df['acc_norm']
    df_comb = df.sort_values('combined')

    print('\n' + '='*80)
    print('TOP 15 COMBINED (MAE + Winner Accuracy)')
    print('='*80)
    print(f"{'Rank':<5} {'Decay':<7} {'HCA_HL':<8} {'Shrink':<8} {'Base':<6} {'MAE':<10} {'Win%':<8}")
    print('-'*65)

    for i, row in df_comb.head(15).iterrows():
        rank = df_comb.index.get_loc(i) + 1
        print(f"{rank:<5} {row['decay']:<7} {row['hca_half_life']:<8} {row['hca_shrink']:<8.1f} "
              f"{row['base_hca']:<6} {row['spread_mae']:<10.4f} {row['winner_acc']*100:<8.2f}")

    # Analysis by parameter
    print('\n' + '='*80)
    print('PARAMETER SENSITIVITY ANALYSIS')
    print('='*80)

    print('\nBy Decay:')
    for d in decays:
        subset = df[df['decay'] == d]
        print(f"  {d}: MAE={subset['spread_mae'].mean():.4f}, Win%={subset['winner_acc'].mean()*100:.2f}%")

    print('\nBy HCA Half-Life:')
    for hl in hca_half_lives:
        subset = df[df['hca_half_life'] == hl]
        print(f"  {hl:>2}: MAE={subset['spread_mae'].mean():.4f}, Win%={subset['winner_acc'].mean()*100:.2f}%")

    print('\nBy Shrinkage:')
    for s in shrinkages:
        subset = df[df['hca_shrink'] == s]
        print(f"  {s}: MAE={subset['spread_mae'].mean():.4f}, Win%={subset['winner_acc'].mean()*100:.2f}%")

    print('\nBy Base HCA:')
    for b in base_hcas:
        subset = df[df['base_hca'] == b]
        print(f"  {b}: MAE={subset['spread_mae'].mean():.4f}, Win%={subset['winner_acc'].mean()*100:.2f}%")

    # Best overall
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)
    best_mae_row = df_mae.iloc[0]
    best_acc_row = df_acc.iloc[0]
    best_comb_row = df_comb.iloc[0]

    print(f"\nBaseline (Flat HCA):     MAE={baseline['spread_mae']:.4f}, Win%={baseline['winner_acc']*100:.2f}%")
    print(f"Best MAE config:         MAE={best_mae_row['spread_mae']:.4f}, Win%={best_mae_row['winner_acc']*100:.2f}%")
    print(f"  -> decay={best_mae_row['decay']}, hca_hl={best_mae_row['hca_half_life']}, shrink={best_mae_row['hca_shrink']}, base={best_mae_row['base_hca']}")
    print(f"Best Winner config:      MAE={best_acc_row['spread_mae']:.4f}, Win%={best_acc_row['winner_acc']*100:.2f}%")
    print(f"  -> decay={best_acc_row['decay']}, hca_hl={best_acc_row['hca_half_life']}, shrink={best_acc_row['hca_shrink']}, base={best_acc_row['base_hca']}")
    print(f"Best Combined config:    MAE={best_comb_row['spread_mae']:.4f}, Win%={best_comb_row['winner_acc']*100:.2f}%")
    print(f"  -> decay={best_comb_row['decay']}, hca_hl={best_comb_row['hca_half_life']}, shrink={best_comb_row['hca_shrink']}, base={best_comb_row['base_hca']}")

    # Improvement vs baseline
    mae_improve = baseline['spread_mae'] - best_mae_row['spread_mae']
    acc_improve = (best_acc_row['winner_acc'] - baseline['winner_acc']) * 100
    print(f"\nMax MAE improvement: {mae_improve:+.4f}")
    print(f"Max Winner% improvement: {acc_improve:+.2f}%")


if __name__ == '__main__':
    main()
