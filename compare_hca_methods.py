"""
Compare HCA methods: Flat vs Static Per-Team vs Dynamic In-Season

This script evaluates three different home court advantage approaches:
1. Flat HCA (constant 1.8 for all teams)
2. Static Per-Team HCA (from previous season, with shrinkage)
3. Dynamic In-Season HCA (building as season progresses)
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / 'nba_games.db'


class BaseModel:
    def __init__(self, decay: float = 0.97, prev_half_life: float = 6.0, b2b: float = 1.0):
        self.decay = decay
        self.prev_half_life = prev_half_life
        self.b2b = b2b
        self.team_games = defaultdict(lambda: defaultdict(lambda: {'ppg': [], 'papg': [], 'wts': []}))
        self.prev_ratings: dict = {}
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

    def _get_rest(self, tid: int, gdate: str) -> int:
        if tid not in self.last_game:
            return 3
        curr = datetime.strptime(gdate[:10], '%Y-%m-%d')
        last = datetime.strptime(self.last_game[tid][:10], '%Y-%m-%d')
        return max(0, min((curr - last).days - 1, 5))

    def get_hca(self, home_id: int, season: int) -> float:
        return 1.8  # Override in subclasses

    def predict(self, home_id: int, away_id: int, season: int, gdate: str) -> tuple[float, float]:
        h_off, h_def = self._get_rating(home_id, season)
        a_off, a_def = self._get_rating(away_id, season)
        hca = self.get_hca(home_id, season)

        pred_h = (h_off + a_def) / 2 + hca / 2
        pred_a = (a_off + h_def) / 2 - hca / 2

        # B2B
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
        self.last_game.clear()


class FlatHCAModel(BaseModel):
    """Constant HCA for all teams."""
    def get_hca(self, home_id: int, season: int) -> float:
        return 1.8


class StaticPerTeamHCA(BaseModel):
    """Per-team HCA from previous season, with shrinkage (current production approach)."""
    HCA_SCALE = 0.36
    HCA_SHRINK = 0.50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.team_hca: dict = {}
        self.league_hca: float = 1.8

    def calculate_hca(self, games_df: pd.DataFrame, season: int):
        sg = games_df[games_df['season'] == season]
        raw_hca = {}

        for tid in sg['home_team_id'].unique():
            home_games = sg[sg['home_team_id'] == tid]
            away_games = sg[sg['away_team_id'] == tid]

            if len(home_games) >= 10 and len(away_games) >= 10:
                home_margin = (home_games['home_score'] - home_games['away_score']).mean()
                away_margin = (away_games['away_score'] - away_games['home_score']).mean()
                raw_hca[tid] = home_margin - away_margin

        if not raw_hca:
            return

        league_mean = np.mean(list(raw_hca.values()))
        self.league_hca = self.HCA_SCALE * league_mean

        for tid, raw in raw_hca.items():
            shrunk = league_mean + self.HCA_SHRINK * (raw - league_mean)
            self.team_hca[tid] = self.HCA_SCALE * shrunk

    def get_hca(self, home_id: int, season: int) -> float:
        return self.team_hca.get(home_id, self.league_hca if self.league_hca else 1.8)


class DynamicHCAModel(BaseModel):
    """Build HCA dynamically within season."""
    HCA_HALF_LIFE = 6.0
    BASE_HCA = 2.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.team_hca_data = defaultdict(
            lambda: defaultdict(lambda: {'home': [], 'away': [], 'wts': []})
        )
        self.prev_hca: dict = {}

    def get_hca(self, home_id: int, season: int) -> float:
        hd = self.team_hca_data[home_id][season]
        n_home = len(hd['home'])
        n_away = len(hd['away'])
        total = n_home + n_away

        if total == 0:
            return self.prev_hca.get(home_id, self.BASE_HCA)

        if n_home > 0 and n_away > 0:
            avg_home = np.mean(hd['home'])
            avg_away = np.mean(hd['away'])
            curr_hca = (avg_home - avg_away) / 2
            curr_hca = max(-2, min(curr_hca, 8))
        else:
            curr_hca = self.BASE_HCA

        prev = self.prev_hca.get(home_id, self.BASE_HCA)
        blend = 0.5 ** (total / self.HCA_HALF_LIFE)
        return blend * prev + (1 - blend) * curr_hca

    def update(self, home_id: int, away_id: int, h_score: int, a_score: int,
               season: int, gdate: str):
        super().update(home_id, away_id, h_score, a_score, season, gdate)

        margin = h_score - a_score
        self.team_hca_data[home_id][season]['home'].append(margin)
        self.team_hca_data[away_id][season]['away'].append(-margin)

    def set_prev_season(self, season: int):
        super().set_prev_season(season)
        prev = season - 1
        for tid in self.team_hca_data:
            if prev in self.team_hca_data[tid]:
                hd = self.team_hca_data[tid][prev]
                if hd['home'] and hd['away']:
                    self.prev_hca[tid] = max(-2, min((np.mean(hd['home']) - np.mean(hd['away'])) / 2, 8))


def evaluate_model(model_class, games_df: pd.DataFrame,
                   all_games_for_hca: pd.DataFrame = None) -> tuple[dict, pd.DataFrame]:
    model = model_class()
    preds = []
    seasons = sorted(games_df.season.unique())

    for season in seasons:
        if season > seasons[0]:
            model.set_prev_season(season)
            pg = games_df[games_df.season == season - 1]
            if len(pg) > 0:
                model.league_avg = {'ppg': pg.home_score.mean(), 'papg': pg.away_score.mean()}

            # For static per-team HCA, calculate from previous season
            if hasattr(model, 'calculate_hca') and all_games_for_hca is not None:
                model.calculate_hca(all_games_for_hca, season - 1)

        for _, g in games_df[games_df.season == season].iterrows():
            ph, pa = model.predict(g.home_team_id, g.away_team_id, season, g.game_date)
            pred_spread = pa - ph
            actual_spread = g.away_score - g.home_score
            actual_total = g.home_score + g.away_score
            pred_total = ph + pa

            preds.append({
                'season': season,
                'pred_spread': pred_spread,
                'actual_spread': actual_spread,
                'pred_total': pred_total,
                'actual_total': actual_total
            })

            model.update(g.home_team_id, g.away_team_id, g.home_score, g.away_score,
                        season, g.game_date)

    df = pd.DataFrame(preds)

    # Overall metrics
    spread_mae = np.abs(df.pred_spread - df.actual_spread).mean()
    total_mae = np.abs(df.pred_total - df.actual_total).mean()

    # Winner accuracy
    df['pred_hw'] = df.pred_spread < 0
    df['actual_hw'] = df.actual_spread < 0
    winner_acc = (df.pred_hw == df.actual_hw).mean()

    return {
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'winner_acc': winner_acc,
        'n_games': len(df)
    }, df


def main():
    # Load all games
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT game_id, season, date as game_date, home_team_id, away_team_id,
               home_score, away_score
        FROM games
        WHERE home_score > 0 AND completed = 1
        ORDER BY date
    ''', conn)
    conn.close()

    print(f'Total games: {len(games)}')
    print(f'Seasons: {sorted(games.season.unique())}')

    print('\n' + '='*70)
    print('HCA COMPARISON: Flat vs Static Per-Team vs Dynamic In-Season')
    print('='*70)

    # Run all three models
    print('\nEvaluating Flat HCA (1.8)...')
    flat_results, flat_df = evaluate_model(FlatHCAModel, games)

    print('Evaluating Static Per-Team HCA (from prev season)...')
    static_results, static_df = evaluate_model(StaticPerTeamHCA, games, games)

    print('Evaluating Dynamic In-Season HCA...')
    dynamic_results, dynamic_df = evaluate_model(DynamicHCAModel, games)

    # Results table
    print('\n' + '-'*70)
    print(f"{'Model':<35} {'Spread MAE':<12} {'Total MAE':<12} {'Winner %':<10}")
    print('-'*70)
    print(f"{'Flat HCA (1.8)':<35} {flat_results['spread_mae']:.3f}       "
          f"{flat_results['total_mae']:.2f}       {flat_results['winner_acc']*100:.2f}%")
    print(f"{'Static Per-Team HCA':<35} {static_results['spread_mae']:.3f}       "
          f"{static_results['total_mae']:.2f}       {static_results['winner_acc']*100:.2f}%")
    print(f"{'Dynamic In-Season HCA':<35} {dynamic_results['spread_mae']:.3f}       "
          f"{dynamic_results['total_mae']:.2f}       {dynamic_results['winner_acc']*100:.2f}%")
    print('-'*70)

    # Improvement calculations
    print('\nIMPROVEMENTS vs Flat HCA:')
    static_mae_diff = flat_results['spread_mae'] - static_results['spread_mae']
    static_acc_diff = (static_results['winner_acc'] - flat_results['winner_acc']) * 100
    dynamic_mae_diff = flat_results['spread_mae'] - dynamic_results['spread_mae']
    dynamic_acc_diff = (dynamic_results['winner_acc'] - flat_results['winner_acc']) * 100

    print(f"  Static Per-Team:   Spread MAE: {static_mae_diff:+.4f}, Winner: {static_acc_diff:+.3f}%")
    print(f"  Dynamic In-Season: Spread MAE: {dynamic_mae_diff:+.4f}, Winner: {dynamic_acc_diff:+.3f}%")

    print('\nStatic vs Dynamic:')
    diff_mae = static_results['spread_mae'] - dynamic_results['spread_mae']
    diff_acc = (dynamic_results['winner_acc'] - static_results['winner_acc']) * 100
    better_worse_mae = 'better' if diff_mae > 0 else 'worse'
    better_worse_acc = 'better' if diff_acc > 0 else 'worse'
    print(f"  Dynamic is {better_worse_mae} by {abs(diff_mae):.4f} MAE")
    print(f"  Dynamic is {better_worse_acc} by {abs(diff_acc):.3f}% winner accuracy")

    # Season-by-season breakdown
    print('\n' + '='*70)
    print('SEASON-BY-SEASON SPREAD MAE')
    print('='*70)
    print(f"{'Season':<10} {'Flat':<10} {'Static':<10} {'Dynamic':<10} {'Best':<10}")
    print('-'*50)

    for season in sorted(games.season.unique()):
        flat_s = np.abs(flat_df[flat_df.season==season].pred_spread -
                       flat_df[flat_df.season==season].actual_spread).mean()
        static_s = np.abs(static_df[static_df.season==season].pred_spread -
                         static_df[static_df.season==season].actual_spread).mean()
        dynamic_s = np.abs(dynamic_df[dynamic_df.season==season].pred_spread -
                          dynamic_df[dynamic_df.season==season].actual_spread).mean()

        best = min(flat_s, static_s, dynamic_s)
        if flat_s == best:
            best_name = 'Flat'
        elif static_s == best:
            best_name = 'Static'
        else:
            best_name = 'Dynamic'

        print(f"{season:<10} {flat_s:<10.3f} {static_s:<10.3f} {dynamic_s:<10.3f} {best_name:<10}")

    # Per-team HCA analysis for most recent complete season
    print('\n' + '='*70)
    print('PER-TEAM HCA VALUES (2025 season)')
    print('='*70)

    # Get team abbreviations
    conn = sqlite3.connect(str(DB_PATH))
    teams = pd.read_sql_query('SELECT team_id, abbreviation FROM teams', conn)
    conn.close()
    team_abbr = dict(zip(teams.team_id, teams.abbreviation))

    # Calculate static HCA for 2025
    static_model = StaticPerTeamHCA()
    static_model.calculate_hca(games, 2025)

    # Run dynamic model through 2025 to get end-of-season HCAs
    dynamic_model = DynamicHCAModel()
    for season in sorted(games.season.unique()):
        if season > games.season.min():
            dynamic_model.set_prev_season(season)
        for _, g in games[games.season == season].iterrows():
            dynamic_model.update(g.home_team_id, g.away_team_id, g.home_score, g.away_score,
                                season, g.game_date)

    # Compare HCAs
    print(f"\n{'Team':<6} {'Static HCA':<12} {'Dynamic HCA':<12} {'Diff':<10}")
    print('-'*40)

    all_teams = set(static_model.team_hca.keys()) | set(dynamic_model.prev_hca.keys())
    hca_data = []
    for tid in all_teams:
        static_hca = static_model.team_hca.get(tid, 1.8)
        dynamic_hca = dynamic_model.prev_hca.get(tid, 2.0)
        abbr = team_abbr.get(tid, str(tid))
        hca_data.append((abbr, static_hca, dynamic_hca, dynamic_hca - static_hca))

    hca_data.sort(key=lambda x: x[1], reverse=True)
    for abbr, static, dynamic, diff in hca_data:
        print(f"{abbr:<6} {static:<12.2f} {dynamic:<12.2f} {diff:+.2f}")


if __name__ == '__main__':
    main()
