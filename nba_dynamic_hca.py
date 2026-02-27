"""
NBA Dynamic HCA Model

Instead of using previous year's HCA (no correlation), build current-year HCA
as games are played - similar to PPG/PAPG accumulation.

Hypothesis: Within-season HCA might be more predictive than cross-season.
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / 'nba_games.db'


class DynamicHCAModel:
    def __init__(
        self,
        decay: float = 0.93,
        prev_season_half_life: float = 6.0,
        base_hca: float = 2.0,
        hca_half_life: float = 6.0,  # Games before current-year HCA dominates
        b2b_penalty: float = 1.0
    ):
        self.decay = decay
        self.prev_season_half_life = prev_season_half_life
        self.base_hca = base_hca
        self.hca_half_life = hca_half_life
        self.b2b_penalty = b2b_penalty

        # PPG/PAPG tracking
        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'weights': []
        }))

        # HCA tracking: separate home and away margins
        self.team_hca: dict = defaultdict(lambda: defaultdict(lambda: {
            'home_margins': [], 'away_margins': [], 'weights': []
        }))

        self.prev_season_ratings: dict = {}
        self.prev_season_hca: dict = {}  # Store end-of-season HCA per team
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}
        self.last_game_date: dict = {}

    def _wavg(self, vals: list, wts: list) -> float | None:
        if not vals:
            return None
        return float(np.average(vals, weights=wts))

    def _get_team_rating(self, team_id: int, season: int) -> tuple[float, float]:
        td = self.team_games[team_id][season]
        games_played = len(td['ppg'])

        curr_ppg = self._wavg(td['ppg'], td['weights'])
        curr_papg = self._wavg(td['papg'], td['weights'])

        prev_off = self.prev_season_ratings.get(team_id, {}).get('off', self.league_avg['ppg'])
        prev_def = self.prev_season_ratings.get(team_id, {}).get('def', self.league_avg['papg'])

        if curr_ppg is None:
            return prev_off, prev_def

        blend = 0.5 ** (games_played / self.prev_season_half_life)
        return blend * prev_off + (1 - blend) * curr_ppg, blend * prev_def + (1 - blend) * curr_papg

    def _get_team_hca(self, team_id: int, season: int) -> float:
        """Get team's current HCA estimate, blended with previous/base."""
        hca_data = self.team_hca[team_id][season]
        home_games = len(hca_data['home_margins'])
        away_games = len(hca_data['away_margins'])
        total_games = home_games + away_games

        if total_games == 0:
            # No current season data - use previous year or base
            return self.prev_season_hca.get(team_id, self.base_hca)

        # Calculate current season HCA
        # HCA = (avg_home_margin - avg_away_margin) / 2
        if home_games > 0 and away_games > 0:
            avg_home = self._wavg(hca_data['home_margins'], hca_data['weights'][:home_games])
            # Away margins stored as team's margin (negative when losing)
            avg_away = self._wavg(hca_data['away_margins'], hca_data['weights'][home_games:])
            if avg_home is not None and avg_away is not None:
                curr_hca = (avg_home - avg_away) / 2
                # Clamp to reasonable range
                curr_hca = max(-2, min(curr_hca, 8))
            else:
                curr_hca = self.base_hca
        else:
            curr_hca = self.base_hca

        # Blend with previous/base
        prev_hca = self.prev_season_hca.get(team_id, self.base_hca)
        blend = 0.5 ** (total_games / self.hca_half_life)
        return blend * prev_hca + (1 - blend) * curr_hca

    def _get_rest_days(self, team_id: int, game_date: str) -> int:
        if team_id not in self.last_game_date:
            return 3
        curr = datetime.strptime(game_date[:10], '%Y-%m-%d')
        last = datetime.strptime(self.last_game_date[team_id][:10], '%Y-%m-%d')
        return max(0, min((curr - last).days - 1, 5))

    def predict(self, home_id: int, away_id: int, season: int, game_date: str) -> tuple[float, float]:
        home_off, home_def = self._get_team_rating(home_id, season)
        away_off, away_def = self._get_team_rating(away_id, season)

        # Get dynamic HCA for home team
        home_hca = self._get_team_hca(home_id, season)

        pred_home = (home_off + away_def) / 2 + home_hca / 2
        pred_away = (away_off + home_def) / 2 - home_hca / 2

        # B2B adjustment
        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)

        adj = 0.0
        if home_rest == 0:
            adj -= self.b2b_penalty
        if away_rest == 0:
            adj += self.b2b_penalty

        return pred_home + adj / 2, pred_away - adj / 2

    def update(self, home_id: int, away_id: int, home_score: int, away_score: int,
               season: int, game_date: str):
        home_margin = home_score - away_score

        # Decay weights for PPG/PAPG
        for tid in [home_id, away_id]:
            self.team_games[tid][season]['weights'] = [
                w * self.decay for w in self.team_games[tid][season]['weights']
            ]

        # Update PPG/PAPG
        self.team_games[home_id][season]['ppg'].append(home_score)
        self.team_games[home_id][season]['papg'].append(away_score)
        self.team_games[home_id][season]['weights'].append(1.0)

        self.team_games[away_id][season]['ppg'].append(away_score)
        self.team_games[away_id][season]['papg'].append(home_score)
        self.team_games[away_id][season]['weights'].append(1.0)

        # Update HCA tracking
        # Home team: store home margin
        self.team_hca[home_id][season]['home_margins'].append(home_margin)
        self.team_hca[home_id][season]['weights'].append(1.0)

        # Away team: store their margin (negative of home margin)
        self.team_hca[away_id][season]['away_margins'].append(-home_margin)
        self.team_hca[away_id][season]['weights'].append(1.0)

        # Update last game dates
        self.last_game_date[home_id] = game_date
        self.last_game_date[away_id] = game_date

    def set_prev_season(self, season: int):
        prev = season - 1

        # Set PPG/PAPG ratings
        for tid in self.team_games:
            if prev in self.team_games[tid] and self.team_games[tid][prev]['ppg']:
                self.prev_season_ratings[tid] = {
                    'off': float(np.mean(self.team_games[tid][prev]['ppg'])),
                    'def': float(np.mean(self.team_games[tid][prev]['papg']))
                }

        # Set HCA from previous season
        for tid in self.team_hca:
            if prev in self.team_hca[tid]:
                hca_data = self.team_hca[tid][prev]
                if hca_data['home_margins'] and hca_data['away_margins']:
                    avg_home = np.mean(hca_data['home_margins'])
                    avg_away = np.mean(hca_data['away_margins'])
                    self.prev_season_hca[tid] = max(-2, min((avg_home - avg_away) / 2, 8))

        self.last_game_date.clear()

    def set_league_avg(self, ppg: float, papg: float):
        self.league_avg = {'ppg': ppg, 'papg': papg}


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT game_id, season, game_date_eastern, home_team_id, away_team_id,
               home_score, away_score, away_score - home_score as spread
        FROM games WHERE home_score > 0
        ORDER BY game_date_eastern, game_id
    ''', conn)
    conn.close()
    return games


def run_test(games: pd.DataFrame, hca_half_life: float = 6.0, use_dynamic: bool = True):
    model = DynamicHCAModel(
        decay=0.93,
        prev_season_half_life=6.0,
        base_hca=2.0,
        hca_half_life=hca_half_life,
        b2b_penalty=1.0
    )

    # If not using dynamic HCA, override the method
    if not use_dynamic:
        model._get_team_hca = lambda tid, s: 2.0  # Fixed HCA

    preds = []
    for season in sorted(games.season.unique()):
        if season > games.season.min():
            model.set_prev_season(season)
            pg = games[games.season == season - 1]
            if len(pg) > 0:
                model.set_league_avg(pg['home_score'].mean(), pg['away_score'].mean())

        for _, g in games[games.season == season].iterrows():
            ph, pa = model.predict(g.home_team_id, g.away_team_id, season, g.game_date_eastern)
            preds.append({'pred_spread': pa - ph, 'actual_spread': g.spread})
            model.update(g.home_team_id, g.away_team_id, g.home_score, g.away_score,
                        season, g.game_date_eastern)

    df = pd.DataFrame(preds)
    df['pred_hw'] = df['pred_spread'] < 0
    df['actual_hw'] = df['actual_spread'] < 0
    return (df['pred_hw'] == df['actual_hw']).mean()


def main():
    games = load_data()
    print('=' * 60)
    print('DYNAMIC IN-SEASON HCA MODEL')
    print('=' * 60)
    print(f'\nLoaded {len(games)} games')

    # Baseline (fixed HCA=2.0)
    baseline = run_test(games, use_dynamic=False)
    print(f'\nBaseline (fixed HCA=2.0): {baseline*100:.2f}%')

    # Test different HCA half-lives
    print(f"\n{'HCA Half-Life':<15} {'Accuracy':<12} {'vs Baseline'}")
    print('-' * 45)

    results = []
    for hl in [3, 4, 5, 6, 8, 10, 15, 20]:
        acc = run_test(games, hca_half_life=hl, use_dynamic=True)
        diff = acc - baseline
        marker = ' <--' if acc > baseline else ''
        print(f'{hl:<15} {acc*100:<11.2f}% ({diff*100:+.2f}%){marker}')
        results.append((hl, acc))

    best_hl, best_acc = max(results, key=lambda x: x[1])
    print(f'\nBest HCA half-life: {best_hl} games')
    print(f'Best accuracy: {best_acc*100:.2f}%')
    print(f'Improvement: {(best_acc - baseline)*100:+.2f}%')

    # Analyze within-season HCA correlation
    print('\n' + '=' * 60)
    print('WITHIN-SEASON HCA ANALYSIS')
    print('=' * 60)

    # Track HCA by team and see if early-season predicts late-season
    conn = sqlite3.connect(str(DB_PATH))
    teams = pd.read_sql_query('SELECT team_id, abbreviation FROM teams', conn)
    conn.close()
    team_abbr = dict(zip(teams.team_id, teams.abbreviation))

    # For 2024 season, compare first-half HCA to second-half performance
    season_2024 = games[games.season == 2024].copy()
    midpoint = len(season_2024) // 2

    first_half = season_2024.iloc[:midpoint]
    second_half = season_2024.iloc[midpoint:]

    def calc_team_hca(df):
        hca = {}
        for tid in df.home_team_id.unique():
            home_games = df[df.home_team_id == tid]
            away_games = df[df.away_team_id == tid]
            if len(home_games) >= 5 and len(away_games) >= 5:
                home_margin = (home_games.home_score - home_games.away_score).mean()
                away_margin = (away_games.away_score - away_games.home_score).mean()
                hca[tid] = (home_margin - away_margin) / 2
        return hca

    first_hca = calc_team_hca(first_half)
    second_hca = calc_team_hca(second_half)

    common = set(first_hca.keys()) & set(second_hca.keys())
    if len(common) >= 10:
        x = [first_hca[t] for t in common]
        y = [second_hca[t] for t in common]
        corr = np.corrcoef(x, y)[0, 1]
        print(f'\n2024 Season: First-half HCA vs Second-half HCA')
        print(f'Teams with enough data: {len(common)}')
        print(f'Correlation: {corr:.3f}')

        if corr > 0.3:
            print('-> Moderate correlation - dynamic HCA might help!')
        elif corr > 0.1:
            print('-> Weak correlation - marginal benefit possible')
        else:
            print('-> No correlation - dynamic HCA unlikely to help')


if __name__ == '__main__':
    main()
