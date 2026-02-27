"""
NBA Player-Adjusted Power Rating Model

Adjusts predictions based on missing players:
- Tracks each player's rolling impact (plus/minus per game)
- When a star player doesn't play, adjusts team's expected performance
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / 'nba_games.db'


class PlayerAdjustedModel:
    def __init__(
        self,
        decay: float = 0.93,
        prev_season_half_life: float = 6.0,
        hca: float = 2.0,
        b2b_penalty: float = 1.0,
        player_adj_factor: float = 0.5,  # How much of missing impact to apply
        min_games_for_impact: int = 5
    ):
        self.decay = decay
        self.prev_season_half_life = prev_season_half_life
        self.hca = hca
        self.b2b_penalty = b2b_penalty
        self.player_adj_factor = player_adj_factor
        self.min_games_for_impact = min_games_for_impact

        # PPG/PAPG tracking
        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'weights': []
        }))

        # Player impact tracking: player_id -> {season: {pm_total, games}}
        self.player_stats: dict = defaultdict(lambda: defaultdict(lambda: {
            'pm_total': 0.0, 'games': 0, 'teams': set()
        }))

        # Team regular players: team_id -> {season: set of player_ids}
        self.team_regulars: dict = defaultdict(lambda: defaultdict(set))

        self.prev_season_ratings: dict = {}
        self.prev_player_impact: dict = {}  # player_id -> impact_per_game
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

    def _get_player_impact(self, player_id: int, season: int) -> float:
        """Get player's impact per game (rolling estimate)."""
        ps = self.player_stats[player_id][season]
        if ps['games'] >= self.min_games_for_impact:
            return ps['pm_total'] / ps['games']
        # Fall back to previous season
        return self.prev_player_impact.get(player_id, 0.0)

    def _get_missing_impact(self, team_id: int, season: int, playing_ids: set[int]) -> float:
        """Calculate total missing impact for players who normally play but aren't today."""
        regulars = self.team_regulars[team_id][season]
        missing = regulars - playing_ids

        total_missing = 0.0
        for pid in missing:
            impact = self._get_player_impact(pid, season)
            if abs(impact) > 0.1:  # Only count significant impacts
                total_missing += impact

        return total_missing

    def _get_rest_days(self, team_id: int, game_date: str) -> int:
        if team_id not in self.last_game_date:
            return 3
        curr = datetime.strptime(game_date[:10], '%Y-%m-%d')
        last = datetime.strptime(self.last_game_date[team_id][:10], '%Y-%m-%d')
        return max(0, min((curr - last).days - 1, 5))

    def predict(
        self,
        home_id: int,
        away_id: int,
        season: int,
        game_date: str,
        home_players: set[int] | None = None,
        away_players: set[int] | None = None
    ) -> tuple[float, float]:
        home_off, home_def = self._get_team_rating(home_id, season)
        away_off, away_def = self._get_team_rating(away_id, season)

        pred_home = (home_off + away_def) / 2 + self.hca / 2
        pred_away = (away_off + home_def) / 2 - self.hca / 2

        # B2B adjustment
        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)

        adj = 0.0
        if home_rest == 0:
            adj -= self.b2b_penalty
        if away_rest == 0:
            adj += self.b2b_penalty

        # Player availability adjustment
        if home_players is not None:
            home_missing = self._get_missing_impact(home_id, season, home_players)
            # Missing impact is positive if good players are out
            # So we subtract (reduce home's expected score)
            adj -= home_missing * self.player_adj_factor

        if away_players is not None:
            away_missing = self._get_missing_impact(away_id, season, away_players)
            adj += away_missing * self.player_adj_factor

        return pred_home + adj / 2, pred_away - adj / 2

    def update(
        self,
        home_id: int,
        away_id: int,
        home_score: int,
        away_score: int,
        season: int,
        game_date: str,
        player_stats_list: list[dict] | None = None
    ):
        # Update PPG/PAPG
        for tid in [home_id, away_id]:
            self.team_games[tid][season]['weights'] = [
                w * self.decay for w in self.team_games[tid][season]['weights']
            ]

        self.team_games[home_id][season]['ppg'].append(home_score)
        self.team_games[home_id][season]['papg'].append(away_score)
        self.team_games[home_id][season]['weights'].append(1.0)

        self.team_games[away_id][season]['ppg'].append(away_score)
        self.team_games[away_id][season]['papg'].append(home_score)
        self.team_games[away_id][season]['weights'].append(1.0)

        # Update player stats
        if player_stats_list:
            for ps in player_stats_list:
                pid = ps['player_id']
                tid = ps['team_id']
                pm = ps['plus_minus']

                self.player_stats[pid][season]['pm_total'] += pm
                self.player_stats[pid][season]['games'] += 1
                self.player_stats[pid][season]['teams'].add(tid)

                # Add to team regulars
                self.team_regulars[tid][season].add(pid)

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

        # Set player impacts from previous season
        for pid in self.player_stats:
            if prev in self.player_stats[pid]:
                ps = self.player_stats[pid][prev]
                if ps['games'] >= self.min_games_for_impact:
                    self.prev_player_impact[pid] = ps['pm_total'] / ps['games']

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

    player_game_stats = pd.read_sql_query('''
        SELECT game_id, player_id, team_id, minutes, plus_minus
        FROM player_game_stats
        WHERE minutes > 0
    ''', conn)

    conn.close()
    return games, player_game_stats


def run_backtest(
    games: pd.DataFrame,
    pgs: pd.DataFrame,
    player_adj_factor: float = 0.5,
    use_player_adj: bool = True
):
    model = PlayerAdjustedModel(
        decay=0.93,
        prev_season_half_life=6.0,
        hca=2.0,
        b2b_penalty=1.0,
        player_adj_factor=player_adj_factor
    )

    # Index player stats by game_id
    pgs_by_game = pgs.groupby('game_id').apply(
        lambda x: x.to_dict('records')
    ).to_dict()

    preds = []
    for season in sorted(games.season.unique()):
        if season > games.season.min():
            model.set_prev_season(season)
            pg = games[games.season == season - 1]
            if len(pg) > 0:
                model.set_league_avg(pg['home_score'].mean(), pg['away_score'].mean())

        season_games = games[games.season == season]

        for _, g in season_games.iterrows():
            gid = g['game_id']
            game_pgs = pgs_by_game.get(gid, [])

            if use_player_adj and game_pgs:
                home_players = {p['player_id'] for p in game_pgs if p['team_id'] == g['home_team_id']}
                away_players = {p['player_id'] for p in game_pgs if p['team_id'] == g['away_team_id']}
            else:
                home_players = None
                away_players = None

            ph, pa = model.predict(
                g['home_team_id'], g['away_team_id'],
                season, g['game_date_eastern'],
                home_players, away_players
            )

            preds.append({
                'game_id': gid,
                'season': season,
                'pred_spread': pa - ph,
                'actual_spread': g['spread']
            })

            model.update(
                g['home_team_id'], g['away_team_id'],
                g['home_score'], g['away_score'],
                season, g['game_date_eastern'],
                game_pgs
            )

    df = pd.DataFrame(preds)
    df['pred_hw'] = df['pred_spread'] < 0
    df['actual_hw'] = df['actual_spread'] < 0
    return df


def main():
    print('=' * 60)
    print('PLAYER-ADJUSTED POWER RATING MODEL')
    print('=' * 60)

    games, pgs = load_data()
    print(f'\nLoaded {len(games)} games, {len(pgs)} player-game records')

    # Only test on seasons with player data
    games = games[games.season >= 2024]
    print(f'Testing on seasons with player data: {sorted(games.season.unique())}')

    # Baseline (no player adjustment)
    baseline_df = run_backtest(games, pgs, use_player_adj=False)
    # Filter to seasons after first (need prev season data)
    baseline_df = baseline_df[baseline_df.season > baseline_df.season.min()]
    baseline_acc = (baseline_df['pred_hw'] == baseline_df['actual_hw']).mean()
    print(f'\nBaseline (no player adj): {baseline_acc*100:.2f}%')

    # Test different player adjustment factors
    print(f"\n{'Factor':<12} {'Accuracy':<12} {'vs Baseline'}")
    print('-' * 40)

    results = []
    for factor in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        df = run_backtest(games, pgs, player_adj_factor=factor, use_player_adj=True)
        df = df[df.season > df.season.min()]
        acc = (df['pred_hw'] == df['actual_hw']).mean()
        diff = acc - baseline_acc
        marker = ' <--' if acc > baseline_acc else ''
        print(f'{factor:<12.1f} {acc*100:<11.2f}% ({diff*100:+.2f}%){marker}')
        results.append((factor, acc))

    best_factor, best_acc = max(results, key=lambda x: x[1])
    print(f'\nBest player_adj_factor: {best_factor}')
    print(f'Best accuracy: {best_acc*100:.2f}%')
    print(f'Improvement over baseline: {(best_acc - baseline_acc)*100:+.2f}%')

    # Analyze where player adjustment helps most
    print('\n' + '=' * 60)
    print('ANALYSIS: WHEN DOES PLAYER ADJUSTMENT HELP?')
    print('=' * 60)

    df = run_backtest(games, pgs, player_adj_factor=best_factor, use_player_adj=True)
    df = df[df.season > df.season.min()]

    # Compare by spread magnitude
    df['spread_abs'] = df['actual_spread'].abs()
    for bucket, (lo, hi) in [('Close (<5)', (0, 5)), ('Medium (5-10)', (5, 10)), ('Blowout (>10)', (10, 100))]:
        subset = df[(df['spread_abs'] >= lo) & (df['spread_abs'] < hi)]
        if len(subset) > 50:
            acc = (subset['pred_hw'] == subset['actual_hw']).mean()
            print(f'{bucket}: {acc*100:.1f}% ({len(subset)} games)')


if __name__ == '__main__':
    main()
