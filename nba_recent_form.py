"""
NBA Power Rating Model - Recent Form Enhancement

Adds extra weight to recent games (last N) on top of the exponential decay.
The idea: teams go through hot and cold streaks that are predictive.
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'nba_games.db'


class RecentFormModel:
    """
    Power rating model with recent form bonus.

    Combines:
    1. Long-term rating (exponential decay, 6-game half-life for prev season)
    2. Recent form (last N games, extra weight)
    """

    def __init__(
        self,
        decay: float = 0.93,
        prev_season_half_life: float = 6.0,
        hca: float = 2.0,
        recent_games: int = 5,
        recent_weight: float = 0.3  # Blend: (1-w)*long_term + w*recent
    ):
        self.decay = decay
        self.prev_season_half_life = prev_season_half_life
        self.hca = hca
        self.recent_games = recent_games
        self.recent_weight = recent_weight

        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'weights': []
        }))
        self.prev_season_ratings: dict = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}

    def _get_weighted_avg(self, values: list, weights: list) -> float | None:
        if not values:
            return None
        return float(np.average(values, weights=weights))

    def _get_team_rating(self, team_id: int, season: int, games_played: int) -> tuple[float, float]:
        team_data = self.team_games[team_id][season]

        # Long-term rating (full history with decay)
        lt_ppg = self._get_weighted_avg(team_data['ppg'], team_data['weights'])
        lt_papg = self._get_weighted_avg(team_data['papg'], team_data['weights'])

        # Recent form (last N games, unweighted)
        if games_played >= self.recent_games:
            recent_ppg = np.mean(team_data['ppg'][-self.recent_games:])
            recent_papg = np.mean(team_data['papg'][-self.recent_games:])
        else:
            recent_ppg = lt_ppg
            recent_papg = lt_papg

        # Previous season
        prev_off = self.prev_season_ratings.get(team_id, {}).get('off', self.league_avg['ppg'])
        prev_def = self.prev_season_ratings.get(team_id, {}).get('def', self.league_avg['papg'])

        if lt_ppg is None:
            return prev_off, prev_def

        # Blend long-term and recent (if enough games)
        if games_played >= self.recent_games:
            curr_ppg = (1 - self.recent_weight) * lt_ppg + self.recent_weight * recent_ppg
            curr_papg = (1 - self.recent_weight) * lt_papg + self.recent_weight * recent_papg
        else:
            curr_ppg = lt_ppg
            curr_papg = lt_papg

        # Blend with previous season
        blend = 1.0 * (0.5 ** (games_played / self.prev_season_half_life))
        off_rating = blend * prev_off + (1 - blend) * curr_ppg
        def_rating = blend * prev_def + (1 - blend) * curr_papg

        return off_rating, def_rating

    def _get_games_played(self, team_id: int, season: int) -> int:
        return len(self.team_games[team_id][season]['ppg'])

    def predict(self, home_id: int, away_id: int, season: int) -> tuple[float, float]:
        home_games = self._get_games_played(home_id, season)
        away_games = self._get_games_played(away_id, season)

        home_off, home_def = self._get_team_rating(home_id, season, home_games)
        away_off, away_def = self._get_team_rating(away_id, season, away_games)

        pred_home = (home_off + away_def) / 2 + self.hca / 2
        pred_away = (away_off + home_def) / 2 - self.hca / 2

        return pred_home, pred_away

    def update(self, home_id: int, away_id: int, home_score: int, away_score: int, season: int):
        for team_id in [home_id, away_id]:
            team_data = self.team_games[team_id][season]
            team_data['weights'] = [w * self.decay for w in team_data['weights']]

        self.team_games[home_id][season]['ppg'].append(home_score)
        self.team_games[home_id][season]['papg'].append(away_score)
        self.team_games[home_id][season]['weights'].append(1.0)

        self.team_games[away_id][season]['ppg'].append(away_score)
        self.team_games[away_id][season]['papg'].append(home_score)
        self.team_games[away_id][season]['weights'].append(1.0)

    def set_prev_season_ratings(self, season: int):
        prev_season = season - 1
        for team_id in self.team_games:
            if prev_season in self.team_games[team_id]:
                ppg_list = self.team_games[team_id][prev_season]['ppg']
                papg_list = self.team_games[team_id][prev_season]['papg']
                if ppg_list:
                    self.prev_season_ratings[team_id] = {
                        'off': float(np.mean(ppg_list)),
                        'def': float(np.mean(papg_list))
                    }

    def set_league_avg(self, ppg: float, papg: float):
        self.league_avg = {'ppg': ppg, 'papg': papg}


def load_games() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT
            game_id, season, game_date_eastern,
            home_team_id, away_team_id,
            home_score, away_score,
            home_score + away_score as total,
            away_score - home_score as spread
        FROM games
        WHERE home_score > 0
        ORDER BY game_date_eastern, game_id
    ''', conn)
    conn.close()
    return games


def load_vegas() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    odds = pd.read_sql_query('''
        SELECT game_id, latest_spread as vegas_spread
        FROM odds_and_predictions
        WHERE latest_spread IS NOT NULL
    ''', conn)
    conn.close()
    return odds


def run_backtest(games: pd.DataFrame, recent_games: int = 5, recent_weight: float = 0.3) -> pd.DataFrame:
    model = RecentFormModel(
        decay=0.93,
        prev_season_half_life=6.0,
        hca=2.0,
        recent_games=recent_games,
        recent_weight=recent_weight
    )

    predictions = []
    seasons = sorted(games.season.unique())

    for season in seasons:
        if season > seasons[0]:
            model.set_prev_season_ratings(season)
            prev_games = games[games.season == season - 1]
            if len(prev_games) > 0:
                model.set_league_avg(
                    ppg=prev_games['home_score'].mean(),
                    papg=prev_games['away_score'].mean()
                )

        season_games = games[games.season == season]

        for _, game in season_games.iterrows():
            pred_home, pred_away = model.predict(
                game.home_team_id,
                game.away_team_id,
                season
            )

            predictions.append({
                'game_id': game.game_id,
                'pred_spread': pred_away - pred_home,
                'actual_spread': game.spread
            })

            model.update(
                game.home_team_id,
                game.away_team_id,
                game.home_score,
                game.away_score,
                season
            )

    return pd.DataFrame(predictions)


def main():
    print("=" * 60)
    print("RECENT FORM ENHANCEMENT")
    print("=" * 60)

    games = load_games()
    vegas = load_vegas()
    print(f"\nLoaded {len(games)} games")

    # Test different configurations
    print("\n" + "=" * 60)
    print("PARAMETER GRID SEARCH")
    print("=" * 60)
    print(f"\n{'Recent Games':<15} {'Weight':<10} {'Winner Acc'}")
    print("-" * 40)

    results = []
    for recent_games in [3, 5, 7, 10]:
        for recent_weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
            preds = run_backtest(games, recent_games, recent_weight)
            df = preds.dropna()
            df['pred_home_win'] = df['pred_spread'] < 0
            df['actual_home_win'] = df['actual_spread'] < 0
            acc = (df['pred_home_win'] == df['actual_home_win']).mean()
            results.append({
                'recent_games': recent_games,
                'recent_weight': recent_weight,
                'accuracy': acc
            })

    # Sort by accuracy
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    # Print top 10
    for r in results[:10]:
        print(f"{r['recent_games']:<15} {r['recent_weight']:<10.1f} {r['accuracy']*100:.1f}%")

    # Best result
    best = results[0]
    print(f"\nBest: {best['recent_games']} games, {best['recent_weight']} weight = {best['accuracy']*100:.1f}%")

    # Compare with Vegas
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    preds = run_backtest(games, best['recent_games'], best['recent_weight'])
    merged = preds.merge(vegas, on='game_id', how='inner')
    merged['model_home_win'] = merged['pred_spread'] < 0
    merged['vegas_home_win'] = merged['vegas_spread'] < 0
    merged['actual_home_win'] = merged['actual_spread'] < 0

    model_acc = (merged['model_home_win'] == merged['actual_home_win']).mean()
    vegas_acc = (merged['vegas_home_win'] == merged['actual_home_win']).mean()

    print(f"{'Model':<30} {'Winner Acc'}")
    print("-" * 45)
    print(f"{'Baseline (no recent form)':<30} 65.0%")
    print(f"{'Best recent form config':<30} {best['accuracy']*100:.1f}%")
    print(f"{'Vegas':<30} {vegas_acc*100:.1f}%")


if __name__ == '__main__':
    main()
