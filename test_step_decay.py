"""
Test step-function decay for previous season blending.

Step function: 100%, 75%, 50%, 25%, 0% for games 0-4+
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DB_PATH = Path(__file__).parent / 'nba_games.db'


class PowerRatingModelStep:
    """Power rating model with step-function decay for previous season."""

    def __init__(self, decay: float = 0.93, hca: float = 2.0):
        self.decay = decay
        self.hca = hca
        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {'ppg': [], 'papg': [], 'weights': []}))
        self.prev_season_ratings: dict = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}

    def _get_weighted_avg(self, values: list, weights: list) -> float | None:
        if not values:
            return None
        weights_arr = np.array(weights)
        values_arr = np.array(values)
        return float(np.sum(values_arr * weights_arr) / np.sum(weights_arr))

    def _get_team_rating(self, team_id: int, season: int, games_played: int) -> tuple[float, float]:
        # Current season data
        team_data = self.team_games[team_id][season]
        curr_ppg = self._get_weighted_avg(team_data['ppg'], team_data['weights'])
        curr_papg = self._get_weighted_avg(team_data['papg'], team_data['weights'])

        # Previous season data (fallback to league avg)
        prev_off = self.prev_season_ratings.get(team_id, {}).get('off', self.league_avg['ppg'])
        prev_def = self.prev_season_ratings.get(team_id, {}).get('def', self.league_avg['papg'])

        # If no current season data, use previous season
        if curr_ppg is None:
            return prev_off, prev_def

        # Step function: 100%, 75%, 50%, 25%, 0% for games 0-4+
        if games_played >= 4:
            blend = 0.0
        else:
            blend = 1.0 - (games_played * 0.25)  # 1.0, 0.75, 0.50, 0.25

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


def run_backtest(games: pd.DataFrame) -> pd.DataFrame:
    model = PowerRatingModelStep(decay=0.93, hca=2.0)

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
                'season': season,
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


def test_step_function(games: pd.DataFrame, steps: list[float], name: str) -> float:
    """Test a custom step function and return winner accuracy."""

    class CustomStepModel(PowerRatingModelStep):
        def _get_team_rating(self, team_id: int, season: int, games_played: int) -> tuple[float, float]:
            team_data = self.team_games[team_id][season]
            curr_ppg = self._get_weighted_avg(team_data['ppg'], team_data['weights'])
            curr_papg = self._get_weighted_avg(team_data['papg'], team_data['weights'])

            prev_off = self.prev_season_ratings.get(team_id, {}).get('off', self.league_avg['ppg'])
            prev_def = self.prev_season_ratings.get(team_id, {}).get('def', self.league_avg['papg'])

            if curr_ppg is None:
                return prev_off, prev_def

            # Custom step function
            if games_played >= len(steps):
                blend = steps[-1]
            else:
                blend = steps[games_played]

            off_rating = blend * prev_off + (1 - blend) * curr_ppg
            def_rating = blend * prev_def + (1 - blend) * curr_papg

            return off_rating, def_rating

    model = CustomStepModel(decay=0.93, hca=2.0)
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

    df = pd.DataFrame(predictions).dropna()
    df['pred_home_win'] = df['pred_spread'] < 0
    df['actual_home_win'] = df['actual_spread'] < 0
    winner_acc = (df['pred_home_win'] == df['actual_home_win']).mean()

    return winner_acc


def main():
    print("=" * 60)
    print("STEP-FUNCTION DECAY VARIATIONS")
    print("=" * 60)

    games = load_games()
    print(f"Loaded {len(games)} games\n")

    # Test various step functions
    step_configs = [
        # (name, steps list where index = games_played)
        ("4-game: 100,75,50,25,0", [1.0, 0.75, 0.50, 0.25, 0.0]),
        ("5-game: 100,80,60,40,20,0", [1.0, 0.80, 0.60, 0.40, 0.20, 0.0]),
        ("6-game: 100,83,67,50,33,17,0", [1.0, 0.83, 0.67, 0.50, 0.33, 0.17, 0.0]),
        ("8-game gradual", [1.0, 0.875, 0.75, 0.625, 0.50, 0.375, 0.25, 0.125, 0.0]),
        ("10-game gradual", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
        ("Keep 10% residual", [1.0, 0.75, 0.50, 0.25, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        ("Keep 20% residual", [1.0, 0.80, 0.60, 0.40, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]),
    ]

    results = []
    for name, steps in step_configs:
        acc = test_step_function(games, steps, name)
        results.append((name, acc))
        print(f"  {name:30s} {acc*100:.1f}%")

    print("\n" + "=" * 60)
    print("COMPARISON WITH EXPONENTIAL DECAY")
    print("=" * 60)
    print("  Half-life 4:                   65.0%")
    print("  Half-life 6:                   65.0%")
    print("  Half-life 8:                   65.0%")
    print("  Half-life 10:                  64.9%")

    best = max(results, key=lambda x: x[1])
    print(f"\n  Best step function: {best[0]} at {best[1]*100:.1f}%")


if __name__ == '__main__':
    main()
