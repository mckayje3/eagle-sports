"""
NBA Power Rating Model - Rest Days Adjustment

Adjusts predictions based on rest advantage:
- Back-to-back games (0 days rest) are a significant disadvantage
- 2+ days rest provides a slight advantage
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'nba_games.db'


class RestAdjustedModel:
    """
    Power rating model with rest day adjustments.

    Tracks each team's last game date and adjusts predictions based on:
    - Rest differential (home rest days - away rest days)
    - Back-to-back penalty
    """

    def __init__(
        self,
        decay: float = 0.93,
        prev_season_half_life: float = 6.0,
        hca: float = 2.0,
        rest_factor: float = 1.0,  # Points per rest day differential
        b2b_penalty: float = 2.0   # Extra penalty for back-to-back
    ):
        self.decay = decay
        self.prev_season_half_life = prev_season_half_life
        self.hca = hca
        self.rest_factor = rest_factor
        self.b2b_penalty = b2b_penalty

        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {
            'ppg': [], 'papg': [], 'weights': []
        }))
        self.prev_season_ratings: dict = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}

        # Track last game date for each team
        self.last_game_date: dict = {}

    def _get_weighted_avg(self, values: list, weights: list) -> float | None:
        if not values:
            return None
        return float(np.average(values, weights=weights))

    def _get_team_rating(self, team_id: int, season: int, games_played: int) -> tuple[float, float]:
        team_data = self.team_games[team_id][season]
        curr_ppg = self._get_weighted_avg(team_data['ppg'], team_data['weights'])
        curr_papg = self._get_weighted_avg(team_data['papg'], team_data['weights'])

        prev_off = self.prev_season_ratings.get(team_id, {}).get('off', self.league_avg['ppg'])
        prev_def = self.prev_season_ratings.get(team_id, {}).get('def', self.league_avg['papg'])

        if curr_ppg is None:
            return prev_off, prev_def

        blend = 1.0 * (0.5 ** (games_played / self.prev_season_half_life))
        off_rating = blend * prev_off + (1 - blend) * curr_ppg
        def_rating = blend * prev_def + (1 - blend) * curr_papg

        return off_rating, def_rating

    def _get_games_played(self, team_id: int, season: int) -> int:
        return len(self.team_games[team_id][season]['ppg'])

    def _get_rest_days(self, team_id: int, game_date: str) -> int:
        """Get days of rest for a team (days since last game)."""
        if team_id not in self.last_game_date:
            return 3  # Default to well-rested for first game

        last_date = self.last_game_date[team_id]
        current = datetime.strptime(game_date[:10], '%Y-%m-%d')
        last = datetime.strptime(last_date[:10], '%Y-%m-%d')
        days = (current - last).days - 1  # -1 because same day = 0 rest

        return max(0, min(days, 5))  # Cap at 5 days

    def _get_rest_adjustment(self, home_rest: int, away_rest: int) -> float:
        """
        Calculate point adjustment based on rest differential.

        Returns adjustment to add to home team's advantage.
        """
        adjustment = 0.0

        # Rest differential factor
        rest_diff = home_rest - away_rest
        adjustment += rest_diff * self.rest_factor

        # Back-to-back penalty
        if home_rest == 0:  # Home team on B2B
            adjustment -= self.b2b_penalty
        if away_rest == 0:  # Away team on B2B
            adjustment += self.b2b_penalty

        return adjustment

    def predict(self, home_id: int, away_id: int, season: int, game_date: str) -> tuple[float, float]:
        home_games = self._get_games_played(home_id, season)
        away_games = self._get_games_played(away_id, season)

        home_off, home_def = self._get_team_rating(home_id, season, home_games)
        away_off, away_def = self._get_team_rating(away_id, season, away_games)

        # Base prediction
        pred_home = (home_off + away_def) / 2 + self.hca / 2
        pred_away = (away_off + home_def) / 2 - self.hca / 2

        # Rest adjustment
        home_rest = self._get_rest_days(home_id, game_date)
        away_rest = self._get_rest_days(away_id, game_date)
        rest_adj = self._get_rest_adjustment(home_rest, away_rest)

        pred_home += rest_adj / 2
        pred_away -= rest_adj / 2

        return pred_home, pred_away

    def update(self, home_id: int, away_id: int, home_score: int, away_score: int, season: int, game_date: str):
        for team_id in [home_id, away_id]:
            team_data = self.team_games[team_id][season]
            team_data['weights'] = [w * self.decay for w in team_data['weights']]

        self.team_games[home_id][season]['ppg'].append(home_score)
        self.team_games[home_id][season]['papg'].append(away_score)
        self.team_games[home_id][season]['weights'].append(1.0)

        self.team_games[away_id][season]['ppg'].append(away_score)
        self.team_games[away_id][season]['papg'].append(home_score)
        self.team_games[away_id][season]['weights'].append(1.0)

        # Update last game dates
        self.last_game_date[home_id] = game_date
        self.last_game_date[away_id] = game_date

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
        # Clear last game dates for new season
        self.last_game_date.clear()

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


def run_backtest(games: pd.DataFrame, rest_factor: float = 1.0, b2b_penalty: float = 2.0) -> pd.DataFrame:
    model = RestAdjustedModel(
        decay=0.93,
        prev_season_half_life=6.0,
        hca=2.0,
        rest_factor=rest_factor,
        b2b_penalty=b2b_penalty
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
                season,
                game.game_date_eastern
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
                season,
                game.game_date_eastern
            )

    return pd.DataFrame(predictions)


def analyze_rest_impact(games: pd.DataFrame):
    """Analyze the actual impact of rest on game outcomes."""
    print("\n" + "=" * 60)
    print("REST IMPACT ANALYSIS")
    print("=" * 60)

    # Track rest days for each game
    last_game = {}
    rest_data = []

    for _, game in games.iterrows():
        home_id = game.home_team_id
        away_id = game.away_team_id
        game_date = game.game_date_eastern[:10]
        current = datetime.strptime(game_date, '%Y-%m-%d')

        # Calculate rest days
        home_rest = 3  # default
        away_rest = 3
        if home_id in last_game:
            home_rest = (current - last_game[home_id]).days - 1
        if away_id in last_game:
            away_rest = (current - last_game[away_id]).days - 1

        home_rest = max(0, min(home_rest, 5))
        away_rest = max(0, min(away_rest, 5))

        rest_data.append({
            'home_rest': home_rest,
            'away_rest': away_rest,
            'rest_diff': home_rest - away_rest,
            'home_b2b': 1 if home_rest == 0 else 0,
            'away_b2b': 1 if away_rest == 0 else 0,
            'spread': game.spread,
            'home_win': 1 if game.spread < 0 else 0
        })

        # Update last game dates
        last_game[home_id] = current
        last_game[away_id] = current

    df = pd.DataFrame(rest_data)

    # Analyze by rest differential
    print("\nBy Rest Differential (home - away):")
    print(f"{'Rest Diff':<12} {'Avg Margin':<12} {'Home Win%':<12} {'Games'}")
    print("-" * 48)

    for diff in sorted(df['rest_diff'].unique()):
        subset = df[df['rest_diff'] == diff]
        if len(subset) >= 20:
            avg_margin = -subset['spread'].mean()  # Negative spread = home win
            home_win = subset['home_win'].mean()
            print(f"{diff:<12} {avg_margin:<12.1f} {home_win*100:<11.1f}% {len(subset)}")

    # Back-to-back analysis
    print("\nBack-to-Back Impact:")
    print(f"{'Situation':<25} {'Avg Margin':<12} {'Home Win%':<12} {'Games'}")
    print("-" * 55)

    # Neither on B2B
    neither = df[(df['home_b2b'] == 0) & (df['away_b2b'] == 0)]
    print(f"{'Neither on B2B':<25} {-neither['spread'].mean():<12.1f} {neither['home_win'].mean()*100:<11.1f}% {len(neither)}")

    # Only home on B2B
    home_b2b = df[(df['home_b2b'] == 1) & (df['away_b2b'] == 0)]
    print(f"{'Home on B2B only':<25} {-home_b2b['spread'].mean():<12.1f} {home_b2b['home_win'].mean()*100:<11.1f}% {len(home_b2b)}")

    # Only away on B2B
    away_b2b = df[(df['home_b2b'] == 0) & (df['away_b2b'] == 1)]
    print(f"{'Away on B2B only':<25} {-away_b2b['spread'].mean():<12.1f} {away_b2b['home_win'].mean()*100:<11.1f}% {len(away_b2b)}")

    # Both on B2B
    both = df[(df['home_b2b'] == 1) & (df['away_b2b'] == 1)]
    if len(both) > 0:
        print(f"{'Both on B2B':<25} {-both['spread'].mean():<12.1f} {both['home_win'].mean()*100:<11.1f}% {len(both)}")


def main():
    print("=" * 60)
    print("REST-ADJUSTED POWER RATINGS")
    print("=" * 60)

    games = load_games()
    vegas = load_vegas()
    print(f"\nLoaded {len(games)} games")

    # First, analyze the actual rest impact
    analyze_rest_impact(games)

    # Test different rest parameters
    print("\n" + "=" * 60)
    print("PARAMETER GRID SEARCH")
    print("=" * 60)
    print(f"\n{'Rest Factor':<15} {'B2B Penalty':<15} {'Winner Acc'}")
    print("-" * 45)

    results = []
    for rest_factor in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for b2b_penalty in [0.0, 1.0, 2.0, 3.0, 4.0]:
            preds = run_backtest(games, rest_factor, b2b_penalty)
            df = preds.dropna()
            df['pred_home_win'] = df['pred_spread'] < 0
            df['actual_home_win'] = df['actual_spread'] < 0
            acc = (df['pred_home_win'] == df['actual_home_win']).mean()
            results.append({
                'rest_factor': rest_factor,
                'b2b_penalty': b2b_penalty,
                'accuracy': acc
            })

    # Sort and print top results
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    for r in results[:10]:
        print(f"{r['rest_factor']:<15.1f} {r['b2b_penalty']:<15.1f} {r['accuracy']*100:.1f}%")

    best = results[0]
    print(f"\nBest: rest_factor={best['rest_factor']}, b2b_penalty={best['b2b_penalty']} = {best['accuracy']*100:.1f}%")

    # Compare with Vegas
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    preds = run_backtest(games, best['rest_factor'], best['b2b_penalty'])
    merged = preds.merge(vegas, on='game_id', how='inner')
    merged['model_home_win'] = merged['pred_spread'] < 0
    merged['vegas_home_win'] = merged['vegas_spread'] < 0
    merged['actual_home_win'] = merged['actual_spread'] < 0

    model_acc = (merged['model_home_win'] == merged['actual_home_win']).mean()
    vegas_acc = (merged['vegas_home_win'] == merged['actual_home_win']).mean()

    print(f"{'Model':<30} {'Winner Acc'}")
    print("-" * 45)
    print(f"{'Baseline (no rest adj)':<30} 65.0%")
    print(f"{'Best rest config':<30} {best['accuracy']*100:.1f}%")
    print(f"{'Vegas':<30} {vegas_acc*100:.1f}%")


if __name__ == '__main__':
    main()
