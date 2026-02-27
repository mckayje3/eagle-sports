"""
NBA Power Rating Model - Opponent Adjusted

Extends baseline model by adjusting PPG/PAPG based on opponent strength.

Key concept:
- If you score 120 vs a defense that allows 115 (league avg), that's average
- If you score 120 vs a defense that allows 105 (good), that's impressive (+15 adjusted)
- If you score 120 vs a defense that allows 125 (bad), that's less impressive (-10 adjusted)
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


class OpponentAdjustedModel:
    """
    Power rating model with opponent adjustments.

    Adjustment formula (additive):
        adjusted_ppg = raw_ppg + (league_avg - opponent_papg)
        adjusted_papg = raw_papg + (league_avg - opponent_ppg)

    If opponent allows 120 (bad D), and league avg is 115:
        adjusted = raw + (115 - 120) = raw - 5 (score is less impressive)

    If opponent allows 110 (good D):
        adjusted = raw + (115 - 110) = raw + 5 (score is more impressive)
    """

    def __init__(
        self,
        decay: float = 0.93,
        prev_season_half_life: float = 6.0,
        hca: float = 2.0,
        adjustment_strength: float = 1.0  # 0 = no adjustment, 1 = full adjustment
    ):
        self.decay = decay
        self.prev_season_half_life = prev_season_half_life
        self.hca = hca
        self.adjustment_strength = adjustment_strength

        # Team games: {team_id: {season: {'ppg': [], 'papg': [], 'weights': []}}}
        # Now stores ADJUSTED values
        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {
            'adj_ppg': [], 'adj_papg': [], 'weights': [],
            'raw_ppg': [], 'raw_papg': []  # Keep raw for opponent lookups
        }))

        self.prev_season_ratings: dict = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}

    def _get_weighted_avg(self, values: list, weights: list) -> float | None:
        if not values:
            return None
        weights_arr = np.array(weights)
        values_arr = np.array(values)
        return float(np.sum(values_arr * weights_arr) / np.sum(weights_arr))

    def _get_opponent_rating(self, team_id: int, season: int) -> tuple[float, float]:
        """Get opponent's current raw offensive and defensive rating (for adjustment)."""
        team_data = self.team_games[team_id][season]

        # Use raw PPG/PAPG for opponent adjustment calculation
        raw_ppg = self._get_weighted_avg(team_data['raw_ppg'], team_data['weights'])
        raw_papg = self._get_weighted_avg(team_data['raw_papg'], team_data['weights'])

        if raw_ppg is None:
            # Fall back to previous season or league avg
            prev = self.prev_season_ratings.get(team_id, {})
            raw_ppg = prev.get('raw_off', self.league_avg['ppg'])
            raw_papg = prev.get('raw_def', self.league_avg['papg'])

        return raw_ppg, raw_papg

    def _get_team_rating(self, team_id: int, season: int, games_played: int) -> tuple[float, float]:
        """Get team's current adjusted offensive and defensive rating."""
        team_data = self.team_games[team_id][season]

        # Use adjusted PPG/PAPG for team rating
        curr_ppg = self._get_weighted_avg(team_data['adj_ppg'], team_data['weights'])
        curr_papg = self._get_weighted_avg(team_data['adj_papg'], team_data['weights'])

        prev_off = self.prev_season_ratings.get(team_id, {}).get('off', self.league_avg['ppg'])
        prev_def = self.prev_season_ratings.get(team_id, {}).get('def', self.league_avg['papg'])

        if curr_ppg is None:
            return prev_off, prev_def

        # Exponential half-life blend
        blend = 1.0 * (0.5 ** (games_played / self.prev_season_half_life))

        off_rating = blend * prev_off + (1 - blend) * curr_ppg
        def_rating = blend * prev_def + (1 - blend) * curr_papg

        return off_rating, def_rating

    def _get_games_played(self, team_id: int, season: int) -> int:
        return len(self.team_games[team_id][season]['adj_ppg'])

    def predict(self, home_id: int, away_id: int, season: int) -> tuple[float, float]:
        home_games = self._get_games_played(home_id, season)
        away_games = self._get_games_played(away_id, season)

        home_off, home_def = self._get_team_rating(home_id, season, home_games)
        away_off, away_def = self._get_team_rating(away_id, season, away_games)

        pred_home = (home_off + away_def) / 2 + self.hca / 2
        pred_away = (away_off + home_def) / 2 - self.hca / 2

        return pred_home, pred_away

    def update(self, home_id: int, away_id: int, home_score: int, away_score: int, season: int):
        """Update ratings after a game with opponent adjustment."""

        # Get opponent ratings BEFORE updating (what we knew going into the game)
        home_opp_off, home_opp_def = self._get_opponent_rating(away_id, season)  # Home's opponent is away
        away_opp_off, away_opp_def = self._get_opponent_rating(home_id, season)  # Away's opponent is home

        # Decay existing weights
        for team_id in [home_id, away_id]:
            team_data = self.team_games[team_id][season]
            team_data['weights'] = [w * self.decay for w in team_data['weights']]

        # Store raw values
        self.team_games[home_id][season]['raw_ppg'].append(home_score)
        self.team_games[home_id][season]['raw_papg'].append(away_score)
        self.team_games[away_id][season]['raw_ppg'].append(away_score)
        self.team_games[away_id][season]['raw_papg'].append(home_score)

        # MULTIPLICATIVE adjustment
        # If opponent defense allows 120 and league avg is 115:
        # multiplier = 115/120 = 0.958, so we reduce the score
        # This scales based on how far from league average the opponent is
        league_ppg = self.league_avg['ppg']

        # Avoid division by zero, clamp to reasonable range
        home_opp_def_safe = max(min(home_opp_def, 130), 100)
        home_opp_off_safe = max(min(home_opp_off, 130), 100)
        away_opp_def_safe = max(min(away_opp_def, 130), 100)
        away_opp_off_safe = max(min(away_opp_off, 130), 100)

        # Calculate multipliers (how much to adjust)
        # strength=0 means multiplier=1 (no adjustment)
        # strength=1 means full adjustment
        home_off_mult = 1.0 + self.adjustment_strength * (league_ppg / home_opp_def_safe - 1.0)
        home_def_mult = 1.0 + self.adjustment_strength * (league_ppg / home_opp_off_safe - 1.0)
        away_off_mult = 1.0 + self.adjustment_strength * (league_ppg / away_opp_def_safe - 1.0)
        away_def_mult = 1.0 + self.adjustment_strength * (league_ppg / away_opp_off_safe - 1.0)

        # Store adjusted values
        self.team_games[home_id][season]['adj_ppg'].append(home_score * home_off_mult)
        self.team_games[home_id][season]['adj_papg'].append(away_score * home_def_mult)
        self.team_games[home_id][season]['weights'].append(1.0)

        self.team_games[away_id][season]['adj_ppg'].append(away_score * away_off_mult)
        self.team_games[away_id][season]['adj_papg'].append(home_score * away_def_mult)
        self.team_games[away_id][season]['weights'].append(1.0)

    def set_prev_season_ratings(self, season: int):
        """Calculate and store previous season final ratings."""
        prev_season = season - 1
        for team_id in self.team_games:
            if prev_season in self.team_games[team_id]:
                adj_ppg = self.team_games[team_id][prev_season]['adj_ppg']
                adj_papg = self.team_games[team_id][prev_season]['adj_papg']
                raw_ppg = self.team_games[team_id][prev_season]['raw_ppg']
                raw_papg = self.team_games[team_id][prev_season]['raw_papg']

                if adj_ppg:
                    self.prev_season_ratings[team_id] = {
                        'off': float(np.mean(adj_ppg)),
                        'def': float(np.mean(adj_papg)),
                        'raw_off': float(np.mean(raw_ppg)),
                        'raw_def': float(np.mean(raw_papg))
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
        SELECT game_id, latest_spread as vegas_spread, latest_total as vegas_total
        FROM odds_and_predictions
        WHERE latest_spread IS NOT NULL
    ''', conn)
    conn.close()
    return odds


def run_backtest(games: pd.DataFrame, adjustment_strength: float = 1.0) -> pd.DataFrame:
    model = OpponentAdjustedModel(
        decay=0.93,
        prev_season_half_life=6.0,
        hca=2.0,
        adjustment_strength=adjustment_strength
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
                'season': season,
                'pred_home': pred_home,
                'pred_away': pred_away,
                'pred_total': pred_home + pred_away,
                'pred_spread': pred_away - pred_home,
                'actual_home': game.home_score,
                'actual_away': game.away_score,
                'actual_total': game.total,
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


def evaluate(predictions: pd.DataFrame, name: str = "Model") -> dict:
    df = predictions.dropna()

    spread_mae = (df['pred_spread'] - df['actual_spread']).abs().mean()
    total_mae = (df['pred_total'] - df['actual_total']).abs().mean()

    df['pred_home_win'] = df['pred_spread'] < 0
    df['actual_home_win'] = df['actual_spread'] < 0
    winner_acc = (df['pred_home_win'] == df['actual_home_win']).mean()

    return {
        'name': name,
        'games': len(df),
        'winner_acc': winner_acc,
        'spread_mae': spread_mae,
        'total_mae': total_mae
    }


def main():
    print("=" * 60)
    print("OPPONENT-ADJUSTED POWER RATINGS")
    print("=" * 60)

    games = load_games()
    vegas = load_vegas()
    print(f"\nLoaded {len(games)} games, {len(vegas)} with Vegas odds")

    # Test different adjustment strengths
    print("\n" + "=" * 60)
    print("TESTING ADJUSTMENT STRENGTH")
    print("=" * 60)
    print(f"\n{'Strength':<12} {'Winner Acc':<12} {'Spread MAE':<12} {'Total MAE'}")
    print("-" * 50)

    results = []
    for strength in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        preds = run_backtest(games, adjustment_strength=strength)
        metrics = evaluate(preds, f"adj={strength}")
        results.append(metrics)
        print(f"{strength:<12.2f} {metrics['winner_acc']*100:<12.1f} {metrics['spread_mae']:<12.2f} {metrics['total_mae']:.2f}")

    # Find best
    best = max(results, key=lambda x: x['winner_acc'])
    print(f"\nBest: adjustment_strength={best['name'].split('=')[1]} at {best['winner_acc']*100:.1f}%")

    # Compare best with Vegas
    print("\n" + "=" * 60)
    print("COMPARISON WITH VEGAS")
    print("=" * 60)

    best_strength = float(best['name'].split('=')[1])
    preds = run_backtest(games, adjustment_strength=best_strength)
    merged = preds.merge(vegas, on='game_id', how='inner')

    # Model metrics
    model_spread_mae = (merged['pred_spread'] - merged['actual_spread']).abs().mean()
    merged['model_home_win'] = merged['pred_spread'] < 0
    merged['actual_home_win'] = merged['actual_spread'] < 0
    model_winner_acc = (merged['model_home_win'] == merged['actual_home_win']).mean()

    # Vegas metrics
    vegas_spread_mae = (merged['vegas_spread'] - merged['actual_spread']).abs().mean()
    merged['vegas_home_win'] = merged['vegas_spread'] < 0
    vegas_winner_acc = (merged['vegas_home_win'] == merged['actual_home_win']).mean()

    print(f"\n{'Metric':<20} {'Model':<12} {'Vegas':<12} {'Diff'}")
    print("-" * 50)
    print(f"{'Winner Accuracy':<20} {model_winner_acc*100:<11.1f}% {vegas_winner_acc*100:<11.1f}% {(model_winner_acc-vegas_winner_acc)*100:+.1f}%")
    print(f"{'Spread MAE':<20} {model_spread_mae:<12.2f} {vegas_spread_mae:<12.2f} {model_spread_mae-vegas_spread_mae:+.2f}")

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print("Baseline (no adjustment):     65.0%")
    print(f"Opponent-adjusted (best):     {best['winner_acc']*100:.1f}%")
    print(f"Vegas:                        {vegas_winner_acc*100:.1f}%")


if __name__ == '__main__':
    main()
