"""
NBA Simple Rating System (SRS) Style Model

Instead of adjusting individual games, we:
1. Compute raw PPG/PAPG ratings
2. Calculate Strength of Schedule (avg opponent rating)
3. Adjust rating = raw rating + SOS adjustment

SRS formula: Rating = Avg Margin + Avg(Opponent Ratings)
Solved iteratively until convergence.
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


class SRSModel:
    """
    Simple Rating System model.

    Each team's rating = average margin + average opponent rating
    Solved iteratively until ratings converge.
    """

    def __init__(
        self,
        decay: float = 0.93,
        prev_season_half_life: float = 6.0,
        hca: float = 2.0,
        srs_iterations: int = 10,
        srs_weight: float = 1.0  # How much to weight SRS vs raw
    ):
        self.decay = decay
        self.prev_season_half_life = prev_season_half_life
        self.hca = hca
        self.srs_iterations = srs_iterations
        self.srs_weight = srs_weight

        # Store game results for SRS calculation
        # {season: [(home_id, away_id, home_margin, weight), ...]}
        self.game_results: dict = defaultdict(list)

        # Raw team stats (for blending with previous season)
        self.team_stats: dict = defaultdict(lambda: defaultdict(lambda: {
            'margins': [], 'weights': [], 'opponents': []
        }))

        self.prev_season_ratings: dict = {}
        self.league_avg_margin = 0.0

    def _compute_srs_ratings(self, season: int) -> dict[int, float]:
        """Compute SRS ratings for all teams in a season."""
        games = self.game_results[season]
        if not games:
            return {}

        # Get all teams
        teams = set()
        for home_id, away_id, _, _ in games:
            teams.add(home_id)
            teams.add(away_id)

        # Initialize ratings to 0
        ratings = {team: 0.0 for team in teams}

        # Compute weighted average margin for each team
        team_margins = defaultdict(lambda: {'sum': 0.0, 'weight': 0.0, 'opponents': []})

        for home_id, away_id, home_margin, weight in games:
            # Home team margin (positive = home win)
            team_margins[home_id]['sum'] += (home_margin - self.hca) * weight  # Remove HCA
            team_margins[home_id]['weight'] += weight
            team_margins[home_id]['opponents'].append((away_id, weight))

            # Away team margin (negative of home margin, add back HCA)
            team_margins[away_id]['sum'] += (-home_margin + self.hca) * weight
            team_margins[away_id]['weight'] += weight
            team_margins[away_id]['opponents'].append((home_id, weight))

        # Calculate average margins
        avg_margins = {}
        for team in teams:
            if team_margins[team]['weight'] > 0:
                avg_margins[team] = team_margins[team]['sum'] / team_margins[team]['weight']
            else:
                avg_margins[team] = 0.0

        # Iterative SRS: rating = avg_margin + avg(opponent_ratings)
        for _ in range(self.srs_iterations):
            new_ratings = {}
            for team in teams:
                if team_margins[team]['weight'] > 0:
                    # Weighted average of opponent ratings
                    opp_sum = 0.0
                    opp_weight = 0.0
                    for opp_id, w in team_margins[team]['opponents']:
                        opp_sum += ratings.get(opp_id, 0.0) * w
                        opp_weight += w

                    avg_opp_rating = opp_sum / opp_weight if opp_weight > 0 else 0.0
                    new_ratings[team] = avg_margins[team] + avg_opp_rating
                else:
                    new_ratings[team] = 0.0

            # Normalize to zero mean
            mean_rating = np.mean(list(new_ratings.values()))
            ratings = {t: r - mean_rating for t, r in new_ratings.items()}

        return ratings

    def _get_team_rating(self, team_id: int, season: int) -> float:
        """Get team's current SRS rating."""
        # Compute current season SRS
        srs_ratings = self._compute_srs_ratings(season)
        curr_rating = srs_ratings.get(team_id, 0.0)

        # Get previous season rating
        prev_rating = self.prev_season_ratings.get(team_id, 0.0)

        # Blend based on games played
        games_played = len(self.team_stats[team_id][season]['margins'])

        if games_played == 0:
            return prev_rating

        blend = 1.0 * (0.5 ** (games_played / self.prev_season_half_life))
        return blend * prev_rating + (1 - blend) * curr_rating

    def predict(self, home_id: int, away_id: int, season: int) -> tuple[float, float]:
        """Predict game score."""
        home_rating = self._get_team_rating(home_id, season)
        away_rating = self._get_team_rating(away_id, season)

        # SRS predicts margin, not score
        # Expected margin = home_rating - away_rating + HCA
        expected_margin = home_rating - away_rating + self.hca

        # Convert to scores using league average
        league_avg = 115.0  # Approximate
        pred_home = league_avg + expected_margin / 2
        pred_away = league_avg - expected_margin / 2

        return pred_home, pred_away

    def update(self, home_id: int, away_id: int, home_score: int, away_score: int, season: int):
        """Record game result."""
        home_margin = home_score - away_score

        # Decay old game weights
        self.game_results[season] = [
            (h, a, m, w * self.decay) for h, a, m, w in self.game_results[season]
        ]

        # Add new game with weight 1.0
        self.game_results[season].append((home_id, away_id, home_margin, 1.0))

        # Track for games played count
        self.team_stats[home_id][season]['margins'].append(home_margin)
        self.team_stats[away_id][season]['margins'].append(-home_margin)

    def set_prev_season_ratings(self, season: int):
        """Store previous season final ratings."""
        prev_ratings = self._compute_srs_ratings(season - 1)
        self.prev_season_ratings = prev_ratings


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


def run_backtest(games: pd.DataFrame, srs_weight: float = 1.0) -> pd.DataFrame:
    model = SRSModel(
        decay=0.93,
        prev_season_half_life=6.0,
        hca=2.0,
        srs_iterations=10,
        srs_weight=srs_weight
    )

    predictions = []
    seasons = sorted(games.season.unique())

    for season in seasons:
        if season > seasons[0]:
            model.set_prev_season_ratings(season)

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
                'pred_total': pred_home + pred_away,
                'pred_spread': pred_away - pred_home,
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


def main():
    print("=" * 60)
    print("SRS-STYLE POWER RATINGS")
    print("=" * 60)

    games = load_games()
    vegas = load_vegas()
    print(f"\nLoaded {len(games)} games, {len(vegas)} with Vegas odds")

    # Run SRS model
    print("\nRunning SRS model...")
    preds = run_backtest(games)
    df = preds.dropna()

    # Metrics
    spread_mae = (df['pred_spread'] - df['actual_spread']).abs().mean()
    df['pred_home_win'] = df['pred_spread'] < 0
    df['actual_home_win'] = df['actual_spread'] < 0
    winner_acc = (df['pred_home_win'] == df['actual_home_win']).mean()

    print(f"\nSRS Model Results ({len(df)} games):")
    print(f"  Winner Accuracy: {winner_acc*100:.1f}%")
    print(f"  Spread MAE:      {spread_mae:.2f}")

    # Compare with Vegas
    merged = preds.merge(vegas, on='game_id', how='inner')
    merged['model_home_win'] = merged['pred_spread'] < 0
    merged['vegas_home_win'] = merged['vegas_spread'] < 0
    merged['actual_home_win'] = merged['actual_spread'] < 0

    model_acc = (merged['model_home_win'] == merged['actual_home_win']).mean()
    vegas_acc = (merged['vegas_home_win'] == merged['actual_home_win']).mean()

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'Winner Acc'}")
    print("-" * 40)
    print(f"{'Baseline (PPG/PAPG)':<25} 65.0%")
    print(f"{'SRS Model':<25} {winner_acc*100:.1f}%")
    print(f"{'Vegas':<25} {vegas_acc*100:.1f}%")


if __name__ == '__main__':
    main()
