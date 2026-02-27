"""
NBA Power Rating Model - Baseline Version

Simple model based on PPG/PAPG with:
- Exponentially weighted rolling averages
- Previous season blending
- Home court advantage

Usage:
    python nba_power_ratings.py
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


class PowerRatingModel:
    """
    Simple power rating model based on PPG/PAPG

    Parameters:
    - decay: Weight decay for older games (0.93 = 7% decay per game)
    - min_games: Minimum games before trusting current season
    - prev_season_weight: Initial weight for previous season (1.0 = 100%)
    - prev_season_half_life: Games until previous season weight halves (6 games)
    - hca: Home court advantage in points (2.0)

    Optimal parameters found via backtesting:
    - decay=0.93, prev_season_half_life=6, hca=2.0
    - Winner accuracy: 65.0% (vs Vegas 66.4%)
    """

    def __init__(
        self,
        decay: float = 0.93,
        min_games: int = 5,
        prev_season_weight: float = 1.0,  # Start at 100% for game 0
        prev_season_half_life: float = 6.0,  # Games until prev season weight halves
        hca: float = 2.0
    ):
        self.decay = decay
        self.min_games = min_games
        self.prev_season_weight = prev_season_weight
        self.prev_season_half_life = prev_season_half_life
        self.hca = hca

        # Team games: {team_id: {season: {'ppg': [], 'papg': [], 'weights': []}}}
        self.team_games: dict = defaultdict(lambda: defaultdict(lambda: {'ppg': [], 'papg': [], 'weights': []}))

        # Previous season final ratings
        self.prev_season_ratings: dict = {}

        # League averages
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}

    def _get_weighted_avg(self, values: list, weights: list) -> float | None:
        """Calculate weighted average"""
        if not values:
            return None
        weights_arr = np.array(weights)
        values_arr = np.array(values)
        return float(np.sum(values_arr * weights_arr) / np.sum(weights_arr))

    def _get_team_rating(self, team_id: int, season: int, games_played: int) -> tuple[float, float]:
        """Get team's current offensive and defensive rating"""

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

        # Blend: previous season weight decays exponentially with games played
        # blend = prev_weight * 0.5^(games / half_life)
        # Game 0: 100%, Game 6: 50%, Game 12: 25%, etc.
        blend = self.prev_season_weight * (0.5 ** (games_played / self.prev_season_half_life))

        off_rating = blend * prev_off + (1 - blend) * curr_ppg
        def_rating = blend * prev_def + (1 - blend) * curr_papg

        return off_rating, def_rating

    def _get_games_played(self, team_id: int, season: int) -> int:
        """Get number of games played by team in season"""
        return len(self.team_games[team_id][season]['ppg'])

    def get_ratings(self, team_id: int, season: int) -> dict:
        """Get current ratings for a team"""
        games_played = self._get_games_played(team_id, season)
        off, def_ = self._get_team_rating(team_id, season, games_played)
        return {
            'team_id': team_id,
            'season': season,
            'games_played': games_played,
            'off_rating': off,
            'def_rating': def_,
            'net_rating': off - def_
        }

    def predict(self, home_id: int, away_id: int, season: int) -> tuple[float, float]:
        """
        Predict game score

        Returns: (predicted_home_score, predicted_away_score)
        """
        home_games = self._get_games_played(home_id, season)
        away_games = self._get_games_played(away_id, season)

        home_off, home_def = self._get_team_rating(home_id, season, home_games)
        away_off, away_def = self._get_team_rating(away_id, season, away_games)

        # Prediction formula:
        # Home score = (home_offense + away_defense) / 2 + HCA/2
        # Away score = (away_offense + home_defense) / 2 - HCA/2
        pred_home = (home_off + away_def) / 2 + self.hca / 2
        pred_away = (away_off + home_def) / 2 - self.hca / 2

        return pred_home, pred_away

    def update(self, home_id: int, away_id: int, home_score: int, away_score: int, season: int):
        """Update ratings after a game"""
        # Decay existing weights for both teams
        for team_id in [home_id, away_id]:
            team_data = self.team_games[team_id][season]
            team_data['weights'] = [w * self.decay for w in team_data['weights']]

        # Add new game data with weight 1.0
        self.team_games[home_id][season]['ppg'].append(home_score)
        self.team_games[home_id][season]['papg'].append(away_score)
        self.team_games[home_id][season]['weights'].append(1.0)

        self.team_games[away_id][season]['ppg'].append(away_score)
        self.team_games[away_id][season]['papg'].append(home_score)
        self.team_games[away_id][season]['weights'].append(1.0)

    def set_prev_season_ratings(self, season: int):
        """Calculate and store previous season final ratings"""
        prev_season = season - 1
        for team_id in self.team_games:
            if prev_season in self.team_games[team_id]:
                ppg_list = self.team_games[team_id][prev_season]['ppg']
                papg_list = self.team_games[team_id][prev_season]['papg']
                if ppg_list:
                    # Use simple average for end-of-season rating
                    self.prev_season_ratings[team_id] = {
                        'off': float(np.mean(ppg_list)),
                        'def': float(np.mean(papg_list))
                    }

    def set_league_avg(self, ppg: float, papg: float):
        """Set league average for baseline"""
        self.league_avg = {'ppg': ppg, 'papg': papg}


def load_games(db_path: Path) -> pd.DataFrame:
    """Load all completed games from database"""
    conn = sqlite3.connect(str(db_path))
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


def load_vegas_odds(db_path: Path) -> pd.DataFrame:
    """Load Vegas odds for comparison"""
    conn = sqlite3.connect(str(db_path))
    odds = pd.read_sql_query('''
        SELECT game_id, latest_spread as vegas_spread, latest_total as vegas_total
        FROM odds_and_predictions
        WHERE latest_spread IS NOT NULL
    ''', conn)
    conn.close()
    return odds


def evaluate_model(predictions: pd.DataFrame, name: str = "Model"):
    """Evaluate prediction accuracy"""
    df = predictions.dropna()

    # Total prediction metrics
    total_mae = (df['pred_total'] - df['actual_total']).abs().mean()
    total_rmse = np.sqrt(((df['pred_total'] - df['actual_total'])**2).mean())
    total_corr = df['pred_total'].corr(df['actual_total'])

    # Spread prediction metrics
    spread_mae = (df['pred_spread'] - df['actual_spread']).abs().mean()
    spread_rmse = np.sqrt(((df['pred_spread'] - df['actual_spread'])**2).mean())
    spread_corr = df['pred_spread'].corr(df['actual_spread'])

    # Winner accuracy
    df['pred_home_win'] = df['pred_spread'] < 0
    df['actual_home_win'] = df['actual_spread'] < 0
    winner_acc = (df['pred_home_win'] == df['actual_home_win']).mean()

    logger.info(f"\n{'='*60}")
    logger.info(f"{name} EVALUATION ({len(df)} games)")
    logger.info(f"{'='*60}")
    logger.info(f"\nTOTAL PREDICTION:")
    logger.info(f"  MAE:         {total_mae:.2f} points")
    logger.info(f"  RMSE:        {total_rmse:.2f} points")
    logger.info(f"  Correlation: {total_corr:.3f}")
    logger.info(f"\nSPREAD PREDICTION:")
    logger.info(f"  MAE:         {spread_mae:.2f} points")
    logger.info(f"  RMSE:        {spread_rmse:.2f} points")
    logger.info(f"  Correlation: {spread_corr:.3f}")
    logger.info(f"\nWINNER PREDICTION:")
    logger.info(f"  Accuracy:    {winner_acc*100:.1f}%")

    return {
        'total_mae': total_mae,
        'total_rmse': total_rmse,
        'total_corr': total_corr,
        'spread_mae': spread_mae,
        'spread_rmse': spread_rmse,
        'spread_corr': spread_corr,
        'winner_acc': winner_acc
    }


def run_backtest(
    games: pd.DataFrame,
    decay: float = 0.93,
    prev_season_weight: float = 1.0,
    prev_season_half_life: float = 6.0,
    hca: float = 2.0
) -> pd.DataFrame:
    """Run backtest with given parameters"""

    model = PowerRatingModel(
        decay=decay,
        prev_season_weight=prev_season_weight,
        prev_season_half_life=prev_season_half_life,
        hca=hca
    )

    predictions = []
    seasons = sorted(games.season.unique())

    for season in seasons:
        # Set previous season ratings
        if season > seasons[0]:
            model.set_prev_season_ratings(season)

            # Update league average from previous season
            prev_games = games[games.season == season - 1]
            if len(prev_games) > 0:
                model.set_league_avg(
                    ppg=prev_games['home_score'].mean(),
                    papg=prev_games['away_score'].mean()
                )

        season_games = games[games.season == season]

        for _, game in season_games.iterrows():
            # Make prediction BEFORE updating
            pred_home, pred_away = model.predict(
                game.home_team_id,
                game.away_team_id,
                season
            )

            predictions.append({
                'game_id': game.game_id,
                'season': season,
                'date': game.game_date_eastern,
                'pred_home': pred_home,
                'pred_away': pred_away,
                'pred_total': pred_home + pred_away,
                'pred_spread': pred_away - pred_home,
                'actual_home': game.home_score,
                'actual_away': game.away_score,
                'actual_total': game.total,
                'actual_spread': game.spread
            })

            # Update model with actual result
            model.update(
                game.home_team_id,
                game.away_team_id,
                game.home_score,
                game.away_score,
                season
            )

    return pd.DataFrame(predictions)


def compare_with_vegas(predictions: pd.DataFrame, vegas: pd.DataFrame):
    """Compare model predictions with Vegas lines"""
    merged = predictions.merge(vegas, on='game_id', how='inner')

    if len(merged) < 100:
        logger.info(f"\nInsufficient Vegas data for comparison ({len(merged)} games)")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARISON WITH VEGAS ({len(merged)} games)")
    logger.info(f"{'='*60}")

    # Total prediction
    model_total_mae = (merged['pred_total'] - merged['actual_total']).abs().mean()
    vegas_total_mae = (merged['vegas_total'] - merged['actual_total']).abs().mean()

    model_total_corr = merged['pred_total'].corr(merged['actual_total'])
    vegas_total_corr = merged['vegas_total'].corr(merged['actual_total'])

    logger.info(f"\nTOTAL PREDICTION:")
    logger.info(f"  {'Metric':<15} {'Model':<12} {'Vegas':<12} {'Diff'}")
    logger.info(f"  {'-'*45}")
    logger.info(f"  {'MAE':<15} {model_total_mae:<12.2f} {vegas_total_mae:<12.2f} {model_total_mae - vegas_total_mae:+.2f}")
    logger.info(f"  {'Correlation':<15} {model_total_corr:<12.3f} {vegas_total_corr:<12.3f} {model_total_corr - vegas_total_corr:+.3f}")

    # Spread prediction
    model_spread_mae = (merged['pred_spread'] - merged['actual_spread']).abs().mean()
    vegas_spread_mae = (merged['vegas_spread'] - merged['actual_spread']).abs().mean()

    model_spread_corr = merged['pred_spread'].corr(merged['actual_spread'])
    vegas_spread_corr = merged['vegas_spread'].corr(merged['actual_spread'])

    logger.info(f"\nSPREAD PREDICTION:")
    logger.info(f"  {'Metric':<15} {'Model':<12} {'Vegas':<12} {'Diff'}")
    logger.info(f"  {'-'*45}")
    logger.info(f"  {'MAE':<15} {model_spread_mae:<12.2f} {vegas_spread_mae:<12.2f} {model_spread_mae - vegas_spread_mae:+.2f}")
    logger.info(f"  {'Correlation':<15} {model_spread_corr:<12.3f} {vegas_spread_corr:<12.3f} {model_spread_corr - vegas_spread_corr:+.3f}")

    # Winner accuracy
    merged['model_home_win'] = merged['pred_spread'] < 0
    merged['vegas_home_win'] = merged['vegas_spread'] < 0
    merged['actual_home_win'] = merged['actual_spread'] < 0

    model_winner_acc = (merged['model_home_win'] == merged['actual_home_win']).mean()
    vegas_winner_acc = (merged['vegas_home_win'] == merged['actual_home_win']).mean()

    logger.info(f"\nWINNER PREDICTION:")
    logger.info(f"  {'Metric':<15} {'Model':<12} {'Vegas':<12} {'Diff'}")
    logger.info(f"  {'-'*45}")
    logger.info(f"  {'Accuracy':<15} {model_winner_acc*100:<11.1f}% {vegas_winner_acc*100:<11.1f}% {(model_winner_acc - vegas_winner_acc)*100:+.1f}%")


def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("NBA POWER RATING MODEL - BASELINE")
    logger.info("="*60)

    # Load data
    logger.info("\nLoading data...")
    games = load_games(DB_PATH)
    vegas = load_vegas_odds(DB_PATH)
    logger.info(f"Loaded {len(games)} games, {len(vegas)} with Vegas odds")

    # Run backtest with optimized parameters
    logger.info("\nRunning backtest with optimized parameters...")
    logger.info("  decay=0.93, prev_season_weight=1.0, prev_season_half_life=6, hca=2.0")
    predictions = run_backtest(
        games,
        decay=0.93,
        prev_season_weight=1.0,
        prev_season_half_life=6.0,
        hca=2.0
    )

    # Evaluate
    evaluate_model(predictions, "OPTIMIZED MODEL")

    # Compare with Vegas
    compare_with_vegas(predictions, vegas)

    # Evaluate by season
    logger.info(f"\n{'='*60}")
    logger.info("BY SEASON")
    logger.info(f"{'='*60}")

    for season in sorted(predictions.season.unique()):
        season_preds = predictions[predictions.season == season]
        metrics = evaluate_model(season_preds, f"Season {season}")

    return predictions


if __name__ == '__main__':
    predictions = main()
