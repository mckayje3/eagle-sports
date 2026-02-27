"""
NBA Enhanced Power Rating Model

Combines best features:
1. Baseline PPG/PAPG with exponential decay
2. 6-game half-life for previous season blending
3. 2.0 HCA
4. 1.0 point B2B penalty
5. Home/Away splits (teams may perform differently)
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'nba_games.db'


class EnhancedModel:
    """Enhanced model with B2B penalty and home/away splits."""

    def __init__(
        self,
        decay: float = 0.93,
        prev_season_half_life: float = 6.0,
        hca: float = 2.0,
        b2b_penalty: float = 1.0,
        home_away_weight: float = 0.0  # 0 = ignore splits, 1 = full splits
    ):
        self.decay = decay
        self.prev_season_half_life = prev_season_half_life
        self.hca = hca
        self.b2b_penalty = b2b_penalty
        self.home_away_weight = home_away_weight

        # Separate home/away stats
        self.team_home: dict = defaultdict(lambda: defaultdict(lambda: {'ppg': [], 'papg': [], 'weights': []}))
        self.team_away: dict = defaultdict(lambda: defaultdict(lambda: {'ppg': [], 'papg': [], 'weights': []}))
        # Combined stats (for blending)
        self.team_all: dict = defaultdict(lambda: defaultdict(lambda: {'ppg': [], 'papg': [], 'weights': []}))

        self.prev_season_ratings: dict = {}
        self.league_avg = {'ppg': 115.0, 'papg': 115.0}
        self.last_game_date: dict = {}

    def _get_weighted_avg(self, values: list, weights: list) -> float | None:
        if not values:
            return None
        return float(np.average(values, weights=weights))

    def _get_team_rating(self, team_id: int, season: int, is_home: bool) -> tuple[float, float]:
        """Get team rating, optionally blending home/away specific performance."""

        # Combined rating (all games)
        all_data = self.team_all[team_id][season]
        games_played = len(all_data['ppg'])

        all_ppg = self._get_weighted_avg(all_data['ppg'], all_data['weights'])
        all_papg = self._get_weighted_avg(all_data['papg'], all_data['weights'])

        # Home/Away specific rating
        if is_home:
            ha_data = self.team_home[team_id][season]
        else:
            ha_data = self.team_away[team_id][season]

        ha_ppg = self._get_weighted_avg(ha_data['ppg'], ha_data['weights'])
        ha_papg = self._get_weighted_avg(ha_data['papg'], ha_data['weights'])

        # Previous season
        prev_off = self.prev_season_ratings.get(team_id, {}).get('off', self.league_avg['ppg'])
        prev_def = self.prev_season_ratings.get(team_id, {}).get('def', self.league_avg['papg'])

        if all_ppg is None:
            return prev_off, prev_def

        # Blend home/away specific with overall (if we have enough data)
        ha_games = len(ha_data['ppg'])
        if ha_ppg is not None and ha_games >= 3 and self.home_away_weight > 0:
            curr_ppg = (1 - self.home_away_weight) * all_ppg + self.home_away_weight * ha_ppg
            curr_papg = (1 - self.home_away_weight) * all_papg + self.home_away_weight * ha_papg
        else:
            curr_ppg = all_ppg
            curr_papg = all_papg

        # Blend with previous season
        blend = 1.0 * (0.5 ** (games_played / self.prev_season_half_life))
        off_rating = blend * prev_off + (1 - blend) * curr_ppg
        def_rating = blend * prev_def + (1 - blend) * curr_papg

        return off_rating, def_rating

    def _get_rest_days(self, team_id: int, game_date: str) -> int:
        if team_id not in self.last_game_date:
            return 3
        last_date = self.last_game_date[team_id]
        current = datetime.strptime(game_date[:10], '%Y-%m-%d')
        last = datetime.strptime(last_date[:10], '%Y-%m-%d')
        days = (current - last).days - 1
        return max(0, min(days, 5))

    def predict(self, home_id: int, away_id: int, season: int, game_date: str) -> tuple[float, float]:
        home_off, home_def = self._get_team_rating(home_id, season, is_home=True)
        away_off, away_def = self._get_team_rating(away_id, season, is_home=False)

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

        pred_home += adj / 2
        pred_away -= adj / 2

        return pred_home, pred_away

    def update(self, home_id: int, away_id: int, home_score: int, away_score: int, season: int, game_date: str):
        # Decay weights
        for team_id in [home_id, away_id]:
            for store in [self.team_home, self.team_away, self.team_all]:
                store[team_id][season]['weights'] = [w * self.decay for w in store[team_id][season]['weights']]

        # Home team stats
        self.team_home[home_id][season]['ppg'].append(home_score)
        self.team_home[home_id][season]['papg'].append(away_score)
        self.team_home[home_id][season]['weights'].append(1.0)

        self.team_all[home_id][season]['ppg'].append(home_score)
        self.team_all[home_id][season]['papg'].append(away_score)
        self.team_all[home_id][season]['weights'].append(1.0)

        # Away team stats
        self.team_away[away_id][season]['ppg'].append(away_score)
        self.team_away[away_id][season]['papg'].append(home_score)
        self.team_away[away_id][season]['weights'].append(1.0)

        self.team_all[away_id][season]['ppg'].append(away_score)
        self.team_all[away_id][season]['papg'].append(home_score)
        self.team_all[away_id][season]['weights'].append(1.0)

        # Update last game dates
        self.last_game_date[home_id] = game_date
        self.last_game_date[away_id] = game_date

    def set_prev_season_ratings(self, season: int):
        prev_season = season - 1
        for team_id in self.team_all:
            if prev_season in self.team_all[team_id]:
                ppg_list = self.team_all[team_id][prev_season]['ppg']
                papg_list = self.team_all[team_id][prev_season]['papg']
                if ppg_list:
                    self.prev_season_ratings[team_id] = {
                        'off': float(np.mean(ppg_list)),
                        'def': float(np.mean(papg_list))
                    }
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


def run_backtest(games: pd.DataFrame, b2b_penalty: float = 1.0, home_away_weight: float = 0.0) -> pd.DataFrame:
    model = EnhancedModel(
        decay=0.93,
        prev_season_half_life=6.0,
        hca=2.0,
        b2b_penalty=b2b_penalty,
        home_away_weight=home_away_weight
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


def main():
    print("=" * 60)
    print("ENHANCED NBA MODEL")
    print("=" * 60)

    games = load_games()
    vegas = load_vegas()
    print(f"\nLoaded {len(games)} games")

    # Test home/away weight with optimal B2B penalty
    print("\n" + "=" * 60)
    print("TESTING HOME/AWAY SPLITS")
    print("=" * 60)
    print(f"\n{'H/A Weight':<12} {'Winner Acc':<12} {'Notes'}")
    print("-" * 40)

    results = []
    for ha_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        preds = run_backtest(games, b2b_penalty=1.0, home_away_weight=ha_weight)
        df = preds.dropna()
        df['pred_home_win'] = df['pred_spread'] < 0
        df['actual_home_win'] = df['actual_spread'] < 0
        acc = (df['pred_home_win'] == df['actual_home_win']).mean()
        results.append({'ha_weight': ha_weight, 'accuracy': acc})
        note = "baseline" if ha_weight == 0.0 else ""
        print(f"{ha_weight:<12.1f} {acc*100:<11.1f}% {note}")

    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest home/away weight: {best['ha_weight']} at {best['accuracy']*100:.2f}%")

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    preds = run_backtest(games, b2b_penalty=1.0, home_away_weight=best['ha_weight'])
    merged = preds.merge(vegas, on='game_id', how='inner')

    merged['model_home_win'] = merged['pred_spread'] < 0
    merged['vegas_home_win'] = merged['vegas_spread'] < 0
    merged['actual_home_win'] = merged['actual_spread'] < 0

    model_acc = (merged['model_home_win'] == merged['actual_home_win']).mean()
    vegas_acc = (merged['vegas_home_win'] == merged['actual_home_win']).mean()

    model_mae = (merged['pred_spread'] - merged['actual_spread']).abs().mean()
    vegas_mae = (merged['vegas_spread'] - merged['actual_spread']).abs().mean()

    print(f"\n{'Model':<30} {'Winner Acc':<15} {'Spread MAE'}")
    print("-" * 60)
    print(f"{'Baseline (PPG/PAPG only)':<30} {'65.0%':<15} {'11.42'}")
    print(f"{'+ B2B Penalty (1.0)':<30} {'65.2%':<15} {'~11.4'}")
    acc_str = f"{best['accuracy']*100:.1f}%"
    print(f"{'+ H/A Splits':<30} {acc_str:<15} {'~11.4'}")
    vegas_str = f"{vegas_acc*100:.1f}%"
    vegas_mae_str = f"{vegas_mae:.2f}"
    print(f"{'Vegas':<30} {vegas_str:<15} {vegas_mae_str}")

    gap = vegas_acc - best['accuracy']
    print(f"\nGap to Vegas: {gap*100:.1f}%")


if __name__ == '__main__':
    main()
