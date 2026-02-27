"""
Test adding injury features to the NBA Ridge model.

Compares model performance with and without injury features.
"""

import sqlite3
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'nba_games.db'


def get_player_importance_at_date(conn, team_id: int, game_date: str, season: int) -> dict:
    """
    Get player importance scores for a team based on games BEFORE this date.
    Returns dict mapping player_id to importance score.
    """
    query = """
    SELECT
        pgs.player_id,
        AVG(pgs.minutes) as avg_minutes,
        AVG(pgs.points) as avg_points,
        AVG(pgs.plus_minus) as avg_pm,
        AVG(pgs.starter) as start_rate,
        COUNT(*) as games
    FROM player_game_stats pgs
    JOIN games g ON pgs.game_id = g.game_id
    WHERE pgs.team_id = ?
      AND g.season = ?
      AND g.date < ?
      AND pgs.did_not_play = 0
      AND pgs.minutes > 0
    GROUP BY pgs.player_id
    HAVING COUNT(*) >= 5
    """

    df = pd.read_sql_query(query, conn, params=(team_id, season, game_date))

    if df.empty:
        return {}

    # Simple importance: minutes share (primary driver of impact)
    total_minutes = df['avg_minutes'].sum()
    if total_minutes == 0:
        return {}

    importance = {}
    for _, row in df.iterrows():
        # Importance based on minutes and points contribution
        imp = (row['avg_minutes'] / total_minutes) * 0.6 + \
              (row['avg_points'] / df['avg_points'].sum()) * 0.4
        importance[row['player_id']] = imp

    return importance


def get_injury_features_for_game(conn, game_id: int, home_team_id: int, away_team_id: int,
                                  game_date: str, season: int) -> dict:
    """
    Get injury-related features for a specific game.
    """
    # Get player importance scores (based on games before this one)
    home_importance = get_player_importance_at_date(conn, home_team_id, game_date, season)
    away_importance = get_player_importance_at_date(conn, away_team_id, game_date, season)

    # Get DNPs for this game
    query = """
    SELECT player_id, team_id, dnp_reason
    FROM player_game_stats
    WHERE game_id = ?
      AND did_not_play = 1
      AND dnp_reason NOT IN ("COACH'S DECISION", "NOT WITH TEAM", "REST")
    """

    dnps = pd.read_sql_query(query, conn, params=(game_id,))

    # Calculate importance lost due to injuries
    home_importance_lost = 0
    away_importance_lost = 0
    home_injured_count = 0
    away_injured_count = 0

    for _, row in dnps.iterrows():
        player_id = row['player_id']
        team_id = row['team_id']

        if team_id == home_team_id:
            home_importance_lost += home_importance.get(player_id, 0)
            if home_importance.get(player_id, 0) > 0.05:  # Rotation player
                home_injured_count += 1
        elif team_id == away_team_id:
            away_importance_lost += away_importance.get(player_id, 0)
            if away_importance.get(player_id, 0) > 0.05:
                away_injured_count += 1

    return {
        'home_importance_lost': home_importance_lost,
        'away_importance_lost': away_importance_lost,
        'importance_diff': home_importance_lost - away_importance_lost,
        'home_injured_count': home_injured_count,
        'away_injured_count': away_injured_count,
    }


def build_dataset_with_injuries(seasons: list[int]) -> pd.DataFrame:
    """Build training dataset with injury features."""
    conn = sqlite3.connect(DB_PATH)

    # Get all completed games
    query = """
    SELECT
        g.game_id,
        g.season,
        g.date,
        g.home_team_id,
        g.away_team_id,
        g.home_score,
        g.away_score,
        o.latest_spread as vegas_spread,
        o.latest_total as vegas_total
    FROM games g
    LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
    WHERE g.completed = 1
      AND g.season IN ({})
      AND g.home_score IS NOT NULL
    ORDER BY g.date
    """.format(','.join('?' * len(seasons)))

    games = pd.read_sql_query(query, conn, params=seasons)
    log.info(f"Processing {len(games)} games...")

    # Get team stats for baseline features
    team_stats = defaultdict(lambda: {'ppg': [], 'papg': [], 'margins': []})

    results = []

    for idx, row in games.iterrows():
        if idx % 500 == 0:
            log.info(f"  Processing game {idx}/{len(games)}...")

        game_id = row['game_id']
        hid = row['home_team_id']
        aid = row['away_team_id']
        date = row['date']
        season = row['season']

        hs = team_stats[hid]
        aws = team_stats[aid]

        # Need minimum games for baseline
        if len(hs['ppg']) >= 10 and len(aws['ppg']) >= 10:
            # Baseline features (simplified)
            h_ppg = np.mean(hs['ppg'][-20:])
            h_papg = np.mean(hs['papg'][-20:])
            a_ppg = np.mean(aws['ppg'][-20:])
            a_papg = np.mean(aws['papg'][-20:])

            baseline_spread_pred = (a_ppg - a_papg) - (h_ppg - h_papg) - 3  # Simple HCA
            baseline_total_pred = h_ppg + a_ppg + h_papg + a_papg
            baseline_total_pred = baseline_total_pred / 2  # Average of offensive outputs

            # Get injury features
            injury_feats = get_injury_features_for_game(conn, game_id, hid, aid, date, season)

            actual_spread = row['away_score'] - row['home_score']
            actual_total = row['home_score'] + row['away_score']

            results.append({
                'game_id': game_id,
                'season': season,
                'date': date,
                'h_ppg': h_ppg,
                'h_papg': h_papg,
                'a_ppg': a_ppg,
                'a_papg': a_papg,
                'h_recent_margin': np.mean(hs['margins'][-5:]) if len(hs['margins']) >= 5 else 0,
                'a_recent_margin': np.mean(aws['margins'][-5:]) if len(aws['margins']) >= 5 else 0,
                'baseline_spread': baseline_spread_pred,
                'baseline_total': baseline_total_pred,
                **injury_feats,
                'actual_spread': actual_spread,
                'actual_total': actual_total,
                'vegas_spread': row['vegas_spread'],
                'vegas_total': row['vegas_total'],
            })

        # Update team stats after processing
        team_stats[hid]['ppg'].append(row['home_score'])
        team_stats[hid]['papg'].append(row['away_score'])
        team_stats[hid]['margins'].append(row['home_score'] - row['away_score'])
        team_stats[aid]['ppg'].append(row['away_score'])
        team_stats[aid]['papg'].append(row['home_score'])
        team_stats[aid]['margins'].append(row['away_score'] - row['home_score'])

    conn.close()
    return pd.DataFrame(results)


def train_and_evaluate(df: pd.DataFrame, include_injury: bool = False):
    """Train model and evaluate performance."""

    # Base features
    base_features = [
        'h_ppg', 'h_papg', 'a_ppg', 'a_papg',
        'h_recent_margin', 'a_recent_margin',
    ]

    injury_features = [
        'home_importance_lost', 'away_importance_lost', 'importance_diff',
    ]

    features = base_features + (injury_features if include_injury else [])

    # Split by season for walk-forward evaluation
    train_seasons = [2023, 2024, 2025]
    test_season = 2026

    train_df = df[df['season'].isin(train_seasons)].dropna(subset=features + ['actual_spread'])
    test_df = df[df['season'] == test_season].dropna(subset=features + ['actual_spread'])

    if test_df.empty:
        # If no 2026 data, use 2025 as test
        test_season = 2025
        train_seasons = [2023, 2024]
        train_df = df[df['season'].isin(train_seasons)].dropna(subset=features + ['actual_spread'])
        test_df = df[df['season'] == test_season].dropna(subset=features + ['actual_spread'])

    X_train = train_df[features].values
    y_train_spread = train_df['actual_spread'].values
    y_train_total = train_df['actual_total'].values

    X_test = test_df[features].values
    y_test_spread = test_df['actual_spread'].values
    y_test_total = test_df['actual_total'].values
    vegas_spread = test_df['vegas_spread'].values
    vegas_total = test_df['vegas_total'].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    spread_model = Ridge(alpha=1.0)
    spread_model.fit(X_train_scaled, y_train_spread)

    total_model = Ridge(alpha=1.0)
    total_model.fit(X_train_scaled, y_train_total)

    # Predict
    spread_preds = spread_model.predict(X_test_scaled)
    total_preds = total_model.predict(X_test_scaled)

    # Evaluate
    spread_mae = mean_absolute_error(y_test_spread, spread_preds)
    total_mae = mean_absolute_error(y_test_total, total_preds)

    valid_spread_mask = ~np.isnan(vegas_spread)
    valid_total_mask = ~np.isnan(vegas_total)
    vegas_spread_mae = mean_absolute_error(y_test_spread[valid_spread_mask], vegas_spread[valid_spread_mask]) if valid_spread_mask.any() else np.nan
    vegas_total_mae = mean_absolute_error(y_test_total[valid_total_mask], vegas_total[valid_total_mask]) if valid_total_mask.any() else np.nan

    # ATS performance
    model_picks_spread = spread_preds > vegas_spread
    actual_covers = y_test_spread > vegas_spread
    valid_spread = ~np.isnan(vegas_spread)
    spread_ats = np.mean(model_picks_spread[valid_spread] == actual_covers[valid_spread])

    return {
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'vegas_spread_mae': vegas_spread_mae,
        'vegas_total_mae': vegas_total_mae,
        'spread_ats': spread_ats,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'test_season': test_season,
        'features': features,
        'coefficients': dict(zip(features, spread_model.coef_)),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("NBA Injury Features Impact Test")
    print("=" * 70)

    # Build dataset
    print("\nBuilding dataset with injury features...")
    df = build_dataset_with_injuries([2023, 2024, 2025, 2026])
    print(f"Dataset: {len(df)} games")

    # Show injury feature distribution
    print("\nInjury Feature Summary:")
    print(f"  Games with home injuries: {(df['home_importance_lost'] > 0).sum()} ({(df['home_importance_lost'] > 0).mean()*100:.1f}%)")
    print(f"  Games with away injuries: {(df['away_importance_lost'] > 0).sum()} ({(df['away_importance_lost'] > 0).mean()*100:.1f}%)")
    print(f"  Avg home importance lost: {df['home_importance_lost'].mean():.4f}")
    print(f"  Avg away importance lost: {df['away_importance_lost'].mean():.4f}")

    # Train without injury features
    print("\n" + "-" * 70)
    print("Model WITHOUT Injury Features:")
    print("-" * 70)
    results_no_injury = train_and_evaluate(df, include_injury=False)
    print(f"  Test Season: {results_no_injury['test_season']}")
    print(f"  Train: {results_no_injury['n_train']} games, Test: {results_no_injury['n_test']} games")
    print(f"  Spread MAE: {results_no_injury['spread_mae']:.2f}")
    print(f"  Total MAE: {results_no_injury['total_mae']:.2f}")
    print(f"  Spread ATS: {results_no_injury['spread_ats']*100:.1f}%")

    # Train with injury features
    print("\n" + "-" * 70)
    print("Model WITH Injury Features:")
    print("-" * 70)
    results_injury = train_and_evaluate(df, include_injury=True)
    print(f"  Test Season: {results_injury['test_season']}")
    print(f"  Train: {results_injury['n_train']} games, Test: {results_injury['n_test']} games")
    print(f"  Spread MAE: {results_injury['spread_mae']:.2f}")
    print(f"  Total MAE: {results_injury['total_mae']:.2f}")
    print(f"  Spread ATS: {results_injury['spread_ats']*100:.1f}%")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON:")
    print("=" * 70)
    spread_improvement = results_no_injury['spread_mae'] - results_injury['spread_mae']
    total_improvement = results_no_injury['total_mae'] - results_injury['total_mae']
    ats_improvement = results_injury['spread_ats'] - results_no_injury['spread_ats']

    print(f"  Spread MAE improvement: {spread_improvement:+.3f} pts")
    print(f"  Total MAE improvement: {total_improvement:+.3f} pts")
    print(f"  ATS improvement: {ats_improvement*100:+.2f}%")

    if spread_improvement > 0 or ats_improvement > 0:
        print("\n  --> Injury features IMPROVE the model")
    else:
        print("\n  --> Injury features do NOT improve the model (or hurt it)")

    # Show injury feature coefficients
    print("\nInjury Feature Coefficients (spread model):")
    for feat in ['home_importance_lost', 'away_importance_lost', 'importance_diff']:
        if feat in results_injury['coefficients']:
            coef = results_injury['coefficients'][feat]
            print(f"  {feat}: {coef:+.3f}")
