"""
CFB Deep Eagle Edge Analysis: Neural Network vs Vegas

Uses MLP (Multi-Layer Perceptron) neural network to compare against Ridge.
Features are pre-extracted from deep_eagle_feature_extractor.py.

SPREAD CONVENTION: actual_spread = away_score - home_score
- Negative = home team won/favored
- Positive = away team won/favored
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Load pre-extracted features
FEATURE_FILES = [
    'cfb_2024_deep_eagle_features.csv',
    'cfb_2025_deep_eagle_features.csv',
]


def load_features():
    """Load and combine feature files."""
    dfs = []
    for f in FEATURE_FILES:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"Loaded {f}: {len(df)} games")
        except FileNotFoundError:
            print(f"File not found: {f}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    # Remove duplicates based on game_id
    combined = combined.drop_duplicates(subset='game_id', keep='last')
    return combined.sort_values(['season', 'week', 'game_id'])


def run_edge_analysis():
    """Run walk-forward edge analysis with MLP neural network."""
    print("=" * 70)
    print("CFB DEEP EAGLE EDGE ANALYSIS (MLP Neural Network)")
    print("=" * 70)

    df = load_features()
    if df is None:
        print("No feature files found!")
        return

    # Filter to games with Vegas odds
    vegas_col = 'odds_latest_spread' if 'odds_latest_spread' in df.columns else 'vegas_spread'
    df = df[df[vegas_col].notna()].copy()
    print(f"\nTotal games with Vegas lines: {len(df)}")

    # Check for neutral site games
    if 'neutral_site' in df.columns:
        neutral_count = df['neutral_site'].sum()
        print(f"Neutral site games: {neutral_count} ({100*neutral_count/len(df):.1f}%)")

    # Target variable - ALWAYS use away_score - home_score for consistency
    df['actual_spread'] = df['away_score'] - df['home_score']

    # Select feature columns (exclude targets and identifiers)
    exclude_cols = [
        'game_id', 'season', 'week', 'date',
        'home_team_id', 'away_team_id',
        'home_score', 'away_score', 'point_spread', 'total_points', 'home_win',
        'actual_spread',
        # Exclude Vegas odds from features (we compare against them)
        'odds_opening_spread', 'odds_latest_spread', 'odds_opening_total', 'odds_latest_total',
        'vegas_spread', 'vegas_total',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    print(f"Feature columns: {len(feature_cols)}")

    # Walk-forward prediction
    results = []
    min_train_games = 200  # CFB has more games, need more training data

    # Sort by season and week for proper walk-forward
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    for i in range(min_train_games, len(df)):
        train_df = df.iloc[:i]
        test_row = df.iloc[i]

        # Get features
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df['actual_spread'].values
        X_test = test_row[feature_cols].fillna(0).values.reshape(1, -1)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train MLP (retrain every 100 games for efficiency)
        if i == min_train_games or i % 100 == 0:
            mlp = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,  # L2 regularization
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                verbose=False
            )
            mlp.fit(X_train_scaled, y_train)

        # Predict
        pred = mlp.predict(X_test_scaled)[0]
        vegas = test_row[vegas_col]
        actual = test_row['actual_spread']
        neutral = test_row.get('neutral_site', 0)

        results.append({
            'pred': pred,
            'vegas': vegas,
            'actual': actual,
            'season': test_row['season'],
            'week': test_row['week'],
            'neutral_site': neutral,
            'edge': pred - vegas,
        })

    results_df = pd.DataFrame(results)

    print(f"\nGames analyzed: {len(results_df)}")
    print(f"Model MAE: {np.abs(results_df['pred'] - results_df['actual']).mean():.2f}")
    print(f"Vegas MAE: {np.abs(results_df['vegas'] - results_df['actual']).mean():.2f}")

    # ATS analysis by edge size
    print("\n" + "=" * 70)
    print("ATS RECORD BY EDGE SIZE (MLP Neural Network)")
    print("(Betting WITH the model when it disagrees with Vegas)")
    print("=" * 70)
    print(f"{'Edge':<12} {'Games':>8} {'Wins':>8} {'Losses':>8} {'Push':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 70)

    for threshold in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]:
        wins, losses, pushes = 0, 0, 0

        # Positive edge: bet away to cover
        mask_pos = results_df['edge'] >= threshold
        for _, r in results_df[mask_pos].iterrows():
            result = r['actual'] - r['vegas']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1
            else:
                pushes += 1

        # Negative edge: bet home to cover
        mask_neg = results_df['edge'] <= -threshold
        for _, r in results_df[mask_neg].iterrows():
            result = r['vegas'] - r['actual']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1
            else:
                pushes += 1

        total = wins + losses
        if total > 0:
            win_pct = wins / total * 100
            profit = wins * 0.909 - losses
            roi = profit / total * 100

            marker = " <-- PROFITABLE" if win_pct > 52.4 else ""
            print(f">= {threshold:<8} {total + pushes:>8} {wins:>8} {losses:>8} {pushes:>8} "
                  f"{win_pct:>7.1f}% {roi:>+9.1f}%{marker}")

    # Neutral site analysis
    print("\n" + "=" * 70)
    print("NEUTRAL SITE vs REGULAR GAMES (>= 5 pt edge)")
    print("=" * 70)

    for is_neutral in [0, 1]:
        sdf = results_df[results_df['neutral_site'] == is_neutral]
        if len(sdf) == 0:
            continue

        wins, losses = 0, 0

        for _, r in sdf[sdf['edge'] >= 5].iterrows():
            result = r['actual'] - r['vegas']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        for _, r in sdf[sdf['edge'] <= -5].iterrows():
            result = r['vegas'] - r['actual']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        if wins + losses > 0:
            pct = wins / (wins + losses) * 100
            label = "Neutral" if is_neutral else "Regular"
            print(f"{label}: {wins}-{losses} ({pct:.1f}%)")

    # By season
    print("\n" + "=" * 70)
    print("EDGE PERFORMANCE BY SEASON (>= 5 pt edge)")
    print("=" * 70)

    for season in sorted(results_df['season'].unique()):
        sdf = results_df[results_df['season'] == season]
        wins, losses = 0, 0

        for _, r in sdf[sdf['edge'] >= 5].iterrows():
            result = r['actual'] - r['vegas']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        for _, r in sdf[sdf['edge'] <= -5].iterrows():
            result = r['vegas'] - r['actual']
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1

        if wins + losses > 0:
            pct = wins / (wins + losses) * 100
            print(f"Season {int(season)}: {wins}-{losses} ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Break-even at -110 odds: 52.4%")
    print("Look for thresholds where Win% consistently exceeds 52.4%")

    return results_df


if __name__ == '__main__':
    run_edge_analysis()
