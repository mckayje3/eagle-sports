"""
NBA Feature Analysis

Debug why 100 features perform worse than simple PPG/PAPG model.
Identify which features help vs hurt.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FEATURES_PATH = Path('C:/Users/jbeast/Documents/Coding/sports/nba_2025_deep_eagle_features.csv')


def load_and_split_data(train_pct=0.75):
    """Load features and split by time."""
    df = pd.read_csv(FEATURES_PATH)
    df = df.sort_values('date').reset_index(drop=True)

    # Exclude columns
    exclude = ['game_id', 'season', 'date', 'games_into_season', 'home_team_id', 'away_team_id',
               'home_score', 'away_score', 'point_spread', 'total_points', 'home_win']

    feature_cols = [c for c in df.columns if c not in exclude]

    split_idx = int(len(df) * train_pct)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    return train, test, feature_cols


def analyze_correlations(train, test, feature_cols):
    """Analyze feature correlations with targets."""
    print('=' * 70)
    print('FEATURE CORRELATION ANALYSIS')
    print('=' * 70)

    # Calculate correlations with spread (our main target)
    spread = train['point_spread']

    correlations = []
    for col in feature_cols:
        if col in train.columns:
            vals = train[col].fillna(0)
            corr = vals.corr(spread)
            correlations.append((col, corr, abs(corr)))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop 20 features correlated with SPREAD (home - away):")
    print(f"{'Feature':<45} {'Correlation':>12}")
    print('-' * 60)
    for col, corr, _ in correlations[:20]:
        print(f"{col:<45} {corr:>12.3f}")

    print(f"\nBottom 20 (weakest correlation):")
    print(f"{'Feature':<45} {'Correlation':>12}")
    print('-' * 60)
    for col, corr, _ in correlations[-20:]:
        print(f"{col:<45} {corr:>12.3f}")

    # Same for total
    total = train['total_points']
    total_corrs = []
    for col in feature_cols:
        if col in train.columns:
            vals = train[col].fillna(0)
            corr = vals.corr(total)
            total_corrs.append((col, corr, abs(corr)))

    total_corrs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n\nTop 20 features correlated with TOTAL:")
    print(f"{'Feature':<45} {'Correlation':>12}")
    print('-' * 60)
    for col, corr, _ in total_corrs[:20]:
        print(f"{col:<45} {corr:>12.3f}")

    return correlations, total_corrs


def test_feature_subsets(train, test, feature_cols):
    """Test model performance with different feature subsets."""
    print('\n' + '=' * 70)
    print('FEATURE SUBSET ANALYSIS')
    print('=' * 70)

    # Prepare data
    X_train_full = train[feature_cols].fillna(0).values
    X_test_full = test[feature_cols].fillna(0).values
    y_train_spread = train['point_spread'].values
    y_test_spread = test['point_spread'].values
    y_train_total = train['total_points'].values
    y_test_total = test['total_points'].values

    # Actual outcomes
    actual_home_win = (test['point_spread'] > 0).values  # point_spread = home - away, so >0 = home win

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)

    results = []

    # Test 1: All features with Ridge regression
    print("\n--- Ridge Regression (all features) ---")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train_spread)
    pred_spread = ridge.predict(X_test_scaled)
    pred_home_win = pred_spread > 0
    acc = (pred_home_win == actual_home_win).mean()
    mae = np.abs(pred_spread - y_test_spread).mean()
    print(f"Winner Acc: {acc*100:.1f}%, Spread MAE: {mae:.2f}")
    results.append(('Ridge (all 100)', acc, mae))

    # Test 2: Lasso to identify important features
    print("\n--- Lasso Feature Selection ---")
    lasso = Lasso(alpha=0.5, max_iter=5000)
    lasso.fit(X_train_scaled, y_train_spread)

    # Get non-zero coefficients
    nonzero_idx = np.where(np.abs(lasso.coef_) > 0.01)[0]
    nonzero_features = [feature_cols[i] for i in nonzero_idx]
    print(f"Lasso selected {len(nonzero_features)} features:")

    coef_importance = [(feature_cols[i], lasso.coef_[i]) for i in nonzero_idx]
    coef_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in coef_importance[:15]:
        print(f"  {feat}: {coef:.3f}")

    if len(nonzero_idx) > 0:
        pred_spread = lasso.predict(X_test_scaled)
        pred_home_win = pred_spread > 0
        acc = (pred_home_win == actual_home_win).mean()
        mae = np.abs(pred_spread - y_test_spread).mean()
        print(f"Winner Acc: {acc*100:.1f}%, Spread MAE: {mae:.2f}")
        results.append((f'Lasso ({len(nonzero_features)} feat)', acc, mae))

    # Test 3: Only PPG-related features (our baseline)
    print("\n--- PPG/PAPG Features Only ---")
    ppg_features = [c for c in feature_cols if 'ppg' in c.lower() or 'papg' in c.lower()]
    print(f"PPG features: {ppg_features}")

    if ppg_features:
        X_train_ppg = train[ppg_features].fillna(0).values
        X_test_ppg = test[ppg_features].fillna(0).values
        scaler_ppg = StandardScaler()
        X_train_ppg = scaler_ppg.fit_transform(X_train_ppg)
        X_test_ppg = scaler_ppg.transform(X_test_ppg)

        ridge_ppg = Ridge(alpha=1.0)
        ridge_ppg.fit(X_train_ppg, y_train_spread)
        pred_spread = ridge_ppg.predict(X_test_ppg)
        pred_home_win = pred_spread > 0
        acc = (pred_home_win == actual_home_win).mean()
        mae = np.abs(pred_spread - y_test_spread).mean()
        print(f"Winner Acc: {acc*100:.1f}%, Spread MAE: {mae:.2f}")
        results.append((f'PPG only ({len(ppg_features)})', acc, mae))

    # Test 4: Only Vegas features
    print("\n--- Vegas Features Only ---")
    vegas_features = [c for c in feature_cols if 'odds_' in c.lower() or 'vegas' in c.lower()]
    print(f"Vegas features: {vegas_features}")

    if vegas_features:
        X_train_vegas = train[vegas_features].fillna(0).values
        X_test_vegas = test[vegas_features].fillna(0).values
        scaler_vegas = StandardScaler()
        X_train_vegas = scaler_vegas.fit_transform(X_train_vegas)
        X_test_vegas = scaler_vegas.transform(X_test_vegas)

        ridge_vegas = Ridge(alpha=1.0)
        ridge_vegas.fit(X_train_vegas, y_train_spread)
        pred_spread = ridge_vegas.predict(X_test_vegas)
        pred_home_win = pred_spread > 0
        acc = (pred_home_win == actual_home_win).mean()
        mae = np.abs(pred_spread - y_test_spread).mean()
        print(f"Winner Acc: {acc*100:.1f}%, Spread MAE: {mae:.2f}")
        results.append((f'Vegas only ({len(vegas_features)})', acc, mae))

    # Test 5: Top correlated features only
    print("\n--- Top 10 Correlated Features ---")
    spread_col = train['point_spread']
    corrs = [(c, abs(train[c].fillna(0).corr(spread_col))) for c in feature_cols]
    corrs.sort(key=lambda x: x[1], reverse=True)
    top10_features = [c for c, _ in corrs[:10]]
    print(f"Top 10: {top10_features}")

    X_train_top = train[top10_features].fillna(0).values
    X_test_top = test[top10_features].fillna(0).values
    scaler_top = StandardScaler()
    X_train_top = scaler_top.fit_transform(X_train_top)
    X_test_top = scaler_top.transform(X_test_top)

    ridge_top = Ridge(alpha=1.0)
    ridge_top.fit(X_train_top, y_train_spread)
    pred_spread = ridge_top.predict(X_test_top)
    pred_home_win = pred_spread > 0
    acc = (pred_home_win == actual_home_win).mean()
    mae = np.abs(pred_spread - y_test_spread).mean()
    print(f"Winner Acc: {acc*100:.1f}%, Spread MAE: {mae:.2f}")
    results.append(('Top 10 correlated', acc, mae))

    # Test 6: Differentials only
    print("\n--- Differential Features Only ---")
    diff_features = [c for c in feature_cols if 'diff' in c.lower()]
    print(f"Differential features ({len(diff_features)}): {diff_features[:10]}...")

    if diff_features:
        X_train_diff = train[diff_features].fillna(0).values
        X_test_diff = test[diff_features].fillna(0).values
        scaler_diff = StandardScaler()
        X_train_diff = scaler_diff.fit_transform(X_train_diff)
        X_test_diff = scaler_diff.transform(X_test_diff)

        ridge_diff = Ridge(alpha=1.0)
        ridge_diff.fit(X_train_diff, y_train_spread)
        pred_spread = ridge_diff.predict(X_test_diff)
        pred_home_win = pred_spread > 0
        acc = (pred_home_win == actual_home_win).mean()
        mae = np.abs(pred_spread - y_test_spread).mean()
        print(f"Winner Acc: {acc*100:.1f}%, Spread MAE: {mae:.2f}")
        results.append((f'Differentials ({len(diff_features)})', acc, mae))

    # Test 7: Simple baseline - just Vegas spread
    print("\n--- Vegas Spread Only (1 feature) ---")
    if 'odds_latest_spread' in train.columns:
        vegas_spread_test = test['odds_latest_spread'].fillna(0).values
        # Vegas spread is away - home, so negative = home favored
        # point_spread is home - away, so we need to flip sign
        pred_home_win = vegas_spread_test < 0  # home favored when vegas spread < 0
        acc = (pred_home_win == actual_home_win).mean()
        mae = np.abs(-vegas_spread_test - y_test_spread).mean()  # flip for comparison
        print(f"Winner Acc: {acc*100:.1f}%, Spread MAE: {mae:.2f}")
        results.append(('Vegas spread only', acc, mae))

    return results


def test_neural_network_sizes(train, test, feature_cols):
    """Test neural networks with different feature counts."""
    print('\n' + '=' * 70)
    print('NEURAL NETWORK FEATURE COUNT ANALYSIS')
    print('=' * 70)

    # Sort features by correlation
    spread_col = train['point_spread']
    corrs = [(c, abs(train[c].fillna(0).corr(spread_col))) for c in feature_cols]
    corrs.sort(key=lambda x: x[1], reverse=True)
    sorted_features = [c for c, _ in corrs]

    y_train = train[['home_score', 'away_score']].values
    y_test = test[['home_score', 'away_score']].values
    actual_home_win = (test['point_spread'] > 0).values

    results = []

    for n_features in [5, 10, 20, 30, 50, 100]:
        if n_features > len(sorted_features):
            continue

        use_features = sorted_features[:n_features]

        X_train = train[use_features].fillna(0).values
        X_test = test[use_features].fillna(0).values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Simple NN
        model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

        # Train
        X_t = torch.FloatTensor(X_train)
        y_t = torch.FloatTensor(y_train)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        best_loss = float('inf')
        patience = 0

        for epoch in range(100):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_t)
                val_loss = criterion(val_pred, y_t).item()

            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break

        # Evaluate
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_test)).numpy()

        pred_spread = pred[:, 0] - pred[:, 1]  # home - away
        pred_home_win = pred_spread > 0

        acc = (pred_home_win == actual_home_win).mean()
        mae = np.abs(pred_spread - (y_test[:, 0] - y_test[:, 1])).mean()

        print(f"Features: {n_features:3d} -> Winner Acc: {acc*100:.1f}%, Spread MAE: {mae:.2f}")
        results.append((n_features, acc, mae))

    return results


def identify_problematic_features(train, test, feature_cols):
    """Identify features that hurt performance."""
    print('\n' + '=' * 70)
    print('PROBLEMATIC FEATURE IDENTIFICATION')
    print('=' * 70)

    # Check for features with high train correlation but poor test generalization
    spread_train = train['point_spread']
    spread_test = test['point_spread']

    problematic = []

    for col in feature_cols:
        train_vals = train[col].fillna(0)
        test_vals = test[col].fillna(0)

        train_corr = train_vals.corr(spread_train)
        test_corr = test_vals.corr(spread_test)

        # Features where correlation flips or drops significantly
        if abs(train_corr) > 0.1:  # Only consider features with some correlation
            corr_drop = abs(train_corr) - abs(test_corr)
            sign_flip = (train_corr > 0) != (test_corr > 0)

            if corr_drop > 0.1 or sign_flip:
                problematic.append({
                    'feature': col,
                    'train_corr': train_corr,
                    'test_corr': test_corr,
                    'corr_drop': corr_drop,
                    'sign_flip': sign_flip
                })

    # Sort by correlation drop
    problematic.sort(key=lambda x: x['corr_drop'], reverse=True)

    print(f"\nFeatures with unstable correlation (train vs test):")
    print(f"{'Feature':<40} {'Train Corr':>10} {'Test Corr':>10} {'Drop':>8} {'Flip'}")
    print('-' * 75)

    for p in problematic[:20]:
        flip_str = 'YES' if p['sign_flip'] else ''
        print(f"{p['feature']:<40} {p['train_corr']:>10.3f} {p['test_corr']:>10.3f} {p['corr_drop']:>8.3f} {flip_str}")

    return problematic


def main():
    print('=' * 70)
    print('NBA DEEP EAGLE FEATURE ANALYSIS')
    print('=' * 70)

    train, test, feature_cols = load_and_split_data()
    print(f"\nTrain: {len(train)} games")
    print(f"Test: {len(test)} games")
    print(f"Features: {len(feature_cols)}")

    # 1. Correlation analysis
    spread_corrs, total_corrs = analyze_correlations(train, test, feature_cols)

    # 2. Feature subset testing
    subset_results = test_feature_subsets(train, test, feature_cols)

    # 3. NN with different feature counts
    nn_results = test_neural_network_sizes(train, test, feature_cols)

    # 4. Problematic features
    problematic = identify_problematic_features(train, test, feature_cols)

    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)

    print("\nFeature Subset Results:")
    print(f"{'Model':<30} {'Winner Acc':>12} {'Spread MAE':>12}")
    print('-' * 55)
    for name, acc, mae in subset_results:
        print(f"{name:<30} {acc*100:>11.1f}% {mae:>12.2f}")

    print("\nNeural Network by Feature Count:")
    print(f"{'Features':>10} {'Winner Acc':>12} {'Spread MAE':>12}")
    print('-' * 35)
    for n, acc, mae in nn_results:
        print(f"{n:>10} {acc*100:>11.1f}% {mae:>12.2f}")

    # Best performers
    best_subset = max(subset_results, key=lambda x: x[1])
    print(f"\nBest subset: {best_subset[0]} at {best_subset[1]*100:.1f}%")

    if nn_results:
        best_nn = max(nn_results, key=lambda x: x[1])
        print(f"Best NN features: {best_nn[0]} at {best_nn[1]*100:.1f}%")


if __name__ == '__main__':
    main()
