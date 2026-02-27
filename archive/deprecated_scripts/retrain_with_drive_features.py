"""
Retrain models with drive-based features (v3)
Simple comparison between v2 (traditional) and v3 (with drives)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

print("=" * 80)
print("RETRAINING MODELS WITH DRIVE FEATURES")
print("=" * 80)

# Load the enhanced features
print("\nLoading enhanced features (v3 with drives)...")
df = pd.read_csv('ml_features_v3_2024.csv')

print(f"  Loaded: {len(df)} games")
print(f"  Features: {len(df.columns)} total columns")

# Prepare features and targets
print("\nPreparing features...")

# Exclude non-feature columns
exclude_cols = ['game_id', 'week', 'home_score', 'away_score', 'home_win', 'point_spread', 'total_points']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"  Using {len(feature_cols)} features")
print(f"\n  Sample features:")
for col in feature_cols[:10]:
    print(f"    - {col}")
print(f"    ... and {len(feature_cols) - 10} more")

# Extract features and targets
X = df[feature_cols].fillna(0)
y_spread = df['point_spread']
y_total = df['total_points']
y_winner = df['home_win']

# Time-based split (train on earlier weeks, test on later weeks)
print("\nSplitting data (time-based)...")
print("  Training: Weeks 2-12")
print("  Testing: Weeks 13-15")

train_mask = df['week'] <= 12
test_mask = df['week'] > 12

X_train, X_test = X[train_mask], X[test_mask]
y_spread_train, y_spread_test = y_spread[train_mask], y_spread[test_mask]
y_total_train, y_total_test = y_total[train_mask], y_total[test_mask]
y_winner_train, y_winner_test = y_winner[train_mask], y_winner[test_mask]

print(f"  Train size: {len(X_train)} games")
print(f"  Test size: {len(X_test)} games")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train spread model
print("\n" + "=" * 80)
print("TRAINING SPREAD PREDICTION MODEL")
print("=" * 80)

spread_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    verbose=1
)

print("\nFitting spread model...")
spread_model.fit(X_train_scaled, y_spread_train)

# Evaluate spread predictions
spread_preds_train = spread_model.predict(X_train_scaled)
spread_preds_test = spread_model.predict(X_test_scaled)

print("\nSPREAD PREDICTION RESULTS:")
print("-" * 80)
print(f"Training Set:")
print(f"  MAE: {mean_absolute_error(y_spread_train, spread_preds_train):.2f} points")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_spread_train, spread_preds_train)):.2f} points")
print(f"  R^2: {r2_score(y_spread_train, spread_preds_train):.3f}")

print(f"\nTest Set (Weeks 13-15):")
print(f"  MAE: {mean_absolute_error(y_spread_test, spread_preds_test):.2f} points")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_spread_test, spread_preds_test)):.2f} points")
print(f"  R^2: {r2_score(y_spread_test, spread_preds_test):.3f}")

# Winner accuracy
winner_preds_test = (spread_preds_test > 0).astype(int)
winner_accuracy = (winner_preds_test == y_winner_test).mean()
print(f"  Winner Accuracy: {winner_accuracy:.1%}")

# Train total points model
print("\n" + "=" * 80)
print("TRAINING TOTAL POINTS MODEL")
print("=" * 80)

total_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    verbose=1
)

print("\nFitting total points model...")
total_model.fit(X_train_scaled, y_total_train)

# Evaluate total predictions
total_preds_train = total_model.predict(X_train_scaled)
total_preds_test = total_model.predict(X_test_scaled)

print("\nTOTAL POINTS PREDICTION RESULTS:")
print("-" * 80)
print(f"Training Set:")
print(f"  MAE: {mean_absolute_error(y_total_train, total_preds_train):.2f} points")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_total_train, total_preds_train)):.2f} points")
print(f"  R^2: {r2_score(y_total_train, total_preds_train):.3f}")

print(f"\nTest Set (Weeks 13-15):")
print(f"  MAE: {mean_absolute_error(y_total_test, total_preds_test):.2f} points")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_total_test, total_preds_test)):.2f} points")
print(f"  R^2: {r2_score(y_total_test, total_preds_test):.3f}")

# Feature importance analysis
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Get feature importances
importances = spread_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print("-" * 80)
for idx, row in feature_importance.head(20).iterrows():
    feature = row['feature']
    imp = row['importance']

    # Highlight drive features
    if any(x in feature for x in ['ppd', 'drive', 'scoring_pct', 'redzone', 'explosive']):
        marker = "[DRIVE]"
    else:
        marker = "[TRAD] "

    print(f"{marker:12} {feature:50} {imp:.4f}")

# Count drive vs traditional features in top 20
top_20 = feature_importance.head(20)
drive_features = sum(
    any(x in feat for x in ['ppd', 'drive', 'scoring_pct', 'redzone', 'explosive', 'three_and_out', 'ypd'])
    for feat in top_20['feature']
)
print(f"\nDrive features in top 20: {drive_features}/20 ({drive_features/20:.0%})")

# Save models
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

# Save spread model
with open('models/spread_model_v3_with_drives.pkl', 'wb') as f:
    pickle.dump(spread_model, f)
print("Spread model saved: models/spread_model_v3_with_drives.pkl")

# Save total model
with open('models/total_model_v3_with_drives.pkl', 'wb') as f:
    pickle.dump(total_model, f)
print("Total model saved: models/total_model_v3_with_drives.pkl")

# Save scaler
with open('models/scaler_v3_with_drives.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved: models/scaler_v3_with_drives.pkl")

# Save feature columns
with open('models/feature_columns_v3.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("Feature columns saved: models/feature_columns_v3.pkl")

print("\n" + "=" * 80)
print("RETRAINING COMPLETE!")
print("=" * 80)

# Summary
print("\nPERFORMANCE SUMMARY:")
print("-" * 80)
print(f"Spread MAE (Test): {mean_absolute_error(y_spread_test, spread_preds_test):.2f} points")
print(f"Winner Accuracy (Test): {winner_accuracy:.1%}")
print(f"Total MAE (Test): {mean_absolute_error(y_total_test, total_preds_test):.2f} points")
print(f"Drive features in top 20: {drive_features}/20")

print("\nNEXT STEPS:")
print("  1. Compare these results to your previous model (v2)")
print("  2. If improved, use these models for Week 14+ predictions")
print("  3. Monitor performance week-by-week")
print("  4. Retrain weekly as new data comes in")

print("\nBENCHMARKS:")
print("  Good spread MAE: < 10 points")
print("  Great spread MAE: < 8 points")
print("  Elite spread MAE: < 7 points")
print("  Good winner accuracy: > 65%")
print("  Great winner accuracy: > 70%")
