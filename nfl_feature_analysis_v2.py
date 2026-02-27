"""NFL Feature Analysis V2 - Find optimal features for spread prediction."""
from __future__ import annotations
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Connect to database
conn = sqlite3.connect('nfl_games.db')

# Load games with spreads and scores
games = pd.read_sql_query('''
    SELECT g.game_id, g.date, g.week, g.season,
           g.home_team_id, g.away_team_id,
           g.home_score, g.away_score,
           o.latest_spread as closing_spread
    FROM games g
    LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
    WHERE o.latest_spread IS NOT NULL
      AND g.home_score IS NOT NULL
      AND g.season >= 2022
      AND g.postseason_type IS NULL
    ORDER BY g.date
''', conn)

# Calculate actual margin and ATS result
games['actual_margin'] = games['home_score'] - games['away_score']
games['spread_result'] = games['actual_margin'] + games['closing_spread']
games['home_covered'] = (games['spread_result'] > 0).astype(int)

print(f"Total games with spreads: {len(games)}")
print(f"Home cover rate: {games['home_covered'].mean():.1%}")
print()

# Load team game stats
team_stats = pd.read_sql_query('''
    SELECT ts.game_id, ts.team_id,
           CASE WHEN ts.team_id = g.home_team_id THEN 1 ELSE 0 END as is_home,
           g.home_score, g.away_score,
           ts.total_yards, ts.passing_yards, ts.rushing_yards,
           ts.turnovers, ts.first_downs, ts.third_down_conversions, ts.third_down_attempts,
           ts.fourth_down_conversions, ts.fourth_down_attempts,
           ts.penalties, ts.penalty_yards, ts.sacks, ts.sack_yards,
           ts.fumbles_lost, ts.interceptions_thrown, ts.possession_time
    FROM team_game_stats ts
    JOIN games g ON ts.game_id = g.game_id
''', conn)

# Add points from game scores
team_stats['points'] = team_stats.apply(
    lambda r: r['home_score'] if r['is_home'] == 1 else r['away_score'], axis=1)
team_stats['points_allowed'] = team_stats.apply(
    lambda r: r['away_score'] if r['is_home'] == 1 else r['home_score'], axis=1)

# Load drives data
drives = pd.read_sql_query('''
    SELECT game_id, team_id, plays, yards,
           time_elapsed_seconds, is_score, result
    FROM drives
    WHERE plays > 0
''', conn)

conn.close()

# Calculate drive stats per game
if len(drives) > 0:
    drive_stats = drives.groupby(['game_id', 'team_id']).agg({
        'plays': ['sum', 'count'],
        'yards': 'sum',
        'is_score': 'sum',
        'time_elapsed_seconds': 'sum'
    }).reset_index()
    drive_stats.columns = ['game_id', 'team_id', 'total_plays', 'num_drives',
                           'total_drive_yards', 'scoring_drives', 'total_drive_time']

    drive_stats['yards_per_drive'] = drive_stats['total_drive_yards'] / drive_stats['num_drives'].clip(lower=1)
    drive_stats['scoring_pct'] = drive_stats['scoring_drives'] / drive_stats['num_drives'].clip(lower=1)
    drive_stats['time_per_drive'] = drive_stats['total_drive_time'] / drive_stats['num_drives'].clip(lower=1)

    team_stats = team_stats.merge(drive_stats[['game_id', 'team_id', 'num_drives', 'yards_per_drive',
                                                'scoring_pct', 'time_per_drive']],
                                  on=['game_id', 'team_id'], how='left')

# Convert possession time from 'MM:SS' to minutes
def parse_possession_time(pt):
    if pd.isna(pt) or pt == '' or pt is None:
        return np.nan
    try:
        if isinstance(pt, str) and ':' in pt:
            parts = pt.split(':')
            return float(parts[0]) + float(parts[1]) / 60
        return float(pt)
    except (ValueError, IndexError):
        return np.nan

team_stats['possession_time'] = team_stats['possession_time'].apply(parse_possession_time)

# Calculate derived stats
team_stats['third_down_pct'] = team_stats['third_down_conversions'] / team_stats['third_down_attempts'].clip(lower=1)
team_stats['fourth_down_pct'] = team_stats['fourth_down_conversions'] / team_stats['fourth_down_attempts'].clip(lower=1)
team_stats['yards_per_point'] = team_stats['total_yards'] / team_stats['points'].clip(lower=1)
team_stats['pass_run_ratio'] = team_stats['passing_yards'] / team_stats['rushing_yards'].clip(lower=1)

# Features to analyze
raw_features = [
    'points', 'points_allowed', 'total_yards', 'passing_yards', 'rushing_yards',
    'turnovers', 'first_downs', 'third_down_pct',
    'penalties', 'sacks',
    'num_drives', 'yards_per_drive', 'scoring_pct', 'time_per_drive',
    'possession_time'
]

# Filter to only features that exist
raw_features = [f for f in raw_features if f in team_stats.columns]
print(f"Available features: {raw_features}")

# ============================================================================
# BUILD FEATURES WITH DIFFERENT DECAY RATES
# ============================================================================
def build_features(games_df, team_stats_df, decay=0.92, min_games=3):
    """Build features with specified decay rate."""
    all_games = games_df.sort_values('date').reset_index(drop=True)
    team_histories = {}

    def get_ewma(history, feature):
        if len(history) < min_games:
            return np.nan
        values = []
        for h in history:
            if feature in h:
                v = h[feature]
                if not pd.isna(v):
                    try:
                        values.append(float(v))
                    except (ValueError, TypeError):
                        pass
        if len(values) < min_games:
            return np.nan
        weights = np.array([decay ** i for i in range(len(values))], dtype=np.float64)[::-1]
        return np.average(np.array(values, dtype=np.float64), weights=weights)

    feature_rows = []

    for idx, game in all_games.iterrows():
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        game_id = game['game_id']

        home_hist = team_histories.get(home_id, [])
        away_hist = team_histories.get(away_id, [])

        if len(home_hist) >= min_games and len(away_hist) >= min_games:
            row = {
                'game_id': game_id,
                'date': game['date'],
                'week': game['week'],
                'season': game['season'],
                'closing_spread': game['closing_spread'],
                'actual_margin': game['actual_margin'],
                'spread_result': game['spread_result'],
                'home_covered': game['home_covered']
            }

            for feat in raw_features:
                home_avg = get_ewma(home_hist, feat)
                away_avg = get_ewma(away_hist, feat)

                if not pd.isna(home_avg) and not pd.isna(away_avg):
                    row[f'{feat}_diff'] = home_avg - away_avg

                    # Also get opponent allowed stats
                    home_opp_avg = get_ewma(home_hist, f'opp_{feat}')
                    away_opp_avg = get_ewma(away_hist, f'opp_{feat}')

                    if not pd.isna(home_opp_avg) and not pd.isna(away_opp_avg):
                        row[f'{feat}_allowed_diff'] = home_opp_avg - away_opp_avg

            feature_rows.append(row)

        # Update histories
        home_stats_row = team_stats_df[(team_stats_df['game_id'] == game_id) &
                                       (team_stats_df['team_id'] == home_id)]
        away_stats_row = team_stats_df[(team_stats_df['game_id'] == game_id) &
                                       (team_stats_df['team_id'] == away_id)]

        if len(home_stats_row) > 0 and len(away_stats_row) > 0:
            home_stats = home_stats_row.iloc[0].to_dict()
            away_stats = away_stats_row.iloc[0].to_dict()

            for feat in raw_features:
                if feat in home_stats:
                    home_stats[f'opp_{feat}'] = away_stats.get(feat, np.nan)
                    away_stats[f'opp_{feat}'] = home_stats.get(feat, np.nan)

            if home_id not in team_histories:
                team_histories[home_id] = []
            if away_id not in team_histories:
                team_histories[away_id] = []

            team_histories[home_id].append(home_stats)
            team_histories[away_id].append(away_stats)

    return pd.DataFrame(feature_rows)

# ============================================================================
# WALK-FORWARD TEST FUNCTION
# ============================================================================
def walk_forward_test(df, features, train_min=100):
    """Walk-forward validation for a set of features."""
    results = []
    df_sorted = df.sort_values('date').reset_index(drop=True)

    # Check feature availability
    available = [f for f in features if f in df_sorted.columns]
    if len(available) < len(features):
        return None

    valid_mask = df_sorted[available].notna().all(axis=1)

    for i in range(len(df_sorted)):
        if i < train_min:
            continue
        if not valid_mask.iloc[i]:
            continue

        train_mask = valid_mask.iloc[:i]
        train = df_sorted.iloc[:i][train_mask]
        test_row = df_sorted.iloc[i]

        if len(train) < 50:
            continue

        X_train = train[available]
        y_train = train['spread_result']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y_train)

        X_test = scaler.transform([test_row[available].values])
        pred = model.predict(X_test)[0]

        results.append({
            'actual': test_row['spread_result'],
            'predicted': pred,
            'spread': test_row['closing_spread'],
            'covered': test_row['home_covered'],
            'week': test_row['week'],
            'season': test_row['season']
        })

    if len(results) < 50:
        return None

    res_df = pd.DataFrame(results)
    res_df['pred_cover'] = (res_df['predicted'] > 0).astype(int)
    ats = (res_df['pred_cover'] == res_df['covered']).mean()

    # Early weeks (1-4)
    early = res_df[res_df['week'] <= 4]
    early_ats = (early['pred_cover'] == early['covered']).mean() if len(early) > 20 else np.nan

    # Late weeks (12+)
    late = res_df[res_df['week'] >= 12]
    late_ats = (late['pred_cover'] == late['covered']).mean() if len(late) > 20 else np.nan

    return {
        'ats': ats,
        'early_ats': early_ats,
        'late_ats': late_ats,
        'n_games': len(res_df)
    }

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
print("Building features with default decay=0.92...")
df = build_features(games, team_stats, decay=0.92, min_games=3)
print(f"Games with features: {len(df)}")
print()

# Available feature diffs
feature_cols = [c for c in df.columns if c.endswith('_diff')]
print(f"Feature columns: {feature_cols}")
print()

# ============================================================================
# PART 1: CORRELATIONS
# ============================================================================
print("=" * 70)
print("PART 1: FEATURE CORRELATIONS WITH SPREAD OUTCOME")
print("=" * 70)

correlations = []
for col in feature_cols:
    if col in df.columns and df[col].notna().sum() > 100:
        valid = df[[col, 'spread_result']].dropna()
        corr, pval = stats.pearsonr(valid[col], valid['spread_result'])
        correlations.append({
            'feature': col,
            'correlation': corr,
            'p_value': pval,
            'n_samples': len(valid)
        })

corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
print("\nTop Features by Correlation:")
print("-" * 70)
for _, row in corr_df.head(20).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"  {row['feature']:35} r={row['correlation']:+.4f} {sig:3}")

# ============================================================================
# PART 2: SINGLE FEATURE TESTING
# ============================================================================
print()
print("=" * 70)
print("PART 2: SINGLE FEATURE ATS PERFORMANCE")
print("=" * 70)

single_results = []
for feat in corr_df['feature'].head(15):
    result = walk_forward_test(df, [feat])
    if result:
        single_results.append({
            'feature': feat,
            **result
        })

single_df = pd.DataFrame(single_results).sort_values('ats', ascending=False)
print("\nBest Single Features:")
print("-" * 70)
for _, row in single_df.head(10).iterrows():
    early = f"{row['early_ats']:.1%}" if not pd.isna(row['early_ats']) else "N/A"
    late = f"{row['late_ats']:.1%}" if not pd.isna(row['late_ats']) else "N/A"
    print(f"  {row['feature']:35} ATS: {row['ats']:.1%}  Early: {early}  Late: {late}")

# ============================================================================
# PART 3: FEATURE COMBINATIONS
# ============================================================================
print()
print("=" * 70)
print("PART 3: FEATURE COMBINATION TESTING")
print("=" * 70)

# Define combinations based on correlations
top_features = corr_df['feature'].head(10).tolist()

# Predefined combos
combos = {
    'Core Points': ['points_diff', 'points_allowed_diff'],
    'Yards Only': ['total_yards_diff', 'passing_yards_diff', 'rushing_yards_diff'],
    'Efficiency': ['yards_per_drive_diff', 'scoring_pct_diff', 'third_down_pct_diff'],
    'Drives': ['num_drives_diff', 'yards_per_drive_diff', 'scoring_pct_diff', 'time_per_drive_diff'],
    'Ball Control': ['possession_time_diff', 'turnovers_diff', 'time_per_drive_diff'],
    'Points + Turnovers': ['points_diff', 'points_allowed_diff', 'turnovers_diff'],
    'Points + Yards': ['points_diff', 'points_allowed_diff', 'total_yards_diff'],
    'Full Offense': ['points_diff', 'total_yards_diff', 'yards_per_drive_diff',
                     'scoring_pct_diff', 'first_downs_diff'],
    'Kitchen Sink': ['points_diff', 'points_allowed_diff', 'total_yards_diff',
                     'turnovers_diff', 'yards_per_drive_diff', 'scoring_pct_diff',
                     'third_down_pct_diff', 'first_downs_diff'],
}

combo_results = []
print("\nTesting predefined combinations...")

for name, features in combos.items():
    result = walk_forward_test(df, features)
    if result:
        combo_results.append({
            'name': name,
            'n_features': len(features),
            **result
        })
        early = f"{result['early_ats']:.1%}" if not pd.isna(result['early_ats']) else "N/A"
        late = f"{result['late_ats']:.1%}" if not pd.isna(result['late_ats']) else "N/A"
        print(f"  {name:25} ATS: {result['ats']:.1%}  Early: {early}  Late: {late}  ({len(features)} feat)")
    else:
        print(f"  {name:25} -- insufficient data")

combo_df = pd.DataFrame(combo_results).sort_values('ats', ascending=False)
print("\nBest Combinations:")
print("-" * 70)
for _, row in combo_df.iterrows():
    early = f"{row['early_ats']:.1%}" if not pd.isna(row['early_ats']) else "N/A"
    late = f"{row['late_ats']:.1%}" if not pd.isna(row['late_ats']) else "N/A"
    print(f"  {row['name']:25} ATS: {row['ats']:.1%}  Early: {early}  Late: {late}")

# ============================================================================
# PART 4: DECAY RATE TESTING
# ============================================================================
print()
print("=" * 70)
print("PART 4: DECAY RATE OPTIMIZATION")
print("=" * 70)

# Use best combo for decay testing
best_features = ['points_diff', 'points_allowed_diff', 'total_yards_diff', 'turnovers_diff']

decay_results = []
for decay in [0.85, 0.88, 0.90, 0.92, 0.94, 0.96]:
    print(f"\nTesting decay={decay}...")
    df_decay = build_features(games, team_stats, decay=decay, min_games=3)
    result = walk_forward_test(df_decay, best_features)
    if result:
        decay_results.append({
            'decay': decay,
            **result
        })
        print(f"  Decay {decay}: ATS={result['ats']:.1%} (n={result['n_games']})")

decay_df = pd.DataFrame(decay_results).sort_values('ats', ascending=False)
print("\nBest Decay Rates:")
print("-" * 50)
for _, row in decay_df.iterrows():
    early = f"{row['early_ats']:.1%}" if not pd.isna(row['early_ats']) else "N/A"
    print(f"  Decay {row['decay']:.2f}: ATS={row['ats']:.1%}  Early: {early}")

# ============================================================================
# PART 5: MIN_GAMES TESTING
# ============================================================================
print()
print("=" * 70)
print("PART 5: MIN_GAMES THRESHOLD TESTING")
print("=" * 70)

best_decay = decay_df.iloc[0]['decay'] if len(decay_df) > 0 else 0.92

mingames_results = []
for min_g in [2, 3, 4, 5]:
    print(f"\nTesting MIN_GAMES={min_g}...")
    df_mg = build_features(games, team_stats, decay=best_decay, min_games=min_g)
    result = walk_forward_test(df_mg, best_features)
    if result:
        mingames_results.append({
            'min_games': min_g,
            **result
        })
        print(f"  MIN_GAMES {min_g}: ATS={result['ats']:.1%} (n={result['n_games']})")

mg_df = pd.DataFrame(mingames_results).sort_values('ats', ascending=False)
print("\nBest MIN_GAMES:")
print("-" * 50)
for _, row in mg_df.iterrows():
    print(f"  MIN_GAMES {row['min_games']}: ATS={row['ats']:.1%} (n={row['n_games']})")

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("=" * 70)
print("SUMMARY: RECOMMENDED NFL MODEL CONFIGURATION")
print("=" * 70)

best_combo_row = combo_df.iloc[0] if len(combo_df) > 0 else None
best_decay_row = decay_df.iloc[0] if len(decay_df) > 0 else None
best_mg_row = mg_df.iloc[0] if len(mg_df) > 0 else None

print(f"""
BEST FEATURE COMBINATION:
  {best_combo_row['name'] if best_combo_row is not None else 'N/A'}
  ATS: {best_combo_row['ats']:.1%} if best_combo_row is not None else 'N/A'

BEST DECAY RATE:
  {best_decay_row['decay'] if best_decay_row is not None else 0.92}

BEST MIN_GAMES:
  {best_mg_row['min_games'] if best_mg_row is not None else 3}

TOP CORRELATING FEATURES:
""")
for _, row in corr_df.head(10).iterrows():
    print(f"  {row['feature']:35} r={row['correlation']:+.4f}")

# Save results
corr_df.to_csv('nfl_feature_correlations.csv', index=False)
combo_df.to_csv('nfl_combo_results.csv', index=False)
print("\nSaved: nfl_feature_correlations.csv, nfl_combo_results.csv")
