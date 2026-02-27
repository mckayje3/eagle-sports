"""
Evaluate Deep Eagle Model Performance by Week
Compare actual model predictions vs Vegas for winner accuracy
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler


class DeepEagleModel(nn.Module):
    """Deep Eagle neural network for score prediction"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(DeepEagleModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        self.home_score_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.away_score_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        home_score = self.home_score_head(features)
        away_score = self.away_score_head(features)
        return home_score, away_score


# Load features
print("Loading features...")
df = pd.read_csv('nfl_2024_2025_features_v4.csv')

# Load model checkpoint
print("Loading model...")
checkpoint = torch.load('models/deep_eagle_nfl_2025.pt', weights_only=False)
feature_cols = checkpoint['feature_cols']
input_dim = len(feature_cols)

# Recreate model architecture
model = DeepEagleModel(input_dim=input_dim)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load scaler
with open('models/deep_eagle_nfl_2025_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Use feature columns from saved model
print(f"Using {len(feature_cols)} features from saved model")

# Filter to completed games only
df = df[(df['home_score'].notna()) & (df['away_score'].notna())].copy()
df['actual_home_win'] = (df['home_score'] > df['away_score']).astype(int)
df['actual_spread'] = df['home_score'] - df['away_score']

print(f"Total completed games: {len(df)}")

# Make predictions
print("Making predictions...")
X = df[feature_cols].values
X_scaled = scaler.transform(X)
X_tensor = torch.FloatTensor(X_scaled)

with torch.no_grad():
    home_score_pred, away_score_pred = model(X_tensor)
    df['pred_home_score'] = home_score_pred.numpy().flatten()
    df['pred_away_score'] = away_score_pred.numpy().flatten()
    df['pred_spread'] = df['pred_home_score'] - df['pred_away_score']
    df['pred_total'] = df['pred_home_score'] + df['pred_away_score']

# Prediction: home wins if predicted spread > 0
df['pred_home_win'] = (df['pred_spread'] > 0).astype(int)
df['our_correct'] = (df['pred_home_win'] == df['actual_home_win']).astype(int)

# Vegas: home favored if spread < 0 (negative spread = home favorite)
# If spread < 0, Vegas predicts home wins
df['vegas_pred_home_win'] = (df['odds_latest_spread'] < 0).astype(int)
df['vegas_correct'] = (df['vegas_pred_home_win'] == df['actual_home_win']).astype(int)

print("\n" + "="*80)
print("DEEP EAGLE MODEL vs VEGAS - WEEK BY WEEK ANALYSIS")
print("="*80)

all_results = []

for season in sorted(df['season'].unique()):
    season_df = df[df['season'] == season]

    print(f"\n{'='*60}")
    print(f"SEASON {int(season)}")
    print("="*60)
    print(f"Week   Games  Model    Vegas    Diff     Winner")
    print("-"*60)

    season_our_wins = 0
    season_vegas_wins = 0
    season_ties = 0

    for week in sorted(season_df['week'].unique()):
        week_df = season_df[season_df['week'] == week]
        n_games = len(week_df)

        # Our model accuracy
        our_pct = 100 * week_df['our_correct'].mean()

        # Vegas accuracy (filter to games with valid spread)
        valid_vegas = week_df['odds_latest_spread'].notna() & (week_df['odds_latest_spread'] != 0)
        if valid_vegas.sum() > 0:
            vegas_pct = 100 * week_df.loc[valid_vegas, 'vegas_correct'].mean()
        else:
            vegas_pct = 50.0

        diff = our_pct - vegas_pct

        if diff > 2:
            winner = "MODEL"
            season_our_wins += 1
        elif diff < -2:
            winner = "VEGAS"
            season_vegas_wins += 1
        else:
            winner = "TIE"
            season_ties += 1

        print(f"{int(week):<6} {n_games:<6} {our_pct:>6.1f}%  {vegas_pct:>6.1f}%  {diff:>+6.1f}%   {winner}")

        all_results.append({
            'season': int(season),
            'week': int(week),
            'games': n_games,
            'model_pct': our_pct,
            'vegas_pct': vegas_pct,
            'diff': diff,
            'winner': winner
        })

    # Season totals
    season_our_correct = season_df['our_correct'].sum()
    season_vegas_correct = season_df.loc[season_df['odds_latest_spread'].notna() & (season_df['odds_latest_spread'] != 0), 'vegas_correct'].sum()
    season_total = len(season_df)
    vegas_total = (season_df['odds_latest_spread'].notna() & (season_df['odds_latest_spread'] != 0)).sum()

    print("-"*60)
    print(f"SEASON TOTAL: Model {season_our_correct}/{season_total} ({100*season_our_correct/season_total:.1f}%), "
          f"Vegas {season_vegas_correct}/{vegas_total} ({100*season_vegas_correct/vegas_total:.1f}%)")
    print(f"Weeks won: Model={season_our_wins}, Vegas={season_vegas_wins}, Ties={season_ties}")

# Overall summary
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)

results_df = pd.DataFrame(all_results)

total_model_wins = (results_df['winner'] == 'MODEL').sum()
total_vegas_wins = (results_df['winner'] == 'VEGAS').sum()
total_ties = (results_df['winner'] == 'TIE').sum()

print(f"\nWeeks won: Model={total_model_wins}, Vegas={total_vegas_wins}, Ties={total_ties}")

# Overall accuracy
total_our_correct = df['our_correct'].sum()
valid_vegas_mask = df['odds_latest_spread'].notna() & (df['odds_latest_spread'] != 0)
total_vegas_correct = df.loc[valid_vegas_mask, 'vegas_correct'].sum()
total_games = len(df)
total_vegas_games = valid_vegas_mask.sum()

print(f"\nOverall accuracy:")
print(f"  Deep Eagle Model: {total_our_correct}/{total_games} = {100*total_our_correct/total_games:.1f}%")
print(f"  Vegas Favorite:   {total_vegas_correct}/{total_vegas_games} = {100*total_vegas_correct/total_vegas_games:.1f}%")

# Best/worst weeks
print("\nBest weeks for Model (vs Vegas):")
best = results_df.nlargest(5, 'diff')
for _, row in best.iterrows():
    print(f"  {row['season']} Week {row['week']}: +{row['diff']:.1f}% ({row['model_pct']:.1f}% vs {row['vegas_pct']:.1f}%)")

print("\nWorst weeks for Model (vs Vegas):")
worst = results_df.nsmallest(5, 'diff')
for _, row in worst.iterrows():
    print(f"  {row['season']} Week {row['week']}: {row['diff']:.1f}% ({row['model_pct']:.1f}% vs {row['vegas_pct']:.1f}%)")

# Week patterns
print("\n" + "="*80)
print("PATTERNS BY WEEK NUMBER (Averaged Across Seasons)")
print("="*80)
print(f"\nWeek   Games  Avg Model  Avg Vegas  Avg Diff   Pattern")
print("-"*70)

week_patterns = results_df.groupby('week').agg({
    'model_pct': 'mean',
    'vegas_pct': 'mean',
    'diff': 'mean',
    'games': 'sum'
}).round(1)

for week, row in week_patterns.iterrows():
    if row['diff'] > 5:
        pattern = "MODEL STRONG"
    elif row['diff'] > 0:
        pattern = "Model slight edge"
    elif row['diff'] < -5:
        pattern = "VEGAS STRONG"
    elif row['diff'] < 0:
        pattern = "Vegas slight edge"
    else:
        pattern = "Even"

    print(f"{week:<6} {int(row['games']):<6} {row['model_pct']:>8.1f}%  {row['vegas_pct']:>8.1f}%   {row['diff']:>+6.1f}%   {pattern}")

# Early vs late season analysis
print("\n" + "="*80)
print("EARLY SEASON (Weeks 1-4) vs LATE SEASON (Weeks 10+)")
print("="*80)

early = df[df['week'] <= 4]
late = df[df['week'] >= 10]

early_model = 100 * early['our_correct'].mean()
early_vegas_mask = early['odds_latest_spread'].notna() & (early['odds_latest_spread'] != 0)
early_vegas = 100 * early.loc[early_vegas_mask, 'vegas_correct'].mean()

late_model = 100 * late['our_correct'].mean()
late_vegas_mask = late['odds_latest_spread'].notna() & (late['odds_latest_spread'] != 0)
late_vegas = 100 * late.loc[late_vegas_mask, 'vegas_correct'].mean()

print(f"\nEarly Season (Weeks 1-4): {len(early)} games")
print(f"  Model: {early_model:.1f}%")
print(f"  Vegas: {early_vegas:.1f}%")
print(f"  Diff:  {early_model - early_vegas:+.1f}%")

print(f"\nLate Season (Weeks 10+): {len(late)} games")
print(f"  Model: {late_model:.1f}%")
print(f"  Vegas: {late_vegas:.1f}%")
print(f"  Diff:  {late_model - late_vegas:+.1f}%")

print("\n" + "="*80)
