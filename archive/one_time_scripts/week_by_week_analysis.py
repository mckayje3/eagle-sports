"""
Week-by-Week Performance Analysis: Deep Eagle vs Vegas

This compares:
- Deep Eagle: Winner prediction based on ppg_differential (positive = home wins)
- Vegas: Winner prediction based on spread (negative spread = home favored to win)

For "outright winner" accuracy, Vegas's prediction is: if spread < 0, home wins; else away wins
"""
import pandas as pd
import numpy as np

# Load the v4 features file (has correct odds data)
df = pd.read_csv('nfl_2024_2025_features_v4.csv')

print('=' * 80)
print('WEEK-BY-WEEK PERFORMANCE: DEEP EAGLE vs VEGAS')
print('=' * 80)

all_results = []

# Analyze by season and week
for season in sorted(df['season'].unique()):
    season_df = df[df['season'] == season]

    print(f'\n{"="*60}')
    print(f'SEASON {season}')
    print('='*60)
    print(f"Week   Games   Us       Vegas    Diff     Winner")
    print('-'*60)

    season_us_wins = 0
    season_vegas_wins = 0
    season_ties = 0

    for week in sorted(season_df['week'].unique()):
        week_df = season_df[season_df['week'] == week]
        n_games = len(week_df)

        # Actual result: did home team win?
        actual_home_won = week_df['home_score'] > week_df['away_score']

        # Our prediction: use ppg_differential (positive = predict home wins)
        # For week 1, we use prev_season_ppg_diff
        if 'ppg_differential' in week_df.columns:
            # Use combined: ppg_differential + weighted_vegas_spread direction
            our_pred = week_df['ppg_differential'].copy()
            # For week 1 where ppg_differential is from prev season, use prev_season_ppg_diff
            if 'prev_season_ppg_diff' in week_df.columns:
                mask = week_df['ppg_differential'] == 0
                our_pred[mask] = week_df.loc[mask, 'prev_season_ppg_diff']

            our_home_pred = our_pred > 0
            our_correct = (our_home_pred == actual_home_won).sum()
            our_pct = 100 * our_correct / n_games if n_games > 0 else 0
        else:
            our_pct = 50

        # Vegas: negative spread = home favored to win outright
        if 'odds_latest_spread' in week_df.columns:
            # Filter to games with valid spread data
            valid_spread = week_df['odds_latest_spread'].notna() & (week_df['odds_latest_spread'] != 0)
            if valid_spread.sum() > 0:
                valid_df = week_df[valid_spread]
                vegas_pred_home = valid_df['odds_latest_spread'] < 0
                actual_home_won_valid = valid_df['home_score'] > valid_df['away_score']
                vegas_correct = (vegas_pred_home == actual_home_won_valid).sum()
                vegas_pct = 100 * vegas_correct / len(valid_df)
                vegas_games = len(valid_df)
            else:
                vegas_pct = 50
                vegas_games = 0
        else:
            vegas_pct = 50
            vegas_games = 0

        diff = our_pct - vegas_pct

        if diff > 2:
            winner = 'US'
            season_us_wins += 1
        elif diff < -2:
            winner = 'VEGAS'
            season_vegas_wins += 1
        else:
            winner = 'TIE'
            season_ties += 1

        print(f'{week:<6} {n_games:<7} {our_pct:>6.1f}%  {vegas_pct:>6.1f}%  {diff:>+6.1f}%   {winner:<10}')

        all_results.append({
            'season': season,
            'week': week,
            'games': n_games,
            'our_pct': our_pct,
            'vegas_pct': vegas_pct,
            'diff': diff,
            'winner': winner
        })

    print('-'*60)
    print(f'Season {season}: US wins {season_us_wins} weeks, Vegas wins {season_vegas_wins} weeks, Ties {season_ties}')

# Overall summary
print('\n' + '='*80)
print('OVERALL SUMMARY')
print('='*80)

results_df = pd.DataFrame(all_results)

# Count total weeks won
total_us = (results_df['winner'] == 'US').sum()
total_vegas = (results_df['winner'] == 'VEGAS').sum()
total_ties = (results_df['winner'] == 'TIE').sum()

print(f'\nTotal weeks won: US={total_us}, Vegas={total_vegas}, Ties={total_ties}')

# Average accuracy
print(f'\nAverage accuracy:')
print(f'  Deep Eagle: {results_df["our_pct"].mean():.1f}%')
print(f'  Vegas:      {results_df["vegas_pct"].mean():.1f}%')

# Best/worst weeks for us
print('\nBest weeks for Deep Eagle (vs Vegas):')
best = results_df.nlargest(5, 'diff')
for _, row in best.iterrows():
    print(f'  {row["season"]} Week {row["week"]}: +{row["diff"]:.1f}% ({row["our_pct"]:.1f}% vs {row["vegas_pct"]:.1f}%)')

print('\nWorst weeks for Deep Eagle (vs Vegas):')
worst = results_df.nsmallest(5, 'diff')
for _, row in worst.iterrows():
    print(f'  {row["season"]} Week {row["week"]}: {row["diff"]:.1f}% ({row["our_pct"]:.1f}% vs {row["vegas_pct"]:.1f}%)')

# Week-by-week patterns across seasons
print('\n' + '='*80)
print('WEEK-BY-WEEK PATTERNS (Averaged Across Seasons)')
print('='*80)

week_patterns = results_df.groupby('week').agg({
    'our_pct': 'mean',
    'vegas_pct': 'mean',
    'diff': 'mean',
    'games': 'sum'
}).round(1)

print(f"\nWeek   Games   Avg Us   Avg Vegas  Avg Diff")
print('-'*50)
for week, row in week_patterns.iterrows():
    print(f'{week:<6} {int(row["games"]):<7} {row["our_pct"]:>6.1f}%  {row["vegas_pct"]:>7.1f}%   {row["diff"]:>+6.1f}%')

print('\n' + '='*80)
