"""
Debug script to see what features are being generated for predictions
"""
import pandas as pd
from deep_eagle_feature_extractor import DeepEagleFeatureExtractor
import sqlite3

# Get one upcoming game
conn = sqlite3.connect('nfl_games.db')
upcoming_df = pd.read_sql_query('''
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team_id,
        g.away_team_id,
        ht.display_name as home_team,
        at.display_name as away_team,
        g.neutral_site,
        g.conference_game
    FROM games g
    JOIN teams ht ON g.home_team_id = ht.team_id
    JOIN teams at ON g.away_team_id = at.team_id
    WHERE g.season = 2025 AND g.week = 13 AND g.completed = 0
    LIMIT 1
''', conn)
conn.close()

if upcoming_df.empty:
    print("No upcoming games found!")
    exit(1)

game = upcoming_df.iloc[0]
print(f"\n{'='*80}")
print(f"TEST GAME: {game['away_team']} @ {game['home_team']} (Week {game['week']})")
print('='*80)

# Extract features
extractor = DeepEagleFeatureExtractor('nfl_games.db', sport='nfl')

game_dict = {
    'game_id': game['game_id'],
    'season': game['season'],
    'week': game['week'],
    'home_team_id': game['home_team_id'],
    'away_team_id': game['away_team_id'],
    'neutral_site': game['neutral_site'],
    'conference_game': game['conference_game'],
    'home_score': 0,  # Dummy
    'away_score': 0,  # Dummy
    'temperature': 70,
    'wind_speed': 0,
    'is_dome': 0
}

features = extractor._extract_game_features(pd.Series(game_dict))

# Check drive features
drive_features = {k: v for k, v in features.items() if 'drive' in k.lower()}

print(f"\nExtracted {len(features)} total features")
print(f"\nDrive features ({len(drive_features)}):")
for key in sorted(drive_features.keys()):
    print(f"  {key}: {drive_features[key]}")

# Check if all drive features are zeros
all_zeros = all(v == 0 for v in drive_features.values())
print(f"\nAll drive features are zero: {all_zeros}")

# Check some other key features
print(f"\nOther key features:")
print(f"  home_hist_ppg: {features.get('home_hist_ppg', 'MISSING')}")
print(f"  away_hist_ppg: {features.get('away_hist_ppg', 'MISSING')}")
print(f"  home_hist_win_pct: {features.get('home_hist_win_pct', 'MISSING')}")
print(f"  away_hist_win_pct: {features.get('away_hist_win_pct', 'MISSING')}")
