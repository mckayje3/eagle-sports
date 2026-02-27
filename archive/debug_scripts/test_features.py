"""Test feature extraction"""
from ml_feature_extraction_v2 import FeatureExtractorV2
import pandas as pd

extractor = FeatureExtractorV2('cfb_games.db')

# Test with a few different games
test_games = [401752773, 401752774, 401752775]

for game_id in test_games:
    print(f"\n{'='*60}")
    print(f"Game ID: {game_id}")
    print('='*60)

    features = extractor.get_game_features(game_id)

    if len(features) <= 1:
        print("WARNING: Only got game_id, no other features!")
        continue

    # Check key stats
    print(f"\nKey features:")
    print(f"  week: {features.get('week', 'N/A')}")
    print(f"  home_points_scored_avg: {features.get('home_points_scored_avg', 'N/A')}")
    print(f"  away_points_scored_avg: {features.get('away_points_scored_avg', 'N/A')}")
    print(f"  home_win_pct: {features.get('home_win_pct', 'N/A')}")
    print(f"  away_win_pct: {features.get('away_win_pct', 'N/A')}")
    print(f"  points_scored_diff: {features.get('points_scored_diff', 'N/A')}")
    print(f"\nTotal features extracted: {len(features)}")

    # Show first 15 features
    print(f"\nFirst 15 features:")
    for i, (k, v) in enumerate(features.items()):
        if i >= 15:
            break
        print(f"  {k}: {v}")
