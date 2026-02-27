"""
Test the _get_historical_stats method directly
"""
from deep_eagle_feature_extractor import DeepEagleFeatureExtractor

extractor = DeepEagleFeatureExtractor('nfl_games.db', sport='nfl')

# Test Eagles (team 21) for Week 13
team_id = 21
season = 2025
week = 13

print(f"Testing _get_historical_stats({team_id}, {season}, {week})...\n")

result = extractor._get_historical_stats(team_id, season, week)

print("Result:")
for key, value in result.items():
    print(f"  {key}: {value}")

# Also test Bears (team 3)
print(f"\n\nTesting _get_historical_stats(3, {season}, {week})...\n")

result2 = extractor._get_historical_stats(3, season, week)

print("Result:")
for key, value in result2.items():
    print(f"  {key}: {value}")
