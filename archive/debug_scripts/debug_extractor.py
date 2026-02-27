"""Debug feature extractor"""
from ml_feature_extraction_v2 import FeatureExtractorV2

extractor = FeatureExtractorV2('cfb_games.db')

# Test get_team_season_stats directly
print("Testing get_team_season_stats...")
print()

stats = extractor.get_team_season_stats(
    team_id=333,  # Alabama
    season=2025,
    up_to_week=13
)

print(f"Stats returned:")
for key, value in stats.items():
    print(f"  {key}: {value}")
