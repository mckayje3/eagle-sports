"""
Extract ML features from both 2024 and 2025 seasons for comprehensive training
"""
import pandas as pd
from ml_feature_extraction_v2 import FeatureExtractorV2

def extract_combined_features():
    """Extract features from both 2024 and 2025 seasons"""
    extractor = FeatureExtractorV2()

    print("="*80)
    print("EXTRACTING FEATURES FROM 2024 + 2025 SEASONS")
    print("="*80 + "\n")

    # Extract 2024 season (completed games, starting from week 3)
    print("Extracting 2024 season features...")
    print("Starting from Week 3 (teams have prior game stats)\n")
    df_2024 = extractor.extract_all_games(season=2024, min_week=3, completed_only=True)
    print(f"[OK] 2024 season: {len(df_2024)} games extracted\n")

    # Extract 2025 season (completed games, starting from week 3)
    print("Extracting 2025 season features...")
    print("Starting from Week 3 (teams have prior game stats)\n")
    df_2025 = extractor.extract_all_games(season=2025, min_week=3, completed_only=True)
    print(f"[OK] 2025 season: {len(df_2025)} games extracted\n")

    # Combine both seasons
    print("Combining datasets...")
    df_combined = pd.concat([df_2024, df_2025], ignore_index=True)
    print(f"[OK] Combined: {len(df_combined)} total games\n")

    # Sort by season and week
    df_combined = df_combined.sort_values(['season', 'week']).reset_index(drop=True)

    # Save combined features
    output_file = 'ml_features_v2_combined.csv'
    df_combined.to_csv(output_file, index=False)

    print("="*80)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nSaved to: {output_file}")
    print(f"Total features: {len(df_combined.columns)}")
    print(f"Total games: {len(df_combined)}")
    print(f"  - 2024 season: {len(df_2024)} games")
    print(f"  - 2025 season: {len(df_2025)} games")

    # Show season distribution
    print("\nGames by season and week:")
    season_week_counts = df_combined.groupby(['season', 'week']).size()
    for (season, week), count in season_week_counts.items():
        print(f"  {season} Week {week}: {count} games")

    # Show feature columns
    print(f"\nFeature columns ({len(df_combined.columns)} total):")
    feature_cols = [col for col in df_combined.columns if col not in ['game_id', 'season', 'week', 'home_team_id', 'away_team_id']]
    for col in feature_cols[:20]:  # Show first 20
        print(f"  - {col}")
    if len(feature_cols) > 20:
        print(f"  ... and {len(feature_cols) - 20} more features")

    print("\n" + "="*80 + "\n")

    return df_combined


if __name__ == '__main__':
    df = extract_combined_features()

    print("Next steps:")
    print("1. Train models with combined data: py train_score_predictor.py")
    print("2. Update training script to use ml_features_v2_combined.csv")
    print("3. Generate predictions: py predict_scores.py\n")
