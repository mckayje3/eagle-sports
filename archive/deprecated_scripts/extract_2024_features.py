"""
Extract ML features from 2024 season for training
"""
from ml_feature_extraction_v2 import FeatureExtractorV2

def main():
    extractor = FeatureExtractorV2()

    print("="*80)
    print("EXTRACTING FEATURES FROM 2024 SEASON")
    print("="*80 + "\n")

    print("Extracting 2024 season features...")
    print("Starting from Week 3 (teams have prior game stats)\n")

    df = extractor.extract_all_games(season=2024, min_week=3, completed_only=True)

    print(f"\n[OK] Extracted features for {len(df)} games")

    # Save to CSV
    output_file = 'ml_features_v2_2024.csv'
    df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nSaved to: {output_file}")
    print(f"Total features: {len(df.columns)}")
    print(f"Total games: {len(df)}")

    # Show distribution by week
    print("\nGames by week:")
    week_counts = df.groupby('week').size()
    for week, count in week_counts.items():
        print(f"  Week {week:2d}: {count:3d} games")

    # Show sample feature columns
    print(f"\nFeature columns ({len(df.columns)} total):")
    feature_cols = [col for col in df.columns if col not in ['game_id', 'season', 'week']]
    for col in feature_cols[:15]:
        print(f"  - {col}")
    if len(feature_cols) > 15:
        print(f"  ... and {len(feature_cols) - 15} more features")

    print("\n" + "="*80 + "\n")

    print("Next step: Train models with this data")
    print("Run: py train_score_predictor.py\n")

    return df

if __name__ == '__main__':
    df = main()
