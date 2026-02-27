"""
Extract drive-based features for ML training
Adds offensive and defensive efficiency metrics from drive data
"""
import sqlite3
import pandas as pd
import numpy as np

def calculate_drive_features(db_path, season=2024):
    """Calculate drive-based features for all teams in a season"""

    conn = sqlite3.connect(db_path)

    print(f"Extracting drive features from {db_path} for {season} season...")

    # Get all drives with game context
    query = """
    SELECT
        d.*,
        g.home_team_id,
        g.away_team_id,
        g.is_dome,
        CASE
            WHEN d.team_id = g.home_team_id THEN 1
            ELSE 0
        END as is_home
    FROM drives d
    JOIN games g ON d.game_id = g.game_id
    WHERE g.season = ? AND g.completed = 1
    """

    drives_df = pd.read_sql_query(query, conn, params=(season,))

    print(f"  Loaded {len(drives_df)} drives from {drives_df['game_id'].nunique()} games")

    # Calculate points from drive result
    def get_points(result):
        if pd.isna(result):
            return 0
        result = str(result).upper()
        if 'TD' in result:
            return 7
        elif 'FG' in result:
            return 3
        elif 'SAFETY' in result:
            return 2
        return 0

    drives_df['points'] = drives_df['result'].apply(get_points)

    # Offensive features (team with possession)
    offensive_features = drives_df.groupby('team_id').agg({
        'points': ['mean', 'sum'],
        'yards': ['mean', 'std'],
        'plays': ['mean', 'std'],
        'time_elapsed_seconds': ['mean', 'std'],
        'is_score': ['mean', 'sum'],
        'drive_number': 'count'
    }).reset_index()

    offensive_features.columns = ['team_id',
        'off_points_per_drive', 'off_total_points',
        'off_yards_per_drive', 'off_yards_std',
        'off_plays_per_drive', 'off_plays_std',
        'off_seconds_per_drive', 'off_seconds_std',
        'off_scoring_pct', 'off_scoring_drives',
        'off_total_drives'
    ]

    # Red zone efficiency (drives starting inside opponent 20)
    red_zone = drives_df[drives_df['start_yards_to_endzone'] <= 20].groupby('team_id').agg({
        'is_score': ['mean', 'count']
    }).reset_index()
    red_zone.columns = ['team_id', 'off_redzone_scoring_pct', 'off_redzone_attempts']

    offensive_features = offensive_features.merge(red_zone, on='team_id', how='left')
    offensive_features['off_redzone_scoring_pct'].fillna(0, inplace=True)
    offensive_features['off_redzone_attempts'].fillna(0, inplace=True)

    # Three-and-out rate (drives with 3 or fewer plays and no score)
    three_and_outs = drives_df[
        (drives_df['plays'] <= 3) & (drives_df['is_score'] == 0)
    ].groupby('team_id').size().reset_index(name='off_three_and_outs')

    offensive_features = offensive_features.merge(three_and_outs, on='team_id', how='left')
    offensive_features['off_three_and_outs'].fillna(0, inplace=True)
    offensive_features['off_three_and_out_pct'] = (
        offensive_features['off_three_and_outs'] / offensive_features['off_total_drives']
    )

    # Explosive play drives (20+ yard plays)
    explosive_drives = drives_df[drives_df['yards'] >= 20].groupby('team_id').size().reset_index(name='off_explosive_drives')
    offensive_features = offensive_features.merge(explosive_drives, on='team_id', how='left')
    offensive_features['off_explosive_drives'].fillna(0, inplace=True)
    offensive_features['off_explosive_pct'] = (
        offensive_features['off_explosive_drives'] / offensive_features['off_total_drives']
    )

    # Home vs Away splits
    home_drives = drives_df[drives_df['is_home'] == 1].groupby('team_id')['points'].mean().reset_index()
    home_drives.columns = ['team_id', 'off_home_ppd']

    away_drives = drives_df[drives_df['is_home'] == 0].groupby('team_id')['points'].mean().reset_index()
    away_drives.columns = ['team_id', 'off_away_ppd']

    offensive_features = offensive_features.merge(home_drives, on='team_id', how='left')
    offensive_features = offensive_features.merge(away_drives, on='team_id', how='left')

    # Dome vs Outdoor splits
    dome_drives = drives_df[drives_df['is_dome'] == 1].groupby('team_id')['points'].mean().reset_index()
    dome_drives.columns = ['team_id', 'off_dome_ppd']

    outdoor_drives = drives_df[drives_df['is_dome'] == 0].groupby('team_id')['points'].mean().reset_index()
    outdoor_drives.columns = ['team_id', 'off_outdoor_ppd']

    offensive_features = offensive_features.merge(dome_drives, on='team_id', how='left')
    offensive_features = offensive_features.merge(outdoor_drives, on='team_id', how='left')

    print(f"  Calculated {len(offensive_features)} team offensive features")

    # Defensive features (opponent drives)
    # For each game, get opponent drives
    defensive_stats = []

    for team_id in drives_df['team_id'].unique():
        # Get all games where this team played
        team_games = drives_df[
            (drives_df['home_team_id'] == team_id) |
            (drives_df['away_team_id'] == team_id)
        ]['game_id'].unique()

        # Get opponent drives (drives by teams other than this team in those games)
        opponent_drives = drives_df[
            (drives_df['game_id'].isin(team_games)) &
            (drives_df['team_id'] != team_id)
        ]

        if len(opponent_drives) > 0:
            defensive_stats.append({
                'team_id': team_id,
                'def_points_per_drive': opponent_drives['points'].mean(),
                'def_yards_per_drive': opponent_drives['yards'].mean(),
                'def_plays_per_drive': opponent_drives['plays'].mean(),
                'def_scoring_pct': opponent_drives['is_score'].mean(),
                'def_total_drives': len(opponent_drives),
                'def_three_and_out_forced': len(opponent_drives[
                    (opponent_drives['plays'] <= 3) & (opponent_drives['is_score'] == 0)
                ]) / len(opponent_drives) if len(opponent_drives) > 0 else 0,
                'def_redzone_stops': 1 - opponent_drives[
                    opponent_drives['start_yards_to_endzone'] <= 20
                ]['is_score'].mean() if len(opponent_drives[opponent_drives['start_yards_to_endzone'] <= 20]) > 0 else 0
            })

    defensive_features = pd.DataFrame(defensive_stats)

    print(f"  Calculated {len(defensive_features)} team defensive features")

    # Merge offensive and defensive features
    all_features = offensive_features.merge(defensive_features, on='team_id', how='outer')

    # Fill NaN values
    all_features.fillna(0, inplace=True)

    # Add efficiency differential features
    all_features['ppd_differential'] = (
        all_features['off_points_per_drive'] - all_features['def_points_per_drive']
    )
    all_features['ypd_differential'] = (
        all_features['off_yards_per_drive'] - all_features['def_yards_per_drive']
    )
    all_features['scoring_pct_differential'] = (
        all_features['off_scoring_pct'] - all_features['def_scoring_pct']
    )

    conn.close()

    print(f"  Final feature set: {len(all_features)} teams, {len(all_features.columns)} features")

    return all_features

def save_features(features_df, output_path):
    """Save features to CSV"""
    features_df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")
    print(f"  Shape: {features_df.shape}")
    print(f"\nSample features:")
    print(features_df.head())
    print(f"\nFeature columns:")
    for col in features_df.columns:
        print(f"  - {col}")

if __name__ == '__main__':
    print("=" * 80)
    print("EXTRACTING DRIVE-BASED FEATURES FOR ML TRAINING")
    print("=" * 80)

    # CFB features
    print("\n1. CFB 2024 Features")
    print("-" * 80)
    cfb_features = calculate_drive_features('cfb_games.db', season=2024)
    save_features(cfb_features, 'cfb_drive_features_2024.csv')

    # NFL features
    print("\n2. NFL 2024 Features")
    print("-" * 80)
    nfl_features = calculate_drive_features('nfl_games.db', season=2024)
    save_features(nfl_features, 'nfl_drive_features_2024.csv')

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the extracted features")
    print("  2. Merge with existing team features")
    print("  3. Retrain ML models with enhanced feature set")
