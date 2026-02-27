"""
Enhanced feature extraction with drive-based metrics
Version 3: Combines traditional stats + drive efficiency metrics
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml_feature_extraction_v2 import FeatureExtractorV2


class FeatureExtractorV3(FeatureExtractorV2):
    """Enhanced feature extractor with drive-based metrics"""

    def get_team_drive_stats(self, team_id, season, up_to_week=None):
        """
        Calculate drive-based statistics for the season up to a specific week
        """
        conn = sqlite3.connect(self.db_path)

        # Convert to Python int
        team_id = int(team_id)
        season = int(season)
        if up_to_week is not None:
            up_to_week = int(up_to_week)

        # Get offensive drives (team with possession)
        query = """
        SELECT
            d.*,
            g.is_dome,
            CASE
                WHEN d.team_id = g.home_team_id THEN 1
                ELSE 0
            END as is_home
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE d.team_id = ?
            AND g.season = ?
            AND g.completed = 1
        """

        params = [team_id, season]

        if up_to_week:
            query += " AND g.week < ?"
            params.append(up_to_week)

        drives_df = pd.read_sql_query(query, conn, params=params)

        if drives_df.empty:
            conn.close()
            return self._empty_drive_stats()

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

        stats = {
            # Offensive efficiency
            'off_points_per_drive': drives_df['points'].mean(),
            'off_yards_per_drive': drives_df['yards'].mean(),
            'off_plays_per_drive': drives_df['plays'].mean(),
            'off_seconds_per_drive': drives_df['time_elapsed_seconds'].mean(),
            'off_scoring_pct': drives_df['is_score'].mean(),

            # Red zone
            'off_redzone_scoring_pct': drives_df[
                drives_df['start_yards_to_endzone'] <= 20
            ]['is_score'].mean() if len(drives_df[drives_df['start_yards_to_endzone'] <= 20]) > 0 else 0,

            # Explosiveness
            'off_explosive_pct': len(drives_df[drives_df['yards'] >= 20]) / len(drives_df) if len(drives_df) > 0 else 0,

            # Efficiency
            'off_three_and_out_pct': len(drives_df[
                (drives_df['plays'] <= 3) & (drives_df['is_score'] == 0)
            ]) / len(drives_df) if len(drives_df) > 0 else 0,

            # Total drives
            'off_total_drives': len(drives_df)
        }

        # Defensive drives (opponent possessions)
        def_query = """
        SELECT
            d.*
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE d.team_id != ?
            AND (g.home_team_id = ? OR g.away_team_id = ?)
            AND g.season = ?
            AND g.completed = 1
        """

        def_params = [team_id, team_id, team_id, season]

        if up_to_week:
            def_query += " AND g.week < ?"
            def_params.append(up_to_week)

        def_drives_df = pd.read_sql_query(def_query, conn, params=def_params)

        if not def_drives_df.empty:
            def_drives_df['points'] = def_drives_df['result'].apply(get_points)

            stats['def_points_per_drive'] = def_drives_df['points'].mean()
            stats['def_yards_per_drive'] = def_drives_df['yards'].mean()
            stats['def_scoring_pct'] = def_drives_df['is_score'].mean()
            stats['def_three_and_out_forced'] = len(def_drives_df[
                (def_drives_df['plays'] <= 3) & (def_drives_df['is_score'] == 0)
            ]) / len(def_drives_df) if len(def_drives_df) > 0 else 0
            stats['def_redzone_stops'] = 1 - def_drives_df[
                def_drives_df['start_yards_to_endzone'] <= 20
            ]['is_score'].mean() if len(def_drives_df[def_drives_df['start_yards_to_endzone'] <= 20]) > 0 else 0
        else:
            stats['def_points_per_drive'] = 0
            stats['def_yards_per_drive'] = 0
            stats['def_scoring_pct'] = 0
            stats['def_three_and_out_forced'] = 0
            stats['def_redzone_stops'] = 0

        # Differential stats
        stats['ppd_differential'] = stats['off_points_per_drive'] - stats['def_points_per_drive']
        stats['ypd_differential'] = stats['off_yards_per_drive'] - stats['def_yards_per_drive']
        stats['scoring_pct_differential'] = stats['off_scoring_pct'] - stats['def_scoring_pct']

        conn.close()
        return stats

    def _empty_drive_stats(self):
        """Return empty drive stats"""
        return {
            'off_points_per_drive': 0,
            'off_yards_per_drive': 0,
            'off_plays_per_drive': 0,
            'off_seconds_per_drive': 0,
            'off_scoring_pct': 0,
            'off_redzone_scoring_pct': 0,
            'off_explosive_pct': 0,
            'off_three_and_out_pct': 0,
            'off_total_drives': 0,
            'def_points_per_drive': 0,
            'def_yards_per_drive': 0,
            'def_scoring_pct': 0,
            'def_three_and_out_forced': 0,
            'def_redzone_stops': 0,
            'ppd_differential': 0,
            'ypd_differential': 0,
            'scoring_pct_differential': 0
        }

    def get_game_features(self, home_team_id, away_team_id, season, week):
        """
        Get combined features for a matchup including traditional + drive stats
        """
        # Get traditional stats
        home_stats = self.get_team_season_stats(home_team_id, season, up_to_week=week)
        away_stats = self.get_team_season_stats(away_team_id, season, up_to_week=week)

        # Get drive stats
        home_drive_stats = self.get_team_drive_stats(home_team_id, season, up_to_week=week)
        away_drive_stats = self.get_team_drive_stats(away_team_id, season, up_to_week=week)

        # Combine all stats
        features = {}

        # Home team traditional stats
        for key, value in home_stats.items():
            features[f'home_{key}'] = value

        # Away team traditional stats
        for key, value in away_stats.items():
            features[f'away_{key}'] = value

        # Home team drive stats
        for key, value in home_drive_stats.items():
            features[f'home_{key}'] = value

        # Away team drive stats
        for key, value in away_drive_stats.items():
            features[f'away_{key}'] = value

        # Matchup differentials
        features['points_scored_diff'] = home_stats['points_scored_avg'] - away_stats['points_scored_avg']
        features['points_allowed_diff'] = home_stats['points_allowed_avg'] - away_stats['points_allowed_avg']
        features['win_pct_diff'] = home_stats['win_pct'] - away_stats['win_pct']
        features['total_yards_diff'] = home_stats['total_yards_avg'] - away_stats['total_yards_avg']
        features['ppd_matchup_diff'] = home_drive_stats['ppd_differential'] - away_drive_stats['ppd_differential']

        return features

    def extract_training_data(self, season, output_path='ml_features_v3_with_drives.csv'):
        """
        Extract training data for an entire season with drive features
        """
        conn = sqlite3.connect(self.db_path)

        # Get all completed games
        query = """
        SELECT
            game_id,
            season,
            week,
            home_team_id,
            away_team_id,
            home_score,
            away_score,
            winner_team_id
        FROM games
        WHERE season = ?
            AND completed = 1
            AND week > 1
        ORDER BY week, game_id
        """

        games_df = pd.read_sql_query(query, conn, params=(season,))
        conn.close()

        print(f"Extracting features for {len(games_df)} games from {season} season...")

        all_features = []

        for idx, game in games_df.iterrows():
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(games_df)} games...")

            try:
                features = self.get_game_features(
                    game['home_team_id'],
                    game['away_team_id'],
                    game['season'],
                    game['week']
                )

                # Add target variables
                features['home_score'] = game['home_score']
                features['away_score'] = game['away_score']
                features['home_win'] = 1 if game['winner_team_id'] == game['home_team_id'] else 0
                features['point_spread'] = game['home_score'] - game['away_score']
                features['total_points'] = game['home_score'] + game['away_score']
                features['game_id'] = game['game_id']
                features['week'] = game['week']

                all_features.append(features)

            except Exception as e:
                print(f"  Error processing game {game['game_id']}: {e}")
                continue

        features_df = pd.DataFrame(all_features)

        # Save to CSV
        features_df.to_csv(output_path, index=False)

        print(f"\nFeatures extracted successfully!")
        print(f"  Total games: {len(features_df)}")
        print(f"  Total features: {len(features_df.columns)}")
        print(f"  Saved to: {output_path}")

        return features_df


if __name__ == '__main__':
    print("=" * 80)
    print("EXTRACTING ENHANCED FEATURES WITH DRIVE DATA")
    print("=" * 80)

    # CFB 2024
    print("\n1. CFB 2024")
    print("-" * 80)
    cfb_extractor = FeatureExtractorV3('cfb_games.db')
    cfb_features = cfb_extractor.extract_training_data(
        season=2024,
        output_path='ml_features_v3_2024.csv'
    )

    print(f"\nSample features (first 5 columns):")
    print(cfb_features[cfb_features.columns[:5]].head())

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 80)
