"""
NFL Data Preparation for Deep-Eagle
Extracts game-by-game data with comprehensive rolling statistics
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class NFLDataPreparation:
    """Prepare NFL data for Deep-Eagle LSTM training"""

    def __init__(self, db_path='nfl_games.db'):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_team_game_sequence(self, team_id, season, max_week=None):
        """
        Get all games for a team in chronological order

        Args:
            team_id: Team ID
            season: Season year
            max_week: Maximum week to include (for walk-forward validation)

        Returns:
            DataFrame with game-by-game stats
        """
        query = '''
            SELECT
                g.game_id,
                g.season,
                g.week,
                g.date,
                g.home_team_id,
                g.away_team_id,
                g.home_score,
                g.away_score,
                g.neutral_site,
                g.completed,
                CASE WHEN g.home_team_id = ? THEN 1 ELSE 0 END as is_home,
                CASE WHEN g.home_team_id = ? THEN g.home_score ELSE g.away_score END as points_scored,
                CASE WHEN g.home_team_id = ? THEN g.away_score ELSE g.home_score END as points_allowed,
                CASE WHEN (g.home_team_id = ? AND g.home_score > g.away_score) OR
                          (g.away_team_id = ? AND g.away_score > g.home_score)
                     THEN 1 ELSE 0 END as win,
                CASE WHEN g.home_team_id = ?
                     THEN g.home_score - g.away_score
                     ELSE g.away_score - g.home_score END as point_differential,
                tgs.total_yards,
                tgs.passing_yards,
                tgs.rushing_yards,
                tgs.turnovers,
                tgs.first_downs,
                tgs.third_down_conversions,
                tgs.third_down_attempts,
                tgs.penalties,
                tgs.penalty_yards,
                -- Vegas lines (home team perspective, latest_spread serves as closing for completed games)
                op.latest_spread as vegas_spread_home,
                op.latest_total as vegas_total,
                -- Team-specific spread (positive = team favored)
                CASE WHEN g.home_team_id = ?
                     THEN -op.latest_spread
                     ELSE op.latest_spread END as vegas_spread,
                -- ATS result (1 = covered, 0 = didn't cover, 0.5 = push)
                CASE
                    WHEN op.latest_spread IS NULL THEN NULL
                    WHEN g.home_team_id = ? THEN
                        CASE
                            WHEN (g.home_score - g.away_score) + op.latest_spread > 0 THEN 1
                            WHEN (g.home_score - g.away_score) + op.latest_spread < 0 THEN 0
                            ELSE 0.5
                        END
                    ELSE
                        CASE
                            WHEN (g.away_score - g.home_score) - op.latest_spread > 0 THEN 1
                            WHEN (g.away_score - g.home_score) - op.latest_spread < 0 THEN 0
                            ELSE 0.5
                        END
                END as covered_spread,
                -- Over/under result
                CASE
                    WHEN op.latest_total IS NULL THEN NULL
                    WHEN (g.home_score + g.away_score) > op.latest_total THEN 1
                    WHEN (g.home_score + g.away_score) < op.latest_total THEN 0
                    ELSE 0.5
                END as went_over
            FROM games g
            LEFT JOIN team_game_stats tgs ON g.game_id = tgs.game_id AND tgs.team_id = ?
            LEFT JOIN odds_and_predictions op ON g.game_id = op.game_id
            WHERE (g.home_team_id = ? OR g.away_team_id = ?)
                AND g.season = ?
                AND g.completed = 1
                {week_filter}
            ORDER BY g.date, g.game_id
        '''

        week_filter = f'AND g.week <= {max_week}' if max_week else ''
        query = query.format(week_filter=week_filter)

        # team_id used for: is_home, points_scored, points_allowed, win(2x), point_diff,
        # vegas_spread, covered_spread(2x), tgs join, home filter, away filter
        params = (team_id, team_id, team_id, team_id, team_id, team_id,
                 team_id, team_id, team_id, team_id, team_id, season)

        df = pd.read_sql_query(query, self.conn, params=params)

        # Calculate third down percentage
        df['third_down_pct'] = df.apply(
            lambda row: row['third_down_conversions'] / row['third_down_attempts']
            if row['third_down_attempts'] and row['third_down_attempts'] > 0 else 0.4,
            axis=1
        )

        return df

    def create_rolling_features(self, df, windows=[3, 5, 10]):
        """
        Create rolling average features

        Args:
            df: Team game sequence DataFrame
            windows: List of rolling window sizes

        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()

        # Fill missing stats with NFL averages before rolling
        stat_defaults = {
            'total_yards': 330.0,
            'passing_yards': 220.0,
            'rushing_yards': 110.0,
            'turnovers': 1.0,
            'first_downs': 20.0,
            'third_down_conversions': 5.0,
            'third_down_attempts': 12.0,
            'penalties': 6.0,
            'penalty_yards': 50.0
        }

        for col, default_val in stat_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)

        # Stats to roll
        roll_cols = [
            'points_scored', 'points_allowed', 'point_differential',
            'total_yards', 'passing_yards', 'rushing_yards',
            'turnovers', 'first_downs', 'third_down_pct',
            'penalties', 'penalty_yards'
        ]

        for window in windows:
            for col in roll_cols:
                if col in df.columns:
                    df[f'{col}_roll{window}'] = df[col].rolling(
                        window=window, min_periods=1
                    ).mean()

        # Win percentage rolling
        for window in windows:
            df[f'win_pct_roll{window}'] = df['win'].rolling(
                window=window, min_periods=1
            ).mean()

        # ATS (Against The Spread) rolling performance
        for window in windows:
            if 'covered_spread' in df.columns:
                df[f'ats_pct_roll{window}'] = df['covered_spread'].rolling(
                    window=window, min_periods=1
                ).mean()

            if 'went_over' in df.columns:
                df[f'over_pct_roll{window}'] = df['went_over'].rolling(
                    window=window, min_periods=1
                ).mean()

        return df

    def create_lag_features(self, df, lags=[1, 2]):
        """
        Create lag features (previous game stats)

        Args:
            df: Team game sequence DataFrame
            lags: List of lag periods

        Returns:
            DataFrame with lag features added
        """
        df = df.copy()

        lag_cols = [
            'points_scored', 'points_allowed', 'point_differential',
            'win', 'total_yards'
        ]

        for lag in lags:
            for col in lag_cols:
                if col in df.columns:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)

        return df

    def create_streak_features(self, df):
        """
        Create winning/losing streak features

        Args:
            df: Team game sequence DataFrame

        Returns:
            DataFrame with streak features added
        """
        df = df.copy()

        # Winning streak
        df['winning_streak'] = (
            df['win']
            .groupby((df['win'] != df['win'].shift()).cumsum())
            .cumcount() + 1
        ) * df['win']

        # Losing streak
        df['losing_streak'] = (
            (1 - df['win'])
            .groupby(((1 - df['win']) != (1 - df['win']).shift()).cumsum())
            .cumcount() + 1
        ) * (1 - df['win'])

        return df

    def get_opponent_stats(self, game_row, season, week):
        """
        Get opponent's rolling stats up to this game

        Args:
            game_row: Row from games DataFrame
            season: Season year
            week: Week number

        Returns:
            Dictionary of opponent stats
        """
        opponent_id = game_row['away_team_id'] if game_row['is_home'] == 1 else game_row['home_team_id']

        # Get opponent's games before this week
        opp_df = self.get_team_game_sequence(opponent_id, season, max_week=week-1)

        if len(opp_df) == 0:
            # No previous games, use NFL defaults
            return {
                'opp_win_pct': 0.5,
                'opp_points_scored_avg': 22.0,  # NFL average
                'opp_points_allowed_avg': 22.0,
                'opp_total_yards_avg': 330.0  # NFL average
            }

        # Calculate opponent averages
        return {
            'opp_win_pct': opp_df['win'].mean(),
            'opp_points_scored_avg': opp_df['points_scored'].mean(),
            'opp_points_allowed_avg': opp_df['points_allowed'].mean(),
            'opp_total_yards_avg': opp_df['total_yards'].mean() if opp_df['total_yards'].notna().any() else 330.0
        }

    def prepare_team_features(self, team_id, season):
        """
        Prepare full feature set for a team's season

        Args:
            team_id: Team ID
            season: Season year

        Returns:
            DataFrame with all features
        """
        # Get team's game sequence
        df = self.get_team_game_sequence(team_id, season)

        if len(df) == 0:
            return None

        # Create rolling features
        df = self.create_rolling_features(df)

        # Create lag features
        df = self.create_lag_features(df)

        # Create streak features
        df = self.create_streak_features(df)

        # Add opponent strength features
        opp_stats_list = []
        for idx, row in df.iterrows():
            opp_stats = self.get_opponent_stats(row, season, row['week'])
            opp_stats_list.append(opp_stats)

        opp_df = pd.DataFrame(opp_stats_list)
        df = pd.concat([df.reset_index(drop=True), opp_df], axis=1)

        # Add rest days (days since last game)
        df['date'] = pd.to_datetime(df['date'])
        df['rest_days'] = df['date'].diff().dt.days.fillna(7)

        # Add season progress (early vs late season indicator)
        df['season_progress'] = df['week'] / 18  # NFL has 18 weeks

        return df

    def prepare_all_data(self, seasons=[2023, 2024, 2025]):
        """
        Prepare data for all teams across multiple seasons

        Args:
            seasons: List of seasons to include

        Returns:
            DataFrame with all team-game features
        """
        self.connect()

        # Get all teams
        teams_df = pd.read_sql_query('SELECT DISTINCT team_id FROM teams', self.conn)
        team_ids = teams_df['team_id'].tolist()

        all_data = []

        print(f"Preparing data for {len(team_ids)} teams across {len(seasons)} seasons...")

        for season in seasons:
            print(f"\nProcessing {season} season...")
            teams_processed = 0

            for team_id in team_ids:
                team_df = self.prepare_team_features(team_id, season)

                if team_df is not None and len(team_df) > 0:
                    team_df['team_id'] = team_id
                    all_data.append(team_df)
                    teams_processed += 1

            print(f"  {season}: {teams_processed} teams processed")

        self.close()

        if not all_data:
            print("No data found!")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        print(f"\nTotal games prepared: {len(combined_df)}")
        print(f"Features per game: {len(combined_df.columns)}")

        return combined_df


def main():
    """Prepare NFL data and save to file"""
    print("=" * 80)
    print("NFL DATA PREPARATION FOR DEEP-EAGLE")
    print("=" * 80)

    prep = NFLDataPreparation()

    # Prepare all data (2023, 2024, 2025)
    df = prep.prepare_all_data(seasons=[2023, 2024, 2025])

    if len(df) == 0:
        print("No data to process!")
        return

    # Fill remaining NaN values (from lag/rolling operations on first few games)
    print(f"\nCleaning data...")
    print(f"  Before: {len(df)} rows")

    # Fill numeric columns with appropriate defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            if 'win' in col or 'pct' in col:
                df[col] = df[col].fillna(0.5)
            elif 'streak' in col:
                df[col] = df[col].fillna(0)
            elif 'lag' in col:
                df[col] = df[col].fillna(df[col].mean() if df[col].notna().any() else 0)
            else:
                df[col] = df[col].fillna(df[col].mean() if df[col].notna().any() else 0)

    # Drop any remaining rows with NaN in critical columns only
    critical_cols = ['game_id', 'points_scored', 'points_allowed', 'season', 'week']
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
    print(f"  After: {len(df)} rows")

    # Save to CSV
    output_file = 'nfl_training_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total games: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Weeks: {df['week'].min()} - {df['week'].max()}")
    print(f"Features: {len(df.columns)}")
    print(f"\nFeature columns:")
    for col in sorted(df.columns):
        print(f"  - {col}")

    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("Ready for Deep-Eagle training")
    print("=" * 80)


if __name__ == '__main__':
    main()
