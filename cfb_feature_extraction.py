"""
CFB Deep Eagle Feature Extractor
Feature extraction for machine learning predictions
Extracts and engineers features from the database for model training
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FeatureExtractor:
    """Extract and engineer features for ML models"""

    def __init__(self, db_path='cfb_games.db'):
        self.db_path = db_path

    def get_team_season_stats(self, team_id, season, up_to_week=None):
        """
        Calculate team's season statistics up to a specific week

        Returns:
            Dictionary with season averages and totals
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                g.week,
                g.home_team_id,
                g.away_team_id,
                g.home_score,
                g.away_score,
                ts.team_id,
                ts.total_yards,
                ts.passing_yards,
                ts.rushing_yards,
                ts.turnovers,
                ts.first_downs,
                ts.third_down_conversions,
                ts.third_down_attempts,
                ts.fourth_down_conversions,
                ts.fourth_down_attempts,
                ts.penalties,
                ts.penalty_yards,
                ts.possession_time
            FROM games g
            LEFT JOIN team_game_stats ts ON g.game_id = ts.game_id AND ts.team_id = ?
            WHERE (g.home_team_id = ? OR g.away_team_id = ?)
                AND g.season = ?
                AND g.completed = 1
        """

        params = [team_id, team_id, team_id, season]

        if up_to_week:
            query += " AND g.week < ?"
            params.append(up_to_week)

        query += " ORDER BY g.week"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return None

        # Calculate offensive stats
        stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'points_scored_avg': 0,
            'points_allowed_avg': 0,
            'point_differential_avg': 0,
            'total_yards_avg': 0,
            'passing_yards_avg': 0,
            'rushing_yards_avg': 0,
            'turnovers_avg': 0,
            'first_downs_avg': 0,
            'third_down_pct': 0,
            'penalties_avg': 0,
            'penalty_yards_avg': 0,
            'win_pct': 0
        }

        points_scored = []
        points_allowed = []
        total_yards_list = []
        passing_yards_list = []
        rushing_yards_list = []
        turnovers_list = []
        first_downs_list = []
        third_down_conv = []
        third_down_att = []
        penalties_list = []
        penalty_yards_list = []

        for _, game in df.iterrows():
            is_home = game['home_team_id'] == team_id

            if is_home:
                team_score = game['home_score']
                opp_score = game['away_score']
            else:
                team_score = game['away_score']
                opp_score = game['home_score']

            if pd.notna(team_score) and pd.notna(opp_score):
                stats['games_played'] += 1
                points_scored.append(team_score)
                points_allowed.append(opp_score)

                if team_score > opp_score:
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1

            # Add team stats if available (only for this team's stats)
            if pd.notna(game['team_id']) and game['team_id'] == team_id:
                if pd.notna(game['total_yards']):
                    total_yards_list.append(game['total_yards'])
                if pd.notna(game['passing_yards']):
                    passing_yards_list.append(game['passing_yards'])
                if pd.notna(game['rushing_yards']):
                    rushing_yards_list.append(game['rushing_yards'])
                if pd.notna(game['turnovers']):
                    turnovers_list.append(game['turnovers'])
                if pd.notna(game['first_downs']):
                    first_downs_list.append(game['first_downs'])
                if pd.notna(game['third_down_conversions']):
                    third_down_conv.append(game['third_down_conversions'])
                if pd.notna(game['third_down_attempts']):
                    third_down_att.append(game['third_down_attempts'])
                if pd.notna(game['penalties']):
                    penalties_list.append(game['penalties'])
                if pd.notna(game['penalty_yards']):
                    penalty_yards_list.append(game['penalty_yards'])

        # Calculate averages
        if stats['games_played'] > 0:
            stats['points_scored_avg'] = np.mean(points_scored)
            stats['points_allowed_avg'] = np.mean(points_allowed)
            stats['point_differential_avg'] = stats['points_scored_avg'] - stats['points_allowed_avg']
            stats['win_pct'] = stats['wins'] / stats['games_played']

        if total_yards_list:
            stats['total_yards_avg'] = np.mean(total_yards_list)
        if passing_yards_list:
            stats['passing_yards_avg'] = np.mean(passing_yards_list)
        if rushing_yards_list:
            stats['rushing_yards_avg'] = np.mean(rushing_yards_list)
        if turnovers_list:
            stats['turnovers_avg'] = np.mean(turnovers_list)
        if first_downs_list:
            stats['first_downs_avg'] = np.mean(first_downs_list)
        if third_down_conv and third_down_att:
            stats['third_down_pct'] = sum(third_down_conv) / sum(third_down_att) if sum(third_down_att) > 0 else 0
        if penalties_list:
            stats['penalties_avg'] = np.mean(penalties_list)
        if penalty_yards_list:
            stats['penalty_yards_avg'] = np.mean(penalty_yards_list)

        return stats

    def get_head_to_head_record(self, team1_id, team2_id, seasons=5):
        """Get head-to-head record between two teams over recent seasons"""
        conn = sqlite3.connect(self.db_path)
        current_season = datetime.now().year

        query = """
            SELECT
                home_team_id,
                away_team_id,
                home_score,
                away_score
            FROM games
            WHERE ((home_team_id = ? AND away_team_id = ?)
                OR (home_team_id = ? AND away_team_id = ?))
                AND season >= ?
                AND completed = 1
        """

        df = pd.read_sql_query(query, conn,
                               params=[team1_id, team2_id, team2_id, team1_id,
                                     current_season - seasons])
        conn.close()

        if df.empty:
            return {'games': 0, 'team1_wins': 0, 'team2_wins': 0, 'win_pct': 0.5}

        team1_wins = 0
        total_games = 0

        for _, game in df.iterrows():
            if pd.notna(game['home_score']) and pd.notna(game['away_score']):
                total_games += 1
                if game['home_team_id'] == team1_id:
                    if game['home_score'] > game['away_score']:
                        team1_wins += 1
                else:
                    if game['away_score'] > game['home_score']:
                        team1_wins += 1

        return {
            'games': total_games,
            'team1_wins': team1_wins,
            'team2_wins': total_games - team1_wins,
            'win_pct': team1_wins / total_games if total_games > 0 else 0.5
        }

    def get_recent_form(self, team_id, season, up_to_week, num_games=5):
        """Get team's performance in recent games"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                g.home_team_id,
                g.away_team_id,
                g.home_score,
                g.away_score
            FROM games g
            WHERE (g.home_team_id = ? OR g.away_team_id = ?)
                AND g.season = ?
                AND g.week < ?
                AND g.completed = 1
            ORDER BY g.week DESC
            LIMIT ?
        """

        df = pd.read_sql_query(query, conn,
                               params=[team_id, team_id, season, up_to_week, num_games])
        conn.close()

        if df.empty:
            return {'recent_wins': 0, 'recent_games': 0, 'recent_win_pct': 0}

        wins = 0
        for _, game in df.iterrows():
            is_home = game['home_team_id'] == team_id
            if is_home:
                if game['home_score'] > game['away_score']:
                    wins += 1
            else:
                if game['away_score'] > game['home_score']:
                    wins += 1

        return {
            'recent_wins': wins,
            'recent_games': len(df),
            'recent_win_pct': wins / len(df) if len(df) > 0 else 0
        }

    def get_game_features(self, game_id):
        """
        Extract all features for a single game

        Returns:
            Dictionary with all features for ML model
        """
        conn = sqlite3.connect(self.db_path)

        # Get game details
        query = """
            SELECT
                g.game_id,
                g.season,
                g.week,
                g.home_team_id,
                g.away_team_id,
                g.home_score,
                g.away_score,
                g.neutral_site,
                home.name as home_name,
                away.name as away_name
            FROM games g
            JOIN teams home ON g.home_team_id = home.team_id
            JOIN teams away ON g.away_team_id = away.team_id
            WHERE g.game_id = ?
        """

        game_df = pd.read_sql_query(query, conn, params=[game_id])

        if game_df.empty:
            conn.close()
            return {'game_id': game_id}  # Return minimal features if game not found

        game = game_df.iloc[0]

        # Get betting odds if available
        odds_query = """
            SELECT
                opening_spread,
                latest_spread,
                
                opening_total,
                current_total,
                closing_total
            FROM odds_and_predictions
            WHERE game_id = ?
        """

        odds_df = pd.read_sql_query(odds_query, conn, params=[game_id])
        conn.close()

        # Extract features
        features = {
            'game_id': game_id,
            'season': game['season'],
            'week': game['week'],
            'neutral_site': 1 if game['neutral_site'] else 0,
        }

        # Get team season statistics
        home_stats = self.get_team_season_stats(
            game['home_team_id'],
            game['season'],
            up_to_week=game['week']
        )

        away_stats = self.get_team_season_stats(
            game['away_team_id'],
            game['season'],
            up_to_week=game['week']
        )

        # Add home team stats with prefix
        if home_stats:
            for key, value in home_stats.items():
                features[f'home_{key}'] = value

        # Add away team stats with prefix
        if away_stats:
            for key, value in away_stats.items():
                features[f'away_{key}'] = value

        # Calculate differential features
        if home_stats and away_stats:
            features['win_pct_diff'] = home_stats['win_pct'] - away_stats['win_pct']
            features['points_diff'] = home_stats['points_scored_avg'] - away_stats['points_scored_avg']
            features['yards_diff'] = home_stats['total_yards_avg'] - away_stats['total_yards_avg']

        # Add head-to-head record
        h2h = self.get_head_to_head_record(game['home_team_id'], game['away_team_id'])
        features['h2h_games'] = h2h['games']
        features['h2h_win_pct'] = h2h['win_pct']

        # Add recent form
        home_form = self.get_recent_form(
            game['home_team_id'],
            game['season'],
            game['week']
        )
        away_form = self.get_recent_form(
            game['away_team_id'],
            game['season'],
            game['week']
        )

        features['home_recent_win_pct'] = home_form['recent_win_pct']
        features['away_recent_win_pct'] = away_form['recent_win_pct']
        features['recent_form_diff'] = home_form['recent_win_pct'] - away_form['recent_win_pct']

        # Add betting odds features
        if not odds_df.empty:
            odds = odds_df.iloc[0]
            features['spread'] = odds['latest_spread'] if pd.notna(odds['latest_spread']) else odds['opening_spread']
            features['total'] = odds['closing_total'] if pd.notna(odds['closing_total']) else odds['opening_total']

        # Add target variable (outcome)
        if pd.notna(game['home_score']) and pd.notna(game['away_score']):
            features['home_score'] = game['home_score']
            features['away_score'] = game['away_score']
            features['home_win'] = 1 if game['home_score'] > game['away_score'] else 0
            features['point_differential'] = game['home_score'] - game['away_score']

            # Against the spread
            if 'spread' in features and pd.notna(features['spread']):
                actual_diff = game['home_score'] - game['away_score']
                features['covered_spread'] = 1 if actual_diff + features['spread'] > 0 else 0

        return features

    def extract_all_games(self, season=None, completed_only=True):
        """
        Extract features for all games (or specific season)

        Returns:
            DataFrame with features for all games
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT game_id FROM games WHERE 1=1"
        params = []

        if season:
            query += " AND season = ?"
            params.append(season)

        if completed_only:
            query += " AND completed = 1"

        query += " ORDER BY season, week, game_id"

        game_ids = pd.read_sql_query(query, conn, params=params)
        conn.close()

        print(f"Extracting features for {len(game_ids)} games...")

        all_features = []
        for i, game_id in enumerate(game_ids['game_id'], 1):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(game_ids)} games...")

            features = self.get_game_features(game_id)
            all_features.append(features)

        df = pd.DataFrame(all_features)
        print(f"Feature extraction complete! Shape: {df.shape}")

        return df


if __name__ == '__main__':
    # Example usage
    extractor = FeatureExtractor()

    # Extract features for 2025 season
    print("Extracting features for 2025 season...")
    df = extractor.extract_all_games(season=2025, completed_only=True)

    # Save to CSV
    df.to_csv('ml_features_2025.csv', index=False)
    print(f"\nSaved features to ml_features_2025.csv")
    print(f"Total features: {len(df.columns)}")
    print(f"Total games: {len(df)}")
    print(f"\nFeature columns:")
    print(df.columns.tolist())

    # Show sample
    print(f"\nSample features (first game):")
    print(df.head(1).T)
