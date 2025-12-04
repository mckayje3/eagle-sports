"""
NBA Deep Eagle Feature Extractor
Adapted for basketball - uses dates instead of weeks, basketball stats

Key differences from football:
- No weeks - uses game dates for calculating rolling stats
- Different stats: FG%, 3P%, rebounds, assists vs yards/turnovers
- No drives data (basketball doesn't have drives)
- All games indoors (no weather)
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class NBAFeatureExtractor:
    """Extract comprehensive features for NBA Deep Eagle models"""

    def __init__(self, db_path='nba_games.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def extract_season_features(self, season):
        """
        Extract complete feature set for a season
        NBA season spans two years (e.g., 2023 = 2023-24 season)

        Returns:
            DataFrame with all features for each game
        """
        print(f"\n{'='*80}")
        print(f"EXTRACTING NBA DEEP EAGLE FEATURES - {season}-{str(season+1)[-2:]} SEASON")
        print('='*80)

        # Get all completed games
        games_df = pd.read_sql_query('''
            SELECT
                g.game_id,
                g.season,
                g.date,
                g.home_team_id,
                g.away_team_id,
                g.home_score,
                g.away_score,
                g.venue_name,
                g.attendance
            FROM games g
            WHERE g.season = ? AND g.completed = 1
            ORDER BY g.date, g.game_id
        ''', self.conn, params=(season,))

        print(f"\nFound {len(games_df)} completed games in {season}-{str(season+1)[-2:]}")

        all_features = []

        for idx, game in games_df.iterrows():
            if (idx + 1) % 200 == 0:
                print(f"  Processing game {idx + 1}/{len(games_df)}...")

            try:
                features = self._extract_game_features(game)
                if features is not None:
                    all_features.append(features)
            except Exception as e:
                print(f"  Error processing game {game['game_id']}: {e}")
                continue

        features_df = pd.DataFrame(all_features)

        print(f"\n[DONE] Feature extraction complete:")
        print(f"  Games processed: {len(features_df)}")
        print(f"  Total features: {len(features_df.columns)}")

        return features_df

    def _extract_game_features(self, game):
        """Extract all features for a single NBA game"""
        # Calculate games into season (0-82 roughly)
        games_into_season = self._get_games_into_season(game['home_team_id'], game['season'], game['date'])
        max_games = 82  # NBA regular season

        features = {
            'game_id': game['game_id'],
            'season': game['season'],
            'date': game['date'],
            'games_into_season': games_into_season,
            'season_progress': min(1.0, games_into_season / max_games),
            'home_team_id': game['home_team_id'],
            'away_team_id': game['away_team_id'],
        }

        # Target variables
        features['home_score'] = game['home_score']
        features['away_score'] = game['away_score']
        features['point_spread'] = game['home_score'] - game['away_score']
        features['total_points'] = game['home_score'] + game['away_score']
        features['home_win'] = 1 if game['home_score'] > game['away_score'] else 0

        # Game context - NBA has no neutral site for regular season
        features['attendance'] = game['attendance'] if pd.notna(game['attendance']) else 0

        # Get historical team stats (season up to this game - avoiding data leakage)
        home_hist = self._get_historical_stats(game['home_team_id'], game['season'], game['date'])
        away_hist = self._get_historical_stats(game['away_team_id'], game['season'], game['date'])

        for key, value in home_hist.items():
            features[f'home_hist_{key}'] = value
        for key, value in away_hist.items():
            features[f'away_hist_{key}'] = value

        # Get recent form (last 10 games)
        home_recent = self._get_recent_form(game['home_team_id'], game['season'], game['date'])
        away_recent = self._get_recent_form(game['away_team_id'], game['season'], game['date'])

        for key, value in home_recent.items():
            features[f'home_recent_{key}'] = value
        for key, value in away_recent.items():
            features[f'away_recent_{key}'] = value

        # Get rest days
        features['home_rest_days'] = self._get_rest_days(game['home_team_id'], game['date'])
        features['away_rest_days'] = self._get_rest_days(game['away_team_id'], game['date'])
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']

        # Back-to-back indicator
        features['home_b2b'] = 1 if features['home_rest_days'] == 0 else 0
        features['away_b2b'] = 1 if features['away_rest_days'] == 0 else 0

        # Get odds data
        odds = self._get_odds_data(game['game_id'])
        for key, value in odds.items():
            features[f'odds_{key}'] = value

        # Calculate matchup differentials
        features['ppg_differential'] = home_hist.get('ppg', 0) - away_hist.get('ppg', 0)
        features['papg_differential'] = home_hist.get('papg', 0) - away_hist.get('papg', 0)
        features['win_pct_differential'] = home_hist.get('win_pct', 0) - away_hist.get('win_pct', 0)
        features['fg_pct_differential'] = home_hist.get('fg_pct', 0) - away_hist.get('fg_pct', 0)
        features['three_pct_differential'] = home_hist.get('three_pct', 0) - away_hist.get('three_pct', 0)
        features['rebound_differential'] = home_hist.get('rebounds_pg', 0) - away_hist.get('rebounds_pg', 0)
        features['assist_differential'] = home_hist.get('assists_pg', 0) - away_hist.get('assists_pg', 0)
        features['turnover_differential'] = home_hist.get('turnovers_pg', 0) - away_hist.get('turnovers_pg', 0)

        # Recent form differentials
        features['recent_ppg_diff'] = home_recent.get('ppg', 0) - away_recent.get('ppg', 0)
        features['recent_win_pct_diff'] = home_recent.get('win_pct', 0) - away_recent.get('win_pct', 0)

        # Season progress-based reliability weighting
        # Early season: trust Vegas more; Late season: trust accumulated stats more
        games_played = home_hist.get('games_played', 0)

        # Stats reliability: 0 at game 1, approaches 1 as season progresses
        features['stats_reliability'] = games_played / (games_played + 10)

        # Vegas reliability: inverse
        features['vegas_reliability'] = 10 / (games_played + 10)

        # Previous season weight: fades out over first 15 games
        prev_season_weight = max(0, 1 - games_played / 15)
        features['prev_season_weight'] = prev_season_weight

        # Get previous season stats for blending
        if prev_season_weight > 0:
            prev_season_stats = self._get_prev_season_stats(
                game['home_team_id'], game['away_team_id'], game['season']
            )
            features['prev_season_ppg_diff'] = prev_season_stats['ppg_diff'] * prev_season_weight
            features['prev_season_win_pct_diff'] = prev_season_stats['win_pct_diff'] * prev_season_weight
        else:
            features['prev_season_ppg_diff'] = 0
            features['prev_season_win_pct_diff'] = 0

        # Weighted differentials
        features['weighted_ppg_diff'] = features['ppg_differential'] * features['stats_reliability']
        features['weighted_vegas_spread'] = features.get('odds_latest_spread', 0) * features['vegas_reliability']

        # Blended PPG diff
        features['blended_ppg_diff'] = features['weighted_ppg_diff'] + features['prev_season_ppg_diff']

        return features

    def _get_games_into_season(self, team_id, season, current_date):
        """Get number of games team has played so far this season"""
        team_id = int(team_id)
        season = int(season)

        query = '''
            SELECT COUNT(*) as games
            FROM games
            WHERE season = ? AND completed = 1
            AND (home_team_id = ? OR away_team_id = ?)
            AND date < ?
        '''
        result = pd.read_sql_query(query, self.conn, params=(season, team_id, team_id, current_date))
        return int(result.iloc[0]['games']) if not result.empty else 0

    def _get_historical_stats(self, team_id, season, current_date):
        """Get team's season stats up to (but not including) current game date"""
        team_id = int(team_id)
        season = int(season)

        # Get PPG and PAPG from games table
        ppg_query = '''
            SELECT
                COUNT(*) as games_played,
                AVG(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as ppg,
                AVG(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as papg,
                SUM(CASE WHEN
                    (home_team_id = ? AND home_score > away_score) OR
                    (away_team_id = ? AND away_score > home_score)
                    THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND date < ?
                AND (home_team_id = ? OR away_team_id = ?)
        '''

        ppg_result = pd.read_sql_query(ppg_query, self.conn,
            params=(team_id, team_id, team_id, team_id, season, current_date, team_id, team_id))

        # Get box score stats from team_game_stats
        stats_query = '''
            SELECT
                AVG(ts.field_goal_pct) as fg_pct,
                AVG(ts.three_point_pct) as three_pct,
                AVG(ts.free_throw_pct) as ft_pct,
                AVG(ts.total_rebounds) as rebounds_pg,
                AVG(ts.offensive_rebounds) as off_rebounds_pg,
                AVG(ts.defensive_rebounds) as def_rebounds_pg,
                AVG(ts.assists) as assists_pg,
                AVG(ts.steals) as steals_pg,
                AVG(ts.blocks) as blocks_pg,
                AVG(ts.turnovers) as turnovers_pg,
                AVG(ts.points_in_paint) as paint_pg,
                AVG(ts.fast_break_points) as fastbreak_pg,
                AVG(ts.bench_points) as bench_pg
            FROM team_game_stats ts
            JOIN games g ON ts.game_id = g.game_id
            WHERE ts.team_id = ?
                AND g.season = ?
                AND g.date < ?
                AND g.completed = 1
        '''

        stats_result = pd.read_sql_query(stats_query, self.conn, params=(team_id, season, current_date))

        # Check if we have current season data
        games_played = ppg_result.iloc[0]['games_played'] if not ppg_result.empty else 0

        # If no current season data, use previous season
        if games_played == 0:
            prev_season = season - 1

            prev_ppg_query = '''
                SELECT
                    COUNT(*) as games_played,
                    AVG(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as ppg,
                    AVG(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as papg,
                    SUM(CASE WHEN
                        (home_team_id = ? AND home_score > away_score) OR
                        (away_team_id = ? AND away_score > home_score)
                        THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_pct
                FROM games
                WHERE season = ? AND completed = 1
                    AND (home_team_id = ? OR away_team_id = ?)
            '''
            ppg_result = pd.read_sql_query(prev_ppg_query, self.conn,
                params=(team_id, team_id, team_id, team_id, prev_season, team_id, team_id))

            prev_stats_query = '''
                SELECT
                    AVG(ts.field_goal_pct) as fg_pct,
                    AVG(ts.three_point_pct) as three_pct,
                    AVG(ts.free_throw_pct) as ft_pct,
                    AVG(ts.total_rebounds) as rebounds_pg,
                    AVG(ts.offensive_rebounds) as off_rebounds_pg,
                    AVG(ts.defensive_rebounds) as def_rebounds_pg,
                    AVG(ts.assists) as assists_pg,
                    AVG(ts.steals) as steals_pg,
                    AVG(ts.blocks) as blocks_pg,
                    AVG(ts.turnovers) as turnovers_pg,
                    AVG(ts.points_in_paint) as paint_pg,
                    AVG(ts.fast_break_points) as fastbreak_pg,
                    AVG(ts.bench_points) as bench_pg
                FROM team_game_stats ts
                JOIN games g ON ts.game_id = g.game_id
                WHERE ts.team_id = ?
                    AND g.season = ?
                    AND g.completed = 1
            '''
            stats_result = pd.read_sql_query(prev_stats_query, self.conn, params=(team_id, prev_season))

        # Combine results
        if ppg_result.empty or ppg_result.iloc[0]['games_played'] == 0:
            return {
                'games_played': 0, 'ppg': 110, 'papg': 110, 'win_pct': 0.5,
                'fg_pct': 45, 'three_pct': 35, 'ft_pct': 75,
                'rebounds_pg': 44, 'off_rebounds_pg': 10, 'def_rebounds_pg': 34,
                'assists_pg': 24, 'steals_pg': 7, 'blocks_pg': 5, 'turnovers_pg': 14,
                'paint_pg': 48, 'fastbreak_pg': 12, 'bench_pg': 35
            }

        result = {
            'games_played': ppg_result.iloc[0]['games_played'],
            'ppg': ppg_result.iloc[0]['ppg'] if pd.notna(ppg_result.iloc[0]['ppg']) else 110,
            'papg': ppg_result.iloc[0]['papg'] if pd.notna(ppg_result.iloc[0]['papg']) else 110,
            'win_pct': ppg_result.iloc[0]['win_pct'] if pd.notna(ppg_result.iloc[0]['win_pct']) else 0.5,
        }

        # Add box score stats
        if not stats_result.empty:
            row = stats_result.iloc[0]
            result['fg_pct'] = row['fg_pct'] if pd.notna(row['fg_pct']) else 45
            result['three_pct'] = row['three_pct'] if pd.notna(row['three_pct']) else 35
            result['ft_pct'] = row['ft_pct'] if pd.notna(row['ft_pct']) else 75
            result['rebounds_pg'] = row['rebounds_pg'] if pd.notna(row['rebounds_pg']) else 44
            result['off_rebounds_pg'] = row['off_rebounds_pg'] if pd.notna(row['off_rebounds_pg']) else 10
            result['def_rebounds_pg'] = row['def_rebounds_pg'] if pd.notna(row['def_rebounds_pg']) else 34
            result['assists_pg'] = row['assists_pg'] if pd.notna(row['assists_pg']) else 24
            result['steals_pg'] = row['steals_pg'] if pd.notna(row['steals_pg']) else 7
            result['blocks_pg'] = row['blocks_pg'] if pd.notna(row['blocks_pg']) else 5
            result['turnovers_pg'] = row['turnovers_pg'] if pd.notna(row['turnovers_pg']) else 14
            result['paint_pg'] = row['paint_pg'] if pd.notna(row['paint_pg']) else 48
            result['fastbreak_pg'] = row['fastbreak_pg'] if pd.notna(row['fastbreak_pg']) else 12
            result['bench_pg'] = row['bench_pg'] if pd.notna(row['bench_pg']) else 35
        else:
            result.update({
                'fg_pct': 45, 'three_pct': 35, 'ft_pct': 75,
                'rebounds_pg': 44, 'off_rebounds_pg': 10, 'def_rebounds_pg': 34,
                'assists_pg': 24, 'steals_pg': 7, 'blocks_pg': 5, 'turnovers_pg': 14,
                'paint_pg': 48, 'fastbreak_pg': 12, 'bench_pg': 35
            })

        return result

    def _get_recent_form(self, team_id, season, current_date, n_games=10):
        """Get team's recent form (last N games)"""
        team_id = int(team_id)
        season = int(season)

        query = '''
            SELECT
                g.home_team_id, g.away_team_id, g.home_score, g.away_score
            FROM games g
            WHERE g.completed = 1 AND g.date < ?
                AND (g.home_team_id = ? OR g.away_team_id = ?)
            ORDER BY g.date DESC
            LIMIT ?
        '''

        result = pd.read_sql_query(query, self.conn, params=(current_date, team_id, team_id, n_games))

        if result.empty:
            return {'ppg': 110, 'papg': 110, 'win_pct': 0.5, 'games': 0}

        wins = 0
        total_scored = 0
        total_allowed = 0

        for _, row in result.iterrows():
            if row['home_team_id'] == team_id:
                total_scored += row['home_score']
                total_allowed += row['away_score']
                if row['home_score'] > row['away_score']:
                    wins += 1
            else:
                total_scored += row['away_score']
                total_allowed += row['home_score']
                if row['away_score'] > row['home_score']:
                    wins += 1

        games = len(result)
        return {
            'ppg': total_scored / games if games > 0 else 110,
            'papg': total_allowed / games if games > 0 else 110,
            'win_pct': wins / games if games > 0 else 0.5,
            'games': games
        }

    def _get_rest_days(self, team_id, current_date):
        """Get number of days since team's last game"""
        team_id = int(team_id)

        query = '''
            SELECT date FROM games
            WHERE completed = 1 AND date < ?
                AND (home_team_id = ? OR away_team_id = ?)
            ORDER BY date DESC
            LIMIT 1
        '''

        result = pd.read_sql_query(query, self.conn, params=(current_date, team_id, team_id))

        if result.empty:
            return 7  # First game of season, assume well-rested

        last_game_date = pd.to_datetime(result.iloc[0]['date'])
        current = pd.to_datetime(current_date)

        # Return days difference, capped at 7
        return min(7, (current - last_game_date).days)

    def _get_prev_season_stats(self, home_team_id, away_team_id, current_season):
        """Get previous season stats for both teams"""
        prev_season = current_season - 1
        home_team_id = int(home_team_id)
        away_team_id = int(away_team_id)

        def get_team_prev_stats(team_id):
            query = '''
                SELECT
                    COUNT(*) as games_played,
                    AVG(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as ppg,
                    SUM(CASE WHEN
                        (home_team_id = ? AND home_score > away_score) OR
                        (away_team_id = ? AND away_score > home_score)
                        THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_pct
                FROM games
                WHERE season = ? AND completed = 1
                    AND (home_team_id = ? OR away_team_id = ?)
            '''
            result = pd.read_sql_query(query, self.conn,
                params=(team_id, team_id, team_id, prev_season, team_id, team_id))

            if result.empty or result.iloc[0]['games_played'] == 0:
                return {'ppg': 110, 'win_pct': 0.5}

            return {
                'ppg': result.iloc[0]['ppg'] if pd.notna(result.iloc[0]['ppg']) else 110,
                'win_pct': result.iloc[0]['win_pct'] if pd.notna(result.iloc[0]['win_pct']) else 0.5
            }

        home_stats = get_team_prev_stats(home_team_id)
        away_stats = get_team_prev_stats(away_team_id)

        return {
            'ppg_diff': home_stats['ppg'] - away_stats['ppg'],
            'win_pct_diff': home_stats['win_pct'] - away_stats['win_pct']
        }

    def _get_odds_data(self, game_id):
        """Get betting odds for the game - uses NBA-specific schema"""
        query = '''
            SELECT
                opening_line, latest_line,
                opening_total_line, latest_total_line,
                closing_spread_home, closing_total,
                opening_spread_home, opening_total
            FROM game_odds
            WHERE game_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
        '''

        result = pd.read_sql_query(query, self.conn, params=(game_id,))

        if result.empty:
            return {
                'opening_spread': 0, 'latest_spread': 0, 'line_movement': 0,
                'opening_total': 220, 'latest_total': 220,
            }

        row = result.iloc[0]

        # Use various column fallbacks for NBA schema
        opening_spread = (
            row['opening_line'] if pd.notna(row.get('opening_line'))
            else row['opening_spread_home'] if pd.notna(row.get('opening_spread_home'))
            else row['closing_spread_home'] if pd.notna(row.get('closing_spread_home'))
            else 0
        )
        latest_spread = (
            row['latest_line'] if pd.notna(row.get('latest_line'))
            else row['closing_spread_home'] if pd.notna(row.get('closing_spread_home'))
            else opening_spread
        )
        # Calculate line movement
        line_movement = (latest_spread - opening_spread) if opening_spread != 0 else 0

        opening_total = (
            row['opening_total_line'] if pd.notna(row.get('opening_total_line'))
            else row['opening_total'] if pd.notna(row.get('opening_total'))
            else row['closing_total'] if pd.notna(row.get('closing_total'))
            else 220
        )
        latest_total = (
            row['latest_total_line'] if pd.notna(row.get('latest_total_line'))
            else row['closing_total'] if pd.notna(row.get('closing_total'))
            else opening_total
        )

        return {
            'opening_spread': opening_spread,
            'latest_spread': latest_spread,
            'line_movement': line_movement,
            'opening_total': opening_total,
            'latest_total': latest_total,
        }

    def save_features(self, features_df, output_path):
        """Save extracted features to CSV"""
        features_df.to_csv(output_path, index=False)
        print(f"\n[SAVED] Features saved to: {output_path}")
        print(f"  Shape: {features_df.shape}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: py nba_feature_extractor.py <season> [output_path]")
        print("Example: py nba_feature_extractor.py 2023 nba_2023_features.csv")
        sys.exit(1)

    season = int(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"nba_{season}_deep_eagle_features.csv"

    extractor = NBAFeatureExtractor()
    features_df = extractor.extract_season_features(season)
    extractor.save_features(features_df, output_path)
    extractor.close()

    print(f"\n{'='*80}")
    print("FEATURE EXTRACTION COMPLETE!")
    print('='*80)
