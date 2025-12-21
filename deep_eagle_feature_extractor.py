"""
Deep Eagle Feature Extractor
Comprehensive feature extraction from all database tables:
- games
- team_game_stats
- odds_and_predictions
- spread_movement, total_movement, moneyline_movement columns in game_odds
- drives

IMPORTANT - DATA LEAKAGE PREVENTION:
=====================================
This extractor outputs TWO types of columns:

1. TARGET COLUMNS (actual game results - DO NOT use as training features):
   - home_score, away_score: The actual final scores (prediction targets)
   - point_spread: Calculated from actual scores (away_score - home_score)
   - total_points: Calculated from actual scores (home_score + away_score)
   - home_win: Binary indicator of actual winner

2. FEATURE COLUMNS (safe to use for training):
   - Historical stats (home_hist_*, away_hist_*): From PREVIOUS games only (week < current)
   - Drive stats (home_drive_*, away_drive_*): From PREVIOUS games only (week < current)
   - Vegas odds (odds_*): Pre-game betting lines for THIS game
   - Game context: neutral_site, conference_game, temperature, etc.
   - Differentials: Calculated from historical stats

The target columns are included in the output CSV for convenience (so you can
evaluate model performance), but train_deep_eagle_<sport>.py EXCLUDES them from features.

When adding new features, ask: "Would I know this BEFORE the game starts?"
- YES → Safe to use as a feature
- NO  → This is a target/leakage column, must be excluded from training

Usage:
    py deep_eagle_feature_extractor.py <sport> <db_path> <season> [output_path]
    py deep_eagle_feature_extractor.py cfb cfb_games.db 2025 cfb_2025_features.csv
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime


class DeepEagleFeatureExtractor:
    """Extract comprehensive features for Deep Eagle models"""

    def __init__(self, db_path, sport='cfb'):
        self.db_path = db_path
        self.sport = sport
        self.conn = sqlite3.connect(db_path)

    def extract_season_features(self, season):
        """
        Extract complete feature set for a season

        Returns:
            DataFrame with all features for each game
        """
        print(f"\n{'='*80}")
        print(f"EXTRACTING DEEP EAGLE FEATURES - {self.sport.upper()} {season}")
        print('='*80)

        # Get all completed games
        games_df = pd.read_sql_query('''
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
                g.conference_game,
                g.temperature,
                g.wind_speed,
                g.is_dome
            FROM games g
            WHERE g.season = ? AND g.completed = 1
            ORDER BY g.week, g.game_id
        ''', self.conn, params=(season,))

        print(f"\nFound {len(games_df)} completed games in {season}")

        all_features = []

        for idx, game in games_df.iterrows():
            if (idx + 1) % 100 == 0:
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
        """Extract all features for a single game"""
        # Determine max week based on sport (NFL=22 with playoffs, CFB=15 regular + bowls)
        max_week = 22 if self.sport == 'nfl' else 18

        features = {
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'week_normalized': game['week'] / max_week,  # Normalized to 0-1 scale
            'home_team_id': game['home_team_id'],
            'away_team_id': game['away_team_id'],
        }

        # TARGET VARIABLES - actual game results
        # WARNING: These are included for evaluation purposes only!
        # They must be EXCLUDED from training features in train_deep_eagle_<sport>.py
        # Using these as features would be DATA LEAKAGE (model sees the answer)
        features['home_score'] = game['home_score']
        features['away_score'] = game['away_score']
        features['point_spread'] = game['away_score'] - game['home_score']  # LEAKAGE if used as feature!
        features['total_points'] = game['home_score'] + game['away_score']  # LEAKAGE if used as feature!
        features['home_win'] = 1 if game['home_score'] > game['away_score'] else 0  # LEAKAGE if used as feature!

        # Game context
        features['neutral_site'] = game['neutral_site']
        features['conference_game'] = game['conference_game']
        features['temperature'] = game['temperature'] if pd.notna(game['temperature']) else 0
        features['wind_speed'] = game['wind_speed'] if pd.notna(game['wind_speed']) else 0
        features['is_dome'] = game['is_dome'] if pd.notna(game['is_dome']) else 0

        # NOTE: We intentionally do NOT use current game stats (_get_team_game_stats)
        # because those stats (yards, turnovers, etc.) are not available before the game.
        # This would cause data leakage during training - the model would "cheat" by seeing
        # the actual game results. Instead, we only use historical stats (prior games).

        # Get historical team stats (season up to this game)
        home_hist = self._get_historical_stats(game['home_team_id'], game['season'], game['week'])
        away_hist = self._get_historical_stats(game['away_team_id'], game['season'], game['week'])

        for key, value in home_hist.items():
            features[f'home_hist_{key}'] = value
        for key, value in away_hist.items():
            features[f'away_hist_{key}'] = value

        # Get odds data
        odds = self._get_odds_data(game['game_id'])
        for key, value in odds.items():
            features[f'odds_{key}'] = value

        # Get drive data (NEW!)
        home_drive_stats = self._get_drive_stats(game['home_team_id'], game['season'], game['week'])
        away_drive_stats = self._get_drive_stats(game['away_team_id'], game['season'], game['week'])

        for key, value in home_drive_stats.items():
            features[f'home_drive_{key}'] = value
        for key, value in away_drive_stats.items():
            features[f'away_drive_{key}'] = value

        # Calculate matchup differentials
        features['ppg_differential'] = away_hist.get('ppg', 0) - home_hist.get('ppg', 0)
        features['papg_differential'] = away_hist.get('papg', 0) - home_hist.get('papg', 0)
        features['win_pct_differential'] = away_hist.get('win_pct', 0) - home_hist.get('win_pct', 0)
        features['ppd_differential'] = away_drive_stats.get('ppd', 0) - home_drive_stats.get('ppd', 0)
        features['scoring_pct_differential'] = away_drive_stats.get('scoring_pct', 0) - home_drive_stats.get('scoring_pct', 0)

        # NEW: Home/away venue-adjusted differentials
        # For home team: use their home_ppg vs away team's away_ppg (how they perform in this venue)
        # This is the KEY feature that captures home field advantage
        features['venue_ppg_differential'] = away_hist.get('away_ppg', 0) - home_hist.get('home_ppg', 0)
        features['venue_win_pct_differential'] = away_hist.get('away_win_pct', 0) - home_hist.get('home_win_pct', 0)

        # Combined home field advantage signal
        # Positive = home team has venue advantage
        features['combined_home_advantage'] = (
            home_hist.get('home_away_ppg_diff', 0) +  # Home team's home boost
            away_hist.get('home_away_ppg_diff', 0)    # Away team usually drops on road (positive value here means they're better at home too)
        ) / 2  # Average the two teams' home/away differentials

        return features

    def _get_team_game_stats(self, game_id, team_id):
        """Get team game stats for this specific game"""
        query = '''
            SELECT
                points, total_yards, passing_yards, rushing_yards,
                passing_completions, passing_attempts, rushing_attempts,
                turnovers, fumbles_lost, interceptions_thrown,
                first_downs, third_down_conversions, third_down_attempts,
                fourth_down_conversions, fourth_down_attempts,
                penalties, penalty_yards, sacks, sack_yards
            FROM team_game_stats
            WHERE game_id = ? AND team_id = ?
        '''

        result = pd.read_sql_query(query, self.conn, params=(game_id, team_id))

        if result.empty:
            return {
                'points': 0, 'total_yards': 0, 'passing_yards': 0, 'rushing_yards': 0,
                'passing_completions': 0, 'passing_attempts': 0, 'rushing_attempts': 0,
                'turnovers': 0, 'fumbles_lost': 0, 'interceptions_thrown': 0,
                'first_downs': 0, 'third_down_conversions': 0, 'third_down_attempts': 0,
                'fourth_down_conversions': 0, 'fourth_down_attempts': 0,
                'penalties': 0, 'penalty_yards': 0, 'sacks': 0, 'sack_yards': 0
            }

        return result.iloc[0].to_dict()

    def _get_historical_stats(self, team_id, season, current_week):
        """Get team's season stats up to (but not including) current week

        Now includes home/away splits to help model understand venue impact.
        """
        # Convert numpy types to Python native types for SQLite compatibility
        team_id = int(team_id)
        season = int(season)
        current_week = int(current_week)

        # Get overall PPG and PAPG from games table
        ppg_query = '''
            SELECT
                COUNT(*) as games_played,
                AVG(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as ppg,
                AVG(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND week < ?
                AND (home_team_id = ? OR away_team_id = ?)
        '''

        ppg_result = pd.read_sql_query(ppg_query, self.conn,
            params=(team_id, team_id, team_id, season, current_week, team_id, team_id))

        # Get HOME-only stats (when this team plays AT HOME)
        home_query = '''
            SELECT
                COUNT(*) as home_games,
                AVG(home_score) as home_ppg,
                AVG(away_score) as home_papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 /
                    NULLIF(COUNT(*), 0) as home_win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND week < ?
                AND home_team_id = ?
        '''
        home_result = pd.read_sql_query(home_query, self.conn,
            params=(team_id, season, current_week, team_id))

        # Get AWAY-only stats (when this team plays ON THE ROAD)
        away_query = '''
            SELECT
                COUNT(*) as away_games,
                AVG(away_score) as away_ppg,
                AVG(home_score) as away_papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 /
                    NULLIF(COUNT(*), 0) as away_win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND week < ?
                AND away_team_id = ?
        '''
        away_result = pd.read_sql_query(away_query, self.conn,
            params=(team_id, season, current_week, team_id))

        # Get other stats from team_game_stats
        stats_query = '''
            SELECT
                AVG(ts.total_yards) as ypg,
                AVG(ts.turnovers) as turnover_pg
            FROM team_game_stats ts
            JOIN games g ON ts.game_id = g.game_id
            WHERE ts.team_id = ?
                AND g.season = ?
                AND g.week < ?
                AND g.completed = 1
        '''

        stats_result = pd.read_sql_query(stats_query, self.conn, params=(team_id, season, current_week))

        # Combine results
        if ppg_result.empty or ppg_result.iloc[0]['games_played'] == 0:
            return {
                'games_played': 0, 'ppg': 0, 'papg': 0, 'ypg': 0,
                'turnover_pg': 0, 'win_pct': 0,
                'home_games': 0, 'home_ppg': 0, 'home_papg': 0, 'home_win_pct': 0,
                'away_games': 0, 'away_ppg': 0, 'away_papg': 0, 'away_win_pct': 0,
                'home_away_ppg_diff': 0
            }

        # Extract home stats
        home_games = home_result.iloc[0]['home_games'] if not home_result.empty else 0
        home_ppg = home_result.iloc[0]['home_ppg'] if not home_result.empty and pd.notna(home_result.iloc[0]['home_ppg']) else 0
        home_papg = home_result.iloc[0]['home_papg'] if not home_result.empty and pd.notna(home_result.iloc[0]['home_papg']) else 0
        home_win_pct = home_result.iloc[0]['home_win_pct'] if not home_result.empty and pd.notna(home_result.iloc[0]['home_win_pct']) else 0

        # Extract away stats
        away_games = away_result.iloc[0]['away_games'] if not away_result.empty else 0
        away_ppg = away_result.iloc[0]['away_ppg'] if not away_result.empty and pd.notna(away_result.iloc[0]['away_ppg']) else 0
        away_papg = away_result.iloc[0]['away_papg'] if not away_result.empty and pd.notna(away_result.iloc[0]['away_papg']) else 0
        away_win_pct = away_result.iloc[0]['away_win_pct'] if not away_result.empty and pd.notna(away_result.iloc[0]['away_win_pct']) else 0

        # Calculate home/away differential (negative = team scores more at home)
        home_away_ppg_diff = away_ppg - home_ppg if home_games > 0 and away_games > 0 else 0

        result = {
            'games_played': ppg_result.iloc[0]['games_played'],
            'ppg': ppg_result.iloc[0]['ppg'] if pd.notna(ppg_result.iloc[0]['ppg']) else 0,
            'papg': ppg_result.iloc[0]['papg'] if pd.notna(ppg_result.iloc[0]['papg']) else 0,
            'win_pct': ppg_result.iloc[0]['win_pct'] if pd.notna(ppg_result.iloc[0]['win_pct']) else 0,
            'ypg': stats_result.iloc[0]['ypg'] if not stats_result.empty and pd.notna(stats_result.iloc[0]['ypg']) else 0,
            'turnover_pg': stats_result.iloc[0]['turnover_pg'] if not stats_result.empty and pd.notna(stats_result.iloc[0]['turnover_pg']) else 0,
            # NEW: Home/away splits
            'home_games': home_games,
            'home_ppg': home_ppg,
            'home_papg': home_papg,
            'home_win_pct': home_win_pct,
            'away_games': away_games,
            'away_ppg': away_ppg,
            'away_papg': away_papg,
            'away_win_pct': away_win_pct,
            'home_away_ppg_diff': home_away_ppg_diff
        }

        return result

    def _get_odds_data(self, game_id):
        """Get betting odds for the game

        Uses 'latest' odds concept to avoid train/test skew:
        - For upcoming games: uses current odds (freshly scraped before prediction)

        This ensures training and prediction see similar data distributions.
        """
        # Get odds from simplified schema
        query = '''
            SELECT
                opening_spread, latest_spread,
                opening_total, latest_total,
                opening_moneyline_home, latest_moneyline_home,
                spread_movement, total_movement, moneyline_movement
            FROM odds_and_predictions
            WHERE game_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
        '''

        result = pd.read_sql_query(query, self.conn, params=(game_id,))

        if result.empty:
            return {
                'opening_spread': 0, 'latest_spread': 0,
                'opening_total': 0, 'latest_total': 0,
                'opening_ml_home': 0, 'latest_ml_home': 0,
                'opening_ml_away': 0, 'latest_ml_away': 0,
                'spread_movement': 0, 'total_movement': 0,
            }

        row = result.iloc[0]

        # Use latest values, fall back to opening
        latest_spread = row['latest_spread'] if pd.notna(row['latest_spread']) else (row['opening_spread'] if pd.notna(row['opening_spread']) else 0)
        latest_total = row['latest_total'] if pd.notna(row['latest_total']) else (row['opening_total'] if pd.notna(row['opening_total']) else 0)
        latest_ml = row['latest_moneyline_home'] if pd.notna(row['latest_moneyline_home']) else (row['opening_moneyline_home'] if pd.notna(row['opening_moneyline_home']) else 0)

        # Calculate movement from opening to latest (NEW: line movement features)
        spread_movement = row['spread_movement'] if pd.notna(row['spread_movement']) else 0
        total_movement = row['total_movement'] if pd.notna(row['total_movement']) else 0

        # If movement columns are 0 but we have both opening and latest, calculate manually
        if spread_movement == 0 and pd.notna(row['opening_spread']) and pd.notna(row['latest_spread']):
            spread_movement = row['latest_spread'] - row['opening_spread']
        if total_movement == 0 and pd.notna(row['opening_total']) and pd.notna(row['latest_total']):
            total_movement = row['latest_total'] - row['opening_total']

        # Threshold-based features: help model focus on significant moves, ignore noise
        # Sport-specific thresholds based on typical spread sizes:
        # - NFL/NBA: 2.0 pts (~35% of avg spread)
        # - CBB: 2.5 pts (~28% of avg spread)
        # - CFB: 4.0 pts (~21% of avg spread, since CFB spreads avg 18.8 pts)
        if self.sport == 'cfb':
            spread_threshold = 4.0
            total_threshold = 3.0
        elif self.sport == 'cbb':
            spread_threshold = 2.5
            total_threshold = 2.0
        else:  # nfl, nba
            spread_threshold = 2.0
            total_threshold = 2.0

        spread_significant = abs(spread_movement) >= spread_threshold
        total_significant = abs(total_movement) >= total_threshold

        return {
            'opening_spread': row['opening_spread'] if pd.notna(row['opening_spread']) else 0,
            'latest_spread': latest_spread,
            'opening_total': row['opening_total'] if pd.notna(row['opening_total']) else 0,
            'latest_total': latest_total,
            'opening_ml_home': row['opening_moneyline_home'] if pd.notna(row['opening_moneyline_home']) else 0,
            'latest_ml_home': latest_ml,
            'opening_ml_away': 0,  # Simplified schema doesn't track away ML separately
            'latest_ml_away': 0,
            'spread_movement': spread_movement,
            'total_movement': total_movement,
            # Magnitude of movement
            'spread_movement_abs': abs(spread_movement),
            'total_movement_abs': abs(total_movement),
            # Threshold features: binary indicators for significant movement
            'spread_movement_significant': 1 if spread_significant else 0,
            'total_movement_significant': 1 if total_significant else 0,
            # Direction ONLY when significant (0 for small moves = ignore)
            # Positive = line moved toward home team (sharps like home)
            # Negative = line moved away from home (sharps like away)
            'spread_movement_sig_direction': spread_movement if spread_significant else 0,
            'total_movement_sig_direction': total_movement if total_significant else 0,
        }

    def _get_drive_stats(self, team_id, season, current_week):
        """Get team's drive efficiency stats up to current week"""
        # Convert numpy types to Python native types for SQLite compatibility
        team_id = int(team_id)
        season = int(season)
        current_week = int(current_week)

        # Offensive drives (team with possession)
        query_off = '''
            SELECT
                COUNT(*) as total_drives,
                AVG(CASE
                    WHEN d.result LIKE '%TD%' THEN 7
                    WHEN d.result LIKE '%FG%' THEN 3
                    WHEN d.result LIKE '%SAFETY%' THEN 2
                    ELSE 0
                END) as ppd,
                AVG(d.yards) as ypd,
                AVG(d.plays) as plays_per_drive,
                AVG(d.time_elapsed_seconds) as seconds_per_drive,
                AVG(d.is_score) as scoring_pct,
                SUM(CASE WHEN d.start_yards_to_endzone <= 20 THEN d.is_score ELSE 0 END) * 1.0 /
                    NULLIF(SUM(CASE WHEN d.start_yards_to_endzone <= 20 THEN 1 ELSE 0 END), 0) as redzone_pct,
                SUM(CASE WHEN d.plays <= 3 AND d.is_score = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as three_and_out_pct,
                SUM(CASE WHEN d.yards >= 20 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as explosive_drive_pct
            FROM drives d
            JOIN games g ON d.game_id = g.game_id
            WHERE d.team_id = ?
                AND g.season = ?
                AND g.week < ?
                AND g.completed = 1
        '''

        result_off = pd.read_sql_query(query_off, self.conn, params=(team_id, season, current_week))

        if result_off.empty or result_off.iloc[0]['total_drives'] == 0 or pd.isna(result_off.iloc[0]['total_drives']):
            return {
                'total_drives': 0,  # Added to match successful query return
                'ppd': 0, 'ypd': 0, 'plays_per_drive': 0, 'seconds_per_drive': 0,
                'scoring_pct': 0, 'redzone_pct': 0, 'three_and_out_pct': 0,
                'explosive_drive_pct': 0, 'def_ppd': 0, 'def_ypd': 0,
                'def_scoring_pct': 0, 'def_three_and_out_forced': 0
            }

        # Convert to dict and replace None with 0
        stats = result_off.iloc[0].to_dict()
        stats = {k: (v if v is not None else 0) for k, v in stats.items()}

        # Defensive stats (opponent drives)
        query_def = '''
            SELECT
                AVG(CASE
                    WHEN d.result LIKE '%TD%' THEN 7
                    WHEN d.result LIKE '%FG%' THEN 3
                    WHEN d.result LIKE '%SAFETY%' THEN 2
                    ELSE 0
                END) as def_ppd,
                AVG(d.yards) as def_ypd,
                AVG(d.is_score) as def_scoring_pct,
                SUM(CASE WHEN d.plays <= 3 AND d.is_score = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as def_three_and_out_forced
            FROM drives d
            JOIN games g ON d.game_id = g.game_id
            WHERE d.team_id != ?
                AND (g.home_team_id = ? OR g.away_team_id = ?)
                AND g.season = ?
                AND g.week < ?
                AND g.completed = 1
        '''

        result_def = pd.read_sql_query(query_def, self.conn, params=(team_id, team_id, team_id, season, current_week))

        if not result_def.empty:
            stats['def_ppd'] = result_def.iloc[0]['def_ppd'] if pd.notna(result_def.iloc[0]['def_ppd']) else 0
            stats['def_ypd'] = result_def.iloc[0]['def_ypd'] if pd.notna(result_def.iloc[0]['def_ypd']) else 0
            stats['def_scoring_pct'] = result_def.iloc[0]['def_scoring_pct'] if pd.notna(result_def.iloc[0]['def_scoring_pct']) else 0
            stats['def_three_and_out_forced'] = result_def.iloc[0]['def_three_and_out_forced'] if pd.notna(result_def.iloc[0]['def_three_and_out_forced']) else 0
        else:
            stats['def_ppd'] = 0
            stats['def_ypd'] = 0
            stats['def_scoring_pct'] = 0
            stats['def_three_and_out_forced'] = 0

        # Replace NaN with 0
        for key in stats:
            if pd.isna(stats[key]):
                stats[key] = 0

        return stats

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

    if len(sys.argv) < 4:
        print("Usage: py deep_eagle_feature_extractor.py <sport> <db_path> <season> [output_path]")
        print("Example: py deep_eagle_feature_extractor.py cfb cfb_games.db 2025 cfb_2025_features.csv")
        sys.exit(1)

    sport = sys.argv[1]
    db_path = sys.argv[2]
    season = int(sys.argv[3])
    output_path = sys.argv[4] if len(sys.argv) > 4 else f"{sport}_{season}_deep_eagle_features.csv"

    extractor = DeepEagleFeatureExtractor(db_path, sport)
    features_df = extractor.extract_season_features(season)
    extractor.save_features(features_df, output_path)
    extractor.close()

    print(f"\n{'='*80}")
    print("FEATURE EXTRACTION COMPLETE!")
    print('='*80)
