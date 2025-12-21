"""
CBB Deep Eagle Feature Extractor
Comprehensive feature extraction for Men's College Basketball predictions

Extracts basketball-specific features:
- Shooting efficiency (FG%, 3PT%, FT%)
- Rebounding (offensive, defensive, total)
- Ball control (assists, turnovers, A/T ratio)
- Tempo and pace metrics
- Home/away performance splits
- Historical matchup stats
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime


class CBBFeatureExtractor:
    """Extract comprehensive features for CBB Deep Eagle models"""

    def __init__(self, db_path='cbb_games.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def extract_season_features(self, season, min_games=3):
        """
        Extract complete feature set for a season

        Args:
            season: Season year (e.g., 2024 for 2023-24 season)
            min_games: Minimum games a team must have played before features are calculated

        Returns:
            DataFrame with all features for each game
        """
        print(f"\n{'='*80}")
        print(f"EXTRACTING CBB DEEP EAGLE FEATURES - {season-1}-{str(season)[-2:]} SEASON")
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
                g.neutral_site,
                g.conference_game
            FROM games g
            WHERE g.season = ? AND g.completed = 1
            ORDER BY g.date, g.game_id
        ''', self.conn, params=(season,))

        print(f"\nFound {len(games_df)} completed games in {season}")

        # Calculate approximate week number from date
        games_df['week'] = self._calculate_week(games_df['date'], season)

        all_features = []
        skipped_early = 0

        for idx, game in games_df.iterrows():
            if (idx + 1) % 500 == 0:
                print(f"  Processing game {idx + 1}/{len(games_df)}...")

            try:
                features = self._extract_game_features(game, min_games)
                if features is not None:
                    all_features.append(features)
                else:
                    skipped_early += 1
            except Exception as e:
                print(f"  Error processing game {game['game_id']}: {e}")
                continue

        features_df = pd.DataFrame(all_features)

        print(f"\n[DONE] Feature extraction complete:")
        print(f"  Games processed: {len(features_df)}")
        print(f"  Games skipped (early season): {skipped_early}")
        print(f"  Total features: {len(features_df.columns)}")

        return features_df

    def _calculate_week(self, dates, season):
        """Calculate approximate week number from date for CBB season"""
        # CBB season starts in early November
        season_start = datetime(season - 1, 11, 1)

        weeks = []
        for date_str in dates:
            game_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
            days_since_start = (game_date - season_start).days
            week = max(1, days_since_start // 7 + 1)
            weeks.append(week)
        return weeks

    def _extract_game_features(self, game, min_games=3):
        """Extract all features for a single game"""
        # Check if both teams have enough games
        home_games = self._get_games_played(game['home_team_id'], game['season'], game['date'])
        away_games = self._get_games_played(game['away_team_id'], game['season'], game['date'])

        if home_games < min_games or away_games < min_games:
            return None

        features = {
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'week_normalized': min(game['week'] / 25, 1.0),  # CBB season ~25 weeks
            'home_team_id': game['home_team_id'],
            'away_team_id': game['away_team_id'],
        }

        # Target variables
        features['home_score'] = game['home_score']
        features['away_score'] = game['away_score']
        features['point_spread'] = game['home_score'] - game['away_score']
        features['total_points'] = game['home_score'] + game['away_score']
        features['home_win'] = 1 if game['home_score'] > game['away_score'] else 0

        # Game context
        features['neutral_site'] = game['neutral_site'] if pd.notna(game['neutral_site']) else 0
        features['conference_game'] = game['conference_game'] if pd.notna(game['conference_game']) else 0

        # Get historical team stats
        home_hist = self._get_historical_stats(game['home_team_id'], game['season'], game['date'])
        away_hist = self._get_historical_stats(game['away_team_id'], game['season'], game['date'])

        for key, value in home_hist.items():
            features[f'home_hist_{key}'] = value
        for key, value in away_hist.items():
            features[f'away_hist_{key}'] = value

        # Get odds data
        odds = self._get_odds_data(game['game_id'])
        for key, value in odds.items():
            features[f'odds_{key}'] = value

        # Calculate matchup differentials
        features['ppg_differential'] = home_hist.get('ppg', 0) - away_hist.get('ppg', 0)
        features['papg_differential'] = home_hist.get('papg', 0) - away_hist.get('papg', 0)
        features['win_pct_differential'] = home_hist.get('win_pct', 0) - away_hist.get('win_pct', 0)

        # Shooting efficiency differentials
        features['fg_pct_differential'] = home_hist.get('fg_pct', 0) - away_hist.get('fg_pct', 0)
        features['three_pct_differential'] = home_hist.get('three_pct', 0) - away_hist.get('three_pct', 0)
        features['ft_pct_differential'] = home_hist.get('ft_pct', 0) - away_hist.get('ft_pct', 0)

        # Rebounding differentials
        features['reb_differential'] = home_hist.get('rpg', 0) - away_hist.get('rpg', 0)
        features['oreb_differential'] = home_hist.get('oreb_pg', 0) - away_hist.get('oreb_pg', 0)

        # Ball control differentials
        features['to_differential'] = home_hist.get('to_pg', 0) - away_hist.get('to_pg', 0)
        features['ast_to_differential'] = home_hist.get('ast_to_ratio', 0) - away_hist.get('ast_to_ratio', 0)

        # Home/away venue-adjusted differentials
        features['venue_ppg_differential'] = home_hist.get('home_ppg', 0) - away_hist.get('away_ppg', 0)
        features['venue_win_pct_differential'] = home_hist.get('home_win_pct', 0) - away_hist.get('away_win_pct', 0)

        # Combined home field advantage signal
        features['combined_home_advantage'] = (
            home_hist.get('home_away_ppg_diff', 0) +
            away_hist.get('home_away_ppg_diff', 0)
        ) / 2

        # Strength of schedule approximation (opponent PPG allowed)
        features['home_sos'] = home_hist.get('opponent_ppg', 0)
        features['away_sos'] = away_hist.get('opponent_ppg', 0)

        return features

    def _get_games_played(self, team_id, season, before_date):
        """Get number of games a team has played before a certain date"""
        query = '''
            SELECT COUNT(*) as games
            FROM games
            WHERE season = ? AND completed = 1
                AND (home_team_id = ? OR away_team_id = ?)
                AND date < ?
        '''
        result = pd.read_sql_query(query, self.conn,
            params=(int(season), int(team_id), int(team_id), before_date))
        return result.iloc[0]['games']

    def _get_historical_stats(self, team_id, season, before_date):
        """Get team's season stats up to (but not including) current date

        Includes shooting splits, rebounding, assists, turnovers, home/away splits
        """
        team_id = int(team_id)
        season = int(season)

        # Get PPG and PAPG from games table
        ppg_query = '''
            SELECT
                COUNT(*) as games_played,
                AVG(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as ppg,
                AVG(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND date < ?
                AND (home_team_id = ? OR away_team_id = ?)
        '''
        ppg_result = pd.read_sql_query(ppg_query, self.conn,
            params=(team_id, team_id, team_id, season, before_date, team_id, team_id))

        # Get HOME-only stats
        home_query = '''
            SELECT
                COUNT(*) as home_games,
                AVG(home_score) as home_ppg,
                AVG(away_score) as home_papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 /
                    NULLIF(COUNT(*), 0) as home_win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND date < ?
                AND home_team_id = ?
        '''
        home_result = pd.read_sql_query(home_query, self.conn,
            params=(team_id, season, before_date, team_id))

        # Get AWAY-only stats
        away_query = '''
            SELECT
                COUNT(*) as away_games,
                AVG(away_score) as away_ppg,
                AVG(home_score) as away_papg,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) * 1.0 /
                    NULLIF(COUNT(*), 0) as away_win_pct
            FROM games
            WHERE season = ? AND completed = 1 AND date < ?
                AND away_team_id = ?
        '''
        away_result = pd.read_sql_query(away_query, self.conn,
            params=(team_id, season, before_date, team_id))

        # Get basketball-specific stats from team_game_stats
        stats_query = '''
            SELECT
                AVG(ts.field_goal_pct) as fg_pct,
                AVG(ts.three_point_pct) as three_pct,
                AVG(ts.free_throw_pct) as ft_pct,
                AVG(ts.total_rebounds) as rpg,
                AVG(ts.offensive_rebounds) as oreb_pg,
                AVG(ts.defensive_rebounds) as dreb_pg,
                AVG(ts.assists) as apg,
                AVG(ts.turnovers) as to_pg,
                AVG(ts.steals) as spg,
                AVG(ts.blocks) as bpg,
                AVG(ts.field_goals_made) as fgm_pg,
                AVG(ts.field_goals_attempted) as fga_pg,
                AVG(ts.three_pointers_made) as tpm_pg,
                AVG(ts.three_pointers_attempted) as tpa_pg,
                AVG(ts.free_throws_made) as ftm_pg,
                AVG(ts.free_throws_attempted) as fta_pg
            FROM team_game_stats ts
            JOIN games g ON ts.game_id = g.game_id
            WHERE ts.team_id = ?
                AND g.season = ?
                AND g.date < ?
                AND g.completed = 1
        '''
        stats_result = pd.read_sql_query(stats_query, self.conn,
            params=(team_id, season, before_date))

        # Get opponent average PPG (rough SOS metric)
        sos_query = '''
            SELECT AVG(opp_ppg) as opponent_ppg FROM (
                SELECT
                    CASE
                        WHEN g.home_team_id = ? THEN
                            (SELECT AVG(home_score) FROM games
                             WHERE (home_team_id = g.away_team_id OR away_team_id = g.away_team_id)
                             AND completed = 1 AND season = ?)
                        ELSE
                            (SELECT AVG(home_score) FROM games
                             WHERE (home_team_id = g.home_team_id OR away_team_id = g.home_team_id)
                             AND completed = 1 AND season = ?)
                    END as opp_ppg
                FROM games g
                WHERE g.season = ? AND g.completed = 1 AND g.date < ?
                    AND (g.home_team_id = ? OR g.away_team_id = ?)
            )
        '''
        sos_result = pd.read_sql_query(sos_query, self.conn,
            params=(team_id, season, season, season, before_date, team_id, team_id))

        # Build result dict
        if ppg_result.empty or ppg_result.iloc[0]['games_played'] == 0:
            return self._empty_stats()

        # Extract home stats
        home_games = int(home_result.iloc[0]['home_games']) if not home_result.empty else 0
        home_ppg = float(home_result.iloc[0]['home_ppg']) if not home_result.empty and pd.notna(home_result.iloc[0]['home_ppg']) else 0
        home_papg = float(home_result.iloc[0]['home_papg']) if not home_result.empty and pd.notna(home_result.iloc[0]['home_papg']) else 0
        home_win_pct = float(home_result.iloc[0]['home_win_pct']) if not home_result.empty and pd.notna(home_result.iloc[0]['home_win_pct']) else 0

        # Extract away stats
        away_games = int(away_result.iloc[0]['away_games']) if not away_result.empty else 0
        away_ppg = float(away_result.iloc[0]['away_ppg']) if not away_result.empty and pd.notna(away_result.iloc[0]['away_ppg']) else 0
        away_papg = float(away_result.iloc[0]['away_papg']) if not away_result.empty and pd.notna(away_result.iloc[0]['away_papg']) else 0
        away_win_pct = float(away_result.iloc[0]['away_win_pct']) if not away_result.empty and pd.notna(away_result.iloc[0]['away_win_pct']) else 0

        # Calculate home/away differential
        home_away_ppg_diff = home_ppg - away_ppg if home_games > 0 and away_games > 0 else 0

        # Extract box score stats
        fg_pct = float(stats_result.iloc[0]['fg_pct']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['fg_pct']) else 0
        three_pct = float(stats_result.iloc[0]['three_pct']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['three_pct']) else 0
        ft_pct = float(stats_result.iloc[0]['ft_pct']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['ft_pct']) else 0
        rpg = float(stats_result.iloc[0]['rpg']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['rpg']) else 0
        oreb_pg = float(stats_result.iloc[0]['oreb_pg']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['oreb_pg']) else 0
        dreb_pg = float(stats_result.iloc[0]['dreb_pg']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['dreb_pg']) else 0
        apg = float(stats_result.iloc[0]['apg']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['apg']) else 0
        to_pg = float(stats_result.iloc[0]['to_pg']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['to_pg']) else 0
        spg = float(stats_result.iloc[0]['spg']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['spg']) else 0
        bpg = float(stats_result.iloc[0]['bpg']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['bpg']) else 0

        # Calculate assist/turnover ratio
        ast_to_ratio = apg / to_pg if to_pg > 0 else apg

        # Calculate true shooting %
        fgm_pg = float(stats_result.iloc[0]['fgm_pg']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['fgm_pg']) else 0
        fga_pg = float(stats_result.iloc[0]['fga_pg']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['fga_pg']) else 0
        fta_pg = float(stats_result.iloc[0]['fta_pg']) if not stats_result.empty and pd.notna(stats_result.iloc[0]['fta_pg']) else 0
        ppg = float(ppg_result.iloc[0]['ppg']) if pd.notna(ppg_result.iloc[0]['ppg']) else 0

        # True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
        true_shooting = 0
        if fga_pg > 0 or fta_pg > 0:
            true_shooting = ppg / (2 * (fga_pg + 0.44 * fta_pg)) if (fga_pg + 0.44 * fta_pg) > 0 else 0

        # Opponent PPG (SOS proxy)
        opponent_ppg = float(sos_result.iloc[0]['opponent_ppg']) if not sos_result.empty and pd.notna(sos_result.iloc[0]['opponent_ppg']) else 0

        result = {
            'games_played': int(ppg_result.iloc[0]['games_played']),
            'ppg': ppg,
            'papg': float(ppg_result.iloc[0]['papg']) if pd.notna(ppg_result.iloc[0]['papg']) else 0,
            'win_pct': float(ppg_result.iloc[0]['win_pct']) if pd.notna(ppg_result.iloc[0]['win_pct']) else 0,
            # Shooting
            'fg_pct': fg_pct,
            'three_pct': three_pct,
            'ft_pct': ft_pct,
            'true_shooting': true_shooting,
            # Rebounding
            'rpg': rpg,
            'oreb_pg': oreb_pg,
            'dreb_pg': dreb_pg,
            # Ball control
            'apg': apg,
            'to_pg': to_pg,
            'ast_to_ratio': ast_to_ratio,
            # Defense
            'spg': spg,
            'bpg': bpg,
            # Home/away splits
            'home_games': home_games,
            'home_ppg': home_ppg,
            'home_papg': home_papg,
            'home_win_pct': home_win_pct,
            'away_games': away_games,
            'away_ppg': away_ppg,
            'away_papg': away_papg,
            'away_win_pct': away_win_pct,
            'home_away_ppg_diff': home_away_ppg_diff,
            # SOS proxy
            'opponent_ppg': opponent_ppg
        }

        return result

    def _empty_stats(self):
        """Return empty stats dict with zeros"""
        return {
            'games_played': 0, 'ppg': 0, 'papg': 0, 'win_pct': 0,
            'fg_pct': 0, 'three_pct': 0, 'ft_pct': 0, 'true_shooting': 0,
            'rpg': 0, 'oreb_pg': 0, 'dreb_pg': 0,
            'apg': 0, 'to_pg': 0, 'ast_to_ratio': 0,
            'spg': 0, 'bpg': 0,
            'home_games': 0, 'home_ppg': 0, 'home_papg': 0, 'home_win_pct': 0,
            'away_games': 0, 'away_ppg': 0, 'away_papg': 0, 'away_win_pct': 0,
            'home_away_ppg_diff': 0,
            'opponent_ppg': 0
        }

    def _get_odds_data(self, game_id):
        """Get betting odds for the game from odds_and_predictions table"""
        query = '''
            SELECT
                opening_spread, latest_spread,
                opening_total, latest_total,
                opening_moneyline_home, latest_moneyline_home,
                opening_moneyline_away, latest_moneyline_away,
                spread_movement, total_movement
            FROM odds_and_predictions
            WHERE game_id = ?
        '''

        result = pd.read_sql_query(query, self.conn, params=(int(game_id),))

        if result.empty:
            return {
                'opening_spread': 0, 'closing_spread': 0,
                'opening_total': 0, 'closing_total': 0,
                'opening_ml_home': 0, 'closing_ml_home': 0,
                'opening_ml_away': 0, 'closing_ml_away': 0,
                'spread_movement': 0, 'total_movement': 0,
                'spread_movement_abs': 0, 'total_movement_abs': 0,
                'spread_movement_significant': 0, 'total_movement_significant': 0,
                'spread_movement_sig_direction': 0, 'total_movement_sig_direction': 0,
            }

        row = result.iloc[0]

        opening_spread = float(row['opening_spread']) if pd.notna(row['opening_spread']) else 0
        latest_spread = float(row['latest_spread']) if pd.notna(row['latest_spread']) else 0
        opening_total = float(row['opening_total']) if pd.notna(row['opening_total']) else 0
        latest_total = float(row['latest_total']) if pd.notna(row['latest_total']) else 0

        # Calculate movement
        spread_movement = float(row['spread_movement']) if pd.notna(row['spread_movement']) else (latest_spread - opening_spread)
        total_movement = float(row['total_movement']) if pd.notna(row['total_movement']) else (latest_total - opening_total)

        # Threshold features: significant movement
        # CBB uses 2.5 pts (~28% of avg spread)
        spread_significant = abs(spread_movement) >= 2.5
        total_significant = abs(total_movement) >= 2.0

        return {
            'opening_spread': opening_spread,
            'closing_spread': latest_spread,
            'opening_total': opening_total,
            'closing_total': latest_total,
            'opening_ml_home': float(row['opening_moneyline_home']) if pd.notna(row['opening_moneyline_home']) else 0,
            'closing_ml_home': float(row['latest_moneyline_home']) if pd.notna(row['latest_moneyline_home']) else 0,
            'opening_ml_away': float(row['opening_moneyline_away']) if pd.notna(row['opening_moneyline_away']) else 0,
            'closing_ml_away': float(row['latest_moneyline_away']) if pd.notna(row['latest_moneyline_away']) else 0,
            # Movement features
            'spread_movement': spread_movement,
            'total_movement': total_movement,
            'spread_movement_abs': abs(spread_movement),
            'total_movement_abs': abs(total_movement),
            # Threshold features: focus on significant moves, ignore noise
            'spread_movement_significant': 1 if spread_significant else 0,
            'total_movement_significant': 1 if total_significant else 0,
            'spread_movement_sig_direction': spread_movement if spread_significant else 0,
            'total_movement_sig_direction': total_movement if total_significant else 0,
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
        print("Usage: py cbb_feature_extractor.py <season> [output_path]")
        print("Example: py cbb_feature_extractor.py 2024 cbb_2024_features.csv")
        sys.exit(1)

    season = int(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"cbb_{season}_deep_eagle_features.csv"

    extractor = CBBFeatureExtractor()
    features_df = extractor.extract_season_features(season)
    extractor.save_features(features_df, output_path)
    extractor.close()

    print(f"\n{'='*80}")
    print("CBB FEATURE EXTRACTION COMPLETE!")
    print('='*80)
