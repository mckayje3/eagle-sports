"""
CFB Bowl-Specific Feature Engineering
Features that capture unique bowl game dynamics:
- Opt-out risk (elite teams, playoff-miss frustration)
- Motivation asymmetry
- Bowl tier/prestige
- Win count differential
- Conference power differential
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime


# NY6 Bowl names (highest prestige)
NY6_BOWLS = [
    'Rose Bowl', 'Sugar Bowl', 'Orange Bowl', 'Cotton Bowl',
    'Peach Bowl', 'Fiesta Bowl', 'Playoff', 'National Championship',
    'Quarterfinal', 'Semifinal'
]

# Power conferences (more likely to have opt-outs, better depth)
POWER_CONFERENCES = {
    8: 'SEC',
    9: 'Big 12',
    4: 'Big Ten',
    1: 'ACC',
    17: 'Pac-12'  # Was power through 2023
}


class BowlFeatureExtractor:
    def __init__(self, db_path='cfb_games.db'):
        self.db_path = db_path

    def get_team_record_before_bowl(self, team_id, season, bowl_date):
        """Get team's win-loss record before the bowl game"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COUNT(*) as games,
                SUM(CASE WHEN winner_team_id = ? THEN 1 ELSE 0 END) as wins
            FROM games
            WHERE season = ?
            AND completed = 1
            AND date < ?
            AND week < 16
            AND (home_team_id = ? OR away_team_id = ?)
        ''', (team_id, season, bowl_date, team_id, team_id))

        row = cursor.fetchone()
        conn.close()

        games = row[0] or 0
        wins = row[1] or 0
        losses = games - wins

        return {'wins': wins, 'losses': losses, 'win_pct': wins / games if games > 0 else 0.5}

    def get_team_conference(self, team_id):
        """Get team's conference ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT conference FROM teams WHERE team_id = ?', (team_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def is_power_conference(self, conference_id):
        """Check if team is in a power conference"""
        return conference_id in POWER_CONFERENCES

    def get_bowl_tier(self, postseason_type, venue_name=None):
        """
        Classify bowl tier:
        3 = Playoff/NY6
        2 = Mid-tier (notable bowls)
        1 = Lower tier
        """
        if postseason_type in ['Playoff Round 1', 'Quarterfinals', 'Semifinals', 'National Championship']:
            return 3

        if venue_name:
            for ny6 in NY6_BOWLS:
                if ny6.lower() in venue_name.lower():
                    return 3

        # Default to mid-tier for regular bowls
        return 2

    def calculate_opt_out_risk(self, team_id, season, wins, is_power_conf):
        """
        Estimate opt-out risk based on:
        - Win count (more wins = more NFL prospects)
        - Power conference (more NFL talent)
        - Season context (did they miss playoff?)
        """
        risk = 0.0

        # High win count = more elite players who might sit
        if wins >= 10:
            risk += 0.3
        elif wins >= 8:
            risk += 0.15

        # Power conference teams have more NFL talent
        if is_power_conf:
            risk += 0.2

        # If team has 11-12 wins but isn't in playoff, frustration opt-outs
        if wins >= 11:
            risk += 0.2

        return min(risk, 1.0)

    def calculate_motivation_score(self, wins, losses, opponent_wins, bowl_tier, is_power_conf):
        """
        Estimate team motivation:
        - 6-6 team happy to be in a bowl
        - 10-2 team that missed playoff might be checked out
        - NY6 bowl = high motivation
        - Mismatch (12-1 vs 6-6) might lead to lookahead
        """
        motivation = 0.5  # Base

        # Bowl tier matters
        if bowl_tier == 3:
            motivation += 0.3  # High stakes
        elif bowl_tier == 1:
            motivation -= 0.1  # Lower prestige

        # Close to .500 = grateful to be there
        if 6 <= wins <= 7:
            motivation += 0.15

        # Elite team in low-tier bowl = possible letdown
        if wins >= 10 and bowl_tier <= 2:
            motivation -= 0.2

        # Big mismatch (favorite by 3+ wins) = possible lookahead
        win_diff = wins - opponent_wins
        if win_diff >= 4:
            motivation -= 0.15
        elif win_diff <= -4:
            motivation += 0.1  # Underdog motivation

        return max(0, min(1, motivation))

    def extract_bowl_features(self, game_row):
        """
        Extract bowl-specific features for a single game
        """
        features = {}

        home_id = game_row['home_team_id']
        away_id = game_row['away_team_id']
        season = game_row['season']
        game_date = game_row['date']
        postseason_type = game_row.get('postseason_type')
        venue_name = game_row.get('venue_name', '')

        # Get records
        home_record = self.get_team_record_before_bowl(home_id, season, game_date)
        away_record = self.get_team_record_before_bowl(away_id, season, game_date)

        # Win differential
        features['win_differential'] = home_record['wins'] - away_record['wins']
        features['home_wins'] = home_record['wins']
        features['away_wins'] = away_record['wins']
        features['home_win_pct'] = home_record['win_pct']
        features['away_win_pct'] = away_record['win_pct']

        # Conference info
        home_conf = self.get_team_conference(home_id)
        away_conf = self.get_team_conference(away_id)
        home_power = self.is_power_conference(home_conf)
        away_power = self.is_power_conference(away_conf)

        features['home_power_conf'] = 1 if home_power else 0
        features['away_power_conf'] = 1 if away_power else 0
        features['power_conf_matchup'] = 1 if (home_power and away_power) else 0
        features['power_vs_group'] = 1 if (home_power != away_power) else 0

        # Bowl tier
        bowl_tier = self.get_bowl_tier(postseason_type, venue_name)
        features['bowl_tier'] = bowl_tier
        features['is_ny6_playoff'] = 1 if bowl_tier == 3 else 0

        # Opt-out risk
        features['home_opt_out_risk'] = self.calculate_opt_out_risk(
            home_id, season, home_record['wins'], home_power
        )
        features['away_opt_out_risk'] = self.calculate_opt_out_risk(
            away_id, season, away_record['wins'], away_power
        )
        features['opt_out_differential'] = features['home_opt_out_risk'] - features['away_opt_out_risk']

        # Motivation
        features['home_motivation'] = self.calculate_motivation_score(
            home_record['wins'], home_record['losses'],
            away_record['wins'], bowl_tier, home_power
        )
        features['away_motivation'] = self.calculate_motivation_score(
            away_record['wins'], away_record['losses'],
            home_record['wins'], bowl_tier, away_power
        )
        features['motivation_differential'] = features['home_motivation'] - features['away_motivation']

        return features


def build_bowl_training_data():
    """Build training dataset with bowl-specific features"""
    extractor = BowlFeatureExtractor()

    conn = sqlite3.connect('cfb_games.db')

    # Get all completed bowl games
    query = '''
        SELECT
            g.game_id, g.season, g.week, g.date,
            g.home_team_id, g.away_team_id,
            ht.display_name as home_team,
            at.display_name as away_team,
            g.neutral_site, g.conference_game, g.postseason_type,
            g.venue_name,
            g.home_score, g.away_score,
            COALESCE(o.latest_spread, o.opening_spread) as vegas_spread,
            COALESCE(o.latest_total, o.opening_total) as vegas_total
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.week >= 16 AND g.completed = 1
        AND g.season IN (2021, 2022, 2023, 2024)
        ORDER BY g.date
    '''

    games_df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Processing {len(games_df)} bowl games...")

    all_features = []

    for idx, game in games_df.iterrows():
        bowl_features = extractor.extract_bowl_features(game)

        # Add game info
        bowl_features['game_id'] = game['game_id']
        bowl_features['season'] = game['season']
        bowl_features['home_team'] = game['home_team']
        bowl_features['away_team'] = game['away_team']
        bowl_features['home_score'] = game['home_score']
        bowl_features['away_score'] = game['away_score']
        bowl_features['actual_margin'] = game['home_score'] - game['away_score']
        bowl_features['actual_total'] = game['home_score'] + game['away_score']
        bowl_features['vegas_spread'] = game['vegas_spread']
        bowl_features['vegas_total'] = game['vegas_total']

        all_features.append(bowl_features)

    df = pd.DataFrame(all_features)

    print(f"\nBowl features dataset: {len(df)} games")
    print(f"Games with Vegas odds: {df['vegas_spread'].notna().sum()}")

    return df


if __name__ == '__main__':
    df = build_bowl_training_data()

    # Show sample of features
    print("\nSample bowl features:")
    print(df[['home_team', 'away_team', 'home_wins', 'away_wins', 'win_differential',
              'home_opt_out_risk', 'home_motivation', 'bowl_tier', 'actual_margin']].head(10))

    # Show correlation with actual margin
    print("\nCorrelation with actual margin:")
    numeric_cols = ['win_differential', 'home_wins', 'away_wins', 'home_win_pct', 'away_win_pct',
                    'home_opt_out_risk', 'away_opt_out_risk', 'opt_out_differential',
                    'home_motivation', 'away_motivation', 'motivation_differential',
                    'bowl_tier', 'power_conf_matchup', 'power_vs_group']

    correlations = df[numeric_cols + ['actual_margin']].corr()['actual_margin'].sort_values()
    for col, corr in correlations.items():
        if col != 'actual_margin':
            print(f"  {col:30s}: {corr:+.3f}")

    # Save
    df.to_csv('cfb_bowl_features.csv', index=False)
    print(f"\nSaved to cfb_bowl_features.csv")
