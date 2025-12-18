"""
NBA Database Setup and Management
Similar structure to NFL/CFB databases
"""
import sqlite3
import os
from timezone_utils import utc_to_eastern_date


def create_nba_database(db_path='nba_games.db'):
    """Create NBA database with required tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Teams table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            abbreviation TEXT,
            display_name TEXT,
            logo_url TEXT,
            color TEXT,
            conference TEXT,
            division TEXT,
            UNIQUE(team_id)
        )
    ''')

    # Games table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY,
            season INTEGER NOT NULL,
            season_type INTEGER DEFAULT 2,
            date TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            home_team_id INTEGER NOT NULL,
            away_team_id INTEGER NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            winner_team_id INTEGER,
            attendance INTEGER,
            venue_name TEXT,
            venue_city TEXT,
            venue_state TEXT,
            broadcast_network TEXT,
            period INTEGER,
            clock TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (home_team_id) REFERENCES teams (team_id),
            FOREIGN KEY (away_team_id) REFERENCES teams (team_id),
            FOREIGN KEY (winner_team_id) REFERENCES teams (team_id),
            UNIQUE(game_id)
        )
    ''')

    # Team game stats table (basketball specific)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_game_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            points INTEGER,
            field_goals_made INTEGER,
            field_goals_attempted INTEGER,
            field_goal_pct REAL,
            three_pointers_made INTEGER,
            three_pointers_attempted INTEGER,
            three_point_pct REAL,
            free_throws_made INTEGER,
            free_throws_attempted INTEGER,
            free_throw_pct REAL,
            offensive_rebounds INTEGER,
            defensive_rebounds INTEGER,
            total_rebounds INTEGER,
            assists INTEGER,
            steals INTEGER,
            blocks INTEGER,
            turnovers INTEGER,
            personal_fouls INTEGER,
            points_in_paint INTEGER,
            fast_break_points INTEGER,
            bench_points INTEGER,
            FOREIGN KEY (game_id) REFERENCES games (game_id),
            FOREIGN KEY (team_id) REFERENCES teams (team_id),
            UNIQUE(game_id, team_id)
        )
    ''')

    # Unified odds and predictions table (replaces separate game_odds and predictions tables)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS odds_and_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER UNIQUE,
            source TEXT,
            opening_spread REAL,
            latest_spread REAL,
            opening_total REAL,
            latest_total REAL,
            opening_moneyline_home INTEGER,
            latest_moneyline_home INTEGER,
            opening_moneyline_away INTEGER,
            latest_moneyline_away INTEGER,
            spread_movement REAL,
            total_movement REAL,
            moneyline_movement INTEGER,
            odds_updated_at TEXT,
            predicted_home_score REAL,
            predicted_away_score REAL,
            predicted_home_MOE REAL,
            predicted_away_MOE REAL,
            predicted_spread_MOE REAL,
            predicted_total_MOE REAL,
            home_win_probability REAL,
            confidence REAL,
            prediction_created TEXT,
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"NBA database created at {db_path}")
    return db_path


class NBADatabase:
    """NBA Database helper class"""

    def __init__(self, db_path='nba_games.db'):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()

    def get_teams(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT team_id, name, abbreviation FROM teams')
        return {row[1]: row[0] for row in cursor.fetchall()}

    def insert_team(self, team_id, name, abbreviation=None, display_name=None,
                    conference=None, division=None):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO teams
            (team_id, name, abbreviation, display_name, conference, division)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (team_id, name, abbreviation, display_name, conference, division))
        self.conn.commit()

    def insert_game(self, game_data):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO games
            (game_id, season, season_type, date, game_date_eastern, completed, home_team_id, away_team_id,
             home_score, away_score, winner_team_id, venue_name, venue_city, venue_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_data['game_id'],
            game_data['season'],
            game_data.get('season_type', 2),
            game_data['date'],
            utc_to_eastern_date(game_data['date']),
            game_data.get('completed', 0),
            game_data['home_team_id'],
            game_data['away_team_id'],
            game_data.get('home_score'),
            game_data.get('away_score'),
            game_data.get('winner_team_id'),
            game_data.get('venue_name'),
            game_data.get('venue_city'),
            game_data.get('venue_state')
        ))
        self.conn.commit()

    def insert_odds(self, game_id, spread, total, source='TheOddsAPI'):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO odds_and_predictions (game_id, source, latest_spread, latest_total, odds_updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(game_id) DO UPDATE SET
                source = excluded.source,
                latest_spread = excluded.latest_spread,
                latest_total = excluded.latest_total,
                odds_updated_at = excluded.odds_updated_at
        ''', (game_id, source, spread, total))
        self.conn.commit()


if __name__ == '__main__':
    create_nba_database()
