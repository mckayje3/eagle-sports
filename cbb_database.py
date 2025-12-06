"""
Men's College Basketball (CBB) Database Setup and Management
Follows the same structure as NBA/NFL/CFB databases
"""
import sqlite3
import os


def create_cbb_database(db_path='cbb_games.db'):
    """Create CBB database with required tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Teams table - stores all D1 teams
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            abbreviation TEXT,
            display_name TEXT,
            logo_url TEXT,
            color TEXT,
            conference TEXT,
            conference_id INTEGER,
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
            neutral_site INTEGER DEFAULT 0,
            conference_game INTEGER DEFAULT 0,
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
            largest_lead INTEGER,
            FOREIGN KEY (game_id) REFERENCES games (game_id),
            FOREIGN KEY (team_id) REFERENCES teams (team_id),
            UNIQUE(game_id, team_id)
        )
    ''')

    # Game odds table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            opening_spread_home REAL,
            closing_spread_home REAL,
            opening_total REAL,
            closing_total REAL,
            opening_moneyline_home INTEGER,
            opening_moneyline_away INTEGER,
            closing_moneyline_home INTEGER,
            closing_moneyline_away INTEGER,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_id) REFERENCES games (game_id),
            UNIQUE(game_id, source)
        )
    ''')

    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            game_id INTEGER PRIMARY KEY,
            pred_home_score REAL,
            pred_away_score REAL,
            pred_spread REAL,
            pred_total REAL,
            pred_home_win INTEGER,
            pred_home_win_prob REAL,
            confidence REAL,
            vegas_spread REAL,
            vegas_total REAL,
            spread_edge REAL,
            total_edge REAL,
            prediction_date TEXT,
            model_version TEXT
        )
    ''')

    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_season ON games(season)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_completed ON games(completed)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_stats_game ON team_game_stats(game_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_stats_team ON team_game_stats(team_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_odds_game ON game_odds(game_id)')

    conn.commit()
    conn.close()
    print(f"CBB database created at {db_path}")
    return db_path


class CBBDatabase:
    """CBB Database helper class"""

    def __init__(self, db_path='cbb_games.db'):
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
                    conference=None, conference_id=None, logo_url=None, color=None):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO teams
            (team_id, name, abbreviation, display_name, conference, conference_id, logo_url, color)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (team_id, name, abbreviation, display_name, conference, conference_id, logo_url, color))
        self.conn.commit()

    def insert_game(self, game_data):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO games
            (game_id, season, season_type, date, completed, home_team_id, away_team_id,
             home_score, away_score, winner_team_id, venue_name, venue_city, venue_state,
             neutral_site, conference_game, attendance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_data['game_id'],
            game_data['season'],
            game_data.get('season_type', 2),
            game_data['date'],
            game_data.get('completed', 0),
            game_data['home_team_id'],
            game_data['away_team_id'],
            game_data.get('home_score'),
            game_data.get('away_score'),
            game_data.get('winner_team_id'),
            game_data.get('venue_name'),
            game_data.get('venue_city'),
            game_data.get('venue_state'),
            game_data.get('neutral_site', 0),
            game_data.get('conference_game', 0),
            game_data.get('attendance')
        ))
        self.conn.commit()

    def insert_team_stats(self, game_id, team_id, stats):
        """Insert team game statistics"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO team_game_stats
            (game_id, team_id, points, field_goals_made, field_goals_attempted, field_goal_pct,
             three_pointers_made, three_pointers_attempted, three_point_pct,
             free_throws_made, free_throws_attempted, free_throw_pct,
             offensive_rebounds, defensive_rebounds, total_rebounds,
             assists, steals, blocks, turnovers, personal_fouls,
             points_in_paint, fast_break_points, bench_points, largest_lead)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_id, team_id,
            stats.get('points'),
            stats.get('field_goals_made'),
            stats.get('field_goals_attempted'),
            stats.get('field_goal_pct'),
            stats.get('three_pointers_made'),
            stats.get('three_pointers_attempted'),
            stats.get('three_point_pct'),
            stats.get('free_throws_made'),
            stats.get('free_throws_attempted'),
            stats.get('free_throw_pct'),
            stats.get('offensive_rebounds'),
            stats.get('defensive_rebounds'),
            stats.get('total_rebounds'),
            stats.get('assists'),
            stats.get('steals'),
            stats.get('blocks'),
            stats.get('turnovers'),
            stats.get('personal_fouls'),
            stats.get('points_in_paint'),
            stats.get('fast_break_points'),
            stats.get('bench_points'),
            stats.get('largest_lead')
        ))
        self.conn.commit()

    def insert_odds(self, game_id, spread, total, source='ESPN'):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO game_odds
            (game_id, source, closing_spread_home, closing_total, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        ''', (game_id, source, spread, total))
        self.conn.commit()

    def get_games_for_season(self, season, completed_only=False):
        """Get all games for a season"""
        cursor = self.conn.cursor()
        query = 'SELECT game_id FROM games WHERE season = ?'
        if completed_only:
            query += ' AND completed = 1'
        cursor.execute(query, (season,))
        return [row[0] for row in cursor.fetchall()]

    def get_games_needing_stats(self, season):
        """Get completed games that don't have stats yet"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT g.game_id FROM games g
            LEFT JOIN team_game_stats s ON g.game_id = s.game_id
            WHERE g.season = ? AND g.completed = 1 AND s.game_id IS NULL
        ''', (season,))
        return [row[0] for row in cursor.fetchall()]


if __name__ == '__main__':
    create_cbb_database()
