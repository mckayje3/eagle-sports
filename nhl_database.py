"""
NHL Database Setup and Management
Similar structure to NBA/NFL databases with hockey-specific stats.
"""
import sqlite3
from timezone_utils import utc_to_eastern_date


def create_nhl_database(db_path='nhl_games.db'):
    """Create NHL database with required tables"""
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
            game_date_eastern TEXT,
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
            overtime INTEGER DEFAULT 0,
            shootout INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (home_team_id) REFERENCES teams (team_id),
            FOREIGN KEY (away_team_id) REFERENCES teams (team_id),
            FOREIGN KEY (winner_team_id) REFERENCES teams (team_id),
            UNIQUE(game_id)
        )
    ''')

    # Team game stats table (hockey specific)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_game_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            goals INTEGER,
            assists INTEGER,
            shots INTEGER,
            shots_on_goal INTEGER,
            blocked_shots INTEGER,
            hits INTEGER,
            giveaways INTEGER,
            takeaways INTEGER,
            faceoffs_won INTEGER,
            faceoffs_total INTEGER,
            faceoff_pct REAL,
            powerplay_goals INTEGER,
            powerplay_opportunities INTEGER,
            powerplay_pct REAL,
            shorthanded_goals INTEGER,
            penalty_minutes INTEGER,
            saves INTEGER,
            save_pct REAL,
            goals_against INTEGER,
            FOREIGN KEY (game_id) REFERENCES games (game_id),
            FOREIGN KEY (team_id) REFERENCES teams (team_id),
            UNIQUE(game_id, team_id)
        )
    ''')

    # Unified odds and predictions table
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
    print(f"NHL database created at {db_path}")
    return db_path


class NHLDatabase:
    """NHL Database helper class"""

    def __init__(self, db_path='nhl_games.db'):
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

    def get_team_by_id(self, team_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT name, abbreviation, display_name FROM teams WHERE team_id = ?', (team_id,))
        row = cursor.fetchone()
        return {'name': row[0], 'abbreviation': row[1], 'display_name': row[2]} if row else None

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
             home_score, away_score, winner_team_id, venue_name, venue_city, venue_state, overtime, shootout)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            game_data.get('venue_state'),
            game_data.get('overtime', 0),
            game_data.get('shootout', 0)
        ))
        self.conn.commit()

    def insert_team_stats(self, game_id, team_id, stats):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO team_game_stats
            (game_id, team_id, goals, assists, shots, shots_on_goal, blocked_shots, hits,
             giveaways, takeaways, faceoffs_won, faceoffs_total, faceoff_pct,
             powerplay_goals, powerplay_opportunities, powerplay_pct, shorthanded_goals,
             penalty_minutes, saves, save_pct, goals_against)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_id,
            team_id,
            stats.get('goals'),
            stats.get('assists'),
            stats.get('shots'),
            stats.get('shots_on_goal'),
            stats.get('blocked_shots'),
            stats.get('hits'),
            stats.get('giveaways'),
            stats.get('takeaways'),
            stats.get('faceoffs_won'),
            stats.get('faceoffs_total'),
            stats.get('faceoff_pct'),
            stats.get('powerplay_goals'),
            stats.get('powerplay_opportunities'),
            stats.get('powerplay_pct'),
            stats.get('shorthanded_goals'),
            stats.get('penalty_minutes'),
            stats.get('saves'),
            stats.get('save_pct'),
            stats.get('goals_against')
        ))
        self.conn.commit()

    def insert_odds(self, game_id, spread, total, source='ESPN'):
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

    def update_prediction(self, game_id, home_score, away_score, confidence=0.85):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO odds_and_predictions (game_id, predicted_home_score, predicted_away_score, confidence, prediction_created)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(game_id) DO UPDATE SET
                predicted_home_score = excluded.predicted_home_score,
                predicted_away_score = excluded.predicted_away_score,
                confidence = excluded.confidence,
                prediction_created = excluded.prediction_created
        ''', (game_id, home_score, away_score, confidence))
        self.conn.commit()


if __name__ == '__main__':
    create_nhl_database()
