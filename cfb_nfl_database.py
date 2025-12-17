"""
Database setup and schema for NCAA D1 FBS and NFL Football game data
"""
import sqlite3
from datetime import datetime
from typing import Optional


class FootballDatabase:
    def __init__(self, db_path: str = 'cfb_games.db'):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def initialize_schema(self):
        """Create all necessary tables"""
        cursor = self.conn.cursor()

        # Teams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                name TEXT,
                display_name TEXT NOT NULL,
                abbreviation TEXT,
                logo_url TEXT,
                color TEXT,
                conference TEXT,
                UNIQUE(team_id)
            )
        ''')

        # Games table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id INTEGER PRIMARY KEY,
                season INTEGER NOT NULL,
                week INTEGER,
                date TEXT NOT NULL,
                neutral_site INTEGER DEFAULT 0,
                conference_game INTEGER DEFAULT 0,
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
                postseason_type TEXT,
                temperature INTEGER,
                wind_speed INTEGER,
                conditions TEXT,
                is_dome INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (home_team_id) REFERENCES teams (team_id),
                FOREIGN KEY (away_team_id) REFERENCES teams (team_id),
                FOREIGN KEY (winner_team_id) REFERENCES teams (team_id),
                UNIQUE(game_id)
            )
        ''')

        # Team game statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_game_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                points INTEGER,
                total_yards INTEGER,
                passing_yards INTEGER,
                rushing_yards INTEGER,
                passing_completions INTEGER,
                passing_attempts INTEGER,
                rushing_attempts INTEGER,
                turnovers INTEGER,
                fumbles_lost INTEGER,
                interceptions_thrown INTEGER,
                possession_time TEXT,
                first_downs INTEGER,
                third_down_conversions INTEGER,
                third_down_attempts INTEGER,
                fourth_down_conversions INTEGER,
                fourth_down_attempts INTEGER,
                penalties INTEGER,
                penalty_yards INTEGER,
                sacks INTEGER,
                sack_yards INTEGER,
                FOREIGN KEY (game_id) REFERENCES games (game_id),
                FOREIGN KEY (team_id) REFERENCES teams (team_id),
                UNIQUE(game_id, team_id)
            )
        ''')

        # Individual player stats table (optional for future expansion)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_game_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                player_id INTEGER,
                position TEXT,
                passing_completions INTEGER,
                passing_attempts INTEGER,
                passing_yards INTEGER,
                passing_touchdowns INTEGER,
                passing_interceptions INTEGER,
                rushing_attempts INTEGER,
                rushing_yards INTEGER,
                rushing_touchdowns INTEGER,
                receiving_receptions INTEGER,
                receiving_yards INTEGER,
                receiving_touchdowns INTEGER,
                FOREIGN KEY (game_id) REFERENCES games (game_id),
                FOREIGN KEY (team_id) REFERENCES teams (team_id)
            )
        ''')

        # Game odds table (simplified schema)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS odds_and_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                opening_spread REAL,
                latest_spread REAL,
                opening_moneyline_home INTEGER,
                latest_moneyline_home INTEGER,
                opening_moneyline_away INTEGER,
                latest_moneyline_away INTEGER,
                opening_total REAL,
                latest_total REAL,
                spread_movement REAL,
                total_movement REAL,
                moneyline_movement INTEGER,
                home_win_probability REAL,
                confidence REAL,
                predicted_home_score REAL,
                predicted_away_score REAL,
                predicted_home_MOE REAL,
                predicted_away_MOE REAL,
                predicted_spread_MOE REAL,
                predicted_total_MOE REAL,
                prediction_created TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                opening_line_timestamp TEXT,
                FOREIGN KEY (game_id) REFERENCES games (game_id),
                UNIQUE(game_id, source)
            )
        ''')

        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_season ON games(season)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_week ON games(week)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_stats_game ON team_game_stats(game_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_stats_game ON player_game_stats(game_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_and_predictions(game_id)')
        # odds_movement table removed - movement tracked in odds_and_predictions columns

        self.conn.commit()
        print("Database schema initialized successfully")

    def insert_or_update_team(self, team_data: dict):
        """Insert or update team information"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO teams
            (team_id, name, abbreviation, display_name, logo_url, color, conference)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            team_data.get('team_id'),
            team_data.get('name'),
            team_data.get('abbreviation'),
            team_data.get('display_name'),
            team_data.get('logo_url'),
            team_data.get('color'),
            team_data.get('conference')
        ))
        self.conn.commit()

    def insert_or_update_game(self, game_data: dict):
        """Insert or update game information"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO games
            (game_id, season, week, date, neutral_site, conference_game, completed,
             home_team_id, away_team_id, home_score, away_score, winner_team_id,
             attendance, venue_name, venue_city, venue_state, broadcast_network,
             postseason_type, temperature, wind_speed, conditions, is_dome, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            game_data.get('game_id'),
            game_data.get('season'),
            game_data.get('week'),
            game_data.get('date'),
            game_data.get('neutral_site', 0),
            game_data.get('conference_game', 0),
            game_data.get('completed', 0),
            game_data.get('home_team_id'),
            game_data.get('away_team_id'),
            game_data.get('home_score'),
            game_data.get('away_score'),
            game_data.get('winner_team_id'),
            game_data.get('attendance'),
            game_data.get('venue_name'),
            game_data.get('venue_city'),
            game_data.get('venue_state'),
            game_data.get('broadcast_network'),
            game_data.get('postseason_type'),
            game_data.get('temperature'),
            game_data.get('wind_speed'),
            game_data.get('conditions'),
            game_data.get('is_dome', 0)
        ))
        self.conn.commit()

    def insert_or_update_team_stats(self, stats_data: dict):
        """Insert or update team game statistics"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO team_game_stats
            (game_id, team_id, points, total_yards, passing_yards, rushing_yards,
             passing_completions, passing_attempts, rushing_attempts, turnovers,
             fumbles_lost, interceptions_thrown, possession_time, first_downs,
             third_down_conversions, third_down_attempts, fourth_down_conversions,
             fourth_down_attempts, penalties, penalty_yards, sacks, sack_yards)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stats_data.get('game_id'),
            stats_data.get('team_id'),
            stats_data.get('points'),
            stats_data.get('total_yards'),
            stats_data.get('passing_yards'),
            stats_data.get('rushing_yards'),
            stats_data.get('passing_completions'),
            stats_data.get('passing_attempts'),
            stats_data.get('rushing_attempts'),
            stats_data.get('turnovers'),
            stats_data.get('fumbles_lost'),
            stats_data.get('interceptions_thrown'),
            stats_data.get('possession_time'),
            stats_data.get('first_downs'),
            stats_data.get('third_down_conversions'),
            stats_data.get('third_down_attempts'),
            stats_data.get('fourth_down_conversions'),
            stats_data.get('fourth_down_attempts'),
            stats_data.get('penalties'),
            stats_data.get('penalty_yards'),
            stats_data.get('sacks'),
            stats_data.get('sack_yards')
        ))
        self.conn.commit()

    def insert_drive(self, drive_data: dict):
        """Insert drive data"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO drives
            (game_id, drive_number, team_id, start_period, start_clock, start_yard_line,
             start_yards_to_endzone, end_period, end_clock, end_yard_line, end_yards_to_endzone,
             plays, yards, time_elapsed_seconds, time_elapsed_display, result, is_score, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            drive_data.get('game_id'),
            drive_data.get('drive_number'),
            drive_data.get('team_id'),
            drive_data.get('start_period'),
            drive_data.get('start_clock'),
            drive_data.get('start_yard_line'),
            drive_data.get('start_yards_to_endzone'),
            drive_data.get('end_period'),
            drive_data.get('end_clock'),
            drive_data.get('end_yard_line'),
            drive_data.get('end_yards_to_endzone'),
            drive_data.get('plays'),
            drive_data.get('yards'),
            drive_data.get('time_elapsed_seconds'),
            drive_data.get('time_elapsed_display'),
            drive_data.get('result'),
            drive_data.get('is_score', 0),
            drive_data.get('description')
        ))
        self.conn.commit()

    def insert_or_update_odds(self, odds_data: dict):
        """Insert or update game odds using simplified schema"""
        cursor = self.conn.cursor()
        
        # Calculate movements if we have both opening and latest
        spread_movement = None
        total_movement = None
        moneyline_movement = None
        
        opening_spread = odds_data.get('opening_spread')
        latest_spread = odds_data.get('latest_spread')
        opening_total = odds_data.get('opening_total')
        latest_total = odds_data.get('latest_total')
        opening_moneyline_home = odds_data.get('opening_moneyline_home')
        latest_moneyline_home = odds_data.get('latest_moneyline_home')
        opening_moneyline_away = odds_data.get('opening_moneyline_away')
        latest_moneyline_away = odds_data.get('latest_moneyline_away')

        if opening_spread is not None and latest_spread is not None:
            spread_movement = latest_spread - opening_spread
        if opening_total is not None and latest_total is not None:
            total_movement = latest_total - opening_total
        if opening_moneyline_home is not None and latest_moneyline_home is not None:
            moneyline_movement = latest_moneyline_home - opening_moneyline_home

        cursor.execute('''
            INSERT OR REPLACE INTO odds_and_predictions
            (game_id, source, opening_spread, latest_spread,
             opening_moneyline_home, latest_moneyline_home,
             opening_moneyline_away, latest_moneyline_away,
             opening_total, latest_total,
             spread_movement, total_movement, moneyline_movement,
             updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            odds_data.get('game_id'),
            odds_data.get('source', 'ESPN'),
            opening_spread,
            latest_spread,
            opening_moneyline_home,
            latest_moneyline_home,
            opening_moneyline_away,
            latest_moneyline_away,
            opening_total,
            latest_total,
            spread_movement,
            total_movement,
            moneyline_movement
        ))
        self.conn.commit()

    def update_lines(self, game_id: int, spread: float, total: float,
                     moneyline: int = None, is_opening: bool = False, 
                     source: str = 'ESPN'):
        """
        Update betting lines using the simplified schema.

        Args:
            game_id: ESPN game ID
            spread: Home team spread (negative = home favored)
            total: Over/under total points
            moneyline: Home team moneyline
            is_opening: True if this is the opening line (only set once)
            source: Data source name

        The simplified schema has:
        - opening_spread/total/moneyline: Fixed once set
        - latest_spread/total/moneyline: Updated on each scrape
        - spread_movement/total_movement/moneyline_movement: Computed differences
        """
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()

        # Check if game exists and has opening line already
        cursor.execute('''
            SELECT opening_spread, opening_total, opening_moneyline_home, opening_line_timestamp
            FROM odds_and_predictions WHERE game_id = ? AND source = ?
        ''', (game_id, source))
        existing = cursor.fetchone()

        if existing is None:
            # First time seeing this game - insert new row
            cursor.execute('''
                INSERT INTO odds_and_predictions
                (game_id, source, opening_spread, latest_spread,
                 opening_total, latest_total, opening_moneyline_home, latest_moneyline_home,
                 opening_moneyline_away, latest_moneyline_away,
                 spread_movement, total_movement, moneyline_movement,
                 opening_line_timestamp, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, 0, 0, ?, CURRENT_TIMESTAMP)
            ''', (game_id, source, spread, spread, total, total,
                  moneyline, moneyline, timestamp))
        elif existing[0] is None or is_opening:
            # Opening line not set yet, or explicitly setting opening
            cursor.execute('''
                UPDATE odds_and_predictions SET
                    opening_spread = ?,
                    latest_spread = ?,
                    opening_total = ?,
                    latest_total = ?,
                    opening_moneyline_home = ?,
                    latest_moneyline_home = ?,
                    spread_movement = 0,
                    total_movement = 0,
                    moneyline_movement = 0,
                    opening_line_timestamp = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE game_id = ? AND source = ?
            ''', (spread, spread, total, total, moneyline, moneyline,
                  timestamp, game_id, source))
        else:
            # Opening already exists - just update latest and compute movement
            opening_spread = existing[0]
            opening_total = existing[1]
            opening_moneyline_home = existing[2]

            spread_movement = spread - opening_spread if opening_spread else 0
            total_movement = total - opening_total if opening_total else 0
            moneyline_movement = moneyline - opening_moneyline_home if opening_moneyline_home and moneyline else 0

            cursor.execute('''
                UPDATE odds_and_predictions SET
                    latest_spread = ?,
                    latest_total = ?,
                    latest_moneyline_home = ?,
                    spread_movement = ?,
                    total_movement = ?,
                    moneyline_movement = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE game_id = ? AND source = ?
            ''', (spread, total, moneyline, spread_movement, total_movement,
                  moneyline_movement, game_id, source))

        self.conn.commit()

    def set_closing_line(self, game_id: int, source: str = 'ESPN'):
        """
        Mark game as completed - latest values become the closing lines.
        The simplified schema doesn't have separate closing columns;
        latest_spread/total/moneyline serve as closing values once game completes.
        """
        # In simplified schema, no separate closing columns needed
        # Just update the timestamp to mark finalization
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE odds_and_predictions SET updated_at = CURRENT_TIMESTAMP
            WHERE game_id = ? AND source = ?
        ''', (game_id, source))
        self.conn.commit()


    def classify_cfb_postseason_game(self, game_id: int = None, week: int = None,
                                       date: str = None, venue_name: str = None) -> str:
        """
        Classify a CFB postseason game type based on week, date, and venue.

        Returns one of:
        - None: Regular season or conference championship (week < 16)
        - 'Bowl': Non-playoff bowl game
        - 'Playoff Round 1': CFP First Round (typically week 16, Dec 20-22 at campus sites)
        - 'Quarterfinals': CFP Quarterfinals (typically week 18, Dec 31-Jan 1)
        - 'Semifinals': CFP Semifinals (typically week 19, Jan ~10-11)
        - 'National Championship': CFP National Championship (typically week 20, Jan ~20)
        """
        if week is None or week < 16:
            return None

        # CFP First Round campus venues (top 4 seeds host at home)
        # These are the major university stadiums that host CFP first round games
        cfp_first_round_campus_venues = [
            'Notre Dame Stadium', 'Beaver Stadium', 'DKR-Texas Memorial Stadium',
            'Ohio Stadium', 'Michigan Stadium', 'Neyland Stadium', 'Kyle Field',
            'Sanford Stadium', 'Bryant-Denny Stadium', 'Tiger Stadium',
            'Memorial Stadium', 'Darrell K Royal', 'Camp Randall'
        ]

        # Week 20 is always National Championship
        if week == 20:
            return 'National Championship'

        # Week 19 is always Semifinals
        if week == 19:
            return 'Semifinals'

        # Week 18 - Quarterfinals are Dec 31 - Jan 2 at New Year's Six bowl venues
        if week == 18:
            if venue_name:
                # New Year's Six bowls (CFP Quarterfinals)
                ny6_venues = ['Rose Bowl', 'Mercedes-Benz Stadium', 'Caesars Superdome',
                              'State Farm Stadium', 'Hard Rock Stadium', 'AT&T Stadium']
                if any(v.lower() in venue_name.lower() for v in ny6_venues):
                    return 'Quarterfinals'
            return 'Bowl'

        # Week 16-17 - CFP First Round at campus sites OR bowl games
        if week == 16:
            if venue_name:
                # CFP First Round games are ONLY at campus sites
                if any(v.lower() in venue_name.lower() for v in cfp_first_round_campus_venues):
                    return 'Playoff Round 1'
            # If not at a campus site, it's a bowl game
            return 'Bowl'

        # Week 17 - All bowl games (no CFP games in week 17)
        return 'Bowl'

    def update_postseason_types(self, season: int = None):
        """
        Update postseason_type for all postseason games (week >= 16).

        Args:
            season: Optional season to filter, or all seasons if None
        """
        cursor = self.conn.cursor()

        query = '''
            SELECT game_id, week, date, venue_name
            FROM games
            WHERE week >= 16
        '''
        params = []
        if season:
            query += ' AND season = ?'
            params.append(season)

        cursor.execute(query, params)
        games = cursor.fetchall()

        updated = 0
        for game in games:
            game_id, week, date, venue_name = game
            postseason_type = self.classify_cfb_postseason_game(
                game_id=game_id, week=week, date=date, venue_name=venue_name
            )

            if postseason_type:
                cursor.execute('''
                    UPDATE games SET postseason_type = ? WHERE game_id = ?
                ''', (postseason_type, game_id))
                updated += 1

        self.conn.commit()
        print(f"Updated {updated} postseason games")
        return updated


if __name__ == '__main__':
    # Initialize database
    db = FootballDatabase()
    db.connect()
    db.initialize_schema()
    db.close()
    print("Database setup complete!")
