"""
ESPN NFL Data Scraper
Fetches NFL game data and statistics from ESPN's API
"""
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from database import FootballDatabase
from timezone_utils import convert_espn_date


class NFLESPNScraper:
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

    def __init__(self, db_path: str = 'nfl_games.db'):
        self.db = FootballDatabase(db_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_scoreboard(self, date: Optional[str] = None, season_type: int = 2, week: Optional[int] = None, season: Optional[int] = None) -> Dict:
        """
        Fetch scoreboard data for a specific date or week

        Args:
            date: Date in YYYYMMDD format (e.g., '20241109')
            season_type: 1=preseason, 2=regular season, 3=postseason
            week: Week number (1-18 for regular season, 1-5 for postseason)
            season: Year (e.g., 2024) - REQUIRED for historical data

        Returns:
            Dictionary containing scoreboard data
        """
        url = f"{self.BASE_URL}/scoreboard"
        params = {'limit': '100'}  # NFL doesn't need groups parameter

        if date:
            params['dates'] = date
        if week:
            params['week'] = week
            params['seasontype'] = season_type
        if season:
            params['year'] = season  # ESPN uses 'year' not 'season'

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching scoreboard: {e}")
            return {}

    def fetch_game_details(self, game_id: str) -> Dict:
        """
        Fetch detailed statistics for a specific game

        Args:
            game_id: ESPN game ID

        Returns:
            Dictionary containing detailed game data and statistics
        """
        url = f"{self.BASE_URL}/summary"
        params = {'event': game_id}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching game {game_id}: {e}")
            return {}

    def parse_team_data(self, team_json: Dict) -> Dict:
        """Parse team information from ESPN JSON"""
        team_id = team_json.get('id')
        return {
            'team_id': int(team_id) if team_id else None,
            'name': team_json.get('name'),
            'abbreviation': team_json.get('abbreviation'),
            'display_name': team_json.get('displayName'),
            'logo_url': team_json.get('logo'),
            'color': team_json.get('color'),
            'conference': team_json.get('conferenceId')
        }

    def parse_game_data(self, event: Dict, season: int) -> Dict:
        """Parse game information from ESPN scoreboard event"""
        game_id = event.get('id')
        competitions = event.get('competitions', [{}])[0]
        competitors = competitions.get('competitors', [])

        # Identify home and away teams
        home_team = next((c for c in competitors if c.get('homeAway') == 'home'), {})
        away_team = next((c for c in competitors if c.get('homeAway') == 'away'), {})

        # Get scores
        home_score = int(home_team.get('score', 0)) if home_team.get('score') else None
        away_score = int(away_team.get('score', 0)) if away_team.get('score') else None

        # Determine winner
        winner_id = None
        if home_score is not None and away_score is not None:
            if home_score > away_score:
                winner_id = home_team.get('team', {}).get('id')
            elif away_score > home_score:
                winner_id = away_team.get('team', {}).get('id')

        # Get venue info
        venue = competitions.get('venue', {})

        # Convert UTC date to Eastern Time
        date_str_utc = event.get('date')
        date_str = convert_espn_date(date_str_utc) or date_str_utc

        return {
            'game_id': int(game_id),
            'season': season,
            'week': event.get('week', {}).get('number'),
            'date': date_str,
            'neutral_site': 1 if competitions.get('neutralSite', False) else 0,
            'conference_game': 1 if competitions.get('conferenceCompetition', False) else 0,
            'completed': 1 if event.get('status', {}).get('type', {}).get('completed', False) else 0,
            'home_team_id': int(home_team.get('team', {}).get('id', 0)),
            'away_team_id': int(away_team.get('team', {}).get('id', 0)),
            'home_score': home_score,
            'away_score': away_score,
            'winner_team_id': int(winner_id) if winner_id else None,
            'attendance': competitions.get('attendance'),
            'venue_name': venue.get('fullName'),
            'venue_city': venue.get('address', {}).get('city'),
            'venue_state': venue.get('address', {}).get('state'),
            'broadcast_network': competitions.get('broadcasts', [{}])[0].get('names', [None])[0] if competitions.get('broadcasts') else None
        }

    def parse_team_stats(self, game_id: int, team_id: int, stats_json: Dict) -> Dict:
        """Parse team statistics from game details"""
        stats = {}

        # Map ESPN stat names to our database field names
        stat_name_mapping = {
            'totalyards': 'total_yards',
            'netpassingyards': 'passing_yards',
            'rushingyards': 'rushing_yards',
            'rushingAttempts': 'rushing_attempts',
            'turnovers': 'turnovers',
            'fumblesLost': 'fumbles_lost',
            'interceptions': 'interceptions_thrown',
            'firstdowns': 'first_downs',
            'possessiontime': 'possession_time',
            'thirddowneff': 'third_down_eff',
            'fourthdowneff': 'fourth_down_eff',
            'completions/passingattempts': 'comp_att',
            'totalpenyards': 'penalties_yards',
        }

        # Parse statistics array
        for stat_category in stats_json:
            raw_name = stat_category.get('name', '')
            name = raw_name.lower().replace(' ', '_')
            value = stat_category.get('displayValue', '0')

            # Map to our field name if exists
            mapped_name = stat_name_mapping.get(name, name)

            # Convert to appropriate type
            try:
                if '-' in value and mapped_name in ['third_down_eff', 'fourth_down_eff', 'comp_att', 'penalties_yards']:
                    # Handle efficiency stats like "7-13" (made-attempts)
                    parts = value.split('-')
                    if len(parts) == 2:
                        stats[f'{mapped_name}_made'] = int(parts[0])
                        stats[f'{mapped_name}_attempts'] = int(parts[1])
                elif '.' in value:
                    stats[mapped_name] = float(value)
                else:
                    stats[mapped_name] = int(value)
            except (ValueError, IndexError):
                stats[mapped_name] = value

        # Map to our schema
        return {
            'game_id': game_id,
            'team_id': team_id,
            'points': stats.get('total_points'),
            'total_yards': stats.get('total_yards'),
            'passing_yards': stats.get('passing_yards'),
            'rushing_yards': stats.get('rushing_yards'),
            'passing_completions': stats.get('comp_att_made'),
            'passing_attempts': stats.get('comp_att_attempts'),
            'rushing_attempts': stats.get('rushing_attempts'),
            'turnovers': stats.get('turnovers'),
            'fumbles_lost': stats.get('fumbles_lost'),
            'interceptions_thrown': stats.get('interceptions_thrown'),
            'first_downs': stats.get('first_downs'),
            'third_down_conversions': stats.get('third_down_eff_made'),
            'third_down_attempts': stats.get('third_down_eff_attempts'),
            'fourth_down_conversions': stats.get('fourth_down_eff_made'),
            'fourth_down_attempts': stats.get('fourth_down_eff_attempts'),
            'penalties': stats.get('penalties_yards_made'),
            'penalty_yards': stats.get('penalties_yards_attempts'),
            'sacks': stats.get('sacks'),
            'sack_yards': stats.get('sack_yards'),
            'possession_time': stats.get('possession_time')
        }

    def process_game_details(self, game_id: str, season: int):
        """Fetch and save detailed game statistics"""
        details = self.fetch_game_details(game_id)

        if not details or 'boxscore' not in details:
            print(f"  No detailed stats available for game {game_id}")
            return

        # Parse team statistics
        teams = details.get('boxscore', {}).get('teams', [])

        for team_data in teams:
            team_id = int(team_data.get('team', {}).get('id', 0))
            stats = team_data.get('statistics', [])

            if team_id and stats:
                team_stats = self.parse_team_stats(int(game_id), team_id, stats)
                self.db.insert_or_update_team_stats(team_stats)

        # Update weather data if available
        self.update_weather_data(int(game_id), details)

        # Process drive data if available
        self.process_drive_data(int(game_id), details)

    def update_weather_data(self, game_id: int, game_details: Dict):
        """Extract and update weather data from game details"""
        game_info = game_details.get('gameInfo', {})
        venue = game_info.get('venue', {})
        weather = game_info.get('weather', {})

        temperature = None
        wind_speed = None
        conditions = None
        is_dome = None

        # Determine if dome/indoor based on venue info
        venue_name = venue.get('fullName', '')
        grass = venue.get('grass')

        # Known domed/indoor/retractable roof stadiums (NFL)
        indoor_stadiums = [
            'Caesars Superdome',
            'Mercedes-Benz Stadium',
            'AT&T Stadium',
            'Lucas Oil Stadium',
            'Ford Field',
            'U.S. Bank Stadium',
            'State Farm Stadium',
            'Allegiant Stadium',
            'NRG Stadium',
            'SoFi Stadium',
            'Lumen Field',  # Partial roof
        ]

        # Determine dome status
        if any(dome in venue_name for dome in indoor_stadiums):
            is_dome = 1
        elif grass is True:  # Grass field = definitely outdoor
            is_dome = 0
        elif grass is False:
            is_dome = 0  # Conservative: assume outdoor unless explicitly in dome list

        # Extract weather if available
        if weather:
            # Temperature
            temp = weather.get('temperature')
            if temp:
                try:
                    temperature = int(temp)
                except (ValueError, TypeError):
                    pass

            # Wind speed
            wind = weather.get('wind')
            if wind:
                try:
                    # Extract just the number
                    wind_speed = int(''.join(filter(str.isdigit, str(wind))))
                except (ValueError, TypeError):
                    pass

            # Conditions
            conditions = weather.get('displayValue') or weather.get('conditionId')

        # Update the game record with weather data
        cursor = self.db.conn.cursor()
        cursor.execute('''
            UPDATE games
            SET temperature = ?,
                wind_speed = ?,
                conditions = ?,
                is_dome = ?
            WHERE game_id = ?
        ''', (temperature, wind_speed, conditions, is_dome, game_id))
        self.db.conn.commit()

    def process_drive_data(self, game_id: int, game_details: Dict):
        """Extract and save drive-by-drive data"""
        drives_data = game_details.get('drives', {})
        previous_drives = drives_data.get('previous', [])

        if not previous_drives:
            return

        drive_number = 0
        for drive in previous_drives:
            drive_number += 1

            # Extract team ID
            team_info = drive.get('team', {})
            team_id = team_info.get('id')
            if not team_id:
                continue

            # Extract start information
            start = drive.get('start', {})
            start_period = start.get('period', {}).get('number')
            start_clock = start.get('clock', {}).get('displayValue')
            start_yard_line = start.get('yardLine')
            start_yards_to_endzone = start.get('yardsToEndzone')

            # Extract end information
            end = drive.get('end', {})
            end_period = end.get('period', {}).get('number')
            end_clock = end.get('clock', {}).get('displayValue')
            end_yard_line = end.get('yardLine')
            end_yards_to_endzone = end.get('yardsToEndzone')

            # Extract drive stats
            plays = drive.get('offensivePlays')
            yards = drive.get('yards')
            time_elapsed_display = drive.get('timeElapsed', {}).get('displayValue')

            # Convert time elapsed to seconds if possible (format: "M:SS")
            time_elapsed_seconds = None
            if time_elapsed_display:
                try:
                    parts = time_elapsed_display.split(':')
                    if len(parts) == 2:
                        minutes, seconds = parts
                        time_elapsed_seconds = int(minutes) * 60 + int(seconds)
                except (ValueError, AttributeError):
                    pass

            # Extract result
            result = drive.get('result')
            is_score = 1 if drive.get('isScore') else 0
            description = drive.get('description')

            # Create drive data dictionary
            drive_data = {
                'game_id': game_id,
                'drive_number': drive_number,
                'team_id': int(team_id),
                'start_period': start_period,
                'start_clock': start_clock,
                'start_yard_line': start_yard_line,
                'start_yards_to_endzone': start_yards_to_endzone,
                'end_period': end_period,
                'end_clock': end_clock,
                'end_yard_line': end_yard_line,
                'end_yards_to_endzone': end_yards_to_endzone,
                'plays': plays,
                'yards': yards,
                'time_elapsed_seconds': time_elapsed_seconds,
                'time_elapsed_display': time_elapsed_display,
                'result': result,
                'is_score': is_score,
                'description': description
            }

            # Insert into database
            self.db.insert_drive(drive_data)

    def scrape_week(self, season: int, week: int, season_type: int = 2):
        """
        Scrape all games for a specific week

        Args:
            season: Season year (e.g., 2024)
            week: Week number
            season_type: 1=preseason, 2=regular season, 3=postseason
        """
        if not self.db.conn:
            self.db.connect()
            self.db.initialize_schema()

        print(f"Scraping {season} Week {week}...")
        scoreboard = self.fetch_scoreboard(season_type=season_type, week=week, season=season)

        events = scoreboard.get('events', [])
        print(f"Found {len(events)} games")

        for event in events:
            # Parse and save game data
            game_data = self.parse_game_data(event, season)

            # Save teams
            competitions = event.get('competitions', [{}])[0]
            for competitor in competitions.get('competitors', []):
                team_data = self.parse_team_data(competitor.get('team', {}))
                # Only save teams with valid ID and name (skip placeholders like TBD)
                if team_data['team_id'] and team_data['name']:
                    self.db.insert_or_update_team(team_data)

            # Save game
            self.db.insert_or_update_game(game_data)

            # Fetch detailed stats if game is completed
            if game_data['completed']:
                game_id = game_data['game_id']
                print(f"  Fetching stats for game {game_id}...")
                self.process_game_details(str(game_id), season)
                time.sleep(0.5)  # Rate limiting

        print(f"Week {week} complete!")

    def scrape_season(self, season: int, start_week: int = 1, end_week: int = 18, season_type: int = 2):
        """
        Scrape multiple weeks of a season

        Args:
            season: Season year
            start_week: Starting week number
            end_week: Ending week number
            season_type: 1=preseason, 2=regular season (18 weeks), 3=postseason (5 weeks)
        """
        if not self.db.conn:
            self.db.connect()
            self.db.initialize_schema()

        print(f"\nStarting scrape for {season} season (weeks {start_week}-{end_week})")

        for week in range(start_week, end_week + 1):
            self.scrape_week(season, week, season_type)
            time.sleep(1)  # Be nice to ESPN's servers

        self.db.conn.commit()
        print(f"\nSeason {season} scrape complete!")


if __name__ == '__main__':
    # Example usage
    print("NFL ESPN Scraper")
    print("="*60)
    print("\nThis scraper fetches NFL games and statistics from ESPN")
    print("\nExample usage:")
    print("  scraper = NFLESPNScraper()")
    print("  scraper.scrape_season(2024, start_week=1, end_week=18)")
