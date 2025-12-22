"""
ESPN College Football Data Scraper
Fetches FBS game data and statistics from ESPN's API
"""
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from cfb_nfl_database import FootballDatabase
from timezone_utils import convert_espn_date


class ESPNScraper:
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

    def __init__(self, db_path: str = 'cfb_games.db'):
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
            week: Week number (1-15 for regular season)
            season: Year (e.g., 2024) - REQUIRED for historical data

        Returns:
            Dictionary containing scoreboard data
        """
        url = f"{self.BASE_URL}/scoreboard"
        params = {'groups': '80', 'limit': '100'}  # Group 80 is FBS

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
        winner_team_id = None
        if home_team.get('winner'):
            winner_team_id = int(home_team.get('team', {}).get('id'))
        elif away_team.get('winner'):
            winner_team_id = int(away_team.get('team', {}).get('id'))

        # Venue information
        venue = competitions.get('venue', {})
        is_dome = 1 if venue.get('indoor') else 0

        # Weather information (only available in some cases)
        weather = competitions.get('weather', {})
        temperature = None
        wind_speed = None
        conditions = None

        if weather:
            # Temperature (convert to integer if present)
            temp = weather.get('temperature')
            if temp:
                try:
                    temperature = int(temp)
                except (ValueError, TypeError):
                    temperature = None

            # Wind speed (extract number from string like "5 MPH")
            wind = weather.get('wind')
            if wind:
                try:
                    # Extract just the number
                    wind_speed = int(''.join(filter(str.isdigit, str(wind))))
                except (ValueError, TypeError):
                    wind_speed = None

            # Conditions (text description)
            conditions = weather.get('displayValue') or weather.get('conditionId')

        # Broadcast information
        broadcast = competitions.get('broadcasts', [{}])[0]
        broadcast_network = broadcast.get('names', [None])[0] if broadcast.get('names') else None

        # Keep raw UTC date - let database derive game_date_eastern
        date_str_utc = event.get('date')

        # Get week from season info
        week = event.get('week', {}).get('number')

        return {
            'game_id': int(game_id) if game_id else None,
            'season': season,
            'week': week,
            'date': date_str_utc,  # Keep UTC for proper timezone conversion
            'neutral_site': 1 if competitions.get('neutralSite') else 0,
            'conference_game': 1 if competitions.get('conferenceCompetition') else 0,
            'completed': 1 if event.get('status', {}).get('type', {}).get('completed') else 0,
            'home_team_id': int(home_team.get('team', {}).get('id')) if home_team.get('team', {}).get('id') else None,
            'away_team_id': int(away_team.get('team', {}).get('id')) if away_team.get('team', {}).get('id') else None,
            'home_score': home_score,
            'away_score': away_score,
            'winner_team_id': winner_team_id,
            'attendance': int(competitions.get('attendance')) if competitions.get('attendance') else None,
            'venue_name': venue.get('fullName'),
            'venue_city': venue.get('address', {}).get('city'),
            'venue_state': venue.get('address', {}).get('state'),
            'broadcast_network': broadcast_network,
            'temperature': temperature,
            'wind_speed': wind_speed,
            'conditions': conditions,
            'is_dome': is_dome
        }

    def parse_team_stats(self, game_id: int, team_id: int, stats_list: List[Dict]) -> Dict:
        """Parse team statistics from game details"""
        stats_dict = {
            'game_id': game_id,
            'team_id': team_id
        }

        # Map ESPN stat names to our database fields
        stat_mapping = {
            'totalYards': 'total_yards',
            'netPassingYards': 'passing_yards',
            'rushingYards': 'rushing_yards',
            'completionAttempts': 'passing_comp_att',
            'rushingAttempts': 'rushing_attempts',
            'turnovers': 'turnovers',
            'fumblesLost': 'fumbles_lost',
            'interceptions': 'interceptions_thrown',
            'possessionTime': 'possession_time',
            'firstDowns': 'first_downs',
            'thirdDownEff': 'third_down_eff',
            'fourthDownEff': 'fourth_down_eff',
            'totalPenaltiesYards': 'penalties_yards',
            'yardsPerPlay': 'yards_per_play'
        }

        for stat in stats_list:
            stat_name = stat.get('name')
            stat_value = stat.get('displayValue')

            if stat_name in stat_mapping:
                db_field = stat_mapping[stat_name]

                # Handle special cases
                if stat_name == 'completionAttempts':
                    # Format is "X-Y" (completions-attempts)
                    if '-' in str(stat_value):
                        comp, att = stat_value.split('-')
                        stats_dict['passing_completions'] = int(comp)
                        stats_dict['passing_attempts'] = int(att)
                elif stat_name == 'thirdDownEff':
                    # Format is "X-Y" (conversions-attempts)
                    if '-' in str(stat_value):
                        conv, att = stat_value.split('-')
                        stats_dict['third_down_conversions'] = int(conv)
                        stats_dict['third_down_attempts'] = int(att)
                elif stat_name == 'fourthDownEff':
                    # Format is "X-Y" (conversions-attempts)
                    if '-' in str(stat_value):
                        conv, att = stat_value.split('-')
                        stats_dict['fourth_down_conversions'] = int(conv)
                        stats_dict['fourth_down_attempts'] = int(att)
                elif stat_name == 'totalPenaltiesYards':
                    # Format is "X-Y" (penalties-yards)
                    if '-' in str(stat_value):
                        pens, yards = stat_value.split('-')
                        stats_dict['penalties'] = int(pens)
                        stats_dict['penalty_yards'] = int(yards)
                elif stat_name == 'possessionTime':
                    stats_dict[db_field] = stat_value
                else:
                    # Try to convert to int
                    try:
                        stats_dict[db_field] = int(stat_value)
                    except (ValueError, TypeError):
                        stats_dict[db_field] = stat_value

        return stats_dict

    def scrape_week(self, season: int, week: int, season_type: int = 2):
        """
        Scrape all games for a specific week

        Args:
            season: Year (e.g., 2024, 2025)
            week: Week number
            season_type: 1=preseason, 2=regular season, 3=postseason
        """
        print(f"Scraping {season} Week {week}...")

        scoreboard = self.fetch_scoreboard(season_type=season_type, week=week, season=season)
        events = scoreboard.get('events', [])

        print(f"Found {len(events)} games")

        self.db.connect()

        for event in events:
            try:
                # Parse and save teams
                competitions = event.get('competitions', [{}])[0]
                competitors = competitions.get('competitors', [])

                for competitor in competitors:
                    team_data = self.parse_team_data(competitor.get('team', {}))
                    if team_data.get('team_id'):
                        self.db.insert_or_update_team(team_data)

                # Parse and save game
                game_data = self.parse_game_data(event, season)
                if game_data.get('game_id'):
                    self.db.insert_or_update_game(game_data)

                    # If game is completed, fetch detailed stats
                    if game_data.get('completed'):
                        print(f"  Fetching stats for game {game_data['game_id']}...")
                        time.sleep(0.5)  # Be nice to ESPN's servers

                        game_details = self.fetch_game_details(str(game_data['game_id']))
                        self.process_game_stats(game_data['game_id'], game_details)

            except Exception as e:
                print(f"  Error processing event: {e}")
                continue

        self.db.close()
        print(f"Week {week} complete!\n")

    def process_game_stats(self, game_id: int, game_details: Dict):
        """Process and save detailed game statistics"""
        box_score = game_details.get('boxscore', {})
        teams = box_score.get('teams', [])

        for team_stats in teams:
            team_id = int(team_stats.get('team', {}).get('id'))
            stats = team_stats.get('statistics', [])

            if stats:
                stats_data = self.parse_team_stats(game_id, team_id, stats)
                self.db.insert_or_update_team_stats(stats_data)

        # Update weather data if available in game details
        self.update_weather_data(game_id, game_details)

        # Process drive data if available
        self.process_drive_data(game_id, game_details)

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

        # Known domed/indoor/retractable roof stadiums (CFB)
        indoor_stadiums = [
            'Caesars Superdome',
            'Mercedes-Benz Stadium',
            'AT&T Stadium',
            'Lucas Oil Stadium',
            'Ford Field',
            'U.S. Bank Stadium',
            'State Farm Stadium',
            'Alamodome',
            'Carrier Dome',  # Syracuse
            'Georgia Dome',
            'Edward Jones Dome',
            'NRG Stadium',
            'Allegiant Stadium',
            'BC Place',  # BC Lions (CFB games sometimes)
            'Idaho Central Credit Union Arena'  # Boise State basketball arena
        ]

        # Determine dome status
        if any(dome in venue_name for dome in indoor_stadiums):
            is_dome = 1
        elif grass is True:  # Grass field = definitely outdoor
            is_dome = 0
        # If grass is False (turf) but not in known domes list, assume outdoor
        # Many outdoor stadiums use turf (e.g., Boise State, Eastern Michigan)
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

    def scrape_season(self, season: int, start_week: int = 1, end_week: int = 15, season_type: int = 2):
        """
        Scrape an entire season

        Args:
            season: Year (e.g., 2024, 2025)
            start_week: Starting week number
            end_week: Ending week number
            season_type: 1=preseason, 2=regular season, 3=postseason
        """
        print(f"Starting scrape for {season} season (weeks {start_week}-{end_week})")

        for week in range(start_week, end_week + 1):
            self.scrape_week(season, week, season_type)
            time.sleep(1)  # Rate limiting

        print(f"Season {season} scrape complete!")

    def scrape_date_range(self, start_date: str, end_date: str):
        """
        Scrape games within a date range

        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
        """
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        current = start

        self.db.connect()

        while current <= end:
            date_str = current.strftime('%Y%m%d')
            print(f"Scraping games for {date_str}...")

            scoreboard = self.fetch_scoreboard(date=date_str)
            events = scoreboard.get('events', [])

            if events:
                print(f"  Found {len(events)} games")

                for event in events:
                    try:
                        # Get season from event
                        season = event.get('season', {}).get('year', datetime.now().year)

                        # Process teams and game
                        competitions = event.get('competitions', [{}])[0]
                        competitors = competitions.get('competitors', [])

                        for competitor in competitors:
                            team_data = self.parse_team_data(competitor.get('team', {}))
                            if team_data.get('team_id'):
                                self.db.insert_or_update_team(team_data)

                        game_data = self.parse_game_data(event, season)
                        if game_data.get('game_id'):
                            self.db.insert_or_update_game(game_data)

                            if game_data.get('completed'):
                                time.sleep(0.5)
                                game_details = self.fetch_game_details(str(game_data['game_id']))
                                self.process_game_stats(game_data['game_id'], game_details)

                    except Exception as e:
                        print(f"  Error processing event: {e}")
                        continue

            current += timedelta(days=1)
            time.sleep(0.5)

        self.db.close()
        print("Date range scrape complete!")


if __name__ == '__main__':
    # Example usage
    scraper = ESPNScraper()

    # Scrape current week (example)
    # scraper.scrape_week(season=2025, week=12)

    # Scrape entire season (example)
    # scraper.scrape_season(season=2024, start_week=1, end_week=15)

    # Scrape specific date range (example)
    # scraper.scrape_date_range('20240901', '20241130')

    print("ESPN Scraper ready. Import and use the ESPNScraper class to fetch data.")
