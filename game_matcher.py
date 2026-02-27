"""
Match games between different odds sources and ESPN database
Handles team name variations and finds correct game_id
"""
import sqlite3
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import re
from difflib import SequenceMatcher


class GameMatcher:
    """Match games from odds sources to ESPN game IDs"""

    # Map VegasInsider abbreviations to ESPN team names
    TEAM_NAME_MAPPING = {
        # MAC
        'OHIO': 'Ohio',
        'UMass': 'UMass',
        'BGSU': 'Bowling Green',
        'WMU': 'Western Michigan',
        'CMU': 'Central Michigan',
        'NIU': 'Northern Illinois',
        'BUFF': 'Buffalo',
        'AKRON': 'Akron',
        'KENT': 'Kent State',
        'EMU': 'Eastern Michigan',
        'TOL': 'Toledo',
        'MIA-OH': 'Miami (OH)',
        'MRSH': 'Marshall',

        # Big Ten
        'OSU': 'Ohio State',
        'MICH': 'Michigan',
        'PSU': 'Penn State',
        'WISC': 'Wisconsin',
        'IOWA': 'Iowa',
        'NW': 'Northwestern',
        'ILL': 'Illinois',
        'IND': 'Indiana',
        'MINN': 'Minnesota',
        'MD': 'Maryland',
        'RUTG': 'Rutgers',
        'MSU': 'Michigan State',
        'NEB': 'Nebraska',
        'PUR': 'Purdue',

        # SEC
        'ALA': 'Alabama',
        'UGA': 'Georgia',
        'LSU': 'LSU',
        'FLA': 'Florida',
        'TENN': 'Tennessee',
        'AUB': 'Auburn',
        'TAMU': 'Texas A&M',
        'ARK': 'Arkansas',
        'MISS': 'Ole Miss',
        'MSST': 'Mississippi State',
        'SC': 'South Carolina',
        'UK': 'Kentucky',
        'MIZZ': 'Missouri',
        'VAN': 'Vanderbilt',

        # ACC
        'FSU': 'Florida State',
        'CLEM': 'Clemson',
        'UNC': 'North Carolina',
        'NCST': 'NC State',
        'MIA': 'Miami',
        'VT': 'Virginia Tech',
        'UVA': 'Virginia',
        'PITT': 'Pittsburgh',
        'LOU': 'Louisville',
        'WAKE': 'Wake Forest',
        'BC': 'Boston College',
        'SYR': 'Syracuse',
        'GT': 'Georgia Tech',
        'DUKE': 'Duke',

        # Big 12
        'TEX': 'Texas',
        'OU': 'Oklahoma',
        'OKST': 'Oklahoma State',
        'KSU': 'Kansas State',
        'KU': 'Kansas',
        'BU': 'Baylor',
        'TCU': 'TCU',
        'TTU': 'Texas Tech',
        'WVU': 'West Virginia',
        'ISU': 'Iowa State',
        'UCF': 'UCF',
        'CIN': 'Cincinnati',
        'HOU': 'Houston',
        'BYU': 'BYU',
        'ZONA': 'Arizona',
        'ARI': 'Arizona',
        'ASU': 'Arizona State',
        'COLO': 'Colorado',
        'UTAH': 'Utah',

        # Pac-12 (remaining)
        'ORE': 'Oregon',
        'WASH': 'Washington',
        'USC': 'USC',
        'UCLA': 'UCLA',
        'CAL': 'California',
        'STAN': 'Stanford',
        'ORST': 'Oregon State',
        'WSU': 'Washington State',

        # Group of 5
        'BOISE': 'Boise State',
        'SDSU': 'San Diego State',
        'FRES': 'Fresno State',
        'SJSU': 'San José State',
        'NEV': 'Nevada',
        'UNM': 'New Mexico',
        'USU': 'Utah State',
        'UNLV': 'UNLV',
        'WYO': 'Wyoming',
        'CSU': 'Colorado State',
        'AFA': 'Air Force',
        'HAW': 'Hawai\'i',

        'TULN': 'Tulane',
        'SMU': 'SMU',
        'MEM': 'Memphis',
        'USF': 'South Florida',
        'ECU': 'East Carolina',
        'TUSL': 'Tulsa',
        'NAVY': 'Navy',
        'ARMY': 'Army',
        'TEMPLE': 'Temple',
        'UAB': 'UAB',
        'UTSA': 'UTSA',
        'UNT': 'North Texas',
        'RICE': 'Rice',
        'FAU': 'Florida Atlantic',
        'FIU': 'FIU',
        'MTSU': 'Middle Tennessee',
        'WKU': 'Western Kentucky',
        'ODU': 'Old Dominion',
        'LT': 'Louisiana Tech',
        'CHAR': 'Charlotte',
        'APST': 'Appalachian State',
        'CCU': 'Coastal Carolina',
        'GASO': 'Georgia Southern',
        'ULL': 'Louisiana',
        'TXST': 'Texas State',
        'USM': 'Southern Miss',
        'USA': 'South Alabama',
        'ARST': 'Arkansas State',
        'ULM': 'Louisiana Monroe',
        'TROY': 'Troy',
        'JMU': 'James Madison',
        'UCONN': 'UConn',
        'UMASS': 'UMass',

        # Add more as needed

        # NFL Teams (for The Odds API matching)
        'Kansas City Chiefs': 'Kansas City Chiefs',
        'Buffalo Bills': 'Buffalo Bills',
        'Philadelphia Eagles': 'Philadelphia Eagles',
        'San Francisco 49ers': 'San Francisco 49ers',
        'Dallas Cowboys': 'Dallas Cowboys',
        'Baltimore Ravens': 'Baltimore Ravens',
        'Detroit Lions': 'Detroit Lions',
        'Miami Dolphins': 'Miami Dolphins',
        'Jacksonville Jaguars': 'Jacksonville Jaguars',
        'Cleveland Browns': 'Cleveland Browns',
        'Cincinnati Bengals': 'Cincinnati Bengals',
        'Los Angeles Chargers': 'Los Angeles Chargers',
        'Los Angeles Rams': 'Los Angeles Rams',
        'Las Vegas Raiders': 'Las Vegas Raiders',
        'Seattle Seahawks': 'Seattle Seahawks',
        'Pittsburgh Steelers': 'Pittsburgh Steelers',
        'Denver Broncos': 'Denver Broncos',
        'Minnesota Vikings': 'Minnesota Vikings',
        'Green Bay Packers': 'Green Bay Packers',
        'Chicago Bears': 'Chicago Bears',
        'New York Giants': 'New York Giants',
        'New York Jets': 'New York Jets',
        'Tennessee Titans': 'Tennessee Titans',
        'Houston Texans': 'Houston Texans',
        'Indianapolis Colts': 'Indianapolis Colts',
        'New England Patriots': 'New England Patriots',
        'Washington Commanders': 'Washington Commanders',
        'Atlanta Falcons': 'Atlanta Falcons',
        'New Orleans Saints': 'New Orleans Saints',
        'Tampa Bay Buccaneers': 'Tampa Bay Buccaneers',
        'Carolina Panthers': 'Carolina Panthers',
        'Arizona Cardinals': 'Arizona Cardinals',
    }

    def __init__(self, db_path: str = 'cfb_games.db'):
        self.db_path = db_path

    def normalize_team_name(self, team_name: str) -> str:
        """Normalize team name for comparison"""
        # Remove common suffixes
        name = team_name.strip()
        name = re.sub(r'\s+(Crimson Tide|Buckeyes|Wolverines|Tigers|Bulldogs|Wildcats|Cardinals|Bears|Cowboys|Red Raiders|Longhorns|Sooners|Seminoles|Hurricanes|Fighting Irish|Trojans|Bruins|Golden Bears|Sun Devils|Utes|Buffaloes|Ducks|Huskies|Cougars|Beavers)$', '', name, flags=re.IGNORECASE)
        name = name.strip()
        return name

    def lookup_team_id(self, team_name: str, season: int = None) -> Optional[int]:
        """
        Find ESPN team_id by name

        Args:
            team_name: Team name (from VegasInsider, The Odds API, etc.)
            season: Optional season to filter (some teams change conferences)

        Returns:
            team_id if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # First try the mapping
        mapped_name = self.TEAM_NAME_MAPPING.get(team_name)
        if mapped_name:
            team_name = mapped_name

        # Normalize the name
        normalized = self.normalize_team_name(team_name)

        # Try exact match first
        cursor.execute('''
            SELECT team_id FROM teams
            WHERE name = ? OR display_name = ? OR abbreviation = ?
        ''', (team_name, team_name, team_name))

        result = cursor.fetchone()
        if result:
            conn.close()
            return result['team_id']

        # Try normalized match
        cursor.execute('''
            SELECT team_id, name, display_name FROM teams
        ''')

        teams = cursor.fetchall()
        best_match = None
        best_score = 0

        for team in teams:
            team_names = [
                team['name'],
                team['display_name'],
                self.normalize_team_name(team['name']),
                self.normalize_team_name(team['display_name'])
            ]

            for tn in team_names:
                if not tn:
                    continue

                # Check for substring match
                if normalized.lower() in tn.lower() or tn.lower() in normalized.lower():
                    conn.close()
                    return team['team_id']

                # Use fuzzy matching
                score = SequenceMatcher(None, normalized.lower(), tn.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = team['team_id']

        conn.close()

        # Return best match if confidence is high enough
        if best_score > 0.8:
            return best_match

        return None

    def find_game_by_teams_and_date(
        self,
        home_team: str,
        away_team: str,
        game_date: str = None,
        week: int = None,
        season: int = None
    ) -> Optional[int]:
        """
        Find ESPN game_id by matching teams and date

        Args:
            home_team: Home team name (any format)
            away_team: Away team name (any format)
            game_date: Game date (ISO format or close approximation)
            week: Week number (helps narrow down)
            season: Season year

        Returns:
            game_id if found, None otherwise
        """
        # Get team IDs
        home_id = self.lookup_team_id(home_team, season)
        away_id = self.lookup_team_id(away_team, season)

        if not home_id or not away_id:
            print(f"Could not find team IDs: {home_team}={home_id}, {away_team}={away_id}")
            return None

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query
        query = '''
            SELECT game_id, date, week, season
            FROM games
            WHERE home_team_id = ? AND away_team_id = ?
        '''
        params = [home_id, away_id]

        if season:
            query += ' AND season = ?'
            params.append(season)

        if week:
            query += ' AND week = ?'
            params.append(week)

        # If we have a date, filter by date range (within 3 days)
        if game_date:
            try:
                # Parse date
                if 'T' in game_date:
                    date_obj = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
                else:
                    date_obj = datetime.strptime(game_date, '%Y-%m-%d')

                # Create date range (3 days before/after)
                start_date = (date_obj - timedelta(days=3)).strftime('%Y-%m-%d')
                end_date = (date_obj + timedelta(days=3)).strftime('%Y-%m-%d')

                query += ' AND date BETWEEN ? AND ?'
                params.extend([start_date, end_date])
            except:
                pass

        cursor.execute(query, params)
        result = cursor.fetchone()
        conn.close()

        if result:
            return result['game_id']

        return None

    def match_vegas_insider_game(self, vi_game: Dict) -> Optional[int]:
        """
        Match a VegasInsider game to ESPN game_id

        Args:
            vi_game: Dictionary with 'home_team', 'away_team' from VI

        Returns:
            ESPN game_id if matched, None otherwise
        """
        home = vi_game.get('home_team')
        away = vi_game.get('away_team')

        if not home or not away:
            return None

        # Current season (or from context)
        current_season = datetime.now().year
        if datetime.now().month < 8:
            current_season -= 1

        # Try to find the game
        # Most VI games are for upcoming week, so check current/next week
        for week in range(1, 16):
            game_id = self.find_game_by_teams_and_date(
                home_team=home,
                away_team=away,
                season=current_season,
                week=week
            )
            if game_id:
                return game_id

        return None

    def match_odds_api_game(self, odds_game: Dict) -> Optional[int]:
        """
        Match an Odds API game to ESPN game_id

        Args:
            odds_game: Game dict from The Odds API

        Returns:
            ESPN game_id if matched, None otherwise
        """
        home = odds_game.get('home_team')
        away = odds_game.get('away_team')
        commence_time = odds_game.get('commence_time')

        if not home or not away:
            return None

        # Parse season from date
        if commence_time:
            try:
                date_obj = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                season = date_obj.year
                if date_obj.month < 8:
                    season -= 1
            except:
                season = None
        else:
            season = None

        return self.find_game_by_teams_and_date(
            home_team=home,
            away_team=away,
            game_date=commence_time,
            season=season
        )

    def batch_match_games(self, games_list: List[Dict], source: str = 'vegasinsider') -> Dict[str, Optional[int]]:
        """
        Match a batch of games and return mapping

        Args:
            games_list: List of game dicts
            source: 'vegasinsider' or 'oddsapi'

        Returns:
            Dictionary mapping game identifier to ESPN game_id
        """
        results = {}

        for i, game in enumerate(games_list):
            identifier = f"{game.get('away_team')}@{game.get('home_team')}"

            if source == 'vegasinsider':
                game_id = self.match_vegas_insider_game(game)
            elif source == 'oddsapi':
                game_id = self.match_odds_api_game(game)
            else:
                game_id = None

            results[identifier] = game_id

            if game_id:
                print(f"✓ Matched: {identifier} → game_id {game_id}")
            else:
                print(f"✗ No match: {identifier}")

        # Summary
        matched = sum(1 for v in results.values() if v is not None)
        print(f"\nMatched {matched}/{len(games_list)} games")

        return results


if __name__ == '__main__':
    # Example usage
    matcher = GameMatcher()

    # Test team lookup
    print("Testing team lookups:")
    print(f"OHIO → {matcher.lookup_team_id('OHIO')}")
    print(f"OSU → {matcher.lookup_team_id('OSU')}")
    print(f"Alabama → {matcher.lookup_team_id('Alabama')}")

    # Test game matching
    print("\nTesting game matching:")
    game_id = matcher.find_game_by_teams_and_date(
        home_team='OHIO',
        away_team='UMass',
        season=2025
    )
    print(f"UMass @ Ohio (2025) → game_id {game_id}")
