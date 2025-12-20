"""
Backfill CFB Regular Season Odds from TheOddsAPI
Fetches historical odds for 2022, 2023, 2024 seasons
"""
import json
import sqlite3
import requests
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Team name mappings from TheOddsAPI to our database
TEAM_NAME_MAPPINGS = {
    # Common variations
    'Alabama Crimson Tide': 'Crimson Tide',
    'Georgia Bulldogs': 'Bulldogs',
    'Ohio State Buckeyes': 'Buckeyes',
    'Michigan Wolverines': 'Wolverines',
    'Texas Longhorns': 'Longhorns',
    'Oregon Ducks': 'Ducks',
    'Penn State Nittany Lions': 'Nittany Lions',
    'Notre Dame Fighting Irish': 'Fighting Irish',
    'Tennessee Volunteers': 'Volunteers',
    'LSU Tigers': 'Tigers',
    'Florida State Seminoles': 'Seminoles',
    'USC Trojans': 'Trojans',
    'Clemson Tigers': 'Tigers',
    'Oklahoma Sooners': 'Sooners',
    'Texas A&M Aggies': 'Aggies',
    'Florida Gators': 'Gators',
    'Washington Huskies': 'Huskies',
    'Utah Utes': 'Utes',
    'Kansas State Wildcats': 'Wildcats',
    'Ole Miss Rebels': 'Rebels',
    'Miami Hurricanes': 'Hurricanes',
    'NC State Wolfpack': 'Wolfpack',
    'Louisville Cardinals': 'Cardinals',
    'Wisconsin Badgers': 'Badgers',
    'Iowa Hawkeyes': 'Hawkeyes',
    'Missouri Tigers': 'Tigers',
    'Kentucky Wildcats': 'Wildcats',
    'Auburn Tigers': 'Tigers',
    'Oklahoma State Cowboys': 'Cowboys',
    'Kansas Jayhawks': 'Jayhawks',
    'Arkansas Razorbacks': 'Razorbacks',
    'Mississippi State Bulldogs': 'Bulldogs',
    'South Carolina Gamecocks': 'Gamecocks',
    'TCU Horned Frogs': 'Horned Frogs',
    'Baylor Bears': 'Bears',
    'UCLA Bruins': 'Bruins',
    'Arizona State Sun Devils': 'Sun Devils',
    'Arizona Wildcats': 'Wildcats',
    'Colorado Buffaloes': 'Buffaloes',
    'Oregon State Beavers': 'Beavers',
    'Washington State Cougars': 'Cougars',
    'Stanford Cardinal': 'Cardinal',
    'California Golden Bears': 'Golden Bears',
    'Syracuse Orange': 'Orange',
    'Duke Blue Devils': 'Blue Devils',
    'Pittsburgh Panthers': 'Panthers',
    'Virginia Tech Hokies': 'Hokies',
    'Virginia Cavaliers': 'Cavaliers',
    'North Carolina Tar Heels': 'Tar Heels',
    'Wake Forest Demon Deacons': 'Demon Deacons',
    'Boston College Eagles': 'Eagles',
    'Georgia Tech Yellow Jackets': 'Yellow Jackets',
    'Rutgers Scarlet Knights': 'Scarlet Knights',
    'Maryland Terrapins': 'Terrapins',
    'Indiana Hoosiers': 'Hoosiers',
    'Illinois Fighting Illini': 'Fighting Illini',
    'Northwestern Wildcats': 'Wildcats',
    'Purdue Boilermakers': 'Boilermakers',
    'Minnesota Golden Gophers': 'Golden Gophers',
    'Nebraska Cornhuskers': 'Cornhuskers',
    'Michigan State Spartans': 'Spartans',
    'Iowa State Cyclones': 'Cyclones',
    'West Virginia Mountaineers': 'Mountaineers',
    'Texas Tech Red Raiders': 'Red Raiders',
    'Cincinnati Bearcats': 'Bearcats',
    'UCF Knights': 'Knights',
    'Houston Cougars': 'Cougars',
    'BYU Cougars': 'Cougars',
    'SMU Mustangs': 'Mustangs',
    'Memphis Tigers': 'Tigers',
    'Tulane Green Wave': 'Green Wave',
    'UNLV Rebels': 'Rebels',
    'Boise State Broncos': 'Broncos',
    'Fresno State Bulldogs': 'Bulldogs',
    'San Diego State Aztecs': 'Aztecs',
    'Colorado State Rams': 'Rams',
    'Air Force Falcons': 'Falcons',
    'Wyoming Cowboys': 'Cowboys',
    'Nevada Wolf Pack': 'Wolf Pack',
    'San Jose State Spartans': 'Spartans',
    'Hawaii Rainbow Warriors': 'Rainbow Warriors',
    'New Mexico Lobos': 'Lobos',
    'Utah State Aggies': 'Aggies',
    'Liberty Flames': 'Flames',
    'James Madison Dukes': 'Dukes',
    'Jacksonville State Gamecocks': 'Gamecocks',
    'Sam Houston State Bearkats': 'Bearkats',
    'Army Black Knights': 'Black Knights',
    'Navy Midshipmen': 'Midshipmen',
    'Appalachian State Mountaineers': 'Mountaineers',
    'Coastal Carolina Chanticleers': 'Chanticleers',
    'Marshall Thundering Herd': 'Thundering Herd',
    'Old Dominion Monarchs': 'Monarchs',
    'Georgia State Panthers': 'Panthers',
    'Georgia Southern Eagles': 'Eagles',
    'South Alabama Jaguars': 'Jaguars',
    'Troy Trojans': 'Trojans',
    'Louisiana Ragin\' Cajuns': 'Ragin\' Cajuns',
    'Arkansas State Red Wolves': 'Red Wolves',
    'Texas State Bobcats': 'Bobcats',
    'Southern Miss Golden Eagles': 'Golden Eagles',
    'Louisiana Tech Bulldogs': 'Bulldogs',
    'UTSA Roadrunners': 'Roadrunners',
    'Rice Owls': 'Owls',
    'North Texas Mean Green': 'Mean Green',
    'UTEP Miners': 'Miners',
    'Florida Atlantic Owls': 'Owls',
    'Florida International Panthers': 'Panthers',
    'Middle Tennessee Blue Raiders': 'Blue Raiders',
    'Western Kentucky Hilltoppers': 'Hilltoppers',
    'Charlotte 49ers': '49ers',
    'UAB Blazers': 'Blazers',
    'East Carolina Pirates': 'Pirates',
    'Temple Owls': 'Owls',
    'Tulsa Golden Hurricane': 'Golden Hurricane',
    'USF Bulls': 'Bulls',
    'Navy Midshipmen': 'Midshipmen',
    'Connecticut Huskies': 'Huskies',
    'UMass Minutemen': 'Minutemen',
    'Vanderbilt Commodores': 'Commodores',
    'Kennesaw State Owls': 'Owls',
}


class CFBOddsBackfiller:
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    SPORT = "americanfootball_ncaaf"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.db_path = 'cfb_games.db'

    def get_remaining_credits(self) -> int:
        """Check remaining API credits"""
        url = f"{self.BASE_URL}/{self.SPORT}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h'
        }
        try:
            resp = self.session.get(url, params=params, timeout=30)
            remaining = resp.headers.get('x-requests-remaining', '0')
            return int(float(remaining))
        except:
            return 0

    def fetch_historical_odds(self, date: str) -> List[Dict]:
        """
        Fetch historical odds for a specific date
        Costs 20 credits (10 per market × 2 markets)
        """
        url = f"{self.BASE_URL}/{self.SPORT}/odds-history"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'spreads,totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso',
            'date': date
        }

        try:
            resp = self.session.get(url, params=params, timeout=30)
            remaining = resp.headers.get('x-requests-remaining', '?')

            if resp.status_code != 200:
                logger.error(f"API error: {resp.status_code} - {resp.text}")
                return []

            data = resp.json()
            games = data.get('data', [])
            logger.info(f"Date {date[:10]}: {len(games)} games, {remaining} credits left")
            return games

        except Exception as e:
            logger.error(f"Error fetching odds for {date}: {e}")
            return []

    def normalize_team_name(self, name: str) -> str:
        """Convert TheOddsAPI team name to our database format"""
        # First check direct mapping
        if name in TEAM_NAME_MAPPINGS:
            return TEAM_NAME_MAPPINGS[name]

        # Try to extract mascot (last word)
        parts = name.split()
        if len(parts) >= 2:
            mascot = parts[-1]
            # Handle multi-word mascots
            if len(parts) >= 3 and parts[-2] in ['State', 'Tech', 'A&M']:
                mascot = parts[-1]
            return mascot

        return name

    def match_game(self, odds_game: Dict, date: str) -> Optional[int]:
        """
        Match a game from TheOddsAPI to our database
        Returns game_id if found, None otherwise
        """
        home_team_api = odds_game.get('home_team', '')
        away_team_api = odds_game.get('away_team', '')

        home_normalized = self.normalize_team_name(home_team_api)
        away_normalized = self.normalize_team_name(away_team_api)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get game date from commence_time
        commence = odds_game.get('commence_time', '')
        if commence:
            game_date = commence[:10]  # YYYY-MM-DD
        else:
            game_date = date[:10]

        # Try to find matching game within 2 days of the date
        cursor.execute('''
            SELECT g.game_id, ht.display_name, at.display_name
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE date(g.date) BETWEEN date(?, '-1 day') AND date(?, '+1 day')
        ''', (game_date, game_date))

        games = cursor.fetchall()
        conn.close()

        # Try to match by team names
        for game_id, home_db, away_db in games:
            # Check if normalized names match
            if (home_normalized.lower() in home_db.lower() or
                home_db.lower() in home_normalized.lower()):
                if (away_normalized.lower() in away_db.lower() or
                    away_db.lower() in away_normalized.lower()):
                    return game_id

            # Also try the original API names
            if (home_team_api.lower() in home_db.lower() or
                home_db.lower() in home_team_api.lower()):
                if (away_team_api.lower() in away_db.lower() or
                    away_db.lower() in away_team_api.lower()):
                    return game_id

        return None

    def parse_odds(self, game: Dict) -> Dict:
        """Parse odds from TheOddsAPI format"""
        result = {
            'spread': None,
            'total': None,
            'moneyline_home': None,
            'moneyline_away': None
        }

        bookmakers = game.get('bookmakers', [])
        if not bookmakers:
            return result

        # Collect all odds for averaging
        spreads = []
        totals = []

        home_team = game.get('home_team', '')

        for book in bookmakers:
            for market in book.get('markets', []):
                if market.get('key') == 'spreads':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == home_team:
                            point = outcome.get('point')
                            if point is not None:
                                spreads.append(float(point))

                elif market.get('key') == 'totals':
                    for outcome in market.get('outcomes', []):
                        point = outcome.get('point')
                        if point is not None:
                            totals.append(float(point))
                            break  # Only need one total

        if spreads:
            result['spread'] = sum(spreads) / len(spreads)
        if totals:
            result['total'] = sum(totals) / len(totals)

        return result

    def save_odds(self, game_id: int, odds: Dict) -> bool:
        """Save odds to database"""
        if not odds.get('spread') and not odds.get('total'):
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO odds_and_predictions (
                    game_id, source, opening_spread, latest_spread,
                    opening_total, latest_total, odds_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    opening_spread = COALESCE(odds_and_predictions.opening_spread, excluded.opening_spread),
                    latest_spread = COALESCE(excluded.latest_spread, odds_and_predictions.latest_spread),
                    opening_total = COALESCE(odds_and_predictions.opening_total, excluded.opening_total),
                    latest_total = COALESCE(excluded.latest_total, odds_and_predictions.latest_total),
                    odds_updated_at = excluded.odds_updated_at
            ''', (
                game_id,
                'TheOddsAPI',
                odds.get('spread'),
                odds.get('spread'),
                odds.get('total'),
                odds.get('total'),
                datetime.now().isoformat()
            ))

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error saving odds for game {game_id}: {e}")
            return False
        finally:
            conn.close()

    def get_season_saturdays(self, season: int) -> List[str]:
        """Get all Saturdays during CFB regular season"""
        # CFB season: late August to early December
        if season == 2022:
            start = datetime(2022, 8, 27)  # Week 0/1
            end = datetime(2022, 12, 3)    # Championship week
        elif season == 2023:
            start = datetime(2023, 8, 26)
            end = datetime(2023, 12, 2)
        elif season == 2024:
            start = datetime(2024, 8, 24)
            end = datetime(2024, 12, 7)
        else:
            return []

        # Find first Saturday
        days_until_saturday = (5 - start.weekday()) % 7
        current = start + timedelta(days=days_until_saturday)

        saturdays = []
        while current <= end:
            saturdays.append(current.strftime('%Y-%m-%dT12:00:00Z'))
            current += timedelta(days=7)

        return saturdays

    def backfill_season(self, season: int, dry_run: bool = False) -> Tuple[int, int]:
        """
        Backfill odds for an entire season

        Args:
            season: Year (2022, 2023, or 2024)
            dry_run: If True, just count games without saving

        Returns:
            Tuple of (games_matched, odds_saved)
        """
        saturdays = self.get_season_saturdays(season)
        logger.info(f"Season {season}: {len(saturdays)} Saturdays to process")

        # Estimated cost
        estimated_credits = len(saturdays) * 20
        current_credits = self.get_remaining_credits()
        logger.info(f"Estimated cost: {estimated_credits} credits, Available: {current_credits}")

        if estimated_credits > current_credits:
            logger.warning(f"Not enough credits! Need {estimated_credits}, have {current_credits}")
            return 0, 0

        total_matched = 0
        total_saved = 0

        for date in saturdays:
            games = self.fetch_historical_odds(date)

            for game in games:
                game_id = self.match_game(game, date)

                if game_id:
                    total_matched += 1
                    odds = self.parse_odds(game)

                    if not dry_run and self.save_odds(game_id, odds):
                        total_saved += 1

            # Be nice to the API
            time.sleep(1)

        logger.info(f"Season {season}: Matched {total_matched} games, saved {total_saved} odds")
        return total_matched, total_saved


def main():
    # Load API key
    try:
        with open('odds_api_config.json', 'r') as f:
            config = json.load(f)
        api_key = config['api_key']
    except FileNotFoundError:
        logger.error("odds_api_config.json not found!")
        return

    backfiller = CFBOddsBackfiller(api_key)

    # Check credits
    credits = backfiller.get_remaining_credits()
    logger.info(f"Starting with {credits} credits")

    # Backfill each season (most recent first)
    results = {}
    for season in [2024, 2023, 2022]:
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKFILLING {season} SEASON")
        logger.info('='*60)

        matched, saved = backfiller.backfill_season(season)
        results[season] = {'matched': matched, 'saved': saved}

        # Check credits after each season
        remaining = backfiller.get_remaining_credits()
        if remaining < 200:
            logger.warning(f"Low credits ({remaining}), stopping backfill")
            break

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BACKFILL SUMMARY")
    logger.info('='*60)
    for season, data in results.items():
        logger.info(f"  {season}: {data['matched']} matched, {data['saved']} saved")

    total_saved = sum(r['saved'] for r in results.values())
    logger.info(f"  Total odds saved: {total_saved}")


if __name__ == '__main__':
    main()
