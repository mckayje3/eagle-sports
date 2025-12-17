"""
The Odds API Integration for NFL Historical Odds
Scrapes NFL odds from The Odds API with your paid credits
"""
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from cfb_nfl_database import FootballDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLOddsAPIScraper:
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    SPORT = "americanfootball_nfl"

    def __init__(self, api_key: str, db_path: str = 'nfl_games.db'):
        """
        Initialize The Odds API scraper for NFL

        Args:
            api_key: Your API key from the-odds-api.com
            db_path: Path to SQLite database
        """
        self.api_key = api_key
        self.db = FootballDatabase(db_path)
        self.session = requests.Session()

    def get_remaining_requests(self) -> Optional[int]:
        """Check how many API requests you have remaining"""
        url = f"{self.BASE_URL}/{self.SPORT}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american'
        }

        try:
            response = self.session.get(url, params=params, timeout=30)

            # Get remaining requests from headers
            remaining = response.headers.get('x-requests-remaining')
            used = response.headers.get('x-requests-used')

            if remaining:
                logger.info(f"API Requests Remaining: {remaining}")
                logger.info(f"API Requests Used: {used}")
                return int(remaining)

            return None
        except Exception as e:
            logger.error(f"Error checking remaining requests: {e}")
            return None

    def fetch_historical_odds(self, date: str, markets: List[str] = None, regions: str = 'us') -> List[Dict]:
        """
        Fetch historical odds for a specific date
        Note: Historical data costs 10 requests per region per market

        Args:
            date: ISO 8601 date string (e.g., '2024-11-09T12:00:00Z')
            markets: List of markets (h2h, spreads, totals)
            regions: Bookmaker regions

        Returns:
            List of games with historical odds
        """
        if markets is None:
            markets = ['spreads', 'totals', 'h2h']

        url = f"{self.BASE_URL}/{self.SPORT}/odds-history"
        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'dateFormat': 'iso',
            'date': date
        }

        try:
            logger.info(f"Fetching historical NFL odds for {date}...")
            logger.info(f"NOTE: This will cost ~10 requests per market")

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Log remaining requests
            remaining = response.headers.get('x-requests-remaining')
            used = response.headers.get('x-requests-used')
            logger.info(f"Requests remaining: {remaining}, Used in this call: {used}")

            response_data = response.json()

            # Historical API returns a dict with 'data' field containing the games
            games_data = response_data.get('data', [])
            logger.info(f"Retrieved {len(games_data)} games")

            return games_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical odds: {e}")
            return []

    def parse_odds_data(self, game_data: Dict) -> Dict:
        """
        Parse odds data from The Odds API format to our database format

        Args:
            game_data: Game data from The Odds API

        Returns:
            Dictionary formatted for our database
        """
        parsed = {
            'game_id': game_data.get('id'),
            'sport_key': game_data.get('sport_key'),
            'commence_time': game_data.get('commence_time'),
            'home_team': game_data.get('home_team'),
            'away_team': game_data.get('away_team'),
            'source': 'TheOddsAPI',
            'bookmakers': []
        }

        # Parse bookmaker odds
        for bookmaker in game_data.get('bookmakers', []):
            bookmaker_data = {
                'key': bookmaker.get('key'),
                'title': bookmaker.get('title'),
                'last_update': bookmaker.get('last_update')
            }

            # Parse markets
            for market in bookmaker.get('markets', []):
                market_key = market.get('key')

                if market_key == 'h2h':  # Moneyline
                    for outcome in market.get('outcomes', []):
                        team = outcome.get('name')
                        price = outcome.get('price')

                        if team == parsed['home_team']:
                            bookmaker_data['moneyline_home'] = price
                        elif team == parsed['away_team']:
                            bookmaker_data['moneyline_away'] = price

                elif market_key == 'spreads':
                    for outcome in market.get('outcomes', []):
                        team = outcome.get('name')
                        point = outcome.get('point')
                        price = outcome.get('price')

                        if team == parsed['home_team']:
                            bookmaker_data['spread_home'] = point
                            bookmaker_data['spread_home_price'] = price
                        elif team == parsed['away_team']:
                            bookmaker_data['spread_away'] = point
                            bookmaker_data['spread_away_price'] = price

                elif market_key == 'totals':
                    for outcome in market.get('outcomes', []):
                        name = outcome.get('name')
                        point = outcome.get('point')
                        price = outcome.get('price')

                        bookmaker_data['total'] = point
                        if name == 'Over':
                            bookmaker_data['over_price'] = price
                        elif name == 'Under':
                            bookmaker_data['under_price'] = price

            parsed['bookmakers'].append(bookmaker_data)

        return parsed

    def get_consensus_odds(self, parsed_game: Dict) -> Dict:
        """
        Calculate consensus odds from multiple bookmakers

        Args:
            parsed_game: Parsed game data with multiple bookmakers

        Returns:
            Dictionary with consensus odds
        """
        if not parsed_game.get('bookmakers'):
            return {}

        # Collect all odds
        spreads_home = []
        spreads_away = []
        totals = []
        ml_home = []
        ml_away = []

        for book in parsed_game['bookmakers']:
            if 'spread_home' in book:
                spreads_home.append(book['spread_home'])
            if 'spread_away' in book:
                spreads_away.append(book['spread_away'])
            if 'total' in book:
                totals.append(book['total'])
            if 'moneyline_home' in book:
                ml_home.append(book['moneyline_home'])
            if 'moneyline_away' in book:
                ml_away.append(book['moneyline_away'])

        # Calculate averages (consensus)
        consensus = {
            'game_id': parsed_game.get('game_id'),
            'home_team': parsed_game.get('home_team'),
            'away_team': parsed_game.get('away_team'),
            'commence_time': parsed_game.get('commence_time'),
            'source': 'TheOddsAPI'
        }

        if spreads_home:
            consensus['spread_home'] = sum(spreads_home) / len(spreads_home)
        if spreads_away:
            consensus['spread_away'] = sum(spreads_away) / len(spreads_away)
        if totals:
            consensus['total'] = sum(totals) / len(totals)
        if ml_home:
            consensus['moneyline_home'] = int(sum(ml_home) / len(ml_home))
        if ml_away:
            consensus['moneyline_away'] = int(sum(ml_away) / len(ml_away))

        return consensus

    def match_team_to_espn_id(self, odds_team_name: str) -> Optional[int]:
        """
        Match Odds API team name to ESPN team_id

        Args:
            odds_team_name: Team name from Odds API

        Returns:
            ESPN team_id if found, None otherwise
        """
        if not self.db.conn:
            self.db.connect()

        cursor = self.db.conn.cursor()

        # Try exact match first
        cursor.execute(
            "SELECT team_id FROM teams WHERE display_name LIKE ? OR name LIKE ? OR abbreviation LIKE ?",
            (f"%{odds_team_name}%", f"%{odds_team_name}%", f"%{odds_team_name}%")
        )

        result = cursor.fetchone()
        if result:
            return result[0]

        # Log unmatched team
        logger.warning(f"Could not match team: {odds_team_name}")
        return None

    def find_game_id(self, home_team_name: str, away_team_name: str, commence_time: str) -> Optional[int]:
        """
        Find ESPN game_id by matching teams and date

        Args:
            home_team_name: Home team name from Odds API
            away_team_name: Away team name from Odds API
            commence_time: Game commence time (ISO format)

        Returns:
            ESPN game_id if found, None otherwise
        """
        home_id = self.match_team_to_espn_id(home_team_name)
        away_id = self.match_team_to_espn_id(away_team_name)

        if not home_id or not away_id:
            return None

        # Parse date
        game_date = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        date_str = game_date.strftime('%Y-%m-%dT%H:%M')

        # Find game in database (within 6 hours of commence time)
        if not self.db.conn:
            self.db.connect()

        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT game_id FROM games
            WHERE home_team_id = ?
            AND away_team_id = ?
            AND date LIKE ?
        ''', (home_id, away_id, f"{date_str[:10]}%"))

        result = cursor.fetchone()
        if result:
            return result[0]

        return None

    def save_odds_to_database(self, consensus_odds: Dict):
        """
        Save odds data to database

        Args:
            consensus_odds: Consensus odds data
        """
        if not self.db.conn:
            self.db.connect()

        # Find matching ESPN game_id
        game_id = self.find_game_id(
            consensus_odds['home_team'],
            consensus_odds['away_team'],
            consensus_odds['commence_time']
        )

        if game_id is None:
            logger.warning(f"No ESPN game found for {consensus_odds['away_team']} @ {consensus_odds['home_team']}")
            return

        odds_data = {
            'game_id': game_id,
            'source': 'TheOddsAPI',
            'latest_spread': consensus_odds.get('spread_home'),
            'latest_moneyline_home': consensus_odds.get('moneyline_home'),
            'latest_moneyline_away': consensus_odds.get('moneyline_away'),
            'latest_total': consensus_odds.get('total')
        }

        self.db.insert_or_update_odds(odds_data)
        logger.info(f"Saved odds for game_id {game_id}: {consensus_odds['away_team']} @ {consensus_odds['home_team']}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("NFL ODDS API SCRAPER")
    print("="*80)
    print("\nThis scraper fetches historical NFL odds from The Odds API")
    print("\nNote: Historical odds cost ~10 requests per market per date")
    print("With 20,000 credits, you can fetch:")
    print("  - ~2,000 dates with 1 market (spreads)")
    print("  - ~650 dates with 3 markets (spreads, totals, h2h)")
    print("\n" + "="*80 + "\n")
