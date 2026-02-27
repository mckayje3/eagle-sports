"""
The Odds API Integration for Historical College Football Odds
Free tier: 500 requests/month
Historical data available from June 2020
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


class OddsAPIScraper:
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    SPORT = "americanfootball_ncaaf"

    def __init__(self, api_key: str, db_path: str = 'cfb_games.db'):
        """
        Initialize The Odds API scraper

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

    def fetch_current_odds(self, markets: List[str] = None, regions: str = 'us') -> List[Dict]:
        """
        Fetch current odds for upcoming NCAAF games

        Args:
            markets: List of markets to fetch (h2h, spreads, totals)
            regions: Bookmaker regions (us, us2, uk, eu, au)

        Returns:
            List of games with odds
        """
        if markets is None:
            markets = ['h2h', 'spreads', 'totals']

        url = f"{self.BASE_URL}/{self.SPORT}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }

        try:
            logger.info(f"Fetching current odds from The Odds API...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Log remaining requests
            remaining = response.headers.get('x-requests-remaining')
            logger.info(f"Requests remaining: {remaining}")

            data = response.json()
            logger.info(f"Retrieved {len(data)} games")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching current odds: {e}")
            return []

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
            markets = ['spreads', 'totals']

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
            logger.info(f"Fetching historical odds for {date}...")
            logger.info(f"NOTE: This will cost 10 requests per market")

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

    def save_odds_to_database(self, game_odds: Dict, game_id: Optional[int] = None):
        """
        Save odds data to database

        Args:
            game_odds: Consensus odds data
            game_id: ESPN game_id if known (for linking)
        """
        if not self.db.conn:
            self.db.connect()

        # If we don't have an ESPN game_id, we'll need to match by team names
        # For now, save with the Odds API game_id
        if game_id is None:
            # Try to match teams to get ESPN game_id
            # This is complex and would require team name matching
            # For simplicity, we'll use a placeholder or the API's game_id
            logger.warning(f"No ESPN game_id provided for {game_odds.get('away_team')} @ {game_odds.get('home_team')}")
            return

        odds_data = {
            'game_id': game_id,
            'source': game_odds.get('source', 'TheOddsAPI'),
            'current_spread_home': game_odds.get('spread_home'),
            'current_spread_away': game_odds.get('spread_away'),
            'current_moneyline_home': game_odds.get('moneyline_home'),
            'current_moneyline_away': game_odds.get('moneyline_away'),
            'current_total': game_odds.get('total')
        }

        self.db.insert_or_update_odds(odds_data)
        logger.info(f"Saved odds for game_id {game_id}")


def setup_guide():
    """Print setup instructions"""
    print("\n" + "="*80)
    print("THE ODDS API SETUP GUIDE")
    print("="*80)
    print("\n1. Get a FREE API Key:")
    print("   - Go to: https://the-odds-api.com/")
    print("   - Click 'Get Started' or 'Sign Up'")
    print("   - Create a free account")
    print("   - You'll receive 500 free requests per month")
    print()
    print("2. Copy your API key")
    print()
    print("3. Save it to a config file:")
    print("   - Create a file: odds_api_config.json")
    print("   - Add your key: {\"api_key\": \"YOUR_KEY_HERE\"}")
    print()
    print("4. Usage:")
    print("   from odds_api_scraper import OddsAPIScraper")
    print("   scraper = OddsAPIScraper(api_key='your_key')")
    print("   games = scraper.fetch_current_odds()")
    print()
    print("="*80)
    print("\nFree Tier Details:")
    print("  - 500 requests/month")
    print("  - 1 request = 1 current odds fetch")
    print("  - 10 requests = 1 historical odds fetch (per market)")
    print("  - Historical data from June 2020")
    print("="*80 + "\n")


if __name__ == '__main__':
    setup_guide()

    # Example usage (commented out - replace with your API key)
    """
    # Load API key from config file
    with open('odds_api_config.json', 'r') as f:
        config = json.load(f)

    scraper = OddsAPIScraper(api_key=config['api_key'])

    # Check remaining requests
    scraper.get_remaining_requests()

    # Fetch current odds
    games = scraper.fetch_current_odds()

    # Display first game
    if games:
        parsed = scraper.parse_odds_data(games[0])
        consensus = scraper.get_consensus_odds(parsed)
        print(json.dumps(consensus, indent=2))
    """
