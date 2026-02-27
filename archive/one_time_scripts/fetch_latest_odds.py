"""
Fetch Latest Odds Script
Fetches current odds from The Odds API and updates the database using the simplified schema.

Usage:
    py fetch_latest_odds.py nfl          # Update NFL odds only
    py fetch_latest_odds.py cfb          # Update CFB odds only
    py fetch_latest_odds.py all          # Update both
    py fetch_latest_odds.py nfl --opening # Set as opening lines (use when lines first release)

The simplified schema:
    - opening_line: Fixed once set (captured when lines first release)
    - latest_line: Updated on each scrape until game completes
    - line_movement: Computed as latest_line - opening_line
"""

import json
import sys
from datetime import datetime, timedelta
import requests
import logging
from database import FootballDatabase
from game_matcher import GameMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OddsAPIFetcher:
    """Fetches odds from The Odds API using the simplified line schema"""

    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    SPORTS = {
        'nfl': 'americanfootball_nfl',
        'cfb': 'americanfootball_ncaaf'
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.requests_remaining = None

    def fetch_current_odds(self, sport: str) -> list:
        """
        Fetch current odds for upcoming games.

        Args:
            sport: 'nfl' or 'cfb'

        Returns:
            List of games with odds data
        """
        sport_key = self.SPORTS.get(sport.lower())
        if not sport_key:
            logger.error(f"Unknown sport: {sport}")
            return []

        url = f"{self.BASE_URL}/{sport_key}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'spreads,totals,h2h',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }

        try:
            logger.info(f"Fetching {sport.upper()} odds from The Odds API...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Track remaining requests
            self.requests_remaining = response.headers.get('x-requests-remaining')
            logger.info(f"API requests remaining: {self.requests_remaining}")

            data = response.json()
            logger.info(f"Retrieved {len(data)} games")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching odds: {e}")
            return []

    def get_consensus_line(self, game_data: dict) -> dict:
        """
        Calculate consensus spread and total from multiple bookmakers.

        Args:
            game_data: Game data from The Odds API

        Returns:
            Dict with consensus spread_home and total
        """
        spreads_home = []
        totals = []
        ml_home = []
        ml_away = []

        home_team = game_data.get('home_team')

        for bookmaker in game_data.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                market_key = market.get('key')

                if market_key == 'spreads':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == home_team:
                            point = outcome.get('point')
                            if point is not None:
                                spreads_home.append(point)

                elif market_key == 'totals':
                    for outcome in market.get('outcomes', []):
                        point = outcome.get('point')
                        if point is not None:
                            totals.append(point)
                            break  # Only need one total per bookmaker

                elif market_key == 'h2h':
                    for outcome in market.get('outcomes', []):
                        price = outcome.get('price')
                        if outcome.get('name') == home_team:
                            if price is not None:
                                ml_home.append(price)
                        else:
                            if price is not None:
                                ml_away.append(price)

        result = {
            'home_team': home_team,
            'away_team': game_data.get('away_team'),
            'commence_time': game_data.get('commence_time'),
            'spread_home': sum(spreads_home) / len(spreads_home) if spreads_home else None,
            'total': sum(totals) / len(totals) if totals else None,
            'moneyline_home': int(sum(ml_home) / len(ml_home)) if ml_home else None,
            'moneyline_away': int(sum(ml_away) / len(ml_away)) if ml_away else None,
            'num_books': len(game_data.get('bookmakers', []))
        }

        return result

    def update_odds(self, sport: str, db_path: str, is_opening: bool = False):
        """
        Fetch and update odds for a sport.

        Args:
            sport: 'nfl' or 'cfb'
            db_path: Path to database
            is_opening: True to set these as opening lines
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"UPDATING {sport.upper()} ODDS")
        if is_opening:
            logger.info("** Setting as OPENING lines **")
        logger.info('='*60)

        # Fetch current odds
        games = self.fetch_current_odds(sport)
        if not games:
            logger.warning("No games retrieved")
            return

        # Initialize database and matcher
        db = FootballDatabase(db_path)
        db.connect()
        matcher = GameMatcher(db_path)

        updated = 0
        skipped = 0

        for game in games:
            try:
                consensus = self.get_consensus_line(game)

                if consensus['spread_home'] is None:
                    logger.debug(f"No spread for {consensus['away_team']} @ {consensus['home_team']}")
                    skipped += 1
                    continue

                # Match to ESPN game_id
                game_id = matcher.match_odds_api_game({
                    'home_team': consensus['home_team'],
                    'away_team': consensus['away_team'],
                    'commence_time': consensus['commence_time']
                })

                if game_id is None:
                    logger.warning(f"Could not match: {consensus['away_team']} @ {consensus['home_team']}")
                    skipped += 1
                    continue

                # Update using the simplified schema
                db.update_lines(
                    game_id=game_id,
                    spread=consensus['spread_home'],
                    total=consensus['total'] or 0,
                    is_opening=is_opening,
                    source='TheOddsAPI'
                )

                logger.info(f"Updated: {consensus['away_team']} @ {consensus['home_team']} "
                           f"(spread: {consensus['spread_home']:+.1f}, total: {consensus['total']:.1f})")
                updated += 1

            except Exception as e:
                logger.error(f"Error processing game: {e}")
                skipped += 1

        db.close()

        logger.info(f"\nSummary:")
        logger.info(f"  Updated: {updated}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  API requests remaining: {self.requests_remaining}")


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    sport = sys.argv[1].lower()
    is_opening = '--opening' in sys.argv

    # Load API key
    try:
        with open('odds_api_config.json', 'r') as f:
            config = json.load(f)
            api_key = config['api_key']
    except FileNotFoundError:
        print("ERROR: odds_api_config.json not found")
        print("Create it with: {\"api_key\": \"YOUR_KEY_HERE\"}")
        sys.exit(1)

    fetcher = OddsAPIFetcher(api_key)

    if sport == 'nfl':
        fetcher.update_odds('nfl', 'nfl_games.db', is_opening)
    elif sport == 'cfb':
        fetcher.update_odds('cfb', 'cfb_games.db', is_opening)
    elif sport == 'all':
        fetcher.update_odds('nfl', 'nfl_games.db', is_opening)
        fetcher.update_odds('cfb', 'cfb_games.db', is_opening)
    else:
        print(f"Unknown sport: {sport}")
        print("Use: nfl, cfb, or all")
        sys.exit(1)


if __name__ == '__main__':
    main()
