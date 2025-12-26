"""
Full NBA Odds Backfill - Daily dates instead of weekly sampling
Uses more API credits but gets better coverage
"""
import json
import time
import sqlite3
import requests
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAGameMatcher:
    """Match Odds API games to database games"""

    def __init__(self, db_path='nba_games.db'):
        self.db_path = db_path
        self.team_mappings = {
            # Odds API name -> ESPN name
            'Los Angeles Lakers': 'Los Angeles Lakers',
            'Los Angeles Clippers': 'LA Clippers',
            'LA Clippers': 'LA Clippers',
            'Golden State Warriors': 'Golden State Warriors',
            'Phoenix Suns': 'Phoenix Suns',
            'Sacramento Kings': 'Sacramento Kings',
            'Denver Nuggets': 'Denver Nuggets',
            'Minnesota Timberwolves': 'Minnesota Timberwolves',
            'Oklahoma City Thunder': 'Oklahoma City Thunder',
            'Portland Trail Blazers': 'Portland Trail Blazers',
            'Utah Jazz': 'Utah Jazz',
            'Dallas Mavericks': 'Dallas Mavericks',
            'Houston Rockets': 'Houston Rockets',
            'Memphis Grizzlies': 'Memphis Grizzlies',
            'New Orleans Pelicans': 'New Orleans Pelicans',
            'San Antonio Spurs': 'San Antonio Spurs',
            'Boston Celtics': 'Boston Celtics',
            'Brooklyn Nets': 'Brooklyn Nets',
            'New York Knicks': 'New York Knicks',
            'Philadelphia 76ers': 'Philadelphia 76ers',
            'Toronto Raptors': 'Toronto Raptors',
            'Chicago Bulls': 'Chicago Bulls',
            'Cleveland Cavaliers': 'Cleveland Cavaliers',
            'Detroit Pistons': 'Detroit Pistons',
            'Indiana Pacers': 'Indiana Pacers',
            'Milwaukee Bucks': 'Milwaukee Bucks',
            'Atlanta Hawks': 'Atlanta Hawks',
            'Charlotte Hornets': 'Charlotte Hornets',
            'Miami Heat': 'Miami Heat',
            'Orlando Magic': 'Orlando Magic',
            'Washington Wizards': 'Washington Wizards',
        }

    def normalize_team_name(self, name):
        """Normalize team name"""
        return self.team_mappings.get(name, name)

    def match_game(self, odds_game):
        """Match an Odds API game to database game"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        home_team = self.normalize_team_name(odds_game.get('home_team', ''))
        away_team = self.normalize_team_name(odds_game.get('away_team', ''))
        game_date = odds_game.get('commence_time', '')[:10]  # YYYY-MM-DD

        # Try to match by teams and date
        cursor.execute('''
            SELECT g.game_id
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE (ht.name LIKE ? OR ht.display_name LIKE ?)
            AND (at.name LIKE ? OR at.display_name LIKE ?)
            AND DATE(g.date) = DATE(?)
        ''', (f'%{home_team}%', f'%{home_team}%',
              f'%{away_team}%', f'%{away_team}%',
              game_date))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None


def fetch_nba_historical_odds(api_key, date, markets=['spreads', 'totals']):
    """Fetch historical NBA odds directly from API"""
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds-history"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': ','.join(markets),
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'date': date
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        remaining = response.headers.get('x-requests-remaining', 'Unknown')
        logger.info(f"API requests remaining: {remaining}")

        if response.status_code != 200:
            logger.error(f"API error: {response.status_code}")
            return [], remaining

        data = response.json()
        games = data.get('data', [])
        return games, remaining

    except Exception as e:
        logger.error(f"Error fetching odds: {e}")
        return [], None


def backfill_nba_odds_full(season_year, start_month=10, end_month=6):
    """
    Full NBA odds backfill - every 3 days instead of weekly
    season_year: The year the season starts (e.g., 2023 for 2023-24)
    """
    print(f"\n{'='*80}")
    print(f"NBA {season_year}-{str(season_year+1)[-2:]} FULL ODDS BACKFILL")
    print(f"{'='*80}\n")

    # Load API key
    try:
        with open('odds_api_config.json', 'r') as f:
            config = json.load(f)
            api_key = config.get('api_key')
    except FileNotFoundError:
        print("ERROR: odds_api_config.json not found!")
        return 0

    if not api_key:
        print("ERROR: No API key")
        return 0

    matcher = NBAGameMatcher()
    conn = sqlite3.connect('nba_games.db')

    # Generate dates for the season - every 3 days for better coverage
    dates = []

    # First part: Oct-Dec of start year
    if start_month >= 10:
        current = datetime(season_year, start_month, 20)  # Start late Oct when season begins
        end_dec = datetime(season_year, 12, 31)
        while current <= end_dec:
            dates.append(current)
            current += timedelta(days=3)

    # Second part: Jan-Jun of next year
    current = datetime(season_year + 1, 1, 1)
    end_jun = datetime(season_year + 1, end_month, 30)
    while current <= end_jun:
        dates.append(current)
        current += timedelta(days=3)

    print(f"Fetching odds for {len(dates)} dates (every 3 days)")

    total_games = 0
    total_matched = 0
    total_saved = 0
    total_new = 0
    remaining = None

    for i, date in enumerate(dates, 1):
        date_str = date.strftime('%Y-%m-%dT17:00:00Z')
        print(f"\n[{i}/{len(dates)}] {date.strftime('%Y-%m-%d')}")

        games, remaining = fetch_nba_historical_odds(api_key, date_str)

        if not games:
            print("  No games found")
            continue

        print(f"  Retrieved {len(games)} games")
        total_games += len(games)

        matched = 0
        saved = 0
        new_odds = 0

        for game in games:
            home = game.get('home_team')
            away = game.get('away_team')

            if not home or not away:
                continue

            game_id = matcher.match_game(game)

            if not game_id:
                continue

            matched += 1

            bookmakers = game.get('bookmakers', [])
            if not bookmakers:
                continue

            spreads_home = []
            totals = []

            for book in bookmakers:
                for market in book.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == home:
                                spreads_home.append(outcome['point'])
                    elif market['key'] == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == 'Over':
                                totals.append(outcome['point'])

            if spreads_home or totals:
                avg_spread = sum(spreads_home) / len(spreads_home) if spreads_home else None
                avg_total = sum(totals) / len(totals) if totals else None

                try:
                    cursor = conn.cursor()
                    cursor.execute('SELECT id FROM game_odds WHERE game_id = ? AND source = ?',
                                   (game_id, 'TheOddsAPI'))
                    existing = cursor.fetchone()

                    if existing:
                        cursor.execute('''
                            UPDATE game_odds SET
                                closing_spread_home = ?,
                                closing_total = ?,
                                latest_line = ?,
                                latest_total_line = ?,
                                updated_at = ?
                            WHERE game_id = ? AND source = ?
                        ''', (
                            round(avg_spread, 1) if avg_spread else None,
                            round(avg_total, 1) if avg_total else None,
                            round(avg_spread, 1) if avg_spread else None,
                            round(avg_total, 1) if avg_total else None,
                            date_str,
                            game_id,
                            'TheOddsAPI'
                        ))
                    else:
                        cursor.execute('''
                            INSERT INTO game_odds
                            (game_id, source, closing_spread_home, closing_total,
                             latest_line, latest_total_line, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            game_id,
                            'TheOddsAPI',
                            round(avg_spread, 1) if avg_spread else None,
                            round(avg_total, 1) if avg_total else None,
                            round(avg_spread, 1) if avg_spread else None,
                            round(avg_total, 1) if avg_total else None,
                            date_str
                        ))
                        new_odds += 1
                    conn.commit()
                    saved += 1
                except Exception as e:
                    print(f"  Error saving: {e}")

        total_matched += matched
        total_saved += saved
        total_new += new_odds
        print(f"  Matched: {matched} | Saved: {saved} | New: {new_odds}")

        time.sleep(1)

    conn.close()

    print(f"\n{'='*80}")
    print(f"NBA FULL ODDS BACKFILL COMPLETE")
    print(f"{'='*80}")
    print(f"Total games fetched: {total_games}")
    print(f"Total matched: {total_matched}")
    print(f"Total saved: {total_saved}")
    print(f"New odds added: {total_new}")
    print(f"API requests remaining: {remaining}")

    return total_saved


def main():
    import sys

    if len(sys.argv) > 1:
        season = int(sys.argv[1])
        backfill_nba_odds_full(season)
    else:
        # Default: backfill both seasons
        backfill_nba_odds_full(2023)  # 2023-24 season
        backfill_nba_odds_full(2024)  # 2024-25 season


if __name__ == '__main__':
    main()
