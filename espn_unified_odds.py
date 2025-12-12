"""
ESPN Unified Odds Scraper - Primary odds source for current season data
Supports: NBA, NFL, CFB, CBB

This is the PRIMARY source for current season odds.
For historical data (previous seasons), use TheOddsAPI.
VegasInsider scraper available as backup/verification.
"""
import requests
import sqlite3
from datetime import datetime
import time
import argparse


class ESPNOddsScraper:
    """Unified ESPN Odds Scraper for all sports"""

    SPORTS_CONFIG = {
        'nba': {
            'sport': 'basketball',
            'league': 'nba',
            'db_path': 'nba_games.db',
            'season_field': 'season',  # Uses calendar year (2025 = 2024-25 season)
        },
        'nfl': {
            'sport': 'football',
            'league': 'nfl',
            'db_path': 'nfl_games.db',
            'season_field': 'season',
        },
        'cfb': {
            'sport': 'football',
            'league': 'college-football',
            'db_path': 'cfb_games.db',
            'season_field': 'season',
        },
        'cbb': {
            'sport': 'basketball',
            'league': 'mens-college-basketball',
            'db_path': 'cbb_games.db',
            'season_field': 'season',
        }
    }

    def __init__(self, sport: str):
        """
        Initialize scraper for a specific sport

        Args:
            sport: One of 'nba', 'nfl', 'cfb'
        """
        if sport.lower() not in self.SPORTS_CONFIG:
            raise ValueError(f"Unsupported sport: {sport}. Use: nba, nfl, cfb")

        self.sport = sport.lower()
        self.config = self.SPORTS_CONFIG[self.sport]
        self.base_url = f"https://sports.core.api.espn.com/v2/sports/{self.config['sport']}/leagues/{self.config['league']}"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_game_odds(self, game_id: int) -> dict:
        """
        Get odds for a specific game from ESPN API

        Args:
            game_id: ESPN game ID

        Returns:
            Dictionary with odds data or None if unavailable
        """
        url = f"{self.base_url}/events/{game_id}/competitions/{game_id}/odds"

        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return None

            data = resp.json()
            items = data.get('items', [])

            if not items:
                return None

            # Get first provider (usually DraftKings)
            odds = items[0]

            result = {
                'game_id': int(game_id),
                'source': odds.get('provider', {}).get('name', 'ESPN'),
                'spread': odds.get('spread'),  # Home team spread
                'over_under': odds.get('overUnder'),
                'over_odds': odds.get('overOdds'),
                'under_odds': odds.get('underOdds'),
            }

            # Helper to parse moneyline (handles "EVEN" = +100)
            def parse_ml(val):
                if not val:
                    return None
                if str(val).upper() == 'EVEN':
                    return 100
                try:
                    return int(str(val).replace('+', ''))
                except (ValueError, TypeError):
                    return None

            # Home team odds
            home_odds = odds.get('homeTeamOdds', {})
            if home_odds:
                result['home_favorite'] = home_odds.get('favorite', False)
                result['home_moneyline'] = home_odds.get('moneyLine')
                result['home_spread_odds'] = home_odds.get('spreadOdds')

                # Opening lines
                home_open = home_odds.get('open', {})
                if home_open:
                    ps = home_open.get('pointSpread', {})
                    if ps.get('american'):
                        try:
                            result['opening_spread'] = float(str(ps['american']).replace('+', ''))
                        except (ValueError, TypeError):
                            pass
                    ml = home_open.get('moneyLine', {})
                    result['opening_ml_home'] = parse_ml(ml.get('american'))

                # Closing lines
                home_current = home_odds.get('current', {})
                if home_current:
                    ml = home_current.get('moneyLine', {})
                    result['closing_ml_home'] = parse_ml(ml.get('american'))

            # Away team odds
            away_odds = odds.get('awayTeamOdds', {})
            if away_odds:
                result['away_moneyline'] = away_odds.get('moneyLine')
                result['away_spread_odds'] = away_odds.get('spreadOdds')

                # Opening lines
                away_open = away_odds.get('open', {})
                if away_open:
                    ml = away_open.get('moneyLine', {})
                    result['opening_ml_away'] = parse_ml(ml.get('american'))

                # Closing lines
                away_current = away_odds.get('current', {})
                if away_current:
                    ml = away_current.get('moneyLine', {})
                    result['closing_ml_away'] = parse_ml(ml.get('american'))

            return result

        except Exception as e:
            print(f"Error getting odds for game {game_id}: {e}")
            return None

    def save_odds_to_db(self, odds_data: dict) -> bool:
        """
        Save odds data to database

        Args:
            odds_data: Dictionary with odds from get_game_odds()

        Returns:
            True if saved successfully
        """
        if not odds_data:
            return False

        conn = sqlite3.connect(self.config['db_path'])
        cursor = conn.cursor()

        try:
            game_id = odds_data['game_id']

            # Verify game exists in database
            cursor.execute('SELECT game_id FROM games WHERE game_id = ?', (game_id,))
            if not cursor.fetchone():
                conn.close()
                return False

            # Determine column name (some DBs use 'source', some use 'sportsbook')
            cursor.execute("PRAGMA table_info(game_odds)")
            columns = [col[1] for col in cursor.fetchall()]
            source_col = 'source' if 'source' in columns else 'sportsbook'

            # Get current values
            opening_spread = odds_data.get('opening_spread')
            latest_spread = odds_data.get('spread')
            opening_total = odds_data.get('over_under')
            latest_total = odds_data.get('over_under')
            opening_ml = odds_data.get('opening_ml_home')
            latest_ml = odds_data.get('closing_ml_home') or odds_data.get('home_moneyline')
            
            # Calculate movements
            spread_movement = (latest_spread - opening_spread) if (opening_spread and latest_spread) else None
            total_movement = (latest_total - opening_total) if (opening_total and latest_total) else None
            ml_movement = (latest_ml - opening_ml) if (opening_ml and latest_ml) else None
            
            # Upsert odds with simplified schema
            cursor.execute(f'''
                INSERT INTO game_odds (
                    game_id, {source_col},
                    opening_spread, latest_spread,
                    opening_total, latest_total,
                    opening_moneyline, latest_moneyline,
                    spread_movement, total_movement, moneyline_movement,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, {source_col}) DO UPDATE SET
                    latest_spread = excluded.latest_spread,
                    latest_total = excluded.latest_total,
                    latest_moneyline = excluded.latest_moneyline,
                    spread_movement = excluded.spread_movement,
                    total_movement = excluded.total_movement,
                    moneyline_movement = excluded.moneyline_movement,
                    updated_at = excluded.updated_at
            ''', (
                game_id,
                odds_data.get('source', 'ESPN'),
                opening_spread,
                latest_spread,
                opening_total,
                latest_total,
                opening_ml,
                latest_ml,
                spread_movement,
                total_movement,
                ml_movement,
                datetime.now().isoformat()
            ))

            conn.commit()
            return True

        except Exception as e:
            print(f"Error saving odds: {e}")
            return False

        finally:
            conn.close()

    def update_prediction_cache(self, game_id: int, spread: float, total: float) -> bool:
        """
        Update prediction_cache in users.db with Vegas odds

        Args:
            game_id: Game ID
            spread: Home team spread
            total: Over/under total

        Returns:
            True if updated
        """
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()

            sport_name = self.sport.upper()

            cursor.execute('''
                UPDATE prediction_cache
                SET vegas_spread = COALESCE(?, vegas_spread),
                    vegas_total = COALESCE(?, vegas_total)
                WHERE game_id = ? AND sport = ?
            ''', (spread, total, game_id, sport_name))

            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return updated

        except Exception:
            return False

    def scrape_season(self, season: int, update_cache: bool = True) -> tuple:
        """
        Scrape odds for all games in a season from database

        Args:
            season: Season year (e.g., 2025)
            update_cache: Whether to update prediction_cache in users.db

        Returns:
            Tuple of (games_with_odds, games_saved)
        """
        conn = sqlite3.connect(self.config['db_path'])
        cursor = conn.cursor()

        cursor.execute('''
            SELECT game_id FROM games
            WHERE season = ?
            ORDER BY date
        ''', (season,))

        games = cursor.fetchall()
        conn.close()

        print(f"Found {len(games)} {self.sport.upper()} {season} games in database")

        total_with_odds = 0
        total_saved = 0

        for (game_id,) in games:
            odds = self.get_game_odds(game_id)

            if odds:
                total_with_odds += 1
                if self.save_odds_to_db(odds):
                    total_saved += 1

                    if update_cache:
                        self.update_prediction_cache(
                            game_id,
                            odds.get('spread'),
                            odds.get('over_under')
                        )

            if total_with_odds > 0 and total_with_odds % 50 == 0:
                print(f"  Processed {total_with_odds} games with odds...")

            time.sleep(0.1)  # Be nice to ESPN

        print(f"\nResults:")
        print(f"  Games with odds: {total_with_odds}")
        print(f"  Saved to database: {total_saved}")

        return total_with_odds, total_saved

    def scrape_recent(self, days: int = 7, update_cache: bool = True) -> tuple:
        """
        Scrape odds for games in the past N days

        Args:
            days: Number of days to look back
            update_cache: Whether to update prediction_cache

        Returns:
            Tuple of (games_with_odds, games_saved)
        """
        conn = sqlite3.connect(self.config['db_path'])
        cursor = conn.cursor()

        cursor.execute('''
            SELECT game_id FROM games
            WHERE date >= date('now', ?)
            AND date <= date('now', '+1 day')
            ORDER BY date
        ''', (f'-{days} days',))

        games = cursor.fetchall()
        conn.close()

        print(f"Found {len(games)} {self.sport.upper()} games in last {days} days")

        total_with_odds = 0
        total_saved = 0

        for (game_id,) in games:
            odds = self.get_game_odds(game_id)

            if odds:
                total_with_odds += 1
                if self.save_odds_to_db(odds):
                    total_saved += 1

                    if update_cache:
                        self.update_prediction_cache(
                            game_id,
                            odds.get('spread'),
                            odds.get('over_under')
                        )

            time.sleep(0.1)

        print(f"Results: {total_saved}/{total_with_odds} games saved")
        return total_with_odds, total_saved


def sync_odds_to_prediction_cache(sport: str):
    """
    Sync all odds from sport database to prediction_cache

    Args:
        sport: One of 'nba', 'nfl', 'cfb'
    """
    config = ESPNOddsScraper.SPORTS_CONFIG[sport.lower()]
    db_path = config['db_path']
    sport_name = sport.upper()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT game_id, latest_spread, latest_total
        FROM game_odds
        WHERE latest_spread IS NOT NULL
    ''')
    odds_records = cursor.fetchall()
    conn.close()

    print(f"Found {len(odds_records)} odds records in {db_path}")

    users_conn = sqlite3.connect('users.db')
    users_cursor = users_conn.cursor()

    updated = 0
    for game_id, spread, total in odds_records:
        users_cursor.execute('''
            UPDATE prediction_cache
            SET vegas_spread = COALESCE(?, vegas_spread),
                vegas_total = COALESCE(?, vegas_total)
            WHERE game_id = ? AND sport = ?
        ''', (spread, total, game_id, sport_name))
        if users_cursor.rowcount > 0:
            updated += 1

    users_conn.commit()
    users_conn.close()

    print(f"Updated {updated} prediction_cache records")
    return updated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ESPN Unified Odds Scraper')
    parser.add_argument('sport', choices=['nba', 'nfl', 'cfb', 'cbb'], help='Sport to scrape')
    parser.add_argument('--season', type=int, help='Season year to scrape')
    parser.add_argument('--days', type=int, default=7, help='Days to look back (default: 7)')
    parser.add_argument('--sync', action='store_true', help='Sync existing odds to prediction_cache')

    args = parser.parse_args()

    print("=" * 60)
    print(f"ESPN {args.sport.upper()} ODDS SCRAPER")
    print("=" * 60)

    if args.sync:
        sync_odds_to_prediction_cache(args.sport)
    else:
        scraper = ESPNOddsScraper(args.sport)

        if args.season:
            scraper.scrape_season(args.season)
        else:
            scraper.scrape_recent(args.days)
