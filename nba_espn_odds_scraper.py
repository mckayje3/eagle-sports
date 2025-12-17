"""
ESPN Odds Scraper - Get NBA odds from ESPN's free API
Provides opening and closing lines for current season games
"""
import requests
import sqlite3
from datetime import datetime, timedelta
import time


def get_nba_games_for_date(date_str):
    """Get list of NBA game IDs for a specific date (YYYYMMDD format)"""
    url = f'https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events?dates={date_str}'
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            game_ids = []
            for item in data.get('items', []):
                ref = item.get('$ref', '')
                # Extract game ID from URL like .../events/401810193?...
                if '/events/' in ref:
                    game_id = ref.split('/events/')[1].split('?')[0]
                    game_ids.append(game_id)
            return game_ids
    except Exception as e:
        print(f"Error getting games for {date_str}: {e}")
    return []


def get_game_odds(game_id):
    """Get odds for a specific game from ESPN API"""
    url = f'https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/odds'
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get('items', [])
            if items:
                odds = items[0]  # Get first provider (usually DraftKings)

                # Extract data
                result = {
                    'game_id': int(game_id),
                    'source': odds.get('provider', {}).get('name', 'ESPN'),
                    'spread': odds.get('spread'),  # Current spread (home perspective)
                    'over_under': odds.get('overUnder'),
                    'over_odds': odds.get('overOdds'),
                    'under_odds': odds.get('underOdds'),
                }

                # Home team odds
                home_odds = odds.get('homeTeamOdds', {})
                if home_odds:
                    result['home_favorite'] = home_odds.get('favorite', False)
                    result['home_moneyline'] = home_odds.get('moneyLine')
                    result['home_spread_odds'] = home_odds.get('spreadOdds')

                    # Helper to parse moneyline (handles "EVEN" = +100)
                    def parse_ml(val):
                        if not val:
                            return None
                        if val.upper() == 'EVEN':
                            return 100
                        return int(val.replace('+', ''))

                    # Opening lines
                    home_open = home_odds.get('open', {})
                    if home_open:
                        ps = home_open.get('pointSpread', {})
                        result['opening_spread_home'] = float(ps.get('american', '0').replace('+', '')) if ps.get('american') else None
                        ml = home_open.get('moneyLine', {})
                        result['opening_ml_home'] = parse_ml(ml.get('american'))

                    # Current lines
                    home_current = home_odds.get('current', {})
                    if home_current:
                        ml = home_current.get('moneyLine', {})
                        result['closing_ml_home'] = parse_ml(ml.get('american'))

                # Away team odds
                away_odds = odds.get('awayTeamOdds', {})
                if away_odds:
                    result['away_moneyline'] = away_odds.get('moneyLine')
                    result['away_spread_odds'] = away_odds.get('spreadOdds')

                    # Helper to parse moneyline (handles "EVEN" = +100)
                    def parse_ml(val):
                        if not val:
                            return None
                        if val.upper() == 'EVEN':
                            return 100
                        return int(val.replace('+', ''))

                    # Opening lines
                    away_open = away_odds.get('open', {})
                    if away_open:
                        ml = away_open.get('moneyLine', {})
                        result['opening_ml_away'] = parse_ml(ml.get('american'))

                    # Current lines
                    away_current = away_odds.get('current', {})
                    if away_current:
                        ml = away_current.get('moneyLine', {})
                        result['closing_ml_away'] = parse_ml(ml.get('american'))

                return result
    except Exception as e:
        print(f"Error getting odds for game {game_id}: {e}")
    return None


def save_odds_to_db(odds_data, db_path='nba_games.db'):
    """Save odds data to odds_and_predictions table"""
    if not odds_data:
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    game_id = odds_data['game_id']

    # Check if game exists in our database
    cursor.execute('SELECT game_id FROM games WHERE game_id = ?', (game_id,))
    if not cursor.fetchone():
        conn.close()
        return False

    # Get current values
    opening_spread = odds_data.get('opening_spread_home')
    latest_spread = odds_data.get('spread')
    opening_total = odds_data.get('over_under')
    latest_total = odds_data.get('over_under')

    # Calculate movements
    spread_movement = (latest_spread - opening_spread) if (opening_spread and latest_spread) else None
    total_movement = (latest_total - opening_total) if (opening_total and latest_total) else None

    # Upsert odds into odds_and_predictions table
    cursor.execute('''
        INSERT INTO odds_and_predictions (
            game_id, source,
            opening_spread, latest_spread,
            opening_total, latest_total,
            opening_moneyline_home, latest_moneyline_home,
            opening_moneyline_away, latest_moneyline_away,
            spread_movement, total_movement,
            odds_updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id) DO UPDATE SET
            source = excluded.source,
            latest_spread = excluded.latest_spread,
            latest_total = excluded.latest_total,
            latest_moneyline_home = excluded.latest_moneyline_home,
            latest_moneyline_away = excluded.latest_moneyline_away,
            spread_movement = excluded.spread_movement,
            total_movement = excluded.total_movement,
            odds_updated_at = excluded.odds_updated_at
    ''', (
        game_id,
        odds_data.get('source', 'ESPN'),
        opening_spread,
        latest_spread,
        opening_total,
        latest_total,
        odds_data.get('opening_ml_home'),
        odds_data.get('closing_ml_home') or odds_data.get('home_moneyline'),
        odds_data.get('opening_ml_away'),
        odds_data.get('closing_ml_away') or odds_data.get('away_moneyline'),
        spread_movement,
        total_movement,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()
    return True


def update_prediction_cache(game_id, spread, total, db_path='users.db'):
    """Update prediction cache with Vegas odds"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE prediction_cache
        SET vegas_spread = COALESCE(?, vegas_spread),
            vegas_total = COALESCE(?, vegas_total)
        WHERE game_id = ? AND sport = 'NBA'
    ''', (spread, total, game_id))

    updated = cursor.rowcount
    conn.commit()
    conn.close()
    return updated > 0


def scrape_nba_odds_for_date_range(start_date, end_date, update_cache=True):
    """Scrape NBA odds for a range of dates"""
    current = start_date
    total_saved = 0
    total_games = 0

    while current <= end_date:
        date_str = current.strftime('%Y%m%d')
        print(f"\n{current.strftime('%Y-%m-%d')}:")

        game_ids = get_nba_games_for_date(date_str)
        print(f"  Found {len(game_ids)} games")

        for game_id in game_ids:
            total_games += 1
            odds = get_game_odds(game_id)

            if odds:
                saved = save_odds_to_db(odds)
                if saved:
                    total_saved += 1
                    print(f"    Game {game_id}: Spread {odds.get('spread')}, O/U {odds.get('over_under')}")

                    if update_cache:
                        update_prediction_cache(
                            int(game_id),
                            odds.get('spread'),
                            odds.get('over_under')
                        )
                else:
                    print(f"    Game {game_id}: Not in database (skipped)")
            else:
                print(f"    Game {game_id}: No odds available")

            time.sleep(0.2)  # Be nice to ESPN

        current += timedelta(days=1)

    return total_saved, total_games


def scrape_current_season_odds():
    """Scrape odds for all NBA games in current 2025 season from database"""
    conn = sqlite3.connect('nba_games.db')
    cursor = conn.cursor()

    # Get all 2025 game IDs
    cursor.execute('''
        SELECT game_id, date, home_team_id, away_team_id
        FROM games
        WHERE season = 2025
        ORDER BY date
    ''')
    games = cursor.fetchall()
    conn.close()

    print(f"Found {len(games)} NBA 2025 games in database")

    total_saved = 0
    total_with_odds = 0

    for game_id, date, home_id, away_id in games:
        odds = get_game_odds(game_id)

        if odds:
            total_with_odds += 1
            saved = save_odds_to_db(odds)
            if saved:
                total_saved += 1
                update_prediction_cache(game_id, odds.get('spread'), odds.get('over_under'))

        if total_with_odds % 50 == 0:
            print(f"  Processed {total_with_odds} games with odds...")

        time.sleep(0.1)

    print(f"\nResults:")
    print(f"  Games with odds: {total_with_odds}")
    print(f"  Saved to database: {total_saved}")

    return total_saved


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Scrape NBA odds from ESPN')
    parser.add_argument('--date', type=str, help='Specific date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=7, help='Number of days to scrape (default 7)')
    parser.add_argument('--season', action='store_true', help='Scrape entire 2025 season')

    args = parser.parse_args()

    print("="*60)
    print("ESPN NBA ODDS SCRAPER")
    print("="*60)

    if args.season:
        scrape_current_season_odds()
    elif args.date:
        date = datetime.strptime(args.date, '%Y-%m-%d')
        scrape_nba_odds_for_date_range(date, date)
    else:
        # Default: scrape last N days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        scrape_nba_odds_for_date_range(start_date, end_date)
