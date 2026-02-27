"""
Non-interactive backfill of 2025 CFB odds
"""
import json
import time
from datetime import datetime
from odds_api_scraper import OddsAPIScraper
from game_matcher import GameMatcher
from database import FootballDatabase

# 2025 CFB Season weeks (up to current - Week 14)
DATES_2025 = [
    ('2025-08-23T17:00:00Z', '2025 Week 0'),
    ('2025-08-30T17:00:00Z', '2025 Week 1'),
    ('2025-09-06T17:00:00Z', '2025 Week 2'),
    ('2025-09-13T17:00:00Z', '2025 Week 3'),
    ('2025-09-20T17:00:00Z', '2025 Week 4'),
    ('2025-09-27T17:00:00Z', '2025 Week 5'),
    ('2025-10-04T17:00:00Z', '2025 Week 6'),
    ('2025-10-11T17:00:00Z', '2025 Week 7'),
    ('2025-10-18T17:00:00Z', '2025 Week 8'),
    ('2025-10-25T17:00:00Z', '2025 Week 9'),
    ('2025-11-01T17:00:00Z', '2025 Week 10'),
    ('2025-11-08T17:00:00Z', '2025 Week 11'),
    ('2025-11-15T17:00:00Z', '2025 Week 12'),
    ('2025-11-22T17:00:00Z', '2025 Week 13'),
    ('2025-11-29T17:00:00Z', '2025 Week 14'),
]


def backfill_2025_odds():
    """Backfill Vegas odds for 2025 CFB season (non-interactive)"""
    print("\n" + "="*80)
    print("2025 CFB SEASON VEGAS ODDS BACKFILL")
    print("="*80 + "\n")

    # Load API key
    try:
        with open('odds_api_config.json', 'r') as f:
            config = json.load(f)
            api_key = config.get('api_key')
    except FileNotFoundError:
        print("ERROR: odds_api_config.json not found!")
        return

    if not api_key:
        print("ERROR: No API key in config file")
        return

    # Initialize
    scraper = OddsAPIScraper(api_key=api_key)
    matcher = GameMatcher()
    db = FootballDatabase()
    db.connect()

    # Check remaining requests
    remaining = scraper.get_remaining_requests()
    print(f"Starting API requests remaining: {remaining}")
    print(f"Total dates to fetch: {len(DATES_2025)}")
    print(f"Estimated cost: ~{len(DATES_2025) * 20} requests\n")

    if remaining and remaining < len(DATES_2025) * 20:
        print("WARNING: May not have enough API requests!")

    # Track stats
    total_games = 0
    total_matched = 0
    total_saved = 0

    # Process each week
    for i, (date, description) in enumerate(DATES_2025, 1):
        print(f"\n[{i}/{len(DATES_2025)}] {description} ({date[:10]})")

        # Fetch historical odds
        games = scraper.fetch_historical_odds(
            date=date,
            markets=['spreads', 'totals']
        )

        if not games:
            print(f"  No games found")
            continue

        print(f"  Retrieved {len(games)} games")
        total_games += len(games)

        matched = 0
        saved = 0

        for game in games:
            home = game.get('home_team')
            away = game.get('away_team')

            if not home or not away:
                continue

            game_id = matcher.match_odds_api_game(game)

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
                    cursor = db.conn.cursor()
                    cursor.execute('SELECT id FROM game_odds WHERE game_id = ? AND source = ?',
                                   (game_id, 'TheOddsAPI'))
                    existing = cursor.fetchone()

                    if existing:
                        cursor.execute('''
                            UPDATE game_odds SET
                                closing_spread_home = ?,
                                closing_total = ?,
                                updated_at = ?
                            WHERE game_id = ? AND source = ?
                        ''', (
                            round(avg_spread, 1) if avg_spread else None,
                            round(avg_total, 1) if avg_total else None,
                            date,
                            game_id,
                            'TheOddsAPI'
                        ))
                    else:
                        cursor.execute('''
                            INSERT INTO game_odds
                            (game_id, source, closing_spread_home, closing_total, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            game_id,
                            'TheOddsAPI',
                            round(avg_spread, 1) if avg_spread else None,
                            round(avg_total, 1) if avg_total else None,
                            date
                        ))
                    db.conn.commit()
                    saved += 1
                except Exception as e:
                    print(f"  Error saving game_id {game_id}: {e}")

        total_matched += matched
        total_saved += saved
        print(f"  Matched: {matched} | Saved: {saved}")

        if i < len(DATES_2025):
            time.sleep(2)

    db.close()

    print("\n" + "="*80)
    print("2025 CFB ODDS BACKFILL COMPLETE")
    print("="*80)
    print(f"Total games fetched: {total_games}")
    print(f"Total matched: {total_matched}")
    print(f"Total saved: {total_saved}")
    print(f"API requests remaining: {scraper.get_remaining_requests()}")

    return total_saved


if __name__ == '__main__':
    backfill_2025_odds()
