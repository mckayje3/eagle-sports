"""
Backfill odds for 2024 CFB Bowl Games (Weeks 16-20)
"""
import json
import time
from odds_api_scraper import OddsAPIScraper
from game_matcher import GameMatcher
from database import FootballDatabase

# 2024 Bowl Season dates
BOWL_DATES_2024 = [
    ('2024-12-14T17:00:00Z', '2024 Bowl Week 1 (Dec 14)'),
    ('2024-12-17T17:00:00Z', '2024 Bowl Week (Dec 17)'),
    ('2024-12-18T17:00:00Z', '2024 Bowl Week (Dec 18)'),
    ('2024-12-19T17:00:00Z', '2024 Bowl Week (Dec 19)'),
    ('2024-12-20T17:00:00Z', '2024 Bowl Week (Dec 20)'),
    ('2024-12-21T17:00:00Z', '2024 Bowl Week (Dec 21)'),
    ('2024-12-23T17:00:00Z', '2024 Bowl Week (Dec 23)'),
    ('2024-12-24T17:00:00Z', '2024 Bowl Week (Dec 24)'),
    ('2024-12-26T17:00:00Z', '2024 Bowl Week (Dec 26)'),
    ('2024-12-27T17:00:00Z', '2024 Bowl Week (Dec 27)'),
    ('2024-12-28T17:00:00Z', '2024 Bowl Week (Dec 28)'),
    ('2024-12-30T17:00:00Z', '2024 Bowl Week (Dec 30)'),
    ('2024-12-31T17:00:00Z', '2024 Bowl Week (Dec 31)'),
    ('2025-01-01T17:00:00Z', '2024 NY6 Bowls (Jan 1)'),
    ('2025-01-02T17:00:00Z', '2024 Bowl (Jan 2)'),
    ('2025-01-04T17:00:00Z', '2024 Bowl (Jan 4)'),
    ('2025-01-09T17:00:00Z', '2024 Playoff Semifinal (Jan 9)'),
    ('2025-01-10T17:00:00Z', '2024 Playoff Semifinal (Jan 10)'),
    ('2025-01-20T17:00:00Z', '2024 National Championship (Jan 20)'),
]


def backfill_bowl_odds():
    """Backfill Vegas odds for 2024 bowl season"""
    print("\n" + "="*80)
    print("2024 CFB BOWL SEASON VEGAS ODDS BACKFILL")
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
    print(f"Total dates to fetch: {len(BOWL_DATES_2024)}")
    print(f"Estimated cost: ~{len(BOWL_DATES_2024) * 10} requests\n")

    if remaining and remaining < len(BOWL_DATES_2024) * 10:
        print("WARNING: May not have enough API requests!")

    # Track stats
    total_games = 0
    total_matched = 0
    total_saved = 0

    # Process each date
    for i, (date, description) in enumerate(BOWL_DATES_2024, 1):
        print(f"\n[{i}/{len(BOWL_DATES_2024)}] {description}")

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
                print(f"    No match: {away} @ {home}")
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
                    print(f"    Saved: {away} @ {home} | Spread: {avg_spread:.1f}, Total: {avg_total:.1f}")
                except Exception as e:
                    print(f"  Error saving game_id {game_id}: {e}")

        total_matched += matched
        total_saved += saved
        print(f"  Matched: {matched} | Saved: {saved}")

        if i < len(BOWL_DATES_2024):
            time.sleep(1)

    db.close()

    print("\n" + "="*80)
    print("2024 BOWL ODDS BACKFILL COMPLETE")
    print("="*80)
    print(f"Total games fetched: {total_games}")
    print(f"Total matched: {total_matched}")
    print(f"Total saved: {total_saved}")
    print(f"API requests remaining: {scraper.get_remaining_requests()}")

    return total_saved


if __name__ == '__main__':
    backfill_bowl_odds()
