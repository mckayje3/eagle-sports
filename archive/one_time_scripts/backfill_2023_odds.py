"""
Backfill 2023 NFL and CFB odds from The Odds API
"""
import json
import time
import requests
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2023 NFL Season dates (Sep 2023 - Feb 2024)
NFL_DATES_2023 = [
    ('2023-09-07T17:00:00Z', '2023 NFL Week 1'),
    ('2023-09-10T17:00:00Z', '2023 NFL Week 1 Sunday'),
    ('2023-09-14T17:00:00Z', '2023 NFL Week 2'),
    ('2023-09-17T17:00:00Z', '2023 NFL Week 2 Sunday'),
    ('2023-09-21T17:00:00Z', '2023 NFL Week 3'),
    ('2023-09-24T17:00:00Z', '2023 NFL Week 3 Sunday'),
    ('2023-09-28T17:00:00Z', '2023 NFL Week 4'),
    ('2023-10-01T17:00:00Z', '2023 NFL Week 4 Sunday'),
    ('2023-10-05T17:00:00Z', '2023 NFL Week 5'),
    ('2023-10-08T17:00:00Z', '2023 NFL Week 5 Sunday'),
    ('2023-10-12T17:00:00Z', '2023 NFL Week 6'),
    ('2023-10-15T17:00:00Z', '2023 NFL Week 6 Sunday'),
    ('2023-10-19T17:00:00Z', '2023 NFL Week 7'),
    ('2023-10-22T17:00:00Z', '2023 NFL Week 7 Sunday'),
    ('2023-10-26T17:00:00Z', '2023 NFL Week 8'),
    ('2023-10-29T17:00:00Z', '2023 NFL Week 8 Sunday'),
    ('2023-11-02T17:00:00Z', '2023 NFL Week 9'),
    ('2023-11-05T17:00:00Z', '2023 NFL Week 9 Sunday'),
    ('2023-11-09T17:00:00Z', '2023 NFL Week 10'),
    ('2023-11-12T17:00:00Z', '2023 NFL Week 10 Sunday'),
    ('2023-11-16T17:00:00Z', '2023 NFL Week 11'),
    ('2023-11-19T17:00:00Z', '2023 NFL Week 11 Sunday'),
    ('2023-11-23T17:00:00Z', '2023 NFL Week 12 Thanksgiving'),
    ('2023-11-26T17:00:00Z', '2023 NFL Week 12 Sunday'),
    ('2023-11-30T17:00:00Z', '2023 NFL Week 13'),
    ('2023-12-03T17:00:00Z', '2023 NFL Week 13 Sunday'),
    ('2023-12-07T17:00:00Z', '2023 NFL Week 14'),
    ('2023-12-10T17:00:00Z', '2023 NFL Week 14 Sunday'),
    ('2023-12-14T17:00:00Z', '2023 NFL Week 15'),
    ('2023-12-17T17:00:00Z', '2023 NFL Week 15 Sunday'),
    ('2023-12-21T17:00:00Z', '2023 NFL Week 16'),
    ('2023-12-24T17:00:00Z', '2023 NFL Week 16 Sunday'),
    ('2023-12-25T17:00:00Z', '2023 NFL Week 16 Christmas'),
    ('2023-12-31T17:00:00Z', '2023 NFL Week 17'),
    ('2024-01-06T17:00:00Z', '2023 NFL Week 18'),
    ('2024-01-07T17:00:00Z', '2023 NFL Week 18 Sunday'),
    ('2024-01-13T17:00:00Z', '2023 NFL Wild Card'),
    ('2024-01-14T17:00:00Z', '2023 NFL Wild Card Sunday'),
    ('2024-01-20T17:00:00Z', '2023 NFL Divisional'),
    ('2024-01-21T17:00:00Z', '2023 NFL Divisional Sunday'),
    ('2024-01-28T17:00:00Z', '2023 NFL Conference Championships'),
    ('2024-02-11T17:00:00Z', '2023 Super Bowl LVIII'),
]

# 2023 CFB Season dates (Aug 2023 - Jan 2024)
CFB_DATES_2023 = [
    ('2023-08-26T17:00:00Z', '2023 CFB Week 0'),
    ('2023-09-02T17:00:00Z', '2023 CFB Week 1'),
    ('2023-09-09T17:00:00Z', '2023 CFB Week 2'),
    ('2023-09-16T17:00:00Z', '2023 CFB Week 3'),
    ('2023-09-23T17:00:00Z', '2023 CFB Week 4'),
    ('2023-09-30T17:00:00Z', '2023 CFB Week 5'),
    ('2023-10-07T17:00:00Z', '2023 CFB Week 6'),
    ('2023-10-14T17:00:00Z', '2023 CFB Week 7'),
    ('2023-10-21T17:00:00Z', '2023 CFB Week 8'),
    ('2023-10-28T17:00:00Z', '2023 CFB Week 9'),
    ('2023-11-04T17:00:00Z', '2023 CFB Week 10'),
    ('2023-11-11T17:00:00Z', '2023 CFB Week 11'),
    ('2023-11-18T17:00:00Z', '2023 CFB Week 12'),
    ('2023-11-25T17:00:00Z', '2023 CFB Week 13'),
    ('2023-12-02T17:00:00Z', '2023 CFB Week 14 (Conf Champs)'),
    ('2023-12-16T17:00:00Z', '2023 CFB Bowl Week 1'),
    ('2023-12-19T17:00:00Z', '2023 CFB Bowl Week'),
    ('2023-12-20T17:00:00Z', '2023 CFB Bowl Week'),
    ('2023-12-21T17:00:00Z', '2023 CFB Bowl Week'),
    ('2023-12-22T17:00:00Z', '2023 CFB Bowl Week'),
    ('2023-12-23T17:00:00Z', '2023 CFB Bowl Week'),
    ('2023-12-26T17:00:00Z', '2023 CFB Bowl Week'),
    ('2023-12-27T17:00:00Z', '2023 CFB Bowl Week'),
    ('2023-12-28T17:00:00Z', '2023 CFB Bowl Week'),
    ('2023-12-29T17:00:00Z', '2023 CFB Bowl Week'),
    ('2023-12-30T17:00:00Z', '2023 CFB Bowl Week'),
    ('2024-01-01T17:00:00Z', '2023 CFB NY6 Bowls'),
    ('2024-01-08T17:00:00Z', '2023 CFB National Championship'),
]


class SimpleGameMatcher:
    """Simple game matcher for NFL/CFB"""

    # Team name mappings (Odds API -> ESPN)
    NFL_MAPPINGS = {
        'Arizona Cardinals': 'Arizona Cardinals',
        'Atlanta Falcons': 'Atlanta Falcons',
        'Baltimore Ravens': 'Baltimore Ravens',
        'Buffalo Bills': 'Buffalo Bills',
        'Carolina Panthers': 'Carolina Panthers',
        'Chicago Bears': 'Chicago Bears',
        'Cincinnati Bengals': 'Cincinnati Bengals',
        'Cleveland Browns': 'Cleveland Browns',
        'Dallas Cowboys': 'Dallas Cowboys',
        'Denver Broncos': 'Denver Broncos',
        'Detroit Lions': 'Detroit Lions',
        'Green Bay Packers': 'Green Bay Packers',
        'Houston Texans': 'Houston Texans',
        'Indianapolis Colts': 'Indianapolis Colts',
        'Jacksonville Jaguars': 'Jacksonville Jaguars',
        'Kansas City Chiefs': 'Kansas City Chiefs',
        'Las Vegas Raiders': 'Las Vegas Raiders',
        'Los Angeles Chargers': 'Los Angeles Chargers',
        'Los Angeles Rams': 'Los Angeles Rams',
        'Miami Dolphins': 'Miami Dolphins',
        'Minnesota Vikings': 'Minnesota Vikings',
        'New England Patriots': 'New England Patriots',
        'New Orleans Saints': 'New Orleans Saints',
        'New York Giants': 'New York Giants',
        'New York Jets': 'New York Jets',
        'Philadelphia Eagles': 'Philadelphia Eagles',
        'Pittsburgh Steelers': 'Pittsburgh Steelers',
        'San Francisco 49ers': 'San Francisco 49ers',
        'Seattle Seahawks': 'Seattle Seahawks',
        'Tampa Bay Buccaneers': 'Tampa Bay Buccaneers',
        'Tennessee Titans': 'Tennessee Titans',
        'Washington Commanders': 'Washington Commanders',
    }

    def __init__(self, db_path):
        self.db_path = db_path

    def match_game(self, odds_game):
        """Match an odds game to database game"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        home_team = odds_game.get('home_team', '')
        away_team = odds_game.get('away_team', '')
        game_date = odds_game.get('commence_time', '')[:10]

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


def fetch_historical_odds(api_key, sport_key, date, markets=['spreads', 'totals']):
    """Fetch historical odds directly from API"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds-history"
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


def backfill_odds(dates, sport_key, db_path, sport_name):
    """Backfill odds for a list of dates"""
    print(f"\n{'='*80}")
    print(f"{sport_name} 2023 ODDS BACKFILL")
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
        print("ERROR: No API key in config file")
        return 0

    matcher = SimpleGameMatcher(db_path)
    conn = sqlite3.connect(db_path)

    print(f"Total dates to fetch: {len(dates)}")

    # Track stats
    total_games = 0
    total_matched = 0
    total_saved = 0
    remaining = None

    for i, (date, description) in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] {description} ({date[:10]})")

        # Fetch historical odds
        games, remaining = fetch_historical_odds(api_key, sport_key, date)

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
                    conn.commit()
                    saved += 1
                except Exception as e:
                    print(f"  Error saving game_id {game_id}: {e}")

        total_matched += matched
        total_saved += saved
        print(f"  Matched: {matched} | Saved: {saved}")

        # Rate limiting
        if i < len(dates):
            time.sleep(1)

    conn.close()

    print(f"\n{'='*80}")
    print(f"{sport_name} 2023 ODDS BACKFILL COMPLETE")
    print(f"{'='*80}")
    print(f"Total games fetched: {total_games}")
    print(f"Total matched: {total_matched}")
    print(f"Total saved: {total_saved}")
    print(f"API requests remaining: {remaining}")

    return total_saved


def main():
    import sys

    # Check what to backfill
    if len(sys.argv) > 1:
        sport = sys.argv[1].lower()
    else:
        sport = 'both'

    if sport in ['nfl', 'both']:
        backfill_odds(
            NFL_DATES_2023,
            'americanfootball_nfl',
            'nfl_games.db',
            'NFL'
        )

    if sport in ['cfb', 'both']:
        backfill_odds(
            CFB_DATES_2023,
            'americanfootball_ncaaf',
            'cfb_games.db',
            'CFB'
        )


if __name__ == '__main__':
    main()
