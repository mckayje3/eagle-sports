"""Try to get odds for Dec 3 missing games by querying Dec 2 evening times"""
import sqlite3
import requests
import json
from datetime import datetime

# API key
API_KEY = '2c938aee93cd51153a6291528e9e5765'

# Games on Dec 3 at early UTC times are Dec 2 evening US time
# Let's query Dec 2 at different times
dates_to_try = ['2025-12-02T22:00:00Z', '2025-12-02T18:00:00Z', '2025-12-03T02:00:00Z']

all_api_games = []
for date in dates_to_try:
    url = f'https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds'
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'spreads,totals',
        'oddsFormat': 'american',
        'date': date
    }

    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            games = data.get('data', [])
            print(f'Date {date}: {len(games)} games')
            for g in games:
                home = g.get('home_team', '')
                away = g.get('away_team', '')
                commence = g.get('commence_time', '')
                all_api_games.append({
                    'home': home,
                    'away': away,
                    'commence': commence,
                    'bookmakers': g.get('bookmakers', []),
                    'query_date': date
                })
        else:
            print(f'Date {date}: Error {resp.status_code}')
    except Exception as e:
        print(f'Date {date}: Exception {e}')

print(f'\nTotal API games collected: {len(all_api_games)}')

# Normalize team names
def normalize(name):
    name = name.lower().strip()
    if 'clippers' in name: return 'clippers'
    if 'lakers' in name: return 'lakers'
    if '76ers' in name or 'sixers' in name: return '76ers'
    if 'blazers' in name: return 'blazers'
    if 'trail' in name: return 'blazers'
    if 'celtics' in name: return 'celtics'
    if 'knicks' in name: return 'knicks'
    if 'wizards' in name: return 'wizards'
    if 'grizzlies' in name: return 'grizzlies'
    if 'spurs' in name: return 'spurs'
    if 'raptors' in name: return 'raptors'
    parts = name.split()
    return parts[-1] if parts else name

# The 4 missing games (away @ home format)
missing = [
    ('Washington Wizards', 'Philadelphia 76ers'),
    ('Portland Trail Blazers', 'Toronto Raptors'),
    ('New York Knicks', 'Boston Celtics'),
    ('Memphis Grizzlies', 'San Antonio Spurs')
]

# Connect to database
conn = sqlite3.connect('nba_games.db')
cursor = conn.cursor()

users_conn = sqlite3.connect('users.db')
users_cursor = users_conn.cursor()

print('\nSearching for missing games...')
saved_count = 0

for away_db, home_db in missing:
    away_norm = normalize(away_db)
    home_norm = normalize(home_db)
    print(f'\nLooking for: {away_db} @ {home_db} (normalized: {away_norm} @ {home_norm})')

    found = False
    for api_game in all_api_games:
        api_home = normalize(api_game['home'])
        api_away = normalize(api_game['away'])

        if api_home == home_norm and api_away == away_norm:
            print(f'  FOUND! {api_game["away"]} @ {api_game["home"]} at {api_game["commence"]}')
            print(f'  Query date: {api_game["query_date"]}')
            print(f'  Bookmakers: {len(api_game["bookmakers"])}')

            # Extract odds
            spreads = []
            totals = []
            for book in api_game['bookmakers']:
                for market in book.get('markets', []):
                    if market.get('key') == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == api_game['home']:
                                spreads.append(outcome.get('point', 0))
                    elif market.get('key') == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == 'Over':
                                totals.append(outcome.get('point', 0))

            avg_spread = sum(spreads) / len(spreads) if spreads else None
            avg_total = sum(totals) / len(totals) if totals else None

            print(f'  Spreads: {spreads} -> Avg: {avg_spread}')
            print(f'  Totals: {totals} -> Avg: {avg_total}')

            # Find game_id in database
            cursor.execute('''
                SELECT g.game_id FROM games g
                JOIN teams ht ON g.home_team_id = ht.team_id
                JOIN teams at ON g.away_team_id = at.team_id
                WHERE ht.display_name = ? AND at.display_name = ?
                AND g.season = 2025
            ''', (home_db, away_db))

            row = cursor.fetchone()
            if row:
                game_id = row[0]
                print(f'  Game ID: {game_id}')

                # Save to game_odds
                if avg_spread is not None or avg_total is not None:
                    cursor.execute('''
                        INSERT OR REPLACE INTO game_odds (game_id, sportsbook, opening_spread_home, closing_spread_home, opening_total, closing_total)
                        VALUES (?, 'consensus', ?, ?, ?, ?)
                    ''', (game_id, avg_spread, avg_spread, avg_total, avg_total))

                    # Update users.db
                    users_cursor.execute('''
                        UPDATE prediction_cache
                        SET vegas_spread = COALESCE(?, vegas_spread),
                            vegas_total = COALESCE(?, vegas_total)
                        WHERE game_id = ? AND sport = 'NBA'
                    ''', (avg_spread, avg_total, game_id))

                    print(f'  SAVED odds!')
                    saved_count += 1
            else:
                print(f'  Game ID not found in database')

            found = True
            break

    if not found:
        print(f'  NOT FOUND in any API response')

conn.commit()
conn.close()
users_conn.commit()
users_conn.close()

print(f'\n{"="*60}')
print(f'Saved odds for {saved_count} games')
print(f'{"="*60}')

# Check final status
users_conn = sqlite3.connect('users.db')
users_cursor = users_conn.cursor()
users_cursor.execute('''
    SELECT COUNT(*) as total,
           SUM(CASE WHEN vegas_spread != 0 AND vegas_spread IS NOT NULL THEN 1 ELSE 0 END) as with_spread,
           SUM(game_completed) as completed
    FROM prediction_cache
    WHERE sport = 'NBA' AND season = 2025 AND week = 7
''')
row = users_cursor.fetchone()
print(f'\nWeek 7 status: {row[1]}/{row[0]} games with odds, {row[2]} completed')
users_conn.close()
