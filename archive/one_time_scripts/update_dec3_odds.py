"""Update the 4 Dec 3 games with odds from web search"""
import sqlite3

# The 4 missing games and their odds (from web search)
# Spread is relative to home team (negative = home favored)
games_odds = [
    # (away_team, home_team, spread, total)
    ('Washington Wizards', 'Philadelphia 76ers', -13.5, 234.5),
    ('Portland Trail Blazers', 'Toronto Raptors', -5.0, 232.5),
    ('New York Knicks', 'Boston Celtics', -1.5, 230.5),
    ('Memphis Grizzlies', 'San Antonio Spurs', -5.5, 230.5),
]

# Connect to databases
nba_conn = sqlite3.connect('nba_games.db')
nba_cursor = nba_conn.cursor()

users_conn = sqlite3.connect('users.db')
users_cursor = users_conn.cursor()

print("Updating Dec 3 games with odds from web search...")
print("="*60)

for away_team, home_team, spread, total in games_odds:
    # Find game_id
    nba_cursor.execute('''
        SELECT g.game_id FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE ht.display_name = ? AND at.display_name = ?
        AND g.season = 2025
        AND g.date LIKE '2025-12-03%'
    ''', (home_team, away_team))

    row = nba_cursor.fetchone()
    if row:
        game_id = row[0]
        print(f"\n{away_team} @ {home_team}")
        print(f"  Game ID: {game_id}")
        print(f"  Spread: {spread}, Total: {total}")

        # Update nba_games.db game_odds
        nba_cursor.execute('''
            INSERT OR REPLACE INTO game_odds (game_id, source, opening_spread_home, closing_spread_home, opening_total, closing_total)
            VALUES (?, 'websearch', ?, ?, ?, ?)
        ''', (game_id, spread, spread, total, total))

        # Update users.db prediction_cache
        users_cursor.execute('''
            UPDATE prediction_cache
            SET vegas_spread = ?,
                vegas_total = ?
            WHERE game_id = ? AND sport = 'NBA'
        ''', (spread, total, game_id))

        print(f"  SAVED!")
    else:
        print(f"\n{away_team} @ {home_team}")
        print(f"  NOT FOUND in database")

nba_conn.commit()
nba_conn.close()

users_conn.commit()
users_conn.close()

print("\n" + "="*60)
print("Done!")

# Check final status
users_conn = sqlite3.connect('users.db')
users_cursor = users_conn.cursor()
users_cursor.execute('''
    SELECT
        SUM(CASE WHEN game_completed = 1 THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN game_completed = 1 AND vegas_spread != 0 AND vegas_spread IS NOT NULL THEN 1 ELSE 0 END) as completed_with_odds,
        COUNT(*) as total,
        SUM(CASE WHEN vegas_spread != 0 AND vegas_spread IS NOT NULL THEN 1 ELSE 0 END) as total_with_odds
    FROM prediction_cache
    WHERE sport = 'NBA' AND season = 2025 AND week = 7
''')
row = users_cursor.fetchone()
print(f"\nWeek 7 final status: {row[1]}/{row[0]} completed games have odds, {row[3]}/{row[2]} total games have odds")
users_conn.close()
