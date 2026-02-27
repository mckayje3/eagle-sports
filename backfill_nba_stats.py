"""
Backfill NBA team game stats from ESPN API
Fetches box score statistics for all completed games
"""
import requests
import sqlite3
import time
import sys

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def fetch_game_stats(game_id):
    """Fetch detailed stats for a game from ESPN summary API"""
    url = f"{BASE_URL}/summary?event={game_id}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception as e:
        print(f"  Error fetching game {game_id}: {e}")
        return None


def parse_team_stats(boxscore_data, team_id):
    """Parse team statistics from boxscore data"""
    stats = {
        'team_id': team_id,
        'points': None,
        'field_goals_made': None,
        'field_goals_attempted': None,
        'field_goal_pct': None,
        'three_pointers_made': None,
        'three_pointers_attempted': None,
        'three_point_pct': None,
        'free_throws_made': None,
        'free_throws_attempted': None,
        'free_throw_pct': None,
        'offensive_rebounds': None,
        'defensive_rebounds': None,
        'total_rebounds': None,
        'assists': None,
        'steals': None,
        'blocks': None,
        'turnovers': None,
        'personal_fouls': None,
        'points_in_paint': None,
        'fast_break_points': None,
        'bench_points': None
    }

    if not boxscore_data:
        return stats

    # Find the team's stats in the boxscore
    for team in boxscore_data.get('teams', []):
        if team.get('team', {}).get('id') == str(team_id):
            team_stats = team.get('statistics', [])

            for stat in team_stats:
                name = stat.get('name', '').lower()
                value = stat.get('displayValue', '')

                try:
                    if name == 'points':
                        stats['points'] = int(value) if value else None
                    elif name == 'fieldgoalsmade-fieldgoalsattempted':
                        parts = value.split('-')
                        if len(parts) == 2:
                            stats['field_goals_made'] = int(parts[0])
                            stats['field_goals_attempted'] = int(parts[1])
                    elif name == 'fieldgoalpct':
                        stats['field_goal_pct'] = float(value) if value else None
                    elif name == 'threepointfieldgoalsmade-threepointfieldgoalsattempted':
                        parts = value.split('-')
                        if len(parts) == 2:
                            stats['three_pointers_made'] = int(parts[0])
                            stats['three_pointers_attempted'] = int(parts[1])
                    elif name == 'threepointfieldgoalpct':
                        stats['three_point_pct'] = float(value) if value else None
                    elif name == 'freethrowsmade-freethrowsattempted':
                        parts = value.split('-')
                        if len(parts) == 2:
                            stats['free_throws_made'] = int(parts[0])
                            stats['free_throws_attempted'] = int(parts[1])
                    elif name == 'freethrowpct':
                        stats['free_throw_pct'] = float(value) if value else None
                    elif name == 'offensiverebounds':
                        stats['offensive_rebounds'] = int(value) if value else None
                    elif name == 'defensiverebounds':
                        stats['defensive_rebounds'] = int(value) if value else None
                    elif name == 'totalrebounds' or name == 'rebounds':
                        stats['total_rebounds'] = int(value) if value else None
                    elif name == 'assists':
                        stats['assists'] = int(value) if value else None
                    elif name == 'steals':
                        stats['steals'] = int(value) if value else None
                    elif name == 'blocks':
                        stats['blocks'] = int(value) if value else None
                    elif name == 'turnovers':
                        stats['turnovers'] = int(value) if value else None
                    elif name == 'fouls' or name == 'personalfouls':
                        stats['personal_fouls'] = int(value) if value else None
                    elif name == 'pointsinpaint':
                        stats['points_in_paint'] = int(value) if value else None
                    elif name == 'fastbreakpoints':
                        stats['fast_break_points'] = int(value) if value else None
                    elif name == 'benchpoints':
                        stats['bench_points'] = int(value) if value else None
                except (ValueError, TypeError):
                    pass

            break

    return stats


def save_team_stats(conn, game_id, stats):
    """Save team stats to database"""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO team_game_stats
        (game_id, team_id, points, field_goals_made, field_goals_attempted, field_goal_pct,
         three_pointers_made, three_pointers_attempted, three_point_pct,
         free_throws_made, free_throws_attempted, free_throw_pct,
         offensive_rebounds, defensive_rebounds, total_rebounds,
         assists, steals, blocks, turnovers, personal_fouls,
         points_in_paint, fast_break_points, bench_points)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        game_id,
        stats['team_id'],
        stats['points'],
        stats['field_goals_made'],
        stats['field_goals_attempted'],
        stats['field_goal_pct'],
        stats['three_pointers_made'],
        stats['three_pointers_attempted'],
        stats['three_point_pct'],
        stats['free_throws_made'],
        stats['free_throws_attempted'],
        stats['free_throw_pct'],
        stats['offensive_rebounds'],
        stats['defensive_rebounds'],
        stats['total_rebounds'],
        stats['assists'],
        stats['steals'],
        stats['blocks'],
        stats['turnovers'],
        stats['personal_fouls'],
        stats['points_in_paint'],
        stats['fast_break_points'],
        stats['bench_points']
    ))
    conn.commit()


def backfill_stats(season_year=None, batch_size=100):
    """Backfill team stats for all completed games"""
    conn = sqlite3.connect('nba_games.db')
    cursor = conn.cursor()

    # Get games that need stats
    if season_year:
        cursor.execute('''
            SELECT g.game_id, g.home_team_id, g.away_team_id, g.date
            FROM games g
            LEFT JOIN team_game_stats tgs ON g.game_id = tgs.game_id
            WHERE g.completed = 1
            AND g.season = ?
            AND tgs.id IS NULL
            ORDER BY g.date
        ''', (season_year,))
    else:
        cursor.execute('''
            SELECT g.game_id, g.home_team_id, g.away_team_id, g.date
            FROM games g
            LEFT JOIN team_game_stats tgs ON g.game_id = tgs.game_id
            WHERE g.completed = 1
            AND tgs.id IS NULL
            ORDER BY g.date
        ''')

    games = cursor.fetchall()
    total_games = len(games)

    if total_games == 0:
        print("No games need stats backfill!")
        conn.close()
        return 0

    print(f"\n{'='*80}")
    print(f"NBA TEAM STATS BACKFILL")
    print(f"{'='*80}")
    print(f"Games to process: {total_games}")
    print(f"Estimated time: {total_games * 0.3 / 60:.1f} minutes\n")

    success_count = 0
    error_count = 0

    for i, (game_id, home_team_id, away_team_id, game_date) in enumerate(games, 1):
        print(f"[{i}/{total_games}] Game {game_id} ({game_date[:10]})...", end=" ")

        # Fetch game data
        data = fetch_game_stats(game_id)

        if not data:
            print("No data")
            error_count += 1
            time.sleep(0.2)
            continue

        # Get boxscore data
        boxscore = data.get('boxscore', {})

        # Parse and save stats for both teams
        try:
            home_stats = parse_team_stats(boxscore, home_team_id)
            away_stats = parse_team_stats(boxscore, away_team_id)

            # If we didn't get stats from boxscore, try to get points from the game data
            if home_stats['points'] is None:
                for team in boxscore.get('teams', []):
                    team_info = team.get('team', {})
                    team_id_str = team_info.get('id')

                    if team_id_str == str(home_team_id):
                        # Try to get score from competition data
                        for competitor in data.get('header', {}).get('competitions', [{}])[0].get('competitors', []):
                            if competitor.get('id') == str(home_team_id):
                                home_stats['points'] = int(competitor.get('score', 0))
                    elif team_id_str == str(away_team_id):
                        for competitor in data.get('header', {}).get('competitions', [{}])[0].get('competitors', []):
                            if competitor.get('id') == str(away_team_id):
                                away_stats['points'] = int(competitor.get('score', 0))

            save_team_stats(conn, game_id, home_stats)
            save_team_stats(conn, game_id, away_stats)

            print(f"OK (H:{home_stats['points']} A:{away_stats['points']})")
            success_count += 1

        except Exception as e:
            print(f"Error: {e}")
            error_count += 1

        # Rate limiting
        time.sleep(0.2)

        # Progress update every 100 games
        if i % 100 == 0:
            print(f"\n--- Progress: {i}/{total_games} ({100*i/total_games:.1f}%) ---\n")

    conn.close()

    print(f"\n{'='*80}")
    print(f"STATS BACKFILL COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")

    return success_count


def main():
    if len(sys.argv) > 1:
        season = int(sys.argv[1])
        backfill_stats(season)
    else:
        # Backfill all seasons
        backfill_stats()


if __name__ == '__main__':
    main()
