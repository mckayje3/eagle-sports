"""
Scrape NBA player game stats from ESPN API.

This script fetches box score data for completed NBA games and populates
the players and player_game_stats tables.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import time
import urllib.request
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "nba_games.db"
ESPN_BOXSCORE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"


def get_games_without_player_stats(conn: sqlite3.Connection, season: int) -> list[tuple]:
    """Get completed games that don't have player stats yet."""
    query = """
        SELECT g.game_id, g.date
        FROM games g
        WHERE g.completed = 1
          AND g.season = ?
          AND g.game_id NOT IN (
              SELECT DISTINCT game_id FROM player_game_stats
          )
        ORDER BY g.date
    """
    return conn.execute(query, (season,)).fetchall()


def fetch_boxscore(game_id: int) -> dict | None:
    """Fetch boxscore data from ESPN API."""
    url = ESPN_BOXSCORE_URL.format(game_id=game_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read())
    except Exception as e:
        logger.error(f"Failed to fetch boxscore for game {game_id}: {e}")
        return None


def parse_stat(stat_str: str) -> tuple[int, int] | int | None:
    """Parse a stat string, handling 'made-attempted' format."""
    if stat_str in ('--', '-', ''):
        return None
    if '-' in stat_str and not stat_str.startswith('-'):
        # Format: "7-20" for made-attempted
        parts = stat_str.split('-')
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                return None
    # Single value (could be negative for plus_minus)
    try:
        return int(stat_str)
    except ValueError:
        return None


def upsert_player(conn: sqlite3.Connection, player_data: dict) -> int:
    """Insert or update player, return player_id."""
    player_id = int(player_data.get('id'))
    name = player_data.get('displayName', '')
    short_name = player_data.get('shortName', '')
    position = player_data.get('position', {}).get('abbreviation', '')
    jersey = player_data.get('jersey', '')

    conn.execute("""
        INSERT INTO players (player_id, name, short_name, position, jersey)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            name = excluded.name,
            short_name = excluded.short_name,
            position = excluded.position,
            jersey = excluded.jersey,
            updated_at = CURRENT_TIMESTAMP
    """, (player_id, name, short_name, position, jersey))

    return player_id


def process_boxscore(conn: sqlite3.Connection, game_id: int, data: dict) -> int:
    """Process boxscore data and insert player stats. Returns count of players processed."""
    boxscore = data.get('boxscore', {})
    players_data = boxscore.get('players', [])

    if not players_data:
        logger.warning(f"No player data in boxscore for game {game_id}")
        return 0

    count = 0
    for team_data in players_data:
        team_id = int(team_data.get('team', {}).get('id', 0))
        statistics = team_data.get('statistics', [])

        for stat_group in statistics:
            athletes = stat_group.get('athletes', [])

            for athlete_data in athletes:
                athlete = athlete_data.get('athlete', {})
                if not athlete.get('id'):
                    continue

                # Upsert player
                player_id = upsert_player(conn, athlete)

                # Parse stats
                stats = athlete_data.get('stats', [])
                starter = 1 if athlete_data.get('starter') else 0
                did_not_play = 1 if athlete_data.get('didNotPlay') else 0
                dnp_reason = athlete_data.get('reason') if did_not_play else None

                # Stats array order: MIN, PTS, FG, 3PT, FT, REB, AST, TO, STL, BLK, OREB, DREB, PF, +/-
                if len(stats) >= 14 and not did_not_play:
                    minutes = parse_stat(stats[0])
                    points = parse_stat(stats[1])
                    fg = parse_stat(stats[2])  # tuple (made, attempted)
                    three = parse_stat(stats[3])  # tuple
                    ft = parse_stat(stats[4])  # tuple
                    rebounds = parse_stat(stats[5])
                    assists = parse_stat(stats[6])
                    turnovers = parse_stat(stats[7])
                    steals = parse_stat(stats[8])
                    blocks = parse_stat(stats[9])
                    off_reb = parse_stat(stats[10])
                    def_reb = parse_stat(stats[11])
                    fouls = parse_stat(stats[12])
                    plus_minus = parse_stat(stats[13])

                    # Unpack shooting stats
                    fg_made, fg_att = fg if isinstance(fg, tuple) else (None, None)
                    three_made, three_att = three if isinstance(three, tuple) else (None, None)
                    ft_made, ft_att = ft if isinstance(ft, tuple) else (None, None)
                else:
                    # DNP or incomplete stats
                    minutes = points = rebounds = off_reb = def_reb = None
                    assists = steals = blocks = turnovers = fouls = plus_minus = None
                    fg_made = fg_att = three_made = three_att = ft_made = ft_att = None

                # Insert player game stats
                try:
                    conn.execute("""
                        INSERT INTO player_game_stats (
                            game_id, player_id, team_id, starter, did_not_play, dnp_reason,
                            minutes, points, rebounds, offensive_rebounds, defensive_rebounds,
                            assists, steals, blocks, turnovers, personal_fouls,
                            field_goals_made, field_goals_attempted,
                            three_pointers_made, three_pointers_attempted,
                            free_throws_made, free_throws_attempted,
                            plus_minus
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game_id, player_id, team_id, starter, did_not_play, dnp_reason,
                        minutes, points, rebounds, off_reb, def_reb,
                        assists, steals, blocks, turnovers, fouls,
                        fg_made, fg_att, three_made, three_att, ft_made, ft_att,
                        plus_minus
                    ))
                    count += 1
                except sqlite3.IntegrityError:
                    # Already exists (shouldn't happen with our query, but just in case)
                    pass

    return count


def scrape_season(season: int, delay: float = 0.5, limit: int | None = None):
    """Scrape player stats for all completed games in a season."""
    conn = sqlite3.connect(DB_PATH)

    games = get_games_without_player_stats(conn, season)
    total_games = len(games)

    if limit:
        games = games[:limit]

    logger.info(f"Found {total_games} games without player stats for season {season}")
    logger.info(f"Processing {len(games)} games...")

    total_players = 0
    for i, (game_id, game_date) in enumerate(games, 1):
        logger.info(f"[{i}/{len(games)}] Fetching game {game_id} ({game_date[:10]})")

        data = fetch_boxscore(game_id)
        if data:
            count = process_boxscore(conn, game_id, data)
            total_players += count
            logger.info(f"  -> {count} player stats inserted")
            conn.commit()

        if i < len(games):
            time.sleep(delay)

    # Summary
    player_count = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    stats_count = conn.execute("SELECT COUNT(*) FROM player_game_stats").fetchone()[0]

    logger.info(f"\n=== Summary ===")
    logger.info(f"Total players in database: {player_count}")
    logger.info(f"Total player game stats: {stats_count}")
    logger.info(f"New stats added this run: {total_players}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Scrape NBA player game stats from ESPN")
    parser.add_argument("--season", type=int, default=2026, help="Season ending year (e.g., 2026 for 2025-26)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls in seconds")
    parser.add_argument("--limit", type=int, help="Limit number of games to process")
    args = parser.parse_args()

    scrape_season(args.season, args.delay, args.limit)


if __name__ == "__main__":
    main()
