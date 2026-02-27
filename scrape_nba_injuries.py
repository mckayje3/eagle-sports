"""
Scrape NBA injury reports from ESPN and populate player_game_availability.

This script fetches current injury data and populates availability status
for upcoming games. Run daily before predictions.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import urllib.request
import json
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "nba_games.db"
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def fetch_injuries() -> list[dict]:
    """Fetch current injury data from ESPN."""
    try:
        with urllib.request.urlopen(ESPN_INJURIES_URL, timeout=30) as response:
            data = json.loads(response.read())
        return data.get('injuries', [])
    except Exception as e:
        logger.error(f"Failed to fetch injuries: {e}")
        return []


def parse_injury(athlete_data: dict, team_id: int) -> dict | None:
    """Parse an injury entry into our schema format."""
    athlete = athlete_data.get('athlete', {})

    # Get player ID - try multiple fields
    player_id = athlete_data.get('id') or athlete.get('id')
    if not player_id:
        return None

    try:
        player_id = int(player_id)
    except (ValueError, TypeError):
        return None

    # Parse status
    status_raw = athlete_data.get('status', '')
    type_info = athlete_data.get('type', {})
    if isinstance(type_info, dict):
        status_name = type_info.get('description', status_raw)
    else:
        status_name = status_raw

    # Normalize status
    status_upper = status_name.upper() if status_name else ''
    if 'OUT' in status_upper:
        status = 'Out'
    elif 'DAY' in status_upper:
        status = 'Day-To-Day'
    elif 'DOUBT' in status_upper:
        status = 'Doubtful'
    elif 'QUESTION' in status_upper:
        status = 'Questionable'
    elif 'PROBABLE' in status_upper:
        status = 'Probable'
    else:
        status = 'Out'  # Default to out if unclear

    # Parse injury details
    details = athlete_data.get('details', {})
    injury_type = details.get('type')  # e.g., 'Knee', 'Ankle'
    injury_location = details.get('location')  # e.g., 'Leg'
    injury_detail_part = details.get('detail')  # e.g., 'Surgery', 'Strain'
    injury_side = details.get('side')  # e.g., 'Left', 'Right'
    return_date = details.get('returnDate')

    # Build injury detail string
    parts = []
    if injury_side and injury_side != 'Not Specified':
        parts.append(injury_side)
    if injury_type:
        parts.append(injury_type)
    if injury_detail_part and injury_detail_part != 'Not Specified':
        parts.append(injury_detail_part)
    injury_detail = ' '.join(parts) if parts else None

    # Categorize injury type
    if injury_location:
        loc_upper = injury_location.upper()
        if loc_upper in ('LEG', 'FOOT'):
            injury_category = 'Lower Body'
        elif loc_upper in ('ARM', 'HAND'):
            injury_category = 'Upper Body'
        elif loc_upper in ('TORSO', 'BACK'):
            injury_category = 'Torso'
        elif loc_upper == 'HEAD':
            injury_category = 'Illness/Head'
        else:
            injury_category = injury_location.title()
    elif injury_type:
        injury_category = injury_type
    else:
        injury_category = None

    # Report date
    report_date = athlete_data.get('date', '')[:10] if athlete_data.get('date') else None

    return {
        'player_id': player_id,
        'player_name': athlete.get('displayName', 'Unknown'),
        'team_id': team_id,
        'status': status,
        'injury_type': injury_category,
        'injury_detail': injury_detail,
        'report_date': report_date,
        'return_date': return_date,
        'short_comment': athlete_data.get('shortComment'),
    }


def get_team_id_map(conn: sqlite3.Connection) -> dict[str, int]:
    """Get mapping of team names/abbreviations to team_id."""
    team_map = {}
    rows = conn.execute('SELECT team_id, name, abbreviation FROM teams').fetchall()
    for team_id, name, abbr in rows:
        team_map[name.lower()] = team_id
        team_map[abbr.lower()] = team_id
    return team_map


def get_upcoming_games(conn: sqlite3.Connection, days_ahead: int = 3) -> list[tuple]:
    """Get upcoming games within the next N days."""
    today = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

    query = """
        SELECT game_id, game_date_eastern, home_team_id, away_team_id
        FROM games
        WHERE completed = 0
          AND game_date_eastern >= ?
          AND game_date_eastern <= ?
        ORDER BY game_date_eastern
    """
    return conn.execute(query, (today, end_date)).fetchall()


def get_player_team_games(conn: sqlite3.Connection, player_id: int, team_id: int,
                          game_ids: set[int]) -> list[tuple[int, int]]:
    """Get (game_id, team_id) tuples where this player's team is playing."""
    if not game_ids:
        return []
    query = """
        SELECT game_id,
               CASE WHEN home_team_id = ? THEN home_team_id ELSE away_team_id END as player_team
        FROM games
        WHERE game_id IN ({})
          AND (home_team_id = ? OR away_team_id = ?)
    """.format(','.join('?' * len(game_ids)))

    params = [team_id] + list(game_ids) + [team_id, team_id]
    return [(r[0], r[1]) for r in conn.execute(query, params).fetchall()]


def ensure_player_exists(conn: sqlite3.Connection, player_id: int, player_name: str):
    """Make sure player exists in players table."""
    conn.execute("""
        INSERT OR IGNORE INTO players (player_id, name)
        VALUES (?, ?)
    """, (player_id, player_name))


def update_availability(conn: sqlite3.Connection, injuries: list[dict],
                        upcoming_game_ids: set[int], today: str) -> int:
    """Update player_game_availability for upcoming games."""
    updated = 0

    for injury in injuries:
        player_id = injury['player_id']
        team_id = injury['team_id']

        if not team_id:
            continue

        # Ensure player exists
        ensure_player_exists(conn, player_id, injury['player_name'])

        # Find this player's upcoming games
        player_games = get_player_team_games(conn, player_id, team_id, upcoming_game_ids)

        for game_id, player_team_id in player_games:
            try:
                conn.execute("""
                    INSERT INTO player_game_availability
                    (game_id, player_id, team_id, status, injury_type, injury_detail, report_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(game_id, player_id) DO UPDATE SET
                        team_id = excluded.team_id,
                        status = excluded.status,
                        injury_type = excluded.injury_type,
                        injury_detail = excluded.injury_detail,
                        report_date = excluded.report_date
                """, (
                    game_id, player_id, team_id, injury['status'],
                    injury['injury_type'], injury['injury_detail'],
                    injury['report_date'] or today
                ))
                updated += 1
            except sqlite3.Error as e:
                logger.warning(f"Failed to update availability for player {player_id}, game {game_id}: {e}")

    return updated


def scrape_injuries(days_ahead: int = 3):
    """Main function to scrape injuries and update availability."""
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime('%Y-%m-%d')

    # Get team ID mapping
    team_map = get_team_id_map(conn)

    # Fetch injuries from ESPN
    logger.info("Fetching injury data from ESPN...")
    injury_data = fetch_injuries()

    if not injury_data:
        logger.warning("No injury data retrieved")
        conn.close()
        return

    # Parse all injuries
    all_injuries = []
    for team_entry in injury_data:
        team_name = team_entry.get('displayName', '')
        team_id = team_map.get(team_name.lower())

        # Try to find team ID from nested data if not found
        if not team_id:
            # Some entries have team info differently structured
            team_id_raw = team_entry.get('id')
            if team_id_raw:
                try:
                    team_id = int(team_id_raw)
                except ValueError:
                    pass

        for athlete_data in team_entry.get('injuries', []):
            injury = parse_injury(athlete_data, team_id)
            if injury:
                all_injuries.append(injury)

    logger.info(f"Parsed {len(all_injuries)} injury records")

    # Get upcoming games
    upcoming_games = get_upcoming_games(conn, days_ahead)
    upcoming_game_ids = {g[0] for g in upcoming_games}
    logger.info(f"Found {len(upcoming_games)} upcoming games in next {days_ahead} days")

    if not upcoming_games:
        logger.info("No upcoming games to update")
        conn.close()
        return

    # Update availability
    updated = update_availability(conn, all_injuries, upcoming_game_ids, today)
    conn.commit()

    logger.info(f"Updated {updated} availability records")

    # Summary
    print("\n=== Injury Report Summary ===")
    rows = conn.execute("""
        SELECT pga.status, COUNT(*) as cnt
        FROM player_game_availability pga
        JOIN games g ON pga.game_id = g.game_id
        WHERE g.completed = 0 AND g.game_date_eastern >= ?
        GROUP BY pga.status
        ORDER BY cnt DESC
    """, (today,)).fetchall()

    for status, count in rows:
        print(f"  {status}: {count} player-games")

    # Show sample of injured players for upcoming games
    print("\n=== Key Injuries (Upcoming Games) ===")
    rows = conn.execute("""
        SELECT p.name, t.abbreviation, pga.status, pga.injury_detail, g.game_date_eastern
        FROM player_game_availability pga
        JOIN players p ON pga.player_id = p.player_id
        JOIN games g ON pga.game_id = g.game_id
        LEFT JOIN teams t ON pga.team_id = t.team_id
        WHERE g.completed = 0
          AND g.game_date_eastern >= ?
          AND pga.status = 'Out'
        GROUP BY pga.player_id, g.game_date_eastern
        ORDER BY g.game_date_eastern, p.name
        LIMIT 20
    """, (today,)).fetchall()

    for name, team, status, detail, game_date in rows:
        detail_str = f" ({detail})" if detail else ""
        print(f"  {game_date}: {name} [{team}] - {status}{detail_str}")

    conn.close()


def reconcile_completed_games(conn: sqlite3.Connection) -> int:
    """
    After games complete, update played/minutes_played from box scores.
    This captures players who were on the injury report but still played.
    """
    result = conn.execute("""
        UPDATE player_game_availability
        SET played = (
                SELECT CASE WHEN pgs.did_not_play = 1 THEN 0 ELSE 1 END
                FROM player_game_stats pgs
                WHERE pgs.game_id = player_game_availability.game_id
                  AND pgs.player_id = player_game_availability.player_id
            ),
            minutes_played = (
                SELECT pgs.minutes
                FROM player_game_stats pgs
                WHERE pgs.game_id = player_game_availability.game_id
                  AND pgs.player_id = player_game_availability.player_id
            )
        WHERE played IS NULL
          AND game_id IN (SELECT game_id FROM games WHERE completed = 1)
          AND EXISTS (
              SELECT 1 FROM player_game_stats pgs
              WHERE pgs.game_id = player_game_availability.game_id
                AND pgs.player_id = player_game_availability.player_id
          )
    """)
    return result.rowcount


def main():
    parser = argparse.ArgumentParser(description="Scrape NBA injuries and update availability")
    parser.add_argument("--days", type=int, default=3, help="Days ahead to update (default: 3)")
    parser.add_argument("--reconcile", action="store_true", help="Reconcile completed games with box scores")
    args = parser.parse_args()

    if args.reconcile:
        conn = sqlite3.connect(DB_PATH)
        updated = reconcile_completed_games(conn)
        conn.commit()

        # Show players who played injured
        print(f"Reconciled {updated} availability records")
        print("\n=== Players Who Played Injured ===")
        rows = conn.execute("""
            SELECT p.name, t.abbreviation, pga.status, pga.injury_detail,
                   pga.minutes_played, g.game_date_eastern
            FROM player_game_availability pga
            JOIN players p ON pga.player_id = p.player_id
            JOIN games g ON pga.game_id = g.game_id
            LEFT JOIN teams t ON pga.team_id = t.team_id
            WHERE pga.played = 1
              AND pga.status IN ('Day-To-Day', 'Questionable', 'Doubtful', 'Probable')
            ORDER BY g.game_date_eastern DESC
            LIMIT 15
        """).fetchall()

        if rows:
            for name, team, status, detail, mins, date in rows:
                detail_str = f" ({detail})" if detail else ""
                print(f"  {date}: {name} [{team}] was {status}{detail_str} - played {mins} min")
        else:
            print("  No 'played injured' records yet (will accumulate over time)")

        conn.close()
    else:
        scrape_injuries(args.days)


if __name__ == "__main__":
    main()
