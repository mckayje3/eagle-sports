"""
Scrape NBA game details: quarterly scores, referees, and coaches.

This script backfills additional game data from ESPN API:
- Quarterly scores (Q1-Q4, OT) for team_game_stats
- Referee assignments for each game
- Coach information for each team
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
ESPN_GAME_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
ESPN_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"


def fetch_json(url: str) -> dict | None:
    """Fetch JSON from URL."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read())
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def scrape_coaches(conn: sqlite3.Connection):
    """Scrape head coaches for all teams."""
    logger.info("Scraping coaches...")

    teams = conn.execute("SELECT team_id FROM teams").fetchall()

    for (team_id,) in teams:
        url = ESPN_ROSTER_URL.format(team_id=team_id)
        data = fetch_json(url)
        if not data:
            continue

        coaches = data.get('coach', [])
        for coach in coaches:
            coach_id = int(coach.get('id', 0))
            if not coach_id:
                continue

            first_name = coach.get('firstName', '')
            last_name = coach.get('lastName', '')
            full_name = f"{first_name} {last_name}".strip()
            experience = coach.get('experience', 0)

            # Upsert coach
            conn.execute("""
                INSERT INTO coaches (coach_id, first_name, last_name, full_name, experience)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(coach_id) DO UPDATE SET
                    first_name = excluded.first_name,
                    last_name = excluded.last_name,
                    full_name = excluded.full_name,
                    experience = excluded.experience,
                    updated_at = CURRENT_TIMESTAMP
            """, (coach_id, first_name, last_name, full_name, experience))

            # Link to team for current season
            conn.execute("""
                INSERT INTO team_coaches (team_id, coach_id, season, is_head_coach)
                VALUES (?, ?, 2025, 1)
                ON CONFLICT(team_id, coach_id, season) DO NOTHING
            """, (team_id, coach_id))

        time.sleep(0.2)

    conn.commit()
    coach_count = conn.execute("SELECT COUNT(*) FROM coaches").fetchone()[0]
    logger.info(f"Scraped {coach_count} coaches")


def get_or_create_referee(conn: sqlite3.Connection, name: str) -> int:
    """Get referee ID, creating if needed."""
    row = conn.execute("SELECT referee_id FROM referees WHERE full_name = ?", (name,)).fetchone()
    if row:
        return row[0]

    cursor = conn.execute("INSERT INTO referees (full_name) VALUES (?)", (name,))
    return cursor.lastrowid


def process_game_details(conn: sqlite3.Connection, game_id: int, data: dict) -> dict:
    """Process game details and return counts of what was updated."""
    results = {'quarters': False, 'refs': 0}

    # Get header for quarterly scores
    header = data.get('header', {})
    competitions = header.get('competitions', [{}])
    if not competitions:
        return results

    competition = competitions[0]
    competitors = competition.get('competitors', [])

    # Process quarterly scores
    for competitor in competitors:
        team_info = competitor.get('team', {})
        team_id = int(team_info.get('id', 0))
        if not team_id:
            continue

        linescores = competitor.get('linescores', [])
        if not linescores:
            continue

        # Parse quarterly scores
        q_scores = [None, None, None, None, None]  # Q1, Q2, Q3, Q4, OT
        for i, ls in enumerate(linescores):
            score = ls.get('displayValue') or ls.get('value')
            if score is not None:
                try:
                    score = int(float(score))
                except (ValueError, TypeError):
                    continue

                if i < 4:
                    q_scores[i] = score
                else:
                    # OT periods - accumulate
                    if q_scores[4] is None:
                        q_scores[4] = 0
                    q_scores[4] += score

        # Update team_game_stats
        if any(s is not None for s in q_scores[:4]):
            conn.execute("""
                UPDATE team_game_stats
                SET q1_points = ?, q2_points = ?, q3_points = ?, q4_points = ?, ot_points = ?
                WHERE game_id = ? AND team_id = ?
            """, (*q_scores, game_id, team_id))
            results['quarters'] = True

    # Process referees
    game_info = data.get('gameInfo', {})
    officials = game_info.get('officials', [])

    for official in officials:
        ref_name = official.get('fullName') or official.get('displayName')
        if not ref_name:
            continue

        position = official.get('order', 0)
        referee_id = get_or_create_referee(conn, ref_name)

        try:
            conn.execute("""
                INSERT INTO game_referees (game_id, referee_id, position)
                VALUES (?, ?, ?)
                ON CONFLICT(game_id, referee_id) DO NOTHING
            """, (game_id, referee_id, position))
            results['refs'] += 1
        except sqlite3.IntegrityError:
            pass

    return results


def get_games_to_process(conn: sqlite3.Connection, season: int, force: bool = False) -> list[int]:
    """Get completed games that need processing."""
    if force:
        # All completed games for season
        query = """
            SELECT game_id FROM games
            WHERE completed = 1 AND season = ?
            ORDER BY date
        """
        return [r[0] for r in conn.execute(query, (season,)).fetchall()]

    # Games without quarterly scores or refs
    query = """
        SELECT DISTINCT g.game_id
        FROM games g
        JOIN team_game_stats tgs ON g.game_id = tgs.game_id
        LEFT JOIN game_referees gr ON g.game_id = gr.game_id
        WHERE g.completed = 1
          AND g.season = ?
          AND (tgs.q1_points IS NULL OR gr.game_id IS NULL)
        ORDER BY g.date
    """
    return [r[0] for r in conn.execute(query, (season,)).fetchall()]


def scrape_game_details(season: int, delay: float = 0.3, force: bool = False, limit: int | None = None):
    """Main function to scrape game details."""
    conn = sqlite3.connect(DB_PATH)

    # First scrape coaches
    existing_coaches = conn.execute("SELECT COUNT(*) FROM coaches").fetchone()[0]
    if existing_coaches == 0:
        scrape_coaches(conn)
    else:
        logger.info(f"Coaches already populated ({existing_coaches} coaches)")

    # Get games to process
    games = get_games_to_process(conn, season, force)

    if limit:
        games = games[:limit]

    logger.info(f"Processing {len(games)} games for season {season}")

    if not games:
        logger.info("No games to process")
        conn.close()
        return

    quarters_updated = 0
    refs_added = 0

    for i, game_id in enumerate(games, 1):
        logger.info(f"[{i}/{len(games)}] Processing game {game_id}")

        url = ESPN_GAME_URL.format(game_id=game_id)
        data = fetch_json(url)

        if data:
            results = process_game_details(conn, game_id, data)
            if results['quarters']:
                quarters_updated += 1
            refs_added += results['refs']

            # Commit every 50 games
            if i % 50 == 0:
                conn.commit()

        if i < len(games):
            time.sleep(delay)

    conn.commit()

    # Summary
    print("\n=== Scrape Summary ===")
    print(f"Games processed: {len(games)}")
    print(f"Quarterly scores updated: {quarters_updated}")
    print(f"Referee assignments added: {refs_added}")

    # Show sample data
    print("\n--- Sample Quarterly Scores ---")
    rows = conn.execute("""
        SELECT t.abbreviation, tgs.q1_points, tgs.q2_points, tgs.q3_points, tgs.q4_points,
               tgs.ot_points, tgs.points, g.date
        FROM team_game_stats tgs
        JOIN teams t ON tgs.team_id = t.team_id
        JOIN games g ON tgs.game_id = g.game_id
        WHERE tgs.q1_points IS NOT NULL
        ORDER BY g.date DESC
        LIMIT 6
    """).fetchall()
    for r in rows:
        ot = f" OT:{r[5]}" if r[5] else ""
        print(f"  {r[7][:10]} {r[0]}: Q1:{r[1]} Q2:{r[2]} Q3:{r[3]} Q4:{r[4]}{ot} = {r[6]}")

    print("\n--- Referees ---")
    ref_count = conn.execute("SELECT COUNT(*) FROM referees").fetchone()[0]
    assignment_count = conn.execute("SELECT COUNT(*) FROM game_referees").fetchone()[0]
    print(f"  Total referees: {ref_count}")
    print(f"  Total assignments: {assignment_count}")

    # Most assigned refs
    print("\n  Top refs by games:")
    rows = conn.execute("""
        SELECT r.full_name, COUNT(*) as games
        FROM game_referees gr
        JOIN referees r ON gr.referee_id = r.referee_id
        GROUP BY gr.referee_id
        ORDER BY games DESC
        LIMIT 5
    """).fetchall()
    for r in rows:
        print(f"    {r[0]}: {r[1]} games")

    print("\n--- Coaches ---")
    rows = conn.execute("""
        SELECT t.abbreviation, c.full_name, c.experience
        FROM team_coaches tc
        JOIN teams t ON tc.team_id = t.team_id
        JOIN coaches c ON tc.coach_id = c.coach_id
        WHERE tc.season = 2025
        ORDER BY t.abbreviation
    """).fetchall()
    for r in rows:
        print(f"  {r[0]}: {r[1]} ({r[2]} yrs)")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Scrape NBA game details (quarters, refs, coaches)")
    parser.add_argument("--season", type=int, default=2026, help="Season ending year (e.g., 2026 for 2025-26)")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between API calls")
    parser.add_argument("--force", action="store_true", help="Reprocess all games")
    parser.add_argument("--limit", type=int, help="Limit games to process")
    args = parser.parse_args()

    scrape_game_details(args.season, args.delay, args.force, args.limit)


if __name__ == "__main__":
    main()
