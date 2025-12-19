"""
Daily Basketball Update Script
Fetches latest odds and regenerates predictions for NBA and CBB.
Can be scheduled via Windows Task Scheduler or run manually.

Usage:
    py daily_basketball_update.py              # Update both NBA and CBB
    py daily_basketball_update.py --nba        # NBA only
    py daily_basketball_update.py --cbb        # CBB only
    py daily_basketball_update.py --no-push    # Don't push to git
"""

import sys
import subprocess
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'basketball_update_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def fetch_odds(sport: str) -> tuple:
    """Fetch latest odds from ESPN"""
    try:
        from espn_unified_odds import ESPNOddsScraper
        scraper = ESPNOddsScraper(sport)
        return scraper.scrape_recent(days=7)
    except Exception as e:
        logger.error(f"Error fetching {sport.upper()} odds: {e}")
        return 0, 0


def update_predictions(sport: str) -> bool:
    """Run the appropriate update script"""
    try:
        if sport == 'nba':
            from update_predictions_nba import update_predictions
            success, _ = update_predictions(days=7)
        elif sport == 'cbb':
            from update_predictions_cbb import update_predictions
            success, _ = update_predictions(days=7)
        else:
            logger.error(f"Unknown sport: {sport}")
            return False
        return success
    except Exception as e:
        logger.error(f"Error updating {sport.upper()} predictions: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_stats(sport: str) -> dict:
    """Get summary stats for a sport"""
    db_map = {'nba': 'nba_games.db', 'cbb': 'cbb_games.db'}
    db_path = db_map.get(sport)

    if not db_path:
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check for game_date_eastern column
    cursor.execute("PRAGMA table_info(games)")
    columns = [col[1] for col in cursor.fetchall()]
    date_col = 'game_date_eastern' if 'game_date_eastern' in columns else 'date(date)'

    # Games today
    cursor.execute(f'''
        SELECT COUNT(*) FROM games
        WHERE {date_col} = date('now')
    ''')
    games_today = cursor.fetchone()[0]

    # Games with odds today
    cursor.execute(f'''
        SELECT COUNT(*) FROM games g
        JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE {date_col} = date('now') AND o.latest_spread IS NOT NULL
    ''')
    with_odds_today = cursor.fetchone()[0]

    # Games tomorrow
    cursor.execute(f'''
        SELECT COUNT(*) FROM games
        WHERE {date_col} = date('now', '+1 day')
    ''')
    games_tomorrow = cursor.fetchone()[0]

    # Games with odds tomorrow
    cursor.execute(f'''
        SELECT COUNT(*) FROM games g
        JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE {date_col} = date('now', '+1 day') AND o.latest_spread IS NOT NULL
    ''')
    with_odds_tomorrow = cursor.fetchone()[0]

    conn.close()

    return {
        'today': games_today,
        'today_with_odds': with_odds_today,
        'tomorrow': games_tomorrow,
        'tomorrow_with_odds': with_odds_tomorrow
    }


def git_commit_and_push() -> bool:
    """Commit changes and push to remote"""
    try:
        # Check if there are changes
        result = subprocess.run(
            ['git', 'status', '--porcelain', 'nba_games.db', 'cbb_games.db', 'users.db'],
            capture_output=True, text=True, cwd=Path(__file__).parent
        )

        if not result.stdout.strip():
            logger.info("No changes to commit")
            return True

        # Stage files
        subprocess.run(
            ['git', 'add', 'nba_games.db', 'cbb_games.db', 'users.db'],
            cwd=Path(__file__).parent, check=True
        )

        # Commit
        commit_msg = f"""Auto-update basketball odds and predictions

Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"""

        subprocess.run(
            ['git', 'commit', '-m', commit_msg],
            cwd=Path(__file__).parent, check=True
        )

        # Push
        subprocess.run(
            ['git', 'push'],
            cwd=Path(__file__).parent, check=True
        )

        logger.info("Successfully committed and pushed to remote")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git error: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("DAILY BASKETBALL UPDATE")
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Parse arguments
    args = sys.argv[1:]
    do_nba = '--nba' in args or (not '--cbb' in args and not '--nba' in args)
    do_cbb = '--cbb' in args or (not '--cbb' in args and not '--nba' in args)
    do_push = '--no-push' not in args

    results = {}

    # Update NBA
    if do_nba:
        logger.info("\n" + "=" * 40)
        logger.info("Updating NBA...")
        logger.info("=" * 40)

        with_odds, saved = fetch_odds('nba')
        logger.info(f"Fetched odds: {saved}/{with_odds} games saved")

        success = update_predictions('nba')
        stats = get_stats('nba')

        results['nba'] = {
            'success': success,
            'odds_fetched': saved,
            'stats': stats
        }

        logger.info(f"NBA Today: {stats.get('today_with_odds', 0)}/{stats.get('today', 0)} games with odds")
        logger.info(f"NBA Tomorrow: {stats.get('tomorrow_with_odds', 0)}/{stats.get('tomorrow', 0)} games with odds")

    # Update CBB
    if do_cbb:
        logger.info("\n" + "=" * 40)
        logger.info("Updating CBB...")
        logger.info("=" * 40)

        with_odds, saved = fetch_odds('cbb')
        logger.info(f"Fetched odds: {saved}/{with_odds} games saved")

        success = update_predictions('cbb')
        stats = get_stats('cbb')

        results['cbb'] = {
            'success': success,
            'odds_fetched': saved,
            'stats': stats
        }

        logger.info(f"CBB Today: {stats.get('today_with_odds', 0)}/{stats.get('today', 0)} games with odds")
        logger.info(f"CBB Tomorrow: {stats.get('tomorrow_with_odds', 0)}/{stats.get('tomorrow', 0)} games with odds")

    # Git push
    if do_push:
        logger.info("\n" + "=" * 40)
        logger.info("Pushing to git...")
        logger.info("=" * 40)
        git_commit_and_push()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("UPDATE COMPLETE")
    logger.info("=" * 60)

    all_success = all(r.get('success', False) for r in results.values())
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
