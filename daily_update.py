"""
Daily Sports Update Script
Automatically determines which sports are in season and updates only those.
Fetches results, odds, updates predictions, and pushes to cloud.

Usage:
    python daily_update.py              # Auto-detect in-season sports
    python daily_update.py --all        # Force update all sports
    python daily_update.py --nba --nfl  # Force specific sports
    python daily_update.py --no-push    # Skip git push
    python daily_update.py --dry-run    # Show what would run without running
"""

import sys
import subprocess
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
        logging.FileHandler(log_dir / f'daily_update_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_nfl_in_season() -> tuple[bool, str]:
    """
    NFL regular season: Week 1 (early Sept) through Week 18 (early Jan)
    Playoffs: January
    Super Bowl: Early February
    """
    today = datetime.now()
    month, day = today.month, today.day

    # Sept 1 - Feb 15 is NFL season (includes playoffs + Super Bowl)
    if month >= 9 or month == 1 or (month == 2 and day <= 15):
        return True, "Regular season/Playoffs"
    return False, f"Offseason (resumes Sept)"


def is_cfb_in_season() -> tuple[bool, str]:
    """
    CFB regular season: Late August through early December
    Bowl season: Mid-December through early January
    National Championship: Early January
    """
    today = datetime.now()
    month, day = today.month, today.day

    # Aug 15 - Jan 15 is CFB season (includes bowls + championship)
    if month >= 8 or (month == 1 and day <= 15):
        return True, "Regular season/Bowls"
    return False, f"Offseason (resumes Aug)"


def is_nba_in_season() -> tuple[bool, str]:
    """
    NBA regular season: Late October through mid-April
    Playoffs: April through June
    """
    today = datetime.now()
    month = today.month

    # Oct 15 - June 30 is NBA season (includes playoffs)
    if month >= 10 or month <= 6:
        return True, "Regular season/Playoffs"
    return False, f"Offseason (resumes Oct)"


def is_cbb_in_season() -> tuple[bool, str]:
    """
    CBB regular season: Early November through early March
    Conference tournaments: Early March
    March Madness: Mid-March through early April
    """
    today = datetime.now()
    month, day = today.month, today.day

    # Nov 1 - April 10 is CBB season (includes March Madness)
    if month >= 11 or month <= 3 or (month == 4 and day <= 10):
        return True, "Regular season/March Madness"
    return False, f"Offseason (resumes Nov)"


def get_active_sports(force_all=False, force_sports=None) -> dict:
    """Determine which sports should be updated today"""

    sports = {
        'nba': {'check': is_nba_in_season, 'db': 'nba_games.db'},
        'cbb': {'check': is_cbb_in_season, 'db': 'cbb_games.db'},
        'nfl': {'check': is_nfl_in_season, 'db': 'nfl_games.db'},
        'cfb': {'check': is_cfb_in_season, 'db': 'cfb_games.db'},
    }

    active = {}

    for sport, info in sports.items():
        if force_all or (force_sports and sport in force_sports):
            active[sport] = {'active': True, 'reason': 'Forced by user'}
        else:
            in_season, reason = info['check']()
            active[sport] = {'active': in_season, 'reason': reason}

    return active


def update_game_results(sport: str, dry_run=False) -> bool:
    """Update game results/scores for a sport"""
    if dry_run:
        return True

    try:
        from update_game_results import update_results
        success, count = update_results(sport, days=7)
        logger.info(f"  Results: {count} games updated")
        return success
    except Exception as e:
        logger.error(f"  Error updating {sport.upper()} results: {e}")
        return False


def update_basketball(sport: str, dry_run=False) -> bool:
    """Update NBA or CBB"""
    logger.info(f"Updating {sport.upper()}...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would fetch results, odds, and update predictions")
        return True

    try:
        # Update results first (get final scores for completed games)
        update_game_results(sport, dry_run)

        # Fetch odds
        from espn_unified_odds import ESPNOddsScraper
        scraper = ESPNOddsScraper(sport)
        found, saved = scraper.scrape_recent(days=7)
        logger.info(f"  Odds: {saved}/{found} games updated")

        # Update predictions
        if sport == 'nba':
            from update_predictions_nba import update_predictions
        else:
            from update_predictions_cbb import update_predictions

        success, count = update_predictions(days=7)
        logger.info(f"  Predictions: {count} games updated")

        return success
    except Exception as e:
        logger.error(f"  Error updating {sport.upper()}: {e}")
        return False


def update_football(sport: str, dry_run=False) -> bool:
    """Update NFL or CFB"""
    logger.info(f"Updating {sport.upper()}...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would fetch results and odds")
        return True

    try:
        # Update results first (get final scores for completed games)
        update_game_results(sport, dry_run)

        # Fetch odds
        from espn_unified_odds import ESPNOddsScraper
        scraper = ESPNOddsScraper(sport)
        found, saved = scraper.scrape_recent(days=7)
        logger.info(f"  Odds: {saved}/{found} games updated")

        # For NFL, also update predictions if it's Tuesday (after MNF)
        if sport == 'nfl' and datetime.now().weekday() == 1:  # Tuesday
            logger.info("  Running Tuesday NFL prediction update...")
            try:
                from update_predictions_nfl import update_predictions
                success, count = update_predictions()
                logger.info(f"  Predictions: {count} games updated")
            except Exception as e:
                logger.warning(f"  NFL prediction update failed: {e}")

        # For CFB, update predictions on Sundays (after Saturday games)
        if sport == 'cfb' and datetime.now().weekday() == 6:  # Sunday
            logger.info("  Running Sunday CFB prediction update...")
            try:
                from update_predictions_cfb import update_predictions
                success, count = update_predictions(days=7)
                logger.info(f"  Predictions: {count} games updated")
            except Exception as e:
                logger.warning(f"  CFB prediction update failed: {e}")

        return True
    except Exception as e:
        logger.error(f"  Error updating {sport.upper()}: {e}")
        return False


def push_to_cloud(dry_run=False) -> bool:
    """Commit and push databases to GitHub"""
    logger.info("Pushing to cloud...")

    if dry_run:
        logger.info("  [DRY RUN] Would push to GitHub")
        return True

    try:
        result = subprocess.run(
            ['python', 'push_databases.py'],
            capture_output=True, text=True,
            cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            logger.info("  Successfully pushed to GitHub")
            return True
        else:
            logger.error(f"  Push failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"  Push error: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("DAILY SPORTS UPDATE")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Day: {datetime.now().strftime('%A')}")
    logger.info("=" * 60)

    # Parse arguments
    args = sys.argv[1:]
    force_all = '--all' in args
    no_push = '--no-push' in args
    dry_run = '--dry-run' in args

    force_sports = set()
    for sport in ['nba', 'cbb', 'nfl', 'cfb']:
        if f'--{sport}' in args:
            force_sports.add(sport)

    if dry_run:
        logger.info("*** DRY RUN MODE ***\n")

    # Determine active sports
    active_sports = get_active_sports(force_all, force_sports if force_sports else None)

    logger.info("Sport Status:")
    for sport, info in active_sports.items():
        status = "ACTIVE" if info['active'] else "SKIPPED"
        logger.info(f"  {sport.upper()}: {status} - {info['reason']}")
    logger.info("")

    # Update each active sport
    results = {}

    for sport, info in active_sports.items():
        if not info['active']:
            continue

        logger.info("-" * 40)

        if sport in ['nba', 'cbb']:
            results[sport] = update_basketball(sport, dry_run)
        else:
            results[sport] = update_football(sport, dry_run)

    # Push to cloud
    if not no_push and any(results.values()):
        logger.info("-" * 40)
        push_to_cloud(dry_run)
    elif no_push:
        logger.info("\nSkipping git push (--no-push flag)")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for sport, success in results.items():
        status = "OK" if success else "FAILED"
        logger.info(f"  {sport.upper()}: {status}")

    skipped = [s.upper() for s, info in active_sports.items() if not info['active']]
    if skipped:
        logger.info(f"  Skipped (offseason): {', '.join(skipped)}")

    logger.info("")
    logger.info("Update complete!")

    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
