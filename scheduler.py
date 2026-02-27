"""
Automated scheduler for updating college football game data
Runs scheduled updates to keep the database current
"""
import schedule
import time
from datetime import datetime, timedelta
import logging
from cfb_espn_scraper import ESPNScraper
from odds_scraper import VegasInsiderOddsScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cfb_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CFBScheduler:
    def __init__(self):
        self.scraper = ESPNScraper()
        self.odds_scraper = VegasInsiderOddsScraper()
        self.current_season = datetime.now().year
        # Football season typically runs August-January
        # Adjust year if we're in Jan-July
        if datetime.now().month < 8:
            self.current_season -= 1

    def update_current_week(self):
        """Update games for the current week"""
        logger.info("Starting current week update...")
        try:
            # Determine current week (approximate)
            # Season typically starts in late August/early September
            current_date = datetime.now()
            season_start = datetime(self.current_season, 8, 25)

            if current_date < season_start:
                logger.info("Season hasn't started yet")
                return

            # Calculate approximate week number
            days_into_season = (current_date - season_start).days
            current_week = min((days_into_season // 7) + 1, 15)

            logger.info(f"Updating {self.current_season} Season, Week {current_week}")
            self.scraper.scrape_week(self.current_season, current_week, season_type=2)

            logger.info("Current week update completed successfully")
        except Exception as e:
            logger.error(f"Error updating current week: {e}")

    def update_recent_games(self):
        """Update games from the last 7 days"""
        logger.info("Starting recent games update...")
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')

            logger.info(f"Updating games from {start_str} to {end_str}")
            self.scraper.scrape_date_range(start_str, end_str)

            logger.info("Recent games update completed successfully")
        except Exception as e:
            logger.error(f"Error updating recent games: {e}")

    def update_yesterday(self):
        """Update games from yesterday"""
        logger.info("Starting yesterday's games update...")
        try:
            yesterday = datetime.now() - timedelta(days=1)
            date_str = yesterday.strftime('%Y%m%d')

            logger.info(f"Updating games for {date_str}")
            self.scraper.scrape_date_range(date_str, date_str)

            logger.info("Yesterday's games update completed successfully")
        except Exception as e:
            logger.error(f"Error updating yesterday's games: {e}")

    def full_season_update(self):
        """Update entire current season"""
        logger.info("Starting full season update...")
        try:
            logger.info(f"Updating entire {self.current_season} season")
            self.scraper.scrape_season(self.current_season, start_week=1, end_week=15)

            logger.info("Full season update completed successfully")
        except Exception as e:
            logger.error(f"Error updating full season: {e}")

    def backfill_historical(self, start_year: int, end_year: int):
        """Backfill historical seasons"""
        logger.info(f"Starting historical backfill from {start_year} to {end_year}...")
        try:
            for year in range(start_year, end_year + 1):
                logger.info(f"Backfilling {year} season")
                self.scraper.scrape_season(year, start_week=1, end_week=15)
                time.sleep(5)  # Longer delay between seasons

            logger.info("Historical backfill completed successfully")
        except Exception as e:
            logger.error(f"Error during historical backfill: {e}")

    def update_current_odds(self):
        """Update current betting odds for upcoming games (VegasInsider)"""
        logger.info("Starting current odds update from VegasInsider...")
        try:
            import subprocess
            # Step 1: Scrape VegasInsider
            result = subprocess.run(['py', 'parse_vegasinsider.py'],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"VegasInsider scraping failed: {result.stderr}")
                return

            # Step 2: Match and save to database
            result = subprocess.run(['py', 'save_odds_with_matching.py', 'vegasinsider'],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Saving odds failed: {result.stderr}")
                return

            logger.info("Current odds update completed successfully (VegasInsider)")
        except Exception as e:
            logger.error(f"Error updating current odds: {e}")

    def update_odds_with_games(self):
        """Update both games and odds together"""
        logger.info("Starting combined games and odds update...")
        try:
            # First update games
            self.update_current_week()
            # Then update odds
            time.sleep(2)  # Brief delay between updates
            self.update_current_odds()
            logger.info("Combined update completed successfully")
        except Exception as e:
            logger.error(f"Error during combined update: {e}")


def run_scheduled_updates():
    """
    Configure and run scheduled updates

    Default schedule:
    - Every day at 9 AM: Update yesterday's games (catch any late finishes)
    - Every Saturday at 11 PM: Update current week (during season)
    - Every Sunday at 9 AM: Update recent week (catch all weekend games)
    """
    scheduler = CFBScheduler()

    # Daily morning update - catch completed games from previous day
    schedule.every().day.at("09:00").do(scheduler.update_yesterday)

    # Saturday night update - get Saturday games
    schedule.every().saturday.at("23:00").do(scheduler.update_current_week)

    # Sunday morning update - comprehensive weekly update
    schedule.every().sunday.at("09:00").do(scheduler.update_recent_games)

    # Weekly comprehensive update
    schedule.every().monday.at("10:00").do(scheduler.update_recent_games)

    # Odds updates - multiple times per week during season
    schedule.every().tuesday.at("14:00").do(scheduler.update_current_odds)
    schedule.every().thursday.at("14:00").do(scheduler.update_current_odds)
    schedule.every().friday.at("14:00").do(scheduler.update_current_odds)
    schedule.every().saturday.at("10:00").do(scheduler.update_current_odds)

    logger.info("Scheduler started. Running scheduled updates...")
    logger.info("Schedule:")
    logger.info("  - Daily at 9:00 AM: Update yesterday's games")
    logger.info("  - Saturday at 11:00 PM: Update current week")
    logger.info("  - Sunday at 9:00 AM: Update recent games")
    logger.info("  - Monday at 10:00 AM: Update recent games")
    logger.info("  - Tuesday/Thursday/Friday at 2:00 PM: Update current odds")
    logger.info("  - Saturday at 10:00 AM: Update current odds")

    # Run continuously
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def run_manual_update(update_type: str = 'current_week', **kwargs):
    """
    Run a manual update

    Args:
        update_type: Type of update to run
            - 'current_week': Update current week
            - 'recent': Update last 7 days
            - 'yesterday': Update yesterday
            - 'full_season': Update entire current season
            - 'historical': Backfill historical data (requires start_year and end_year)
            - 'week': Update specific week (requires week and optionally season)
            - 'date_range': Update specific date range (requires start_date and end_date)
            - 'odds': Update current betting odds
            - 'odds_and_games': Update both games and odds together
    """
    scheduler = CFBScheduler()

    if update_type == 'current_week':
        scheduler.update_current_week()
    elif update_type == 'recent':
        scheduler.update_recent_games()
    elif update_type == 'yesterday':
        scheduler.update_yesterday()
    elif update_type == 'full_season':
        scheduler.full_season_update()
    elif update_type == 'historical':
        start_year = kwargs.get('start_year', 2020)
        end_year = kwargs.get('end_year', 2024)
        scheduler.backfill_historical(start_year, end_year)
    elif update_type == 'week':
        week = kwargs.get('week', 1)
        season = kwargs.get('season', scheduler.current_season)
        season_type = kwargs.get('season_type', 2)
        scheduler.scraper.scrape_week(season, week, season_type)
    elif update_type == 'date_range':
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        if start_date and end_date:
            scheduler.scraper.scrape_date_range(start_date, end_date)
        else:
            logger.error("start_date and end_date required for date_range update")
    elif update_type == 'odds':
        scheduler.update_current_odds()
    elif update_type == 'odds_and_games':
        scheduler.update_odds_with_games()
    else:
        logger.error(f"Unknown update type: {update_type}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # Manual mode with command-line arguments
        command = sys.argv[1]

        if command == 'schedule':
            # Run scheduled updates
            run_scheduled_updates()
        elif command == 'current':
            # Update current week
            run_manual_update('current_week')
        elif command == 'recent':
            # Update recent games
            run_manual_update('recent')
        elif command == 'yesterday':
            # Update yesterday
            run_manual_update('yesterday')
        elif command == 'full':
            # Update full current season
            run_manual_update('full_season')
        elif command == 'historical':
            # Backfill historical data
            start_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2020
            end_year = int(sys.argv[3]) if len(sys.argv) > 3 else 2024
            run_manual_update('historical', start_year=start_year, end_year=end_year)
        elif command == 'week':
            # Update specific week
            week = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            season = int(sys.argv[3]) if len(sys.argv) > 3 else datetime.now().year
            run_manual_update('week', week=week, season=season)
        elif command == 'odds':
            # Update current betting odds
            run_manual_update('odds')
        elif command == 'odds_and_games':
            # Update both games and odds
            run_manual_update('odds_and_games')
        else:
            print("Unknown command. Available commands:")
            print("  schedule     - Run scheduled automatic updates")
            print("  current      - Update current week")
            print("  recent       - Update last 7 days")
            print("  yesterday    - Update yesterday's games")
            print("  full         - Update full current season")
            print("  historical <start_year> <end_year> - Backfill historical data")
            print("  week <week_num> <season> - Update specific week")
            print("  odds         - Update current betting odds")
            print("  odds_and_games - Update both games and odds together")
    else:
        print("CFB Data Scheduler")
        print("==================")
        print()
        print("Usage: python scheduler.py <command> [args]")
        print()
        print("Commands:")
        print("  schedule     - Run scheduled automatic updates")
        print("  current      - Update current week")
        print("  recent       - Update last 7 days")
        print("  yesterday    - Update yesterday's games")
        print("  full         - Update full current season")
        print("  historical <start_year> <end_year> - Backfill historical data")
        print("  week <week_num> <season> - Update specific week")
        print("  odds         - Update current betting odds")
        print("  odds_and_games - Update both games and odds together")
        print()
        print("Examples:")
        print("  python scheduler.py schedule")
        print("  python scheduler.py current")
        print("  python scheduler.py historical 2020 2024")
        print("  python scheduler.py week 12 2024")
        print("  python scheduler.py odds")
        print("  python scheduler.py odds_and_games")
