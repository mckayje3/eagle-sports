"""
NFL Tuesday Update Script
Runs every Tuesday after Monday Night Football to:
1. Scrape final game stats from previous week
2. Set closing lines for completed games
3. Fetch opening lines for current week
4. Generate predictions for current week

Usage:
    py nfl_tuesday_update.py              # Run full Tuesday update
    py nfl_tuesday_update.py --test       # Dry run (no database changes)
    py nfl_tuesday_update.py --week 14    # Force specific week
"""

import sys
import json
import sqlite3
import logging
from datetime import datetime, timedelta
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nfl_tuesday_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_current_nfl_week():
    """Calculate current NFL week based on date"""
    # 2025 NFL season starts Sept 4, 2025 (Week 1)
    # 2024 NFL season started Sept 5, 2024
    today = datetime.now()

    if today.year == 2025:
        season_start = datetime(2025, 9, 4)
    else:
        season_start = datetime(2024, 9, 5)

    if today < season_start:
        return 1, today.year

    days_since_start = (today - season_start).days
    week = (days_since_start // 7) + 1
    week = min(week, 18)  # Cap at week 18

    return week, season_start.year


def scrape_previous_week_games(season, week, dry_run=False):
    """Scrape final stats for games from the previous week"""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 1: Scraping Week {week} final game stats")
    logger.info('='*60)

    if dry_run:
        logger.info("[DRY RUN] Would scrape game stats")
        return True

    try:
        from nfl_espn_scraper import NFLESPNScraper
        scraper = NFLESPNScraper('nfl_games.db')

        # Scrape the week
        games = scraper.scrape_week(season, week, season_type=2)
        logger.info(f"Scraped {len(games) if games else 0} games")

        return True
    except Exception as e:
        logger.error(f"Error scraping games: {e}")
        return False


def set_closing_lines_for_completed_games(dry_run=False):
    """Set closing lines for all completed games that don't have them"""
    logger.info(f"\n{'='*60}")
    logger.info("STEP 2: Setting closing lines for completed games")
    logger.info('='*60)

    if dry_run:
        logger.info("[DRY RUN] Would set closing lines")
        return True

    try:
        from cfb_nfl_database import FootballDatabase
        db = FootballDatabase('nfl_games.db')
        db.connect()

        # Find completed games without closing lines
        cursor = db.conn.cursor()
        cursor.execute('''
            SELECT g.game_id, op.latest_spread, op.latest_total
            FROM games g
            JOIN odds_and_predictions op ON g.game_id = op.game_id
            WHERE g.completed = 1
              AND op.latest_spread IS NOT NULL
        ''')
        games_to_update = cursor.fetchall()

        updated = 0
        for game_id, latest_line, latest_total in games_to_update:
            db.set_closing_line(game_id)
            updated += 1

        logger.info(f"Set closing lines for {updated} games")
        db.close()
        return True

    except Exception as e:
        logger.error(f"Error setting closing lines: {e}")
        return False


def fetch_opening_lines_for_current_week(season, week, dry_run=False):
    """Fetch opening lines for the current week's games"""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 3: Fetching Week {week} opening lines")
    logger.info('='*60)

    if dry_run:
        logger.info("[DRY RUN] Would fetch opening lines")
        return True

    try:
        # Load API key
        with open('odds_api_config.json', 'r') as f:
            config = json.load(f)
            api_key = config['api_key']

        from fetch_latest_odds import OddsAPIFetcher
        fetcher = OddsAPIFetcher(api_key)

        # Fetch and set as opening lines
        fetcher.update_odds('nfl', 'nfl_games.db', is_opening=True)

        return True

    except Exception as e:
        logger.error(f"Error fetching opening lines: {e}")
        return False


def generate_predictions(season, week, dry_run=False):
    """Generate predictions for current week using Deep Eagle"""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 4: Generating Week {week} predictions")
    logger.info('='*60)

    if dry_run:
        logger.info("[DRY RUN] Would generate predictions")
        return True

    try:
        from predict_deep_eagle import predict_upcoming_games, display_predictions

        model_path = f'models/deep_eagle_nfl_{season}.pt'
        scaler_path = f'models/deep_eagle_nfl_{season}_scaler.pkl'

        predictions_df = predict_upcoming_games(
            sport='nfl',
            season=season,
            db_path='nfl_games.db',
            model_path=model_path,
            scaler_path=scaler_path,
            min_week=week
        )

        if predictions_df is not None:
            display_predictions(predictions_df)

            # Save predictions
            output_path = f'nfl_week{week}_predictions.csv'
            predictions_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to: {output_path}")

            # Also save to a standard location for the dashboard
            predictions_df.to_csv('nfl_current_predictions.csv', index=False)
            logger.info("Updated nfl_current_predictions.csv")

        return True

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_predictions_to_database(season, week, dry_run=False):
    """Save predictions to the database for dashboard access"""
    logger.info(f"\n{'='*60}")
    logger.info("STEP 5: Saving predictions to database")
    logger.info('='*60)

    if dry_run:
        logger.info("[DRY RUN] Would save predictions to database")
        return True

    try:
        import pandas as pd

        # Read predictions
        try:
            predictions_df = pd.read_csv('nfl_current_predictions.csv')
        except FileNotFoundError:
            logger.warning("No predictions file found")
            return True

        conn = sqlite3.connect('nfl_games.db')
        cursor = conn.cursor()

        # Create predictions table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                game_id INTEGER PRIMARY KEY,
                pred_home_score REAL,
                pred_away_score REAL,
                pred_spread REAL,
                pred_total REAL,
                pred_home_win INTEGER,
                pred_home_win_prob REAL,
                vegas_spread REAL,
                vegas_total REAL,
                spread_edge REAL,
                total_edge REAL,
                prediction_date TEXT,
                model_version TEXT
            )
        ''')

        # Insert/update predictions
        for _, row in predictions_df.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO predictions
                (game_id, pred_home_score, pred_away_score, pred_spread, pred_total,
                 pred_home_win, pred_home_win_prob, vegas_spread, vegas_total,
                 spread_edge, total_edge, prediction_date, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(row['game_id']),
                row.get('pred_home_score'),
                row.get('pred_away_score'),
                row.get('pred_spread'),
                row.get('pred_total'),
                int(row.get('pred_home_win', 0)),
                row.get('pred_home_win_prob'),
                row.get('vegas_spread'),
                row.get('vegas_total'),
                row.get('spread_edge'),
                row.get('total_edge'),
                datetime.now().isoformat(),
                f'deep_eagle_nfl_{season}'
            ))

        conn.commit()
        conn.close()

        logger.info(f"Saved {len(predictions_df)} predictions to database")
        return True

    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        return False


def run_tuesday_update(force_week=None, dry_run=False):
    """Run the complete Tuesday update process"""

    logger.info("\n" + "="*80)
    logger.info("NFL TUESDAY UPDATE")
    logger.info("="*80)
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if dry_run:
        logger.info("*** DRY RUN MODE - No database changes will be made ***")

    # Determine weeks
    current_week, season = get_current_nfl_week()

    if force_week:
        current_week = force_week

    previous_week = current_week - 1 if current_week > 1 else 1

    logger.info(f"Season: {season}")
    logger.info(f"Previous week (to finalize): {previous_week}")
    logger.info(f"Current week (to predict): {current_week}")

    success = True

    # Step 1: Scrape previous week's final stats
    if previous_week >= 1:
        if not scrape_previous_week_games(season, previous_week, dry_run):
            logger.warning("Game scraping had issues, continuing anyway...")

    # Step 2: Set closing lines for completed games
    if not set_closing_lines_for_completed_games(dry_run):
        logger.warning("Setting closing lines had issues, continuing anyway...")

    # Step 3: Fetch opening lines for current week
    if not fetch_opening_lines_for_current_week(season, current_week, dry_run):
        logger.warning("Fetching opening lines had issues, continuing anyway...")

    # Step 4: Generate predictions
    if not generate_predictions(season, current_week, dry_run):
        logger.error("Prediction generation failed!")
        success = False

    # Step 5: Save predictions to database
    if not save_predictions_to_database(season, current_week, dry_run):
        logger.warning("Saving predictions to database had issues")

    # Summary
    logger.info("\n" + "="*80)
    if success:
        logger.info("TUESDAY UPDATE COMPLETED SUCCESSFULLY")
    else:
        logger.info("TUESDAY UPDATE COMPLETED WITH ERRORS")
    logger.info("="*80)

    return success


def main():
    force_week = None
    dry_run = False

    for arg in sys.argv[1:]:
        if arg == '--test':
            dry_run = True
        elif arg.startswith('--week'):
            if '=' in arg:
                force_week = int(arg.split('=')[1])
            elif len(sys.argv) > sys.argv.index(arg) + 1:
                force_week = int(sys.argv[sys.argv.index(arg) + 1])

    success = run_tuesday_update(force_week=force_week, dry_run=dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
