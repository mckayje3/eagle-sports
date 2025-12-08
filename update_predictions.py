"""
Update Predictions Script
Fetches latest odds and regenerates predictions without full scrape.
Called by the dashboard "Update Predictions" button.

Usage:
    py update_predictions.py              # Update with latest odds
    py update_predictions.py --week 14    # Force specific week
"""

import sys
import json
import sqlite3
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_current_nfl_week():
    """Calculate current NFL week based on date"""
    today = datetime.now()

    if today.year == 2025:
        season_start = datetime(2025, 9, 4)
    else:
        season_start = datetime(2024, 9, 5)

    if today < season_start:
        return 1, today.year

    days_since_start = (today - season_start).days
    week = (days_since_start // 7) + 1
    week = min(week, 18)

    return week, season_start.year


def fetch_latest_odds():
    """Fetch latest odds from The Odds API"""
    logger.info("Fetching latest odds...")

    try:
        with open('odds_api_config.json', 'r') as f:
            config = json.load(f)
            api_key = config['api_key']

        from fetch_latest_odds import OddsAPIFetcher
        fetcher = OddsAPIFetcher(api_key)

        # Update latest lines (NOT opening)
        fetcher.update_odds('nfl', 'nfl_games.db', is_opening=False)

        return True

    except Exception as e:
        logger.error(f"Error fetching odds: {e}")
        return False


def generate_predictions(season, week):
    """Generate predictions for current week"""
    logger.info(f"Generating Week {week} predictions...")

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
            # Save predictions
            predictions_df.to_csv('nfl_current_predictions.csv', index=False)
            predictions_df.to_csv(f'nfl_week{week}_predictions.csv', index=False)

            # Save to database
            save_to_database(predictions_df, season)

            logger.info(f"Generated {len(predictions_df)} predictions")
            return predictions_df

        return None

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_database(predictions_df, season):
    """Save predictions to database"""
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

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


def update_predictions(force_week=None):
    """Main function to update predictions with latest odds"""
    logger.info("\n" + "="*60)
    logger.info("UPDATING NFL PREDICTIONS")
    logger.info("="*60)

    current_week, season = get_current_nfl_week()
    if force_week:
        current_week = force_week

    logger.info(f"Season: {season}, Week: {current_week}")

    # Step 1: Fetch latest odds
    fetch_latest_odds()

    # Step 2: Generate predictions
    predictions_df = generate_predictions(season, current_week)

    if predictions_df is not None:
        logger.info("\n" + "="*60)
        logger.info("UPDATE COMPLETE")
        logger.info("="*60)
        return True, predictions_df
    else:
        logger.error("Failed to generate predictions")
        return False, None


def main():
    force_week = None

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--week' and i < len(sys.argv) - 1:
            force_week = int(sys.argv[i + 1])
        elif arg.startswith('--week='):
            force_week = int(arg.split('=')[1])

    success, _ = update_predictions(force_week)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
