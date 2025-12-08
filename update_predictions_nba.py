"""
Update NBA Predictions Script
Fetches latest odds from ESPN and regenerates predictions.
Called by the dashboard "Update Predictions" button.

Usage:
    py update_predictions_nba.py              # Update with latest odds
    py update_predictions_nba.py --days 7     # Predictions for next N days
"""

import sys
import subprocess
import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_latest_odds():
    """Fetch latest odds from ESPN via direct import"""
    logger.info("Fetching latest NBA odds from ESPN...")

    try:
        from espn_unified_odds import ESPNOddsScraper
        scraper = ESPNOddsScraper()
        scraper.fetch_odds('nba', days=7)
        logger.info("Odds fetched successfully")
        return True
    except ImportError:
        logger.warning("espn_unified_odds not available, continuing with cached odds")
        return True
    except Exception as e:
        logger.warning(f"Odds fetch failed: {e}, continuing with cached odds")
        return True


def generate_predictions(days=7):
    """Generate predictions using Deep Eagle model"""
    logger.info(f"Generating NBA predictions for next {days} days...")

    try:
        from nba_predictor import NBAPredictor

        predictor = NBAPredictor()

        if predictor.model is None:
            logger.warning("NBA model not loaded - no predictions generated")
            return None

        predictions_df = predictor.predict_upcoming(days=days)

        if predictions_df is not None and len(predictions_df) > 0:
            # Save predictions
            predictions_df.to_csv('nba_current_predictions.csv', index=False)

            # Save to database
            save_to_database(predictions_df)

            logger.info(f"Generated {len(predictions_df)} predictions")
            return predictions_df

        logger.warning("No upcoming NBA games found")
        return None

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_database(predictions_df):
    """Save predictions to nba_games.db"""
    conn = sqlite3.connect('nba_games.db')
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
            row.get('pred_home_win_prob', row.get('confidence', 0.5)),
            row.get('vegas_spread'),
            row.get('vegas_total'),
            row.get('spread_edge'),
            row.get('total_edge'),
            datetime.now().isoformat(),
            'deep_eagle_nba_2025'
        ))

    conn.commit()
    conn.close()


def sync_to_cache():
    """Sync predictions to users.db prediction_cache for dashboard"""
    try:
        nba_conn = sqlite3.connect('nba_games.db')
        users_conn = sqlite3.connect('users.db')

        # Get predictions with game info
        query = '''
            SELECT
                p.game_id, g.date, g.season,
                ht.name as home_team, at.name as away_team,
                p.pred_home_score, p.pred_away_score,
                p.pred_spread, p.pred_total,
                p.pred_home_win_prob as confidence
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.completed = 0
        '''

        predictions = pd.read_sql_query(query, nba_conn)

        if len(predictions) > 0:
            cursor = users_conn.cursor()

            for _, row in predictions.iterrows():
                winner = row['home_team'] if row['pred_spread'] > 0 else row['away_team']
                confidence = row['confidence'] if row['confidence'] else 0.5

                cursor.execute('''
                    INSERT OR REPLACE INTO prediction_cache
                    (game_id, sport, season, week, game_date, home_team, away_team,
                     predicted_winner, predicted_spread, predicted_total, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(row['game_id']), 'NBA', int(row['season']), 0,
                    row['date'], row['home_team'], row['away_team'],
                    winner, row['pred_spread'], row['pred_total'], confidence
                ))

            users_conn.commit()
            logger.info(f"Synced {len(predictions)} predictions to cache")

        nba_conn.close()
        users_conn.close()
        return True, f"Synced {len(predictions)} predictions"

    except Exception as e:
        logger.error(f"Error syncing to cache: {e}")
        return False, str(e)


def update_predictions(days=7):
    """Main function to update predictions with latest odds"""
    logger.info("\n" + "="*60)
    logger.info("UPDATING NBA PREDICTIONS")
    logger.info("="*60)

    # Step 1: Fetch latest odds from ESPN
    fetch_latest_odds()

    # Step 2: Generate predictions
    predictions_df = generate_predictions(days)

    if predictions_df is not None:
        # Step 3: Sync to dashboard cache
        sync_to_cache()

        logger.info("\n" + "="*60)
        logger.info("UPDATE COMPLETE")
        logger.info("="*60)
        return True, predictions_df
    else:
        logger.error("Failed to generate predictions")
        return False, None


def main():
    days = 7

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--days' and i < len(sys.argv) - 1:
            days = int(sys.argv[i + 1])
        elif arg.startswith('--days='):
            days = int(arg.split('=')[1])

    success, _ = update_predictions(days)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
