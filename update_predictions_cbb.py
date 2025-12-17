"""
Update CBB Predictions Script
Fetches latest odds from ESPN and regenerates predictions.
Called by the dashboard "Update Predictions" button.

Usage:
    py update_predictions_cbb.py              # Update with latest odds
    py update_predictions_cbb.py --days 7     # Predictions for next N days
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
    logger.info("Fetching latest CBB odds from ESPN...")

    try:
        from espn_unified_odds import ESPNOddsScraper
        scraper = ESPNOddsScraper()
        scraper.fetch_odds('cbb', days=7)
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
    logger.info(f"Generating CBB predictions for next {days} days...")

    try:
        from cbb_predictor import CBBPredictor

        predictor = CBBPredictor()
        predictions_df = predictor.predict_upcoming(days=days)

        if predictions_df is not None and len(predictions_df) > 0:
            # Save predictions
            predictions_df.to_csv('cbb_predictions.csv', index=False)

            # Save to database
            save_to_database(predictions_df)

            logger.info(f"Generated {len(predictions_df)} predictions")
            return predictions_df

        return None

    except ImportError:
        # Fallback: use predict_deep_eagle if cbb_predictor doesn't exist
        logger.info("Using Deep Eagle predictor fallback...")
        try:
            from predict_deep_eagle import predict_upcoming_games

            predictions_df = predict_upcoming_games(
                sport='cbb',
                season=2025,
                db_path='cbb_games.db',
                model_path='models/deep_eagle_cbb_2025.pt',
                scaler_path='models/deep_eagle_cbb_2025_scaler.pkl'
            )

            if predictions_df is not None and len(predictions_df) > 0:
                predictions_df.to_csv('cbb_predictions.csv', index=False)
                save_to_database(predictions_df)
                logger.info(f"Generated {len(predictions_df)} predictions")
                return predictions_df

            return None

        except Exception as e:
            logger.error(f"Error with Deep Eagle fallback: {e}")
            return None

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_database(predictions_df):
    """Save predictions to cbb_games.db odds_and_predictions table"""
    conn = sqlite3.connect('cbb_games.db')
    cursor = conn.cursor()

    for _, row in predictions_df.iterrows():
        game_id = int(row['game_id'])

        # Update existing odds_and_predictions row or insert new one
        cursor.execute('''
            INSERT INTO odds_and_predictions (game_id, predicted_home_score, predicted_away_score,
                predicted_home_MOE, predicted_away_MOE, predicted_spread_MOE, predicted_total_MOE,
                home_win_probability, confidence, prediction_created)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                predicted_home_score = excluded.predicted_home_score,
                predicted_away_score = excluded.predicted_away_score,
                predicted_home_MOE = excluded.predicted_home_MOE,
                predicted_away_MOE = excluded.predicted_away_MOE,
                predicted_spread_MOE = excluded.predicted_spread_MOE,
                predicted_total_MOE = excluded.predicted_total_MOE,
                home_win_probability = excluded.home_win_probability,
                confidence = excluded.confidence,
                prediction_created = excluded.prediction_created
        ''', (
            game_id,
            row.get('pred_home_score'),
            row.get('pred_away_score'),
            row.get('pred_home_MOE'),
            row.get('pred_away_MOE'),
            row.get('pred_spread_MOE'),
            row.get('pred_total_MOE'),
            row.get('pred_home_win_prob', row.get('confidence', 0.5)),
            row.get('confidence', 0.5),
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()


def sync_to_cache():
    """Sync predictions to users.db prediction_cache for dashboard"""
    try:
        cbb_conn = sqlite3.connect('cbb_games.db')
        users_conn = sqlite3.connect('users.db')

        # Get predictions with game info from odds_and_predictions table
        query = '''
            SELECT
                op.game_id, g.date, g.season,
                ht.name as home_team, at.name as away_team,
                op.predicted_home_score as pred_home_score,
                op.predicted_away_score as pred_away_score,
                (op.predicted_home_score - op.predicted_away_score) as pred_spread,
                (op.predicted_home_score + op.predicted_away_score) as pred_total,
                op.confidence,
                op.prediction_created as prediction_date
            FROM odds_and_predictions op
            JOIN games g ON op.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.completed = 0 AND op.predicted_home_score IS NOT NULL
        '''

        predictions = pd.read_sql_query(query, cbb_conn)

        if len(predictions) > 0:
            cursor = users_conn.cursor()
            now = datetime.now().isoformat()

            for _, row in predictions.iterrows():
                winner = row['home_team'] if row['pred_spread'] > 0 else row['away_team']
                confidence = row['confidence'] if row['confidence'] else 0.5
                # Use prediction_date from cbb_games.db if available, otherwise use now
                created_at = row.get('prediction_date') or now

                cursor.execute('''
                    INSERT OR REPLACE INTO prediction_cache
                    (game_id, sport, season, week, game_date, home_team, away_team,
                     predicted_home_score, predicted_away_score,
                     predicted_spread, predicted_total, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(row['game_id']), 'CBB', int(row['season']), 0,
                    row['date'], row['home_team'], row['away_team'],
                    row['pred_home_score'], row['pred_away_score'],
                    row['pred_spread'], row['pred_total'], confidence,
                    created_at
                ))

            users_conn.commit()
            logger.info(f"Synced {len(predictions)} predictions to cache with timestamp")

        cbb_conn.close()
        users_conn.close()
        return True, f"Synced {len(predictions)} predictions"

    except Exception as e:
        logger.error(f"Error syncing to cache: {e}")
        return False, str(e)


def update_predictions(days=7):
    """Main function to update predictions with latest odds"""
    logger.info("\n" + "="*60)
    logger.info("UPDATING CBB PREDICTIONS")
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
