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
from timezone_utils import utc_to_eastern_date

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
        scraper = ESPNOddsScraper('cbb')
        scraper.scrape_recent(days=7)
        logger.info("Odds fetched successfully")
        return True
    except ImportError:
        logger.warning("espn_unified_odds not available, continuing with cached odds")
        return True
    except Exception as e:
        logger.warning(f"Odds fetch failed: {e}, continuing with cached odds")
        return True




def check_missing_vegas_lines(days=7):
    """
    Check for upcoming games missing Vegas lines and warn before predictions.

    Returns:
        Tuple of (games_missing_lines, total_upcoming_games)
    """
    conn = sqlite3.connect('cbb_games.db')
    cursor = conn.cursor()

    # Find upcoming games without Vegas lines (use game_date_eastern for accurate dates)
    cursor.execute('''
        SELECT g.game_id, g.game_date_eastern,
               ht.name as home_team, at.name as away_team,
               o.latest_spread, o.latest_total
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 0
          AND g.game_date_eastern >= date('now')
          AND g.game_date_eastern <= date('now', ?)
        ORDER BY g.game_date_eastern
    ''', (f'+{days} days',))

    games = cursor.fetchall()
    conn.close()

    missing_lines = []
    for game_id, game_date, home, away, spread, total in games:
        if spread is None or total is None:
            missing_lines.append({
                'game_id': game_id,
                'date': game_date,
                'matchup': f'{away} @ {home}',
                'spread': spread,
                'total': total
            })

    if missing_lines:
        logger.warning("=" * 60)
        logger.warning(f"WARNING: {len(missing_lines)} games missing Vegas lines!")
        logger.warning("=" * 60)
        for game in missing_lines[:10]:  # Show first 10 only (CBB has many games)
            missing = []
            if game['spread'] is None:
                missing.append('spread')
            if game['total'] is None:
                missing.append('total')
            logger.warning(f"  {game['date']} - {game['matchup']} - missing: {', '.join(missing)}")
        if len(missing_lines) > 10:
            logger.warning(f"  ... and {len(missing_lines) - 10} more games")
        logger.warning("")
        logger.warning("Predictions for these games will use fallback values.")
        logger.warning("Consider fetching odds before generating predictions.")
        logger.warning("=" * 60)
    else:
        logger.info(f"All {len(games)} upcoming games have Vegas lines")

    return len(missing_lines), len(games)

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

        # Get predictions with game info and odds from odds_and_predictions table
        query = '''
            SELECT
                op.game_id, g.date, g.season,
                ht.name as home_team, at.name as away_team,
                op.predicted_home_score as pred_home_score,
                op.predicted_away_score as pred_away_score,
                (op.predicted_away_score - op.predicted_home_score) as pred_spread,
                (op.predicted_home_score + op.predicted_away_score) as pred_total,
                op.confidence,
                op.prediction_created as prediction_date,
                COALESCE(op.latest_spread, op.opening_spread) as vegas_spread,
                COALESCE(op.latest_total, op.opening_total) as vegas_total
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
                # Vegas convention: spread = away - home
                # Negative spread = HOME favored, Positive spread = AWAY favored
                # (winner variable removed - not used in cache insert)
                confidence = row['confidence'] if row['confidence'] else 0.5
                # Use prediction_date from cbb_games.db if available, otherwise use now
                created_at = row.get('prediction_date') or now

                # Convert UTC date to Eastern date for correct day display
                # ESPN stores dates in UTC (e.g., 2025-12-18T01:00Z for 8PM ET Dec 17)
                game_date_eastern = utc_to_eastern_date(row['date']) or row['date']

                cursor.execute('''
                    INSERT OR REPLACE INTO prediction_cache
                    (game_id, sport, season, week, game_date, home_team, away_team,
                     predicted_home_score, predicted_away_score,
                     predicted_spread, predicted_total, confidence, created_at,
                     vegas_spread, vegas_total)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(row['game_id']), 'CBB', int(row['season']), 0,
                    game_date_eastern, row['home_team'], row['away_team'],
                    row['pred_home_score'], row['pred_away_score'],
                    row['pred_spread'], row['pred_total'], confidence,
                    created_at,
                    row.get('vegas_spread'), row.get('vegas_total')
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

    # Step 2: Check for missing Vegas lines (warn but continue)
    missing, total = check_missing_vegas_lines(days)
    if missing > 0:
        logger.warning(f"Proceeding with {missing}/{total} games missing lines...")

    # Step 3: Generate predictions
    predictions_df = generate_predictions(days)

    if predictions_df is not None:
        # Step 4: Sync to dashboard cache
        sync_to_cache()

        # Step 5: Generate betting recommendations
        try:
            from betting_tracker import BettingTracker
            tracker = BettingTracker()
            recs = tracker.generate_recommendations('CBB')
            saved = tracker.save_recommendations(recs)
            logger.info(f"Generated {len(recs)} betting recommendations, saved {saved}")
        except Exception as e:
            logger.warning(f"Could not generate betting recommendations: {e}")

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
