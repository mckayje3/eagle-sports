"""
Update CFB Predictions Script
Fetches latest odds from ESPN and regenerates predictions.
Called by the dashboard "Update Predictions" button.

Usage:
    py update_predictions_cfb.py              # Update with latest odds
    py update_predictions_cfb.py --week 14    # Force specific week
"""

import sys
import subprocess
import sqlite3
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_current_cfb_week():
    """
    Calculate current CFB week based on date.
    Season starts the last Saturday of August each year.

    Returns:
        tuple: (week_number, season_year)

    Week numbers:
    - 0-15: Regular season and conference championships
    - 16-17: Bowl games and CFP First Round
    - 18: Bowl games and CFP Quarterfinals
    - 19: CFP Semifinals
    - 20: CFP National Championship
    """
    today = datetime.now()
    year = today.year
    from datetime import timedelta

    def get_season_start(yr):
        """Find last Saturday of August for a given year"""
        aug_31 = datetime(yr, 8, 31)
        days_to_subtract = (aug_31.weekday() - 5) % 7  # Saturday = 5
        return aug_31 - timedelta(days=days_to_subtract)

    season_start = get_season_start(year)

    # Check if we're in previous year's bowl season
    if today < season_start:
        if today.month <= 1:
            season_start = get_season_start(year - 1)
            year = year - 1
        else:
            return 0, year

    days_since_start = (today - season_start).days
    week = (days_since_start // 7) + 1
    week = min(week, 20)  # Cap at week 20 (National Championship)

    return week, year


def fetch_latest_odds():
    """Fetch latest odds from ESPN via direct import"""
    logger.info("Fetching latest CFB odds from ESPN...")

    try:
        from espn_unified_odds import ESPNOddsScraper
        scraper = ESPNOddsScraper()
        scraper.fetch_odds('cfb', days=7)
        logger.info("Odds fetched successfully")
        return True
    except ImportError:
        logger.warning("espn_unified_odds not available, continuing with cached odds")
        return True
    except Exception as e:
        logger.warning(f"Odds fetch failed: {e}, continuing with cached odds")
        return True




def check_missing_vegas_lines(season, week):
    """
    Check for games in the specified week missing Vegas lines.

    Returns:
        Tuple of (games_missing_lines, total_games)
    """
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    # Find games in this week without Vegas lines
    cursor.execute('''
        SELECT g.game_id, g.date,
               ht.name as home_team, at.name as away_team,
               o.latest_spread, o.latest_total
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.season = ? AND g.week = ? AND g.completed = 0
        ORDER BY g.date
    ''', (season, week))

    games = cursor.fetchall()
    conn.close()

    missing_lines = []
    for game_id, game_date, home, away, spread, total in games:
        if spread is None or total is None:
            missing_lines.append({
                'game_id': game_id,
                'date': game_date[:10] if game_date else 'Unknown',
                'matchup': f'{away} @ {home}',
                'spread': spread,
                'total': total
            })

    if missing_lines:
        logger.warning("=" * 60)
        logger.warning(f"WARNING: {len(missing_lines)} games missing Vegas lines!")
        logger.warning("=" * 60)
        for game in missing_lines:
            missing = []
            if game['spread'] is None:
                missing.append('spread')
            if game['total'] is None:
                missing.append('total')
            logger.warning(f"  {game['date']} - {game['matchup']} - missing: {', '.join(missing)}")
        logger.warning("")
        logger.warning("Predictions for these games will use fallback values.")
        logger.warning("Consider fetching odds before generating predictions.")
        logger.warning("=" * 60)
    else:
        logger.info(f"All {len(games)} Week {week} games have Vegas lines")

    return len(missing_lines), len(games)

def generate_predictions(season, week):
    """Generate predictions using Deep Eagle model"""
    logger.info(f"Generating Week {week} CFB predictions...")

    try:
        from cfb_deep_eagle_predictor import CFBDeepEaglePredictor

        predictor = CFBDeepEaglePredictor(
            model_path=f'models/deep_eagle_cfb_{season}.pt',
            scaler_path=f'models/deep_eagle_cfb_{season}_scaler.pkl'
        )

        if predictor.model is None:
            logger.warning("CFB model not loaded - no predictions generated")
            return None

        predictions_df = predictor.predict_upcoming(week=week)

        if predictions_df is not None and len(predictions_df) > 0:
            # Save predictions
            predictions_df.to_csv('cfb_current_predictions.csv', index=False)
            predictions_df.to_csv(f'cfb_week{week}_predictions.csv', index=False)

            # Save to database
            save_to_database(predictions_df, season)

            logger.info(f"Generated {len(predictions_df)} predictions")
            return predictions_df

        logger.warning("No upcoming CFB games found")
        return None

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_database(predictions_df, season):
    """Save predictions to odds_and_predictions table in cfb_games.db"""
    conn = sqlite3.connect('cfb_games.db')
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    for _, row in predictions_df.iterrows():
        game_id = int(row['game_id'])

        # Check if row exists for this game
        cursor.execute('SELECT id FROM odds_and_predictions WHERE game_id = ?', (game_id,))
        existing = cursor.fetchone()

        if existing:
            # Update existing row with predictions
            cursor.execute('''
                UPDATE odds_and_predictions SET
                    predicted_home_score = ?,
                    predicted_away_score = ?,
                    predicted_home_MOE = ?,
                    predicted_away_MOE = ?,
                    predicted_spread_MOE = ?,
                    predicted_total_MOE = ?,
                    home_win_probability = ?,
                    confidence = ?,
                    prediction_created = ?
                WHERE game_id = ?
            ''', (
                row.get('pred_home_score'),
                row.get('pred_away_score'),
                row.get('pred_home_MOE'),
                row.get('pred_away_MOE'),
                row.get('pred_spread_MOE'),
                row.get('pred_total_MOE'),
                row.get('pred_home_win_prob'),
                row.get('confidence'),
                now,
                game_id
            ))
        else:
            # Insert new row with predictions (odds may be added later)
            cursor.execute('''
                INSERT INTO odds_and_predictions
                (game_id, source, predicted_home_score, predicted_away_score,
                 predicted_home_MOE, predicted_away_MOE,
                 predicted_spread_MOE, predicted_total_MOE,
                 home_win_probability, confidence, prediction_created)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game_id,
                'DeepEagle',
                row.get('pred_home_score'),
                row.get('pred_away_score'),
                row.get('pred_home_MOE'),
                row.get('pred_away_MOE'),
                row.get('pred_spread_MOE'),
                row.get('pred_total_MOE'),
                row.get('pred_home_win_prob'),
                row.get('confidence'),
                now
            ))

    conn.commit()
    conn.close()
    logger.info(f"Saved {len(predictions_df)} predictions to odds_and_predictions table")


def sync_to_cache():
    """Sync predictions to users.db prediction_cache for dashboard"""
    try:
        cfb_conn = sqlite3.connect('cfb_games.db')
        users_conn = sqlite3.connect('users.db')

        # Get predictions from odds_and_predictions table with game info
        query = '''
            SELECT
                op.game_id, g.date, g.week, g.season,
                ht.name as home_team, at.name as away_team,
                op.predicted_home_score, op.predicted_away_score,
                (op.predicted_home_score - op.predicted_away_score) as pred_spread,
                (op.predicted_home_score + op.predicted_away_score) as pred_total,
                op.confidence,
                op.prediction_created,
                g.postseason_type,
                op.latest_spread as vegas_spread,
                op.latest_total as vegas_total
            FROM odds_and_predictions op
            JOIN games g ON op.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.completed = 0
              AND op.predicted_home_score IS NOT NULL
        '''

        predictions = pd.read_sql_query(query, cfb_conn)

        if len(predictions) > 0:
            cursor = users_conn.cursor()
            now = datetime.now().isoformat()

            for _, row in predictions.iterrows():
                # Use prediction_created from cfb_games.db if available, otherwise use now
                created_at = row.get('prediction_created') or now
                postseason_type = row.get('postseason_type')

                cursor.execute('''
                    INSERT OR REPLACE INTO prediction_cache
                    (game_id, sport, season, week, game_date, home_team, away_team,
                     predicted_home_score, predicted_away_score,
                     predicted_spread, predicted_total, confidence, created_at,
                     postseason_type, vegas_spread, vegas_total)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(row['game_id']), 'CFB', int(row['season']), int(row['week']),
                    row['date'], row['home_team'], row['away_team'],
                    row['predicted_home_score'], row['predicted_away_score'],
                    row['pred_spread'], row['pred_total'], row['confidence'],
                    created_at, postseason_type,
                    row.get('vegas_spread'), row.get('vegas_total')
                ))

            users_conn.commit()
            logger.info(f"Synced {len(predictions)} predictions to cache with timestamp")

        cfb_conn.close()
        users_conn.close()
        return True, f"Synced {len(predictions)} predictions"

    except Exception as e:
        logger.error(f"Error syncing to cache: {e}")
        return False, str(e)


def update_predictions(force_week=None):
    """Main function to update predictions with latest odds"""
    logger.info("\n" + "="*60)
    logger.info("UPDATING CFB PREDICTIONS")
    logger.info("="*60)

    current_week, season = get_current_cfb_week()
    if force_week:
        current_week = force_week

    # Use 2025 season for current predictions
    season = 2025

    logger.info(f"Season: {season}, Week: {current_week}")

    # Step 1: Fetch latest odds from ESPN
    fetch_latest_odds()

    # Step 2: Check for missing Vegas lines (warn but continue)
    missing, total = check_missing_vegas_lines(season, current_week)
    if missing > 0:
        logger.warning(f"Proceeding with {missing}/{total} games missing lines...")

    # Step 3: Generate predictions
    predictions_df = generate_predictions(season, current_week)

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
