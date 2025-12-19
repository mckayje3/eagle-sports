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
        fetcher.update_odds('nfl', 'nfl_games.db', is_opening=False)
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
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    # Find games in this week without Vegas lines
    cursor.execute('''
        SELECT g.game_id, g.date,
               ht.display_name as home_team, at.display_name as away_team,
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
    """Generate predictions for current week"""
    logger.info(f"Generating Week {week} predictions...")

    try:
        from nfl_predictor import NFLPredictor

        predictor = NFLPredictor(
            model_path=f'models/deep_eagle_nfl_{season}.pt',
            scaler_path=f'models/deep_eagle_nfl_{season}_scaler.pkl'
        )

        if predictor.model is None:
            logger.warning("NFL model not loaded - no predictions generated")
            return None

        predictions_df = predictor.predict_upcoming(week=week)

        if predictions_df is not None and len(predictions_df) > 0:
            predictions_df.to_csv('nfl_current_predictions.csv', index=False)
            predictions_df.to_csv(f'nfl_week{week}_predictions.csv', index=False)
            save_to_database(predictions_df, season)
            logger.info(f"Generated {len(predictions_df)} predictions")
            return predictions_df

        logger.warning("No upcoming NFL games found")
        return None

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_database(predictions_df, season):
    """Save predictions to odds_and_predictions table"""
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    model_version = f'deep_eagle_nfl_{season}'

    for _, row in predictions_df.iterrows():
        game_id = int(row['game_id'])

        # Check if odds entry exists for this game
        cursor.execute('SELECT id FROM odds_and_predictions WHERE game_id = ?', (game_id,))
        existing = cursor.fetchone()

        if existing:
            # Update existing entry
            cursor.execute('''
                UPDATE odds_and_predictions
                SET predicted_home_score = ?,
                    predicted_away_score = ?,
                    prediction_date = ?,
                    model_version = ?
                WHERE game_id = ?
            ''', (
                row.get('pred_home_score'),
                row.get('pred_away_score'),
                now,
                model_version,
                game_id
            ))
        else:
            # Insert new entry with predictions
            cursor.execute('''
                INSERT INTO odds_and_predictions
                (game_id, source, predicted_home_score, predicted_away_score,
                 prediction_date, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                game_id,
                'deep_eagle',
                row.get('pred_home_score'),
                row.get('pred_away_score'),
                now,
                model_version
            ))

    conn.commit()
    logger.info(f"Saved {len(predictions_df)} predictions to odds_and_predictions")
    conn.close()


def sync_to_cache():
    """Sync predictions to users.db prediction_cache for dashboard"""
    try:
        nfl_conn = sqlite3.connect('nfl_games.db')
        users_conn = sqlite3.connect('users.db')

        # Get predictions from odds_and_predictions table
        query = '''
            SELECT
                o.game_id, g.date, g.week, g.season,
                ht.display_name as home_team, at.display_name as away_team,
                o.predicted_home_score, o.predicted_away_score,
                o.predicted_home_score - o.predicted_away_score as pred_spread,
                o.predicted_home_score + o.predicted_away_score as pred_total,
                o.prediction_date,
                o.latest_spread as vegas_spread,
                o.latest_total as vegas_total,
                g.completed,
                g.home_score as actual_home,
                g.away_score as actual_away
            FROM odds_and_predictions o
            JOIN games g ON o.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE o.predicted_home_score IS NOT NULL
        '''

        predictions = pd.read_sql_query(query, nfl_conn)

        if len(predictions) > 0:
            cursor = users_conn.cursor()
            now = datetime.now().isoformat()

            for _, row in predictions.iterrows():
                pred_spread = row['pred_spread'] or 0
                confidence = 0.5 + min(0.45, abs(pred_spread) / 20)  # Higher spread = higher confidence
                created_at = row.get('prediction_date') or now

                cursor.execute('''
                    INSERT OR REPLACE INTO prediction_cache
                    (game_id, sport, season, week, game_date, home_team, away_team,
                     predicted_home_score, predicted_away_score,
                     predicted_spread, predicted_total, confidence,
                     vegas_spread, vegas_total,
                     game_completed, actual_home_score, actual_away_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(row['game_id']), 'NFL', int(row['season']), int(row['week']),
                    row['date'], row['home_team'], row['away_team'],
                    row['predicted_home_score'], row['predicted_away_score'],
                    pred_spread, row['pred_total'], confidence,
                    row['vegas_spread'], row['vegas_total'],
                    int(row['completed']), row['actual_home'], row['actual_away'],
                    created_at
                ))

            users_conn.commit()
            logger.info(f"Synced {len(predictions)} predictions to cache")

        nfl_conn.close()
        users_conn.close()
        return True, f"Synced {len(predictions)} predictions"

    except Exception as e:
        logger.error(f"Error syncing to cache: {e}")
        return False, str(e)


def update_predictions(force_week=None):
    """Main function to update predictions with latest odds"""
    logger.info("=" * 60)
    logger.info("UPDATING NFL PREDICTIONS")
    logger.info("=" * 60)

    current_week, season = get_current_nfl_week()
    if force_week:
        current_week = force_week

    logger.info(f"Season: {season}, Week: {current_week}")

    fetch_latest_odds()

    # Check for missing Vegas lines (warn but continue)
    missing, total = check_missing_vegas_lines(season, current_week)
    if missing > 0:
        logger.warning(f"Proceeding with {missing}/{total} games missing lines...")

    predictions_df = generate_predictions(season, current_week)

    if predictions_df is not None:
        # Sync to dashboard cache
        sync_to_cache()

        logger.info("=" * 60)
        logger.info("UPDATE COMPLETE")
        logger.info("=" * 60)
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
