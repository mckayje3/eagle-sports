"""
Update NHL Predictions Script
Fetches latest odds from ESPN and regenerates predictions.
Called by the dashboard "Update Predictions" button.

Uses simple Ridge regression model (goals-based features).

Usage:
    py update_predictions_nhl.py              # Update with latest odds
    py update_predictions_nhl.py --days 7     # Predictions for next N days
"""
from __future__ import annotations

import sys
import sqlite3
import logging
from datetime import datetime

import pandas as pd

from timezone_utils import utc_to_eastern_date

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = 'nhl_games.db'


def fetch_latest_odds() -> bool:
    """Fetch latest odds from ESPN via direct import."""
    logger.info("Fetching latest NHL odds from ESPN...")

    try:
        from espn_unified_odds import ESPNOddsScraper
        scraper = ESPNOddsScraper('nhl')
        scraper.scrape_recent(days=7)
        logger.info("Odds fetched successfully")
        return True
    except ImportError:
        logger.warning("espn_unified_odds not available, continuing with cached odds")
        return True
    except Exception as e:
        logger.warning(f"Odds fetch failed: {e}, continuing with cached odds")
        return True


def check_missing_vegas_lines(days: int = 7) -> tuple[int, int]:
    """
    Check for upcoming games missing Vegas lines and warn before predictions.

    Returns:
        Tuple of (games_missing_lines, total_upcoming_games)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT g.game_id, g.game_date_eastern,
               ht.display_name as home_team, at.display_name as away_team,
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
        for game in missing_lines:
            missing = []
            if game['spread'] is None:
                missing.append('spread')
            if game['total'] is None:
                missing.append('total')
            logger.warning(f"  {game['date']} - {game['matchup']} - missing: {', '.join(missing)}")
        logger.warning("")
        logger.warning("Predictions for these games will use fallback values.")
        logger.warning("=" * 60)
    else:
        logger.info(f"All {len(games)} upcoming games have Vegas lines")

    return len(missing_lines), len(games)


def generate_predictions(days: int = 7) -> pd.DataFrame | None:
    """Generate predictions using Ridge model."""
    logger.info(f"Generating NHL predictions for next {days} days...")

    try:
        from nhl_predictor import NHLPredictor

        predictor = NHLPredictor()

        if not predictor.load_models():
            logger.warning("NHL model not loaded - training new model")
            predictor.train(seasons=[2023, 2024])

        if predictor.spread_model is None:
            logger.error("NHL model still not available after training")
            return None

        # Build current team stats from completed games
        # Season 2024 = 2024-25 season, Season 2026 = 2025-26 season (current)
        conn = sqlite3.connect(DB_PATH)
        query = '''
            SELECT game_id, date, season, home_team_id, away_team_id,
                   home_score, away_score, completed
            FROM games WHERE completed = 1 AND season IN (2024, 2026)
            ORDER BY date
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()

        for _, row in df.iterrows():
            hid = row['home_team_id']
            aid = row['away_team_id']
            season = row['season']
            date = row['date']

            weight = 0.97 ** len(predictor.team_stats[hid][season]['goals_for'])

            predictor.team_stats[hid][season]['goals_for'].append(row['home_score'])
            predictor.team_stats[hid][season]['goals_against'].append(row['away_score'])
            predictor.team_stats[hid][season]['weights'].append(weight)

            predictor.team_stats[aid][season]['goals_for'].append(row['away_score'])
            predictor.team_stats[aid][season]['goals_against'].append(row['home_score'])
            predictor.team_stats[aid][season]['weights'].append(weight)

            predictor.last_game[hid] = date
            predictor.last_game[aid] = date

        predictions_df = predictor.predict_upcoming(days=days)

        if predictions_df is not None and len(predictions_df) > 0:
            predictions_df.to_csv('nhl_current_predictions.csv', index=False)
            save_to_database(predictions_df)
            logger.info(f"Generated {len(predictions_df)} predictions")
            return predictions_df

        logger.warning("No upcoming NHL games found")
        return None

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_database(predictions_df: pd.DataFrame) -> None:
    """Save predictions to nhl_games.db odds_and_predictions table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for _, row in predictions_df.iterrows():
        game_id = int(row['game_id'])

        # Calculate predicted scores from spread and total
        pred_spread = row.get('pred_spread', 0)
        pred_total = row.get('pred_total', 6.0)
        pred_home = (pred_total - pred_spread) / 2
        pred_away = (pred_total + pred_spread) / 2

        cursor.execute('''
            INSERT INTO odds_and_predictions (game_id, predicted_home_score, predicted_away_score,
                confidence, prediction_created)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                predicted_home_score = excluded.predicted_home_score,
                predicted_away_score = excluded.predicted_away_score,
                confidence = excluded.confidence,
                prediction_created = excluded.prediction_created
        ''', (
            game_id,
            round(pred_home, 1),
            round(pred_away, 1),
            0.5,  # Default confidence for Ridge model
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()


def sync_to_cache() -> tuple[bool, str]:
    """Sync predictions to users.db prediction_cache for dashboard."""
    try:
        nhl_conn = sqlite3.connect(DB_PATH)
        users_conn = sqlite3.connect('users.db')

        query = '''
            SELECT
                op.game_id, g.date, g.season,
                ht.display_name as home_team, at.display_name as away_team,
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

        predictions = pd.read_sql_query(query, nhl_conn)

        if len(predictions) > 0:
            cursor = users_conn.cursor()
            now = datetime.now().isoformat()

            for _, row in predictions.iterrows():
                confidence = row['confidence'] if row['confidence'] else 0.5
                created_at = row.get('prediction_date') or now
                game_date_eastern = utc_to_eastern_date(row['date']) or row['date']

                cursor.execute('''
                    INSERT OR REPLACE INTO prediction_cache
                    (game_id, sport, season, week, game_date, home_team, away_team,
                     predicted_home_score, predicted_away_score,
                     predicted_spread, predicted_total, confidence, created_at,
                     vegas_spread, vegas_total)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(row['game_id']), 'NHL', int(row['season']), 0,
                    game_date_eastern, row['home_team'], row['away_team'],
                    row['pred_home_score'], row['pred_away_score'],
                    row['pred_spread'], row['pred_total'], confidence,
                    created_at,
                    row.get('vegas_spread'), row.get('vegas_total')
                ))

            users_conn.commit()
            logger.info(f"Synced {len(predictions)} predictions to cache")

        nhl_conn.close()
        users_conn.close()
        return True, f"Synced {len(predictions)} predictions"

    except Exception as e:
        logger.error(f"Error syncing to cache: {e}")
        return False, str(e)


def update_predictions(days: int = 7) -> tuple[bool, pd.DataFrame | None]:
    """Main function to update predictions with latest odds."""
    logger.info("\n" + "=" * 60)
    logger.info("UPDATING NHL PREDICTIONS")
    logger.info("=" * 60)

    fetch_latest_odds()

    missing, total = check_missing_vegas_lines(days)
    if missing > 0:
        logger.warning(f"Proceeding with {missing}/{total} games missing lines...")

    predictions_df = generate_predictions(days)

    if predictions_df is not None:
        sync_to_cache()

        logger.info("\n" + "=" * 60)
        logger.info("UPDATE COMPLETE")
        logger.info("=" * 60)
        return True, predictions_df
    else:
        logger.error("Failed to generate predictions")
        return False, None


def main() -> None:
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
