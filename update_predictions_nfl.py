"""
Update Predictions Script
Fetches latest odds and regenerates predictions without full scrape.
Called by the dashboard "Update Predictions" button.

Usage:
    py update_predictions.py              # Update with latest odds
    py update_predictions.py --week 14    # Force specific week
"""

import sys
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
    """
    Calculate current NFL week and season based on date.
    Returns (week, season, is_playoffs).

    NFL season convention:
    - Season year = year the season STARTS (e.g., 2024 season runs Sept 2024 - Feb 2025)
    - Regular season: Weeks 1-18 (Sept - Jan)
    - Playoffs: Jan-Feb (stored as weeks 1-5 in DB for postseason)

    Note: Database stores playoff weeks as 1-5, not 19-22.
    """
    today = datetime.now()

    # Determine season year (NFL season spans two calendar years)
    # Jan-Feb = previous year's playoffs
    # March-Aug = offseason
    # Sept-Dec = current year's regular season
    if today.month >= 9:
        season = today.year
    elif today.month <= 2:
        # Jan/Feb = we're in the previous year's playoffs
        season = today.year - 1
    else:
        # March-August: offseason, use previous season for reference
        season = today.year - 1

    # Season start dates
    season_starts = {
        2024: datetime(2024, 9, 5),
        2025: datetime(2025, 9, 4),
        2026: datetime(2026, 9, 10),
    }
    season_start = season_starts.get(season, datetime(season, 9, 5))

    # Calculate week number
    if today < season_start:
        # Before season starts
        return 1, season, False

    days_since_start = (today - season_start).days
    week = (days_since_start // 7) + 1

    # After week 18, we're in playoffs
    # (This will be > 18 for Jan dates since we calculated from Sept)
    is_playoffs = week > 18 or today.month in (1, 2)

    return week, season, is_playoffs


def fetch_latest_odds():
    """Fetch latest odds from ESPN API"""
    logger.info("Fetching latest NFL odds from ESPN...")

    try:
        from espn_unified_odds import ESPNOddsScraper
        scraper = ESPNOddsScraper('nfl')
        scraper.scrape_recent(days=7)
        logger.info("Odds fetched successfully")
        return True
    except ImportError:
        logger.warning("espn_unified_odds not available, continuing with cached odds")
        return True
    except Exception as e:
        logger.warning(f"Odds fetch failed: {e}, continuing with cached odds")
        return True


def get_upcoming_games(season: int, is_playoffs: bool = False):
    """
    Get upcoming NFL games from database.

    For regular season: queries by week
    For playoffs: queries by date (next 10 days of incomplete games, any season)

    Returns list of game dicts with odds.
    """
    conn = sqlite3.connect('nfl_games.db')

    if is_playoffs:
        # Playoffs: get all upcoming incomplete games regardless of season/week
        # This handles cases where system date doesn't match DB season
        query = '''
            SELECT g.game_id, g.date, g.week, g.season,
                   g.home_team_id, g.away_team_id,
                   ht.display_name as home_team, at.display_name as away_team,
                   o.latest_spread as vegas_spread, o.latest_total as vegas_total
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.completed = 0
              AND g.date >= date('now')
              AND g.date <= date('now', '+10 days')
            ORDER BY g.date
        '''
        games = pd.read_sql_query(query, conn)
    else:
        # Regular season: handled by existing week-based logic
        games = pd.DataFrame()

    conn.close()
    return games


def check_missing_vegas_lines(season, week=None, is_playoffs=False):
    """
    Check for games missing Vegas lines.

    Args:
        season: NFL season year
        week: Week number (for regular season)
        is_playoffs: If True, check upcoming playoff games by date

    Returns:
        Tuple of (games_missing_lines, total_games)
    """
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    if is_playoffs:
        # Playoffs: check by date, ignore season (handles date mismatches)
        cursor.execute('''
            SELECT g.game_id, g.date, g.week,
                   ht.display_name as home_team, at.display_name as away_team,
                   o.latest_spread, o.latest_total
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.completed = 0
              AND g.date >= date('now')
              AND g.date <= date('now', '+10 days')
            ORDER BY g.date
        ''')
        week_label = "Playoff"
    else:
        # Regular season: check by week
        cursor.execute('''
            SELECT g.game_id, g.date, g.week,
                   ht.display_name as home_team, at.display_name as away_team,
                   o.latest_spread, o.latest_total
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.season = ? AND g.week = ? AND g.completed = 0
            ORDER BY g.date
        ''', (season, week))
        week_label = f"Week {week}"

    games = cursor.fetchall()
    conn.close()

    missing_lines = []
    for game_id, game_date, game_week, home, away, spread, total in games:
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
        logger.info(f"All {len(games)} {week_label} games have Vegas lines")

    return len(missing_lines), len(games)


def generate_predictions(season, week=None, is_playoffs=False):
    """
    Generate predictions for NFL games.

    Args:
        season: NFL season year
        week: Week number (for regular season)
        is_playoffs: If True, generate playoff predictions

    Returns:
        DataFrame of predictions or None
    """
    if is_playoffs:
        logger.info("Generating Playoff predictions...")
        return generate_playoff_predictions(season)
    else:
        logger.info(f"Generating Week {week} predictions...")
        return generate_regular_season_predictions(season, week)


def generate_regular_season_predictions(season, week):
    """Generate predictions for regular season week."""
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


def generate_playoff_predictions(season):
    """
    Generate predictions for playoff games using simple Ridge model.

    Uses hardcoded PLAYOFF_GAMES from predict_nfl_playoffs.py since
    playoff matchups aren't in database until each round completes.
    """
    try:
        # Import the playoff model builder and games list
        from predict_nfl_playoffs import build_model_state, predict_game, PLAYOFF_GAMES

        if not PLAYOFF_GAMES:
            logger.warning("No playoff games defined in predict_nfl_playoffs.py")
            return None

        logger.info(f"Found {len(PLAYOFF_GAMES)} playoff games to predict")

        # Build model state
        state = build_model_state()
        logger.info(f"Model trained on {len(state['team_stats'])} teams")

        predictions = []
        for away_team, home_team, vegas_spread, game_date, time_slot in PLAYOFF_GAMES:
            # Default total if not available
            vegas_total = 44.0

            pred, err = predict_game(
                state,
                away_team,
                home_team,
                vegas_spread,
                game_date
            )

            if err:
                logger.warning(f"Could not predict {away_team} @ {home_team}: {err}")
                continue

            # Calculate scores from spread and total
            # spread = away - home, total = away + home
            # Solving: home = (total - spread) / 2, away = (total + spread) / 2
            pred_spread = pred['model_spread']
            pred_home = (vegas_total - pred_spread) / 2
            pred_away = (vegas_total + pred_spread) / 2

            predictions.append({
                'game_id': 0,  # No DB game_id for hardcoded games
                'date': game_date,
                'time_slot': time_slot,
                'week': 'WC',  # Wild Card, will be DIV, CONF, SB for later rounds
                'home_team': home_team,
                'away_team': away_team,
                'pred_home_score': round(pred_home, 1),
                'pred_away_score': round(pred_away, 1),
                'pred_spread': round(pred_spread, 1),
                'pred_total': round(vegas_total, 1),
                'vegas_spread': vegas_spread,
                'vegas_total': vegas_total,
                'edge': round(pred['edge'], 1),
                'is_playoff': True,
                # Include additional stats for display
                'home_ppg': pred['home_ppg'],
                'away_ppg': pred['away_ppg'],
                'home_form': pred['home_form'],
                'away_form': pred['away_form'],
            })

        if not predictions:
            logger.warning("No predictions generated for playoff games")
            return None

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv('nfl_current_predictions.csv', index=False)
        predictions_df.to_csv('nfl_playoff_predictions.csv', index=False)

        # Log summary
        logger.info(f"Generated {len(predictions_df)} playoff predictions")
        for _, p in predictions_df.iterrows():
            edge_dir = "AWAY" if p['edge'] > 0 else "HOME"
            logger.info(f"  {p['away_team']} @ {p['home_team']}: "
                       f"Vegas {p['vegas_spread']:+.1f}, Model {p['pred_spread']:+.1f}, "
                       f"Edge {p['edge']:+.1f} ({edge_dir})")

        return predictions_df

    except ImportError as e:
        logger.error(f"Could not import playoff predictor: {e}")
        return None
    except Exception as e:
        logger.error(f"Error generating playoff predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_database(predictions_df, season, is_playoff=False):
    """Save predictions to odds_and_predictions table"""
    conn = sqlite3.connect('nfl_games.db')
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    model_version = f'playoff_ridge_{season}' if is_playoff else f'deep_eagle_nfl_{season}'

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
                o.predicted_away_score - o.predicted_home_score as pred_spread,
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


def update_predictions(force_week=None, force_playoffs=None):
    """
    Main function to update predictions with latest odds.

    Args:
        force_week: Force specific week number
        force_playoffs: Force playoff mode (True/False), or None for auto-detect
    """
    logger.info("=" * 60)
    logger.info("UPDATING NFL PREDICTIONS")
    logger.info("=" * 60)

    current_week, season, is_playoffs = get_current_nfl_week()

    # Allow forcing week or playoff mode
    if force_week:
        current_week = force_week
        is_playoffs = False  # Explicit week = regular season
    if force_playoffs is not None:
        is_playoffs = force_playoffs

    if is_playoffs:
        logger.info(f"Season: {season}, Mode: PLAYOFFS")
    else:
        logger.info(f"Season: {season}, Week: {current_week}")

    fetch_latest_odds()

    # Check for missing Vegas lines (warn but continue)
    missing, total = check_missing_vegas_lines(season, week=current_week, is_playoffs=is_playoffs)
    if missing > 0:
        logger.warning(f"Proceeding with {missing}/{total} games missing lines...")

    predictions_df = generate_predictions(season, week=current_week, is_playoffs=is_playoffs)

    if predictions_df is not None:
        # Sync to dashboard cache
        sync_to_cache()

        # Generate betting recommendations
        try:
            from betting_tracker import BettingTracker
            tracker = BettingTracker()
            recs = tracker.generate_recommendations('NFL', week=current_week)
            saved = tracker.save_recommendations(recs)
            logger.info(f"Generated {len(recs)} betting recommendations, saved {saved}")
        except Exception as e:
            logger.warning(f"Could not generate betting recommendations: {e}")

        logger.info("=" * 60)
        logger.info("UPDATE COMPLETE")
        logger.info("=" * 60)
        return True, predictions_df
    else:
        logger.error("Failed to generate predictions")
        return False, None


def main():
    force_week = None
    force_playoffs = None

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--week' and i < len(sys.argv) - 1:
            force_week = int(sys.argv[i + 1])
        elif arg.startswith('--week='):
            force_week = int(arg.split('=')[1])
        elif arg == '--playoffs':
            force_playoffs = True

    success, _ = update_predictions(force_week, force_playoffs)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
