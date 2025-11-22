"""
Populate prediction cache with REAL games from cfb_games.db
Uses ENSEMBLE predictor for predictions with confidence intervals
"""
import sqlite3
from datetime import datetime
from deep_eagle_ensemble import EnsemblePredictor

# Connect to both databases
cfb_conn = sqlite3.connect('cfb_games.db')
users_conn = sqlite3.connect('users.db')

cfb_cursor = cfb_conn.cursor()
users_cursor = users_conn.cursor()

# Use the ensemble predictor which combines stats + power ratings
print("[INFO] Using ENSEMBLE predictor (stats + power ratings)")
print("       Predictions include confidence intervals")
predictor = EnsemblePredictor(sport='cfb')

def get_team_name(team_id):
    """Get team name from ID"""
    cfb_cursor.execute('SELECT COALESCE(school_name, display_name, name) FROM teams WHERE team_id = ?', (team_id,))
    result = cfb_cursor.fetchone()
    return result[0] if result else f"Team_{team_id}"

def populate_week(week=13, season=2024):
    """Populate predictions for a specific week with real games"""
    print(f"Populating Week {week}, Season {season} with REAL games")
    print("=" * 60)

    # First, clear any existing fake predictions for this week
    users_cursor.execute(
        'DELETE FROM prediction_cache WHERE week=? AND season=?',
        (week, season)
    )
    users_conn.commit()
    print(f"Cleared existing predictions for Week {week}")

    # Get real games from cfb_games.db (excluding TBD placeholders)
    cfb_cursor.execute('''
        SELECT
            game_id,
            home_team_id,
            away_team_id,
            date,
            home_score,
            away_score,
            completed
        FROM games
        WHERE season = ? AND week = ?
        AND home_team_id > 0 AND away_team_id > 0
        ORDER BY date
    ''', (season, week))

    games = cfb_cursor.fetchall()
    print(f"Found {len(games)} real games for Week {week}")
    print()

    count = 0
    for game_id, home_id, away_id, date, home_score, away_score, completed in games:
        home_team = get_team_name(home_id)
        away_team = get_team_name(away_id)

        # Generate prediction using ensemble predictor (includes confidence)
        prediction = predictor.predict_game(home_id, away_id, season, week)

        pred_home = prediction['predicted_home_score']
        pred_away = prediction['predicted_away_score']
        spread = prediction['predicted_spread']
        total = prediction['predicted_total']
        win_prob = prediction['home_win_probability']
        confidence = prediction.get('confidence', 0.85)
        spread_low = prediction.get('spread_low', spread - 3)
        spread_high = prediction.get('spread_high', spread + 3)
        total_low = prediction.get('total_low', total - 5)
        total_high = prediction.get('total_high', total + 5)

        # Insert into prediction_cache with confidence intervals
        users_cursor.execute('''
            INSERT INTO prediction_cache (
                game_id, sport, season, week,
                home_team, away_team, game_date,
                predicted_home_score, predicted_away_score,
                predicted_spread, predicted_total,
                home_win_probability,
                confidence, spread_low, spread_high, total_low, total_high,
                actual_home_score, actual_away_score,
                game_completed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_id, 'cfb', season, week,
            home_team, away_team, date,
            pred_home, pred_away,
            spread, total,
            win_prob,
            confidence, spread_low, spread_high, total_low, total_high,
            home_score if completed else None,
            away_score if completed else None,
            completed
        ))

        status = f"Final: {away_score}-{home_score}" if completed else "Upcoming"
        conf_icon = "🟢" if confidence >= 0.90 else "🟡" if confidence >= 0.80 else "🔴"
        print(f"[OK] {away_team} @ {home_team} ({date[:10]}) - {status}")
        print(f"     Predicted: {pred_away:.0f}-{pred_home:.0f} | Spread: {spread:+.1f} | Total: {total:.1f} | Conf: {conf_icon} {confidence:.0%}")
        count += 1

    users_conn.commit()
    print()
    print("=" * 60)
    print(f"Added {count} real games with ENSEMBLE PREDICTIONS!")
    print("Using ensemble: stats-based (50%) + power ratings (25%) + LSTM (25%)")
    print("Confidence intervals based on model agreement")

# Populate Week 14 (upcoming rivalry week)
populate_week(week=14, season=2024)

# Close connections
cfb_conn.close()
users_conn.close()
