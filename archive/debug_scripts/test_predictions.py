from deep_learning_predictor import DeepLearningPredictor
import sqlite3

predictor = DeepLearningPredictor()
conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

# Get 3 different games from Week 14
cursor.execute('SELECT game_id, home_team_id, away_team_id FROM games WHERE season=2024 AND week=14 ORDER BY game_id LIMIT 3')
games = cursor.fetchall()

print('Testing updated feature extraction with real stats:\n')

for gid, home_id, away_id in games:
    # Get team names
    cursor.execute("SELECT COALESCE(school_name, name) FROM teams WHERE team_id=?", (home_id,))
    home = cursor.fetchone()[0]
    cursor.execute("SELECT COALESCE(school_name, name) FROM teams WHERE team_id=?", (away_id,))
    away = cursor.fetchone()[0]

    # Get prediction
    pred = predictor.predict_game(home_id, away_id, 2024, 14)

    print(f'{away} @ {home}:')
    print(f'  Score: {pred["predicted_away_score"]}-{pred["predicted_home_score"]}')
    print(f'  Spread: {pred["predicted_spread"]:+.1f}, Total: {pred["predicted_total"]:.1f}')
    print(f'  Home Win%: {pred["home_win_probability"]:.1%}\n')

conn.close()
