from deep_learning_predictor import DeepLearningPredictor
import sqlite3
import numpy as np

predictor = DeepLearningPredictor()
conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

# Test Missouri @ Arkansas
cursor.execute("SELECT team_id, COALESCE(school_name, name) FROM teams WHERE name IN ('Razorbacks', 'Tigers')")
teams = {name: tid for tid, name in cursor.fetchall()}
print('Teams:', teams)
print()

# Get Arkansas (home) and Missouri (away) - need to check which is which
cursor.execute('SELECT home_team_id, away_team_id FROM games WHERE season=2024 AND week=14 LIMIT 1')
home_id, away_id = cursor.fetchone()

print(f'Testing game: Home={home_id}, Away={away_id}')
print()

# Extract features
X = predictor.get_team_features(home_id, away_id, 2024, 14)
print(f'Raw features shape: {X.shape}')
print(f'First 20 features: {X[0][:20]}')
print()

# Scale features
X_scaled = predictor.win_scaler.transform(X)
print(f'Scaled features (first 20): {X_scaled[0][:20]}')
print()

# Predict
win_prob = predictor.win_model.predict(X_scaled, verbose=0)[0][0]
print(f'Win probability: {win_prob}')

conn.close()
