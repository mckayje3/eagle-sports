import pickle
from deep_learning_predictor import DeepLearningPredictor
import sqlite3

# Load expected feature columns
data = pickle.load(open('cfb_model_v2_scaler.pkl', 'rb'))
expected_features = data['feature_columns']

# Get features being generated
predictor = DeepLearningPredictor()
conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

cursor.execute('SELECT home_team_id, away_team_id FROM games WHERE season=2024 AND week=14 LIMIT 1')
home_id, away_id = cursor.fetchone()

# Get the feature dict before it's converted to vector
# Need to modify get_team_features temporarily to return the dict

# Let's just manually check by building the feature list
print('Expected feature order (from scaler):')
for i, feat in enumerate(expected_features[:20]):
    print(f'{i+1:2d}. {feat}')

print('\nFeature values we generated (first 20):')
X = predictor.get_team_features(home_id, away_id, 2024, 14)
for i in range(min(20, len(X[0]))):
    print(f'{i+1:2d}. {expected_features[i]:30s} = {X[0][i]}')

conn.close()
