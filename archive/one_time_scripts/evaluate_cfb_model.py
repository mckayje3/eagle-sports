"""
Evaluate CFB Deep Eagle Model vs Vegas
"""
import pandas as pd
import numpy as np
import torch
import pickle

# Load test data
df = pd.read_csv('cfb_2023_2024_deep_eagle_features.csv')
test_df = df[df['week'].isin([11, 12, 13, 14, 15])].copy()
print(f'Test set: {len(test_df)} games (weeks 11-15)')

# Load model and scaler
checkpoint = torch.load('models/deep_eagle_cfb_2024.pt', weights_only=False)
feature_cols = checkpoint['feature_cols']

with open('models/deep_eagle_cfb_2024_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare features
X_test = test_df[feature_cols].values
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
X_scaled = scaler.transform(X_test)

# Load model architecture
class DeepEagleModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(DeepEagleModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.3))
            prev_dim = hidden_dim
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.home_score_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        )
        self.away_score_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        home_score = self.home_score_head(features)
        away_score = self.away_score_head(features)
        return torch.cat([home_score, away_score], dim=1)

model = DeepEagleModel(input_dim=len(feature_cols))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    predictions = model(X_tensor).numpy()

test_df['pred_home_score'] = predictions[:, 0]
test_df['pred_away_score'] = predictions[:, 1]
test_df['pred_spread'] = test_df['pred_home_score'] - test_df['pred_away_score']
test_df['actual_spread'] = test_df['home_score'] - test_df['away_score']
test_df['vegas_spread'] = test_df['odds_latest_spread']

# Filter games with odds
games = test_df[abs(test_df['vegas_spread']) > 0.01].copy()
print(f'Games with odds: {len(games)}')

# Spread conventions:
# pred_spread = home_score - away_score (positive = home wins)
# vegas_spread = home team spread (negative = home favored, positive = home underdog)
# actual_spread = home_score - away_score (positive = home won)

# Who does Vegas say wins?
games['vegas_favors_home'] = games['vegas_spread'] < 0

# Who does model say wins?
games['model_favors_home'] = games['pred_spread'] > 0

# Who actually won?
games['home_won'] = games['actual_spread'] > 0

# Winner accuracy
model_winner = (games['model_favors_home'] == games['home_won']).mean()
vegas_winner = (games['vegas_favors_home'] == games['home_won']).mean()

# ATS: Did the home team cover?
# Home covers if they beat the spread: actual_spread > vegas_spread
# Example: vegas=-7, actual=+10 -> 10 > -7, home covered
games['home_covered'] = games['actual_spread'] > games['vegas_spread']

# Model's ATS pick: if model expects home to do better than Vegas, take home
# pred_spread > vegas_spread means model thinks home will cover
games['model_picks_home_ats'] = games['pred_spread'] > games['vegas_spread']
games['model_ats_correct'] = games['model_picks_home_ats'] == games['home_covered']
model_ats = games['model_ats_correct'].mean()

print()
print('=' * 70)
print('CFB DEEP EAGLE MODEL vs VEGAS - WEEKS 11-15')
print('=' * 70)
print(f'Games evaluated: {len(games)}')
print()
print('STRAIGHT-UP WINNER ACCURACY:')
print(f'  Model: {model_winner:.1%}')
print(f'  Vegas: {vegas_winner:.1%}')
print(f'  Edge:  {(model_winner - vegas_winner)*100:+.1f}%')
print()
print('AGAINST THE SPREAD (ATS):')
print(f'  Model ATS: {model_ats:.1%}')
print(f'  (Baseline: 50%)')

# Sample games for verification
print()
print('=' * 95)
print('SAMPLE GAMES (ATS Verification):')
print('=' * 95)
print('Actual       | Model Pred   | Vegas   | Actual  | Pick   | Cover  | Result')
print('Home  Away   | H_pred A_pred Pred_Sp | Spread  | Spread  | ATS    | ATS    |')
print('-' * 95)

np.random.seed(42)
sample = games.sample(12)
for idx, row in sample.iterrows():
    pick = 'HOME' if row['model_picks_home_ats'] else 'AWAY'
    cover = 'HOME' if row['home_covered'] else 'AWAY'
    result = 'WIN' if row['model_ats_correct'] else 'LOSS'
    print(f"{row['home_score']:4.0f}  {row['away_score']:4.0f}   | "
          f"{row['pred_home_score']:5.1f}  {row['pred_away_score']:5.1f}  {row['pred_spread']:6.1f}  | "
          f"{row['vegas_spread']:6.1f}  | {row['actual_spread']:6.0f}  | "
          f"{pick:6} | {cover:6} | {result}")

# Weekly breakdown
print()
print('=' * 70)
print('WEEKLY BREAKDOWN:')
print('-' * 70)
print(f"{'Week':>5} | {'Games':>5} | {'Model Win':>10} | {'Vegas Win':>10} | {'Model ATS':>10}")
print('-' * 70)
for week in sorted(games['week'].unique()):
    wk = games[games['week'] == week]
    m_win = (wk['model_favors_home'] == wk['home_won']).mean()
    v_win = (wk['vegas_favors_home'] == wk['home_won']).mean()
    m_ats = wk['model_ats_correct'].mean()
    print(f"{week:>5} | {len(wk):>5} | {m_win:>10.1%} | {v_win:>10.1%} | {m_ats:>10.1%}")
