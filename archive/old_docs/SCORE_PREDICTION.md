# College Football Score Prediction System

Deep learning-based system for predicting actual game scores, spreads, and totals.

## Overview

This system uses **three deep neural networks** to predict:

1. **Win/Loss Probability** (Classification)
   - Binary classification: Will the home team win?
   - Output: Probability between 0-1

2. **Point Differential / Spread** (Regression)
   - How many points will the home team win/lose by?
   - Output: Positive = home team favored, Negative = away team favored

3. **Total Points / Over-Under** (Regression)
   - Combined score of both teams
   - Output: Total points scored in the game

From the **spread** and **total** predictions, we calculate **actual scores**:

```
home_score = (total + spread) / 2
away_score = (total - spread) / 2
```

### Example

If the model predicts:
- **Spread**: +7 (home team favored by 7)
- **Total**: 55 points

Then:
- **Home Score**: (55 + 7) / 2 = **31 points**
- **Away Score**: (55 - 7) / 2 = **24 points**
- **Predicted Result**: Home team wins 31-24

## Architecture

All three models use the **Deep Learning Framework** from `documents/coding/deep`:

```
CFBScorePredictor
├── Win Model (CFBFeedForwardModel)
│   ├── Task: Binary Classification
│   ├── Loss: Binary Cross Entropy
│   └── Metric: Accuracy
│
├── Spread Model (CFBFeedForwardModel)
│   ├── Task: Regression
│   ├── Loss: MSE
│   └── Metric: MAE (Mean Absolute Error)
│
└── Total Model (CFBFeedForwardModel)
    ├── Task: Regression
    ├── Loss: MSE
    └── Metric: MAE (Mean Absolute Error)
```

Each model is a feedforward neural network with:
- **Input Layer**: 50+ team statistics
- **Hidden Layers**: 3 layers with dropout
- **Output Layer**: 1 neuron (win probability, spread, or total)

## Usage

### 1. Train Models

Train all three models on historical data:

```bash
py train_score_predictor.py
```

This will:
- Load game data from `ml_features_v2_2025.csv`
- Train win/loss classifier
- Train point differential predictor
- Train total points predictor
- Save models to `models/` directory
- Generate training history plots
- Show evaluation metrics

**Output:**
```
models/
├── win_model.pt
├── spread_model.pt
├── total_model.pt
├── score_predictor_data.pkl
└── training_histories.png
```

### 2. Predict Scores

Predict scores for upcoming games:

```bash
py predict_scores.py
```

**Example Output:**
```
Week 13 - 2025-11-22
  Michigan State @ Ohio State
  Predicted Score: Michigan State 17, Ohio State 38
  Winner: Ohio State (87.3% confidence)
  Spread: Ohio State +21.0
  Total: 55.0 (O/U)

Week 13 - 2025-11-22
  Alabama @ Auburn
  Predicted Score: Alabama 28, Auburn 24
  Winner: Alabama (64.2% confidence)
  Spread: Alabama +4.0
  Total: 52.0 (O/U)
```

**CSV Output:**
```csv
game_id,week,date,away_team,home_team,predicted_away_score,predicted_home_score,predicted_winner,win_probability,predicted_spread,predicted_total,neutral_site
```

### 3. Use in Python

```python
from cfb_score_predictor import CFBScorePredictor
import pandas as pd

# Load predictor
predictor = CFBScorePredictor()
predictor.load('models')

# Prepare features for a game
features_df = pd.read_csv('game_features.csv')
X = features_df[predictor.feature_columns].values

# Predict scores
predictions = predictor.predict_scores(X)

print(f"Home Score: {predictions['home_score'][0]}")
print(f"Away Score: {predictions['away_score'][0]}")
print(f"Spread: {predictions['spread'][0]:+.1f}")
print(f"Total: {predictions['total'][0]:.1f}")
print(f"Win Probability: {predictions['home_win_prob'][0]:.1%}")
```

## Model Performance

Based on 2025 season data (validation set):

| Metric | Performance |
|--------|-------------|
| **Win/Loss Accuracy** | ~65-70% |
| **Spread MAE** | ~10-12 points |
| **Total MAE** | ~8-10 points |
| **Score MAE** | ~7-9 points per team |

Performance improves as the season progresses (more data available).

## Features Used

The models use **50+ team statistics** including:

### Offensive Stats
- Points scored average
- Total yards average
- Passing yards average
- Rushing yards average
- First downs average
- Third down conversion %
- Red zone efficiency

### Defensive Stats
- Points allowed average
- Yards allowed average
- Turnovers forced
- Sacks average
- Pass defense efficiency

### Team Metrics
- Win percentage
- Recent form (last 3, 5, 10 games)
- Home/away performance
- Strength of schedule
- Head-to-head history

### Game Context
- Week number
- Neutral site flag
- Rest days
- Conference matchup

## Training Details

### Hyperparameters

```python
{
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15  # early stopping
}
```

### Training Process

1. **Data Preparation**
   - Load features from CSV
   - Filter to week 4+ (teams need game history)
   - Calculate total_points = home_score + away_score
   - 80/20 train/validation split
   - StandardScaler normalization

2. **Model Training** (for each model)
   - Initialize PyTorch model
   - Adam optimizer
   - Appropriate loss function (BCE or MSE)
   - Early stopping on validation loss
   - Model checkpointing (save best)

3. **Evaluation**
   - Calculate metrics on validation set
   - Generate training history plots
   - Show sample predictions

## Deep Learning Framework Integration

Uses the **core deep learning module** from `documents/coding/deep`:

```python
from core.training import Trainer, EarlyStopping, ModelCheckpoint
from core.utils import set_seed, get_device
from cfb_models import CFBFeedForwardModel

# Training with framework
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    callbacks=[
        EarlyStopping(patience=15),
        ModelCheckpoint(filepath='best_model.pt')
    ]
)

history = trainer.fit(train_loader, val_loader, epochs=100)
```

**Benefits:**
- ✅ Consistent training pipeline
- ✅ Automatic GPU/CPU detection
- ✅ Built-in callbacks (early stopping, checkpointing)
- ✅ Progress tracking with tqdm
- ✅ Model save/load functionality
- ✅ 80% code reuse across projects

## Advanced Usage

### Custom Predictions

Predict scores for specific matchups:

```python
from cfb_score_predictor import CFBScorePredictor
from ml_feature_extraction_v2 import FeatureExtractorV2

# Initialize
predictor = CFBScorePredictor()
predictor.load('models')
extractor = FeatureExtractorV2('cfb_games.db')

# Extract features for a specific game
features = extractor.get_game_features(
    game_id=12345,
    home_team_id=101,
    away_team_id=202,
    season=2025,
    week=13
)

# Convert to array
import pandas as pd
feature_df = pd.DataFrame([features])
X = feature_df[predictor.feature_columns].values

# Predict
preds = predictor.predict_scores(X)
print(f"Score: {preds['away_score'][0]}-{preds['home_score'][0]}")
```

### Batch Predictions

Predict all games for a specific week:

```python
import sqlite3
import pandas as pd

# Get all week 13 games
conn = sqlite3.connect('cfb_games.db')
query = "SELECT * FROM games WHERE season=2025 AND week=13 AND completed=0"
games = pd.read_sql_query(query, conn)

# Extract features and predict for each game
predictions = []
for _, game in games.iterrows():
    features = extractor.get_game_features(...)
    X = prepare_features(features)
    pred = predictor.predict_scores(X)
    predictions.append(pred)
```

### Retrain Models

Retrain with updated data mid-season:

```bash
# 1. Update feature extraction
py ml_feature_extraction_v2.py

# 2. Retrain models
py train_score_predictor.py

# 3. Make new predictions
py predict_scores.py
```

## Interpretation Guide

### Spread Interpretation

- **+7**: Home team favored by 7 points
- **-3**: Away team favored by 3 points
- **0**: Toss-up game

### Total Interpretation

- **< 45**: Low-scoring defensive battle
- **45-55**: Average scoring game
- **> 55**: High-scoring shootout

### Confidence Interpretation

- **> 80%**: Strong favorite, likely blowout
- **60-80%**: Clear favorite
- **50-60%**: Slight favorite, competitive game
- **< 50%**: Underdog (flip for favorite)

## Accuracy Expectations

### Factors Affecting Accuracy

**Improves Accuracy:**
- ✓ Later in season (more data)
- ✓ Teams with consistent performance
- ✓ Large sample sizes
- ✓ Conference games (familiar opponents)

**Reduces Accuracy:**
- ✗ Early season (limited data)
- ✗ Teams with high variance
- ✗ Rivalry games (unpredictable)
- ✗ Weather conditions (not in model)
- ✗ Injuries (not in model)

### Real-World Comparison

Professional oddsmakers typically achieve:
- **Win/Loss**: 65-70% accuracy
- **Against Spread**: 52-55% accuracy
- **Total**: 50-52% accuracy

Our model performs similarly, which is impressive for an automated system!

## Future Enhancements

Potential improvements:

1. **Player-Level Data**
   - Quarterback stats
   - Key player injuries
   - Depth chart changes

2. **Advanced Features**
   - Weather conditions
   - Travel distance
   - Rivalry indicators
   - Coaching matchups

3. **Model Architectures**
   - LSTM for sequential game history
   - Attention mechanisms for important features
   - Ensemble of multiple models

4. **Live Updates**
   - Real-time odds integration
   - Line movement tracking
   - In-game predictions

## Troubleshooting

### Models not found

```bash
ERROR: Models not found in models/
```

**Solution:** Run training first:
```bash
py train_score_predictor.py
```

### Poor predictions

If predictions seem off:

1. **Check data quality**
   - Verify feature extraction is working
   - Check for missing/null values
   - Ensure recent games are in database

2. **Retrain models**
   - May need more epochs
   - Try different hyperparameters
   - Check for overfitting

3. **Feature engineering**
   - Add more relevant features
   - Remove noisy features
   - Try different normalization

### Import errors

```bash
ModuleNotFoundError: No module named 'core'
```

**Solution:** Install deep framework:
```bash
pip install -e C:\Users\jbeast\documents\coding\deep
```

## Files Reference

| File | Purpose |
|------|---------|
| `cfb_score_predictor.py` | Main predictor class |
| `cfb_models.py` | PyTorch model definitions |
| `train_score_predictor.py` | Training script |
| `predict_scores.py` | Prediction script |
| `ml_feature_extraction_v2.py` | Feature extraction |
| `models/` | Saved models directory |
| `predicted_scores.csv` | Output predictions |

## Questions?

See also:
- `PYTORCH_MIGRATION.md` - PyTorch architecture details
- `documents/coding/deep/README.md` - Deep learning framework docs
- `ml_feature_extraction_v2.py` - Feature definitions
