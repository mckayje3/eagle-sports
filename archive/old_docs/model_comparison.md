# Deep Learning Model Comparison

## Current TensorFlow/Keras Predictor vs Deep-Eagle PyTorch

---

## 🔍 Architecture Comparison

| Aspect | Current (TensorFlow/Keras) | Deep-Eagle (PyTorch LSTM) |
|--------|---------------------------|---------------------------|
| **Framework** | TensorFlow/Keras | PyTorch |
| **Model Type** | Dense/CNN (static) | LSTM (sequential) |
| **Hidden Units** | Unknown (loaded from .keras) | 128 |
| **Layers** | Unknown | 2 LSTM layers |
| **Dropout** | Unknown | 0.2 |
| **Input Format** | 51 static features | 15-game sequences |

---

## 🎯 Prediction Capabilities

| Feature | Current | Deep-Eagle |
|---------|---------|------------|
| **Win/Loss** | ✅ (broken - 100% home) | ✅ |
| **Spread Prediction** | ❌ (estimated from prob) | ✅ Direct regression |
| **Total Points** | ❌ (estimated average) | ✅ Direct regression |
| **Score Prediction** | ❌ (calculated) | ✅ Direct regression |

---

## 📊 Feature Engineering

### Current Approach (Static):
```python
Features (51 total):
- week, neutral_site
- home_games_played, home_wins, home_losses
- home_points_scored_avg, home_points_allowed_avg
- away_points_scored_avg, away_points_allowed_avg
- home_total_yards_avg, passing_yards_avg, rushing_yards_avg
- turnovers_avg, first_downs_avg, third_down_pct
- Differentials: win_pct_diff, points_diff, yards_diff
- H2H: h2h_games, h2h_win_pct
- Recent form: last 3 games win %
```

**Issues:**
- All features weighted equally
- No temporal awareness (early season = late season)
- Manually calculated averages
- No momentum/streak detection

### Deep-Eagle Approach (Sequential):
```python
Rolling Windows (automatic):
- 3-game rolling average (recent form)
- 5-game rolling average (short-term trend)
- 10-game rolling average (medium-term form)

Lag Features:
- lag1: Previous game stats
- lag2: Two games ago
- lag7: Week ago

Streak Features:
- winning_streak
- losing_streak
- unbeaten_streak

Context Features:
- rest_days (days since last game)
- home_away indicator
- days_in_season
- back_to_back games
```

**Advantages:**
- Captures momentum and trends
- Recent games automatically weighted more
- Built-in feature engineering
- Time series aware

---

## 🏆 Expected Performance

### Current Model:
```
Status: BROKEN
- Win/Loss: 100% home win (clearly overfitted/corrupted)
- Spread: Calculated from broken win probability
- Total: Estimated from season average (no real prediction)
```

### Deep-Eagle Model:
```
Expected (from documentation):
- Point Spread MAE: < 8 points
- Score Prediction RMSE: < 10 points
- Win/Loss Accuracy: > 60%
```

**Vegas Benchmarks:**
- Vegas spread accuracy: ~7-8 point MAE
- Vegas total accuracy: ~9-10 point MAE

**Conclusion:** Deep-Eagle targets beating Vegas, Current doesn't work

---

## 🔬 Data Format Comparison

### Current Model Input:
```python
# Single row per game (static snapshot)
[week, neutral, home_wins, home_avg_points, away_wins, away_avg_points, ...]
Shape: (1, 51)
```

### Deep-Eagle Input:
```python
# Sequence of 15 games per prediction
[
    [game_-14_features],
    [game_-13_features],
    ...
    [game_-1_features]  # Most recent
]
Shape: (15, n_features)
```

**Why Sequences Matter:**
- Team playing 5-game winning streak ≠ Team that went 3-2
- Both have same record, but momentum differs
- LSTMs capture this, static models don't

---

## ⚡ Training Differences

| Aspect | Current | Deep-Eagle |
|--------|---------|------------|
| **Loss Function** | Binary cross-entropy | MSE (regression) |
| **Optimizer** | Unknown | Adam |
| **Learning Rate** | Unknown | 0.001 |
| **Batch Size** | Unknown | 32 |
| **Early Stopping** | Unknown | Yes (patience: 15) |
| **Validation** | Standard split | Walk-forward (time series) |

**Critical Difference:** Walk-forward validation simulates real prediction (can't see future)

---

## 💾 Model Persistence

### Current:
```
cfb_model_v2.keras           # Win/loss model (broken)
cfb_model_v2_scaler.pkl      # Feature scaler
spread_model.keras           # Spread model (optional, not found)
```

### Deep-Eagle:
```
checkpoints/sports_best_model.pth   # Best model checkpoint
sports_prediction_config.yaml        # Full configuration
```

**Advantage:** Deep-Eagle saves config, reproducible

---

## 🎓 Use Case Alignment

### Your Goals:
1. ✅ Predict spreads and totals (not just win/loss)
2. ✅ Beat Vegas lines
3. ✅ Use 2024+2025 sequential game data
4. ✅ Weight 2025 season more (recent = important)
5. ✅ Rolling predictions based on stats

### Current Model Alignment:
1. ❌ Win/loss only (spreads calculated, not predicted)
2. ❌ Broken (100% home win)
3. ⚠️ Static features (doesn't use sequences)
4. ❌ No temporal weighting
5. ❌ Manual averages, no rolling

### Deep-Eagle Alignment:
1. ✅ Direct spread/total regression
2. ✅ Targets < 8 MAE (competitive with Vegas)
3. ✅ LSTM built for sequences
4. ✅ Recent games weighted via rolling windows
5. ✅ Automatic rolling features

---

## 🧪 Testing Plan

### Test 1: Data Preparation
- Prepare CFB data in both formats
- Compare feature extraction time
- Check data quality

### Test 2: Training
- Train Deep-Eagle on 2024+2025 data
- Cannot retrain current (models are pre-trained .keras files)

### Test 3: Prediction Accuracy
- Use completed 2025 games as test set
- Compare predictions to actual results
- Calculate MAE for spreads, RMSE for scores
- Compare to Vegas lines from database

### Test 4: Vegas Benchmark
- Load odds from game_odds table
- Compare model predictions to Vegas spreads/totals
- Identify games where model disagrees with Vegas
- Check if model would beat the spread

---

## 🏁 Recommendation

**Use Deep-Eagle for the following reasons:**

1. **Functional vs Broken:** Current models predict 100% home win
2. **Purpose-Built:** Deep-Eagle designed for spread/total prediction
3. **Time Series:** College football is sequential, LSTM captures this
4. **Feature Engineering:** Automatic rolling stats match your needs
5. **Performance Target:** < 8 MAE aligns with beating Vegas
6. **Data Alignment:** You have sequential game data (perfect for LSTM)
7. **Recent Weighting:** Rolling windows automatically weight 2025 more
8. **Proven Architecture:** Sports prediction guide shows it works

**Migration Path:**
1. Install PyTorch (if not already installed)
2. Prepare CFB data in Deep-Eagle format
3. Train model on 2024+2025 seasons
4. Test on completed games
5. Compare to Vegas lines
6. Deploy for Week 15+ predictions

**Keep Current Model?**
- No - it's broken and designed for wrong task
- Could try retraining, but architecture not suited for spreads/totals
- Deep-Eagle is better fit from ground up
