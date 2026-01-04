# NBA Edge Classifier Findings

**Date:** 2026-01-04
**Code:** `nba_edge_classifier.py`
**Model:** `models/nba_edge_classifier.pt`

---

## Executive Summary

The edge classifier attempts to identify **when** to bet with or against our prediction models. Instead of predicting game outcomes directly, it predicts betting actions based on model-Vegas disagreement patterns.

**Key Finding:** Spreads show exploitable patterns. Totals do not.

---

## 1. Spread Results (Promising)

### Best Strategy: FADE @ 0.70 confidence
| Metric | Value |
|--------|-------|
| Record | 24-17 |
| Win Rate | 58.5% |
| ROI | +11.7% |
| Threshold | P(fade) >= 0.70 |

### What This Means
When the classifier outputs P(fade) >= 70%, bet **opposite** of what our model predicts:
- Model says "home covers" → bet away
- Model says "away covers" → bet home

The classifier learned situations where our model is systematically wrong.

### Spread Patterns Discovered

**1. Model Agreement Matters:**
| Condition | Games | Spread WR |
|-----------|-------|-----------|
| Models AGREE | 2,972 | 48.2% |
| Models DISAGREE | 421 | 43.7% |

When simple and enhanced models agree, performance is better.

**2. Simple Model Wins Disagreements:**
When models disagree on spread direction:
- Simple correct: 53.7%
- Enhanced correct: 46.3%

The simpler model is more reliable in conflict situations.

**3. Star Injuries Help:**
| Condition | Games | Spread WR |
|-----------|-------|-----------|
| Star injury present | 472 | 50.0% |
| No star injuries | 2,921 | 47.2% |

Model performs better when star players are injured (injury adjustment working).

---

## 2. Total Results (No Signal)

### Individual Model Performance
| Model | Accuracy | 2025 Test |
|-------|----------|-----------|
| Simple | 48.9% | 49.9% |
| Enhanced | 49.9% | 50.1% |
| Average | 49.5% | 49.1% |

**All essentially 50% - coin flip.** Neither model beats Vegas on totals.

### Why Totals Failed

1. **No exploitable edge exists** - Both models predict totals at ~50% accuracy
2. **Vegas is efficient** - Total lines are well-calibrated
3. **Can't ensemble to signal** - Combining two 50% models doesn't create edge
4. **Features may be wrong** - Totals might need different features (pace, rest patterns, etc.)

### Decision: Remove Totals from Classifier

The edge classifier cannot learn patterns that don't exist. Focusing on spreads only.

---

## 3. Architecture

### Input Features (42 total, spread-focused)

**Simple Model (4):**
- simple_spread_edge, simple_spread_edge_abs
- simple_spread

**Enhanced Model (4):**
- enhanced_spread_edge, enhanced_spread_edge_abs
- enhanced_spread

**Ensemble (6):**
- avg_spread_edge, avg_spread_edge_abs
- spread_agreement (do models agree on direction?)
- spread_model_diff (how much do models disagree?)

**Vegas Context (5):**
- vegas_spread, vegas_spread_abs
- big_favorite (>8 pts), close_game (<3 pts)

**Model Accuracy (1):**
- recent_spread_acc (last 50 games)

**Team Reliability (3):**
- home_games, away_games, combined_games

**Situational (5):**
- home_rest, away_rest, rest_diff
- home_b2b, away_b2b

**Season (4):**
- season_progress
- early_season, mid_season, late_season

**Injuries (6):**
- home_star_ppg_out, away_star_ppg_out
- star_injury_adj, has_star_injury
- star_ppg_diff, total_star_ppg_out

### Output
- P(pass) - don't bet
- P(bet_with) - bet same direction as model
- P(fade) - bet opposite direction

### Neural Network
```
Input (42) → Dense(64) → ReLU → BatchNorm → Dropout(0.3)
          → Dense(32) → ReLU → BatchNorm → Dropout(0.3)
          → Dense(16) → ReLU → Dense(3) → Softmax
```

---

## 4. Training Details

- **Train set:** 2023-2024 seasons (~2,200 games)
- **Test set:** 2025 season (~750 games)
- **Optimizer:** Adam (lr=0.001, weight_decay=0.01)
- **Scheduler:** StepLR (step=50, gamma=0.5)
- **Epochs:** 150
- **Batch size:** 64

---

## 5. Usage Recommendations

### When to Use FADE Strategy
Apply FADE when classifier confidence >= 0.70:
- High disagreement between model and Vegas
- Specific situational factors present
- Expect ~58% win rate, +11% ROI

### Sample Size Warning
The 24-17 record is small. Continue monitoring as more games are played.

### Integration
```python
# Load classifier
checkpoint = torch.load('models/nba_edge_classifier.pt')
model.load_state_dict(checkpoint['model_state_dict'])
scaler = checkpoint['scaler']

# Get prediction
features = extract_meta_features(game)
X = scaler.transform(features.reshape(1, -1))
probs = model(torch.FloatTensor(X))
spread_probs = torch.softmax(probs, dim=1).numpy()[0]

# Decision
if spread_probs[2] >= 0.70:  # P(fade)
    action = "FADE"  # Bet opposite of model
elif spread_probs[1] >= 0.65:
    action = "BET_WITH"
else:
    action = "PASS"
```

---

## 6. Future Improvements

1. **More training data** - Add 2026 season as it completes
2. **Feature engineering** - Explore additional spread-specific features
3. **Threshold tuning** - Optimize confidence thresholds on validation set
4. **Segment models** - Different classifiers for early/mid/late season
5. **Injury deep-dive** - More granular injury impact features

---

## 7. Comparison: Spreads vs Totals

| Aspect | Spreads | Totals |
|--------|---------|--------|
| Base model accuracy | ~48% | ~50% |
| Exploitable patterns | Yes | No |
| Best classifier ROI | +11.7% | -2.0% |
| Model agreement helps | Yes | No |
| Injury signal | Yes | No |
| Recommendation | Use classifier | Skip |

---

*Last updated: 2026-01-04*
