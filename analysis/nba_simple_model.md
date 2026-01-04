# NBA Simple Model

**Date:** 2026-01-04
**Code:** `nba_simple_model.py`
**Model:** `models/nba_simple_model.pkl`

---

## Overview

A clean, minimal Ridge regression model for NBA spread and total predictions. Designed as a stable baseline with no dynamic adjustments or injury features.

---

## Model Philosophy

- **Simplicity over complexity** - Only 12 features, all with meaningful coefficients
- **Flat HCA** - Universal home court advantage (learned by intercept)
- **No injury data** - Avoids noise from inconsistent injury reporting
- **Per-stat weight tracking** - Handles missing box score data gracefully

---

## Features (12)

| # | Feature | Description |
|---|---------|-------------|
| 0 | PPG diff | Points per game differential |
| 1 | PAPG diff | Points allowed per game differential |
| 2 | FG% diff | Field goal percentage differential |
| 3 | Reb diff | Rebounds differential |
| 4 | TOV diff | Turnovers differential |
| 5 | Rest diff | Days of rest differential |
| 6 | Home B2B | Home team on back-to-back (0/1) |
| 7 | Away B2B | Away team on back-to-back (0/1) |
| 8 | Net rating diff | (PPG - PAPG) differential |
| 9 | Margin diff | Same as net rating (redundant but kept) |
| 10 | Home reliability | min(home_games / 20, 1.0) |
| 11 | Away reliability | min(away_games / 20, 1.0) |

### Removed Features (from original 18)
- 3P% diff - Weak signal, high variance
- FT% diff - Weak signal
- Assists diff - Collinear with other stats
- Steals diff - Weak signal
- Blocks diff - Weak signal
- HCA constant - Redundant with intercept

---

## Performance (2025 Season Test - Fair Comparison)

| Metric | Model | Vegas |
|--------|-------|-------|
| Spread MAE | 11.05 | 10.75 |
| Winner Acc | 67.1% | 68.5% |
| Total MAE | 14.78 | 14.25 |

**Test set:** 867 games (2025 season with Vegas lines)
**Training set:** 2,379 games (2023-2024 seasons)

Note: Fair comparison uses only games with actual Vegas lines (not defaults).

---

## Coefficient Importance

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| Net rating diff | -1.53 | Strongest predictor |
| Margin diff | -1.53 | (Same as net rating) |
| PAPG diff | +1.25 | Defense matters more |
| FG% diff | -0.93 | Shooting efficiency |
| TOV diff | +0.88 | Turnovers hurt |
| Home B2B | +0.81 | Home fatigue penalty |
| Away B2B | -0.76 | Away fatigue penalty |
| PPG diff | -0.63 | Offense secondary |
| Rest diff | -0.55 | Rest advantage |
| Reb diff | -0.45 | Rebounding edge |

All 12 features have |coef| > 0.4, confirming signal.

---

## Technical Details

### Hyperparameters
```python
DECAY = 0.97          # Exponential decay for weighted averages
PREV_HALF_LIFE = 6.0  # Games until 50% blend with prev season
MIN_GAMES = 10        # Minimum games before making predictions
BLEND_WEIGHT = 0.90   # Model weight in Vegas blend (spread)
TOTAL_BLEND = 0.40    # Model weight in Vegas blend (total) - Vegas better for totals
```

### Total Features (6)
| # | Feature | Description |
|---|---------|-------------|
| 0 | Combined PPG | Home + Away PPG |
| 1 | Combined PAPG | Home + Away PAPG |
| 2 | Home B2B | Home team back-to-back (0/1) |
| 3 | Away B2B | Away team back-to-back (0/1) |
| 4 | Combined Reb | Home + Away rebounds |
| 5 | Combined TOV | Home + Away turnovers |

### Per-Stat Weight Tracking

Each box score stat has its own weight list to handle missing data:
```python
team_games[team_id][season] = {
    'ppg': [], 'ppg_wts': [],      # Always populated
    'fg_pct': [], 'fg_wts': [],    # Only when box score exists
    ...
}
```

This prevents NaN propagation when games have missing box scores (e.g., preseason).

### Previous Season Blending

Stats blend with previous season based on games played:
```python
blend = 0.5 ** (games_played / PREV_HALF_LIFE)
current_ppg = blend * prev_season_ppg + (1 - blend) * this_season_ppg
```

| Games | Current Season | Previous Season |
|-------|----------------|-----------------|
| 0 | 0% | 100% |
| 6 | 50% | 50% |
| 12 | 75% | 25% |
| 20 | 90% | 10% |

---

## Usage

### Training
```bash
python nba_simple_model.py
```

### Making Predictions
```python
from nba_simple_model import NBASimpleModel
import pickle

# Load trained model
with open('models/nba_simple_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict a game
result = model.predict(
    home_id=1,
    away_id=2,
    season=2026,
    game_date='2026-01-05',
    vegas_spread=-5.0,
    vegas_total=220.0
)

print(f"Predicted spread: {result['predicted_spread']:.1f}")
print(f"Predicted total: {result['predicted_total']:.1f}")
```

---

## Comparison with Enhanced Model

| Aspect | Simple Model | Enhanced Model |
|--------|--------------|----------------|
| Spread Features | 12 | 17 |
| Total Features | 6 | 15 |
| HCA | Flat (intercept) | Dynamic per-team |
| Injuries | None | Star injury adjustment |
| Box scores | FG%, Reb, TOV | FG%, Reb, TOV |
| Spread MAE | 11.05 | 11.09 |
| Total MAE | 14.78 | 14.86 |
| Winner Acc | 67.1% | 66.7% |

**Verdict:** Simple model slightly outperforms Enhanced on both spread and total. Extra features add complexity without improving accuracy. Vegas beats both.

---

## Model Files

- `nba_simple_model.py` - Model class and training code
- `models/nba_simple_model.pkl` - Trained model artifact
- `analysis/nba_simple_model.md` - This documentation

---

*Last updated: 2026-01-04*
