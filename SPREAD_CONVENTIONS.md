# Spread Conventions Guide

This document explains how spreads are stored, calculated, and displayed in Eagle Eye Sports Tracker.

## Standard Betting (Vegas) Notation

In sports betting, spreads are shown from the **home team's perspective**:
- **Negative spread** = Home team is favored (must win by that many points)
- **Positive spread** = Home team is underdog (can lose by up to that many points)

### Examples:
- "Chiefs -7.0" = Chiefs (home) are 7-point favorites
- "Broncos +3.0" = Broncos (home) are 3-point underdogs

## Internal Data Storage

### ALL Spreads Use Vegas Convention

Both model predictions and Vegas lines use the same convention:
```
spread = away_score - home_score
```
- **Negative value** = Home team favored (home scored more)
- **Positive value** = Away team favored (away scored more)

This applies to:
- `predicted_spread` (model output)
- `vegas_spread` / `latest_spread` / `opening_spread` (from sportsbooks)
- `actual_spread` (game result)

### Why This Convention?

Using consistent conventions throughout the system:
1. Eliminates confusing negation logic
2. Allows direct comparison: `deviation = model_spread - vegas_spread`
3. Matches industry standard for spread representation

## Deviation Calculation

```python
deviation = model_spread - vegas_spread
```

- **Negative deviation** = Model favors HOME more than Vegas
  - Model spread is more negative (home winning by more)
  - Bet: HOME to cover

- **Positive deviation** = Model favors AWAY more than Vegas
  - Model spread is less negative (or more positive)
  - Bet: AWAY to cover

### Example:
- Vegas: -7 (home favored by 7)
- Model: -10 (model thinks home wins by 10)
- Deviation: -10 - (-7) = -3
- Interpretation: Model favors HOME 3 pts more than Vegas → Bet HOME

## Display

Spreads display directly with their sign:
```python
st.write(f"{spread:+.1f}")  # Shows "+7.0" or "-3.5"
```

## Quick Reference

| Scenario | Spread Value | Display |
|----------|-------------|---------|
| Home favored by 7 | `-7.0` | "-7.0" |
| Away favored by 3 | `+3.0` | "+3.0" |
| Pick 'em | `0.0` | "+0.0" |

## Files Using This Convention

- `betting_tracker.py` - Recommendation generation
- `streamlit_app.py` - Display and comparison
- `update_predictions_*.py` - Prediction sync (all 4 sports)
- All database queries calculating `pred_spread`
