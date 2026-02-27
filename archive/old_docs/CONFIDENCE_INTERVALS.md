# Prediction Confidence Intervals & Margin of Error

## Overview

The score predictions now include **margin of error** and **confidence intervals** to show the uncertainty in predictions. This helps you understand how reliable each prediction is.

## What is Margin of Error?

The margin of error represents the average prediction error based on the model's validation performance:

- **Spread Margin**: ±12.4 points
- **Total Margin**: ±11.4 points

These values come from the **Mean Absolute Error (MAE)** during model validation, which measures how far off predictions typically are from actual results.

## Example: Tennessee vs Florida

```
Predicted Score: Tennessee 36, Florida 25
Winner: Tennessee Volunteers (83.0% confidence)

SPREAD:
  Prediction: Florida -11.4 points (Tennessee favored by 11.4)
  Margin of Error: ±12.4 points
  Confidence Range: -23.8 to +1.0

TOTAL:
  Prediction: 60.5 points
  Margin of Error: ±11.4 points
  Confidence Range: 49.1 to 71.9
```

### Interpreting the Spread

- **Prediction**: Tennessee favored by 11.4 points
- **Confidence Range**: Tennessee could win by as much as 23.8 points OR Florida could win by as much as 1.0 point
- **What it means**: While Tennessee is favored, the margin is within the error range, so this could be closer than predicted

### Interpreting the Total

- **Prediction**: 60.5 combined points
- **Confidence Range**: Could be as low as 49.1 or as high as 71.9
- **What it means**: Actual combined score will likely fall within this 23-point range

## How to Use These Values

### 1. Assess Prediction Reliability

**High Confidence Prediction:**
```
Spread: +30.0 ±12.4
Range: +17.6 to +42.4
```
Even with the margin of error, the favorite is still heavily favored (17.6+ point margin).

**Low Confidence Prediction:**
```
Spread: +3.0 ±12.4
Range: -9.4 to +15.4
```
The margin of error includes both teams potentially winning. This is a toss-up game.

### 2. Identify Close Games

If the confidence range crosses zero (includes both positive and negative values), the game could go either way:

```
Spread: -5.0 ±12.4
Range: -17.4 to +7.4  ← Crosses zero, could go either way
```

### 3. Compare to Vegas Lines

If Vegas has a different spread, check if it falls within your confidence range:

```
Model: Team A -7.0 ±12.4 (Range: +5.4 to +19.4)
Vegas: Team A -10.0

Vegas line of -10.0 falls within our confidence range,
so our model agrees with Vegas within margin of error.
```

## Statistical Interpretation

### Confidence Intervals

The ranges shown represent approximately **80% confidence intervals**, meaning:

- **80% of the time**, the actual value will fall within this range
- **20% of the time**, the actual value will be outside this range

### Why These Numbers?

The margins come from validation performance:

1. **Training Data**: 498 games from 2025 season (weeks 4-12)
2. **Validation Split**: 80% training, 20% validation
3. **Performance Metrics**:
   - Spread MAE: 12.4 points (average error)
   - Total MAE: 11.4 points (average error)

## Real-World Examples

### Example 1: Clear Favorite
```
Alabama Crimson Tide @ Auburn Tigers
Predicted: Auburn 7, Alabama 44
Spread: Alabama +36.7 ±12.4
Range: +24.3 to +49.1

Analysis: Even with the margin of error, Alabama is heavily
favored. The worst case is still a 24-point Alabama win.
```

### Example 2: Toss-Up Game
```
Pittsburgh Panthers @ Georgia Tech Yellow Jackets
Predicted: Pittsburgh 30, Georgia Tech 32
Spread: Georgia Tech +1.7 ±12.4
Range: -10.7 to +14.1

Analysis: The range crosses zero, meaning either team could
realistically win. This is a true toss-up despite the slight
Georgia Tech prediction.
```

### Example 3: High-Scoring Game
```
Total: 73.4 ±11.4
Range: 62.0 to 84.8

Analysis: Even the low end (62.0) is high-scoring. This game
is very likely to be an offensive shootout.
```

### Example 4: Defensive Battle
```
Total: 40.7 ±11.4
Range: 29.3 to 52.1

Analysis: The high end (52.1) is still relatively low-scoring.
Expect a defensive game.
```

## Dashboard Display

In the dashboard, predictions now show:

```
Spread & Total
--------------
Spread: Georgia Tech Yellow Jackets +1.7
Margin: ±12.4 pts
Range: -10.7 to +14.1

Total: 62.6 (O/U)
Margin: ±11.4 pts
Range: 51.2 to 74.0
```

## Betting Implications

### When to Bet

**High Confidence (Outside Margin of Error):**
- Model shows Team A -20 ±12.4
- Even worst case is Team A -7.6
- Vegas has Team A -10
- ✓ Model is very confident in larger margin

**Avoid Bets:**
- Prediction is close (within 3-4 points)
- Confidence range is wide (crosses zero)
- Vegas line is far outside your range

### Value Identification

Look for cases where:
1. Vegas line is outside your confidence range
2. Your model shows high confidence (narrow effective range)
3. Teams are consistently over/underperforming spreads

## Improving Predictions

As more games are played and the models retrain:

1. **Margin of error should decrease** - More data = better predictions
2. **Confidence ranges narrow** - Model becomes more certain
3. **Accuracy improves** - Later season predictions are typically better

## Technical Notes

### Why MAE for Margin?

- **MAE (Mean Absolute Error)** is easy to interpret
- Directly represents "typical prediction error"
- More robust to outliers than RMSE
- In same units as prediction (points)

### Calculating Confidence Intervals

```python
spread_confidence_low = spread - spread_margin
spread_confidence_high = spread + spread_margin
```

For a more formal statistical approach, you could use:
- Standard error × 1.28 for 80% confidence
- Standard error × 1.96 for 95% confidence

### Limitations

1. **Assumes normal distribution** of errors
2. **Same margin for all games** (could vary by matchup)
3. **Based on validation set** (may not generalize perfectly)
4. **Doesn't account for**:
   - Weather conditions
   - Player injuries
   - Motivational factors
   - Coaching changes

## Summary

✅ **Margin of Error** shows typical prediction uncertainty
✅ **Confidence Ranges** give realistic outcome bounds
✅ **Crossing Zero** indicates toss-up games
✅ **Wide Ranges** mean less confidence
✅ **Narrow Ranges** mean more confidence
✅ **Compare to Vegas** to find value bets

Use these tools to make more informed predictions and understand the reliability of each forecast!
