# Dashboard Update - Score Predictions Integration

The dashboard's **View Predictions** page has been updated to display comprehensive score predictions from the deep learning model.

## What's New

### Enhanced View Predictions Page

The View Predictions page now shows:

✅ **Predicted Actual Scores** - Both home and away team scores
✅ **Point Spread** - Predicted margin of victory
✅ **Total Points (O/U)** - Combined score prediction
✅ **Win Probability** - Confidence percentage
✅ **School Names** - Full team names (e.g., "Oregon Ducks" instead of just "Ducks")
✅ **Summary Metrics** - Quick stats on average totals, spreads, close games, and blowouts

## How to Use

### 1. Generate Predictions

First, run the score predictor to generate predictions:

```bash
py predict_scores.py
```

This creates **predicted_scores.csv** with 122 game predictions.

### 2. Launch Dashboard

```bash
streamlit run dashboard.py
```

### 3. View Predictions

Navigate to the **View Predictions** page from the sidebar.

## Display Format

Each game prediction now shows:

### Column 1: Predicted Score
```
27 - 24
Akron Zips
Bowling Green Falcons
Confidence: 62.4%
```

### Column 2: Spread & Total
```
Spread: Bowling Green Falcons -3.6
Total: 50.7 (O/U)
```

### Column 3: Result
```
Pending
(or actual result once game is played)
```

## Summary Metrics

At the top of the page, you'll see:

- **Avg Total Points** - Average combined score across all games
- **Avg Spread** - Average point differential
- **Close Games (<7)** - Number of games predicted within 7 points
- **Blowouts (>21)** - Number of games predicted with >21 point margin

## Filters

You can filter predictions by:
- **Status**: All, Pending, Completed
- **Week**: All or specific week number
- **Confidence**: High (>75%), Medium (65-75%), Low (<65%)

## Data Source Priority

The dashboard loads predictions in this order:
1. **predicted_scores.csv** (from `predict_scores.py`) ← **NEW**
2. enhanced_predictions_week_13.csv (legacy)
3. predictions_log.csv (fallback)

## Example View

```
📊 Showing 122 predictions from Deep Learning Score Predictor (PyTorch)

┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Avg Total   │ Avg Spread  │ Close Games │ Blowouts    │
│ 54.9        │ 12.7 pts    │ 40          │ 27          │
└─────────────┴─────────────┴─────────────┴─────────────┘

Week 13: Tennessee Volunteers @ Florida Gators
├─ Predicted Score: 36 - 25
├─ Winner: Tennessee Volunteers (83.0%)
├─ Spread: Florida Gators -11.4
└─ Total: 60.5 (O/U)

Week 13: Pittsburgh Panthers @ Georgia Tech Yellow Jackets
├─ Predicted Score: 30 - 32
├─ Winner: Georgia Tech Yellow Jackets (52.4%)
├─ Spread: Georgia Tech Yellow Jackets +1.7
└─ Total: 62.6 (O/U)
```

## Integration with Existing Features

The dashboard still supports:
- All existing prediction viewing features
- Result tracking (when games complete)
- Filtering and sorting
- Refresh functionality

## Technical Details

### Updated Functions

1. **load_predictions()** - Now tries `predicted_scores.csv` first
2. **show_view_predictions()** - Enhanced display with scores, spreads, totals
3. Added summary metrics section
4. Improved formatting with school names

### New Columns Displayed

- `predicted_away_score` - Away team predicted score
- `predicted_home_score` - Home team predicted score
- `predicted_spread` - Point differential (+ = home favored)
- `predicted_total` - Combined points (over/under)
- `win_probability` - Confidence percentage

## Troubleshooting

### No predictions showing?

Run the score predictor:
```bash
py predict_scores.py
```

### Dashboard won't start?

Check Streamlit is installed:
```bash
pip install streamlit
```

### Data looks wrong?

Refresh the dashboard:
1. Click the "🔄 Refresh Data" button
2. Or restart the dashboard

## Future Enhancements

Potential additions:
- Live score tracking
- Prediction accuracy metrics
- Comparison with Vegas lines
- Historical prediction trends
- Export filtered predictions
