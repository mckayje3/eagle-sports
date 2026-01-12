# Eagle Eye Sports Dashboard

Multi-sport prediction dashboard for NFL, CFB, NBA, and CBB.

## Quick Launch

**Double-click:** `launch_dashboard.bat`

Or from command line:
```bash
streamlit run streamlit_app.py
```

Dashboard opens at http://localhost:8501

---

## Sports Pages

### NFL Predictions
- **Week selector**: Weeks 1-18 + Wild Card, Divisional, Conference, Super Bowl
- Regular season: Pulls from database
- Playoffs: Pulls from `nfl_playoff_predictions.csv`
- Games sorted by edge magnitude (best picks first)

### CFB Predictions
- Week selector (1-20)
- Includes bowl games and playoffs
- Same 3-row card format as NFL

### NBA Predictions
- **Date picker**: 7-day calendar (3 days back, today, 3 days forward)
- Daily games from `nba_games.db`
- Higher totals thresholds (NBA scores more)

### CBB Predictions
- Same date picker as NBA
- College basketball from `cbb_games.db`

---

## Game Card Format

All sports use the same 3-row format:

```
Row 1: Away @ Home          | Date/Time     | ✓ (if completed)
Row 2: Spread | Vegas: +X.X | Model: +X.X | Edge: +X.X → Pick | ⭐⭐⭐
Row 3: Total  | Vegas: X.X  | Model: X.X  | Edge: +X.X → OVER/UNDER | ⭐⭐
```

### Confidence Stars

| Spread Edge | Stars | Total Edge | Stars |
|-------------|-------|------------|-------|
| >= 5 pts | ⭐⭐⭐ | >= 8 pts | ⭐⭐⭐ |
| >= 3 pts | ⭐⭐ | >= 5 pts | ⭐⭐ |
| < 3 pts | ⭐ | < 5 pts | ⭐ |

---

## Other Features

### System Health Banner
Top of page shows system status:
- **Green**: All systems operational
- **Yellow**: Stale data or missing predictions
- **Red**: Database errors or failed updates

### Betting Tracker
Track your bets and performance:
- Log bets with stake, odds, result
- View win/loss record
- Calculate ROI

### News Feed
RSS aggregation from ESPN, CBS Sports, Yahoo Sports.

### Database Explorer
Browse raw data from all databases.

---

## Updating Predictions

### Manual Update (from dashboard)
Click "Update Predictions" button on any sport page.

### Command Line
```bash
python predict_nfl_playoffs.py  # NFL playoffs
python nba_predictor.py         # NBA
python cbb_predictor.py         # CBB
python cfb_predictor.py         # CFB
```

### Automatic (daily at 9am)
The scheduled task runs `daily_update.py` which:
1. Updates game results
2. Fetches latest odds
3. Regenerates predictions
4. Pushes to cloud

---

## Cloud vs Local

Both apps read from the same data:
- **Local**: Reads directly from local files
- **Cloud**: Reads from GitHub repo

To sync: `python push_databases.py`

The daily update auto-syncs at 9am.

---

## Troubleshooting

### Predictions not showing
- Check the date/week selector
- Run the predictor script for that sport
- Refresh the page

### Cloud app outdated
- Run `python push_databases.py`
- Wait 2-3 minutes for Streamlit Cloud to redeploy

### Database errors
- Check that `*_games.db` files exist
- Run `python daily_update.py` to refresh

### Stale predictions warning
- Run "Update Predictions" from dashboard
- Or run predictor script from command line

---

## Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main dashboard application |
| `system_health.py` | Health monitoring logic |
| `betting_tracker.py` | Bet tracking module |
| `timezone_utils.py` | Eastern timezone handling |
| `launch_dashboard.bat` | Quick launch script |
