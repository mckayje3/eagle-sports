# Hybrid Odds Tracking System - Complete Guide

## Overview

Your system now uses a **hybrid approach** to track betting odds:

- **VegasInsider** (FREE): Primary source for ongoing 2025 season odds
- **The Odds API** (FREE TIER): 495 requests/month for spot-checking and verification

This gives you the best of both worlds: free continuous tracking + API backup for verification.

---

## What You Have Now

### Database
- **868 ESPN games** (entire 2025 season, Weeks 1-15)
- **232 teams** with full details
- **57 games with Odds API data** (current week)
- **Game matching system** (95% success rate)

### API Quota
- **495 requests remaining** this month (resets monthly on the 1st)
- Each current game fetch costs 1-2 requests
- Historical data requires paid plan ($30/month)

---

## Daily/Weekly Workflow

### Option 1: Manual Updates

**Update odds from VegasInsider** (recommended 3x per week):
```bash
py update_all_odds.py
```

This will:
1. Scrape current odds from VegasInsider
2. Match games to your ESPN database
3. Save odds and line movements
4. Show summary of matched games

**Check your Odds API quota:**
```bash
py update_all_odds.py quota
```

**View current database status:**
```bash
py update_all_odds.py status
```

### Option 2: Automated Schedule

**Start the automated scheduler:**
```bash
py scheduler.py schedule
```

This runs in the background with the following schedule:

**Game Updates:**
- Daily at 9:00 AM: Update yesterday's games
- Saturday at 11:00 PM: Update current week
- Sunday at 9:00 AM: Update recent games
- Monday at 10:00 AM: Update recent games

**Odds Updates (VegasInsider):**
- Tuesday at 2:00 PM: Opening lines
- Thursday at 2:00 PM: Mid-week adjustment
- Friday at 2:00 PM: Late week movement
- Saturday at 10:00 AM: Final pre-game lines

---

## Recommended Weekly Schedule

### Tuesday (Opening Lines)
```bash
py update_all_odds.py
```
- Captures opening lines for the week
- Records baseline for line movement tracking

### Friday (Closing Lines)
```bash
py update_all_odds.py
```
- Captures final lines before games
- Shows how lines moved during the week

### Sunday (Verify Results)
```bash
py scheduler.py current
```
- Updates final scores from weekend games
- Calculates against-the-spread (ATS) results

---

## Using Your Odds API Requests Wisely

You have **495 free requests/month**. Here's how to use them strategically:

### Strategy 1: Weekly Verification (Recommended)
Use 10-15 requests per week to verify VegasInsider data:

```bash
# Fetch current week from Odds API
py test_odds_api.py
```

This costs ~2 requests and gives you:
- Consensus odds from multiple sportsbooks
- Verification of VegasInsider accuracy
- Additional data points for big games

**Monthly cost:** ~40 requests (8 requests/week × 5 weeks)
**Leaves:** ~455 requests as buffer

### Strategy 2: Spot-Check Big Games
Save API calls for rivalry games, ranked matchups, or unusual line movements.

### Strategy 3: End-of-Month Bulk Check
Use remaining requests at month-end to capture snapshot of entire upcoming slate.

---

## Quick Reference Commands

### Odds Updates
```bash
# Update from VegasInsider (free, use often)
py update_all_odds.py

# Check Odds API quota
py update_all_odds.py quota

# View database status
py update_all_odds.py status
```

### Game Updates
```bash
# Update current week
py scheduler.py current

# Update last 7 days
py scheduler.py recent

# Update yesterday
py scheduler.py yesterday

# Update entire 2025 season
py scheduler.py full
```

### Combined Updates
```bash
# Update both games and odds
py scheduler.py odds_and_games
```

### Start Automated Schedule
```bash
# Run continuous automated updates
py scheduler.py schedule
```

---

## Checking Your Data

### View Saved Odds
```bash
py check_saved_odds.py
```

Shows:
- Total odds records
- Sample odds with game info
- Line movements tracked

### View Backfill Progress
```bash
py check_backfill_progress.py
```

Shows:
- Games by week
- Total games in database
- Team count

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     WEEKLY CYCLE                             │
└─────────────────────────────────────────────────────────────┘

Tuesday (Opening Lines):
  VegasInsider → parse_vegasinsider.py → parsed_games.json
                → save_odds_with_matching.py → Database
                → Records: opening_spread, opening_total, opening_ML

Friday (Closing Lines):
  VegasInsider → parse_vegasinsider.py → parsed_games.json
                → save_odds_with_matching.py → Database
                → Records: closing_spread, closing_total, closing_ML
                → Tracks: line movement from Tuesday

Saturday (Games Played):
  ESPN → espn_scraper.py → Database
        → Records: final scores

Sunday (Verification):
  Odds API → test_odds_api.py → odds_api_current_games.json
            → save_odds_with_matching.py → Database
            → Cross-reference: VegasInsider vs Odds API
            → Calculate: ATS results

┌─────────────────────────────────────────────────────────────┐
│                   DATABASE RESULT                            │
└─────────────────────────────────────────────────────────────┘

games table:
  - game_id, home_team, away_team, date, scores

game_odds table:
  - game_id (FK)
  - opening_spread, current_spread, closing_spread
  - opening_total, current_total, closing_total
  - source (VegasInsider or TheOddsAPI)

odds_movement table:
  - game_id (FK)
  - spread, total, timestamp
  - Tracks every scrape for line movement history
```

---

## Troubleshooting

### VegasInsider Returns No Games
- Website structure may have changed
- Check `vegasinsider_raw.html` for raw data
- VegasInsider only shows upcoming games (removes completed ones)

### Low Match Rate (<80%)
- Add more team name mappings to `game_matcher.py`
- VegasInsider may show games from next week
- Some FCS opponents won't match (by design)

### Odds API Quota Exhausted
- Quota resets on 1st of each month
- Rely on VegasInsider until reset
- Consider paid plan ($30/month) if needed

### Database Errors
- Check database exists: `py database.py`
- Verify schema is up to date
- Check file permissions

---

## Future Enhancements

### If You Decide to Upgrade Odds API ($30/month)

**Benefits:**
- 20,000 requests/month
- Access to historical odds data back to June 2020
- Can backfill entire 2020-2024 seasons

**Cost Analysis:**
- Historical backfill: ~1,000 dates × 20 requests = 20,000 requests
- Use 1 month to backfill everything
- Then downgrade back to free tier or continue for current season

**Backfill Strategy:**
```bash
# If you upgrade, run this to backfill all historical data
py backfill_historical_odds.py
```

---

## Best Practices

### For Best Data Quality

1. **Run VegasInsider** scraper 2-3 times per week minimum
2. **Capture opening lines** early in the week (Tuesday/Wednesday)
3. **Capture closing lines** right before games (Saturday morning)
4. **Update scores** after games complete (Sunday morning)
5. **Use Odds API** to verify on big games or unusual movements

### For API Quota Management

1. **Don't** fetch current odds from Odds API daily (wastes requests)
2. **Do** use VegasInsider for regular updates (free unlimited)
3. **Do** save API requests for verification and spot-checks
4. **Do** check quota regularly: `py update_all_odds.py quota`

### For Historical Analysis

1. **Build database** going forward from now
2. **By end of 2025 season**, you'll have complete odds + results data
3. **Can backfill** 2024 season from ESPN game data (already available)
4. **Historical odds** require paid API or manual collection

---

## Summary

**Your Hybrid System:**

✅ **Free continuous odds tracking** via VegasInsider
✅ **868 ESPN games** from 2025 season
✅ **495 free API requests** for verification
✅ **95% match rate** linking odds to games
✅ **Line movement tracking** with timestamps
✅ **Automated scheduling** available

**Next Steps:**

1. Run `py update_all_odds.py` 2-3 times per week
2. Or start automated scheduler: `py scheduler.py schedule`
3. Monitor API quota: `py update_all_odds.py quota`
4. Build your historical database going forward
5. Decide on paid API upgrade at end of month if needed

**You now have a complete, working system for tracking college football games and betting odds!**
