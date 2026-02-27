

# Hybrid Odds System Setup Guide

Complete setup instructions for using both The Odds API (historical) and VegasInsider (current) to track college football betting odds.

## Overview

**The Strategy:**
- **The Odds API**: Backfill historical odds data (2020-2024)
- **VegasInsider**: Track current 2025 season odds (free scraping)
- **Combined**: Build comprehensive odds database

## Part 1: The Odds API Setup (Historical Data)

### Step 1: Get Free API Key

1. Go to [https://the-odds-api.com/](https://the-odds-api.com/)
2. Click "Get Started" or "Sign Up"
3. Create a free account
4. Copy your API key from the dashboard

**Free Tier Details:**
- 500 requests/month
- Current odds: 1 request per fetch
- Historical odds: 10 requests per market per fetch
- Data from June 2020 onwards

### Step 2: Save API Key

Create a config file to store your API key securely:

```bash
# Create config file
echo {"api_key": "YOUR_KEY_HERE"} > odds_api_config.json
```

Or manually create `odds_api_config.json`:
```json
{
  "api_key": "your_actual_api_key_here"
}
```

**Important**: Add to `.gitignore` to keep your key private!

### Step 3: Test The Odds API

Run the test script:

```bash
py test_odds_api.py
```

Or provide key directly:
```bash
py test_odds_api.py YOUR_API_KEY
```

This will:
- ✅ Verify your API connection
- ✅ Show remaining requests
- ✅ Fetch current NCAAF odds (sample)
- ✅ Optionally test historical fetch (costs 20 requests)

### Step 4: Fetch Historical Data

Once tested, fetch historical data strategically:

```python
from odds_api_scraper import OddsAPIScraper
import json

# Load API key
with open('odds_api_config.json', 'r') as f:
    config = json.load(f)

scraper = OddsAPIScraper(api_key=config['api_key'])

# Fetch specific game days from 2024 season
# Each fetch costs 10 requests per market

# Week 1 of 2024 season
games_week1 = scraper.fetch_historical_odds(
    date='2024-08-31T12:00:00Z',
    markets=['spreads', 'totals']
)

# Save for later processing
with open('historical_2024_week1.json', 'w') as f:
    json.dump(games_week1, f, indent=2)
```

**Budget Your Requests:**
- 500 free requests/month
- Historical = 10 requests per market
- With 2 markets (spreads, totals) = 20 requests per date
- 500 ÷ 20 = **~25 dates per month**
- Strategy: Fetch one weekend per week of past season

### Step 5: Process Historical Data

After fetching, match teams to your ESPN database:

```python
# Parse and save to database
for game in games_week1:
    parsed = scraper.parse_odds_data(game)
    consensus = scraper.get_consensus_odds(parsed)

    # Match to ESPN game_id (manual matching required)
    # Then save to database
    # scraper.save_odds_to_database(consensus, espn_game_id)
```

## Part 2: VegasInsider Setup (Current Season)

### Step 1: Install Dependencies

Already done! But verify:

```bash
py -m pip install requests beautifulsoup4 pandas lxml schedule
```

### Step 2: Test VegasInsider Scraper

```bash
py parse_vegasinsider.py
```

This will:
- ✅ Scrape current week's games
- ✅ Get opening and current lines
- ✅ Calculate line movements
- ✅ Save to `parsed_games.json`

### Step 3: Review Current Data

Check the output:
- `vegasinsider_table_1.csv` - Raw scraped data
- `parsed_games.json` - Processed odds data

Look for:
- ✅ Team names
- ✅ Opening spreads/totals/moneylines
- ✅ Current (consensus) lines
- ✅ Line movements

### Step 4: Set Up Automated Scraping

Add to your scheduler to run multiple times per week:

```python
# In scheduler.py (already configured!)

# Tuesday at 2 PM - Opening lines
schedule.every().tuesday.at("14:00").do(scraper.update_current_odds)

# Friday at 2 PM - Mid-week lines
schedule.every().friday.at("14:00").do(scraper.update_current_odds)

# Saturday at 10 AM - Closing lines
schedule.every().saturday.at("10:00").do(scraper.update_current_odds)
```

Run scheduler:
```bash
py scheduler.py schedule
```

### Step 5: Manual Updates During Season

For current week's games:
```bash
py scheduler.py odds
```

For games + odds together:
```bash
py scheduler.py odds_and_games
```

## Part 3: Hybrid Workflow

### Weekly Routine (During Season)

**Tuesday:**
- VegasInsider scrapes opening lines
- Save to database with timestamp

**Friday:**
- VegasInsider scrapes again (mid-week lines)
- Track movement from Tuesday

**Saturday Morning:**
- VegasInsider scrapes final time (closing lines)
- Track final movement

**After Games Complete:**
- ESPN scraper gets final scores
- Calculate against-the-spread results

### Monthly Historical Backfill

**With 500 free Odds API requests:**
- Pick 25 key dates from past seasons
- Fetch Saturday game days (most games)
- Prioritize:
  - Week 1 (season opener)
  - Rivalry weeks
  - Conference championship games
  - Playoff games
  - Key ranked matchups

**Example dates for 2024:**
```python
key_dates_2024 = [
    '2024-08-31T12:00:00Z',  # Week 1
    '2024-09-07T12:00:00Z',  # Week 2
    '2024-09-14T12:00:00Z',  # Week 3
    # ... continue for important weeks
    '2024-11-30T12:00:00Z',  # Rivalry week
    '2024-12-07T12:00:00Z',  # Championship week
]

for date in key_dates_2024:
    games = scraper.fetch_historical_odds(date=date)
    # Process and save
    time.sleep(1)  # Be nice to the API
```

## Part 4: Data Quality Checks

### Verify The Odds API Data

Check JSON files for:
```python
import json

with open('odds_api_current_games.json', 'r') as f:
    games = json.load(f)

print(f"Total games: {len(games)}")
print(f"Sample game: {json.dumps(games[0], indent=2)}")

# Check data completeness
for game in games:
    assert 'home_team' in game
    assert 'away_team' in game
    assert 'bookmakers' in game
    print(f"✓ {game['away_team']} @ {game['home_team']}")
```

### Verify VegasInsider Data

Check parsed data:
```python
import json

with open('parsed_games.json', 'r') as f:
    games = json.load(f)

print(f"Total games: {len(games)}")

# Check for line movements
movements = [g for g in games
             if g.get('opening_total') != g.get('current_total')]

print(f"Games with line movement: {len(movements)}")
```

## Part 5: Database Integration

### Match Teams Between Sources

Create team mapping:
```python
team_mapping = {
    # The Odds API name -> VegasInsider abbreviation
    'Ohio Bobcats': 'OHIO',
    'Miami (OH) RedHawks': 'MIA-OH',
    'Florida State Seminoles': 'FSU',
    # ... etc
}
```

### Save to Database

```python
from database import FootballDatabase

db = FootballDatabase()
db.connect()

# For each game with matched ESPN game_id
odds_data = {
    'game_id': espn_game_id,
    'source': 'TheOddsAPI',  # or 'VegasInsider'
    'opening_spread_home': -7.5,
    'current_spread_home': -9.5,
    'closing_spread_home': -10.0,
    'opening_total': 54.5,
    'current_total': 55.5,
    'closing_total': 56.0,
    # ... etc
}

db.insert_or_update_odds(odds_data)
```

## Part 6: Cost Analysis

### The Odds API Free Tier (500 requests)

**Option A: All Historical**
- 500 requests ÷ 20 (per date) = 25 dates
- Cover ~2 full seasons sampling key games

**Option B: Mixed Use**
- 300 requests historical (15 dates)
- 200 requests current season monitoring
- Balance between backfill and current

**Option C: Current Only**
- All 500 for current season
- 3x per week = 12 per month = 48 in 4 months
- 500 requests = 41 weeks of monitoring

### Paid Tier ($30/month = 20,000 requests)

If you decide to upgrade:
- 20,000 ÷ 20 = **1,000 dates of historical data**
- Can backfill **entire 2020-2024** seasons
- Plus ongoing current season monitoring
- Plenty of headroom

## Part 7: Troubleshooting

### The Odds API Issues

**"Unauthorized" error:**
- Check API key is correct
- Verify not expired
- Check account status at the-odds-api.com

**No games returned:**
- Check date format (ISO 8601)
- Verify sport key: `americanfootball_ncaaf`
- Try different date (may be off-season)

**Rate limit exceeded:**
- Check remaining requests
- Wait for monthly reset (1st of month)
- Consider upgrading to paid tier

### VegasInsider Issues

**No data scraped:**
- Website structure may have changed
- Check `vegasinsider_raw.html` file
- Update CSS selectors in `odds_scraper.py`

**Team names don't match:**
- Create team mapping dictionary
- Update `match_team_to_database()` function
- Use abbreviations for better matching

**Historical data missing:**
- VegasInsider doesn't archive completed games
- Use The Odds API for past data
- Start scraping BEFORE games are played

## Part 8: Next Steps

### Immediate Actions

1. **Sign up for The Odds API** (free)
2. **Run test_odds_api.py** to verify connection
3. **Fetch sample historical data** (1-2 dates)
4. **Review data quality** in JSON files
5. **Run VegasInsider scraper** for current week

### Short Term (This Week)

1. **Set up automated VegasInsider** scraping
2. **Capture this weekend's** opening/closing lines
3. **Test database** storage of odds
4. **Compare data quality** between sources

### Long Term (This Season)

1. **Run VegasInsider** 3x per week all season
2. **Build historical dataset** going forward
3. **Strategically use** Odds API for key historical games
4. **Analyze betting trends** and line movements

### Decision Point (End of Free Trial)

After using both for a month:

**If The Odds API data is great:**
- Consider $30/month upgrade
- Backfill all 2020-2024 data
- Use VegasInsider as backup

**If VegasInsider is sufficient:**
- Continue with free scraping
- Build your own historical database
- Save The Odds API for spot checks

**If you want both:**
- The Odds API for historical accuracy
- VegasInsider for real-time current data
- Best of both worlds

## Resources

- [The Odds API Docs](https://the-odds-api.com/liveapi/guides/v4/)
- [VegasInsider College Football](https://www.vegasinsider.com/college-football/)
- [ODDS_GUIDE.md](./ODDS_GUIDE.md) - Detailed betting odds documentation
- [README.md](./README.md) - Main project documentation

## Support

Questions? Issues?
- Check logs: `cfb_scraper.log`
- Review JSON output files
- Test individual components
- Adjust as needed for your use case

---

**Remember:** Start small, test quality, scale up as needed. You have 500 free requests - use them wisely to evaluate before committing to paid tier!
