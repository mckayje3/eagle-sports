# College Football Betting Odds Guide

Complete guide to using the betting odds tracking features of the CFB Data Scraper.

## Overview

The odds tracking system captures and stores betting lines from VegasInsider, including:
- **Point Spreads** (opening, current, closing)
- **Moneylines** (home/away odds)
- **Totals/Over-Under** (game total points)
- **Line Movement** (historical tracking of odds changes)

## Database Schema

### game_odds Table

Stores current and historical odds for each game:

```sql
game_id              -- Links to ESPN game
source               -- Odds source (e.g., 'VegasInsider')
opening_spread_home  -- Opening point spread for home team
opening_spread_away  -- Opening point spread for away team
current_spread_home  -- Current/latest spread for home team
current_spread_away  -- Current/latest spread for away team
closing_spread_home  -- Final spread before game starts
closing_spread_away  -- Final spread before game starts
opening_moneyline_home  -- Opening moneyline for home team
opening_moneyline_away  -- Opening moneyline for away team
current_moneyline_home  -- Current moneyline for home team
current_moneyline_away  -- Current moneyline for away team
closing_moneyline_home  -- Closing moneyline for home team
closing_moneyline_away  -- Closing moneyline for away team
opening_total        -- Opening over/under total
current_total        -- Current over/under total
closing_total        -- Closing over/under total
opening_over_odds    -- Odds for over bet (opening)
opening_under_odds   -- Odds for under bet (opening)
current_over_odds    -- Odds for over bet (current)
current_under_odds   -- Odds for under bet (current)
closing_over_odds    -- Odds for over bet (closing)
closing_under_odds   -- Odds for under bet (closing)
```

### odds_movement Table

Tracks line movements over time:

```sql
game_id          -- Links to ESPN game
source           -- Odds source
spread_home      -- Spread at this timestamp
spread_away      -- Spread at this timestamp
moneyline_home   -- Moneyline at this timestamp
moneyline_away   -- Moneyline at this timestamp
total            -- Total at this timestamp
over_odds        -- Over odds at this timestamp
under_odds       -- Under odds at this timestamp
timestamp        -- When odds were captured
```

## Fetching Odds Data

### Manual Updates

Update current odds for upcoming games:
```bash
python scheduler.py odds
```

Update both games and odds together:
```bash
python scheduler.py odds_and_games
```

### Automated Updates

The scheduler automatically updates odds:
- **Tuesday at 2:00 PM** - Mid-week update
- **Thursday at 2:00 PM** - Pre-weekend update
- **Friday at 2:00 PM** - Day before games
- **Saturday at 10:00 AM** - Morning of games

Enable automated updates:
```bash
python scheduler.py schedule
```

### Programmatic Usage

```python
from odds_scraper import VegasInsiderOddsScraper

# Create scraper
scraper = VegasInsiderOddsScraper()

# Scrape current week's odds
scraper.scrape_current_week_odds()

# Save odds to database
odds_data = {
    'game_id': 401234567,
    'source': 'VegasInsider',
    'spread_home': -7.5,
    'spread_away': 7.5,
    'moneyline_home': -300,
    'moneyline_away': 250,
    'total': 54.5
}
scraper.save_odds_to_database(odds_data, game_id)
```

## Querying Odds Data

### Using Query Helper Functions

```python
from query_examples import CFBQuery

query = CFBQuery()

# Get odds for a specific game
odds = query.get_game_odds(game_id=401234567)

# Get all odds for a week
week_odds = query.get_odds_for_week(season=2024, week=12)

# Get odds movement history for a game
movement = query.get_odds_movement_for_game(game_id=401234567)

# Get betting trends for a team
trends = query.get_betting_trends(season=2024, team_name="Alabama")

# Get line movements for a week
movements = query.get_line_movements(season=2024, week=12)

# Get favorites and underdogs
dogs = query.get_dogs_and_favorites(season=2024, week=12)
```

### Direct SQL Queries

#### Get Games with Odds

```sql
SELECT
    g.date,
    t1.name as home_team,
    t2.name as away_team,
    o.current_spread_home,
    o.current_moneyline_home,
    o.current_moneyline_away,
    o.current_total
FROM games g
JOIN teams t1 ON g.home_team_id = t1.team_id
JOIN teams t2 ON g.away_team_id = t2.team_id
LEFT JOIN game_odds o ON g.game_id = o.game_id
WHERE g.season = 2024 AND g.week = 12;
```

#### Track Line Movement

```sql
SELECT
    t1.name as home_team,
    t2.name as away_team,
    o.opening_spread_home,
    o.current_spread_home,
    o.closing_spread_home,
    (o.closing_spread_home - o.opening_spread_home) as line_movement
FROM games g
JOIN teams t1 ON g.home_team_id = t1.team_id
JOIN teams t2 ON g.away_team_id = t2.team_id
JOIN game_odds o ON g.game_id = o.game_id
WHERE g.season = 2024
  AND o.opening_spread_home IS NOT NULL
  AND o.closing_spread_home IS NOT NULL
ORDER BY ABS(line_movement) DESC;
```

#### Calculate Against the Spread (ATS) Record

```sql
SELECT
    t.name,
    COUNT(*) as games,
    SUM(CASE
        WHEN (g.home_team_id = t.team_id AND g.home_score + o.current_spread_home > g.away_score)
          OR (g.away_team_id = t.team_id AND g.away_score + o.current_spread_away > g.home_score)
        THEN 1 ELSE 0
    END) as ats_wins,
    SUM(CASE
        WHEN (g.home_team_id = t.team_id AND g.home_score + o.current_spread_home < g.away_score)
          OR (g.away_team_id = t.team_id AND g.away_score + o.current_spread_away < g.home_score)
        THEN 1 ELSE 0
    END) as ats_losses
FROM teams t
JOIN games g ON (g.home_team_id = t.team_id OR g.away_team_id = t.team_id)
JOIN game_odds o ON g.game_id = o.game_id
WHERE g.season = 2024
  AND g.completed = 1
  AND o.current_spread_home IS NOT NULL
GROUP BY t.team_id
ORDER BY ats_wins DESC;
```

#### Find Best Value Bets (Historical)

```sql
-- Find underdogs that covered the spread
SELECT
    g.date,
    t1.name as home_team,
    t2.name as away_team,
    g.home_score,
    g.away_score,
    o.closing_spread_home,
    CASE
        WHEN o.closing_spread_home > 0 THEN 'Home Underdog Covered'
        WHEN o.closing_spread_home < 0 THEN 'Away Underdog Covered'
    END as result
FROM games g
JOIN teams t1 ON g.home_team_id = t1.team_id
JOIN teams t2 ON g.away_team_id = t2.team_id
JOIN game_odds o ON g.game_id = o.game_id
WHERE g.season = 2024
  AND g.completed = 1
  AND (
    (o.closing_spread_home > 0 AND g.home_score + o.closing_spread_home > g.away_score)
    OR (o.closing_spread_home < 0 AND g.away_score - o.closing_spread_home > g.home_score)
  )
ORDER BY g.date DESC;
```

#### Over/Under Analysis

```sql
SELECT
    t1.name as home_team,
    t2.name as away_team,
    (g.home_score + g.away_score) as actual_total,
    o.closing_total as betting_total,
    CASE
        WHEN (g.home_score + g.away_score) > o.closing_total THEN 'Over'
        WHEN (g.home_score + g.away_score) < o.closing_total THEN 'Under'
        ELSE 'Push'
    END as result,
    ABS((g.home_score + g.away_score) - o.closing_total) as margin
FROM games g
JOIN teams t1 ON g.home_team_id = t1.team_id
JOIN teams t2 ON g.away_team_id = t2.team_id
JOIN game_odds o ON g.game_id = o.game_id
WHERE g.season = 2024
  AND g.completed = 1
  AND o.closing_total IS NOT NULL
ORDER BY margin DESC
LIMIT 20;
```

## Understanding Betting Odds

### Point Spreads

- **Negative (-7.5)**: Team is favored by 7.5 points
- **Positive (+7.5)**: Team is underdog by 7.5 points
- **PK or 0**: Pick'em - no favorite

Example: If Alabama is -7.5 vs Auburn, Alabama must win by more than 7.5 points to cover.

### Moneylines

- **Negative (-150)**: Amount you must bet to win $100
- **Positive (+200)**: Amount you win if you bet $100

Example: -150 means bet $150 to win $100. +200 means bet $100 to win $200.

### Totals (Over/Under)

The combined score of both teams. Bet whether actual total will be over or under the line.

Example: Total is 54.5. If final score is 31-28 (59 total), Over wins.

### Line Movement

When the spread changes from opening to closing:
- **Sharp money**: Professional bettors moving the line
- **Public money**: Recreational bettors
- **Injuries/News**: Team news affecting odds

## Data Sources

### VegasInsider (Primary)

- Current odds for upcoming games
- Opening lines when available
- Multiple sportsbook consensus
- Free access (scraping required)

**Limitations:**
- Requires web scraping
- Historical data limited
- Team name matching needed

### Alternative Sources

If you need more comprehensive historical data, consider:

1. **The Odds API** (Paid)
   - Historical data from June 2020
   - 5-10 minute snapshots
   - JSON API
   - Multiple bookmakers

2. **SportsDataIO** (Paid)
   - 30+ days historical data
   - API access
   - Player props available

## Tips for Using Odds Data

### For Analysis

1. **Track line movement** - Significant moves indicate sharp money
2. **Compare opening vs closing** - See where the smart money went
3. **ATS records** - More predictive than straight wins/losses
4. **Home/away splits** - Teams perform differently ATS at home vs away
5. **Conference games** - Often have different betting patterns

### For Research

1. **Correlate with stats** - Do offensive yards correlate with covering spreads?
2. **Weather impact** - How do totals change with weather?
3. **Rest days** - Do teams off bye weeks perform better ATS?
4. **Time of season** - Early season vs late season betting patterns
5. **Public vs sharp money** - Track which side the public takes

### Best Practices

1. **Update regularly** - Run odds scraper multiple times per week
2. **Track movement** - Save snapshots to see line changes
3. **Validate data** - Check odds against multiple sources when possible
4. **Document methodology** - Keep notes on your analysis methods
5. **Test strategies** - Backtest betting strategies on historical data

## Troubleshooting

### No Odds Data

- VegasInsider may have updated their website structure
- Check `odds_scraper.py` parsing logic
- Verify table class names in HTML
- Consider switching to a paid API

### Team Matching Issues

- VegasInsider team names may differ from ESPN
- Check the `match_team_to_database()` function
- Add manual mapping for problematic teams
- Use abbreviations as fallback

### Missing Historical Odds

- VegasInsider limited historical access
- For comprehensive history, use The Odds API
- Can backfill using archived websites (e.g., Wayback Machine)

## Examples

### Complete Workflow Example

```python
from espn_scraper import ESPNScraper
from odds_scraper import VegasInsiderOddsScraper
from query_examples import CFBQuery

# 1. Fetch games
espn = ESPNScraper()
espn.scrape_week(2024, 12)

# 2. Fetch odds
odds = VegasInsiderOddsScraper()
odds.scrape_current_week_odds()

# 3. Query combined data
query = CFBQuery()
week_data = query.get_odds_for_week(2024, 12)

for game in week_data:
    print(f"{game['away_team']} @ {game['home_team']}")
    print(f"Spread: {game['current_spread_home']}")
    print(f"Total: {game['current_total']}")
    print()
```

### Calculate ROI on Betting Strategy

```python
from query_examples import CFBQuery
import sqlite3

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

# Strategy: Bet on home underdogs
cursor.execute('''
    SELECT COUNT(*) as bets,
           SUM(CASE
               WHEN g.home_score + o.closing_spread_home > g.away_score THEN 1
               ELSE -1.1
           END) as profit_units
    FROM games g
    JOIN game_odds o ON g.game_id = o.game_id
    WHERE g.season = 2024
      AND g.completed = 1
      AND o.closing_spread_home > 0  -- Home team is underdog
''')

result = cursor.fetchone()
bets, profit = result
roi = (profit / bets) * 100 if bets > 0 else 0

print(f"Bets: {bets}")
print(f"Profit: {profit:.2f} units")
print(f"ROI: {roi:.2f}%")
```

## Future Enhancements

Potential improvements to the odds system:

1. **Multiple sportsbooks** - Track odds from different books
2. **Player props** - Individual player betting lines
3. **Live odds** - In-game betting lines
4. **Automated alerts** - Notify on significant line moves
5. **Machine learning** - Predict line movements
6. **API integration** - Direct connection to odds providers

## Resources

- [VegasInsider](https://www.vegasinsider.com/college-football/)
- [The Odds API Documentation](https://the-odds-api.com/liveapi/guides/v4/)
- [Sports Betting Explained](https://www.actionnetwork.com/education/betting-101)
- [Sharp vs Public Money](https://www.actionnetwork.com/education/sharp-money)

---

For questions or issues with odds tracking, check the main README.md or open an issue on GitHub.
