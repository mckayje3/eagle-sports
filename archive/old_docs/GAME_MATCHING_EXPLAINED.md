

# Game Matching System - How It Works

## The Problem

**Different odds sources use different team names and don't provide ESPN game IDs.**

### ESPN Data (Source of Truth)
```
game_id: 401628404
home_team_id: 195 → "Ohio Bobcats"
away_team_id: 113 → "UMass Minutemen"
date: "2025-11-16T12:00:00Z"
```

### VegasInsider Data (What We Scrape)
```
home_team: "OHIO"
away_team: "UMass"
opening_spread: "OHIO -29.5"
current_spread: "OHIO -32.5"
NO ESPN GAME_ID! ❌
```

### The Odds API Data
```
id: "abc123xyz" (Their own ID system)
home_team: "Ohio Bobcats"
away_team: "Massachusetts Minutemen"
commence_time: "2025-11-16T17:00:00Z"
NO ESPN GAME_ID! ❌
```

**Problem:** Our database requires `game_id` (ESPN's ID) as the foreign key, but odds sources don't provide it!

---

## The Solution: Game Matching

The `game_matcher.py` system matches games using:
1. **Team Name Mapping** - Dictionary of abbreviations to full names
2. **Fuzzy String Matching** - Algorithm to find similar names
3. **Date + Context Matching** - Match by game date and season

---

## Complete Workflow

### Step 1: ESPN Scraper (Fills Core Data)

```bash
py scheduler.py current  # Or any ESPN scraping command
```

**What happens:**
```python
# ESPN API returns
{
  "game_id": 401628404,
  "home_team": {"id": 195, "name": "Ohio", "displayName": "Ohio Bobcats"},
  "away_team": {"id": 113, "name": "UMass", "displayName": "UMass Minutemen"},
  "date": "2025-11-16T17:00:00Z"
}

# Saves to database:
INSERT INTO teams (team_id=195, name="Ohio", display_name="Ohio Bobcats")
INSERT INTO teams (team_id=113, name="UMass", display_name="UMass Minutemen")
INSERT INTO games (game_id=401628404, home_team_id=195, away_team_id=113, date="2025-11-16...")
```

**Result:**
- ✅ `games` table has game with `game_id = 401628404`
- ✅ `teams` table has Ohio (195) and UMass (113)

---

### Step 2: VegasInsider Scraper (Gets Odds)

```bash
py parse_vegasinsider.py
```

**What happens:**
```python
# VegasInsider HTML contains
{
  "home_team": "OHIO",
  "away_team": "UMass",
  "opening_spread": "OHIO -29.5",
  "current_spread": "OHIO -32.5",
  "opening_total": "o51.5"
}

# Saves to JSON (NOT database yet!)
# File: parsed_games.json
```

**Result:**
- ✅ Odds data scraped
- ❌ NOT in database yet (no game_id to link it!)

---

### Step 3: Game Matching (Links Them Together)

```bash
py save_odds_with_matching.py vegasinsider
```

**What happens:**
```python
# 1. Load parsed VegasInsider data
vi_game = {
  "home_team": "OHIO",
  "away_team": "UMass"
}

# 2. Initialize matcher
matcher = GameMatcher()

# 3. Match teams to ESPN IDs
home_id = matcher.lookup_team_id("OHIO")
# → Checks mapping: "OHIO" → "Ohio"
# → Queries database: WHERE name = "Ohio"
# → Returns: team_id = 195 ✅

away_id = matcher.lookup_team_id("UMass")
# → Direct match in database
# → Returns: team_id = 113 ✅

# 4. Find the game
game_id = matcher.find_game_by_teams_and_date(
    home_team="OHIO",    # Resolves to team_id 195
    away_team="UMass",   # Resolves to team_id 113
    season=2025
)
# → Queries: SELECT game_id FROM games
#            WHERE home_team_id=195 AND away_team_id=113 AND season=2025
# → Returns: game_id = 401628404 ✅

# 5. Save odds with the matched game_id
odds_data = {
    'game_id': 401628404,  # ← NOW WE HAVE IT!
    'source': 'VegasInsider',
    'opening_spread_home': -29.5,
    'current_spread_home': -32.5,
    'opening_total': 51.5,
    # ...
}

db.insert_or_update_odds(odds_data)
```

**Result:**
- ✅ Odds saved with correct `game_id`
- ✅ Can now query: `SELECT * FROM game_odds WHERE game_id = 401628404`

---

## Matching Algorithms

### 1. Direct Mapping (Fastest)

```python
TEAM_NAME_MAPPING = {
    'OHIO': 'Ohio',
    'OSU': 'Ohio State',
    'UMass': 'UMass',
    'BGSU': 'Bowling Green',
    # 200+ mappings...
}

# VegasInsider says "OHIO" → Mapper says "Ohio" → Database finds team_id 195
```

### 2. Fuzzy String Matching (Fallback)

```python
# If direct mapping fails, compare strings
input: "Massachusetts"
database: ["UMass Minutemen", "UMass", "Massachusetts Minutemen"]

# Calculate similarity score
SequenceMatcher("Massachusetts", "UMass Minutemen").ratio() = 0.65
SequenceMatcher("Massachusetts", "Massachusetts Minutemen").ratio() = 0.95 ✅

# Pick best match if > 80% confidence
```

### 3. Date + Team Context

```python
# Multiple games between same teams? Use date to disambiguate
find_game_by_teams_and_date(
    home_team="Alabama",
    away_team="Auburn",
    game_date="2025-11-30",  # Iron Bowl always last Saturday in November
    season=2025
)

# Searches date range: 2025-11-27 to 2025-12-03
# Finds the correct game even if exact date is slightly off
```

---

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        ESPN SCRAPER                              │
│                                                                   │
│  ESPN API → games table (game_id, home_team_id, away_team_id)   │
│          → teams table (team_id, name, display_name)            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ (Database has games and teams)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VEGASINSIDER SCRAPER                           │
│                                                                   │
│  Website → parsed_games.json (teams="OHIO" vs "UMass")          │
│                              (NO game_id yet!)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ (JSON file with odds, no game_id)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GAME MATCHER                                │
│                                                                   │
│  1. Read: parsed_games.json                                      │
│  2. For each game:                                               │
│     a) Lookup: "OHIO" → team_id 195                             │
│     b) Lookup: "UMass" → team_id 113                            │
│     c) Query: Find game with these team_ids                      │
│     d) Returns: game_id = 401628404                              │
│  3. Combine: odds data + game_id                                 │
│  4. Save: INSERT INTO game_odds (game_id=401628404, ...)         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ (Odds now linked to games!)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATABASE                                 │
│                                                                   │
│  games          (game_id, home_team_id, away_team_id)           │
│      ↓                                                            │
│  game_odds      (game_id, spreads, totals, moneylines)          │
│                                                                   │
│  ✅ FULLY LINKED - Can query both together!                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Example Query After Matching

Once odds are matched and saved, you can query everything together:

```sql
SELECT
    g.game_id,
    g.date,
    home.name as home_team,
    away.name as away_team,
    g.home_score,
    g.away_score,
    o.opening_spread_home,
    o.closing_spread_home,
    o.opening_total,
    o.closing_total
FROM games g
JOIN teams home ON g.home_team_id = home.team_id
JOIN teams away ON g.away_team_id = away.team_id
LEFT JOIN game_odds o ON g.game_id = o.game_id
WHERE g.season = 2025 AND g.week = 12
ORDER BY g.date;
```

**Result:**
```
game_id     | date       | home_team  | away_team | home_score | away_score | opening_spread | closing_spread | opening_total | closing_total
401628404   | 2025-11-16 | Ohio       | UMass     | NULL       | NULL       | -29.5          | -32.5          | 51.5          | 51.5
401628405   | 2025-11-16 | Ohio State | Rutgers   | NULL       | NULL       | -32.5          | -30.5          | 56.5          | 55.5
...
```

---

## Handling Match Failures

### Why Games Might Not Match

1. **Team not in database yet**
   - ESPN scraper hasn't run for that team
   - Solution: Run `py scheduler.py current` first

2. **Team name mapping missing**
   - VegasInsider uses abbreviation not in mapping
   - Solution: Add to `TEAM_NAME_MAPPING` in `game_matcher.py`

3. **Game not in database**
   - Future game ESPN hasn't posted yet
   - Solution: Wait for ESPN to post schedule, or add game manually

4. **FCS/Non-FBS opponent**
   - Small schools may not be in ESPN database
   - Solution: These won't match (by design - we focus on FBS)

### Improving Match Rate

**Add missing teams to mapping:**
```python
# In game_matcher.py, add to TEAM_NAME_MAPPING
TEAM_NAME_MAPPING = {
    # ... existing mappings ...
    'NEW_ABBREV': 'Full Team Name',
}
```

**Check fuzzy matching threshold:**
```python
# In game_matcher.py, adjust confidence threshold
if best_score > 0.8:  # Lower to 0.7 for more lenient matching
    return best_match
```

**Manual override:**
```python
# For specific problematic games
game_id = matcher.find_game_by_teams_and_date(
    home_team='Tricky Team Name',
    away_team='Other Team',
    season=2025,
    week=12  # Add week to narrow it down
)
```

---

## Complete Usage Example

### Full Workflow from Scratch

```bash
# 1. Initialize database
py database.py

# 2. Fetch games from ESPN (gets game_ids and team info)
py scheduler.py current

# 3. Scrape odds from VegasInsider
py parse_vegasinsider.py

# 4. Match and save odds to database
py save_odds_with_matching.py vegasinsider

# 5. Query the combined data
py query_examples.py  # Use any query functions
```

### With The Odds API

```bash
# 1. Get API key and test
py test_odds_api.py

# 2. Data is automatically saved to odds_api_current_games.json

# 3. Match and save
py save_odds_with_matching.py oddsapi

# 4. Query as normal
```

---

## Troubleshooting

### "Could not find team IDs"

**Problem:** Matcher can't resolve team name

**Solution:**
1. Check if team is in database: `SELECT * FROM teams WHERE name LIKE '%Ohio%'`
2. Add mapping: Update `TEAM_NAME_MAPPING` in `game_matcher.py`
3. Run ESPN scraper if team is missing entirely

### "No match found"

**Problem:** Game not in database

**Solution:**
1. Verify ESPN has the game: Run `py scheduler.py current`
2. Check season/week: Make sure you're looking at right timeframe
3. Verify team IDs are correct: Use manual lookup

### "✓ Matched X/120 games"

**Good match rate:** 100-115 out of 120 (83-96%)
- Some games won't match (FCS opponents, future games not posted yet)

**Poor match rate:** <80 out of 120 (<67%)
- Need to add more team name mappings
- ESPN scraper may not have run yet
- Check for errors in team name normalization

---

## Benefits of This System

✅ **Flexible**: Works with multiple odds sources
✅ **Robust**: Multiple matching strategies (direct, fuzzy, date-based)
✅ **Expandable**: Easy to add new team name mappings
✅ **Maintains Integrity**: All odds properly linked via foreign keys
✅ **Historical Tracking**: Can track same game from multiple sources
✅ **No Duplication**: Unique constraints prevent duplicate odds

---

## Future Enhancements

### Auto-Learning Team Names

```python
# After manual confirmation, save new mappings
def add_team_mapping(abbrev, full_name):
    # Save to persistent mapping file
    mappings = load_mappings()
    mappings[abbrev] = full_name
    save_mappings(mappings)
```

### Confidence Scores

```python
# Return match confidence with game_id
def find_game_with_confidence(home, away, date):
    game_id, confidence = matcher.match(...)
    if confidence < 0.9:
        warn_user("Low confidence match")
    return game_id
```

### Interactive Matching

```python
# When match fails, prompt user
if not game_id:
    print(f"Could not match: {away} @ {home}")
    print("Possible matches:")
    # Show similar games
    user_choice = input("Select game_id: ")
    game_id = int(user_choice)
```

---

## Summary

**The game matching system solves the core problem:**

VegasInsider/Odds API → [Game Matcher] → ESPN game_id → Database

Without matching: ❌ Orphaned odds data, no queries work
With matching: ✅ Fully integrated database, rich analytics possible

The system is **already built and working** - you just need to:
1. Run ESPN scraper first (populate games/teams)
2. Run odds scrapers second (get odds data)
3. Run matching script third (link them together)
4. Query everything combined!
