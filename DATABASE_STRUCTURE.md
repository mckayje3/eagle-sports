# Database Structure Documentation

## Overview
You have 6 SQLite databases for tracking sports predictions and game data:

1. **cfb_games.db** - College Football game data
2. **nfl_games.db** - NFL game data
3. **nba_games.db** - NBA game data
4. **cbb_games.db** - College Basketball game data
5. **nhl_games.db** - NHL game data
6. **users.db** - User accounts and predictions cache

All sport databases share a common schema with `game_date_eastern` column for consistent date filtering.

---

## 1. CFB_GAMES.DB (College Football)

### Tables Summary
- **games**: 1,830 rows - Core game data (2024 + 2025)
- **teams**: 240 rows - Team information
- **game_odds**: 1,506 rows - Betting lines from various sources
- **odds_movement**: 188 rows - Historical odds changes
- **team_game_stats**: 3,420 rows - Team performance statistics
- **drives**: 29,184 rows - **NEW: Drive-by-drive data** (397 games 2024, 412 games 2025)
- **player_game_stats**: 0 rows - Player statistics (not populated)

### GAMES Table
Primary table for all CFB games.

**Key Columns:**
- `game_id` (INTEGER, PRIMARY KEY) - Unique game identifier from ESPN
- `season` (INTEGER) - Year (e.g., 2024, 2025)
- `week` (INTEGER) - Week number
- `date` (TEXT) - Game date/time in ISO format (UTC)
- `game_date_eastern` (TEXT) - Game date in Eastern timezone (YYYY-MM-DD) - **use for filtering**
- `home_team_id` (INTEGER) - Foreign key to teams table
- `away_team_id` (INTEGER) - Foreign key to teams table
- `home_score` (INTEGER) - Final home team score
- `away_score` (INTEGER) - Final away team score
- `winner_team_id` (INTEGER) - ID of winning team
- `completed` (INTEGER) - 0=scheduled, 1=completed
- `neutral_site` (INTEGER) - 0=home game, 1=neutral site
- `conference_game` (INTEGER) - 0=non-conference, 1=conference game
- `venue_name`, `venue_city`, `venue_state` (TEXT)
- `broadcast_network` (TEXT)
- `attendance` (INTEGER)
- `temperature` (INTEGER) - Temperature in Fahrenheit (NEW 2024)
- `wind_speed` (INTEGER) - Wind speed in MPH (NEW 2024)
- `conditions` (TEXT) - Weather conditions (NEW 2024)
- `is_dome` (INTEGER) - 1=indoor/dome, 0=outdoor (NEW 2024)

**Current Data:**
- Seasons: 2024, 2025
- Most recent: 2025 Week 15 (9 games)
- Week 13 2025: 97 games (8 completed)

### TEAMS Table
All CFB teams tracked.

**Key Columns:**
- `team_id` (INTEGER, PRIMARY KEY)
- `name` (TEXT) - Team name (e.g., "Tigers", "Crimson Tide")
- `abbreviation` (TEXT) - Short code
- `display_name` (TEXT) - Full name (e.g., "Alabama Crimson Tide")
- `logo_url` (TEXT) - ESPN logo URL
- `color` (TEXT) - Primary team color hex code
- `conference` (TEXT) - Conference ID/name
- `school_name` (TEXT)

**Total Teams:** 240

### GAME_ODDS Table
Betting lines and odds for games.

**Key Columns:**
- `id` (INTEGER, PRIMARY KEY)
- `game_id` (INTEGER) - Links to games table
- `source` (TEXT) - Odds provider (e.g., "VegasInsider", "TheOddsAPI")
- `opening_spread_home` (REAL) - Opening point spread for home team
- `current_spread_home` (REAL) - Current/latest spread
- `closing_spread_home` (REAL) - Final spread at game time
- `opening_spread_away` (REAL) - Opening spread for away team
- `current_spread_away` (REAL)
- `closing_spread_away` (REAL)
- `opening_moneyline_home` (INTEGER) - Moneyline odds
- `current_moneyline_home` (INTEGER)
- `closing_moneyline_home` (INTEGER)
- `opening_moneyline_away` (INTEGER)
- `current_moneyline_away` (INTEGER)
- `closing_moneyline_away` (INTEGER)
- `opening_total` (REAL) - Over/under total points
- `current_total` (REAL)
- `closing_total` (REAL)
- `opening_over_odds` (INTEGER)
- `current_over_odds` (INTEGER)
- `closing_over_odds` (INTEGER)
- `opening_under_odds` (INTEGER)
- `current_under_odds` (INTEGER)
- `closing_under_odds` (INTEGER)
- `timestamp` (TEXT) - When odds were recorded
- `updated_at` (TEXT) - Last update time

**Note:** Spread is from home team perspective (negative = home favored)

### ODDS_MOVEMENT Table
Historical tracking of odds changes over time.

**Key Columns:**
- `id` (INTEGER, PRIMARY KEY)
- `game_id` (INTEGER)
- `source` (TEXT)
- `spread_home` (REAL)
- `spread_away` (REAL)
- `moneyline_home` (INTEGER)
- `moneyline_away` (INTEGER)
- `total` (REAL)
- `over_odds` (INTEGER)
- `under_odds` (INTEGER)
- `timestamp` (TEXT) - When this snapshot was taken

**Purpose:** Track how lines move leading up to game time

### TEAM_GAME_STATS Table
Detailed team performance statistics for each game.

**Key Columns:**
- `id` (INTEGER, PRIMARY KEY)
- `game_id` (INTEGER)
- `team_id` (INTEGER)
- `points` (INTEGER) - Points scored
- `total_yards` (INTEGER) - Total offense yards
- `passing_yards` (INTEGER)
- `rushing_yards` (INTEGER)
- `passing_completions` (INTEGER)
- `passing_attempts` (INTEGER)
- `rushing_attempts` (INTEGER)
- `turnovers` (INTEGER) - Total turnovers
- `fumbles_lost` (INTEGER)
- `interceptions_thrown` (INTEGER)
- `possession_time` (TEXT) - Time of possession
- `first_downs` (INTEGER)
- `third_down_conversions` (INTEGER)
- `third_down_attempts` (INTEGER)
- `fourth_down_conversions` (INTEGER)
- `fourth_down_attempts` (INTEGER)
- `penalties` (INTEGER) - Number of penalties
- `penalty_yards` (INTEGER) - Yards penalized
- `sacks` (INTEGER) - Sacks recorded (defensive stat)
- `sack_yards` (INTEGER)

**Note:** There are 2 rows per game (one for each team)

### DRIVES Table (NEW - Critical for Predictions!)
Drive-by-drive data providing detailed offensive/defensive efficiency metrics.

**Key Columns:**
- `id` (INTEGER, PRIMARY KEY)
- `game_id` (INTEGER) - Links to games table
- `drive_number` (INTEGER) - Sequential drive number in game
- `team_id` (INTEGER) - Team with possession
- `start_period` (INTEGER) - Quarter/period when drive started
- `start_clock` (TEXT) - Clock time at start
- `start_yard_line` (INTEGER) - Starting field position (0-100)
- `start_yards_to_endzone` (INTEGER) - Distance to opponent endzone (1-99)
- `end_period` (INTEGER) - Quarter when drive ended
- `end_clock` (TEXT) - Clock time at end
- `end_yard_line` (INTEGER) - Ending field position
- `end_yards_to_endzone` (INTEGER) - Distance to endzone at end
- `plays` (INTEGER) - Number of plays in drive
- `yards` (INTEGER) - Net yards gained (can be negative)
- `time_elapsed_seconds` (INTEGER) - Total seconds consumed
- `time_elapsed_display` (TEXT) - Time in MM:SS format
- `result` (TEXT) - Outcome: "TD", "FG", "PUNT", "DOWNS", "INT", "FUMBLE", etc.
- `is_score` (INTEGER) - 1 if drive resulted in points (TD/FG), 0 otherwise
- `description` (TEXT) - Drive summary

**Current Data:**
- CFB 2024: 397 games with drives (29,184 total drives)
- CFB 2025: 412 games with drives (currently scraping more)
- Average: ~25-35 drives per game

**Why This Matters:**
Drive data provides **efficiency metrics** that are 2-3x more predictive than traditional volume stats:
- Points per drive (PPD) - Most predictive single metric
- Yards per drive - Measures consistency
- Scoring percentage - % of drives ending in points
- Red zone efficiency - Scoring rate inside opponent 20
- Three-and-out rate - Drives with ≤3 plays and no score
- Explosive drive rate - Drives with 20+ yards

**Example Queries:**
```sql
-- Calculate team's points per drive
SELECT team_id,
       AVG(CASE
           WHEN result LIKE '%TD%' THEN 7
           WHEN result LIKE '%FG%' THEN 3
           WHEN result LIKE '%SAFETY%' THEN 2
           ELSE 0
       END) as points_per_drive
FROM drives
WHERE game_id IN (SELECT game_id FROM games WHERE season = 2025)
GROUP BY team_id;

-- Red zone efficiency
SELECT team_id,
       COUNT(*) as redzone_attempts,
       SUM(is_score) as scores,
       ROUND(100.0 * SUM(is_score) / COUNT(*), 1) as redzone_pct
FROM drives
WHERE start_yards_to_endzone <= 20
GROUP BY team_id;
```

### PLAYER_GAME_STATS Table
Individual player statistics (currently empty).

**Columns:** Similar to team stats but per player
- `player_name`, `player_id`, `position`
- Passing, rushing, receiving stats

---

## 2. NFL_GAMES.DB (NFL)

### Tables Summary
- **games**: 567 rows - NFL game data (2023-2025)
- **teams**: 34 rows - All NFL teams
- **game_odds**: 419 rows - NFL betting lines
- **odds_movement**: 0 rows - No movement tracking yet
- **team_game_stats**: 922 rows - Team stats
- **drives**: 7,541 rows - **NEW: Drive-by-drive data** (178 games 2025 only)
- **player_game_stats**: 0 rows - Not populated

### Structure
**Identical schema to cfb_games.db** but for NFL data.

**Key Differences:**
- Uses NFL team IDs
- Weeks go up to 22 (includes playoffs)
- Season 2025 Week 12: 14 games total, 1 completed

**Odds Source:** Primarily "TheOddsAPI"

---

## 3. USERS.DB (Predictions & Users)

### Tables Summary
- **users**: 2 rows - User accounts
- **prediction_cache**: 140 rows - Model predictions cache
- **prediction_views**: 80 rows - User viewing history
- **sqlite_sequence**: System table

### USERS Table
User accounts for the dashboard/system.

**Key Columns:**
- `id` (INTEGER, PRIMARY KEY)
- `username` (VARCHAR, UNIQUE)
- `password_hash` (VARCHAR) - Hashed password
- `email` (VARCHAR)
- `created_at` (DATETIME)
- `is_active` (BOOLEAN)

**Current Users:** 2 accounts

### PREDICTION_CACHE Table
Stores predictions made by your ML models.

**Key Columns:**
- `id` (INTEGER, PRIMARY KEY)
- `sport` (TEXT) - "CFB" or "NFL"
- `game_id` (INTEGER) - Links to game in respective DB
- `season` (INTEGER)
- `week` (INTEGER)
- `game_date` (TEXT)
- `home_team` (TEXT) - Team name
- `away_team` (TEXT) - Team name
- `predicted_home_score` (REAL) - Your model's predicted home score
- `predicted_away_score` (REAL) - Your model's predicted away score
- `predicted_spread` (REAL) - Home team spread (predicted_home - predicted_away)
- `predicted_total` (REAL) - Total points (home + away)
- `home_win_probability` (REAL) - Probability home team wins (0-1)
- `vegas_spread` (REAL) - Vegas line for comparison
- `vegas_total` (REAL) - Vegas total for comparison
- `game_completed` (INTEGER) - 0=upcoming, 1=completed
- `actual_home_score` (REAL) - Actual home score (after game)
- `actual_away_score` (REAL) - Actual away score (after game)
- `confidence` (REAL) - Confidence level (0-1)
- `spread_low` (REAL) - Confidence interval low
- `spread_high` (REAL) - Confidence interval high
- `total_low` (REAL) - Total CI low
- `total_high` (REAL) - Total CI high
- `created_at` (TEXT) - When prediction was made

**Purpose:** Cache predictions to avoid re-running models

### PREDICTION_VIEWS Table
Tracks which predictions users have viewed.

**Key Columns:**
- `id` (INTEGER, PRIMARY KEY)
- `user_id` (INTEGER) - Links to users table
- `game_id` (VARCHAR)
- `sport` (VARCHAR)
- `season`, `week` (INTEGER)
- `viewed_at` (DATETIME)
- `home_team`, `away_team` (VARCHAR)
- `predicted_home_score`, `predicted_away_score` (INTEGER)
- `predicted_spread`, `predicted_total` (FLOAT)
- `home_win_probability` (FLOAT)
- `actual_home_score`, `actual_away_score` (INTEGER)
- `prediction_correct` (BOOLEAN) - Was the prediction correct?

**Purpose:** User activity tracking for dashboard

---

## 4. SPORTS_PREDICTIONS.DB

### Structure
Contains 2 tables (from output):
- **nfl_predictions**: 0 rows
- **cfb_predictions**: 0 rows

**Note:** This database appears to be empty/unused. Predictions are stored in users.db instead.

---

## Data Flow & Relationships

### Game Data Flow:
1. **Scraping** → Games scraped from ESPN API
2. **Storage** → Stored in games table (cfb_games.db or nfl_games.db)
3. **Odds** → Betting lines fetched and stored in game_odds table
4. **Stats** → Team/player stats stored in respective tables

### Prediction Flow:
1. **Model** → ML model generates predictions
2. **Cache** → Predictions stored in users.db → prediction_cache
3. **CSV** → Also exported to CSV files (enhanced_predictions_week_13.csv)
4. **Display** → Dashboard reads from prediction_cache
5. **Tracking** → User views logged in prediction_views

### Analysis Flow:
1. **Match** → Join predictions (from CSV or cache) with games (by game_id)
2. **Filter** → Filter for completed games (completed=1)
3. **Compare** → Compare predicted vs actual scores/spreads
4. **Vegas** → Compare your predictions to vegas_spread from game_odds

---

## Key Insights

### What's Working:
- ✅ Games table populated with historical and future games
- ✅ Odds data being collected from multiple sources
- ✅ Predictions being generated and cached
- ✅ Week 13 2025: 64 predictions made

### What Needs Attention:
- ✅ **DRIVES TABLE ADDED** - Critical efficiency metrics now available!
- ⚠️ NFL 2024 needs drive data backfill (currently 0 games)
- ⚠️ CFB 2024 drive data partial (28% coverage)
- ⚠️ Player stats tables are empty (not being populated)
- ⚠️ NFL odds_movement table is empty
- ⚠️ sports_predictions.db is unused
- 🔄 CFB 2025 and NFL 2025 scraping in progress (Week 12+)

### Missing Data:
- Most Week 13 2025 games not yet completed
- Player-level statistics not tracked
- Historical odds movement tracking incomplete

---

## Important Notes

### Game IDs:
- Game IDs come from ESPN (e.g., 401762862)
- These are **consistent** across your predictions and database
- Same game_id used in CFB/NFL games tables and predictions

### Team Names:
- Database stores full names (e.g., "Crimson Tide", "Tigers")
- Multiple teams have same mascot (e.g., "Tigers" = Auburn, LSU, Missouri, etc.)
- **Always match by team_id or game_id, not team name**

### Spreads Convention:
- **Negative spread** = Team is favored (e.g., -7 means favored by 7)
- **Positive spread** = Team is underdog (e.g., +7 means 7-point underdog)
- `predicted_spread` = home_score - away_score (from home perspective)

### Completed Games:
- `completed = 0` → Game scheduled but not played
- `completed = 1` → Game finished, scores available
- Only analyze games where `completed = 1`
