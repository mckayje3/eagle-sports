# Additional ESPN API Data Available for Scraping

Based on analysis of ESPN's CFB API, here's what additional data we can scrape beyond what we currently collect:

## ✅ Currently Scraped
- Basic game info (date, teams, scores, venue, attendance)
- Team game statistics (yards, turnovers, time of possession, etc.)
- Team information (names, logos, colors, conferences)
- Betting odds (spread, moneyline, totals)

## 📊 Available But NOT Currently Scraped

### 1. **Player Statistics** (HIGHLY VALUABLE)
**Endpoint:** `boxscore.players`

Individual player performance data including:
- **Passing:** completions, attempts, yards, TDs, INTs, QBR
- **Rushing:** attempts, yards, TDs, fumbles, longest run
- **Receiving:** receptions, yards, TDs, targets, longest catch
- **Defense:** tackles, sacks, TFLs, pass deflections, INTs
- **Kicking:** FGs made/attempted, XPs, longest FG
- **Punting:** punts, average, inside 20, longest
- **Returns:** kick returns, punt returns, yards, TDs

**Value:** Essential for:
- Player prop betting predictions
- Player performance tracking
- Injury impact analysis
- Team depth analysis

### 2. **Drive-by-Drive Data** (EXCELLENT)
**Endpoint:** `drives.previous`

Detailed information for each offensive drive:
- **Drive metrics:** yards gained, plays run, time elapsed
- **Start/end position:** field position data
- **Result:** touchdown, field goal, punt, turnover, etc.
- **Individual plays:** play-by-play within each drive

**Value:**
- Offensive/defensive efficiency metrics
- Red zone efficiency
- Field position analysis
- Time management analysis

### 3. **Scoring Plays** (GOOD)
**Endpoint:** `scoringPlays`

Chronological list of all scoring plays:
- Type of score (TD, FG, safety, etc.)
- Quarter and time
- Score after play
- Team that scored
- Play description

**Value:**
- Scoring patterns (when teams score)
- Comeback analysis
- Momentum shifts

### 4. **Win Probability** (VERY VALUABLE)
**Endpoint:** `winprobability`

Play-by-play win probability throughout the game:
- Win probability after each play
- Probability swings/momentum
- Game flow visualization data

**Value:**
- Model validation (compare our predictions to ESPN's)
- Close game identification
- Comeback probability
- Game excitement metrics

### 5. **Weather Data** (USEFUL)
**Endpoint:** `gameInfo.weather`

Game-time weather conditions:
- Temperature
- Wind speed/direction
- Precipitation
- Weather conditions (clear, rain, snow, etc.)

**Value:**
- Weather impact on scoring
- Home field advantage adjustments
- Passing vs rushing game predictions

### 6. **Odds History & Movements** (VALUABLE)
**Endpoint:** `odds`

Historical betting lines:
- Opening lines
- Line movements throughout week
- Multiple sportsbooks
- Consensus lines

**Value:**
- Sharp money detection
- Line value identification
- Public vs sharp betting trends

### 7. **Against The Spread (ATS) Records** (GOOD)
**Endpoint:** `againstTheSpread`

Team ATS performance:
- Season ATS record
- Home/away ATS splits
- Conference ATS records
- Trends

**Value:**
- Betting performance tracking
- Team over/under performance vs expectations

### 8. **Standings & Rankings** (USEFUL)
**Endpoint:** `standings`

Current conference standings:
- Conference records
- Division standings
- Playoff implications

**Value:**
- Strength of schedule
- Playoff race context
- Motivation factors

### 9. **Game Leaders** (NICE TO HAVE)
**Endpoint:** `leaders`

Top performers in key categories:
- Passing yards leader
- Rushing yards leader
- Receiving yards leader
- Tackles leader

**Value:**
- Quick game summary
- Star player identification

### 10. **Broadcast Information** (BASIC)
**Endpoint:** `broadcasts`

TV/streaming details:
- Network
- Announcers
- Broadcast time

**Value:**
- Game visibility (national vs regional)
- Primetime vs daytime

---

## 🎯 Recommendations: What to Add First

### Priority 1: **Player Statistics**
- Most valuable for detailed analysis
- Essential for player props
- Enables deeper team analysis
- Requires new `player_stats` table (already exists but empty)

### Priority 2: **Drive Data**
- Excellent for efficiency metrics
- Red zone performance
- Time of possession by drive
- Field position analysis
- Requires new `drives` table

### Priority 3: **Win Probability**
- Great for model validation
- Momentum analysis
- Close game identification
- Requires new `win_probability` table

### Priority 4: **Weather Data**
- Easy to add (just new columns)
- Useful for model features
- Add to existing `games` table

### Priority 5: **Enhanced Odds Tracking**
- Line movement tracking
- Multiple sportsbooks
- Already partially implemented in `odds_movement`

---

## 💾 Required Database Changes

### New Tables Needed:

#### 1. **player_game_stats** (exists but not populated)
```sql
Already exists! Just need to populate it.
```

#### 2. **drives** (new)
```sql
CREATE TABLE drives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    drive_number INTEGER,
    team_id INTEGER NOT NULL,
    start_period INTEGER,
    start_clock TEXT,
    start_yard_line INTEGER,
    end_period INTEGER,
    end_clock TEXT,
    end_yard_line INTEGER,
    plays INTEGER,
    yards INTEGER,
    time_elapsed TEXT,
    result TEXT,  -- TD, FG, Punt, TO, etc.
    is_score INTEGER,
    FOREIGN KEY (game_id) REFERENCES games (game_id),
    FOREIGN KEY (team_id) REFERENCES teams (team_id)
);
```

#### 3. **scoring_plays** (new)
```sql
CREATE TABLE scoring_plays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    period INTEGER,
    clock TEXT,
    scoring_type TEXT,  -- TD, FG, Safety, etc.
    away_score INTEGER,
    home_score INTEGER,
    play_text TEXT,
    FOREIGN KEY (game_id) REFERENCES games (game_id),
    FOREIGN KEY (team_id) REFERENCES teams (team_id)
);
```

#### 4. **win_probability** (new)
```sql
CREATE TABLE win_probability (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    play_id TEXT,
    home_win_probability REAL,
    play_number INTEGER,
    period INTEGER,
    clock TEXT,
    FOREIGN KEY (game_id) REFERENCES games (game_id)
);
```

#### 5. **weather** (add to games table)
```sql
ALTER TABLE games ADD COLUMN temperature INTEGER;
ALTER TABLE games ADD COLUMN wind_speed INTEGER;
ALTER TABLE games ADD COLUMN conditions TEXT;
```

---

## 🚀 Implementation Estimate

### Phase 1: Player Stats (2-3 hours)
- Update scraper to fetch player data
- Populate `player_game_stats` table
- Test on recent games

### Phase 2: Weather (30 minutes)
- Add columns to `games` table
- Update scraper to extract weather
- Backfill recent games

### Phase 3: Drives & Scoring (1-2 hours)
- Create new tables
- Update scraper for drive data
- Test on recent games

### Phase 4: Win Probability (1 hour)
- Create new table
- Add to scraper
- Historical data collection

---

## 📈 Impact on ML Models

Adding this data would enable:

1. **Better predictions:**
   - Weather-adjusted scoring models
   - Player availability impact
   - Offensive efficiency metrics
   - Defensive efficiency metrics

2. **Player props:**
   - Individual player predictions
   - Over/under on player stats

3. **Live betting:**
   - Win probability models
   - In-game adjustments

4. **Advanced analytics:**
   - Drive success rate
   - Red zone efficiency
   - Field position value
   - Situational performance

Would you like me to implement any of these additions?
