# Sports Betting Edge Prediction System

Spread and total predictions for NFL, CFB, NBA, CBB, and NHL using Ridge regression and neural network models. Finds edges vs Vegas lines.

## Key Commands

```bash
# Daily update (auto-detects in-season sports, syncs to cloud)
python daily_update.py

# Generate predictions for specific sports
python predict_nfl_playoffs.py      # NFL playoff predictions (updates CSV)
python cfb_predictor.py             # CFB predictions
python nba_predictor.py             # NBA predictions
python cbb_predictor.py             # CBB predictions
python nhl_predictor.py             # NHL predictions

# Retrain models (after significant data updates)
python nfl_simple_model.py          # Retrain NFL simple model
python train_deep_eagle_nfl.py      # Retrain NFL deep model (requires deep-eagle lib)
python cbb_enhanced_ridge.py        # Retrain CBB enhanced model
python nba_enhanced_ridge.py        # Retrain NBA enhanced model

# Update odds from ESPN
python espn_unified_odds.py         # Scrape current odds for all sports

# Sync to cloud (DBs + prediction CSVs)
python push_databases.py

# Streamlit dashboard
streamlit run streamlit_app.py      # Local: http://localhost:8501
```

## Architecture

```
├── *_games.db              # Per-sport SQLite databases (nfl, cfb, nba, cbb, nhl)
├── *_predictor.py          # Main prediction scripts per sport
├── *_simple_model.py       # Ridge regression models (primary)
├── *_enhanced_ridge.py     # Ridge + form/HCA/drive features
├── *_edge_classifier.py    # Neural net meta-model for bet confidence
├── *_espn_scraper.py       # Per-sport ESPN data scrapers
├── espn_unified_odds.py    # Unified odds scraper (all sports)
├── streamlit_app.py        # Streamlit dashboard + bet tracking
├── system_health.py        # Health monitoring for dashboard
├── daily_update.py         # Orchestrates daily data refresh
├── models/                 # Saved model weights and scalers
├── archive/                # Deprecated scripts, one-time migrations
└── api/                    # FastAPI backend (auth, predictions API)
```

## Database Schema

Each sport has identical structure in `{sport}_games.db`:

| Table | Purpose |
|-------|---------|
| `teams` | Team metadata (team_id, name, abbreviation) |
| `games` | Game results (game_id, date, game_date_eastern, week, home/away_team_id, scores) |
| `team_game_stats` | Per-team box scores (yards, turnovers, etc.) |
| `odds_and_predictions` | Vegas lines + model predictions per game |
| `drives` | Drive-level data (NFL/CFB only) |

**Key joins:**
```sql
-- Games with team names and odds
SELECT g.*, ht.name as home_team, at.name as away_team, o.latest_spread, o.latest_total
FROM games g
JOIN teams ht ON g.home_team_id = ht.team_id
JOIN teams at ON g.away_team_id = at.team_id
LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
```

**Gotchas:**
- `game_id` is ESPN's ID (INTEGER, unique per sport)
- Team IDs are also ESPN IDs, not sequential
- `completed = 1` means game has final score
- NBA has player availability tables; others don't
- **All sports have `game_date_eastern` column** - use this for date filtering/grouping
- The `date` column stores UTC timestamps (e.g., `2026-01-14T23:00Z`), but `game_date_eastern` gives the correct Eastern date (`2026-01-13` for late games)
- CFB `date` column is date-only format (no timestamps)
- **CFB postseason games** (championship, bowl games in January) are stored as `season=N+1, week=1` but team stats are in `season=N`. The predictor auto-handles this by using previous season stats when `week <= 1`.

### Prediction Columns (CRITICAL)

The `odds_and_predictions` table has multiple prediction columns. **Always use the ADJUSTED values:**

| Column | Contains | Use? |
|--------|----------|------|
| `avg_pred_spread` | **ADJUSTED** spread (with post-prediction adjustments) | ✅ YES |
| `avg_pred_total` | **ADJUSTED** total (with post-prediction adjustments) | ✅ YES |
| `predicted_home_score` | Raw model score prediction | ❌ Only for display |
| `predicted_away_score` | Raw model score prediction | ❌ Only for display |

**Correct query pattern:**
```sql
-- Use COALESCE to fall back to raw calculation if adjusted not available
COALESCE(o.avg_pred_spread, o.predicted_away_score - o.predicted_home_score) as predicted_spread,
COALESCE(o.avg_pred_total, o.predicted_home_score + o.predicted_away_score) as predicted_total
```

**Column availability by sport:**
| Sport | Has `avg_pred_spread`? | Has adjustments? |
|-------|------------------------|------------------|
| NFL | ✅ Yes | Big underdog, rest, bye |
| NBA | ✅ Yes | Big underdog, fade, struggling home |
| CBB | ✅ Yes | Big underdog, fade close games |
| CFB | ❌ No | Use raw calculation |
| NHL | ✅ Yes | Underdog bias (+0.15 goals) - **Strategy: bet HOME UNDERDOGS on MONEYLINE** |

### Schema Change Protocol (MANDATORY)

**When changing database schema or prediction logic:**
1. Update the predictor script (`*_predictor.py`)
2. Update the database saver (`update_predictions_*.py`)
3. **ALWAYS update `streamlit_app.py` queries** - there are multiple queries per sport!
4. Search for all occurrences: `grep -n "predicted_spread\|predicted_total" streamlit_app.py`
5. Test the dashboard shows correct values before committing

**Why this matters:** The app queries raw scores and calculates spread/total directly. If adjustments are stored in `avg_pred_spread` but the app uses `predicted_away_score - predicted_home_score`, the displayed values will be WRONG by up to 8 points!

### Backtest/Threshold Update Protocol (MANDATORY)

**When updating backtest results or star rating thresholds:**
1. Update the predictor script (`*_predictor.py`) with new adjustments
2. Update `streamlit_app.py` - specifically:
   - `display_top_picks()` function thresholds for 3-star plays
   - Sport-specific game card display (star rating logic)
3. Update `CLAUDE.md` documentation:
   - Confidence Stars tables (per sport)
   - 3-Star Plays Section table
   - Post-Prediction Adjustments section (if applicable)
4. Update `MODEL_IMPROVEMENTS.md` with new backtest findings

**Files to update (checklist):**
- [ ] `*_predictor.py` - adjustment constants and logic
- [ ] `streamlit_app.py` - `display_top_picks()` thresholds + game card star ratings
- [ ] `CLAUDE.md` - Confidence Stars tables + 3-Star Plays table
- [ ] `MODEL_IMPROVEMENTS.md` - backtest results and recommendations

**Why this matters:** Star ratings drive betting decisions. If backtest shows 65% ATS at 1+ goal edge but the app still shows 2 stars, users miss profitable plays. Always keep predictor, app, and docs in sync!

### Model Improvement Protocol (Fade Logic Check)

**CRITICAL:** When improving a model's accuracy, always re-evaluate any existing fade logic.

**Current fade strategies:**
- NFL Ridge Totals: Fade UNDER at 5+ pts (61.9% when fading)

**Why this matters:** Fade strategies exist because a model consistently fails in one direction. If you improve the model and it no longer fails that way, the fade becomes counterproductive (you'd be betting opposite of a now-accurate prediction).

**When retraining or tuning a model:**
1. Run backtest on the new model
2. Check if any fade strategies apply to this model (see list above)
3. If fade exists: re-run fade analysis to verify it's still beneficial
4. Update or remove fade logic as needed
5. Document changes in `MODEL_IMPROVEMENTS.md`

### ESPN API Gotchas

**CBB Scoreboard Limitation:** ESPN's scoreboard API (`/scoreboard?dates=YYYYMMDD`) only returns ~12 featured games per day, NOT all games. The `update_game_results.py` script handles this by also querying incomplete games directly via the event summary API.

**Supported Sports:** `update_game_results.py` supports all 5 sports: `nba`, `nfl`, `cfb`, `cbb`, `nhl`

## Conventions

### Spread Convention (CRITICAL)
All spreads stored as `away_score - home_score`:
- **Negative** = home team favored
- **Positive** = away team favored
- Vegas line `-7` stored as `-7.0` (home favored by 7)

See `SPREAD_CONVENTIONS.md` for full details.

### Edge Calculation
```python
edge = model_spread - vegas_spread
# Negative edge → bet HOME
# Positive edge → bet AWAY
```

### NBA Model (Ridge V2 + Rule-Based Confidence - PROFITABLE)
The NBA predictor uses Ridge V2 (pure model, no Vegas blend) with rule-based confidence scoring:

**Ridge V2 Features:**
- SRS opponent-adjusted ratings
- Reduced HCA (1.5 pts vs traditional 2.2)
- Road favorite penalty (+1.5 pts)
- NO Vegas blending (pure model for edge detection)

**Walk-forward Results (2025-2026 combined, 1615 games):**
- Base 5+ pt edges: 59.0% ATS (308/522)
- **With rule filter (2+ stars): 64.3% ATS (126/196)** - BEST
- **3+ stars: 63.1% ATS (41/65)**
- Totals: ~50% at all thresholds - NOT profitable (skip totals)
- Model files: `models/nba_ridge_v2.pkl`

**Key Rule: FADE road favorite picks** (35% ATS -> **65% when faded**)

### CBB Model (Enhanced Ridge - Promising at 2-4 pt edges)
The CBB predictor uses an Enhanced Ridge model with an unusual pattern:

- Walk-forward validation: Train on 2024+2025, test on 2026 YTD (353 games)
- **2-4 pt edges: 62.9% ATS (44/70)** - promising but small sample (p=0.02, doesn't survive multiple testing)
- Pattern held in both time splits: 69% (first half) and 58.5% (second half)
- 5+ pt edges: 50.3% - consistently degrades at high edges
- Totals: ~50% at all thresholds - NOT profitable
- Model files: `models/cbb_enhanced_model.pkl`

**Caution:** CBB shows opposite pattern of other sports (medium edges > high edges) but sample is small. Monitor as more games accumulate.

### NFL Post-Prediction Adjustments
The NFL predictor (Deep Eagle) applies automatic adjustments after model prediction:

1. **Big underdog** (+1.0 pts): When team is 7+ point underdog
2. **Rest advantage** (+0.5 pts): When team has 3+ days extra rest
3. **Post-bye boost** (+1.0 pts): When team is coming off bye week

NFL Deep Eagle backtest: 58.4% ATS at 5+ pt edges, 63.0% at 7+ pt edges.

### NHL Betting Strategy (MONEYLINE - NOT Puck Line!)

**IMPORTANT:** NHL puck lines (±1.5) have lopsided juice (+200/-200 range), NOT -110 like NFL/NBA spreads. Our puck line analysis was misleading. Use MONEYLINES instead.

**Moneyline Backtest (707 games with moneylines):**

| Strategy | Games | Win Rate | ROI | Significance |
|----------|-------|----------|-----|--------------|
| **Home Underdogs (all)** | 223 | 52.0% | +17.5% | p=0.009 ✓ |
| Home dogs +100-120 | 94 | 59.6% | +26.1% | Best range! |
| Home dogs +140-180 | 43 | 55.8% | +42.7% | Also good |
| Away Underdogs | 484 | 39.5% | -7.2% | AVOID |

**Strategy: BET HOME UNDERDOGS ON THE MONEYLINE**
- 3-star plays: Home dog +100 to +120 (59.6% win rate)
- 2-star plays: Home dog +121 to +180 (55%+ win rate)
- 1-star plays: Home dog +181+ (smaller sample)

### Feature Naming
- `*_ppg` = points per game
- `*_papg` = points allowed per game
- `*_yards` = total yards
- Form = avg margin last 4 games
- Streak = consecutive W/L count (negative = losses)

## Dashboard Features

### 3-Star Plays Section
Each sport page shows a "⭐⭐⭐ 3-Star Plays" section at the top highlighting profitable picks based on **walk-forward validation**:

| Sport | Bet Type | Threshold | Win Rate | Sample | Status |
|-------|----------|-----------|----------|--------|--------|
| NHL | **Moneyline** | Home dog +100-120 | 59.6% | 94 | **CONFIRMED** (+26% ROI, p=0.009) |
| NBA | Spread | **2+ stars (rule-based)** | **64.3% ATS** | 196 | **BEST** - rule filter |
| CBB | Spread | 2-4 pts | 62.9% ATS | 70 | **Promising** (small sample) |
| CFB | Spread | 5+ pts | 57.1% ATS | - | **PROFITABLE** |
| NFL | Spread | 5+ pts | 55.3% ATS | - | **PROFITABLE** |

**Note:** NHL uses MONEYLINE (not puck line) because puck lines have lopsided juice.
**NBA Rule Filter:** Skip road fav picks; prefer home picks, home favs, close games.
**Totals:** Not profitable for any sport - all show 1 star for totals.

**Walk-forward validation** = train on season N, test on season N+1 (simulates real deployment).

### Unified Game Card Format
All sports (NFL, CFB, NBA, CBB) use a consistent 3-row game card:
```
Row 1: Away @ Home          | 🕐 7:00 PM ET | ✓ (if completed)
Row 2: Spread | Vegas: +X.X | Model: +X.X | Edge: +X.X → Pick | ⭐⭐⭐
Row 3: Total  | Vegas: X.X  | Model: X.X  | Edge: +X.X → OVER/UNDER | ⭐⭐
```

Games are sorted by game time (earliest first). NBA/CBB show Eastern times.

### System Health Banner
The dashboard shows a color-coded health banner:
- **Green**: All systems operational
- **Yellow**: Warning (stale data, missing predictions)
- **Red**: Error (database issues, failed updates)

Check `system_health.py` for health logic. Monitors:
- Database freshness (warns if >24h, errors if >48h)
- Prediction coverage for upcoming games
- Vegas odds availability
- Daily update log success/failure

### NFL Week Selector
NFL predictions use a unified week selector:
- Weeks 1-18: Regular season
- Wild Card, Divisional, Conference, Super Bowl: Playoff rounds

During playoffs (Jan-Feb), select the playoff round to see predictions.
Playoff predictions come from `predict_nfl_playoffs.py` and `nfl_playoff_predictions.csv`.

### Confidence Stars

**NFL (Deep Eagle - profitability-based):**

*Spreads:*
| Edge | Stars | Win % | Profitable? |
|------|-------|-------|-------------|
| >= 5 pts | ⭐⭐⭐ | 58.4% | Yes |
| < 5 pts | ⭐ | ~52% | No (below breakeven) |

*Totals:*
| Edge | Stars | Win % | Profitable? |
|------|-------|-------|-------------|
| >= 7 pts | ⭐⭐⭐ | TBD | Needs analysis |
| >= 4 pts | ⭐⭐ | TBD | Needs analysis |
| < 4 pts | ⭐ | TBD | Needs analysis |

**CFB (not currently profitable):**
| Spread Edge | Stars | Total Edge | Stars |
|-------------|-------|------------|-------|
| >= 5 pts | ⭐⭐⭐ | >= 8 pts | ⭐⭐⭐ |
| >= 3 pts | ⭐⭐ | >= 5 pts | ⭐⭐ |
| < 3 pts | ⭐ | < 5 pts | ⭐ |

Note: CFB models show ~50% ATS at all thresholds - not currently profitable.

**NBA (Ridge V2 + Rule-Based Confidence - PROFITABLE):**

*Spreads (1615 games, 2025-2026):*
| Criteria | Stars | Win % | Sample | Notes |
|----------|-------|-------|--------|-------|
| Road Fav picks - **FADE** | ⭐⭐ | **65%** | 17 | Bet HOME DOG instead |
| 5+ edge, basic | ⭐ | 57.1% | 312 | Marginal |
| 5+ edge + home/close/big edge | ⭐⭐ | **64.9%** | 131 | **BET** |
| Multiple positive factors | ⭐⭐⭐ | **63.1%** | 65 | **STRONG BET** |
| All factors aligned | ⭐⭐⭐⭐ | 75% | 8 | Small sample |

**Rule-Based Scoring (5+ pt edge required):**
1. **FADE road favorite picks** (bet home dog instead - 65% ATS)
2. +1 for home picks (62.5% vs 55.4%)
3. +1 for home favorites (67% ATS)
4. +1 for close games (Vegas < 4 pts)
5. -1 for blowouts (Vegas 10+ pts)
6. +1 for big edges (7+ pts)

*Totals:* All 1 star (~50% at all thresholds - not profitable)

**CBB (Enhanced Ridge - Promising at 2-4 pt edges):**

*Spreads (353 games, 2026 YTD):*
| Edge | Stars | Win % | Sample | Notes |
|------|-------|-------|--------|-------|
| 2-4 pts | ⭐⭐⭐ | 62.9% | 70 | Promising but small sample (p=0.02) |
| 5+ pts | ⭐ | 50.3% | 171 | Consistently poor |
| < 2 pts | ⭐ | 50.0% | 112 | Noise |

**Note:** CBB shows opposite pattern - high edges degrade. Pattern held in both time splits (69% early, 58.5% late). Monitor as sample grows.

*Totals:* All 1 star (~50% ATS, not profitable)

**NHL (707 games with moneylines, MONEYLINE STRATEGY):**

**IMPORTANT:** Puck lines (±1.5) have lopsided juice (+200/-200), NOT -110 like NFL/NBA. Use MONEYLINES instead.

*Moneyline (Home Underdogs):*
| ML Range | Stars | Win % | ROI | Notes |
|----------|-------|-------|-----|-------|
| +100 to +120 | ⭐⭐⭐ | 59.6% | +26.1% | Best range (94 games) |
| +121 to +180 | ⭐⭐ | ~55% | +20%+ | Good range |
| +181+ | ⭐ | ~46% | Varies | Smaller sample |
| Away underdogs | ❌ | 39.5% | -7.2% | AVOID |

**Simple NHL Strategy:** Bet HOME UNDERDOGS on the moneyline = 52.0% win rate, +17.5% ROI (p=0.009, statistically significant)

---

Note: Breakeven at -110 vig is ~52.4%. Only 3-star and 2-star picks are profitable.

## Cloud Sync

The daily update automatically syncs local and cloud apps:

**What gets committed:**
- Databases: `nfl_games.db`, `cfb_games.db`, `nba_games.db`, `cbb_games.db`, `nhl_games.db`
- Predictions: `*_predictions.csv`, `*_current_predictions.csv`

**How it works:**
1. `daily_update.py` runs at 9am (scheduled task)
2. Auto-detects in-season sports (all 5: NFL, CFB, NBA, CBB, NHL)
3. For each sport: scrapes games, updates results, fetches odds, generates predictions
4. Calls `push_databases.py` which commits DBs + CSVs
5. Cloud app auto-deploys from GitHub

**Manual sync:**
```bash
python push_databases.py  # Commit and push all DBs + CSVs
```

## Don't Touch

- **`archive/one_time_scripts/`** - Migration scripts, run once only
- **`models/*.pt` + `*_scaler.pkl`** - Trained model pairs, always update together
- **Deep Eagle models** - Require external `deep-eagle` library at `DEEP_EAGLE_PATH`
- **`users.db`** - Auth database for API/dashboard

## Troubleshooting

### Enhanced Ridge Models Return None for All Games
If CBB/NBA predictions fail with "model returned None" for every game:
- **Cause:** The saved model file is missing `team_stats` data (teams appear to have 0 games)
- **Fix:** Retrain the model: `python cbb_enhanced_ridge.py` or `python nba_enhanced_ridge.py`
- This repopulates team stats from the database and saves them in the model file

### CFB Championship/Postseason Games Missing Stats
If CFB predicts with "No game data for [team] - using defaults" for January games:
- **Cause:** Game stored as Season N+1, Week 1, but stats are in Season N
- **Fix:** Already handled automatically - predictor uses previous season for week ≤ 1

## Common Tasks

### Find Edges for Upcoming Games
```bash
python predict_nfl_playoffs.py   # Or cfb/nba/cbb/nhl_predictor.py
```
Look for 3+ point edges on spreads, 5+ on totals. For NHL, 1+ goal edge = 3-star play.

### Update NFL Playoff Lines
Edit `PLAYOFF_GAMES` in `predict_nfl_playoffs.py`:
```python
# Format: (away, home, vegas_spread, vegas_total, date, time_slot)
PLAYOFF_GAMES = [
    ('Bills', 'Jaguars', +1.5, 52.5, '2026-01-11', 'SAT 1:00 PM'),
    # ... more games
]
```
Then run `python predict_nfl_playoffs.py` to regenerate predictions.

### Add New Game Data
ESPN scrapers auto-fetch. For manual:
```python
from cfb_espn_scraper import CFBScraper
scraper = CFBScraper()
scraper.scrape_week(2025, 15)  # Season, week
```

### Check Model Performance
```bash
python analyze_predictions.py    # Historical ATS/totals performance
python fair_comparison.py        # Compare models head-to-head
```

### Backfill Historical Odds
Scripts in `archive/one_time_scripts/backfill_*.py` - check before running, most are one-time.

---
*See `~/.claude/CLAUDE.md` for Python/SQL coding standards.*
