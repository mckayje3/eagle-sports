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
| NHL | ✅ Yes | Underdog bias (+0.15 goals) - **Strategy: always bet underdog +1.5** |

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

### NBA Model (Enhanced Ridge - Profitable at 5+ pt edges)
The NBA predictor uses an Enhanced Ridge model with dynamic HCA, injuries, momentum, and streaks:

- Walk-forward validation: Train on 2024+2025, test on 2026 YTD (512 games)
- **5+ pt edges: 56.1% ATS (37/66), +7% ROI** - PROFITABLE
- **6+ pt edges: 56.8% ATS (21/37), +8% ROI** - PROFITABLE
- 3-4 pt edges: 53.2% ATS - marginally profitable
- Totals: ~50% at all thresholds - NOT profitable (skip totals)
- Model files: `models/nba_ridge_enhanced.pkl`

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

### NHL Post-Prediction Adjustments
The NHL predictor applies a small adjustment after model prediction:

1. **Underdog bias** (+0.15 goals): Small adjustment toward underdog

NHL edge analysis backtest (720 games - 3+ seasons):
- **Model edges >= 1.0 goal: 65.0% ATS** (+24.1% ROI) - Best strategy
- **Underdog +1.5 (all games): 60.7% ATS** (+15.9% ROI) - Simple strategy
- Overs hit 53.5% (marginal)

**Strategies:**
1. **Best:** Use 3-star picks (model edge >= 1.0 goal) for 65% ATS
2. **Simple:** Bet underdog +1.5 on every game for 60.7% ATS

### Feature Naming
- `*_ppg` = points per game
- `*_papg` = points allowed per game
- `*_yards` = total yards
- Form = avg margin last 4 games
- Streak = consecutive W/L count (negative = losses)

## Dashboard Features

### 3-Star Plays Section
Each sport page shows a "⭐⭐⭐ 3-Star Plays" section at the top highlighting profitable picks based on **walk-forward validation**:

| Sport | Spread Threshold | Walk-Forward ATS | Sample | Status |
|-------|------------------|------------------|--------|--------|
| NHL | 1+ goal | 65.0% ATS | 720 | **CONFIRMED** (+24% ROI) |
| CBB | 2-4 pts | 62.9% ATS | 70 | **Promising** (small sample) |
| CFB | 5+ pts | 57.1% ATS (2025) | - | **PROFITABLE** |
| NBA | 5+ pts | 56.1% ATS | 66 | **PROFITABLE** (+7% ROI) |
| NFL | 5+ pts | 55.3% ATS (2025) | - | **PROFITABLE** |

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

**NBA (Enhanced Ridge - PROFITABLE at 5+ pt edges):**

*Spreads (512 games, 2026 YTD):*
| Edge | Stars | Win % | Sample | ROI |
|------|-------|-------|--------|-----|
| >= 5 pts | ⭐⭐⭐ | 56.1% | 66 | +7.0% |
| >= 3 pts | ⭐⭐ | 53.2% | 173 | +1.5% |
| < 3 pts | ⭐ | ~50% | 339 | Negative |

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

**NHL (720 games - 3+ seasons, CONFIRMED PROFITABLE):**

*Puck Line (±1.5):*
| Edge | Stars | Win % | Profitable? |
|------|-------|-------|-------------|
| >= 1.0 goal | ⭐⭐⭐ | 65.0% | Yes (+24.1% ROI) |
| < 1.0 goal | ⭐ | ~52% | No (below breakeven) |

*Totals:*
| Edge | Stars | Win % | Profitable? |
|------|-------|-------|-------------|
| All | ⭐ | 53.5% | No (marginal) |

**Simple NHL Strategy:** Bet underdog +1.5 on every game = 60.7% ATS (+15.9% ROI)

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
