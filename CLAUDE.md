# Sports Betting Edge Prediction System

Spread and total predictions for NFL, CFB, NBA, and CBB using Ridge regression and neural network models. Finds edges vs Vegas lines.

## Key Commands

```bash
# Daily update (auto-detects in-season sports, syncs to cloud)
python daily_update.py

# Generate predictions for specific sports
python predict_nfl_playoffs.py      # NFL playoff predictions (updates CSV)
python cfb_predictor.py             # CFB predictions
python nba_predictor.py             # NBA predictions
python cbb_predictor.py             # CBB predictions

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
├── *_games.db              # Per-sport SQLite databases (nfl, cfb, nba, cbb)
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
| `games` | Game results (game_id, date, week, home/away_team_id, scores) |
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

### Model Confidence Thresholds
| Edge | Confidence |
|------|------------|
| >= 4 pts | HIGH (***) |
| >= 2 pts | MEDIUM (**) |
| >= 1 pt | LOW (*) |

Totals edges need ~1.5x magnitude for equivalent confidence.

### Feature Naming
- `*_ppg` = points per game
- `*_papg` = points allowed per game
- `*_yards` = total yards
- Form = avg margin last 4 games
- Streak = consecutive W/L count (negative = losses)

## Dashboard Features

### Unified Game Card Format
All sports (NFL, CFB, NBA, CBB) use a consistent 3-row game card:
```
Row 1: Away @ Home          | Date/Time     | ✓ (if completed)
Row 2: Spread | Vegas: +X.X | Model: +X.X | Edge: +X.X → Pick | ⭐⭐⭐
Row 3: Total  | Vegas: X.X  | Model: X.X  | Edge: +X.X → OVER/UNDER | ⭐⭐
```

Games are sorted by edge magnitude (best picks first).

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
| Spread Edge | Stars | Total Edge | Stars |
|-------------|-------|------------|-------|
| >= 5 pts | ⭐⭐⭐ | >= 8 pts | ⭐⭐⭐ |
| >= 3 pts | ⭐⭐ | >= 5 pts | ⭐⭐ |
| < 3 pts | ⭐ | < 5 pts | ⭐ |

## Cloud Sync

The daily update automatically syncs local and cloud apps:

**What gets committed:**
- Databases: `*_games.db`
- Predictions: `*_predictions.csv`, `*_current_predictions.csv`

**How it works:**
1. `daily_update.py` runs at 9am (scheduled task)
2. Updates results, odds, predictions for in-season sports
3. Calls `push_databases.py` which commits DBs + CSVs
4. Cloud app auto-deploys from GitHub

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
python predict_nfl_playoffs.py   # Or cfb/nba/cbb_predictor.py
```
Look for 3+ point edges on spreads, 5+ on totals.

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
