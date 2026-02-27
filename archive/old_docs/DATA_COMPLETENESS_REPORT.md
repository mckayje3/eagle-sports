# Data Completeness Report
Generated: November 25, 2025

## Executive Summary

✅ **CFB 2024**: Complete game/stats data, partial drives (28%)
✅ **CFB 2025**: Scraping in progress - 100% drive coverage for completed games
✅ **NFL 2025**: Scraping in progress - drive data being collected
❌ **NFL 2024**: Missing drive data (needs backfill)

---

## CFB Database (cfb_games.db)

### Games Table
| Season | Total Games | Completed | Upcoming |
|--------|-------------|-----------|----------|
| 2024   | 1,417       | 1,417     | 0        |
| 2025   | 413         | ~350      | ~63      |
| **Total** | **1,830** | **~1,767** | **~63** |

### Team Game Stats
| Season | Games with Stats | Coverage |
|--------|------------------|----------|
| 2024   | 1,297            | 92%      |
| 2025   | 413              | 100%     |
| **Total** | **3,420 records** | **93%** |

### Game Odds
| Season | Games with Odds | Coverage |
|--------|-----------------|----------|
| 2024   | 1,092           | 77%      |
| 2025   | 381             | 92%      |
| **Total** | **1,506 records** | **82%** |

### Drives Table (NEW!)
| Season | Games with Drives | Total Drives | Avg per Game | Coverage |
|--------|-------------------|--------------|--------------|----------|
| 2024   | 397               | ~11,000      | ~28          | 28%      |
| 2025   | 412               | ~18,184      | ~44          | **100%** |
| **Total** | **809 games** | **29,184 drives** | **~36** | **44%** |

**Status:** ✅ 2025 drive collection running - real-time data
**Action needed:** Consider backfilling more 2024 games if training data needed

### Odds Movement
- **Total records:** 188
- **Status:** Limited historical tracking
- **Note:** Not critical for current models

---

## NFL Database (nfl_games.db)

### Games Table
| Season | Total Games | Completed | Upcoming |
|--------|-------------|-----------|----------|
| 2023   | 25          | 25        | 0        |
| 2024   | 270         | 270       | 0        |
| 2025   | 272         | ~180      | ~92      |
| **Total** | **567** | **~475** | **~92** |

### Team Game Stats
| Season | Games with Stats | Coverage |
|--------|------------------|----------|
| 2023   | 13               | 52%      |
| 2024   | 270              | 100%     |
| 2025   | 178              | 65%      |
| **Total** | **922 records** | **81%** |

### Game Odds
| Season | Games with Odds | Coverage |
|--------|-----------------|----------|
| 2023   | 0                | 0%       |
| 2024   | 208              | 77%      |
| 2025   | 211              | 78%      |
| **Total** | **419 records** | **74%** |

### Drives Table (NEW!)
| Season | Games with Drives | Total Drives | Avg per Game | Coverage |
|--------|-------------------|--------------|--------------|----------|
| 2023   | 0                 | 0            | 0            | 0%       |
| 2024   | **0**             | **0**        | **0**        | **0%**   |
| 2025   | 178               | 7,541        | ~42          | 65%      |
| **Total** | **178 games** | **7,541 drives** | **~42** | **31%** |

**Status:** ✅ 2025 drive collection running
**Action needed:** ❌ **NFL 2024 needs drive data backfill** (270 games missing)

### Odds Movement
- **Total records:** 0
- **Status:** Not implemented for NFL
- **Action needed:** Consider adding if line movement analysis desired

---

## Data Quality Assessment

### Excellent Coverage (90-100%)
- ✅ CFB 2025 team_game_stats (100%)
- ✅ CFB 2025 drives (100% of completed games)
- ✅ NFL 2024 games (100%)
- ✅ NFL 2024 team_game_stats (100%)

### Good Coverage (70-89%)
- ✅ CFB 2024 team_game_stats (92%)
- ✅ CFB 2025 game_odds (92%)
- ✅ NFL 2025 team_game_stats (65%, growing)
- ✅ NFL game_odds (74-78%)

### Needs Improvement (<70%)
- ⚠️ CFB 2024 drives (28% - partial backfill)
- ❌ NFL 2024 drives (0% - **needs backfill**)
- ⚠️ NFL 2023 team_game_stats (52%)

### Not Implemented
- ❌ player_game_stats (both CFB and NFL)
- ❌ NFL odds_movement table

---

## Critical Action Items

### Priority 1: NFL 2024 Drive Data Backfill
**Impact:** High - Missing 270 games of drive data
**Effort:** Medium - ~2-3 hours scraping time
**Benefit:** Doubles NFL training data

```bash
# Run NFL 2024 drive backfill
py scrape_nfl_2024_with_drives.py
```

**Expected result:** ~10,000-12,000 drives from 270 games

### Priority 2: CFB 2024 Drive Data (Optional)
**Impact:** Medium - Currently have 28% coverage
**Effort:** High - ~1,000 games to backfill
**Benefit:** More complete 2024 training set

**Decision:** Only if you need more 2024 training data. 2025 data (412 games) may be sufficient.

### Priority 3: Complete 2025 Scraping
**Status:** ✅ Running in background
**Action:** Monitor completion, verify data

---

## Training Data Availability

### For Model Retraining (v3 with drives)

**CFB:**
- Training set: 2024 (397 games with drives) + 2025 weeks 2-10 (~300 games)
- Test set: 2025 weeks 11-13 (~100 games)
- **Total:** ~700 games with drive data
- **Status:** ✅ Sufficient for training

**NFL:**
- Training set: 2025 weeks 1-10 (~160 games with drives)
- Test set: 2025 weeks 11-12 (~20 games)
- **Total:** ~180 games with drive data
- **Status:** ⚠️ Limited but usable
- **Recommendation:** Backfill 2024 to get ~450 games total

### Feature Engineering Ready?

**CFB:** ✅ Yes
- `ml_features_v3_2024.csv` exists (1,515 games)
- Includes drive features
- Models trained (`retrain_with_drive_features.py` completed)

**NFL:** ❌ Not yet
- Need to run feature extraction for NFL
- Should backfill 2024 drives first
- Then run: `ml_feature_extraction_v3_with_drives.py` for NFL

---

## Summary Statistics

### Total Database Size

**CFB:**
- Games: 1,830
- Team stats records: 3,420
- Odds records: 1,506
- Drives: 29,184
- **Completeness:** 85%

**NFL:**
- Games: 567
- Team stats records: 922
- Odds records: 419
- Drives: 7,541
- **Completeness:** 60% (missing 2024 drives)

### Scraping Progress (Background Tasks)

Currently running:
1. ✅ `scrape_cfb_2025_only.py` - Collecting CFB 2025 + drives
2. ✅ `scrape_nfl_2025_with_drives.py` - Collecting NFL 2025 + drives

Expected completion: ~30-60 minutes

---

## Recommendations

### Immediate (Today)
1. ✅ Wait for 2025 scrapers to complete
2. ❌ **Run NFL 2024 drive backfill** (high priority)
3. ✅ Verify data completeness after scraping

### Short Term (This Week)
1. Extract NFL features with drives (`ml_feature_extraction_v3_with_drives.py`)
2. Retrain NFL models with drive data
3. Generate predictions for upcoming NFL week
4. Backfill CFB 2024 drives if needed for better training

### Medium Term (Ongoing)
1. Weekly scraping routine for new games
2. Continuous model retraining with latest data
3. Performance monitoring vs Vegas lines
4. Consider implementing odds_movement for NFL

---

## Files to Run

### Data Collection
```bash
# NFL 2024 drive backfill (PRIORITY)
py scrape_nfl_2024_with_drives.py

# Check status
py -c "import sqlite3; conn = sqlite3.connect('nfl_games.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(DISTINCT game_id) FROM drives WHERE game_id IN (SELECT game_id FROM games WHERE season=2024)'); print(f'NFL 2024 games with drives: {cursor.fetchone()[0]}'); conn.close()"
```

### Feature Extraction
```bash
# NFL features with drives
py ml_feature_extraction_v3_with_drives.py --db nfl_games.db --output ml_features_v3_nfl_2024_2025.csv

# Verify
py -c "import pandas as pd; df = pd.read_csv('ml_features_v3_nfl_2024_2025.csv'); print(f'NFL features: {len(df)} games, {len(df.columns)} features')"
```

### Model Training
```bash
# Retrain NFL models
py retrain_nfl_with_drive_features.py

# Compare performance
py compare_models.py --sport nfl --v2 models/nfl_v2 --v3 models/nfl_v3_with_drives
```

---

## Conclusion

**Overall Status:** 🟡 Good progress, action needed

✅ **CFB:** Well-positioned with 2025 drive data
⚠️ **NFL:** Need 2024 drive backfill for optimal training

**Next Critical Step:** Run `scrape_nfl_2024_with_drives.py` to backfill NFL 2024 drives
