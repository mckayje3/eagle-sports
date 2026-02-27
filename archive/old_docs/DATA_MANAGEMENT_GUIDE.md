# Data Management Guide - Preventing Data Loss

## Issue: 2025 CFB Data Was Lost

### What Happened
The 2025 CFB data was successfully scraped but disappeared from the database. Here's why:

1. **Original scrape**: `scrape_2025_season.py` ran successfully and collected all 2025 data
2. **Concurrent scrapers**: Multiple scrapers ran in the background simultaneously:
   - `scrape_2024_with_drives.py` (CFB 2024)
   - `scrape_nfl_2024_with_drives.py` (NFL 2024)
   - `ml_feature_extraction_v3_with_drives.py`

3. **Potential causes**:
   - SQLite database locks from concurrent writes
   - Transaction rollbacks
   - One scraper may have recreated tables or cleared data

### Current Solution (November 2025)

Created two dedicated scrapers that preserve existing data:
- `scrape_cfb_2025_only.py` - CFB 2025 with drive data
- `scrape_nfl_2025_with_drives.py` - NFL 2025 with drive data

Both scrapers:
- ✅ Check existing data before starting
- ✅ Report what data is preserved
- ✅ Add new data without removing old data
- ✅ Include drive-by-drive statistics
- ✅ Verify final state after completion

### Best Practices Going Forward

#### 1. Sequential Scraping (NOT Concurrent)
```bash
# GOOD: Run scrapers one at a time
py scrape_cfb_2025_only.py
# Wait for completion
py scrape_nfl_2025_with_drives.py
```

```bash
# BAD: Don't run multiple scrapers simultaneously
py scrape_cfb_2025_only.py &
py scrape_nfl_2025_with_drives.py &  # NO!
```

#### 2. Verify Before and After
Always check data state:
```python
import sqlite3

# Before scraping
conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()
cursor.execute('SELECT season, COUNT(*) FROM games GROUP BY season')
print("Before:", cursor.fetchall())
conn.close()

# Run scraper...

# After scraping
conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()
cursor.execute('SELECT season, COUNT(*) FROM games GROUP BY season')
print("After:", cursor.fetchall())
conn.close()
```

#### 3. Database Backups
Before major scraping operations:
```bash
# Windows
copy cfb_games.db cfb_games.backup.db
copy nfl_games.db nfl_games.backup.db

# Use current date in backup name
copy cfb_games.db cfb_games_2025_11_25.db
```

#### 4. Season-Specific Scripts
Use dedicated scripts for each season/year combination:
- `scrape_cfb_2024_with_drives.py` - Only 2024 CFB
- `scrape_cfb_2025_only.py` - Only 2025 CFB
- `scrape_nfl_2024_with_drives.py` - Only 2024 NFL
- `scrape_nfl_2025_with_drives.py` - Only 2025 NFL

**Never use generic scrapers that might overwrite data!**

#### 5. Check for Running Processes
Before starting a scraper:
```bash
# Check what's running
# (Claude Code has /tasks command for this)

# Kill old scrapers if needed
# Make sure no conflicts exist
```

### Database Schema Safety

Both CFB and NFL databases use the same schema:
- `games` table - Game metadata
- `teams` table - Team information
- `team_game_stats` table - Detailed stats per team per game
- `drives` table - Drive-by-drive data (NEW in 2024)
- `weather` columns - Temperature, wind, is_dome (NEW in 2024)

**Important**: Scrapers use `INSERT OR REPLACE` which is safe for:
- Re-scraping the same game (updates with latest data)
- Adding new games without affecting old ones

### Recovery Process

If data is lost:

1. **Check backups first**
   ```bash
   dir *.db
   # Look for backup files
   ```

2. **Re-run the appropriate scraper**
   ```bash
   py scrape_cfb_2025_only.py
   ```

3. **Verify the data**
   ```python
   import sqlite3
   conn = sqlite3.connect('cfb_games.db')
   cursor = conn.cursor()

   # Check game counts
   cursor.execute('SELECT season, COUNT(*) FROM games GROUP BY season')
   print(cursor.fetchall())

   # Check drive counts
   cursor.execute('''
       SELECT g.season, COUNT(d.drive_id)
       FROM games g
       LEFT JOIN drives d ON g.game_id = d.game_id
       GROUP BY g.season
   ''')
   print(cursor.fetchall())
   conn.close()
   ```

### Current Status (Nov 25, 2025)

**Running Now:**
- ✅ `scrape_cfb_2025_only.py` (background)
- ✅ `scrape_nfl_2025_with_drives.py` (background)

**Expected Results:**
- CFB 2025: ~800-1000 games (weeks 1-15)
- NFL 2025: ~180-200 games (weeks 1-12, current week)
- Both include drive-by-drive data

**When Complete:**
The databases will contain:
- CFB: 2024 (1,830 games) + 2025 (~800-1000 games)
- NFL: 2023 (25 games) + 2024 (542 games) + 2025 (~180-200 games)

### Training Data Updates

After 2025 data is collected:

1. **Extract features for 2025**:
   ```bash
   # Update feature extractor to include 2025
   py ml_feature_extraction_v3_with_drives.py
   ```

2. **Retrain models with both seasons**:
   ```bash
   # Update training script to use 2024+2025
   py retrain_with_drive_features.py
   ```

3. **Use time-based validation**:
   - Train on: 2024 weeks 2-12 + 2025 weeks 2-10
   - Test on: 2025 weeks 11-13
   - This gives more training data and tests on most recent games

### Prevention Checklist

Before running any scraper:
- [ ] Check current database state
- [ ] Create a backup of existing databases
- [ ] Verify no other scrapers are running
- [ ] Use season-specific scraper scripts
- [ ] Run scrapers sequentially (not in parallel)
- [ ] Verify data after completion
- [ ] Document any issues encountered

### Quick Reference

**Check data:**
```bash
py -c "import sqlite3; conn = sqlite3.connect('cfb_games.db'); cursor = conn.cursor(); cursor.execute('SELECT season, COUNT(*) FROM games GROUP BY season'); print(cursor.fetchall()); conn.close()"
```

**Backup:**
```bash
copy cfb_games.db cfb_games_backup.db
```

**Monitor scraper:**
```bash
# Use Claude Code /tasks command or BashOutput tool
```

---

## Summary

The 2025 data loss was likely due to concurrent scraping operations causing database conflicts. Going forward:

1. Use dedicated season-specific scrapers
2. Run scrapers sequentially, never concurrently
3. Always backup before major operations
4. Verify data before and after scraping
5. Use the new `scrape_cfb_2025_only.py` and `scrape_nfl_2025_with_drives.py` scripts

The scrapers are currently running and will restore all 2025 data with drive-by-drive statistics!
