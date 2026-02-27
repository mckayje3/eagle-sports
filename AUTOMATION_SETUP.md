# NFL Predictions Automation Setup

## Overview

This system automates NFL prediction updates:
1. **Tuesday mornings**: Full update (game stats, closing lines, opening lines, predictions)
2. **On-demand**: Update predictions with latest odds via dashboard button

## Files

| File | Purpose |
|------|---------|
| `nfl_tuesday_update.py` | Full Tuesday update script |
| `nfl_tuesday_update.bat` | Batch file for Task Scheduler |
| `update_predictions.py` | On-demand prediction update |
| `fetch_latest_odds.py` | Fetch odds from The Odds API |

## Setting Up Windows Task Scheduler

### Option 1: Using Task Scheduler GUI

1. Open **Task Scheduler** (search in Start menu)
2. Click **Create Basic Task**
3. Configure:
   - **Name**: `NFL Tuesday Update`
   - **Description**: `Update NFL game stats and predictions after Monday Night Football`
   - **Trigger**: Weekly, every Tuesday at 8:00 AM
   - **Action**: Start a program
   - **Program**: `C:\Users\jbeast\documents\coding\sports\nfl_tuesday_update.bat`
   - **Start in**: `C:\Users\jbeast\documents\coding\sports`

4. In the task properties, check:
   - "Run whether user is logged on or not" (optional)
   - "Run with highest privileges"

### Option 2: Using PowerShell (Run as Admin)

```powershell
$action = New-ScheduledTaskAction -Execute "C:\Users\jbeast\documents\coding\sports\nfl_tuesday_update.bat" -WorkingDirectory "C:\Users\jbeast\documents\coding\sports"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At 8:00AM
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "NFL Tuesday Update" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Update NFL game stats and predictions after Monday Night Football"
```

### Option 3: Using schtasks command (Run as Admin)

```cmd
schtasks /create /tn "NFL Tuesday Update" /tr "C:\Users\jbeast\documents\coding\sports\nfl_tuesday_update.bat" /sc weekly /d TUE /st 08:00 /ru %USERNAME%
```

## Manual Testing

Test the scripts manually before setting up automation:

```cmd
# Test Tuesday update (dry run - no database changes)
py nfl_tuesday_update.py --test

# Test Tuesday update (full run)
py nfl_tuesday_update.py

# Test on-demand update
py update_predictions.py
```

## Dashboard Usage

The **Update Odds** button in the NFL predictions section:
1. Fetches latest odds from The Odds API
2. Regenerates predictions using the Deep Eagle model
3. Syncs predictions to the dashboard cache

## Log Files

- `nfl_tuesday_update.log` - Tuesday update log
- `nfl_scraper.log` - ESPN scraper log

## API Key

Ensure `odds_api_config.json` contains your API key:
```json
{
    "api_key": "YOUR_ODDS_API_KEY"
}
```

## Troubleshooting

1. **Script not running**: Check Task Scheduler history
2. **Python not found**: Ensure Python is in PATH or use full path
3. **Database locked**: Close any SQLite browsers before running
4. **API errors**: Check remaining API credits in logs
