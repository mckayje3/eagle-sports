# NCAA D1 FBS College Football Data Scraper

Automatically fetch and store NCAA Division 1 FBS college football game results and statistics from ESPN into a SQLite database.

## Features

- Fetch game scores, team information, and detailed statistics from ESPN
- **Betting odds tracking** from VegasInsider (spreads, moneylines, totals)
- Store data in a structured SQLite database
- Support for both current and historical seasons
- Automated scheduled updates
- **Line movement tracking** to monitor odds changes over time
- Comprehensive team and game statistics including:
  - Final scores and game metadata (date, venue, attendance, broadcast network)
  - Total yards, passing yards, rushing yards
  - Completions, attempts, turnovers
  - Time of possession
  - First downs, third/fourth down conversions
  - Penalties and penalty yards
  - **Opening, current, and closing betting lines**
  - And more!

## Installation

1. Ensure you have Python 3.7+ installed

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python database.py
```

## Quick Start

### Initialize the Database

```bash
python database.py
```

This creates `cfb_games.db` with all necessary tables.

### Fetch Current Week's Games

```bash
python scheduler.py current
```

### Fetch Recent Games (Last 7 Days)

```bash
python scheduler.py recent
```

### Fetch Yesterday's Games

```bash
python scheduler.py yesterday
```

### Fetch Entire Current Season

```bash
python scheduler.py full
```

### Fetch Historical Data

Backfill multiple seasons:
```bash
python scheduler.py historical 2020 2024
```

### Fetch Specific Week

```bash
python scheduler.py week 12 2024
```

### Fetch Betting Odds

Fetch current betting odds:
```bash
python scheduler.py odds
```

Fetch both games and odds together:
```bash
python scheduler.py odds_and_games
```

## Automated Scheduled Updates

Run the scheduler to automatically update data on a schedule:

```bash
python scheduler.py schedule
```

Default schedule:
- **Daily at 9:00 AM**: Update yesterday's games (catch late finishes)
- **Saturday at 11:00 PM**: Update current week
- **Sunday at 9:00 AM**: Update recent games
- **Monday at 10:00 AM**: Weekly comprehensive update
- **Tuesday/Thursday/Friday at 2:00 PM**: Update betting odds
- **Saturday at 10:00 AM**: Update betting odds

The scheduler will run continuously and log all activities to `cfb_scraper.log`.

### Running as a Background Service

**Windows (using Task Scheduler):**
1. Open Task Scheduler
2. Create a new task
3. Set trigger to "At startup"
4. Set action to run: `python C:\Users\jbeast\documents\coding\sports\scheduler.py schedule`
5. Configure to run whether user is logged in or not

**Linux/Mac (using systemd or cron):**

Create a systemd service file `/etc/systemd/system/cfb-scraper.service`:
```ini
[Unit]
Description=CFB Data Scraper
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/path/to/sports
ExecStart=/usr/bin/python3 /path/to/sports/scheduler.py schedule
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl enable cfb-scraper
sudo systemctl start cfb-scraper
```

## Database Schema

### Tables

**teams**
- team_id (PRIMARY KEY)
- name, abbreviation, display_name
- logo_url, color
- conference

**games**
- game_id (PRIMARY KEY)
- season, week, date
- home_team_id, away_team_id
- home_score, away_score, winner_team_id
- venue_name, venue_city, venue_state
- attendance, broadcast_network
- neutral_site, conference_game, completed

**team_game_stats**
- game_id, team_id
- points, total_yards
- passing_yards, passing_completions, passing_attempts
- rushing_yards, rushing_attempts
- turnovers, fumbles_lost, interceptions_thrown
- possession_time
- first_downs
- third_down_conversions, third_down_attempts
- fourth_down_conversions, fourth_down_attempts
- penalties, penalty_yards
- sacks, sack_yards

**player_game_stats** (available for future expansion)
- Individual player statistics

**game_odds** (betting odds tracking)
- game_id, source
- opening_spread_home, opening_spread_away
- current_spread_home, current_spread_away
- closing_spread_home, closing_spread_away
- opening_moneyline_home, opening_moneyline_away
- current_moneyline_home, current_moneyline_away
- closing_moneyline_home, closing_moneyline_away
- opening_total, current_total, closing_total
- over/under odds (opening, current, closing)

**odds_movement** (line movement history)
- game_id, source, timestamp
- spread_home, spread_away
- moneyline_home, moneyline_away
- total, over_odds, under_odds

For detailed information on betting odds tracking, see [ODDS_GUIDE.md](ODDS_GUIDE.md)

## Usage Examples

### Python Script Usage

```python
from cfb_espn_scraper import ESPNScraper

# Create scraper instance
scraper = ESPNScraper()

# Scrape a specific week
scraper.scrape_week(season=2024, week=12)

# Scrape entire season
scraper.scrape_season(season=2024, start_week=1, end_week=15)

# Scrape date range
scraper.scrape_date_range('20240901', '20241130')
```

### Querying the Database

```python
import sqlite3

# Connect to database
conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

# Get all games for a team
cursor.execute('''
    SELECT g.date, t1.name as home, t2.name as away,
           g.home_score, g.away_score
    FROM games g
    JOIN teams t1 ON g.home_team_id = t1.team_id
    JOIN teams t2 ON g.away_team_id = t2.team_id
    WHERE g.season = 2024
    ORDER BY g.date
''')

for row in cursor.fetchall():
    print(row)

# Get team statistics for a game
cursor.execute('''
    SELECT t.name, s.points, s.total_yards,
           s.passing_yards, s.rushing_yards
    FROM team_game_stats s
    JOIN teams t ON s.team_id = t.team_id
    WHERE s.game_id = ?
''', (401628404,))

for row in cursor.fetchall():
    print(row)

conn.close()
```

### SQL Query Examples

**Find highest scoring games:**
```sql
SELECT date, home_score + away_score as total_points,
       (SELECT name FROM teams WHERE team_id = home_team_id) as home,
       (SELECT name FROM teams WHERE team_id = away_team_id) as away
FROM games
WHERE season = 2024 AND completed = 1
ORDER BY total_points DESC
LIMIT 10;
```

**Team season statistics:**
```sql
SELECT t.name,
       COUNT(*) as games_played,
       AVG(s.points) as avg_points,
       AVG(s.total_yards) as avg_yards
FROM team_game_stats s
JOIN teams t ON s.team_id = t.team_id
JOIN games g ON s.game_id = g.game_id
WHERE g.season = 2024
GROUP BY t.team_id
ORDER BY avg_points DESC;
```

**Conference standings:**
```sql
SELECT t.name, t.conference,
       SUM(CASE WHEN g.winner_team_id = t.team_id THEN 1 ELSE 0 END) as wins,
       SUM(CASE WHEN g.winner_team_id != t.team_id AND
           (g.home_team_id = t.team_id OR g.away_team_id = t.team_id)
           THEN 1 ELSE 0 END) as losses
FROM teams t
JOIN games g ON (g.home_team_id = t.team_id OR g.away_team_id = t.team_id)
WHERE g.season = 2024 AND g.completed = 1
GROUP BY t.team_id
ORDER BY t.conference, wins DESC;
```

## Project Structure

```
sports/
├── database.py          # Database schema and operations
├── cfb_espn_scraper.py  # CFB ESPN data fetching and parsing
├── scheduler.py         # Automated scheduling and manual updates
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── cfb_games.db        # SQLite database (created after initialization)
└── cfb_scraper.log     # Log file (created when scheduler runs)
```

## Data Source

Data is fetched from ESPN's public API endpoints used by their website. This scraper:
- Respects rate limits with built-in delays
- Only fetches publicly available data
- Is designed for personal use and research

## Troubleshooting

**Import errors:**
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**Database locked errors:**
Close any database connections or SQLite browser tools before running updates.

**No games found:**
- Check that the season/week/date parameters are correct
- Football season typically runs late August through early January
- Some weeks may have no games (bye weeks, off weeks)

**Rate limiting:**
The scraper includes built-in delays. If you encounter issues, the delays can be increased in `cfb_espn_scraper.py`.

## Customization

### Modify Update Schedule

Edit the schedule in `scheduler.py` function `run_scheduled_updates()`:

```python
# Change from Saturday 11 PM to Saturday 10 PM
schedule.every().saturday.at("22:00").do(scheduler.update_current_week)
```

### Add More Statistics

Extend the database schema in `database.py` and update the parsing logic in `cfb_espn_scraper.py` to capture additional statistics available in ESPN's API.

### Export Data

Create a simple export script:

```python
import sqlite3
import csv

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

# Export to CSV
cursor.execute('SELECT * FROM games WHERE season = 2024')
with open('games_2024.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([description[0] for description in cursor.description])
    writer.writerows(cursor.fetchall())
```

## License

This project is for personal and educational use. Please respect ESPN's terms of service and use responsibly.

## Contributing

Feel free to extend this project with:
- Additional statistics tracking
- Player-level statistics parsing
- Data visualization tools
- Export to other formats (Excel, JSON, etc.)
- Advanced analytics and reporting
