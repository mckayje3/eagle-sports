"""
VegasInsider NBA Odds Scraper using Selenium
Scrapes current NBA odds from vegasinsider.com
"""
import sqlite3
import time
import re
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def download_chromedriver_cft():
    """Download ChromeDriver from Chrome for Testing (CfT) for Chrome 115+"""
    import os
    import zipfile
    import urllib.request
    import json
    import shutil

    # Get installed Chrome version
    chrome_version = "142.0.7444.177"  # From error message

    # ChromeDriver download URL for Chrome for Testing
    # For 142.x we need the matching driver
    major_version = chrome_version.split('.')[0]

    # Check if we already have a chromedriver
    driver_dir = os.path.join(os.path.dirname(__file__), '.chromedriver')
    driver_path = os.path.join(driver_dir, 'chromedriver.exe')

    if os.path.exists(driver_path):
        return driver_path

    os.makedirs(driver_dir, exist_ok=True)

    # Get the latest known good versions
    try:
        known_good_url = 'https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json'
        req = urllib.request.Request(known_good_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

        # Find matching version
        matching_version = None
        for version_info in reversed(data.get('versions', [])):
            ver = version_info.get('version', '')
            if ver.startswith(f"{major_version}."):
                matching_version = version_info
                break

        if matching_version:
            downloads = matching_version.get('downloads', {}).get('chromedriver', [])
            for download in downloads:
                if download.get('platform') == 'win32':
                    url = download.get('url')
                    print(f"Downloading ChromeDriver {matching_version['version']}...")

                    zip_path = os.path.join(driver_dir, 'chromedriver.zip')
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        with open(zip_path, 'wb') as f:
                            f.write(resp.read())

                    # Extract
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        for name in z.namelist():
                            if name.endswith('chromedriver.exe'):
                                z.extract(name, driver_dir)
                                # Move from nested folder to driver_dir
                                extracted = os.path.join(driver_dir, name)
                                if extracted != driver_path:
                                    shutil.move(extracted, driver_path)
                                break

                    os.remove(zip_path)
                    print(f"ChromeDriver installed at {driver_path}")
                    return driver_path

    except Exception as e:
        print(f"Error downloading ChromeDriver: {e}")

    return None


def setup_driver():
    """Setup Chrome driver with options"""
    options = Options()
    options.add_argument('--headless=new')  # New headless mode for Chrome 109+
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36')

    # Try to download/use Chrome for Testing driver
    driver_path = download_chromedriver_cft()
    if driver_path:
        service = Service(driver_path)
    else:
        # Fall back to letting Selenium find the driver
        service = Service()

    driver = webdriver.Chrome(service=service, options=options)
    return driver


def normalize_team_name(name):
    """Normalize team names for matching"""
    name = name.lower().strip()

    # Common abbreviations and variations
    mappings = {
        'la lakers': 'los angeles lakers',
        'la clippers': 'los angeles clippers',
        'ny knicks': 'new york knicks',
        'gs warriors': 'golden state warriors',
        'golden state': 'golden state warriors',
        'okc thunder': 'oklahoma city thunder',
        'okc': 'oklahoma city thunder',
        'sa spurs': 'san antonio spurs',
        'no pelicans': 'new orleans pelicans',
        'nop': 'new orleans pelicans',
        'phx suns': 'phoenix suns',
        'phoenix': 'phoenix suns',
        'por blazers': 'portland trail blazers',
        'portland': 'portland trail blazers',
        'phi 76ers': 'philadelphia 76ers',
        'philly': 'philadelphia 76ers',
        'was wizards': 'washington wizards',
        'washington': 'washington wizards',
        'mem grizzlies': 'memphis grizzlies',
        'memphis': 'memphis grizzlies',
        'mil bucks': 'milwaukee bucks',
        'milwaukee': 'milwaukee bucks',
        'min wolves': 'minnesota timberwolves',
        'minnesota': 'minnesota timberwolves',
        'det pistons': 'detroit pistons',
        'detroit': 'detroit pistons',
        'ind pacers': 'indiana pacers',
        'indiana': 'indiana pacers',
        'atl hawks': 'atlanta hawks',
        'atlanta': 'atlanta hawks',
        'bos celtics': 'boston celtics',
        'boston': 'boston celtics',
        'bkn nets': 'brooklyn nets',
        'brooklyn': 'brooklyn nets',
        'cha hornets': 'charlotte hornets',
        'charlotte': 'charlotte hornets',
        'chi bulls': 'chicago bulls',
        'chicago': 'chicago bulls',
        'cle cavs': 'cleveland cavaliers',
        'cleveland': 'cleveland cavaliers',
        'dal mavs': 'dallas mavericks',
        'dallas': 'dallas mavericks',
        'den nuggets': 'denver nuggets',
        'denver': 'denver nuggets',
        'hou rockets': 'houston rockets',
        'houston': 'houston rockets',
        'mia heat': 'miami heat',
        'miami': 'miami heat',
        'orl magic': 'orlando magic',
        'orlando': 'orlando magic',
        'sac kings': 'sacramento kings',
        'sacramento': 'sacramento kings',
        'tor raptors': 'toronto raptors',
        'toronto': 'toronto raptors',
        'uta jazz': 'utah jazz',
        'utah': 'utah jazz',
    }

    # Check direct mappings first
    if name in mappings:
        return mappings[name]

    # Return as-is if full name
    return name


def get_team_id_by_name(cursor, team_name):
    """Get team_id from database by team name"""
    normalized = normalize_team_name(team_name)

    # Try exact match on display_name
    cursor.execute('''
        SELECT team_id, display_name FROM teams
        WHERE LOWER(display_name) = ?
    ''', (normalized,))
    row = cursor.fetchone()
    if row:
        return row[0], row[1]

    # Try LIKE match
    cursor.execute('''
        SELECT team_id, display_name FROM teams
        WHERE LOWER(display_name) LIKE ?
    ''', (f'%{normalized.split()[-1]}%',))
    row = cursor.fetchone()
    if row:
        return row[0], row[1]

    return None, None


def scrape_vegasinsider_nba():
    """Scrape NBA odds from VegasInsider"""
    print("Setting up Selenium driver...")
    driver = setup_driver()

    games_data = []

    # NBA team name mappings from VegasInsider short names
    team_map = {
        'lakers': 'Los Angeles Lakers',
        'clippers': 'Los Angeles Clippers',
        'celtics': 'Boston Celtics',
        'nets': 'Brooklyn Nets',
        'knicks': 'New York Knicks',
        '76ers': 'Philadelphia 76ers',
        'sixers': 'Philadelphia 76ers',
        'raptors': 'Toronto Raptors',
        'bulls': 'Chicago Bulls',
        'cavaliers': 'Cleveland Cavaliers',
        'cavs': 'Cleveland Cavaliers',
        'pistons': 'Detroit Pistons',
        'pacers': 'Indiana Pacers',
        'bucks': 'Milwaukee Bucks',
        'hawks': 'Atlanta Hawks',
        'hornets': 'Charlotte Hornets',
        'heat': 'Miami Heat',
        'magic': 'Orlando Magic',
        'wizards': 'Washington Wizards',
        'nuggets': 'Denver Nuggets',
        'timberwolves': 'Minnesota Timberwolves',
        'wolves': 'Minnesota Timberwolves',
        'thunder': 'Oklahoma City Thunder',
        'blazers': 'Portland Trail Blazers',
        'trail blazers': 'Portland Trail Blazers',
        'jazz': 'Utah Jazz',
        'warriors': 'Golden State Warriors',
        'suns': 'Phoenix Suns',
        'kings': 'Sacramento Kings',
        'mavericks': 'Dallas Mavericks',
        'mavs': 'Dallas Mavericks',
        'rockets': 'Houston Rockets',
        'grizzlies': 'Memphis Grizzlies',
        'pelicans': 'New Orleans Pelicans',
        'spurs': 'San Antonio Spurs',
    }

    try:
        url = 'https://www.vegasinsider.com/nba/odds/las-vegas/'
        print(f"Loading {url}...")
        driver.get(url)

        # Wait for content to load
        time.sleep(5)

        print("Parsing page content...")

        # Get all text content and parse structure
        body_text = driver.find_element(By.TAG_NAME, "body").text
        lines = [l.strip() for l in body_text.split('\n') if l.strip()]

        print(f"  Found {len(lines)} text lines")

        # Parse games - look for pattern: rotation number, team name, spread
        # Format: "511\nLakers\n+6.5\n..." or "511 Lakers +6.5 ..."
        i = 0
        current_away = None
        current_home = None

        while i < len(lines):
            line = lines[i].lower()

            # Check if this line is a team name
            for short_name, full_name in team_map.items():
                if short_name in line and len(line) < 30:
                    # Found a team - now look for spread
                    spread = None
                    total = None

                    # Check next few lines for numbers
                    for j in range(i, min(i + 10, len(lines))):
                        # Look for spread pattern like "+6.5" or "-3.5"
                        spread_match = re.search(r'([+-]\d+\.?\d*)', lines[j])
                        if spread_match:
                            val = float(spread_match.group(1))
                            if -25 <= val <= 25 and spread is None:
                                spread = val
                            elif 180 <= val <= 280:
                                total = val

                    if spread is not None:
                        if current_away is None:
                            # First team = away team
                            current_away = {
                                'team': full_name,
                                'spread': spread
                            }
                        else:
                            # Second team = home team
                            current_home = {
                                'team': full_name,
                                'spread': spread
                            }

                            # We have a complete game - home spread is the one we want
                            game_data = {
                                'away_team': current_away['team'],
                                'home_team': current_home['team'],
                                'spread': current_home['spread'],  # Home team spread
                                'total': total
                            }
                            games_data.append(game_data)
                            print(f"  {current_away['team']} @ {current_home['team']}: Spread {current_home['spread']}")

                            # Reset for next game
                            current_away = None
                            current_home = None

                    break
            i += 1

        print(f"\nParsed {len(games_data)} games")

    except Exception as e:
        print(f"Error scraping VegasInsider: {e}")
        import traceback
        traceback.print_exc()

    finally:
        driver.quit()

    return games_data


def save_odds_to_db(games_data, db_path='nba_games.db', users_db_path='users.db'):
    """Save scraped odds to database and prediction_cache"""
    if not games_data:
        print("No games data to save")
        return 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Also update users.db prediction_cache
    users_conn = sqlite3.connect(users_db_path)
    users_cursor = users_conn.cursor()

    saved = 0
    for game in games_data:
        try:
            # Find game in database
            home_id, _ = get_team_id_by_name(cursor, game['home_team'])
            away_id, _ = get_team_id_by_name(cursor, game['away_team'])

            if not home_id or not away_id:
                print(f"  Could not find teams: {game['home_team']} vs {game['away_team']}")
                continue

            # Find matching game
            cursor.execute('''
                SELECT game_id FROM games
                WHERE home_team_id = ? AND away_team_id = ?
                AND season = 2025
                AND date >= date('now', '-1 day')
                AND date <= date('now', '+7 days')
            ''', (home_id, away_id))

            row = cursor.fetchone()
            if row:
                game_id = row[0]
                spread = game.get('spread')
                total = game.get('total')

                # Upsert odds to odds_and_predictions table
                cursor.execute('''
                    INSERT INTO odds_and_predictions (game_id, source, latest_spread, latest_total, odds_updated_at)
                    VALUES (?, 'VegasInsider', ?, ?, ?)
                    ON CONFLICT(game_id) DO UPDATE SET
                        source = excluded.source,
                        latest_spread = excluded.latest_spread,
                        latest_total = excluded.latest_total,
                        odds_updated_at = excluded.odds_updated_at
                ''', (game_id, spread, total, datetime.now().isoformat()))

                # Update prediction_cache in users.db
                users_cursor.execute('''
                    UPDATE prediction_cache
                    SET vegas_spread = COALESCE(?, vegas_spread),
                        vegas_total = COALESCE(?, vegas_total)
                    WHERE game_id = ? AND sport = 'NBA'
                ''', (spread, total, game_id))

                saved += 1
                print(f"  Saved: Game {game_id} - Spread {spread}, Total {total}")

        except Exception as e:
            print(f"  Error saving game: {e}")

    conn.commit()
    conn.close()
    users_conn.commit()
    users_conn.close()
    return saved


if __name__ == '__main__':
    print("=" * 60)
    print("VEGASINSIDER NBA ODDS SCRAPER")
    print("=" * 60)

    games = scrape_vegasinsider_nba()

    if games:
        print(f"\nFound {len(games)} games with odds")
        saved = save_odds_to_db(games)
        print(f"Saved {saved} games to database")
    else:
        print("\nNo games parsed - the page structure may have changed")
        print("Check the debug output above to understand the page layout")
