"""
Scrape NFL data using date ranges (required for historical data)
ESPN ignores 'year' parameter for NFL week-based queries
"""
import requests
import time
import sqlite3
from datetime import datetime, timedelta
from nfl_espn_scraper import NFLESPNScraper
from cfb_nfl_database import FootballDatabase

class NFLDateScraper:
    """Scrape NFL using date ranges for historical data"""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

    # NFL 2024 season weeks (Thu-Mon of each week)
    NFL_2024_WEEKS = [
        ("2024-09-05", "2024-09-10", 1),
        ("2024-09-12", "2024-09-17", 2),
        ("2024-09-19", "2024-09-24", 3),
        ("2024-09-26", "2024-10-01", 4),
        ("2024-10-03", "2024-10-08", 5),
        ("2024-10-10", "2024-10-15", 6),
        ("2024-10-17", "2024-10-22", 7),
        ("2024-10-24", "2024-10-29", 8),
        ("2024-10-31", "2024-11-05", 9),
        ("2024-11-07", "2024-11-12", 10),
        ("2024-11-14", "2024-11-19", 11),
        ("2024-11-21", "2024-11-26", 12),  # Current week
        ("2024-11-28", "2024-12-03", 13),  # Thanksgiving
        ("2024-12-05", "2024-12-10", 14),
        ("2024-12-12", "2024-12-17", 15),
        ("2024-12-19", "2024-12-24", 16),
        ("2024-12-26", "2024-12-31", 17),
        ("2025-01-02", "2025-01-07", 18),
    ]

    # NFL 2024 Postseason
    NFL_2024_PLAYOFFS = [
        ("2025-01-11", "2025-01-14", 1),  # Wild Card
        ("2025-01-18", "2025-01-21", 2),  # Divisional
        ("2025-01-26", "2025-01-27", 3),  # Conference Championships
        ("2025-02-09", "2025-02-10", 4),  # Super Bowl
    ]

    # NFL 2023 season (for backfill if needed)
    NFL_2023_WEEKS = [
        ("2023-09-07", "2023-09-12", 1),
        ("2023-09-14", "2023-09-19", 2),
        ("2023-09-21", "2023-09-26", 3),
        ("2023-09-28", "2023-10-03", 4),
        ("2023-10-05", "2023-10-10", 5),
        ("2023-10-12", "2023-10-17", 6),
        ("2023-10-19", "2023-10-24", 7),
        ("2023-10-26", "2023-10-31", 8),
        ("2023-11-02", "2023-11-07", 9),
        ("2023-11-09", "2023-11-14", 10),
        ("2023-11-16", "2023-11-21", 11),
        ("2023-11-23", "2023-11-28", 12),
        ("2023-11-30", "2023-12-05", 13),
        ("2023-12-07", "2023-12-12", 14),
        ("2023-12-14", "2023-12-19", 15),
        ("2023-12-21", "2023-12-26", 16),
        ("2023-12-28", "2024-01-02", 17),
        ("2024-01-06", "2024-01-08", 18),
    ]

    NFL_2023_PLAYOFFS = [
        ("2024-01-13", "2024-01-16", 1),
        ("2024-01-20", "2024-01-22", 2),
        ("2024-01-28", "2024-01-29", 3),
        ("2024-02-11", "2024-02-12", 4),
    ]

    def __init__(self, db_path: str = 'nfl_games.db'):
        self.db = FootballDatabase(db_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.scraper = NFLESPNScraper(db_path)

    def fetch_scoreboard_by_dates(self, start_date: str, end_date: str) -> dict:
        """Fetch games in a date range"""
        url = f"{self.BASE_URL}/scoreboard"
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        params = {
            'limit': '100',
            'dates': f"{start}-{end}"
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching scoreboard: {e}")
            return {}

    def scrape_week_by_date(self, start_date: str, end_date: str, season: int, week: int):
        """Scrape a week of games using date range"""
        print(f"Scraping {season} Week {week} ({start_date} to {end_date})...")

        scoreboard = self.fetch_scoreboard_by_dates(start_date, end_date)
        events = scoreboard.get('events', [])

        print(f"  Found {len(events)} games")

        self.db.connect()
        self.db.initialize_schema()

        for event in events:
            try:
                # Parse and save teams
                competitions = event.get('competitions', [{}])[0]
                competitors = competitions.get('competitors', [])

                for competitor in competitors:
                    team_data = self.scraper.parse_team_data(competitor.get('team', {}))
                    if team_data.get('team_id') and team_data.get('name'):
                        self.db.insert_or_update_team(team_data)

                # Parse and save game
                game_data = self.scraper.parse_game_data(event, season)
                game_data['week'] = week

                if game_data.get('game_id'):
                    self.db.insert_or_update_game(game_data)

                    if game_data.get('completed'):
                        print(f"    Fetching stats for game {game_data['game_id']}...")
                        time.sleep(0.3)
                        game_details = self.scraper.fetch_game_details(str(game_data['game_id']))
                        # Process stats
                        box_score = game_details.get('boxscore', {})
                        teams = box_score.get('teams', [])
                        for team_stats in teams:
                            team_id = team_stats.get('team', {}).get('id')
                            stats = team_stats.get('statistics', [])
                            if team_id and stats:
                                stats_data = self.scraper.parse_team_stats(game_data['game_id'], int(team_id), stats)
                                self.db.insert_or_update_team_stats(stats_data)

            except Exception as e:
                print(f"    Error processing event: {e}")
                continue

        self.db.close()
        print(f"  Week {week} complete!")

    def scrape_2024_regular_season(self):
        """Scrape 2024 NFL regular season"""
        print("\n" + "="*80)
        print("SCRAPING 2024 NFL REGULAR SEASON")
        print("="*80 + "\n")

        for start_date, end_date, week in self.NFL_2024_WEEKS:
            self.scrape_week_by_date(start_date, end_date, 2024, week)
            time.sleep(1)

        print("\n2024 Regular Season scrape complete!")

    def scrape_2024_playoffs(self):
        """Scrape 2024 NFL playoffs"""
        print("\n" + "="*80)
        print("SCRAPING 2024 NFL PLAYOFFS")
        print("="*80 + "\n")

        for start_date, end_date, week in self.NFL_2024_PLAYOFFS:
            self.scrape_week_by_date(start_date, end_date, 2024, 18 + week)
            time.sleep(1)

        print("\n2024 Playoffs scrape complete!")

    def scrape_2023_postseason(self):
        """Scrape 2023 NFL playoffs (we already have regular season)"""
        print("\n" + "="*80)
        print("SCRAPING 2023 NFL PLAYOFFS")
        print("="*80 + "\n")

        for start_date, end_date, week in self.NFL_2023_PLAYOFFS:
            self.scrape_week_by_date(start_date, end_date, 2023, 18 + week)
            time.sleep(1)

        print("\n2023 Playoffs scrape complete!")

    def scrape_all(self):
        """Scrape all available NFL data"""
        # First, get 2023 playoffs we're missing
        self.scrape_2023_postseason()

        # Then 2024 regular season
        self.scrape_2024_regular_season()

        # Playoffs if available
        # self.scrape_2024_playoffs()  # Not yet played

        # Print summary
        print("\n" + "="*80)
        print("NFL SCRAPE COMPLETE!")
        print("="*80 + "\n")

        conn = sqlite3.connect('nfl_games.db')
        cursor = conn.cursor()
        cursor.execute('SELECT season, COUNT(*) FROM games GROUP BY season ORDER BY season')
        print("Games by Season:")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]} games")
        conn.close()


if __name__ == '__main__':
    scraper = NFLDateScraper()
    scraper.scrape_all()
