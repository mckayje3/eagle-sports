"""
Scrape CFB data using date ranges (required for historical data)
ESPN ignores 'year' parameter for CFB week-based queries
"""
import requests
import time
import sqlite3
from datetime import datetime, timedelta
from cfb_espn_scraper import ESPNScraper
from cfb_nfl_database import FootballDatabase

class CFBDateScraper:
    """Scrape CFB using date ranges for historical data"""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

    # CFB 2024 season weeks (start dates)
    CFB_2024_WEEKS = [
        ("2024-08-24", "2024-08-26", 0),   # Week 0
        ("2024-08-29", "2024-09-02", 1),   # Week 1
        ("2024-09-05", "2024-09-09", 2),   # Week 2
        ("2024-09-12", "2024-09-16", 3),   # Week 3
        ("2024-09-19", "2024-09-23", 4),   # Week 4
        ("2024-09-26", "2024-09-30", 5),   # Week 5
        ("2024-10-03", "2024-10-07", 6),   # Week 6
        ("2024-10-10", "2024-10-14", 7),   # Week 7
        ("2024-10-17", "2024-10-21", 8),   # Week 8
        ("2024-10-24", "2024-10-28", 9),   # Week 9
        ("2024-10-31", "2024-11-04", 10),  # Week 10
        ("2024-11-07", "2024-11-11", 11),  # Week 11
        ("2024-11-14", "2024-11-18", 12),  # Week 12
        ("2024-11-21", "2024-11-25", 13),  # Week 13 (Rivalry Week)
        ("2024-11-28", "2024-12-02", 14),  # Week 14 (Championship Week)
        ("2024-12-05", "2024-12-09", 15),  # Week 15 (Conference Championships)
    ]

    # Bowl Season 2024-2025
    CFB_2024_BOWLS = [
        ("2024-12-14", "2024-12-22", 1),   # Early bowls
        ("2024-12-23", "2024-12-29", 2),   # Mid bowls
        ("2024-12-30", "2025-01-02", 3),   # New Year's bowls
        ("2025-01-09", "2025-01-11", 4),   # CFP Semifinals
        ("2025-01-20", "2025-01-21", 5),   # CFP Championship
    ]

    def __init__(self, db_path: str = 'cfb_games.db'):
        self.db = FootballDatabase(db_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.scraper = ESPNScraper(db_path)  # Reuse parsing methods

    def fetch_scoreboard_by_dates(self, start_date: str, end_date: str) -> dict:
        """
        Fetch games in a date range

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        url = f"{self.BASE_URL}/scoreboard"

        # Convert to ESPN format (YYYYMMDD-YYYYMMDD)
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        params = {
            'groups': '80',  # FBS
            'limit': '200',
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

        for event in events:
            try:
                # Parse and save teams
                competitions = event.get('competitions', [{}])[0]
                competitors = competitions.get('competitors', [])

                for competitor in competitors:
                    team_data = self.scraper.parse_team_data(competitor.get('team', {}))
                    if team_data.get('team_id') and team_data.get('name'):
                        self.db.insert_or_update_team(team_data)

                # Parse and save game (override season with our value)
                game_data = self.scraper.parse_game_data(event, season)
                game_data['week'] = week  # Override week with our value

                if game_data.get('game_id'):
                    self.db.insert_or_update_game(game_data)

                    # Fetch detailed stats if completed
                    if game_data.get('completed'):
                        print(f"    Fetching stats for game {game_data['game_id']}...")
                        time.sleep(0.3)
                        game_details = self.scraper.fetch_game_details(str(game_data['game_id']))
                        # Process stats using our db connection
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
        """Scrape complete 2024 regular season"""
        print("\n" + "="*80)
        print("SCRAPING 2024 CFB REGULAR SEASON (Using Date Ranges)")
        print("="*80 + "\n")

        for start_date, end_date, week in self.CFB_2024_WEEKS:
            self.scrape_week_by_date(start_date, end_date, 2024, week)
            time.sleep(1)  # Rate limiting

        print("\n2024 Regular Season scrape complete!")

    def scrape_2024_bowls(self):
        """Scrape 2024-2025 bowl season"""
        print("\n" + "="*80)
        print("SCRAPING 2024-2025 CFB BOWL SEASON")
        print("="*80 + "\n")

        for start_date, end_date, week in self.CFB_2024_BOWLS:
            self.scrape_week_by_date(start_date, end_date, 2024, 15 + week)  # Weeks 16-20 for bowls
            time.sleep(1)

        print("\n2024-2025 Bowl Season scrape complete!")

    def scrape_full_2024(self):
        """Scrape complete 2024 season including bowls"""
        self.scrape_2024_regular_season()
        self.scrape_2024_bowls()

        # Print summary
        print("\n" + "="*80)
        print("2024 SEASON SCRAPE COMPLETE!")
        print("="*80 + "\n")

        conn = sqlite3.connect('cfb_games.db')
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM games WHERE season=2024')
        total = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM games WHERE season=2024 AND completed=1')
        completed = cursor.fetchone()[0]

        conn.close()

        print(f"Total 2024 games: {total}")
        print(f"Completed games: {completed}")


if __name__ == '__main__':
    scraper = CFBDateScraper()
    scraper.scrape_full_2024()
