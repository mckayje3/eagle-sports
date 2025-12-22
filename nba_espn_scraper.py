"""
NBA ESPN Scraper - Fetch game data from ESPN API
Similar to NFL/CFB scrapers
"""
import requests
import sqlite3
from datetime import datetime, timedelta
import time
from nba_database import NBADatabase, create_nba_database
from timezone_utils import convert_espn_date


class NBAESPNScraper:
    """Scrape NBA data from ESPN API"""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

    def __init__(self, db_path='nba_games.db'):
        self.db_path = db_path
        self.db = NBADatabase(db_path)

    def fetch_teams(self):
        """Fetch all NBA teams"""
        url = f"{self.BASE_URL}/teams"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error fetching teams: {response.status_code}")
            return []

        data = response.json()
        teams = []

        for team in data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []):
            team_info = team.get('team', {})
            teams.append({
                'team_id': int(team_info.get('id', 0)),
                'name': team_info.get('displayName', ''),
                'abbreviation': team_info.get('abbreviation', ''),
                'location': team_info.get('location', ''),
                'color': team_info.get('color', ''),
                'logo_url': team_info.get('logos', [{}])[0].get('href', '') if team_info.get('logos') else ''
            })

        return teams

    def save_teams(self, teams):
        """Save teams to database"""
        self.db.connect()
        for team in teams:
            self.db.insert_team(
                team_id=team['team_id'],
                name=team['name'],
                abbreviation=team['abbreviation'],
                display_name=team['name']
            )
        self.db.close()
        print(f"Saved {len(teams)} teams")

    def fetch_scoreboard(self, date_str):
        """Fetch scoreboard for a specific date (YYYYMMDD format)"""
        url = f"{self.BASE_URL}/scoreboard?dates={date_str}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error fetching scoreboard for {date_str}: {response.status_code}")
            return []

        data = response.json()
        games = []

        for event in data.get('events', []):
            game_id = int(event.get('id', 0))
            competition = event.get('competitions', [{}])[0]

            # Get teams
            home_team = None
            away_team = None
            for competitor in competition.get('competitors', []):
                if competitor.get('homeAway') == 'home':
                    home_team = competitor
                else:
                    away_team = competitor

            if not home_team or not away_team:
                continue

            home_score = int(home_team.get('score', 0)) if home_team.get('score') else None
            away_score = int(away_team.get('score', 0)) if away_team.get('score') else None

            completed = competition.get('status', {}).get('type', {}).get('completed', False)

            # Determine winner
            winner_id = None
            if completed and home_score is not None and away_score is not None:
                if home_score > away_score:
                    winner_id = int(home_team.get('team', {}).get('id', 0))
                else:
                    winner_id = int(away_team.get('team', {}).get('id', 0))

            # Get venue
            venue = competition.get('venue', {})

            # Keep raw UTC date - let database derive game_date_eastern
            date_str_utc = event.get('date', '')

            games.append({
                'game_id': game_id,
                'date': date_str_utc,  # Keep UTC for proper timezone conversion
                'home_team_id': int(home_team.get('team', {}).get('id', 0)),
                'away_team_id': int(away_team.get('team', {}).get('id', 0)),
                'home_score': home_score,
                'away_score': away_score,
                'completed': 1 if completed else 0,
                'winner_team_id': winner_id,
                'venue_name': venue.get('fullName', ''),
                'venue_city': venue.get('address', {}).get('city', ''),
                'venue_state': venue.get('address', {}).get('state', ''),
                'season_type': event.get('season', {}).get('type', 2)
            })

        return games

    def fetch_game_details(self, game_id):
        """Fetch detailed stats for a game"""
        url = f"{self.BASE_URL}/summary?event={game_id}"
        response = requests.get(url)

        if response.status_code != 200:
            return None

        return response.json()

    def scrape_season(self, season_year, start_month=10, end_month=6):
        """
        Scrape an entire NBA season
        season_year: The year the season starts (e.g., 2023 for 2023-24 season)
        """
        print(f"\n{'='*80}")
        print(f"SCRAPING NBA {season_year}-{str(season_year+1)[-2:]} SEASON")
        print(f"{'='*80}\n")

        # First fetch and save teams
        teams = self.fetch_teams()
        if teams:
            self.save_teams(teams)

        self.db.connect()

        # Generate date range for the season
        # NBA season typically runs Oct-Jun
        if start_month >= 10:
            start_date = datetime(season_year, start_month, 1)
        else:
            start_date = datetime(season_year + 1, start_month, 1)

        if end_month <= 6:
            end_date = datetime(season_year + 1, end_month, 30)
        else:
            end_date = datetime(season_year, end_month, 30)

        current_date = start_date
        total_games = 0

        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            games = self.fetch_scoreboard(date_str)

            if games:
                print(f"{current_date.strftime('%Y-%m-%d')}: {len(games)} games")
                for game in games:
                    game['season'] = season_year
                    self.db.insert_game(game)
                    total_games += 1

            current_date += timedelta(days=1)
            time.sleep(0.2)  # Rate limiting

        self.db.close()

        print(f"\n{'='*80}")
        print(f"SEASON SCRAPE COMPLETE")
        print(f"Total games saved: {total_games}")
        print(f"{'='*80}")

        return total_games


def main():
    import sys

    # Create database if it doesn't exist
    create_nba_database()

    scraper = NBAESPNScraper()

    if len(sys.argv) > 1:
        season = int(sys.argv[1])
        scraper.scrape_season(season)
    else:
        # Default: scrape 2023-24 and 2024-25 seasons
        scraper.scrape_season(2023)
        scraper.scrape_season(2024)


if __name__ == '__main__':
    main()
