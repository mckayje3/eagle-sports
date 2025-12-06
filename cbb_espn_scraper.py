"""
CBB ESPN Scraper - Fetch Men's College Basketball game data from ESPN API
Supports scraping by date range for full season coverage
"""
import requests
import sqlite3
from datetime import datetime, timedelta
import time
from cbb_database import CBBDatabase, create_cbb_database


class CBBESPNScraper:
    """Scrape Men's College Basketball data from ESPN API"""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
    CORE_URL = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"

    def __init__(self, db_path='cbb_games.db'):
        self.db_path = db_path
        self.db = CBBDatabase(db_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_teams(self, limit=500):
        """Fetch all D1 college basketball teams"""
        url = f"{self.BASE_URL}/teams?limit={limit}&groups=50"  # groups=50 is D1
        response = self.session.get(url, timeout=30)

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
                'display_name': team_info.get('displayName', ''),
                'color': team_info.get('color', ''),
                'logo_url': team_info.get('logos', [{}])[0].get('href', '') if team_info.get('logos') else '',
                'conference': team_info.get('standingSummary', ''),
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
                display_name=team['display_name'],
                logo_url=team.get('logo_url'),
                color=team.get('color'),
                conference=team.get('conference')
            )
        self.db.close()
        print(f"Saved {len(teams)} teams")

    def fetch_scoreboard(self, date_str):
        """Fetch scoreboard for a specific date (YYYYMMDD format)"""
        url = f"{self.BASE_URL}/scoreboard?dates={date_str}&limit=200&groups=50"
        response = self.session.get(url, timeout=30)

        if response.status_code != 200:
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

            # Check if neutral site or conference game
            neutral_site = competition.get('neutralSite', False)
            conference_competition = competition.get('conferenceCompetition', False)

            games.append({
                'game_id': game_id,
                'date': event.get('date', ''),
                'home_team_id': int(home_team.get('team', {}).get('id', 0)),
                'away_team_id': int(away_team.get('team', {}).get('id', 0)),
                'home_score': home_score,
                'away_score': away_score,
                'completed': 1 if completed else 0,
                'winner_team_id': winner_id,
                'venue_name': venue.get('fullName', ''),
                'venue_city': venue.get('address', {}).get('city', ''),
                'venue_state': venue.get('address', {}).get('state', ''),
                'neutral_site': 1 if neutral_site else 0,
                'conference_game': 1 if conference_competition else 0,
                'season_type': event.get('season', {}).get('type', 2),
                'attendance': competition.get('attendance')
            })

        return games

    def fetch_game_stats(self, game_id):
        """Fetch detailed box score stats for a game"""
        url = f"{self.BASE_URL}/summary?event={game_id}"

        try:
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return None

            data = response.json()
            boxscore = data.get('boxscore', {})
            teams_data = boxscore.get('teams', [])

            if len(teams_data) != 2:
                return None

            result = {}

            for team_data in teams_data:
                team_info = team_data.get('team', {})
                team_id = int(team_info.get('id', 0))
                stats_list = team_data.get('statistics', [])

                # Parse statistics into dict
                stats = {}
                stat_mapping = {
                    'fieldGoalsMade-fieldGoalsAttempted': ('field_goals_made', 'field_goals_attempted'),
                    'fieldGoalPct': 'field_goal_pct',
                    'threePointFieldGoalsMade-threePointFieldGoalsAttempted': ('three_pointers_made', 'three_pointers_attempted'),
                    'threePointFieldGoalPct': 'three_point_pct',
                    'freeThrowsMade-freeThrowsAttempted': ('free_throws_made', 'free_throws_attempted'),
                    'freeThrowPct': 'free_throw_pct',
                    'totalRebounds': 'total_rebounds',
                    'offensiveRebounds': 'offensive_rebounds',
                    'defensiveRebounds': 'defensive_rebounds',
                    'assists': 'assists',
                    'steals': 'steals',
                    'blocks': 'blocks',
                    'turnovers': 'turnovers',
                    'fouls': 'personal_fouls',
                    'technicalFouls': 'technical_fouls',
                    'points': 'points',
                    'largestLead': 'largest_lead',
                }

                for stat_item in stats_list:
                    name = stat_item.get('name', '')
                    display_value = stat_item.get('displayValue', '')

                    if name in stat_mapping:
                        mapping = stat_mapping[name]

                        if isinstance(mapping, tuple):
                            # Split value like "25-70"
                            parts = display_value.split('-')
                            if len(parts) == 2:
                                try:
                                    stats[mapping[0]] = int(parts[0])
                                    stats[mapping[1]] = int(parts[1])
                                except ValueError:
                                    pass
                        else:
                            try:
                                if '.' in display_value:
                                    stats[mapping] = float(display_value)
                                else:
                                    stats[mapping] = int(display_value)
                            except ValueError:
                                pass

                result[team_id] = stats

            return result

        except Exception as e:
            print(f"Error fetching stats for game {game_id}: {e}")
            return None

    def scrape_season(self, season_year, start_month=11, end_month=4):
        """
        Scrape an entire CBB season

        Args:
            season_year: The calendar year for the season (e.g., 2024 for 2023-24 season)
                        CBB uses the ending year of the season
            start_month: Month to start scraping (default Nov)
            end_month: Month to end scraping (default April)
        """
        print(f"\n{'='*80}")
        print(f"SCRAPING CBB {season_year-1}-{str(season_year)[-2:]} SEASON")
        print(f"{'='*80}\n")

        # First fetch and save teams
        teams = self.fetch_teams()
        if teams:
            self.save_teams(teams)

        self.db.connect()

        # CBB season runs Nov-Apr (roughly)
        # Start from previous year's November
        start_date = datetime(season_year - 1, start_month, 1)
        end_date = datetime(season_year, end_month, 30)

        # If we're scraping current season, don't go past today
        today = datetime.now()
        if end_date > today:
            end_date = today

        current_date = start_date
        total_games = 0
        dates_processed = 0

        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            games = self.fetch_scoreboard(date_str)

            if games:
                for game in games:
                    game['season'] = season_year
                    self.db.insert_game(game)
                    total_games += 1

                if len(games) > 0:
                    print(f"{current_date.strftime('%Y-%m-%d')}: {len(games)} games")

            dates_processed += 1
            if dates_processed % 30 == 0:
                print(f"  ... processed {dates_processed} days, {total_games} games so far")

            current_date += timedelta(days=1)
            time.sleep(0.15)  # Rate limiting

        self.db.close()

        print(f"\n{'='*80}")
        print(f"SEASON SCRAPE COMPLETE")
        print(f"Total games saved: {total_games}")
        print(f"{'='*80}")

        return total_games

    def backfill_stats(self, season, batch_size=50):
        """Backfill box score stats for completed games"""
        print(f"\n{'='*80}")
        print(f"BACKFILLING CBB {season} STATS")
        print(f"{'='*80}\n")

        self.db.connect()
        games_needing_stats = self.db.get_games_needing_stats(season)
        print(f"Found {len(games_needing_stats)} games needing stats")

        total_saved = 0

        for i, game_id in enumerate(games_needing_stats):
            stats = self.fetch_game_stats(game_id)

            if stats:
                for team_id, team_stats in stats.items():
                    self.db.insert_team_stats(game_id, team_id, team_stats)
                total_saved += 1

            if (i + 1) % batch_size == 0:
                print(f"  Processed {i + 1}/{len(games_needing_stats)} games, saved {total_saved}")

            time.sleep(0.15)

        self.db.close()

        print(f"\nStats backfill complete: {total_saved} games updated")
        return total_saved


def main():
    import sys

    # Create database if it doesn't exist
    create_cbb_database()

    scraper = CBBESPNScraper()

    if len(sys.argv) > 1:
        action = sys.argv[1]

        if action == 'teams':
            teams = scraper.fetch_teams()
            scraper.save_teams(teams)

        elif action == 'season' and len(sys.argv) > 2:
            season = int(sys.argv[2])
            scraper.scrape_season(season)

        elif action == 'stats' and len(sys.argv) > 2:
            season = int(sys.argv[2])
            scraper.backfill_stats(season)

        elif action == 'today':
            # Scrape today's games
            scraper.db.connect()
            date_str = datetime.now().strftime('%Y%m%d')
            games = scraper.fetch_scoreboard(date_str)
            for game in games:
                game['season'] = 2025  # Current season
                scraper.db.insert_game(game)
            scraper.db.close()
            print(f"Saved {len(games)} games for today")

        else:
            print("Usage: python cbb_espn_scraper.py [teams|season YEAR|stats YEAR|today]")
    else:
        # Default: scrape 2024 (2023-24) and 2025 (2024-25) seasons
        print("Scraping CBB 2023-24 and 2024-25 seasons...")
        scraper.scrape_season(2024)
        scraper.scrape_season(2025)


if __name__ == '__main__':
    main()
