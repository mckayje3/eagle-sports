"""
Scrape NFL 2025 season data using date ranges
"""
import requests
import time
import sqlite3
from nfl_espn_scraper import NFLESPNScraper
from database import FootballDatabase

# NFL 2025 season weeks (Thu-Mon of each week) - up to current week 12
NFL_2025_WEEKS = [
    ("2025-09-04", "2025-09-09", 1),
    ("2025-09-11", "2025-09-16", 2),
    ("2025-09-18", "2025-09-23", 3),
    ("2025-09-25", "2025-09-30", 4),
    ("2025-10-02", "2025-10-07", 5),
    ("2025-10-09", "2025-10-14", 6),
    ("2025-10-16", "2025-10-21", 7),
    ("2025-10-23", "2025-10-28", 8),
    ("2025-10-30", "2025-11-04", 9),
    ("2025-11-06", "2025-11-11", 10),
    ("2025-11-13", "2025-11-18", 11),
    ("2025-11-20", "2025-11-25", 12),  # Current week
]

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


def fetch_scoreboard_by_dates(session, start_date: str, end_date: str) -> dict:
    """Fetch games in a date range"""
    url = f"{BASE_URL}/scoreboard"
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    params = {
        'limit': '100',
        'dates': f"{start}-{end}"
    }

    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching scoreboard: {e}")
        return {}


def scrape_2025_season():
    """Scrape NFL 2025 regular season"""
    print("\n" + "="*80)
    print("SCRAPING NFL 2025 REGULAR SEASON")
    print("="*80 + "\n")

    db = FootballDatabase('nfl_games.db')
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    scraper = NFLESPNScraper('nfl_games.db')

    db.connect()
    db.initialize_schema()

    total_games = 0

    for start_date, end_date, week in NFL_2025_WEEKS:
        print(f"Scraping 2025 Week {week} ({start_date} to {end_date})...")

        scoreboard = fetch_scoreboard_by_dates(session, start_date, end_date)
        events = scoreboard.get('events', [])

        print(f"  Found {len(events)} games")
        total_games += len(events)

        for event in events:
            try:
                competitions = event.get('competitions', [{}])[0]
                competitors = competitions.get('competitors', [])

                for competitor in competitors:
                    team_data = scraper.parse_team_data(competitor.get('team', {}))
                    if team_data.get('team_id') and team_data.get('name'):
                        db.insert_or_update_team(team_data)

                game_data = scraper.parse_game_data(event, 2025)
                game_data['week'] = week

                if game_data.get('game_id'):
                    db.insert_or_update_game(game_data)

                    if game_data.get('completed'):
                        print(f"    Fetching stats for game {game_data['game_id']}...")
                        time.sleep(0.3)
                        game_details = scraper.fetch_game_details(str(game_data['game_id']))
                        box_score = game_details.get('boxscore', {})
                        teams = box_score.get('teams', [])
                        for team_stats in teams:
                            team_id = team_stats.get('team', {}).get('id')
                            stats = team_stats.get('statistics', [])
                            if team_id and stats:
                                stats_data = scraper.parse_team_stats(game_data['game_id'], int(team_id), stats)
                                db.insert_or_update_team_stats(stats_data)

            except Exception as e:
                print(f"    Error processing event: {e}")
                continue

        time.sleep(1)

    db.close()

    print("\n" + "="*80)
    print("NFL 2025 SCRAPE COMPLETE!")
    print(f"Total games scraped: {total_games}")
    print("="*80 + "\n")


if __name__ == '__main__':
    scrape_2025_season()
