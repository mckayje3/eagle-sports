"""
ESPN Game Results Updater - Updates final scores for completed games
Supports: NBA, NFL, CFB, CBB

This runs as part of daily_update.py to fill in scores for games
that have completed since they were initially added to the database.
"""
from __future__ import annotations

import requests
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class ESPNResultsUpdater:
    """Update game results from ESPN scoreboard API"""

    SPORTS_CONFIG = {
        'nba': {
            'sport': 'basketball',
            'league': 'nba',
            'db_path': 'nba_games.db',
        },
        'nfl': {
            'sport': 'football',
            'league': 'nfl',
            'db_path': 'nfl_games.db',
        },
        'cfb': {
            'sport': 'football',
            'league': 'college-football',
            'db_path': 'cfb_games.db',
        },
        'cbb': {
            'sport': 'basketball',
            'league': 'mens-college-basketball',
            'db_path': 'cbb_games.db',
        }
    }

    def __init__(self, sport: str):
        if sport.lower() not in self.SPORTS_CONFIG:
            raise ValueError(f"Unsupported sport: {sport}. Use: nba, nfl, cfb, cbb")

        self.sport = sport.lower()
        self.config = self.SPORTS_CONFIG[self.sport]
        self.base_url = f"https://site.api.espn.com/apis/site/v2/sports/{self.config['sport']}/{self.config['league']}"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_scoreboard(self, date_str: str) -> list[dict]:
        """
        Fetch scoreboard for a specific date

        Args:
            date_str: Date in YYYYMMDD format

        Returns:
            List of game dictionaries with scores
        """
        url = f"{self.base_url}/scoreboard"
        params = {'dates': date_str}

        try:
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code != 200:
                return []

            data = response.json()
            games = []

            for event in data.get('events', []):
                game_id = int(event.get('id', 0))
                competition = event.get('competitions', [{}])[0]

                # Get teams and scores
                home_team = None
                away_team = None
                for competitor in competition.get('competitors', []):
                    if competitor.get('homeAway') == 'home':
                        home_team = competitor
                    else:
                        away_team = competitor

                if not home_team or not away_team:
                    continue

                # Get scores (handle None/empty)
                home_score = home_team.get('score')
                away_score = away_team.get('score')

                if home_score is not None and home_score != '':
                    home_score = int(home_score)
                else:
                    home_score = None

                if away_score is not None and away_score != '':
                    away_score = int(away_score)
                else:
                    away_score = None

                # Check completion status
                status = competition.get('status', {}).get('type', {})
                completed = status.get('completed', False)

                # Determine winner
                winner_id = None
                if completed and home_score is not None and away_score is not None:
                    if home_score > away_score:
                        winner_id = int(home_team.get('team', {}).get('id', 0))
                    elif away_score > home_score:
                        winner_id = int(away_team.get('team', {}).get('id', 0))

                games.append({
                    'game_id': game_id,
                    'home_score': home_score,
                    'away_score': away_score,
                    'completed': 1 if completed else 0,
                    'winner_team_id': winner_id,
                })

            return games

        except Exception as e:
            logger.error(f"Error fetching scoreboard for {date_str}: {e}")
            return []

    def update_game_result(self, game_data: dict) -> bool:
        """
        Update a single game's result in the database

        Args:
            game_data: Dict with game_id, home_score, away_score, completed, winner_team_id

        Returns:
            True if game was updated
        """
        conn = sqlite3.connect(self.config['db_path'])
        cursor = conn.cursor()

        try:
            game_id = game_data['game_id']

            # Only update if game exists and we have scores
            cursor.execute('SELECT game_id, completed FROM games WHERE game_id = ?', (game_id,))
            row = cursor.fetchone()

            if not row:
                conn.close()
                return False

            # Update the game
            cursor.execute('''
                UPDATE games SET
                    home_score = COALESCE(?, home_score),
                    away_score = COALESCE(?, away_score),
                    completed = ?,
                    winner_team_id = COALESCE(?, winner_team_id),
                    updated_at = ?
                WHERE game_id = ?
            ''', (
                game_data.get('home_score'),
                game_data.get('away_score'),
                game_data.get('completed', 0),
                game_data.get('winner_team_id'),
                datetime.now().isoformat(),
                game_id
            ))

            conn.commit()
            updated = cursor.rowcount > 0
            conn.close()
            return updated

        except Exception as e:
            logger.error(f"Error updating game {game_data.get('game_id')}: {e}")
            conn.close()
            return False

    def sync_result_to_cache(self, game_id: int, home_score: int, away_score: int) -> bool:
        """
        Sync actual scores to prediction_cache in users.db

        Args:
            game_id: Game ID
            home_score: Final home score
            away_score: Final away score

        Returns:
            True if updated
        """
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()

            sport_name = self.sport.upper()

            cursor.execute('''
                UPDATE prediction_cache
                SET actual_home_score = ?,
                    actual_away_score = ?
                WHERE game_id = ? AND sport = ?
            ''', (home_score, away_score, game_id, sport_name))

            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return updated

        except Exception:
            return False

    def update_recent(self, days: int = 7) -> tuple[int, int]:
        """
        Update results for games in the past N days

        Args:
            days: Number of days to look back

        Returns:
            Tuple of (games_found, games_updated)
        """
        total_found = 0
        total_updated = 0

        # Iterate through each day
        for i in range(days + 1):  # +1 to include today
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%Y%m%d')

            games = self.fetch_scoreboard(date_str)

            for game in games:
                total_found += 1

                # Only update completed games with scores
                if game.get('completed') and game.get('home_score') is not None:
                    if self.update_game_result(game):
                        total_updated += 1

                        # Sync to prediction cache
                        self.sync_result_to_cache(
                            game['game_id'],
                            game['home_score'],
                            game['away_score']
                        )

        return total_found, total_updated



def update_results(sport: str, days: int = 7) -> tuple[bool, int]:
    """
    Main entry point for updating results

    Args:
        sport: One of 'nba', 'nfl', 'cfb', 'cbb'
        days: Number of days to look back

    Returns:
        Tuple of (success, games_updated)
    """
    try:
        updater = ESPNResultsUpdater(sport)
        found, updated = updater.update_recent(days)

        logger.info(f"  Results: {updated}/{found} games updated")
        return True, updated

    except Exception as e:
        logger.error(f"Error updating {sport.upper()} results: {e}")
        return False, 0


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='ESPN Game Results Updater')
    parser.add_argument('sport', choices=['nba', 'nfl', 'cfb', 'cbb', 'all'],
                        help='Sport to update (or "all")')
    parser.add_argument('--days', type=int, default=7,
                        help='Days to look back (default: 7)')

    args = parser.parse_args()

    print("=" * 60)
    print("ESPN GAME RESULTS UPDATER")
    print("=" * 60)

    if args.sport == 'all':
        sports = ['nba', 'nfl', 'cfb', 'cbb']
    else:
        sports = [args.sport]

    for sport in sports:
        print(f"\nUpdating {sport.upper()} results...")
        success, count = update_results(sport, args.days)
        status = "OK" if success else "FAILED"
        print(f"  {sport.upper()}: {status} ({count} games)")
