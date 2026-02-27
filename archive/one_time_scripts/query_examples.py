"""
Example queries for the CFB database
Provides helper functions to easily query common data
"""
import sqlite3
from typing import List, Dict, Optional
import json


class CFBQuery:
    def __init__(self, db_path: str = 'cfb_games.db'):
        self.db_path = db_path

    def _execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a query and return results as list of dictionaries"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_team_games(self, team_name: str, season: Optional[int] = None) -> List[Dict]:
        """Get all games for a specific team"""
        query = '''
            SELECT g.game_id, g.date, g.week, g.season,
                   t1.name as home_team,
                   t2.name as away_team,
                   g.home_score,
                   g.away_score,
                   t3.name as winner,
                   g.venue_name,
                   g.broadcast_network
            FROM games g
            JOIN teams t1 ON g.home_team_id = t1.team_id
            JOIN teams t2 ON g.away_team_id = t2.team_id
            LEFT JOIN teams t3 ON g.winner_team_id = t3.team_id
            WHERE (t1.name LIKE ? OR t2.name LIKE ?)
        '''
        params = [f'%{team_name}%', f'%{team_name}%']

        if season:
            query += ' AND g.season = ?'
            params.append(season)

        query += ' ORDER BY g.date'

        return self._execute_query(query, tuple(params))

    def get_team_stats(self, team_name: str, season: Optional[int] = None) -> List[Dict]:
        """Get aggregated statistics for a team"""
        query = '''
            SELECT t.name,
                   COUNT(DISTINCT g.game_id) as games_played,
                   AVG(s.points) as avg_points,
                   AVG(s.total_yards) as avg_yards,
                   AVG(s.passing_yards) as avg_passing_yards,
                   AVG(s.rushing_yards) as avg_rushing_yards,
                   AVG(s.turnovers) as avg_turnovers,
                   AVG(s.first_downs) as avg_first_downs,
                   SUM(CASE WHEN g.winner_team_id = t.team_id THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN g.winner_team_id IS NOT NULL
                            AND g.winner_team_id != t.team_id
                            AND (g.home_team_id = t.team_id OR g.away_team_id = t.team_id)
                       THEN 1 ELSE 0 END) as losses
            FROM teams t
            JOIN team_game_stats s ON t.team_id = s.team_id
            JOIN games g ON s.game_id = g.game_id
            WHERE t.name LIKE ? AND g.completed = 1
        '''
        params = [f'%{team_name}%']

        if season:
            query += ' AND g.season = ?'
            params.append(season)

        query += ' GROUP BY t.team_id'

        return self._execute_query(query, tuple(params))

    def get_game_details(self, game_id: int) -> Dict:
        """Get detailed information for a specific game"""
        game_query = '''
            SELECT g.*,
                   t1.name as home_team_name,
                   t2.name as away_team_name,
                   t3.name as winner_name
            FROM games g
            JOIN teams t1 ON g.home_team_id = t1.team_id
            JOIN teams t2 ON g.away_team_id = t2.team_id
            LEFT JOIN teams t3 ON g.winner_team_id = t3.team_id
            WHERE g.game_id = ?
        '''

        stats_query = '''
            SELECT t.name as team_name, s.*
            FROM team_game_stats s
            JOIN teams t ON s.team_id = t.team_id
            WHERE s.game_id = ?
        '''

        game = self._execute_query(game_query, (game_id,))
        stats = self._execute_query(stats_query, (game_id,))

        if game:
            result = game[0]
            result['team_stats'] = stats
            return result
        return {}

    def get_highest_scoring_games(self, season: int, limit: int = 10) -> List[Dict]:
        """Get highest scoring games for a season"""
        query = '''
            SELECT g.game_id, g.date, g.week,
                   t1.name as home_team,
                   t2.name as away_team,
                   g.home_score,
                   g.away_score,
                   (g.home_score + g.away_score) as total_points,
                   g.venue_name
            FROM games g
            JOIN teams t1 ON g.home_team_id = t1.team_id
            JOIN teams t2 ON g.away_team_id = t2.team_id
            WHERE g.season = ? AND g.completed = 1
            ORDER BY total_points DESC
            LIMIT ?
        '''
        return self._execute_query(query, (season, limit))

    def get_conference_standings(self, conference: str, season: int) -> List[Dict]:
        """Get standings for a specific conference"""
        query = '''
            SELECT t.name,
                   COUNT(DISTINCT CASE WHEN g.completed = 1 THEN g.game_id END) as games_played,
                   SUM(CASE WHEN g.winner_team_id = t.team_id THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN g.winner_team_id IS NOT NULL
                            AND g.winner_team_id != t.team_id
                            AND (g.home_team_id = t.team_id OR g.away_team_id = t.team_id)
                       THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN g.winner_team_id = t.team_id AND g.conference_game = 1
                       THEN 1 ELSE 0 END) as conf_wins,
                   SUM(CASE WHEN g.winner_team_id != t.team_id AND g.conference_game = 1
                            AND g.completed = 1
                       THEN 1 ELSE 0 END) as conf_losses
            FROM teams t
            LEFT JOIN games g ON (g.home_team_id = t.team_id OR g.away_team_id = t.team_id)
            WHERE t.conference = ? AND (g.season = ? OR g.season IS NULL)
            GROUP BY t.team_id
            ORDER BY wins DESC, losses ASC
        '''
        return self._execute_query(query, (conference, season))

    def get_teams_by_offensive_yards(self, season: int, limit: int = 25) -> List[Dict]:
        """Get top teams by average offensive yards"""
        query = '''
            SELECT t.name,
                   COUNT(DISTINCT g.game_id) as games,
                   ROUND(AVG(s.total_yards), 1) as avg_total_yards,
                   ROUND(AVG(s.passing_yards), 1) as avg_passing_yards,
                   ROUND(AVG(s.rushing_yards), 1) as avg_rushing_yards
            FROM teams t
            JOIN team_game_stats s ON t.team_id = s.team_id
            JOIN games g ON s.game_id = g.game_id
            WHERE g.season = ? AND g.completed = 1
            GROUP BY t.team_id
            HAVING games >= 5
            ORDER BY avg_total_yards DESC
            LIMIT ?
        '''
        return self._execute_query(query, (season, limit))

    def get_teams_by_defense(self, season: int, limit: int = 25) -> List[Dict]:
        """Get top defensive teams (least yards allowed)"""
        query = '''
            SELECT t.name,
                   COUNT(DISTINCT g.game_id) as games,
                   ROUND(AVG(opp_stats.total_yards), 1) as avg_yards_allowed,
                   ROUND(AVG(opp_stats.passing_yards), 1) as avg_passing_allowed,
                   ROUND(AVG(opp_stats.rushing_yards), 1) as avg_rushing_allowed,
                   ROUND(AVG(opp_stats.points), 1) as avg_points_allowed
            FROM teams t
            JOIN games g ON (g.home_team_id = t.team_id OR g.away_team_id = t.team_id)
            JOIN team_game_stats opp_stats ON (
                opp_stats.game_id = g.game_id
                AND opp_stats.team_id != t.team_id
            )
            WHERE g.season = ? AND g.completed = 1
            GROUP BY t.team_id
            HAVING games >= 5
            ORDER BY avg_yards_allowed ASC
            LIMIT ?
        '''
        return self._execute_query(query, (season, limit))

    def get_week_schedule(self, season: int, week: int) -> List[Dict]:
        """Get all games for a specific week"""
        query = '''
            SELECT g.game_id, g.date,
                   t1.name as home_team,
                   t2.name as away_team,
                   g.home_score,
                   g.away_score,
                   g.completed,
                   g.venue_name,
                   g.broadcast_network
            FROM games g
            JOIN teams t1 ON g.home_team_id = t1.team_id
            JOIN teams t2 ON g.away_team_id = t2.team_id
            WHERE g.season = ? AND g.week = ?
            ORDER BY g.date
        '''
        return self._execute_query(query, (season, week))

    def search_teams(self, search_term: str) -> List[Dict]:
        """Search for teams by name"""
        query = '''
            SELECT team_id, name, display_name, abbreviation, conference
            FROM teams
            WHERE name LIKE ? OR display_name LIKE ? OR abbreviation LIKE ?
            ORDER BY name
        '''
        term = f'%{search_term}%'
        return self._execute_query(query, (term, term, term))

    def get_game_odds(self, game_id: int) -> List[Dict]:
        """Get betting odds for a specific game"""
        query = '''
            SELECT o.*, g.date,
                   t1.name as home_team,
                   t2.name as away_team
            FROM game_odds o
            JOIN games g ON o.game_id = g.game_id
            JOIN teams t1 ON g.home_team_id = t1.team_id
            JOIN teams t2 ON g.away_team_id = t2.team_id
            WHERE o.game_id = ?
        '''
        return self._execute_query(query, (game_id,))

    def get_odds_for_week(self, season: int, week: int) -> List[Dict]:
        """Get betting odds for all games in a week"""
        query = '''
            SELECT g.game_id, g.date, g.week,
                   t1.name as home_team,
                   t2.name as away_team,
                   o.current_spread_home,
                   o.current_moneyline_home,
                   o.current_moneyline_away,
                   o.current_total,
                   o.source
            FROM games g
            JOIN teams t1 ON g.home_team_id = t1.team_id
            JOIN teams t2 ON g.away_team_id = t2.team_id
            LEFT JOIN game_odds o ON g.game_id = o.game_id
            WHERE g.season = ? AND g.week = ?
            ORDER BY g.date
        '''
        return self._execute_query(query, (season, week))

    def get_odds_movement_for_game(self, game_id: int) -> List[Dict]:
        """Get odds movement history for a game"""
        query = '''
            SELECT om.*,
                   t1.name as home_team,
                   t2.name as away_team
            FROM odds_movement om
            JOIN games g ON om.game_id = g.game_id
            JOIN teams t1 ON g.home_team_id = t1.team_id
            JOIN teams t2 ON g.away_team_id = t2.team_id
            WHERE om.game_id = ?
            ORDER BY om.timestamp
        '''
        return self._execute_query(query, (game_id,))

    def get_betting_trends(self, season: int, team_name: str) -> List[Dict]:
        """Get betting performance for a team (ATS record)"""
        query = '''
            SELECT g.game_id, g.date, g.week,
                   CASE
                       WHEN g.home_team_id = t.team_id THEN 'Home'
                       ELSE 'Away'
                   END as location,
                   CASE
                       WHEN g.home_team_id = t.team_id THEN t2.name
                       ELSE t1.name
                   END as opponent,
                   CASE
                       WHEN g.home_team_id = t.team_id THEN g.home_score
                       ELSE g.away_score
                   END as team_score,
                   CASE
                       WHEN g.home_team_id = t.team_id THEN g.away_score
                       ELSE g.home_score
                   END as opp_score,
                   CASE
                       WHEN g.home_team_id = t.team_id THEN o.current_spread_home
                       ELSE o.current_spread_away
                   END as spread,
                   CASE
                       WHEN g.home_team_id = t.team_id THEN o.current_moneyline_home
                       ELSE o.current_moneyline_away
                   END as moneyline
            FROM teams t
            JOIN games g ON (g.home_team_id = t.team_id OR g.away_team_id = t.team_id)
            JOIN teams t1 ON g.home_team_id = t1.team_id
            JOIN teams t2 ON g.away_team_id = t2.team_id
            LEFT JOIN game_odds o ON g.game_id = o.game_id
            WHERE t.name LIKE ? AND g.season = ? AND g.completed = 1
            ORDER BY g.date
        '''
        return self._execute_query(query, (f'%{team_name}%', season))

    def get_line_movements(self, season: int, week: int) -> List[Dict]:
        """Get opening vs closing line movements for a week"""
        query = '''
            SELECT g.game_id,
                   t1.name as home_team,
                   t2.name as away_team,
                   o.opening_spread_home,
                   o.closing_spread_home,
                   (o.closing_spread_home - o.opening_spread_home) as spread_movement,
                   o.opening_total,
                   o.closing_total,
                   (o.closing_total - o.opening_total) as total_movement
            FROM games g
            JOIN teams t1 ON g.home_team_id = t1.team_id
            JOIN teams t2 ON g.away_team_id = t2.team_id
            JOIN game_odds o ON g.game_id = o.game_id
            WHERE g.season = ? AND g.week = ?
              AND o.opening_spread_home IS NOT NULL
              AND o.closing_spread_home IS NOT NULL
            ORDER BY ABS(o.closing_spread_home - o.opening_spread_home) DESC
        '''
        return self._execute_query(query, (season, week))

    def get_dogs_and_favorites(self, season: int, week: int) -> List[Dict]:
        """Get underdogs and favorites for a week"""
        query = '''
            SELECT g.game_id, g.date,
                   t1.name as home_team,
                   t2.name as away_team,
                   o.current_spread_home,
                   CASE
                       WHEN o.current_spread_home < 0 THEN t1.name
                       ELSE t2.name
                   END as favorite,
                   CASE
                       WHEN o.current_spread_home > 0 THEN t1.name
                       ELSE t2.name
                   END as underdog,
                   ABS(o.current_spread_home) as spread_size
            FROM games g
            JOIN teams t1 ON g.home_team_id = t1.team_id
            JOIN teams t2 ON g.away_team_id = t2.team_id
            LEFT JOIN game_odds o ON g.game_id = o.game_id
            WHERE g.season = ? AND g.week = ? AND o.current_spread_home IS NOT NULL
            ORDER BY ABS(o.current_spread_home) DESC
        '''
        return self._execute_query(query, (season, week))


def print_results(results: List[Dict], title: str = "Results"):
    """Pretty print query results"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

    if not results:
        print("No results found.")
        return

    # Print as JSON for easy reading
    for result in results:
        print(json.dumps(result, indent=2, default=str))
        print("-" * 80)

    print(f"\nTotal results: {len(results)}\n")


if __name__ == '__main__':
    # Example usage
    query = CFBQuery()

    print("CFB Database Query Examples")
    print("===========================\n")

    # Example 1: Search for a team
    print("Example 1: Search for teams")
    teams = query.search_teams("Michigan")
    print_results(teams, "Teams matching 'Michigan'")

    # Example 2: Get team games
    # print("Example 2: Get team games")
    # games = query.get_team_games("Alabama", season=2024)
    # print_results(games, "Alabama 2024 Games")

    # Example 3: Get team statistics
    # print("Example 3: Get team statistics")
    # stats = query.get_team_stats("Georgia", season=2024)
    # print_results(stats, "Georgia 2024 Statistics")

    # Example 4: Get highest scoring games
    # print("Example 4: Highest scoring games")
    # high_scoring = query.get_highest_scoring_games(2024, limit=5)
    # print_results(high_scoring, "Top 5 Highest Scoring Games 2024")

    # Example 5: Get top offensive teams
    # print("Example 5: Top offensive teams")
    # offense = query.get_teams_by_offensive_yards(2024, limit=10)
    # print_results(offense, "Top 10 Offensive Teams 2024")

    # Example 6: Get week schedule
    # print("Example 6: Week schedule")
    # schedule = query.get_week_schedule(2024, week=12)
    # print_results(schedule, "Week 12 Schedule 2024")

    print("\nTo use these queries, uncomment the examples above or import CFBQuery class")
    print("Example: from query_examples import CFBQuery")
