"""
VegasInsider Odds Scraper for College Football
Scrapes betting odds including spreads, moneylines, and totals
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from database import FootballDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VegasInsiderOddsScraper:
    BASE_URL = "https://www.vegasinsider.com"

    def __init__(self, db_path: str = 'cfb_games.db'):
        self.db = FootballDatabase(db_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

    def fetch_current_odds(self, odds_type: str = 'las-vegas') -> Optional[pd.DataFrame]:
        """
        Fetch current college football odds from VegasInsider

        Args:
            odds_type: Type of odds ('las-vegas', 'offshore', etc.)

        Returns:
            DataFrame with odds data or None if error
        """
        url = f"{self.BASE_URL}/college-football/odds/{odds_type}/"

        try:
            logger.info(f"Fetching odds from: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Replace <br> tags with newlines for better parsing
            for br in soup.select("br"):
                br.replace_with("\n")

            # Try to find the odds table
            # VegasInsider uses various class names, try common ones
            table_classes = ['.frodds-data-tbl', 'table.frodds-data-tbl', '.main-table',
                           'table[class*="odds"]', 'table[class*="frodds"]']

            df = None
            for table_class in table_classes:
                try:
                    table = soup.select_one(table_class)
                    if table:
                        df = pd.read_html(str(table))[0]
                        logger.info(f"Successfully parsed table with class: {table_class}")
                        break
                except Exception as e:
                    logger.debug(f"Failed to parse with class {table_class}: {e}")
                    continue

            # If specific class search didn't work, try getting all tables
            if df is None:
                try:
                    tables = pd.read_html(response.content)
                    # Usually the main odds table is one of the larger tables
                    df = max(tables, key=lambda x: len(x.columns) * len(x))
                    logger.info("Parsed largest table from page")
                except Exception as e:
                    logger.error(f"Failed to parse any table: {e}")
                    return None

            if df is not None and not df.empty:
                logger.info(f"Retrieved {len(df)} games with odds")
                return df

            logger.warning("No odds data found")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching odds: {e}")
            return None

    def fetch_matchup_odds(self, matchup_url: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch odds from the matchups page which often has more detailed information

        Args:
            matchup_url: Specific matchup URL or None for current week

        Returns:
            DataFrame with matchup odds data
        """
        if matchup_url is None:
            url = f"{self.BASE_URL}/college-football/matchups/"
        else:
            url = matchup_url

        try:
            logger.info(f"Fetching matchup odds from: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Replace <br> tags
            for br in soup.select("br"):
                br.replace_with("\n")

            tables = pd.read_html(response.content)

            if tables:
                # Return the first substantial table
                for table in tables:
                    if len(table) > 0 and len(table.columns) > 3:
                        logger.info(f"Retrieved matchup data for {len(table)} games")
                        return table

            return None

        except Exception as e:
            logger.error(f"Error fetching matchup odds: {e}")
            return None

    def parse_spread(self, spread_str: str) -> Optional[float]:
        """Parse spread string to float (e.g., '-7.5' or 'PK')"""
        if pd.isna(spread_str) or spread_str == '' or spread_str == 'NL':
            return None

        spread_str = str(spread_str).strip()

        # Handle 'PK' (pick'em)
        if spread_str.upper() in ['PK', 'PICK', 'EVEN']:
            return 0.0

        # Extract numeric value
        match = re.search(r'[-+]?\d+\.?\d*', spread_str)
        if match:
            return float(match.group())

        return None

    def parse_moneyline(self, ml_str: str) -> Optional[int]:
        """Parse moneyline string to integer (e.g., '-150' or '+200')"""
        if pd.isna(ml_str) or ml_str == '' or ml_str == 'NL':
            return None

        ml_str = str(ml_str).strip()

        # Extract numeric value with sign
        match = re.search(r'[-+]?\d+', ml_str)
        if match:
            return int(match.group())

        return None

    def parse_total(self, total_str: str) -> Optional[float]:
        """Parse total (over/under) string to float"""
        if pd.isna(total_str) or total_str == '' or total_str == 'NL':
            return None

        total_str = str(total_str).strip()

        # Remove 'o', 'u', 'ov', 'un' prefixes
        total_str = re.sub(r'^(o|u|ov|un|over|under)\s*', '', total_str, flags=re.IGNORECASE)

        # Extract numeric value
        match = re.search(r'\d+\.?\d*', total_str)
        if match:
            return float(match.group())

        return None

    def match_team_to_database(self, team_name: str) -> Optional[int]:
        """
        Try to match a team name from VegasInsider to a team_id in the database

        Args:
            team_name: Team name from VegasInsider

        Returns:
            team_id if found, None otherwise
        """
        if not self.db.conn:
            self.db.connect()

        cursor = self.db.conn.cursor()

        # Clean the team name
        team_name = team_name.strip()

        # Try exact match first
        cursor.execute('''
            SELECT team_id FROM teams
            WHERE name = ? OR display_name = ? OR abbreviation = ?
        ''', (team_name, team_name, team_name))

        result = cursor.fetchone()
        if result:
            return result[0]

        # Try fuzzy match with LIKE
        cursor.execute('''
            SELECT team_id, name FROM teams
            WHERE name LIKE ? OR display_name LIKE ?
        ''', (f'%{team_name}%', f'%{team_name}%'))

        result = cursor.fetchone()
        if result:
            return result[0]

        logger.debug(f"Could not match team: {team_name}")
        return None

    def process_odds_dataframe(self, df: pd.DataFrame, source: str = 'VegasInsider') -> List[Dict]:
        """
        Process a DataFrame of odds data into structured format

        Args:
            df: DataFrame with odds data
            source: Source name (e.g., 'VegasInsider')

        Returns:
            List of odds dictionaries
        """
        odds_list = []

        logger.info(f"Processing DataFrame with columns: {df.columns.tolist()}")

        # The structure varies, but we need to identify:
        # - Team names (home/away)
        # - Spreads
        # - Moneylines
        # - Totals

        for idx, row in df.iterrows():
            try:
                odds_data = {
                    'source': source,
                    'raw_data': row.to_dict()
                }

                # Try to extract team names
                # Common column names: 'Team', 'Teams', 'Matchup', 'Rotation', etc.
                team_columns = [col for col in df.columns if 'team' in str(col).lower()
                               or 'matchup' in str(col).lower()]

                if team_columns:
                    team_data = row[team_columns[0]]
                    # Often the team data contains both teams
                    # Format could be "Team A vs Team B" or separate rows

                # Extract spread
                spread_columns = [col for col in df.columns if 'spread' in str(col).lower()
                                 or 'line' in str(col).lower()]
                if spread_columns:
                    spread_value = self.parse_spread(row[spread_columns[0]])
                    odds_data['spread'] = spread_value

                # Extract moneyline
                ml_columns = [col for col in df.columns if 'money' in str(col).lower()
                             or 'ml' in str(col).lower()]
                if ml_columns:
                    ml_value = self.parse_moneyline(row[ml_columns[0]])
                    odds_data['moneyline'] = ml_value

                # Extract total
                total_columns = [col for col in df.columns if 'total' in str(col).lower()
                                or 'o/u' in str(col).lower() or 'over' in str(col).lower()]
                if total_columns:
                    total_value = self.parse_total(row[total_columns[0]])
                    odds_data['total'] = total_value

                odds_list.append(odds_data)

            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue

        return odds_list

    def save_odds_to_database(self, odds_data: Dict, game_id: int):
        """
        Save odds data to the database

        Args:
            odds_data: Dictionary containing odds information
            game_id: ESPN game ID to associate odds with
        """
        if not self.db.conn:
            self.db.connect()

        # Prepare odds data for database
        db_odds = {
            'game_id': game_id,
            'source': odds_data.get('source', 'VegasInsider'),
            'latest_spread': odds_data.get('spread_home'),
            'current_spread_away': odds_data.get('spread_away'),
            'current_moneyline_home': odds_data.get('moneyline_home'),
            'current_moneyline_away': odds_data.get('moneyline_away'),
            'current_total': odds_data.get('total'),
            'current_over_odds': odds_data.get('over_odds'),
            'current_under_odds': odds_data.get('under_odds'),
        }

        # If we have opening lines
        if odds_data.get('opening_spread'):
            db_odds['opening_spread'] = odds_data['opening_spread']
            db_odds['opening_spread_away'] = odds_data['opening_spread_away']

        if odds_data.get('opening_moneyline_home'):
            db_odds['opening_moneyline_home'] = odds_data['opening_moneyline_home']
            db_odds['opening_moneyline_away'] = odds_data['opening_moneyline_away']

        if odds_data.get('opening_total'):
            db_odds['opening_total'] = odds_data['opening_total']

        # Save to database
        self.db.insert_or_update_odds(db_odds)

        # Also save to odds movement history
        movement_data = {
            'game_id': game_id,
            'source': odds_data.get('source', 'VegasInsider'),
            'spread_home': odds_data.get('spread_home'),
            'spread_away': odds_data.get('spread_away'),
            'moneyline_home': odds_data.get('moneyline_home'),
            'moneyline_away': odds_data.get('moneyline_away'),
            'total': odds_data.get('total'),
            'over_odds': odds_data.get('over_odds'),
            'under_odds': odds_data.get('under_odds'),
            'timestamp': datetime.now().isoformat()
        }

        # odds_movement table removed

    def scrape_current_week_odds(self):
        """Scrape odds for the current week's games"""
        logger.info("Scraping current week odds from VegasInsider...")

        # Fetch odds data
        df = self.fetch_current_odds()

        if df is None or df.empty:
            logger.warning("No odds data retrieved")
            return

        # Process the data
        odds_list = self.process_odds_dataframe(df)

        logger.info(f"Processed {len(odds_list)} odds entries")

        # Note: Matching VegasInsider games to ESPN game_ids is complex
        # For now, we'll log the data structure to help with manual mapping
        logger.info("Sample odds data structure:")
        if odds_list:
            logger.info(odds_list[0])


if __name__ == '__main__':
    # Example usage
    scraper = VegasInsiderOddsScraper()

    # Try to fetch current odds
    scraper.scrape_current_week_odds()

    print("\nVegasInsider Odds Scraper ready.")
    print("Note: Matching VegasInsider teams to ESPN game IDs requires manual mapping.")
    print("Check the logs to see the data structure retrieved.")
