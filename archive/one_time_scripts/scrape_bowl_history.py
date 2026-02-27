"""
Scrape historical bowl game data from Sports Reference
Gets bowl results for training bowl prediction model
"""
import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import time
import re
from datetime import datetime


class BowlHistoryScraper:
    """Scrape historical bowl game results from Sports-Reference"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.base_url = "https://www.sports-reference.com/cfb/years/{}-bowls.html"

    def scrape_season_bowls(self, season: int) -> list:
        """
        Scrape all bowl games for a given season

        Args:
            season: The fall year (e.g., 2023 = 2023-24 bowl season)

        Returns:
            List of bowl game dictionaries
        """
        url = self.base_url.format(season)
        print(f"Scraping {url}...")

        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"  Error: HTTP {resp.status_code}")
                return []

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Find the bowl games table
            table = soup.find('table', {'id': 'bowls'})
            if not table:
                print(f"  No bowl table found for {season}")
                return []

            games = []
            rows = table.find('tbody').find_all('tr')

            for row in rows:
                # Skip header rows
                if row.get('class') and 'thead' in row.get('class'):
                    continue

                cols = row.find_all(['td', 'th'])
                if len(cols) < 7:
                    continue

                try:
                    # Parse date
                    date_str = cols[0].get_text(strip=True)

                    # Parse bowl name
                    bowl_name = cols[2].get_text(strip=True) if len(cols) > 2 else ""

                    # Parse winner info
                    winner_cell = cols[3] if len(cols) > 3 else None
                    winner_name = ""
                    winner_rank = None
                    if winner_cell:
                        winner_text = winner_cell.get_text(strip=True)
                        # Extract rank if present (e.g., "(1) Michigan")
                        rank_match = re.match(r'\((\d+)\)\s*(.+)', winner_text)
                        if rank_match:
                            winner_rank = int(rank_match.group(1))
                            winner_name = rank_match.group(2).strip()
                        else:
                            winner_name = winner_text

                    # Winner score
                    winner_score = None
                    if len(cols) > 4:
                        try:
                            winner_score = int(cols[4].get_text(strip=True))
                        except ValueError:
                            pass

                    # Parse loser info
                    loser_cell = cols[5] if len(cols) > 5 else None
                    loser_name = ""
                    loser_rank = None
                    if loser_cell:
                        loser_text = loser_cell.get_text(strip=True)
                        rank_match = re.match(r'\((\d+)\)\s*(.+)', loser_text)
                        if rank_match:
                            loser_rank = int(rank_match.group(1))
                            loser_name = rank_match.group(2).strip()
                        else:
                            loser_name = loser_text

                    # Loser score
                    loser_score = None
                    if len(cols) > 6:
                        try:
                            loser_score = int(cols[6].get_text(strip=True))
                        except ValueError:
                            pass

                    if winner_name and loser_name and winner_score is not None:
                        game = {
                            'season': season,
                            'date': date_str,
                            'bowl_name': bowl_name,
                            'winner': winner_name,
                            'winner_rank': winner_rank,
                            'winner_score': winner_score,
                            'loser': loser_name,
                            'loser_rank': loser_rank,
                            'loser_score': loser_score,
                            'margin': winner_score - (loser_score or 0),
                            'total_points': winner_score + (loser_score or 0)
                        }
                        games.append(game)

                except Exception as e:
                    print(f"  Error parsing row: {e}")
                    continue

            print(f"  Found {len(games)} bowl games")
            return games

        except Exception as e:
            print(f"  Error scraping {season}: {e}")
            return []

    def scrape_multiple_seasons(self, seasons: list) -> pd.DataFrame:
        """
        Scrape bowl games for multiple seasons

        Args:
            seasons: List of season years

        Returns:
            DataFrame with all bowl games
        """
        all_games = []

        for season in seasons:
            games = self.scrape_season_bowls(season)
            all_games.extend(games)
            time.sleep(2)  # Be nice to the server

        if all_games:
            df = pd.DataFrame(all_games)
            return df
        return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save bowl games to CSV"""
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} bowl games to {filename}")

    def save_to_database(self, df: pd.DataFrame, db_path: str = 'cfb_games.db'):
        """
        Save bowl games to a dedicated bowl_history table

        Args:
            df: DataFrame with bowl games
            db_path: Path to database
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create bowl_history table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bowl_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER,
                date TEXT,
                bowl_name TEXT,
                winner TEXT,
                winner_rank INTEGER,
                winner_score INTEGER,
                loser TEXT,
                loser_rank INTEGER,
                loser_score INTEGER,
                margin INTEGER,
                total_points INTEGER,
                UNIQUE(season, bowl_name)
            )
        ''')

        # Insert games
        inserted = 0
        for _, row in df.iterrows():
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO bowl_history
                    (season, date, bowl_name, winner, winner_rank, winner_score,
                     loser, loser_rank, loser_score, margin, total_points)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['season'],
                    row['date'],
                    row['bowl_name'],
                    row['winner'],
                    row.get('winner_rank'),
                    row['winner_score'],
                    row['loser'],
                    row.get('loser_rank'),
                    row['loser_score'],
                    row['margin'],
                    row['total_points']
                ))
                inserted += 1
            except Exception as e:
                print(f"  Error inserting: {e}")

        conn.commit()
        conn.close()
        print(f"Saved {inserted} bowl games to {db_path}")


def main():
    scraper = BowlHistoryScraper()

    # Scrape 2021, 2022, and 2023 bowl seasons
    seasons = [2021, 2022, 2023]

    print("=" * 60)
    print("SCRAPING HISTORICAL BOWL GAMES")
    print("=" * 60)

    df = scraper.scrape_multiple_seasons(seasons)

    if not df.empty:
        # Save to CSV
        scraper.save_to_csv(df, 'bowl_history.csv')

        # Save to database
        scraper.save_to_database(df)

        # Print summary
        print("\n" + "=" * 60)
        print("BOWL HISTORY SUMMARY")
        print("=" * 60)
        print(f"Total games: {len(df)}")
        for season in seasons:
            count = len(df[df['season'] == season])
            print(f"  {season}: {count} bowl games")

        # Interesting stats
        print(f"\nAverage margin: {df['margin'].mean():.1f}")
        print(f"Average total: {df['total_points'].mean():.1f}")

        # Ranked vs unranked
        ranked_wins = len(df[df['winner_rank'].notna()])
        ranked_losses = len(df[df['loser_rank'].notna()])
        print(f"\nRanked team wins: {ranked_wins}")
        print(f"Upsets (ranked loser): {ranked_losses}")

        return df
    else:
        print("No data scraped")
        return None


if __name__ == '__main__':
    main()
