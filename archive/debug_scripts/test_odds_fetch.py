"""
Test script to fetch college football odds from VegasInsider
This will try to fetch last week's odds data
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from datetime import datetime

def fetch_vegasinsider_page(url):
    """Fetch and parse a VegasInsider page"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    try:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        print(f"Status: {response.status_code}")

        # Save raw HTML for inspection
        with open('vegasinsider_raw.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Saved raw HTML to vegasinsider_raw.html")

        soup = BeautifulSoup(response.content, 'html.parser')

        # Replace <br> tags
        for br in soup.select("br"):
            br.replace_with("\n")

        # Try to find tables
        tables = soup.find_all('table')
        print(f"\nFound {len(tables)} table(s) on the page")

        if tables:
            # Try pandas to parse tables
            try:
                dfs = pd.read_html(response.content)
                print(f"Pandas found {len(dfs)} parseable table(s)")

                for i, df in enumerate(dfs):
                    print(f"\nTable {i+1} shape: {df.shape}")
                    print(f"Columns: {df.columns.tolist()}")
                    if len(df) > 0:
                        print(f"First few rows:")
                        print(df.head(3))

                        # Save to CSV
                        filename = f'vegasinsider_table_{i+1}.csv'
                        df.to_csv(filename, index=False)
                        print(f"Saved to {filename}")

                return dfs
            except Exception as e:
                print(f"Error parsing with pandas: {e}")

        # Look for specific data in the HTML
        print("\nSearching for game/odds data in HTML...")

        # Look for common class names
        game_containers = soup.find_all(['div', 'tr'], class_=lambda x: x and ('game' in x.lower() or 'match' in x.lower() or 'odds' in x.lower()))
        print(f"Found {len(game_containers)} potential game containers")

        if game_containers:
            print("\nFirst few containers:")
            for i, container in enumerate(game_containers[:3]):
                print(f"\nContainer {i+1}:")
                print(container.get_text()[:200])

        return None

    except Exception as e:
        print(f"Error: {e}")
        return None

def test_multiple_urls():
    """Test multiple VegasInsider URLs"""

    # Current week (Week 12 in 2025)
    current_week = 12
    last_week = 11

    urls_to_try = [
        # Last week's odds
        f"https://www.vegasinsider.com/college-football/odds/las-vegas/week-{last_week}/",

        # Current week's odds
        f"https://www.vegasinsider.com/college-football/odds/las-vegas/week-{current_week}/",

        # Main odds page
        "https://www.vegasinsider.com/college-football/odds/las-vegas/",

        # Matchups page
        "https://www.vegasinsider.com/college-football/matchups/",
    ]

    results = {}

    for url in urls_to_try:
        print("\n" + "="*80)
        print(f"TESTING: {url}")
        print("="*80)

        data = fetch_vegasinsider_page(url)
        results[url] = data

        print("\n" + "-"*80 + "\n")

    return results

if __name__ == '__main__':
    print("VegasInsider Odds Fetch Test")
    print("=" * 80)
    print(f"Date: {datetime.now()}")
    print(f"Testing fetch for Week 11 (last week) odds\n")

    results = test_multiple_urls()

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for url, data in results.items():
        status = "SUCCESS - Data retrieved" if data else "NO DATA"
        print(f"{url}")
        print(f"  Status: {status}\n")

    print("\nCheck the generated CSV files and vegasinsider_raw.html for details")
