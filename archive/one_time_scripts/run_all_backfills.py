"""
Master script to run all odds backfills
Uses remaining API credits before month end
"""
import subprocess
import sys

def check_api_credits():
    """Check remaining API credits"""
    import requests
    import json

    with open('odds_api_config.json', 'r') as f:
        config = json.load(f)
        api_key = config.get('api_key')

    resp = requests.get(f'https://api.the-odds-api.com/v4/sports/?apiKey={api_key}')
    remaining = resp.headers.get('x-requests-remaining', 'Unknown')
    used = resp.headers.get('x-requests-used', 'Unknown')
    print(f"\n{'='*60}")
    print(f"API Credits: {remaining} remaining | {used} used")
    print(f"{'='*60}\n")
    return float(remaining) if remaining != 'Unknown' else 0


def main():
    print("="*80)
    print("COMPREHENSIVE ODDS BACKFILL")
    print("="*80)

    # Check starting credits
    credits = check_api_credits()

    if credits < 100:
        print("WARNING: Low API credits. Proceeding anyway...")

    # Step 1: Backfill 2023 NFL odds
    print("\n" + "="*80)
    print("STEP 1: Backfill 2023 NFL Odds")
    print("="*80)
    subprocess.run([sys.executable, 'backfill_2023_odds.py', 'nfl'])

    check_api_credits()

    # Step 2: Backfill 2023 CFB odds
    print("\n" + "="*80)
    print("STEP 2: Backfill 2023 CFB Odds")
    print("="*80)
    subprocess.run([sys.executable, 'backfill_2023_odds.py', 'cfb'])

    check_api_credits()

    # Step 3: Create NBA database and scrape game data
    print("\n" + "="*80)
    print("STEP 3: Create NBA Database & Scrape Games")
    print("="*80)
    subprocess.run([sys.executable, 'nba_database.py'])
    subprocess.run([sys.executable, 'nba_espn_scraper.py'])

    # Step 4: Backfill NBA 2023-24 odds
    print("\n" + "="*80)
    print("STEP 4: Backfill NBA 2023-24 Odds")
    print("="*80)
    subprocess.run([sys.executable, 'nba_odds_backfill.py', '2023'])

    check_api_credits()

    # Step 5: Backfill NBA 2024-25 odds
    print("\n" + "="*80)
    print("STEP 5: Backfill NBA 2024-25 Odds")
    print("="*80)
    subprocess.run([sys.executable, 'nba_odds_backfill.py', '2024'])

    # Final credit check
    remaining = check_api_credits()

    print("\n" + "="*80)
    print("ALL BACKFILLS COMPLETE!")
    print("="*80)
    print(f"Remaining API credits: {remaining}")


if __name__ == '__main__':
    main()
