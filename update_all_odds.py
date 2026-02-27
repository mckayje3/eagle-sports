"""
Unified odds update script - Hybrid approach
Uses VegasInsider (free) for regular updates
Saves Odds API requests for spot-checking
"""
import json
import sys
from datetime import datetime


def update_vegasinsider_odds():
    """Update odds from VegasInsider (free)"""
    print("\n" + "="*80)
    print("UPDATING ODDS FROM VEGASINSIDER (FREE)")
    print("="*80 + "\n")

    # Step 1: Scrape VegasInsider
    print("Step 1: Scraping VegasInsider...")
    import subprocess
    result = subprocess.run(['py', 'parse_vegasinsider.py'], capture_output=True, text=True)

    if result.returncode != 0:
        print("ERROR: VegasInsider scraping failed")
        print(result.stderr)
        return False

    print("OK: VegasInsider data scraped\n")

    # Step 2: Match and save to database
    print("Step 2: Matching and saving to database...")
    result = subprocess.run(['py', 'save_odds_with_matching.py', 'vegasinsider'],
                          capture_output=True, text=True)

    if result.returncode != 0:
        print("ERROR: Saving to database failed")
        print(result.stderr)
        return False

    # Show summary from output
    for line in result.stdout.split('\n'):
        if 'SUMMARY' in line or 'Total games' in line or 'Matched' in line or 'Saved' in line:
            print(line)

    print("\nOK: VegasInsider odds updated successfully")
    return True


def check_api_quota():
    """Check remaining Odds API requests"""
    import os

    print("\n" + "="*80)
    print("ODDS API QUOTA CHECK")
    print("="*80 + "\n")

    # Get API key from environment variable (secure)
    api_key = os.getenv('ODDS_API_KEY')

    if not api_key:
        print("ERROR: ODDS_API_KEY environment variable not set")
        print("Set it with: set ODDS_API_KEY=your_api_key")
        return

    from odds_api_scraper import OddsAPIScraper
    scraper = OddsAPIScraper(api_key=api_key)
    remaining = scraper.get_remaining_requests()

    if remaining:
        print(f"Odds API requests remaining: {remaining}/500")
        print(f"Monthly reset: 1st of each month")
        print(f"\nRecommended usage:")
        print(f"  - Save for spot-checking VegasInsider data")
        print(f"  - Use for games with unusual line movements")
        print(f"  - Verify consensus on big games")
    else:
        print("Could not check API quota")


def main():
    """Main update routine"""
    print("\n" + "="*80)
    print("HYBRID ODDS TRACKING SYSTEM")
    print("="*80)
    print(f"\nRun time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check what to do
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = 'vegasinsider'  # Default to free source

    if command == 'vegasinsider' or command == 'vi':
        update_vegasinsider_odds()
    elif command == 'quota' or command == 'check':
        check_api_quota()
    elif command == 'status':
        # Show current database status
        print("\n" + "="*80)
        print("DATABASE STATUS")
        print("="*80 + "\n")
        import subprocess
        subprocess.run(['py', 'check_saved_odds.py'])
    else:
        print(f"\nUnknown command: {command}")
        print("\nUsage:")
        print("  py update_all_odds.py                # Update from VegasInsider")
        print("  py update_all_odds.py vegasinsider   # Update from VegasInsider")
        print("  py update_all_odds.py quota          # Check Odds API quota")
        print("  py update_all_odds.py status         # Show database status")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
