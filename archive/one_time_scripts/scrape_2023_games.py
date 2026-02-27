"""
Scrape 2023 NFL and CFB game data from ESPN
"""
import sys


def scrape_nfl_2023():
    """Scrape 2023 NFL season"""
    print("\n" + "="*80)
    print("SCRAPING 2023 NFL SEASON")
    print("="*80 + "\n")

    from nfl_espn_scraper import NFLESPNScraper

    scraper = NFLESPNScraper()

    # Regular season (weeks 1-18)
    print("Scraping regular season...")
    scraper.scrape_season(2023, start_week=1, end_week=18, season_type=2)

    # Playoffs (season_type=3)
    print("\nScraping playoffs...")
    scraper.scrape_season(2023, start_week=1, end_week=5, season_type=3)

    print("\n2023 NFL scrape complete!")


def scrape_cfb_2023():
    """Scrape 2023 CFB season"""
    print("\n" + "="*80)
    print("SCRAPING 2023 CFB SEASON")
    print("="*80 + "\n")

    from espn_scraper import ESPNScraper

    scraper = ESPNScraper()

    # Regular season (weeks 0-15)
    print("Scraping regular season...")
    scraper.scrape_season(2023, start_week=0, end_week=15, season_type=2)

    # Bowl games (season_type=3)
    print("\nScraping bowl games...")
    scraper.scrape_season(2023, start_week=1, end_week=5, season_type=3)

    print("\n2023 CFB scrape complete!")


def main():
    if len(sys.argv) > 1:
        sport = sys.argv[1].lower()
        if sport == 'nfl':
            scrape_nfl_2023()
        elif sport == 'cfb':
            scrape_cfb_2023()
        else:
            print(f"Unknown sport: {sport}")
    else:
        # Scrape both
        scrape_nfl_2023()
        scrape_cfb_2023()


if __name__ == '__main__':
    main()
