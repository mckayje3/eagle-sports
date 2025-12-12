"""
Complete workflow: Scrape odds and save to database with game matching
"""
import json
from game_matcher import GameMatcher
from database import FootballDatabase
from datetime import datetime


def save_vegasinsider_odds_to_db():
    """
    Load parsed VegasInsider data and save to database with matching
    """
    print("="*80)
    print("SAVING VEGASINSIDER ODDS TO DATABASE")
    print("="*80 + "\n")

    # Load parsed games
    try:
        with open('parsed_games.json', 'r') as f:
            games = json.load(f)
    except FileNotFoundError:
        print("Error: parsed_games.json not found!")
        print("Run: py parse_vegasinsider.py first")
        return

    print(f"Loaded {len(games)} games from VegasInsider\n")

    # Initialize matcher and database
    matcher = GameMatcher()
    db = FootballDatabase()
    db.connect()

    # Match and save each game
    matched_count = 0
    saved_count = 0

    for i, game in enumerate(games, 1):
        home = game.get('home_team')
        away = game.get('away_team')

        print(f"\n[{i}/{len(games)}] {away} @ {home}")

        # Match to ESPN game_id
        game_id = matcher.match_vegas_insider_game(game)

        if not game_id:
            print(f"  X Could not match to ESPN database")
            continue

        matched_count += 1
        print(f"  OK Matched to game_id: {game_id}")

        # Parse odds data
        odds_data = {
            'game_id': game_id,
            'source': 'VegasInsider'
        }

        # Opening lines
        opening_spread = game.get('opening_spread', '')
        if opening_spread and opening_spread != 'nan':
            # Parse spread like "OHIO -29.5"
            parts = opening_spread.split()
            if len(parts) >= 2:
                try:
                    spread_value = float(parts[1])
                    # Assume first team is favorite if negative
                    if spread_value < 0:
                        odds_data['opening_spread'] = spread_value
                        # away spread removed = -spread_value
                    else:
                        # away spread removed = spread_value
                        odds_data['opening_spread'] = -spread_value
                except ValueError:
                    pass

        # Opening total
        opening_total = game.get('opening_total', '')
        if opening_total and opening_total != 'nan':
            try:
                # Extract number from "o51.5"
                total_value = float(opening_total.replace('o', '').replace('u', ''))
                odds_data['opening_total'] = total_value
            except ValueError:
                pass

        # Opening moneyline
        opening_ml = game.get('opening_moneyline', '')
        if opening_ml and opening_ml != 'nan' and opening_ml:
            try:
                ml_value = int(opening_ml)
                # Assume this is for the favorite
                if ml_value < 0:
                    odds_data['opening_moneyline'] = ml_value
                else:
                    # away ml removed = ml_value
            except ValueError:
                pass

        # Current/Consensus lines
        current_spread = game.get('current_spread', '')
        if current_spread and current_spread != 'nan':
            parts = current_spread.split()
            if len(parts) >= 2:
                try:
                    spread_value = float(parts[1])
                    if spread_value < 0:
                        odds_data['latest_spread'] = spread_value
                        # away spread removed = -spread_value
                    else:
                        # away spread removed = spread_value
                        odds_data['latest_spread'] = -spread_value
                except ValueError:
                    pass

        # Current total
        current_total = game.get('current_total', '')
        if current_total and current_total != 'nan':
            try:
                total_value = float(current_total.replace('o', '').replace('u', ''))
                odds_data['latest_total'] = total_value
            except ValueError:
                pass

        # Current moneyline
        current_ml = game.get('current_moneyline', '')
        if current_ml and current_ml != 'nan' and current_ml:
            try:
                ml_value = int(current_ml)
                if ml_value < 0:
                    odds_data['latest_moneyline'] = ml_value
                else:
                    # away ml removed = ml_value
            except ValueError:
                pass

        # Save to database
        try:
            db.insert_or_update_odds(odds_data)
            saved_count += 1
            print(f"  OK Saved odds to database")



        except Exception as e:
            print(f"  X Error saving: {e}")

    db.close()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total games: {len(games)}")
    print(f"Matched to ESPN: {matched_count}")
    print(f"Saved to database: {saved_count}")
    print(f"Failed to match: {len(games) - matched_count}")
    print("="*80 + "\n")


def save_odds_api_data_to_db(json_file='odds_api_current_games.json'):
    """
    Load The Odds API data and save to database with matching
    """
    print("="*80)
    print("SAVING ODDS API DATA TO DATABASE")
    print("="*80 + "\n")

    # Load data
    try:
        with open(json_file, 'r') as f:
            games = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found!")
        print("Run: py test_odds_api.py first")
        return

    print(f"Loaded {len(games)} games from The Odds API\n")

    # Initialize
    matcher = GameMatcher()
    db = FootballDatabase()
    db.connect()

    matched_count = 0
    saved_count = 0

    for i, game in enumerate(games, 1):
        home = game.get('home_team')
        away = game.get('away_team')

        print(f"\n[{i}/{len(games)}] {away} @ {home}")

        # Match to ESPN game_id
        game_id = matcher.match_odds_api_game(game)

        if not game_id:
            print(f"  X Could not match to ESPN database")
            continue

        matched_count += 1
        print(f"  OK Matched to game_id: {game_id}")

        # Parse odds from bookmakers
        bookmakers = game.get('bookmakers', [])
        if not bookmakers:
            print(f"  X No bookmaker data")
            continue

        # Calculate consensus from all bookmakers
        spreads_home = []
        totals = []
        ml_home = []
        ml_away = []

        for book in bookmakers:
            for market in book.get('markets', []):
                if market['key'] == 'spreads':
                    for outcome in market.get('outcomes', []):
                        if outcome['name'] == home:
                            spreads_home.append(outcome['point'])

                elif market['key'] == 'totals':
                    for outcome in market.get('outcomes', []):
                        if outcome['name'] == 'Over':
                            totals.append(outcome['point'])

                elif market['key'] == 'h2h':
                    for outcome in market.get('outcomes', []):
                        if outcome['name'] == home:
                            ml_home.append(outcome['price'])
                        elif outcome['name'] == away:
                            ml_away.append(outcome['price'])

        # Create odds data with consensus
        odds_data = {
            'game_id': game_id,
            'source': 'TheOddsAPI'
        }

        if spreads_home:
            avg_spread = sum(spreads_home) / len(spreads_home)
            odds_data['latest_spread'] = round(avg_spread, 1)
            # away spread removed = round(-avg_spread, 1)

        if totals:
            avg_total = sum(totals) / len(totals)
            odds_data['latest_total'] = round(avg_total, 1)

        if ml_home:
            avg_ml_home = int(sum(ml_home) / len(ml_home))
            odds_data['latest_moneyline'] = avg_ml_home

        if ml_away:
            avg_ml_away = int(sum(ml_away) / len(ml_away))
            # away ml removed = avg_ml_away

        # Save to database
        try:
            db.insert_or_update_odds(odds_data)
            saved_count += 1
            print(f"  OK Saved odds to database")

            # Save movement snapshot
            movement_data = {
                'game_id': game_id,
                'source': 'TheOddsAPI',
                'spread_home': odds_data.get('latest_spread'),
                'spread_away': None,
                'moneyline_home': odds_data.get('latest_moneyline'),
                'moneyline_away': None,
                'total': odds_data.get('latest_total'),
                'timestamp': datetime.now().isoformat()
            }
            # odds_movement table removed

        except Exception as e:
            print(f"  X Error saving: {e}")

    db.close()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total games: {len(games)}")
    print(f"Matched to ESPN: {matched_count}")
    print(f"Saved to database: {saved_count}")
    print(f"Failed to match: {len(games) - matched_count}")
    print("="*80 + "\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        source = sys.argv[1].lower()
    else:
        print("Usage:")
        print("  py save_odds_with_matching.py vegasinsider")
        print("  py save_odds_with_matching.py oddsapi")
        sys.exit(1)

    if source == 'vegasinsider':
        save_vegasinsider_odds_to_db()
    elif source == 'oddsapi':
        save_odds_api_data_to_db()
    else:
        print(f"Unknown source: {source}")
        print("Use: vegasinsider or oddsapi")
