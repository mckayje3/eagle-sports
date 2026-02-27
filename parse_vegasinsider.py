"""
Parse VegasInsider data and save to database
"""
import pandas as pd
import re
from datetime import datetime
from cfb_nfl_database import FootballDatabase

def parse_matchups_data(csv_file='vegasinsider_table_1.csv'):
    """Parse the VegasInsider matchups CSV data"""

    df = pd.read_csv(csv_file)

    games = []
    current_game = {}

    for idx, row in df.iterrows():
        # Get the first column value
        first_col = str(row['Unnamed: 0']).strip() if pd.notna(row['Unnamed: 0']) else ''

        # Check if this is a team row (has number prefix)
        if re.match(r'^\d+\s+', first_col):
            # Extract team info
            parts = first_col.split(' ', 1)
            team_num = parts[0]
            team_name = parts[1] if len(parts) > 1 else ''

            # If we don't have away team yet, this is the away team
            if 'away_team' not in current_game:
                current_game['away_team'] = team_name
                current_game['away_team_num'] = team_num
            else:
                # This is the home team
                current_game['home_team'] = team_name
                current_game['home_team_num'] = team_num

        # Check if this is the "Open" row with opening odds
        elif first_col == 'Open':
            spread = str(row['Spread']) if pd.notna(row['Spread']) else ''
            total = str(row['Total']) if pd.notna(row['Total']) else ''
            moneyline = str(row['Moneyline']) if pd.notna(row['Moneyline']) else ''

            current_game['opening_spread'] = spread
            current_game['opening_total'] = total
            current_game['opening_moneyline'] = moneyline

        # Check if this is the "Consensus" row with current odds
        elif first_col == 'Consensus':
            spread = str(row['Spread']) if pd.notna(row['Spread']) else ''
            total = str(row['Total']) if pd.notna(row['Total']) else ''
            moneyline = str(row['Moneyline']) if pd.notna(row['Moneyline']) else ''

            current_game['current_spread'] = spread
            current_game['current_total'] = total
            current_game['current_moneyline'] = moneyline

            # Game data is complete, add to list
            if 'away_team' in current_game and 'home_team' in current_game:
                games.append(current_game.copy())

            # Reset for next game
            current_game = {}

    return games

def parse_spread(spread_str):
    """Parse spread string to get team and value"""
    if not spread_str or spread_str == 'nan':
        return None, None

    # Handle formats like "OHIO -29.5", "BGSU -4.5", "PK"
    if spread_str.upper() in ['PK', 'PICK']:
        return None, 0.0

    # Try to extract team abbreviation and value
    match = re.match(r'([A-Z]+)\s+([-+]?\d+\.?\d*)', spread_str)
    if match:
        team = match.group(1)
        value = float(match.group(2))
        return team, value

    # Just a number
    match = re.match(r'([-+]?\d+\.?\d*)', spread_str)
    if match:
        return None, float(match.group(1))

    return None, None

def parse_total(total_str):
    """Parse total string"""
    if not total_str or total_str == 'nan':
        return None

    # Handle formats like "o51.5", "u47.5", "51.5"
    match = re.search(r'(\d+\.?\d*)', total_str)
    if match:
        return float(match.group(1))

    return None

def parse_moneyline(ml_str):
    """Parse moneyline string"""
    if not ml_str or ml_str == 'nan':
        return None

    # Extract number with sign
    match = re.search(r'([-+]?\d+)', ml_str)
    if match:
        return int(match.group(1))

    return None

def display_games(games):
    """Display parsed games in a readable format"""
    print(f"\nParsed {len(games)} games from VegasInsider:")
    print("=" * 100)

    for i, game in enumerate(games, 1):
        print(f"\nGame {i}:")
        print(f"  {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')}")

        # Opening odds
        print(f"\n  Opening Line:")
        print(f"    Spread: {game.get('opening_spread', 'N/A')}")
        print(f"    Total:  {game.get('opening_total', 'N/A')}")
        print(f"    ML:     {game.get('opening_moneyline', 'N/A')}")

        # Current odds
        print(f"\n  Current Line:")
        print(f"    Spread: {game.get('current_spread', 'N/A')}")
        print(f"    Total:  {game.get('current_total', 'N/A')}")
        print(f"    ML:     {game.get('current_moneyline', 'N/A')}")

        # Calculate movement
        opening_total = parse_total(game.get('opening_total', ''))
        current_total = parse_total(game.get('current_total', ''))

        if opening_total and current_total:
            movement = current_total - opening_total
            if movement != 0:
                print(f"\n  Line Movement:")
                print(f"    Total moved {movement:+.1f} points")

        print("-" * 100)

def main():
    """Main execution"""
    print("VegasInsider Data Parser")
    print("=" * 100)

    # Parse the data
    games = parse_matchups_data()

    # Display results
    display_games(games)

    # Save to JSON for inspection
    import json
    with open('parsed_games.json', 'w') as f:
        json.dump(games, f, indent=2)
    print(f"\nSaved parsed data to parsed_games.json")

    print(f"\n\nSummary:")
    print(f"  Total games found: {len(games)}")
    print(f"  Games with opening lines: {sum(1 for g in games if g.get('opening_spread'))}")
    print(f"  Games with current lines: {sum(1 for g in games if g.get('current_spread'))}")

    return games

if __name__ == '__main__':
    games = main()
