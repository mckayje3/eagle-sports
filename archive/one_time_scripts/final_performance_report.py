"""
Week 13 Performance Report
Match predictions to actual results using team names
"""
import sqlite3
import pandas as pd
import numpy as np

def normalize_team_name(name):
    """Normalize team names for matching"""
    # Common mappings
    mappings = {
        'Zips': 'Akron',
        'Minutemen': 'UMass',
        'RedHawks': 'Miami (OH)',
        'Golden Flashes': 'Kent State',
        'Chippewas': 'Central Michigan',
        'Ragin\' Cajuns': 'Louisiana',
        'Red Wolves': 'Arkansas State',
        'Rainbow Warriors': 'Hawai\'i',
        'Rebels': 'UNLV',
        'Golden Gophers': 'Minnesota',
        'Scarlet Knights': 'Rutgers',
        'Buckeyes': 'Ohio State',
        'Aggies': 'Texas A&M',
        'Bulldogs': 'Georgia',
        'Sooners': 'Oklahoma',
        'Tigers': 'Missouri',
        'Wildcats': 'Kentucky',
        'Mustangs': 'SMU',
        'Cardinals': 'Louisville',
        'Hokies': 'Virginia Tech',
        'Hurricanes': 'Miami',
        'Demon Deacons': 'Wake Forest',
        'Blue Hens': 'Delaware',
        'Cyclones': 'Iowa State',
        'Jayhawks': 'Kansas',
        'Black Knights': 'Army',
        'Golden Hurricane': 'Tulsa',
        '49ers': 'Charlotte',
        'Dukes': 'James Madison',
        'Cougars': 'BYU',
        'Bears': 'Baylor',
        'Eagles': 'Georgia Southern',
        'Monarchs': 'Old Dominion',
        'Crimson Tide': 'Alabama',
        'Panthers': 'Georgia State',
        'Owls': 'Florida Atlantic',
        'Cowboys': 'Oklahoma State',
        'Wolf Pack': 'Nevada',
        'Rockets': 'Toledo',
        'Mountaineers': 'Appalachian State',
        'Thundering Herd': 'Marshall',
        'Huskies': 'Connecticut',
        'Flames': 'Liberty',
        'Blue Raiders': 'Middle Tennessee',
        'Bearkats': 'Sam Houston',
        'Miners': 'UTEP',
        'Blazers': 'UAB',
        'Bulls': 'Buffalo',
        'Longhorns': 'Texas',
        'Razorbacks': 'Arkansas',
        'Commodores': 'Vanderbilt',
        'Ducks': 'Oregon',
        'Trojans': 'USC',
        'Hawkeyes': 'Iowa',
        'Spartans': 'Michigan State',
        'Tar Heels': 'North Carolina',
        'Blue Devils': 'Duke',
        'Fighting Irish': 'Notre Dame',
        'Orange': 'Syracuse',
        'Gamecocks': 'South Carolina',
        'Jaguars': 'South Alabama',
        'Golden Eagles': 'Southern Miss',
        'Roadrunners': 'UTSA',
        'Pirates': 'East Carolina',
        'Green Wave': 'Tulane',
        'Terrapins': 'Maryland',
        'Wolverines': 'Michigan',
        'Horned Frogs': 'TCU',
        'Knights': 'UCF',
        'Utes': 'Utah',
        'Nittany Lions': 'Penn State',
        'Cornhuskers': 'Nebraska',
        'Yellow Jackets': 'Georgia Tech',
        'Falcons': 'Air Force',
        'Lobos': 'New Mexico',
        'Broncos': 'Boise State',
        'Rams': 'Colorado State',
        'Gators': 'Florida',
        'Volunteers': 'Tennessee',
        'Badgers': 'Wisconsin',
        'Fighting Illini': 'Illinois',
        'Cardinal': 'Stanford',
        'Golden Bears': 'California',
        'Mean Green': 'North Texas',
        'Hilltoppers': 'Western Kentucky',
        'Bearcats': 'Cincinnati',
        'Buffaloes': 'Colorado',
        'Sun Devils': 'Arizona State',
        'Bruins': 'UCLA',
        'Aztecs': 'San Diego State',
        'Wolfpack': 'NC State',
        'Seminoles': 'Florida State',
        'Bobcats': 'Ohio',
        'Warhawks': 'Louisiana Monroe'
    }

    return mappings.get(name, name)

def main():
    print("=" * 80)
    print("WEEK 13 CFB PREDICTIONS PERFORMANCE REPORT")
    print("=" * 80)

    # Load predictions
    print("\nLoading predictions...")
    preds = pd.read_csv('enhanced_predictions_week_13.csv')
    preds['home_norm'] = preds['home_team'].apply(normalize_team_name)
    preds['away_norm'] = preds['away_team'].apply(normalize_team_name)
    print(f"  Loaded {len(preds)} predictions")

    # Load completed games
    print("Loading completed games...")
    conn = sqlite3.connect('cfb_games.db')

    query = """
        SELECT
            g.game_id,
            g.week,
            g.date,
            ht.name as home_team,
            at.name as away_team,
            g.home_score,
            g.away_score,
            (g.home_score - g.away_score) as actual_margin,
            CASE
                WHEN g.winner_team_id = g.home_team_id THEN ht.name
                WHEN g.winner_team_id = g.away_team_id THEN at.name
                ELSE 'Tie'
            END as actual_winner,
            go.current_spread_home as vegas_spread
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN game_odds go ON g.game_id = go.game_id
        WHERE g.season = 2024
        AND g.week = 13
        AND g.completed = 1
    """

    results = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Loaded {len(results)} completed games")

    # Match by team names
    print("\nMatching predictions to results...")
    merged = preds.merge(
        results,
        left_on=['home_norm', 'away_norm'],
        right_on=['home_team', 'away_team'],
        how='inner',
        suffixes=('_pred', '_actual')
    )
    print(f"  Matched {len(merged)} games")

    if len(merged) == 0:
        print("ERROR: No matches found. Cannot generate report.")
        return

    # Calculate metrics
    merged['correct_winner'] = merged['predicted_winner'].apply(normalize_team_name) == merged['actual_winner']
    merged['spread_error'] = abs(merged['predicted_spread'] - merged['actual_margin'])

    # Vegas comparison
    vegas_mask = ~merged['vegas_spread'].isna()
    merged['vegas_error'] = np.nan
    merged.loc[vegas_mask, 'vegas_error'] = abs(
        merged.loc[vegas_mask, 'vegas_spread'] - merged.loc[vegas_mask, 'actual_margin']
    )
    merged['model_better'] = merged['spread_error'] < merged['vegas_error']

    # REPORT
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    total = len(merged)
    correct = merged['correct_winner'].sum()
    accuracy = (correct / total) * 100

    print(f"\nGames analyzed: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")

    # Spread accuracy
    avg_spread_error = merged['spread_error'].mean()
    median_spread_error = merged['spread_error'].median()

    print(f"\nSPREAD PREDICTION")
    print(f"  Average error: {avg_spread_error:.2f} points")
    print(f"  Median error: {median_spread_error:.2f} points")

    # Vegas comparison
    vegas_games = merged[vegas_mask]
    if len(vegas_games) > 0:
        vegas_avg = vegas_games['vegas_error'].mean()
        model_avg = vegas_games['spread_error'].mean()
        model_better_count = vegas_games['model_better'].sum()
        model_better_pct = (model_better_count / len(vegas_games)) * 100

        print(f"\nVS VEGAS LINES ({len(vegas_games)} games)")
        print(f"  Vegas average error: {vegas_avg:.2f} points")
        print(f"  Our average error: {model_avg:.2f} points")

        diff = vegas_avg - model_avg
        if diff > 0:
            print(f"  We were {diff:.2f} points BETTER on average")
        else:
            print(f"  We were {abs(diff):.2f} points WORSE on average")

        print(f"  Games we were more accurate: {model_better_count}/{len(vegas_games)} ({model_better_pct:.1f}%)")

    # Confidence analysis
    print(f"\nBY CONFIDENCE LEVEL")
    for conf_name, conf_min, conf_max in [('High (90%+)', 0.9, 1.0), ('Medium (70-90%)', 0.7, 0.9), ('Low (<70%)', 0.0, 0.7)]:
        subset = merged[(merged['confidence'] >= conf_min) & (merged['confidence'] < conf_max)]
        if len(subset) > 0:
            acc = (subset['correct_winner'].sum() / len(subset)) * 100
            print(f"  {conf_name}: {len(subset)} games, {acc:.1f}% accurate")

    # Biggest misses
    print(f"\n" + "=" * 80)
    print("BIGGEST UPSETS (High confidence incorrect predictions)")
    print("=" * 80)

    incorrect = merged[~merged['correct_winner']]
    if len(incorrect) > 0:
        upsets = incorrect.nlargest(min(5, len(incorrect)), 'confidence')
        for _, row in upsets.iterrows():
            print(f"\n{row['away_team_pred']} @ {row['home_team_pred']}")
            print(f"  Predicted: {row['predicted_winner']} ({row['confidence']:.1%} confident)")
            print(f"  Actual: {row['actual_winner']} won {row['away_score']}-{row['home_score']}")
            print(f"  Spread error: {row['spread_error']:.2f} points")

    # Best predictions
    print(f"\n" + "=" * 80)
    print("BEST SPREAD PREDICTIONS")
    print("=" * 80)

    best = merged.nsmallest(5, 'spread_error')
    for _, row in best.iterrows():
        print(f"\n{row['away_team_pred']} @ {row['home_team_pred']}")
        print(f"  Final: {row['away_score']}-{row['home_score']} (margin: {row['actual_margin']:.0f})")
        print(f"  Predicted spread: {row['predicted_spread']:.1f}, Error: {row['spread_error']:.2f} pts")
        if not pd.isna(row.get('vegas_error')):
            print(f"  Vegas error: {row['vegas_error']:.2f} pts")

    # Save results
    output = merged[[
        'home_team_pred', 'away_team_pred',
        'predicted_winner', 'confidence', 'predicted_spread',
        'home_score', 'away_score', 'actual_winner', 'actual_margin',
        'correct_winner', 'spread_error',
        'vegas_spread', 'vegas_error', 'model_better'
    ]].copy()

    output.columns = [
        'home_team', 'away_team',
        'predicted_winner', 'confidence', 'predicted_spread',
        'home_score', 'away_score', 'actual_winner', 'actual_margin',
        'correct_winner', 'spread_error',
        'vegas_spread', 'vegas_error', 'beat_vegas'
    ]

    output.to_csv('week_13_final_report.csv', index=False)
    print(f"\n" + "=" * 80)
    print(f"Detailed results saved to: week_13_final_report.csv")
    print("=" * 80)

if __name__ == '__main__':
    main()
