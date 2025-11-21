"""
Analyze Deep-Eagle Predictions vs Vegas Lines
Run this after games complete to compare model performance vs Vegas
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def analyze_predictions(sport='CFB', week=None, season=2025):
    """
    Analyze completed predictions and compare to Vegas

    Args:
        sport: 'CFB' or 'NFL'
        week: Week number (None for all weeks)
        season: Season year
    """
    print("=" * 80)
    print(f"DEEP-EAGLE vs VEGAS ANALYSIS - {sport} {season}")
    print("=" * 80)

    # Load predictions from cache
    conn = sqlite3.connect('users.db')

    if week:
        query = """
            SELECT * FROM prediction_cache
            WHERE sport = ? AND season = ? AND week = ? AND game_completed = 1
        """
        df = pd.read_sql_query(query, conn, params=(sport.upper(), season, week))
    else:
        query = """
            SELECT * FROM prediction_cache
            WHERE sport = ? AND season = ? AND game_completed = 1
        """
        df = pd.read_sql_query(query, conn, params=(sport.upper(), season))

    conn.close()

    if len(df) == 0:
        print(f"No completed games found for {sport} {season}" + (f" Week {week}" if week else ""))
        return None

    print(f"\nAnalyzing {len(df)} completed games")
    if week:
        print(f"Week: {week}")

    # Calculate actual results
    df['actual_spread'] = df['actual_home_score'] - df['actual_away_score']
    df['actual_total'] = df['actual_home_score'] + df['actual_away_score']

    # Model errors
    df['model_spread_error'] = abs(df['predicted_spread'] - df['actual_spread'])
    df['model_total_error'] = abs(df['predicted_total'] - df['actual_total'])

    # Vegas errors (only where Vegas lines exist)
    df_with_vegas = df[df['vegas_spread'].notna()].copy()
    df_with_vegas['vegas_spread_error'] = abs(df_with_vegas['vegas_spread'] - df_with_vegas['actual_spread'])

    df_with_total = df[df['vegas_total'].notna()].copy()
    df_with_total['vegas_total_error'] = abs(df_with_total['vegas_total'] - df_with_total['actual_total'])

    # ATS (Against The Spread) Analysis
    # Model ATS: Did model pick the correct side?
    df['model_pick_home'] = df['predicted_spread'] > 0  # Model thinks home wins by more
    df['actual_home_covers'] = df['actual_spread'] > df['vegas_spread'].fillna(0)  # Home covered
    df['model_ats_correct'] = (
        (df['model_pick_home'] & df['actual_home_covers']) |
        (~df['model_pick_home'] & ~df['actual_home_covers'])
    )

    # Over/Under Analysis
    df['model_pick_over'] = df['predicted_total'] > df['vegas_total'].fillna(df['predicted_total'])
    df['actual_over'] = df['actual_total'] > df['vegas_total'].fillna(df['actual_total'])
    df['model_ou_correct'] = df['model_pick_over'] == df['actual_over']

    # Print Results
    print("\n" + "-" * 40)
    print("SPREAD PREDICTION ACCURACY")
    print("-" * 40)

    model_spread_mae = df['model_spread_error'].mean()
    print(f"Model Spread MAE:     {model_spread_mae:.2f} points")

    if len(df_with_vegas) > 0:
        vegas_spread_mae = df_with_vegas['vegas_spread_error'].mean()
        print(f"Vegas Spread MAE:     {vegas_spread_mae:.2f} points")
        print(f"Difference:           {model_spread_mae - vegas_spread_mae:+.2f} points")
        print(f"Games with Vegas:     {len(df_with_vegas)}")

        # ATS Record
        ats_correct = df_with_vegas['model_ats_correct'].sum()
        ats_total = len(df_with_vegas)
        ats_pct = ats_correct / ats_total * 100 if ats_total > 0 else 0
        print(f"\nModel ATS Record:     {ats_correct}-{ats_total - ats_correct} ({ats_pct:.1f}%)")
        print(f"Break-even ATS:       52.4% (to beat vig)")

    print("\n" + "-" * 40)
    print("TOTAL PREDICTION ACCURACY")
    print("-" * 40)

    model_total_mae = df['model_total_error'].mean()
    print(f"Model Total MAE:      {model_total_mae:.2f} points")

    if len(df_with_total) > 0:
        vegas_total_mae = df_with_total['vegas_total_error'].mean()
        print(f"Vegas Total MAE:      {vegas_total_mae:.2f} points")
        print(f"Difference:           {model_total_mae - vegas_total_mae:+.2f} points")
        print(f"Games with Vegas:     {len(df_with_total)}")

        # O/U Record
        ou_correct = df_with_total['model_ou_correct'].sum()
        ou_total = len(df_with_total)
        ou_pct = ou_correct / ou_total * 100 if ou_total > 0 else 0
        print(f"\nModel O/U Record:     {ou_correct}-{ou_total - ou_correct} ({ou_pct:.1f}%)")

    print("\n" + "-" * 40)
    print("STRAIGHT UP (PICKING WINNERS)")
    print("-" * 40)

    # SU (Straight Up) Analysis
    df['model_pick_winner'] = df['predicted_home_score'] > df['predicted_away_score']
    df['actual_winner_home'] = df['actual_home_score'] > df['actual_away_score']
    df['model_su_correct'] = df['model_pick_winner'] == df['actual_winner_home']

    su_correct = df['model_su_correct'].sum()
    su_total = len(df)
    su_pct = su_correct / su_total * 100 if su_total > 0 else 0
    print(f"Model SU Record:      {su_correct}-{su_total - su_correct} ({su_pct:.1f}%)")

    # Best/Worst predictions
    print("\n" + "-" * 40)
    print("BEST PREDICTIONS (Closest to actual)")
    print("-" * 40)
    best = df.nsmallest(5, 'model_spread_error')[['away_team', 'home_team', 'predicted_spread', 'actual_spread', 'model_spread_error']]
    for _, row in best.iterrows():
        print(f"  {row['away_team'][:20]:<20} @ {row['home_team'][:20]:<20}  Pred: {row['predicted_spread']:+.1f}  Actual: {row['actual_spread']:+.0f}  Err: {row['model_spread_error']:.1f}")

    print("\n" + "-" * 40)
    print("WORST PREDICTIONS (Furthest from actual)")
    print("-" * 40)
    worst = df.nlargest(5, 'model_spread_error')[['away_team', 'home_team', 'predicted_spread', 'actual_spread', 'model_spread_error']]
    for _, row in worst.iterrows():
        print(f"  {row['away_team'][:20]:<20} @ {row['home_team'][:20]:<20}  Pred: {row['predicted_spread']:+.1f}  Actual: {row['actual_spread']:+.0f}  Err: {row['model_spread_error']:.1f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
Sport:                {sport}
Season:               {season}
Week(s):              {'All' if week is None else week}
Games Analyzed:       {len(df)}
Games with Vegas:     {len(df_with_vegas)}

Model Performance:
  - Spread MAE:       {model_spread_mae:.2f} pts
  - Total MAE:        {model_total_mae:.2f} pts
  - SU Record:        {su_correct}-{su_total - su_correct} ({su_pct:.1f}%)
""")

    if len(df_with_vegas) > 0:
        print(f"""
vs Vegas:
  - ATS Record:       {ats_correct}-{ats_total - ats_correct} ({ats_pct:.1f}%)
  - O/U Record:       {ou_correct}-{ou_total - ou_correct} ({ou_pct:.1f}%)
  - Spread MAE Diff:  {model_spread_mae - vegas_spread_mae:+.2f} pts {'(worse)' if model_spread_mae > vegas_spread_mae else '(better!)'}
  - Total MAE Diff:   {model_total_mae - vegas_total_mae:+.2f} pts {'(worse)' if model_total_mae > vegas_total_mae else '(better!)'}
""")

    # Save detailed results to CSV
    output_file = f'analysis_{sport}_{season}_week{week if week else "all"}_{datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

    return df


def update_completed_games():
    """Update prediction cache with actual scores from completed games"""
    print("Updating prediction cache with completed game results...")

    # Get predictions that need updating
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Get pending predictions
    cursor.execute("""
        SELECT game_id, sport FROM prediction_cache
        WHERE game_completed = 0 OR actual_home_score IS NULL
    """)
    pending = cursor.fetchall()
    print(f"Found {len(pending)} predictions to check")

    updated = 0

    for game_id, sport in pending:
        # Check appropriate database
        db_path = 'cfb_games.db' if sport == 'CFB' else 'nfl_games.db'

        try:
            game_conn = sqlite3.connect(db_path)
            game_cursor = game_conn.cursor()

            game_cursor.execute("""
                SELECT completed, home_score, away_score
                FROM games WHERE game_id = ?
            """, (game_id,))
            result = game_cursor.fetchone()
            game_conn.close()

            if result and result[0] == 1:  # Game completed
                cursor.execute("""
                    UPDATE prediction_cache
                    SET game_completed = 1,
                        actual_home_score = ?,
                        actual_away_score = ?
                    WHERE game_id = ? AND sport = ?
                """, (result[1], result[2], game_id, sport))
                updated += 1

        except Exception as e:
            print(f"  Error checking game {game_id}: {e}")

    conn.commit()
    conn.close()
    print(f"Updated {updated} predictions with actual results")


def main():
    """Main analysis function"""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Deep-Eagle predictions vs Vegas')
    parser.add_argument('--sport', default='both', help='Sport: CFB, NFL, or both')
    parser.add_argument('--week', type=int, help='Week number (omit for all weeks)')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    parser.add_argument('--update', action='store_true', help='Update completed games first')

    args = parser.parse_args()

    if args.update:
        update_completed_games()
        print()

    if args.sport.lower() == 'both':
        print("\n" + "=" * 80)
        print("COLLEGE FOOTBALL ANALYSIS")
        print("=" * 80)
        analyze_predictions('CFB', args.week, args.season)

        print("\n" + "=" * 80)
        print("NFL ANALYSIS")
        print("=" * 80)
        analyze_predictions('NFL', args.week, args.season)
    else:
        analyze_predictions(args.sport.upper(), args.week, args.season)


if __name__ == '__main__':
    main()
