"""
Model Evaluation: Academic vs Practical Metrics

This script evaluates our models two ways:
1. ACADEMIC (MAE): Does our model beat Vegas in raw prediction accuracy?
2. PRACTICAL (Directional): When we disagree with Vegas, are we right about direction?

Evaluates both SPREADS and TOTALS (over/under).

Key insight: You don't need better MAE to make money. You need to correctly
identify which direction Vegas is wrong.

Usage:
    python model_evaluation.py           # Full evaluation (spreads + totals)
    python model_evaluation.py spreads   # Spreads only
    python model_evaluation.py totals    # Totals only
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import sys

DATABASES = [
    ('nfl_games.db', 'NFL'),
    ('cfb_games.db', 'CFB'),
    ('nba_games.db', 'NBA'),
    ('cbb_games.db', 'CBB')
]


def evaluate_sport(db_path, sport):
    """Load completed games with predictions and Vegas lines."""
    conn = sqlite3.connect(db_path)

    query = '''
    SELECT
        g.game_id,
        g.season,
        g.home_score,
        g.away_score,
        g.away_score - g.home_score as actual_spread,
        g.home_score + g.away_score as actual_total,
        o.latest_spread as vegas_spread,
        o.latest_total as vegas_total,
        o.predicted_home_score,
        o.predicted_away_score,
        o.predicted_away_score - o.predicted_home_score as model_spread,
        o.predicted_home_score + o.predicted_away_score as model_total
    FROM games g
    JOIN odds_and_predictions o ON g.game_id = o.game_id
    WHERE g.home_score IS NOT NULL
      AND o.predicted_home_score IS NOT NULL
      AND o.latest_spread IS NOT NULL
    '''

    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) == 0:
        return None

    # Calculate errors
    df['vegas_spread_error'] = abs(df['vegas_spread'] - df['actual_spread'])
    df['model_spread_error'] = abs(df['model_spread'] - df['actual_spread'])
    df['vegas_total_error'] = abs(df['vegas_total'] - df['actual_total'])
    df['model_total_error'] = abs(df['model_total'] - df['actual_total'])

    # Deviation from Vegas (positive = we favor away more, negative = we favor home more)
    df['deviation'] = df['model_spread'] - df['vegas_spread']
    df['deviation_abs'] = abs(df['deviation'])

    # Directional accuracy: Did actual move in our predicted direction from Vegas?
    # deviation > 0 means we think away is stronger -> actual should be > vegas
    # deviation < 0 means we think home is stronger -> actual should be < vegas
    df['cover_correct'] = (
        ((df['deviation'] > 0) & (df['actual_spread'] > df['vegas_spread'])) |
        ((df['deviation'] < 0) & (df['actual_spread'] < df['vegas_spread']))
    )

    df['sport'] = sport
    return df


def calc_ev(win_rate):
    """Calculate expected value per bet at -110 odds."""
    win_pct = win_rate / 100
    # Win: +1.00 unit, Lose: -1.10 units
    ev = win_pct * 1.00 - (1 - win_pct) * 1.10
    roi = ev / 1.10 * 100
    return ev, roi


def print_sport_analysis(df, sport):
    """Print detailed analysis for one sport."""
    print(f'\n{sport} ({len(df)} games with predictions + Vegas)')
    print('-' * 50)

    # MAE comparison
    vegas_mae = df['vegas_spread_error'].mean()
    model_mae = df['model_spread_error'].mean()
    vegas_total_mae = df['vegas_total_error'].mean()
    model_total_mae = df['model_total_error'].mean()

    print(f'SPREAD MAE:  Vegas={vegas_mae:.2f} | Model={model_mae:.2f} | Diff={model_mae-vegas_mae:+.2f}')
    print(f'TOTAL MAE:   Vegas={vegas_total_mae:.2f} | Model={model_total_mae:.2f} | Diff={model_total_mae-vegas_total_mae:+.2f}')

    # Directional accuracy at thresholds
    print(f'\nDIRECTIONAL ACCURACY (when we disagree with Vegas):')
    for thresh in [0, 3, 5, 7, 10]:
        subset = df[df['deviation_abs'] >= thresh]
        if len(subset) > 0:
            pct = subset['cover_correct'].mean() * 100
            print(f'  Deviation >= {thresh:2d} pts: {pct:5.1f}% correct ({len(subset):4d} games)')

    # Split by direction
    print(f'\nBY DIRECTION:')
    home_bias = df[df['deviation'] < 0]
    away_bias = df[df['deviation'] > 0]

    if len(home_bias) > 0:
        home_pct = home_bias['cover_correct'].mean() * 100
        print(f'  Model favors HOME more: {home_pct:.1f}% ({len(home_bias)} games)')
    if len(away_bias) > 0:
        away_pct = away_bias['cover_correct'].mean() * 100
        print(f'  Model favors AWAY more: {away_pct:.1f}% ({len(away_bias)} games)')


def find_edges(results, min_sample=10, min_win_rate=55):
    """Find profitable betting edges across all sports."""
    edges = []

    for sport, df in results.items():
        for thresh in [3, 5, 7, 10]:
            for direction, label in [(-1, 'HOME'), (1, 'AWAY')]:
                if direction == -1:
                    subset = df[(df['deviation'] < 0) & (df['deviation_abs'] >= thresh)]
                else:
                    subset = df[(df['deviation'] > 0) & (df['deviation_abs'] >= thresh)]

                if len(subset) >= min_sample:
                    pct = subset['cover_correct'].mean() * 100
                    if pct >= min_win_rate:
                        ev, roi = calc_ev(pct)
                        edges.append({
                            'sport': sport,
                            'direction': label,
                            'threshold': thresh,
                            'win_rate': pct,
                            'games': len(subset),
                            'ev': ev,
                            'roi': roi
                        })

    return sorted(edges, key=lambda x: -x['roi'])


# =============================================================================
# TOTALS (OVER/UNDER) EVALUATION
# =============================================================================

def evaluate_totals(db_path, sport):
    """Load completed games for totals analysis."""
    conn = sqlite3.connect(db_path)

    query = '''
    SELECT
        g.game_id,
        g.home_score + g.away_score as actual_total,
        o.latest_total as vegas_total,
        o.predicted_home_score + o.predicted_away_score as model_total
    FROM games g
    JOIN odds_and_predictions o ON g.game_id = o.game_id
    WHERE g.home_score IS NOT NULL
      AND o.predicted_home_score IS NOT NULL
      AND o.latest_total IS NOT NULL
    '''

    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) == 0:
        return None

    # Deviation: positive = we predict OVER, negative = we predict UNDER
    df['deviation'] = df['model_total'] - df['vegas_total']
    df['deviation_abs'] = abs(df['deviation'])

    # Did actual move in our direction?
    df['ou_correct'] = (
        ((df['deviation'] > 0) & (df['actual_total'] > df['vegas_total'])) |
        ((df['deviation'] < 0) & (df['actual_total'] < df['vegas_total']))
    )

    # MAE
    df['vegas_error'] = abs(df['vegas_total'] - df['actual_total'])
    df['model_error'] = abs(df['model_total'] - df['actual_total'])

    df['sport'] = sport
    return df


def print_totals_analysis(df, sport):
    """Print detailed totals analysis for one sport."""
    print(f'\n{sport} ({len(df)} games)')
    print('-' * 50)

    # MAE
    vegas_mae = df['vegas_error'].mean()
    model_mae = df['model_error'].mean()
    print(f'TOTAL MAE:  Vegas={vegas_mae:.2f} | Model={model_mae:.2f} | Diff={model_mae-vegas_mae:+.2f}')

    # Directional accuracy at thresholds
    print(f'\nDIRECTIONAL ACCURACY:')
    for thresh in [0, 3, 5, 7, 10]:
        subset = df[df['deviation_abs'] >= thresh]
        if len(subset) > 0:
            pct = subset['ou_correct'].mean() * 100
            print(f'  Deviation >= {thresh:2d} pts: {pct:5.1f}% correct ({len(subset):4d} games)')

    # Split by direction
    print(f'\nBY DIRECTION:')
    over_bias = df[df['deviation'] > 0]
    under_bias = df[df['deviation'] < 0]

    if len(over_bias) > 0:
        over_pct = over_bias['ou_correct'].mean() * 100
        print(f'  Model predicts OVER:  {over_pct:.1f}% ({len(over_bias)} games)')
    if len(under_bias) > 0:
        under_pct = under_bias['ou_correct'].mean() * 100
        print(f'  Model predicts UNDER: {under_pct:.1f}% ({len(under_bias)} games)')


def find_totals_edges(results, min_sample=10, min_win_rate=55):
    """Find profitable O/U betting edges across all sports."""
    edges = []

    for sport, df in results.items():
        for thresh in [3, 5, 7, 10]:
            for direction, label in [(1, 'OVER'), (-1, 'UNDER')]:
                if direction == 1:
                    subset = df[(df['deviation'] > 0) & (df['deviation_abs'] >= thresh)]
                else:
                    subset = df[(df['deviation'] < 0) & (df['deviation_abs'] >= thresh)]

                if len(subset) >= min_sample:
                    pct = subset['ou_correct'].mean() * 100
                    if pct >= min_win_rate:
                        ev, roi = calc_ev(pct)
                        edges.append({
                            'sport': sport,
                            'direction': label,
                            'threshold': thresh,
                            'win_rate': pct,
                            'games': len(subset),
                            'ev': ev,
                            'roi': roi
                        })

    return sorted(edges, key=lambda x: -x['roi'])


def run_totals_evaluation():
    """Run totals (O/U) model evaluation."""
    print('=' * 70)
    print('TOTALS (OVER/UNDER) EVALUATION')
    print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 70)

    results = {}
    for db, sport in DATABASES:
        df = evaluate_totals(db, sport)
        if df is not None and len(df) > 0:
            results[sport] = df
            print_totals_analysis(df, sport)

    # Find and print edges
    edges = find_totals_edges(results)

    print('\n' + '=' * 70)
    print('TOTALS EDGES (>= 55% win rate, min 10 games)')
    print('=' * 70)

    if edges:
        print(f'\n{"Sport":<6} {"Edge":<25} {"Win%":>7} {"Games":>6} {"EV/bet":>10} {"ROI":>8}')
        print('-' * 65)

        for edge in edges:
            desc = f"{edge['direction']} bias @ {edge['threshold']}+ pts"
            print(f"{edge['sport']:<6} {desc:<25} {edge['win_rate']:6.1f}% {edge['games']:6d} {edge['ev']:+.3f}u    {edge['roi']:+6.1f}%")
    else:
        print('\nNo profitable O/U edges found.')

    return results, edges


# =============================================================================
# SPREAD EVALUATION
# =============================================================================

def run_spread_evaluation():
    """Run spread model evaluation."""
    print('=' * 70)
    print('SPREAD EVALUATION: MAE vs VEGAS + DIRECTIONAL ACCURACY')
    print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 70)

    results = {}
    for db, sport in DATABASES:
        df = evaluate_sport(db, sport)
        if df is not None and len(df) > 0:
            results[sport] = df
            print_sport_analysis(df, sport)

    # Find and print edges
    edges = find_edges(results)

    print('\n' + '=' * 70)
    print('PROFITABLE EDGES (>= 55% win rate, min 10 games)')
    print('=' * 70)
    print(f'\n{"Sport":<6} {"Edge":<25} {"Win%":>7} {"Games":>6} {"EV/bet":>10} {"ROI":>8}')
    print('-' * 65)

    for edge in edges:
        desc = f"{edge['direction']} bias @ {edge['threshold']}+ pts"
        print(f"{edge['sport']:<6} {desc:<25} {edge['win_rate']:6.1f}% {edge['games']:6d} {edge['ev']:+.3f}u    {edge['roi']:+6.1f}%")

    return results, edges


# =============================================================================
# COMBINED EVALUATION
# =============================================================================

def run_full_evaluation():
    """Run both spread and totals evaluation."""
    spread_results, spread_edges = run_spread_evaluation()
    print('\n\n')
    totals_results, totals_edges = run_totals_evaluation()

    # Combined summary
    print('\n' + '=' * 70)
    print('COMBINED SUMMARY')
    print('=' * 70)
    print('''
ACADEMIC (MAE): Vegas beats our model on spread/total accuracy for most sports.
This is expected - Vegas has more resources and information.

PRACTICAL (Directional): When we strongly disagree with Vegas, we can be
profitable despite worse MAE. The model adds value by identifying systematic
biases in Vegas lines.

KEY INSIGHT: We don't need to be more accurate than Vegas.
We need to know WHICH DIRECTION Vegas is wrong.

Breakeven at -110 odds: 52.38%
''')

    # Top edges across both
    all_edges = []
    for e in spread_edges:
        e['type'] = 'SPREAD'
        all_edges.append(e)
    for e in totals_edges:
        e['type'] = 'TOTAL'
        all_edges.append(e)

    all_edges = sorted(all_edges, key=lambda x: -x['roi'])[:10]

    print('TOP 10 EDGES (ALL MARKETS):')
    print(f'{"Type":<7} {"Sport":<5} {"Edge":<22} {"Win%":>6} {"Games":>5} {"ROI":>7}')
    print('-' * 55)
    for e in all_edges:
        desc = f"{e['direction']} @ {e['threshold']}+ pts"
        print(f"{e['type']:<7} {e['sport']:<5} {desc:<22} {e['win_rate']:5.1f}% {e['games']:5d} {e['roi']:+6.1f}%")

    return {
        'spread_results': spread_results,
        'spread_edges': spread_edges,
        'totals_results': totals_results,
        'totals_edges': totals_edges
    }


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if mode == 'spreads':
        run_spread_evaluation()
    elif mode == 'totals':
        run_totals_evaluation()
    else:
        run_full_evaluation()
