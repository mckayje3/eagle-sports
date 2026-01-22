"""
CBB Full Walk-Forward Validation

True online learning: train on all past games, predict next game, repeat.
Analyzes edge thresholds and pick patterns to find rule-based filters.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).parent / 'cbb_games.db'

# Constants
DECAY = 0.92
MIN_GAMES = 5
PREV_HALF_LIFE = 6.0


def main():
    conn = sqlite3.connect(str(DB_PATH))

    # Get all completed games with Vegas lines
    games = pd.read_sql_query('''
        SELECT
            g.game_id, g.season, g.date, g.game_date_eastern,
            g.home_team_id, g.away_team_id,
            g.home_score, g.away_score,
            ht.name as home_team, at.name as away_team,
            o.latest_spread as vegas_spread,
            o.latest_total as vegas_total
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
    ''', conn)
    conn.close()

    games['actual_spread'] = games['away_score'] - games['home_score']
    games['actual_total'] = games['home_score'] + games['away_score']

    print(f"Total games: {len(games)}")
    print(f"Seasons: {sorted(games['season'].unique())}")
    print(f"Games with Vegas spread: {games['vegas_spread'].notna().sum()}")

    # Team state tracking
    team_stats = defaultdict(lambda: defaultdict(lambda: {
        'ppg': [], 'papg': [], 'wts': [],
        'margins': [], 'wins': [],
    }))
    team_hca = defaultdict(lambda: defaultdict(lambda: {
        'home_margins': [], 'away_margins': []
    }))
    prev_ratings = {}
    prev_hca = {}
    last_game = {}
    league_avg = {'ppg': 72.0, 'papg': 72.0}

    def wavg(vals, wts):
        if not vals or not wts:
            return None
        n = min(len(vals), len(wts))
        return np.average(vals[-n:], weights=wts[-n:])

    def get_rest(tid, date):
        if tid not in last_game:
            return 3
        try:
            from datetime import datetime
            curr = datetime.strptime(date[:10], '%Y-%m-%d')
            last = datetime.strptime(last_game[tid][:10], '%Y-%m-%d')
            return max(0, min((curr - last).days - 1, 5))
        except Exception:
            return 2

    def get_stats(tid, season):
        td = team_stats[tid][season]
        n = len(td['ppg'])

        if n == 0:
            prev = prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', league_avg['ppg']),
                'papg': prev.get('papg', league_avg['papg']),
                'games': 0,
                'margins': [],
                'wins': [],
            }

        ppg = wavg(td['ppg'], td['wts'])
        papg = wavg(td['papg'], td['wts'])

        prev = prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', league_avg['papg']) + (1 - blend) * papg,
            'games': n,
            'margins': td['margins'],
            'wins': td['wins'],
        }

    def get_dynamic_hca(home_id, season):
        hd = team_hca[home_id][season]
        n_home = len(hd['home_margins'])
        n_away = len(hd['away_margins'])
        total = n_home + n_away

        default_hca = 3.5  # CBB default

        if total == 0:
            return prev_hca.get(home_id, default_hca)

        if n_home > 0 and n_away > 0:
            raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
            raw = max(-2, min(raw, 10))
        else:
            raw = default_hca

        shrunk = default_hca + 0.5 * (raw - default_hca)
        prev = prev_hca.get(home_id, default_hca)
        blend = 0.5 ** (total / 20.0)

        return blend * prev + (1 - blend) * shrunk

    def recent_form(margins, n=5):
        if len(margins) < n:
            return 0.0
        return float(np.mean(margins[-n:]))

    def get_features(hid, aid, season, date):
        hs = get_stats(hid, season)
        aws = get_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = get_rest(hid, date)
        ar = get_rest(aid, date)
        hca = get_dynamic_hca(hid, season)

        h_form = recent_form(hs['margins'])
        a_form = recent_form(aws['margins'])

        return np.array([
            hs['ppg'] - aws['ppg'],           # 0: PPG diff
            hs['papg'] - aws['papg'],         # 1: PAPG diff
            (hs['ppg'] - hs['papg']) - (aws['ppg'] - aws['papg']),  # 2: Net rating
            h_form - a_form,                   # 3: Form diff
            hr - ar,                           # 4: Rest diff
            1 if hr == 0 else 0,               # 5: Home B2B
            1 if ar == 0 else 0,               # 6: Away B2B
            hca,                               # 7: Dynamic HCA
            min(hs['games'] / 15, 1),          # 8: Home reliability
            min(aws['games'] / 15, 1),         # 9: Away reliability
        ])

    def update_team(tid, season, date, pf, pa, is_home):
        td = team_stats[tid][season]
        td['wts'] = [w * DECAY for w in td['wts']]
        td['ppg'].append(pf)
        td['papg'].append(pa)
        td['wts'].append(1.0)

        margin = pf - pa
        td['margins'].append(margin)
        td['wins'].append(1 if margin > 0 else 0)

        hd = team_hca[tid][season]
        if is_home:
            hd['home_margins'].append(margin)
        else:
            hd['away_margins'].append(-margin)

        last_game[tid] = date

    def set_prev_season(season):
        nonlocal prev_ratings, prev_hca
        prev = season - 1

        for tid in team_stats:
            if prev in team_stats[tid]:
                td = team_stats[tid][prev]
                if td['ppg']:
                    prev_ratings[tid] = {
                        'ppg': np.mean(td['ppg']),
                        'papg': np.mean(td['papg']),
                    }

        for tid in team_hca:
            if prev in team_hca[tid]:
                hd = team_hca[tid][prev]
                if hd['home_margins'] and hd['away_margins']:
                    raw = (np.mean(hd['home_margins']) - np.mean(hd['away_margins'])) / 2
                    raw = max(-2, min(raw, 10))
                    prev_hca[tid] = 3.5 + 0.5 * (raw - 3.5)

        last_game.clear()

    # Process by season
    seasons = sorted(games['season'].unique())
    X_all, y_all = [], []
    results = []

    for season in seasons:
        if season > seasons[0]:
            set_prev_season(season)
            prev_games = games[games['season'] == season - 1]
            if len(prev_games) > 0:
                league_avg['ppg'] = prev_games['home_score'].mean()
                league_avg['papg'] = prev_games['away_score'].mean()

        season_games = games[games['season'] == season]

        for _, g in season_games.iterrows():
            feat = get_features(g['home_team_id'], g['away_team_id'], season, g['date'])

            if feat is not None and pd.notna(g['vegas_spread']):
                # Train on all previous data and predict this game
                if len(X_all) >= 100:  # Need minimum training data
                    X_train = np.array(X_all)
                    y_train = np.array(y_all)

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_train)

                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_scaled, y_train)

                    feat_scaled = scaler.transform(feat.reshape(1, -1))
                    pred = ridge.predict(feat_scaled)[0]

                    # Calculate edge
                    edge = pred - g['vegas_spread']

                    # Record result
                    results.append({
                        'game_id': g['game_id'],
                        'season': season,
                        'date': g['date'],
                        'home_team': g['home_team'],
                        'away_team': g['away_team'],
                        'pred': pred,
                        'actual': g['actual_spread'],
                        'vegas': g['vegas_spread'],
                        'edge': edge,
                        'games_trained': len(X_all),
                        # Pick classification
                        'pick_home': edge < 0,  # Negative edge = pick home
                        'vegas_home_fav': g['vegas_spread'] < 0,
                        'abs_vegas': abs(g['vegas_spread']),
                    })

                X_all.append(feat)
                y_all.append(g['actual_spread'])

            # Update team states
            update_team(g['home_team_id'], season, g['date'],
                       g['home_score'], g['away_score'], is_home=True)
            update_team(g['away_team_id'], season, g['date'],
                       g['away_score'], g['home_score'], is_home=False)

    # Analyze results
    df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("CBB WALK-FORWARD RESULTS")
    print(f"{'='*70}")
    print(f"\nTotal predictions: {len(df)}")
    print(f"Seasons covered: {df['season'].unique()}")

    # Overall accuracy
    mae_model = np.abs(df['pred'] - df['actual']).mean()
    mae_vegas = np.abs(df['vegas'] - df['actual']).mean()

    print(f"\n{'Model':<15} {'MAE':<10}")
    print("-" * 30)
    print(f"{'Ridge':<15} {mae_model:.2f}")
    print(f"{'Vegas':<15} {mae_vegas:.2f}")
    print(f"{'Difference':<15} {mae_model - mae_vegas:+.2f}")

    # ATS by edge threshold
    print(f"\n{'='*70}")
    print("ATS BY EDGE THRESHOLD")
    print(f"{'='*70}")

    def calc_ats(df_subset):
        if len(df_subset) == 0:
            return 0, 0, 0

        # edge > 0 means pick away, edge < 0 means pick home
        # Win if pick direction matches actual vs vegas
        wins = 0
        for _, r in df_subset.iterrows():
            if r['edge'] > 0:  # Pick away
                if r['actual'] > r['vegas']:
                    wins += 1
            else:  # Pick home
                if r['actual'] < r['vegas']:
                    wins += 1

        total = len(df_subset)
        pct = wins / total * 100 if total > 0 else 0
        return wins, total, pct

    print(f"\n{'Threshold':<15} {'Record':<15} {'ATS %':<10} {'ROI':<10}")
    print("-" * 50)

    for lo, hi in [(0, 2), (2, 4), (4, 5), (5, 7), (7, 100)]:
        mask = (df['edge'].abs() >= lo) & (df['edge'].abs() < hi)
        wins, total, pct = calc_ats(df[mask])
        if total > 20:
            roi = (wins * 0.91 - (total - wins)) / total * 100
            print(f"{lo}-{hi} pts{'':<7} {wins}-{total-wins:<10} {pct:.1f}%{'':<5} {roi:+.1f}%")

    # Cumulative thresholds
    print(f"\n{'Cumulative':<15}")
    print("-" * 50)
    for thresh in [2, 3, 4, 5, 6, 7]:
        mask = df['edge'].abs() >= thresh
        wins, total, pct = calc_ats(df[mask])
        if total > 20:
            roi = (wins * 0.91 - (total - wins)) / total * 100
            print(f"{thresh}+ pts{'':<9} {wins}-{total-wins:<10} {pct:.1f}%{'':<5} {roi:+.1f}%")

    # By pick type
    print(f"\n{'='*70}")
    print("BY PICK TYPE (5+ pt edges)")
    print(f"{'='*70}")

    df_5plus = df[df['edge'].abs() >= 5].copy()

    # Home vs Away picks
    home_picks = df_5plus[df_5plus['pick_home']]
    away_picks = df_5plus[~df_5plus['pick_home']]

    h_wins, h_total, h_pct = calc_ats(home_picks)
    a_wins, a_total, a_pct = calc_ats(away_picks)

    print(f"\n{'Pick Type':<20} {'Record':<15} {'ATS %':<10}")
    print("-" * 45)
    print(f"{'Home picks':<20} {h_wins}-{h_total-h_wins:<10} {h_pct:.1f}%")
    print(f"{'Away picks':<20} {a_wins}-{a_total-a_wins:<10} {a_pct:.1f}%")

    # Favorite vs Underdog picks
    # Pick fav = (pick_home AND vegas_home_fav) OR (pick_away AND NOT vegas_home_fav)
    df_5plus['pick_fav'] = (
        (df_5plus['pick_home'] & df_5plus['vegas_home_fav']) |
        (~df_5plus['pick_home'] & ~df_5plus['vegas_home_fav'])
    )

    fav_picks = df_5plus[df_5plus['pick_fav']]
    dog_picks = df_5plus[~df_5plus['pick_fav']]

    f_wins, f_total, f_pct = calc_ats(fav_picks)
    d_wins, d_total, d_pct = calc_ats(dog_picks)

    print(f"{'Favorite picks':<20} {f_wins}-{f_total-f_wins:<10} {f_pct:.1f}%")
    print(f"{'Underdog picks':<20} {d_wins}-{d_total-d_wins:<10} {d_pct:.1f}%")

    # Road favorite picks (like NBA)
    df_5plus['road_fav_pick'] = ~df_5plus['pick_home'] & ~df_5plus['vegas_home_fav']
    road_fav = df_5plus[df_5plus['road_fav_pick']]
    rf_wins, rf_total, rf_pct = calc_ats(road_fav)

    print(f"{'Road fav picks':<20} {rf_wins}-{rf_total-rf_wins:<10} {rf_pct:.1f}%")

    # By spread size
    print(f"\n{'='*70}")
    print("BY VEGAS SPREAD SIZE (5+ pt edges)")
    print(f"{'='*70}")

    print(f"\n{'Spread Range':<20} {'Record':<15} {'ATS %':<10}")
    print("-" * 45)

    for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 100)]:
        mask = (df_5plus['abs_vegas'] >= lo) & (df_5plus['abs_vegas'] < hi)
        w, t, p = calc_ats(df_5plus[mask])
        if t > 10:
            print(f"Vegas {lo}-{hi} pts{'':<6} {w}-{t-w:<10} {p:.1f}%")

    # By season (test for consistency)
    print(f"\n{'='*70}")
    print("BY SEASON (2-4 pt edges - the promising range)")
    print(f"{'='*70}")

    df_2to4 = df[(df['edge'].abs() >= 2) & (df['edge'].abs() < 4)]

    print(f"\n{'Season':<15} {'Record':<15} {'ATS %':<10}")
    print("-" * 40)

    for season in sorted(df_2to4['season'].unique()):
        mask = df_2to4['season'] == season
        w, t, p = calc_ats(df_2to4[mask])
        if t > 10:
            print(f"{season:<15} {w}-{t-w:<10} {p:.1f}%")

    # IMPORTANT: Analyze why 5+ pt edges fail
    print(f"\n{'='*70}")
    print("ANALYSIS: WHY DO HIGH EDGES FAIL?")
    print(f"{'='*70}")

    df_5plus = df[df['edge'].abs() >= 5].copy()
    df_2to4 = df[(df['edge'].abs() >= 2) & (df['edge'].abs() < 4)].copy()

    # Add pick_fav and road_fav_pick to both dataframes
    df_5plus['pick_fav'] = (
        (df_5plus['pick_home'] & df_5plus['vegas_home_fav']) |
        (~df_5plus['pick_home'] & ~df_5plus['vegas_home_fav'])
    )
    df_5plus['road_fav_pick'] = ~df_5plus['pick_home'] & ~df_5plus['vegas_home_fav']

    df_2to4['pick_fav'] = (
        (df_2to4['pick_home'] & df_2to4['vegas_home_fav']) |
        (~df_2to4['pick_home'] & ~df_2to4['vegas_home_fav'])
    )
    df_2to4['road_fav_pick'] = ~df_2to4['pick_home'] & ~df_2to4['vegas_home_fav']

    # Compare characteristics
    print(f"\n{'Metric':<25} {'2-4 pt edges':<20} {'5+ pt edges':<20}")
    print("-" * 65)

    # Average Vegas spread
    print(f"{'Avg Vegas spread':<25} {df_2to4['abs_vegas'].mean():.1f}{'':<15} {df_5plus['abs_vegas'].mean():.1f}")

    # % picking favorite
    pct_fav_2to4 = df_2to4['pick_fav'].mean() * 100
    pct_fav_5plus = df_5plus['pick_fav'].mean() * 100
    print(f"{'% picking favorite':<25} {pct_fav_2to4:.1f}%{'':<14} {pct_fav_5plus:.1f}%")

    # % picking home
    pct_home_2to4 = df_2to4['pick_home'].mean() * 100
    pct_home_5plus = df_5plus['pick_home'].mean() * 100
    print(f"{'% picking home':<25} {pct_home_2to4:.1f}%{'':<14} {pct_home_5plus:.1f}%")

    # % road favorite picks
    pct_rf_2to4 = df_2to4['road_fav_pick'].mean() * 100
    pct_rf_5plus = df_5plus['road_fav_pick'].mean() * 100
    print(f"{'% road fav picks':<25} {pct_rf_2to4:.1f}%{'':<14} {pct_rf_5plus:.1f}%")

    # Test potential filters on 5+ pt edges
    print(f"\n{'='*70}")
    print("POTENTIAL FILTERS FOR 5+ PT EDGES")
    print(f"{'='*70}")

    print(f"\n{'Filter':<35} {'Record':<15} {'ATS %':<10}")
    print("-" * 60)

    # Exclude road fav picks
    no_road_fav = df_5plus[~df_5plus['road_fav_pick']]
    w, t, p = calc_ats(no_road_fav)
    print(f"{'Exclude road fav picks':<35} {w}-{t-w:<10} {p:.1f}%")

    # Only home picks
    w, t, p = calc_ats(df_5plus[df_5plus['pick_home']])
    print(f"{'Home picks only':<35} {w}-{t-w:<10} {p:.1f}%")

    # Only underdog picks
    w, t, p = calc_ats(df_5plus[~df_5plus['pick_fav']])
    print(f"{'Underdog picks only':<35} {w}-{t-w:<10} {p:.1f}%")

    # Home underdog picks (best in NBA)
    home_dog = df_5plus[df_5plus['pick_home'] & ~df_5plus['vegas_home_fav']]
    w, t, p = calc_ats(home_dog)
    print(f"{'Home underdog picks':<35} {w}-{t-w:<10} {p:.1f}%")

    # Close games only (Vegas < 10)
    close = df_5plus[df_5plus['abs_vegas'] < 10]
    w, t, p = calc_ats(close)
    print(f"{'Close games (Vegas < 10)':<35} {w}-{t-w:<10} {p:.1f}%")

    # Big favorites (Vegas 15+)
    big_fav = df_5plus[df_5plus['abs_vegas'] >= 15]
    w, t, p = calc_ats(big_fav)
    print(f"{'Big spread games (Vegas 15+)':<35} {w}-{t-w:<10} {p:.1f}%")

    # Combined: exclude road fav + close games
    combined = df_5plus[~df_5plus['road_fav_pick'] & (df_5plus['abs_vegas'] < 10)]
    w, t, p = calc_ats(combined)
    print(f"{'No road fav + close games':<35} {w}-{t-w:<10} {p:.1f}%")

    # SPECIAL: Analyze 2026 only (to match CLAUDE.md claim of 62.9% at 2-4pt edges)
    print(f"\n{'='*70}")
    print("2026 SEASON ONLY (to compare with previous analysis)")
    print(f"{'='*70}")

    df_2026 = df[df['season'] == 2026].copy()
    print(f"\n2026 games: {len(df_2026)}")

    print(f"\n{'Threshold':<15} {'Record':<15} {'ATS %':<10}")
    print("-" * 40)

    for lo, hi in [(0, 2), (2, 4), (4, 5), (5, 7), (7, 100)]:
        mask = (df_2026['edge'].abs() >= lo) & (df_2026['edge'].abs() < hi)
        w, t, p = calc_ats(df_2026[mask])
        if t > 5:
            print(f"{lo}-{hi} pts{'':<7} {w}-{t-w:<10} {p:.1f}%")

    # Cumulative for 2026
    print(f"\nCumulative:")
    for thresh in [2, 3, 4, 5]:
        mask = df_2026['edge'].abs() >= thresh
        w, t, p = calc_ats(df_2026[mask])
        if t > 10:
            print(f"  {thresh}+ pts: {w}-{t-w} ({p:.1f}%)")

    # Save detailed results for further analysis
    output_path = Path(__file__).parent / 'cbb_walkforward_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
