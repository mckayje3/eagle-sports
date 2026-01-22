"""
NBA Totals Walk-Forward Analysis

Analyze if Ridge V2 or other approaches have any edge on totals.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).parent / 'nba_games.db'


def main():
    conn = sqlite3.connect(str(DB_PATH))

    games = pd.read_sql_query('''
        SELECT
            g.game_id, g.season, g.date,
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
    print(f"Games with Vegas total: {games['vegas_total'].notna().sum()}")
    print(f"Seasons: {sorted(games['season'].unique())}")

    # Team tracking
    DECAY = 0.93
    MIN_GAMES = 10
    PREV_HALF_LIFE = 6.0

    team_stats = defaultdict(lambda: defaultdict(lambda: {
        'ppg': [], 'papg': [], 'wts': [],
        'totals': [],  # Track game totals
    }))
    prev_ratings = {}
    last_game = {}
    league_avg = {'ppg': 115.0, 'total': 230.0}

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
                'papg': prev.get('papg', league_avg['ppg']),
                'pace': prev.get('pace', league_avg['total']),
                'games': 0,
            }

        ppg = wavg(td['ppg'], td['wts'])
        papg = wavg(td['papg'], td['wts'])
        pace = wavg(td['totals'], td['wts']) if td['totals'] else league_avg['total']

        prev = prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)

        return {
            'ppg': blend * prev.get('ppg', league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', league_avg['ppg']) + (1 - blend) * papg,
            'pace': blend * prev.get('pace', league_avg['total']) + (1 - blend) * pace,
            'games': n,
        }

    def get_total_features(hid, aid, season, date):
        hs = get_stats(hid, season)
        aws = get_stats(aid, season)

        if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
            return None

        hr = get_rest(hid, date)
        ar = get_rest(aid, date)

        # Multiple approaches to predict total

        # Simple: average of team paces
        simple_total = (hs['pace'] + aws['pace']) / 2

        # Matchup-based: home offense vs away defense, away offense vs home defense
        matchup_total = (hs['ppg'] + aws['papg']) / 2 + (aws['ppg'] + hs['papg']) / 2

        features = np.array([
            hs['ppg'] + aws['ppg'],           # Combined PPG
            hs['papg'] + aws['papg'],         # Combined PAPG
            hs['pace'],                        # Home pace
            aws['pace'],                       # Away pace
            (hs['ppg'] + hs['papg']) / 2,     # Home game average
            (aws['ppg'] + aws['papg']) / 2,   # Away game average
            1 if hr == 0 else 0,               # Home B2B
            1 if ar == 0 else 0,               # Away B2B
            min(hs['games'] / 30, 1),          # Home reliability
            min(aws['games'] / 30, 1),         # Away reliability
        ])

        return features, simple_total, matchup_total

    def update_team(tid, season, date, pf, pa, game_total):
        td = team_stats[tid][season]
        td['wts'] = [w * DECAY for w in td['wts']]
        td['ppg'].append(pf)
        td['papg'].append(pa)
        td['totals'].append(game_total)
        td['wts'].append(1.0)
        last_game[tid] = date

    def set_prev_season(season):
        nonlocal prev_ratings
        prev = season - 1

        for tid in team_stats:
            if prev in team_stats[tid]:
                td = team_stats[tid][prev]
                if td['ppg']:
                    prev_ratings[tid] = {
                        'ppg': np.mean(td['ppg']),
                        'papg': np.mean(td['papg']),
                        'pace': np.mean(td['totals']) if td['totals'] else league_avg['total'],
                    }
        last_game.clear()

    # Walk-forward
    seasons = sorted(games['season'].unique())
    X_all, y_all = [], []
    results = []

    for season in seasons:
        if season > seasons[0]:
            set_prev_season(season)
            prev_games = games[games['season'] == season - 1]
            if len(prev_games) > 0:
                league_avg['ppg'] = (prev_games['home_score'].mean() + prev_games['away_score'].mean()) / 2
                league_avg['total'] = prev_games['actual_total'].mean()

        season_games = games[games['season'] == season]

        for _, g in season_games.iterrows():
            result = get_total_features(g['home_team_id'], g['away_team_id'], season, g['date'])

            if result is not None and pd.notna(g['vegas_total']):
                feat, simple_pred, matchup_pred = result

                # Train Ridge on all previous data
                if len(X_all) >= 100:
                    X_train = np.array(X_all)
                    y_train = np.array(y_all)

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_train)

                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_scaled, y_train)

                    feat_scaled = scaler.transform(feat.reshape(1, -1))
                    ridge_pred = ridge.predict(feat_scaled)[0]

                    # Calculate edges
                    ridge_edge = ridge_pred - g['vegas_total']
                    simple_edge = simple_pred - g['vegas_total']
                    matchup_edge = matchup_pred - g['vegas_total']

                    results.append({
                        'game_id': g['game_id'],
                        'season': season,
                        'date': g['date'],
                        'home_team': g['home_team'],
                        'away_team': g['away_team'],
                        'actual': g['actual_total'],
                        'vegas': g['vegas_total'],
                        'ridge_pred': ridge_pred,
                        'simple_pred': simple_pred,
                        'matchup_pred': matchup_pred,
                        'ridge_edge': ridge_edge,
                        'simple_edge': simple_edge,
                        'matchup_edge': matchup_edge,
                    })

                X_all.append(feat)
                y_all.append(g['actual_total'])

            # Update team states
            game_total = g['home_score'] + g['away_score']
            update_team(g['home_team_id'], season, g['date'],
                       g['home_score'], g['away_score'], game_total)
            update_team(g['away_team_id'], season, g['date'],
                       g['away_score'], g['home_score'], game_total)

    df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("NBA TOTALS WALK-FORWARD RESULTS")
    print(f"{'='*70}")
    print(f"\nTotal predictions: {len(df)}")

    # MAE comparison
    print(f"\n{'Model':<20} {'MAE':<10} {'vs Vegas':<10}")
    print("-" * 40)

    mae_ridge = np.abs(df['ridge_pred'] - df['actual']).mean()
    mae_simple = np.abs(df['simple_pred'] - df['actual']).mean()
    mae_matchup = np.abs(df['matchup_pred'] - df['actual']).mean()
    mae_vegas = np.abs(df['vegas'] - df['actual']).mean()

    print(f"{'Vegas':<20} {mae_vegas:.2f}")
    print(f"{'Ridge':<20} {mae_ridge:.2f}{'':<5} {mae_ridge - mae_vegas:+.2f}")
    print(f"{'Simple (pace avg)':<20} {mae_simple:.2f}{'':<5} {mae_simple - mae_vegas:+.2f}")
    print(f"{'Matchup':<20} {mae_matchup:.2f}{'':<5} {mae_matchup - mae_vegas:+.2f}")

    # O/U Analysis function
    def calc_ou(df_subset, edge_col):
        if len(df_subset) == 0:
            return 0, 0, 0, 0, 0, 0

        # OVER: edge > 0 (model predicts higher than Vegas)
        # UNDER: edge < 0 (model predicts lower than Vegas)
        over_mask = df_subset[edge_col] > 0
        under_mask = df_subset[edge_col] < 0

        over_wins = ((df_subset.loc[over_mask, 'actual'] > df_subset.loc[over_mask, 'vegas'])).sum()
        over_total = over_mask.sum()

        under_wins = ((df_subset.loc[under_mask, 'actual'] < df_subset.loc[under_mask, 'vegas'])).sum()
        under_total = under_mask.sum()

        total_wins = over_wins + under_wins
        total_games = over_total + under_total

        return over_wins, over_total, under_wins, under_total, total_wins, total_games

    # Overall O/U by model
    print(f"\n{'='*70}")
    print("OVER/UNDER PERFORMANCE BY MODEL")
    print(f"{'='*70}")

    for model, edge_col in [('Ridge', 'ridge_edge'), ('Simple', 'simple_edge'), ('Matchup', 'matchup_edge')]:
        ow, ot, uw, ut, tw, tg = calc_ou(df, edge_col)
        print(f"\n{model}:")
        print(f"  OVER:  {ow}-{ot-ow} ({ow/ot*100:.1f}%)" if ot > 0 else "  OVER: N/A")
        print(f"  UNDER: {uw}-{ut-uw} ({uw/ut*100:.1f}%)" if ut > 0 else "  UNDER: N/A")
        print(f"  TOTAL: {tw}-{tg-tw} ({tw/tg*100:.1f}%)" if tg > 0 else "  TOTAL: N/A")

    # By edge threshold (Ridge)
    print(f"\n{'='*70}")
    print("RIDGE TOTALS BY EDGE THRESHOLD")
    print(f"{'='*70}")

    print(f"\n{'Threshold':<15} {'OVER':<20} {'UNDER':<20} {'TOTAL':<15}")
    print("-" * 70)

    for thresh in [0, 3, 5, 7, 10]:
        mask = df['ridge_edge'].abs() >= thresh
        subset = df[mask]
        ow, ot, uw, ut, tw, tg = calc_ou(subset, 'ridge_edge')

        if tg > 20:
            over_pct = f"{ow}/{ot} ({ow/ot*100:.1f}%)" if ot > 0 else "N/A"
            under_pct = f"{uw}/{ut} ({uw/ut*100:.1f}%)" if ut > 0 else "N/A"
            total_pct = f"{tw}/{tg} ({tw/tg*100:.1f}%)"
            print(f"{thresh}+ pts{'':<9} {over_pct:<20} {under_pct:<20} {total_pct:<15}")

    # Check for directional bias
    print(f"\n{'='*70}")
    print("DIRECTIONAL ANALYSIS (Is model biased OVER or UNDER?)")
    print(f"{'='*70}")

    # Check if model systematically over/under predicts
    ridge_bias = (df['ridge_pred'] - df['actual']).mean()
    vegas_bias = (df['vegas'] - df['actual']).mean()

    print(f"\nRidge avg prediction error: {ridge_bias:+.2f} pts (positive = predicts too high)")
    print(f"Vegas avg prediction error: {vegas_bias:+.2f} pts")

    # If model is biased, maybe fade the bias?
    print(f"\n{'='*70}")
    print("FADE ANALYSIS (Bet opposite of model direction)")
    print(f"{'='*70}")

    # When model says OVER by 5+ pts, what if we bet UNDER?
    print(f"\n{'Scenario':<40} {'Record':<15} {'Win %':<10}")
    print("-" * 65)

    # Fade OVER
    over_5 = df[df['ridge_edge'] >= 5]
    if len(over_5) > 10:
        # Fading = bet UNDER when model says OVER
        fade_wins = (over_5['actual'] < over_5['vegas']).sum()
        print(f"{'Fade OVER 5+ (bet UNDER)':<40} {fade_wins}-{len(over_5)-fade_wins:<10} {fade_wins/len(over_5)*100:.1f}%")

    over_7 = df[df['ridge_edge'] >= 7]
    if len(over_7) > 10:
        fade_wins = (over_7['actual'] < over_7['vegas']).sum()
        print(f"{'Fade OVER 7+ (bet UNDER)':<40} {fade_wins}-{len(over_7)-fade_wins:<10} {fade_wins/len(over_7)*100:.1f}%")

    # Fade UNDER
    under_5 = df[df['ridge_edge'] <= -5]
    if len(under_5) > 10:
        # Fading = bet OVER when model says UNDER
        fade_wins = (under_5['actual'] > under_5['vegas']).sum()
        print(f"{'Fade UNDER 5+ (bet OVER)':<40} {fade_wins}-{len(under_5)-fade_wins:<10} {fade_wins/len(under_5)*100:.1f}%")

    under_7 = df[df['ridge_edge'] <= -7]
    if len(under_7) > 10:
        fade_wins = (under_7['actual'] > under_7['vegas']).sum()
        print(f"{'Fade UNDER 7+ (bet OVER)':<40} {fade_wins}-{len(under_7)-fade_wins:<10} {fade_wins/len(under_7)*100:.1f}%")

    # By season
    print(f"\n{'='*70}")
    print("BY SEASON (Ridge 5+ pt edges)")
    print(f"{'='*70}")

    print(f"\n{'Season':<10} {'OVER':<20} {'UNDER':<20} {'TOTAL':<15}")
    print("-" * 65)

    for season in sorted(df['season'].unique()):
        mask = (df['season'] == season) & (df['ridge_edge'].abs() >= 5)
        subset = df[mask]
        ow, ot, uw, ut, tw, tg = calc_ou(subset, 'ridge_edge')

        if tg > 10:
            over_pct = f"{ow}/{ot} ({ow/ot*100:.1f}%)" if ot > 0 else "N/A"
            under_pct = f"{uw}/{ut} ({uw/ut*100:.1f}%)" if ut > 0 else "N/A"
            total_pct = f"{tw}/{tg} ({tw/tg*100:.1f}%)"
            print(f"{season:<10} {over_pct:<20} {under_pct:<20} {total_pct:<15}")

    # Back-to-back analysis
    print(f"\n{'='*70}")
    print("BACK-TO-BACK GAME ANALYSIS")
    print(f"{'='*70}")

    # Games where at least one team is on B2B
    # We need to recalculate this from features... let's use a simpler approach
    # by looking at games where the model's B2B features would have been set

    print("\n(B2B analysis requires additional feature tracking - skipped)")

    # Save results
    output_path = Path(__file__).parent / 'nba_totals_analysis_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
