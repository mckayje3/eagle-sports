"""
NBA Per-Team HCA Analysis

Tests whether per-team home court advantage improves MAE over flat 2.0
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent / 'nba_games.db'


def load_games():
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query('''
        SELECT
            g.game_id, g.season, g.date, g.season_type,
            g.home_team_id, g.away_team_id,
            g.home_score, g.away_score,
            ht.abbreviation as home_abbr, at.abbreviation as away_abbr,
            hs.field_goal_pct as home_fg, hs.three_point_pct as home_three,
            hs.total_rebounds as home_reb, hs.assists as home_ast, hs.turnovers as home_tov,
            aws.field_goal_pct as away_fg, aws.three_point_pct as away_three,
            aws.total_rebounds as away_reb, aws.assists as away_ast, aws.turnovers as away_tov,
            o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
    ''', conn)
    conn.close()

    games['actual_spread'] = games['away_score'] - games['home_score']
    games['actual_total'] = games['home_score'] + games['away_score']
    return games


def calculate_team_hca(games: pd.DataFrame, season: int) -> dict:
    """Calculate per-team HCA from a single season."""
    season_games = games[games['season'] == season]

    team_hca = {}
    team_abbr = {}

    for team_id in season_games['home_team_id'].unique():
        home_games = season_games[season_games['home_team_id'] == team_id]
        away_games = season_games[season_games['away_team_id'] == team_id]

        if len(home_games) >= 10 and len(away_games) >= 10:
            home_margin = (home_games['home_score'] - home_games['away_score']).mean()
            away_margin = (away_games['away_score'] - away_games['home_score']).mean()
            team_hca[team_id] = (home_margin - away_margin) / 2
            team_abbr[team_id] = home_games['home_abbr'].iloc[0]
        else:
            team_hca[team_id] = 2.0

    return team_hca, team_abbr


def run_walk_forward_test(games: pd.DataFrame, test_season: int, team_hca: dict, decay: float = 0.95):
    """Run walk-forward prediction test on a season."""

    team_stats = defaultdict(lambda: {'ppg': [], 'papg': [], 'wts': []})

    def get_weighted_avg(values, weights):
        if not values:
            return None
        n = min(len(values), len(weights))
        return np.average(values[-n:], weights=weights[-n:])

    def get_team_stats(team_id):
        ts = team_stats[team_id]
        if not ts['ppg']:
            return None
        return {
            'ppg': get_weighted_avg(ts['ppg'], ts['wts']),
            'papg': get_weighted_avg(ts['papg'], ts['wts']),
            'games': len(ts['ppg'])
        }

    def update_team(team_id, pts_for, pts_against):
        ts = team_stats[team_id]
        ts['wts'] = [w * decay for w in ts['wts']]
        ts['ppg'].append(pts_for)
        ts['papg'].append(pts_against)
        ts['wts'].append(1.0)

    season_games = games[games['season'] == test_season].copy()

    results = []

    for idx, g in season_games.iterrows():
        home_stats = get_team_stats(g['home_team_id'])
        away_stats = get_team_stats(g['away_team_id'])

        if home_stats and away_stats and home_stats['games'] >= 5 and away_stats['games'] >= 5:
            ppg_diff = home_stats['ppg'] - away_stats['ppg']

            # Flat HCA
            pred_flat = -ppg_diff - 2.0

            # Per-team HCA
            home_hca = team_hca.get(g['home_team_id'], 2.0)
            pred_team = -ppg_diff - home_hca

            results.append({
                'game_id': g['game_id'],
                'home_abbr': g['home_abbr'],
                'away_abbr': g['away_abbr'],
                'home_hca': home_hca,
                'pred_flat': pred_flat,
                'pred_team': pred_team,
                'actual': g['actual_spread'],
                'vegas': g['vegas_spread'],
                'games_played': min(home_stats['games'], away_stats['games']),
            })

        # Update stats
        update_team(g['home_team_id'], g['home_score'], g['away_score'])
        update_team(g['away_team_id'], g['away_score'], g['home_score'])

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("NBA PER-TEAM HCA ANALYSIS")
    print("=" * 70)

    games = load_games()
    print(f"Loaded {len(games)} games")

    # Calculate HCA from 2023 to use for 2024
    team_hca_2023, team_abbr = calculate_team_hca(games, 2023)

    print("\n" + "=" * 70)
    print("PER-TEAM HCA (from 2023 season)")
    print("=" * 70)

    sorted_hca = sorted(team_hca_2023.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'Team':<6} {'HCA':>6}")
    print("-" * 15)
    for tid, hca in sorted_hca:
        abbr = team_abbr.get(tid, str(tid))
        print(f"{abbr:<6} {hca:>+6.2f}")

    hca_values = list(team_hca_2023.values())
    print(f"\nRange: {min(hca_values):.2f} to {max(hca_values):.2f}")
    print(f"Mean: {np.mean(hca_values):.2f}, Std: {np.std(hca_values):.2f}")

    # Run test on 2024
    print("\n" + "=" * 70)
    print("WALK-FORWARD TEST ON 2024 SEASON")
    print("=" * 70)

    results = run_walk_forward_test(games, 2024, team_hca_2023)

    # Filter to games with Vegas
    results_v = results[results['vegas'].notna()].copy()

    mae_flat = np.abs(results_v['pred_flat'] - results_v['actual']).mean()
    mae_team = np.abs(results_v['pred_team'] - results_v['actual']).mean()
    mae_vegas = np.abs(results_v['vegas'] - results_v['actual']).mean()

    mse_flat = ((results_v['pred_flat'] - results_v['actual']) ** 2).mean()
    mse_team = ((results_v['pred_team'] - results_v['actual']) ** 2).mean()
    mse_vegas = ((results_v['vegas'] - results_v['actual']) ** 2).mean()

    print(f"\nOverall (N={len(results_v)}):")
    print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10}")
    print("-" * 40)
    print(f"{'Flat HCA (2.0)':<20} {mae_flat:<10.3f} {np.sqrt(mse_flat):<10.3f}")
    print(f"{'Per-Team HCA':<20} {mae_team:<10.3f} {np.sqrt(mse_team):<10.3f}")
    print(f"{'Vegas':<20} {mae_vegas:<10.3f} {np.sqrt(mse_vegas):<10.3f}")
    print(f"\nImprovement from per-team HCA: {mae_flat - mae_team:+.4f} MAE")

    # By season segment
    print("\n" + "=" * 70)
    print("MAE BY GAMES INTO SEASON")
    print("=" * 70)

    segments = [
        (5, 15, 'Early (5-15)'),
        (16, 30, 'Mid-Early (16-30)'),
        (31, 50, 'Mid-Late (31-50)'),
        (51, 82, 'Late (51-82)'),
    ]

    print(f"\n{'Segment':<18} {'Flat':<8} {'Team':<8} {'Vegas':<8} {'Diff':<8} {'N':<6}")
    print("-" * 60)

    for min_g, max_g, label in segments:
        mask = (results_v['games_played'] >= min_g) & (results_v['games_played'] <= max_g)
        if mask.sum() > 20:
            seg = results_v[mask]
            mae_f = np.abs(seg['pred_flat'] - seg['actual']).mean()
            mae_t = np.abs(seg['pred_team'] - seg['actual']).mean()
            mae_v = np.abs(seg['vegas'] - seg['actual']).mean()
            diff = mae_t - mae_f
            print(f"{label:<18} {mae_f:<8.3f} {mae_t:<8.3f} {mae_v:<8.3f} {diff:<+8.4f} {mask.sum():<6}")

    # Analysis by home team HCA
    print("\n" + "=" * 70)
    print("MAE BY HOME TEAM HCA MAGNITUDE")
    print("=" * 70)

    hca_bins = [
        (0, 1.5, 'Low HCA (<1.5)'),
        (1.5, 2.5, 'Medium HCA (1.5-2.5)'),
        (2.5, 3.5, 'High HCA (2.5-3.5)'),
        (3.5, 10, 'Very High HCA (>3.5)'),
    ]

    print(f"\n{'HCA Range':<22} {'Flat':<8} {'Team':<8} {'Vegas':<8} {'Diff':<8} {'N':<6}")
    print("-" * 65)

    for min_hca, max_hca, label in hca_bins:
        mask = (results_v['home_hca'] >= min_hca) & (results_v['home_hca'] < max_hca)
        if mask.sum() > 20:
            seg = results_v[mask]
            mae_f = np.abs(seg['pred_flat'] - seg['actual']).mean()
            mae_t = np.abs(seg['pred_team'] - seg['actual']).mean()
            mae_v = np.abs(seg['vegas'] - seg['actual']).mean()
            diff = mae_t - mae_f
            print(f"{label:<22} {mae_f:<8.3f} {mae_t:<8.3f} {mae_v:<8.3f} {diff:<+8.4f} {mask.sum():<6}")

    # Look at biggest HCA mismatches
    print("\n" + "=" * 70)
    print("EXTREME HCA GAMES (where flat vs team HCA differs most)")
    print("=" * 70)

    results_v['hca_diff'] = np.abs(results_v['home_hca'] - 2.0)
    results_v['error_flat'] = np.abs(results_v['pred_flat'] - results_v['actual'])
    results_v['error_team'] = np.abs(results_v['pred_team'] - results_v['actual'])
    results_v['error_improvement'] = results_v['error_flat'] - results_v['error_team']

    extreme = results_v[results_v['hca_diff'] > 2.0].copy()
    if len(extreme) > 0:
        mae_f = extreme['error_flat'].mean()
        mae_t = extreme['error_team'].mean()
        print(f"\nGames with HCA > 2.0 from flat (N={len(extreme)}):")
        print(f"  Flat MAE: {mae_f:.3f}")
        print(f"  Team MAE: {mae_t:.3f}")
        print(f"  Improvement: {mae_f - mae_t:+.3f}")


if __name__ == '__main__':
    main()
