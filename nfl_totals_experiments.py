"""
NFL Totals Experiments — Systematic Feature & Alpha Testing

Tests multiple total model configurations via walk-forward validation.
Only 2024 (259 games) and 2025 (282 games) have Vegas total coverage,
so we train on 2024 and test on 2025 (single split, ~259 train / ~282 test).

Experiments:
  A. Pure model (no Vegas anchor) — 16 features, baseline
  B. Vegas-anchored model — 17 features (adds Vegas total)
  C. Per-team pace + matchup features — 18/19 features
  D. Alpha tuning — test alpha [1, 5, 10, 25, 50, 100]
  E. Best combination from above
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nfl_ridge_v2 import NFLRidgeV2, DB_PATH

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_data(db_path: Path = DB_PATH):
    """Load games, drives, and build model state."""
    conn = sqlite3.connect(str(db_path))
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.game_date_eastern,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site,
               hs.total_yards as home_yards, hs.passing_yards as home_pass_yards,
               hs.rushing_yards as home_rush_yards, hs.turnovers as home_to,
               hs.first_downs as home_fd,
               hs.third_down_conversions as home_3dc, hs.third_down_attempts as home_3da,
               aws.total_yards as away_yards, aws.passing_yards as away_pass_yards,
               aws.rushing_yards as away_rush_yards, aws.turnovers as away_to,
               aws.first_downs as away_fd,
               aws.third_down_conversions as away_3dc, aws.third_down_attempts as away_3da,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id
            AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id
            AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date, g.week
    ''', conn)

    drives = pd.read_sql_query('''
        SELECT DISTINCT d.game_id, d.team_id, d.yards, d.is_score,
               d.plays, d.time_elapsed_seconds
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE g.completed = 1
    ''', conn)
    conn.close()

    # Merge drive efficiency
    if not drives.empty:
        drive_stats = drives.groupby(['game_id', 'team_id']).agg(
            total_yards=('yards', 'sum'),
            num_drives=('yards', 'count'),
            scores=('is_score', 'sum'),
        ).reset_index()
        drive_stats['ypd'] = drive_stats['total_yards'] / drive_stats['num_drives'].replace(0, 1)
        drive_stats['scoring_pct'] = drive_stats['scores'] / drive_stats['num_drives'].replace(0, 1)

        for prefix, id_col in [('home', 'home_team_id'), ('away', 'away_team_id')]:
            renamed = drive_stats.rename(columns={
                'ypd': f'{prefix}_ypd', 'scoring_pct': f'{prefix}_scoring_pct'
            })
            games = games.merge(
                renamed[['game_id', 'team_id', f'{prefix}_ypd', f'{prefix}_scoring_pct']],
                left_on=['game_id', id_col],
                right_on=['game_id', 'team_id'],
                how='left'
            ).drop(columns=['team_id'], errors='ignore')

    # Third down pct
    for prefix in ['home', 'away']:
        games[f'{prefix}_3d_pct'] = games.apply(
            lambda r: 100 * r[f'{prefix}_3dc'] / r[f'{prefix}_3da']
            if pd.notna(r[f'{prefix}_3dc']) and pd.notna(r[f'{prefix}_3da']) and r[f'{prefix}_3da'] > 0
            else None,
            axis=1
        )

    return games


def extract_total_features_config(model, home_id, away_id, season, game_date, week,
                                  vegas_total=None, config='pure'):
    """Extract total features with different configurations.

    Configs:
        'pure'     — 12 features: PPG sums, yards, turnovers, drive eff, form, bye, progress, reliability
        'pace'     — 14 features: pure + per-team pace (home_pace, away_pace)
        'matchup'  — 14 features: pure + matchup scoring (home_off_vs_away_def, away_off_vs_home_def)
        'full'     — 16 features: pure + pace + matchup
        'vegas'    — 13 features: pure + Vegas total
        'vegas_full' — 17 features: pure + pace + matchup + Vegas total
    """
    hs = model._get_team_stats(home_id, season)
    aws = model._get_team_stats(away_id, season)

    if hs['games'] < 2 or aws['games'] < 2:
        return None

    hr = model._get_rest_days(home_id, game_date)
    ar = model._get_rest_days(away_id, game_date)

    # Base features (always included) — 12 features
    features = [
        hs['ppg'] + aws['ppg'],                          # Combined PPG
        hs['papg'] + aws['papg'],                        # Combined PAPG
        hs['yards'] + aws['yards'],                      # Combined yards
        hs['turnovers'] + aws['turnovers'],              # Combined turnovers
        hs['scoring_pct'] + aws['scoring_pct'],          # Combined scoring %
        hs['ypd'] + aws['ypd'],                          # Combined YPD
        abs(model._recent_form(hs['margins'])) + abs(model._recent_form(aws['margins'])),
        1.0 if hr >= 13 else 0.0,                        # Home post-bye
        1.0 if ar >= 13 else 0.0,                        # Away post-bye
        min(week / 17.0, 1.0),                           # Season progress
        min(hs['games'] / 10.0, 1.0),                   # Home reliability
        min(aws['games'] / 10.0, 1.0),                  # Away reliability
    ]

    # Optional: per-team pace
    if config in ('pace', 'full', 'vegas_full'):
        features.append((hs['ppg'] + hs['papg']) / 2)   # Home team pace
        features.append((aws['ppg'] + aws['papg']) / 2)  # Away team pace

    # Optional: matchup scoring
    if config in ('matchup', 'full', 'vegas_full'):
        features.append((hs['ppg'] + aws['papg']) / 2)   # Home off vs away def
        features.append((aws['ppg'] + hs['papg']) / 2)   # Away off vs home def

    # Optional: Vegas total anchor
    if config in ('vegas', 'vegas_full') and vegas_total is not None:
        features.append(vegas_total)

    return np.array(features)


def run_experiment(games, config, alpha=1.0, test_season=2025):
    """Run one experiment: train on prior seasons with Vegas data, test on test_season."""

    # Fresh model instance
    model = NFLRidgeV2()

    seasons = sorted(games['season'].unique())

    # Process all seasons through the model to build state
    X_total, y_total, vegas_totals, total_seasons_list = [], [], [], []

    for season in seasons:
        if season > seasons[0]:
            model.set_previous_season(season)
            prev_games = games[games['season'] == season - 1]
            if len(prev_games) > 0:
                model.league_avg['ppg'] = prev_games['home_score'].mean()
                model.league_avg['papg'] = prev_games['away_score'].mean()

        season_games = games[games['season'] == season]
        games_processed = 0

        for _, g in season_games.iterrows():
            hid = g['home_team_id']
            aid = g['away_team_id']
            actual_total = g['home_score'] + g['away_score']
            week = g['week']

            if games_processed > 0 and games_processed % 30 == 0:
                model.team_srs[season] = model._calculate_srs(season)

            home_games = model.team_games[hid][season]['game_count']
            away_games = model.team_games[aid][season]['game_count']

            vt = g['vegas_total'] if pd.notna(g['vegas_total']) else None

            if home_games >= 2 and away_games >= 2 and vt is not None:
                feat = extract_total_features_config(
                    model, hid, aid, season, g['date'], week,
                    vegas_total=vt, config=config
                )
                if feat is not None:
                    X_total.append(feat)
                    y_total.append(actual_total)
                    vegas_totals.append(vt)
                    total_seasons_list.append(season)

            # Always update state
            neutral = g['neutral_site'] == 1
            model.update_team(
                hid, aid, season, g['date'],
                g['home_score'], g['away_score'], is_home=not neutral,
                yards=g['home_yards'], pass_yards=g['home_pass_yards'],
                rush_yards=g['home_rush_yards'], turnovers=g['home_to'],
                third_down_pct=g['home_3d_pct'], first_downs=g['home_fd'],
                scoring_pct=g.get('home_scoring_pct'), ypd=g.get('home_ypd')
            )
            model.update_team(
                aid, hid, season, g['date'],
                g['away_score'], g['home_score'], is_home=False,
                yards=g['away_yards'], pass_yards=g['away_pass_yards'],
                rush_yards=g['away_rush_yards'], turnovers=g['away_to'],
                third_down_pct=g['away_3d_pct'], first_downs=g['away_fd'],
                scoring_pct=g.get('away_scoring_pct'), ypd=g.get('away_ypd')
            )
            games_processed += 1

        model.team_srs[season] = model._calculate_srs(season)

    # Convert to arrays
    X = np.array(X_total)
    y = np.array(y_total)
    vt_arr = np.array(vegas_totals)
    seasons_arr = np.array(total_seasons_list)

    # Clean NaN
    nan_mask = np.isnan(X).any(axis=1)
    X, y, vt_arr, seasons_arr = X[~nan_mask], y[~nan_mask], vt_arr[~nan_mask], seasons_arr[~nan_mask]

    # Split: train on < test_season, test on test_season
    train_mask = seasons_arr < test_season
    test_mask = seasons_arr == test_season

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    vt_test = vt_arr[test_mask]

    if len(X_train) < 10:
        return None  # Not enough training data

    # Train
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model_ridge = Ridge(alpha=alpha)
    model_ridge.fit(X_train_sc, y_train)
    pred = model_ridge.predict(X_test_sc)

    # Evaluate
    model_mae = np.abs(pred - y_test).mean()
    vegas_mae = np.abs(vt_test - y_test).mean()
    actual_total = y_test
    total_edge = pred - vt_test

    results = {
        'config': config,
        'alpha': alpha,
        'test_season': test_season,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X.shape[1],
        'model_mae': model_mae,
        'vegas_mae': vegas_mae,
    }

    # O/U at various thresholds
    for thresh in [0, 3, 5, 7]:
        over_mask = total_edge >= thresh
        if over_mask.sum() > 0:
            over_correct = int((actual_total[over_mask] > vt_test[over_mask]).sum())
            over_push = int((np.abs(actual_total[over_mask] - vt_test[over_mask]) < 0.5).sum())
            over_valid = int(over_mask.sum()) - over_push
            results[f'over_{thresh}_wins'] = over_correct
            results[f'over_{thresh}_total'] = over_valid
            results[f'over_{thresh}_pct'] = over_correct / over_valid * 100 if over_valid > 0 else 0

        under_mask = total_edge <= -thresh
        if under_mask.sum() > 0:
            under_correct = int((actual_total[under_mask] < vt_test[under_mask]).sum())
            under_push = int((np.abs(actual_total[under_mask] - vt_test[under_mask]) < 0.5).sum())
            under_valid = int(under_mask.sum()) - under_push
            results[f'under_{thresh}_wins'] = under_correct
            results[f'under_{thresh}_total'] = under_valid
            results[f'under_{thresh}_pct'] = under_correct / under_valid * 100 if under_valid > 0 else 0

    # Overall O/U
    over_picks = total_edge > 0
    under_picks = total_edge < 0
    over_correct = int((actual_total[over_picks] > vt_test[over_picks]).sum()) if over_picks.sum() > 0 else 0
    under_correct = int((actual_total[under_picks] < vt_test[under_picks]).sum()) if under_picks.sum() > 0 else 0
    total_correct = over_correct + under_correct
    total_valid = int(over_picks.sum() + under_picks.sum())
    results['ou_overall_wins'] = total_correct
    results['ou_overall_total'] = total_valid
    results['ou_overall_pct'] = total_correct / total_valid * 100 if total_valid > 0 else 0

    # Bias
    results['bias'] = float(np.mean(pred - y_test))

    # Coefficients (top 3)
    coef_abs = np.abs(model_ridge.coef_)
    top_idx = np.argsort(coef_abs)[-3:][::-1]
    results['top_coefs'] = [(int(i), float(model_ridge.coef_[i])) for i in top_idx]

    return results


def print_result(r):
    """Print one experiment result."""
    if r is None:
        log.info("  SKIPPED (insufficient training data)")
        return

    log.info(f"  Config: {r['config']:<12} Alpha: {r['alpha']:<5} Features: {r['n_features']}")
    log.info(f"  Train: {r['n_train']} games, Test: {r['n_test']} games (season {r['test_season']})")
    log.info(f"  MAE: {r['model_mae']:.2f} (Vegas: {r['vegas_mae']:.2f}, gap: {r['model_mae'] - r['vegas_mae']:+.2f})")
    log.info(f"  Overall O/U: {r['ou_overall_wins']}/{r['ou_overall_total']} ({r['ou_overall_pct']:.1f}%)")
    log.info(f"  Bias: {r['bias']:+.2f} pts")

    for thresh in [0, 3, 5]:
        over_key = f'over_{thresh}_pct'
        under_key = f'under_{thresh}_pct'
        if over_key in r:
            over_n = r.get(f'over_{thresh}_total', 0)
            over_w = r.get(f'over_{thresh}_wins', 0)
            log.info(f"    OVER  {thresh}+: {over_w}/{over_n} ({r[over_key]:.1f}%)")
        if under_key in r:
            under_n = r.get(f'under_{thresh}_total', 0)
            under_w = r.get(f'under_{thresh}_wins', 0)
            log.info(f"    UNDER {thresh}+: {under_w}/{under_n} ({r[under_key]:.1f}%)")


def main():
    log.info("=" * 70)
    log.info("NFL TOTALS EXPERIMENTS")
    log.info("=" * 70)
    log.info("Testing feature configurations and alpha values")
    log.info("Walk-forward: train on 2024, test on 2025")
    log.info("")

    games = load_data()

    # ── Experiment A: Pure model (no Vegas) ────────────────────────────
    log.info("\n" + "=" * 50)
    log.info("EXPERIMENT A: Pure model (12 features, no Vegas)")
    log.info("=" * 50)
    r_pure = run_experiment(games, config='pure', alpha=1.0)
    print_result(r_pure)

    # ── Experiment B: Vegas-anchored ────────────────────────────────────
    log.info("\n" + "=" * 50)
    log.info("EXPERIMENT B: Vegas-anchored (13 features)")
    log.info("=" * 50)
    r_vegas = run_experiment(games, config='vegas', alpha=1.0)
    print_result(r_vegas)

    # ── Experiment C: Per-team pace ────────────────────────────────────
    log.info("\n" + "=" * 50)
    log.info("EXPERIMENT C: Per-team pace (14 features)")
    log.info("=" * 50)
    r_pace = run_experiment(games, config='pace', alpha=1.0)
    print_result(r_pace)

    # ── Experiment D: Matchup scoring ──────────────────────────────────
    log.info("\n" + "=" * 50)
    log.info("EXPERIMENT D: Matchup scoring (14 features)")
    log.info("=" * 50)
    r_matchup = run_experiment(games, config='matchup', alpha=1.0)
    print_result(r_matchup)

    # ── Experiment E: Full (pace + matchup) ────────────────────────────
    log.info("\n" + "=" * 50)
    log.info("EXPERIMENT E: Full features (16 features)")
    log.info("=" * 50)
    r_full = run_experiment(games, config='full', alpha=1.0)
    print_result(r_full)

    # ── Experiment F: Vegas + full features ────────────────────────────
    log.info("\n" + "=" * 50)
    log.info("EXPERIMENT F: Vegas + pace + matchup (17 features)")
    log.info("=" * 50)
    r_vfull = run_experiment(games, config='vegas_full', alpha=1.0)
    print_result(r_vfull)

    # ── Experiment G: Alpha tuning on best configs ─────────────────────
    log.info("\n" + "=" * 50)
    log.info("EXPERIMENT G: Alpha tuning")
    log.info("=" * 50)

    best_configs = ['pure', 'vegas', 'vegas_full']
    for config in best_configs:
        log.info(f"\n  --- {config} ---")
        for alpha in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
            r = run_experiment(games, config=config, alpha=alpha)
            if r:
                ou_pct = r['ou_overall_pct']
                mae = r['model_mae']
                bias = r['bias']
                over_0 = r.get('over_0_pct', 0)
                log.info(f"    alpha={alpha:<5.0f} O/U={ou_pct:.1f}% MAE={mae:.2f} bias={bias:+.2f} OVER_0+={over_0:.1f}%")

    # ── Summary ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("SUMMARY — All experiments at alpha=1.0")
    log.info("=" * 70)
    log.info(f"\n{'Config':<15} {'Features':<10} {'MAE':<8} {'Bias':<8} {'O/U %':<8} {'OVER 0+':<10} {'OVER 3+':<10}")
    log.info("-" * 75)

    for name, r in [('Pure', r_pure), ('Vegas', r_vegas), ('Pace', r_pace),
                    ('Matchup', r_matchup), ('Full', r_full), ('Vegas+Full', r_vfull)]:
        if r:
            over_0 = f"{r.get('over_0_wins', 0)}/{r.get('over_0_total', 0)}" if 'over_0_wins' in r else "N/A"
            over_3 = f"{r.get('over_3_wins', 0)}/{r.get('over_3_total', 0)}" if 'over_3_wins' in r else "N/A"
            log.info(f"{name:<15} {r['n_features']:<10} {r['model_mae']:<8.2f} {r['bias']:<+8.2f} "
                     f"{r['ou_overall_pct']:<8.1f} {over_0:<10} {over_3:<10}")

    # Bootstrap CI on best result
    all_results = [('Pure', r_pure), ('Vegas', r_vegas), ('Pace', r_pace),
                   ('Matchup', r_matchup), ('Full', r_full), ('Vegas+Full', r_vfull)]
    best_name, best_r = max(
        [(n, r) for n, r in all_results if r is not None],
        key=lambda x: x[1]['ou_overall_pct']
    )

    log.info(f"\nBest config: {best_name} ({best_r['ou_overall_pct']:.1f}% O/U)")

    # Bootstrap CI
    rng = np.random.default_rng(42)
    wins, total = best_r['ou_overall_wins'], best_r['ou_overall_total']
    outcomes = np.zeros(total)
    outcomes[:wins] = 1.0
    boot_means = np.array([rng.choice(outcomes, size=total, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    log.info(f"95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]")

    # P-value
    random_wins = rng.binomial(total, 0.5, 50000)
    p_value = (random_wins >= wins).mean()
    log.info(f"P-value vs 50%: {p_value:.4f}")

    if p_value < 0.05:
        log.info("→ SIGNIFICANT at 95% level ✓")
    elif p_value < 0.10:
        log.info("→ Marginally significant ~")
    else:
        log.info("→ NOT statistically significant ✗")


if __name__ == '__main__':
    main()
