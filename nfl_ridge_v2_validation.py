"""
NFL Ridge V2 — Proper Walk-Forward Validation

This script tests for false confidence by:
1. True per-season walk-forward: for each test season, train only on prior seasons
2. Reporting results per-season to check consistency (not just aggregate)
3. Testing if threshold patterns hold across EACH season independently
4. Bootstrap confidence intervals on ATS percentages
5. Comparing against random baselines

If "OVER 3+" is 65.9% overall but 80% in one season and 50% in another,
that's a red flag — the aggregate number is driven by one lucky season.
"""
from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nfl_ridge_v2 import NFLRidgeV2, DB_PATH

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_all_games(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load games and drives from DB."""
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
        SELECT d.game_id, d.team_id, d.yards, d.is_score
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE g.completed = 1
    ''', conn)
    conn.close()
    return games, drives


def merge_drive_data(games: pd.DataFrame, drives: pd.DataFrame) -> pd.DataFrame:
    """Merge per-team drive efficiency into games."""
    if drives.empty:
        return games
    drive_stats = drives.groupby(['game_id', 'team_id']).agg(
        total_yards=('yards', 'sum'),
        num_drives=('yards', 'count'),
        scores=('is_score', 'sum'),
    ).reset_index()
    drive_stats['ypd'] = drive_stats['total_yards'] / drive_stats['num_drives'].replace(0, 1)
    drive_stats['scoring_pct'] = drive_stats['scores'] / drive_stats['num_drives'].replace(0, 1)

    home_drives = drive_stats.rename(columns={'ypd': 'home_ypd', 'scoring_pct': 'home_scoring_pct'})
    games = games.merge(
        home_drives[['game_id', 'team_id', 'home_ypd', 'home_scoring_pct']],
        left_on=['game_id', 'home_team_id'],
        right_on=['game_id', 'team_id'],
        how='left'
    ).drop(columns=['team_id'], errors='ignore')

    away_drives = drive_stats.rename(columns={'ypd': 'away_ypd', 'scoring_pct': 'away_scoring_pct'})
    games = games.merge(
        away_drives[['game_id', 'team_id', 'away_ypd', 'away_scoring_pct']],
        left_on=['game_id', 'away_team_id'],
        right_on=['game_id', 'team_id'],
        how='left'
    ).drop(columns=['team_id'], errors='ignore')

    return games


def calc_third_down_pct(games: pd.DataFrame) -> pd.DataFrame:
    """Add third-down percentage columns."""
    games['home_3d_pct'] = games.apply(
        lambda r: 100 * r['home_3dc'] / r['home_3da']
        if pd.notna(r['home_3dc']) and pd.notna(r['home_3da']) and r['home_3da'] > 0 else None,
        axis=1
    )
    games['away_3d_pct'] = games.apply(
        lambda r: 100 * r['away_3dc'] / r['away_3da']
        if pd.notna(r['away_3dc']) and pd.notna(r['away_3da']) and r['away_3da'] > 0 else None,
        axis=1
    )
    return games


def process_season(model: NFLRidgeV2, games: pd.DataFrame, season: int):
    """Process one season of games through the model (extract features + update stats).

    Returns lists of (features, targets, metadata) for games with sufficient history.
    """
    spread_feats, total_feats = [], []
    y_spreads, y_totals = [], []
    vegas_spreads, vegas_totals = [], []
    weeks, home_favs = [], []

    season_games = games[games['season'] == season]
    games_processed = 0

    for _, g in season_games.iterrows():
        hid = g['home_team_id']
        aid = g['away_team_id']
        actual_spread = g['away_score'] - g['home_score']
        actual_total = g['home_score'] + g['away_score']
        neutral = g['neutral_site'] == 1
        week = g['week']

        # SRS recalculation
        if games_processed > 0 and games_processed % 30 == 0:
            model.team_srs[season] = model._calculate_srs(season)

        home_games = model.team_games[hid][season]['game_count']
        away_games = model.team_games[aid][season]['game_count']

        if home_games >= 2 and away_games >= 2:
            sf = model.extract_spread_features(hid, aid, season, g['date'], week, neutral_site=neutral)
            tf = model.extract_total_features(hid, aid, season, g['date'], week)

            if sf is not None and tf is not None:
                spread_feats.append(sf)
                total_feats.append(tf)
                y_spreads.append(actual_spread)
                y_totals.append(actual_total)
                weeks.append(week)
                vegas_spreads.append(g['vegas_spread'] if pd.notna(g['vegas_spread']) else np.nan)
                vegas_totals.append(g['vegas_total'] if pd.notna(g['vegas_total']) else np.nan)
                home_favs.append(1 if pd.notna(g['vegas_spread']) and g['vegas_spread'] < 0 else 0)

        # Update team states with actual results
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

    # Final SRS for season
    model.team_srs[season] = model._calculate_srs(season)

    return {
        'X_spread': np.array(spread_feats) if spread_feats else np.empty((0, 18)),
        'X_total': np.array(total_feats) if total_feats else np.empty((0, 14)),
        'y_spread': np.array(y_spreads),
        'y_total': np.array(y_totals),
        'vegas_spread': np.array(vegas_spreads),
        'vegas_total': np.array(vegas_totals),
        'weeks': np.array(weeks),
        'home_fav': np.array(home_favs),
    }


def ats_record(model_vals, vegas_vals, actual_vals, mask=None):
    """Calculate ATS record. Returns (wins, losses, pushes)."""
    if mask is None:
        mask = np.ones(len(model_vals), dtype=bool)
    edge = model_vals[mask] - vegas_vals[mask]
    result = actual_vals[mask] - vegas_vals[mask]
    push = np.abs(result) < 0.5
    wins = ((edge > 0) & (result > 0)) | ((edge < 0) & (result < 0))
    wins = wins & ~push
    losses = ~wins & ~push
    return int(wins.sum()), int(losses.sum()), int(push.sum())


def ou_record(model_totals, vegas_totals, actual_totals, over_mask):
    """Calculate O/U record for a given mask. Returns (wins, losses, pushes)."""
    actual_over = actual_totals > vegas_totals
    actual_under = actual_totals < vegas_totals
    push = np.abs(actual_totals - vegas_totals) < 0.5
    # over_mask = True means we're betting OVER
    wins = (over_mask & actual_over) | (~over_mask & actual_under)
    wins = wins & ~push
    losses = ~wins & ~push
    return int(wins.sum()), int(losses.sum()), int(push.sum())


def bootstrap_ci(wins: int, total: int, n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for a win rate."""
    if total == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    outcomes = np.zeros(total)
    outcomes[:wins] = 1.0
    boot_means = np.array([rng.choice(outcomes, size=total, replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def main():
    log.info("=" * 70)
    log.info("NFL RIDGE V2 — PROPER WALK-FORWARD VALIDATION")
    log.info("=" * 70)
    log.info("Testing for false confidence via:")
    log.info("  1. Per-season walk-forward (train < N, test = N)")
    log.info("  2. Per-season consistency check")
    log.info("  3. Bootstrap confidence intervals")
    log.info("  4. Random baseline comparison")
    log.info("")

    games, drives = load_all_games()
    games = merge_drive_data(games, drives)
    games = calc_third_down_pct(games)

    seasons = sorted(games['season'].unique())
    log.info(f"Available seasons: {seasons}")
    log.info(f"Total completed games: {len(games)}")

    # We need at least 2 seasons of training data, so test starts from season[2]
    min_train_seasons = 2
    test_seasons = seasons[min_train_seasons:]
    log.info(f"Test seasons (each trained on all prior): {test_seasons}")

    # ── Per-Season Walk-Forward ────────────────────────────────────────
    all_results = []

    for test_season in test_seasons:
        train_seasons = [s for s in seasons if s < test_season]
        log.info(f"\n{'='*50}")
        log.info(f"TEST SEASON: {test_season} | Training on: {train_seasons}")
        log.info(f"{'='*50}")

        # Fresh model for each fold
        model = NFLRidgeV2()

        # Process training seasons (build stats + extract features)
        train_data = {'X_spread': [], 'X_total': [], 'y_spread': [], 'y_total': []}
        for s in train_seasons:
            if s > train_seasons[0]:
                model.set_previous_season(s)
                prev_games = games[games['season'] == s - 1]
                if len(prev_games) > 0:
                    model.league_avg['ppg'] = prev_games['home_score'].mean()
                    model.league_avg['papg'] = prev_games['away_score'].mean()
            result = process_season(model, games, s)
            if len(result['X_spread']) > 0:
                train_data['X_spread'].append(result['X_spread'])
                train_data['X_total'].append(result['X_total'])
                train_data['y_spread'].append(result['y_spread'])
                train_data['y_total'].append(result['y_total'])

        # Prepare test season
        model.set_previous_season(test_season)
        prev_games = games[games['season'] == test_season - 1]
        if len(prev_games) > 0:
            model.league_avg['ppg'] = prev_games['home_score'].mean()
            model.league_avg['papg'] = prev_games['away_score'].mean()

        test_result = process_season(model, games, test_season)

        if len(test_result['X_spread']) == 0:
            log.info(f"  No test data for season {test_season}, skipping")
            continue

        # Combine training data
        X_train_s = np.vstack(train_data['X_spread'])
        y_train_s = np.concatenate(train_data['y_spread'])
        X_train_t = np.vstack(train_data['X_total'])
        y_train_t = np.concatenate(train_data['y_total'])

        # Clean NaN rows
        nan_mask = np.isnan(X_train_s).any(axis=1) | np.isnan(y_train_s)
        X_train_s = X_train_s[~nan_mask]
        y_train_s = y_train_s[~nan_mask]
        X_train_t = X_train_t[~nan_mask[:len(X_train_t)]]
        y_train_t = y_train_t[~nan_mask[:len(y_train_t)]]

        X_test_s = test_result['X_spread']
        y_test_s = test_result['y_spread']
        X_test_t = test_result['X_total']
        y_test_t = test_result['y_total']

        test_nan = np.isnan(X_test_s).any(axis=1) | np.isnan(y_test_s)
        X_test_s = X_test_s[~test_nan]
        y_test_s = y_test_s[~test_nan]
        X_test_t = X_test_t[~test_nan[:len(X_test_t)]]
        y_test_t = y_test_t[~test_nan[:len(y_test_t)]]

        vs = test_result['vegas_spread'][~test_nan]
        vt = test_result['vegas_total'][~test_nan]
        wk = test_result['weeks'][~test_nan]

        log.info(f"  Train: {len(X_train_s)} games | Test: {len(X_test_s)} games")

        # Train fresh model on training data only
        spread_scaler = StandardScaler()
        X_train_s_sc = spread_scaler.fit_transform(X_train_s)
        X_test_s_sc = spread_scaler.transform(X_test_s)

        spread_model = Ridge(alpha=1.0)
        spread_model.fit(X_train_s_sc, y_train_s)
        pred_spread = spread_model.predict(X_test_s_sc)

        total_scaler = StandardScaler()
        X_train_t_sc = total_scaler.fit_transform(X_train_t)
        X_test_t_sc = total_scaler.transform(X_test_t)

        total_model = Ridge(alpha=1.0)
        total_model.fit(X_train_t_sc, y_train_t)
        pred_total = total_model.predict(X_test_t_sc)

        # Filter to games with Vegas lines
        has_vs = ~np.isnan(vs)
        has_vt = ~np.isnan(vt)

        # Store per-season results
        season_result = {
            'season': test_season,
            'n_games': int(has_vs.sum()),
            'pred_spread': pred_spread[has_vs],
            'y_spread': y_test_s[has_vs],
            'vegas_spread': vs[has_vs],
            'pred_total': pred_total[has_vt],
            'y_total': y_test_t[has_vt],
            'vegas_total': vt[has_vt],
        }
        all_results.append(season_result)

        # Per-season summary
        ms, ys_v, vs_v = season_result['pred_spread'], season_result['y_spread'], season_result['vegas_spread']
        mt, yt_v, vt_v = season_result['pred_total'], season_result['y_total'], season_result['vegas_total']

        model_mae = np.abs(ms - ys_v).mean()
        vegas_mae = np.abs(vs_v - ys_v).mean()
        log.info(f"  Spread MAE: {model_mae:.2f} (Vegas: {vegas_mae:.2f})")

        # ATS at various thresholds
        for thresh in [0, 3, 5, 7]:
            edge = np.abs(ms - vs_v)
            mask = edge >= thresh
            if mask.sum() > 0:
                w, l, p = ats_record(ms, vs_v, ys_v, mask)
                total = w + l
                pct = w / total * 100 if total > 0 else 0
                log.info(f"  ATS {thresh}+ edge: {w}-{l} ({pct:.1f}%) n={total}")

        # Totals
        total_edge = mt - vt_v
        total_bias = np.mean(total_edge - (yt_v - vt_v))

        for thresh in [0, 3, 5]:
            over_mask = total_edge >= thresh
            if over_mask.sum() > 0:
                over_correct = (yt_v[over_mask] > vt_v[over_mask]).sum()
                over_total = int(over_mask.sum())
                push = int((np.abs(yt_v[over_mask] - vt_v[over_mask]) < 0.5).sum()) if over_total > 0 else 0
                valid = over_total - push
                pct = over_correct / valid * 100 if valid > 0 else 0
                log.info(f"  OVER  {thresh}+ edge: {over_correct}/{valid} ({pct:.1f}%) n={over_total}")

            under_mask = total_edge <= -thresh
            if under_mask.sum() > 0:
                under_correct = (yt_v[under_mask] < vt_v[under_mask]).sum()
                under_total = int(under_mask.sum())
                valid = under_total
                pct = under_correct / valid * 100 if valid > 0 else 0
                # Also show FADE performance
                fade_correct = (yt_v[under_mask] > vt_v[under_mask]).sum()
                fade_pct = fade_correct / valid * 100 if valid > 0 else 0
                log.info(f"  UNDER {thresh}+ edge: {under_correct}/{valid} ({pct:.1f}%) "
                         f"| FADE→OVER: {fade_correct}/{valid} ({fade_pct:.1f}%)")

        log.info(f"  Total bias: {total_bias:+.2f} pts")

    # ── Aggregate Results ──────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("AGGREGATE RESULTS (all test seasons combined)")
    log.info("=" * 70)

    # Combine all seasons
    all_ms = np.concatenate([r['pred_spread'] for r in all_results])
    all_ys = np.concatenate([r['y_spread'] for r in all_results])
    all_vs = np.concatenate([r['vegas_spread'] for r in all_results])
    all_mt = np.concatenate([r['pred_total'] for r in all_results])
    all_yt = np.concatenate([r['y_total'] for r in all_results])
    all_vt = np.concatenate([r['vegas_total'] for r in all_results])

    total_games = len(all_ms)
    log.info(f"\nTotal test games: {total_games}")

    # Spread ATS with bootstrap CI
    log.info("\n--- SPREAD ATS (with 95% Bootstrap CI) ---")
    for thresh in [0, 3, 5, 7]:
        edge = np.abs(all_ms - all_vs)
        mask = edge >= thresh
        w, l, p = ats_record(all_ms, all_vs, all_ys, mask)
        total = w + l
        if total > 0:
            pct = w / total * 100
            roi = (w * 0.91 - l) / total * 100
            lo, hi = bootstrap_ci(w, total)
            log.info(f"  {thresh}+ edge: {w}-{l} ({pct:.1f}%) ROI:{roi:+.1f}% "
                     f"95% CI: [{lo*100:.1f}%, {hi*100:.1f}%] n={total}")

    # Totals with bootstrap CI
    log.info("\n--- TOTALS O/U (with 95% Bootstrap CI) ---")
    total_edge = all_mt - all_vt
    for thresh in [0, 3, 5]:
        over_mask = total_edge >= thresh
        if over_mask.sum() > 0:
            over_correct = int((all_yt[over_mask] > all_vt[over_mask]).sum())
            over_total = int(over_mask.sum())
            push = int((np.abs(all_yt[over_mask] - all_vt[over_mask]) < 0.5).sum())
            valid = over_total - push
            if valid > 0:
                lo, hi = bootstrap_ci(over_correct, valid)
                pct = over_correct / valid * 100
                log.info(f"  OVER  {thresh}+: {over_correct}/{valid} ({pct:.1f}%) "
                         f"95% CI: [{lo*100:.1f}%, {hi*100:.1f}%] n={over_total}")

        under_mask = total_edge <= -thresh
        if under_mask.sum() > 0:
            under_correct = int((all_yt[under_mask] < all_vt[under_mask]).sum())
            under_total = int(under_mask.sum())
            # FADE performance
            fade_correct = int((all_yt[under_mask] > all_vt[under_mask]).sum())
            push = int((np.abs(all_yt[under_mask] - all_vt[under_mask]) < 0.5).sum())
            valid = under_total - push
            if valid > 0:
                lo, hi = bootstrap_ci(fade_correct, valid)
                fade_pct = fade_correct / valid * 100
                raw_pct = under_correct / valid * 100
                log.info(f"  UNDER {thresh}+: raw {under_correct}/{valid} ({raw_pct:.1f}%) "
                         f"| FADE→OVER: {fade_correct}/{valid} ({fade_pct:.1f}%) "
                         f"95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]")

    # ── Consistency Check ──────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PER-SEASON CONSISTENCY CHECK")
    log.info("(If one season drives the aggregate, the signal is unreliable)")
    log.info("=" * 70)

    for thresh_name, check_fn in [
        ("Spread ATS 5+", lambda r: _season_ats(r, 5)),
        ("OVER 3+", lambda r: _season_over(r, 3)),
        ("FADE UNDER 5+", lambda r: _season_fade_under(r, 5)),
    ]:
        log.info(f"\n--- {thresh_name} ---")
        season_records = []
        for r in all_results:
            w, l, n = check_fn(r)
            season_records.append((r['season'], w, l, n))
            if n > 0:
                pct = w / (w + l) * 100 if (w + l) > 0 else 0
                log.info(f"  Season {r['season']}: {w}-{l} ({pct:.1f}%) n={n}")
            else:
                log.info(f"  Season {r['season']}: no qualifying games")

        # Check consistency
        valid_seasons = [(s, w, l) for s, w, l, n in season_records if (w + l) >= 5]
        if len(valid_seasons) >= 2:
            pcts = [w / (w + l) * 100 for _, w, l in valid_seasons]
            log.info(f"  Seasons with 5+ games: {len(valid_seasons)}")
            log.info(f"  Win % range: {min(pcts):.1f}% - {max(pcts):.1f}%")
            log.info(f"  Std dev: {np.std(pcts):.1f}%")
            all_above_52 = all(p > 52.4 for p in pcts)
            log.info(f"  All seasons > 52.4% (breakeven)? {'YES ✓' if all_above_52 else 'NO ✗'}")
        elif len(valid_seasons) == 1:
            log.info(f"  ⚠ Only 1 season with 5+ games — cannot assess consistency")
        else:
            log.info(f"  ⚠ No seasons with 5+ games — unreliable signal")

    # ── Random Baseline ────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("RANDOM BASELINE (how often does random selection hit these rates?)")
    log.info("=" * 70)

    rng = np.random.default_rng(42)
    n_simulations = 50000

    for label, observed_wins, observed_total in [
        ("Spread 5+ ATS", *_aggregate_ats(all_results, 5)),
        ("OVER 3+", *_aggregate_over(all_results, 3)),
        ("FADE UNDER 5+", *_aggregate_fade(all_results, 5)),
    ]:
        if observed_total == 0:
            continue
        observed_pct = observed_wins / observed_total
        # How often would random coin flips (50%) produce this result?
        random_wins = rng.binomial(observed_total, 0.5, n_simulations)
        p_value = (random_wins >= observed_wins).mean()
        log.info(f"\n  {label}: {observed_wins}/{observed_total} ({observed_pct*100:.1f}%)")
        log.info(f"  P-value (vs 50% baseline): {p_value:.4f}")
        if p_value < 0.05:
            log.info(f"  → Statistically significant at 95% (p < 0.05) ✓")
        elif p_value < 0.10:
            log.info(f"  → Marginally significant (0.05 < p < 0.10) ~")
        else:
            log.info(f"  → NOT statistically significant (p >= 0.10) ✗")


# ── Helper functions for consistency checks ────────────────────────────

def _season_ats(result: dict, thresh: int) -> tuple[int, int, int]:
    """ATS record at threshold for one season."""
    ms, vs, ys = result['pred_spread'], result['vegas_spread'], result['y_spread']
    edge = np.abs(ms - vs)
    mask = edge >= thresh
    if mask.sum() == 0:
        return 0, 0, 0
    w, l, p = ats_record(ms, vs, ys, mask)
    return w, l, w + l


def _season_over(result: dict, thresh: int) -> tuple[int, int, int]:
    """OVER record at threshold for one season."""
    mt, vt, yt = result['pred_total'], result['vegas_total'], result['y_total']
    total_edge = mt - vt
    mask = total_edge >= thresh
    if mask.sum() == 0:
        return 0, 0, 0
    correct = int((yt[mask] > vt[mask]).sum())
    push = int((np.abs(yt[mask] - vt[mask]) < 0.5).sum())
    valid = int(mask.sum()) - push
    return correct, valid - correct, valid


def _season_fade_under(result: dict, thresh: int) -> tuple[int, int, int]:
    """FADE UNDER (bet OVER when model says UNDER) for one season."""
    mt, vt, yt = result['pred_total'], result['vegas_total'], result['y_total']
    total_edge = mt - vt
    mask = total_edge <= -thresh
    if mask.sum() == 0:
        return 0, 0, 0
    # Fade = bet OVER when model says UNDER
    correct = int((yt[mask] > vt[mask]).sum())
    push = int((np.abs(yt[mask] - vt[mask]) < 0.5).sum())
    valid = int(mask.sum()) - push
    return correct, valid - correct, valid


def _aggregate_ats(results: list, thresh: int) -> tuple[int, int]:
    """Combined ATS across all seasons."""
    total_w, total_n = 0, 0
    for r in results:
        w, l, n = _season_ats(r, thresh)
        total_w += w
        total_n += n
    return total_w, total_n


def _aggregate_over(results: list, thresh: int) -> tuple[int, int]:
    """Combined OVER across all seasons."""
    total_w, total_n = 0, 0
    for r in results:
        w, l, n = _season_over(r, thresh)
        total_w += w
        total_n += n
    return total_w, total_n


def _aggregate_fade(results: list, thresh: int) -> tuple[int, int]:
    """Combined FADE UNDER across all seasons."""
    total_w, total_n = 0, 0
    for r in results:
        w, l, n = _season_fade_under(r, thresh)
        total_w += w
        total_n += n
    return total_w, total_n


if __name__ == '__main__':
    main()
