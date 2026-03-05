"""
NFL Ridge V2 - Directional Performance Analysis

Deep analysis of home vs away pick performance to determine:
1. Is the away pick underperformance real or low-N noise?
2. Can we fade away picks profitably?
3. What should star tiers look like?

Methodology:
- Per-season walk-forward: train on seasons < N, test on season N
- 3 test seasons (2023, 2024, 2025) tested independently
- Statistical tests: binomial p-values, Wilson confidence intervals, Fisher exact test
- Comparison with NBA road favorite fade pattern

ANALYSIS ONLY - does not modify any production files or models.
"""
from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Import the model class (read-only usage)
from nfl_ridge_v2 import NFLRidgeV2

DB_PATH = Path(__file__).parent / 'nfl_games.db'


def load_all_games(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all completed games with stats and drive data."""
    conn = sqlite3.connect(str(db_path))

    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.game_date_eastern,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site,
               ht.display_name as home_team, at.display_name as away_team,
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
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
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
    return games, drives


def merge_drive_data(games: pd.DataFrame, drives: pd.DataFrame) -> pd.DataFrame:
    """Merge drive efficiency data into games dataframe."""
    if drives.empty:
        return games

    drive_stats = drives.groupby(['game_id', 'team_id']).agg(
        total_yards=('yards', 'sum'),
        num_drives=('yards', 'count'),
        scores=('is_score', 'sum'),
    ).reset_index()
    drive_stats['ypd'] = drive_stats['total_yards'] / drive_stats['num_drives'].replace(0, 1)
    drive_stats['scoring_pct'] = drive_stats['scores'] / drive_stats['num_drives'].replace(0, 1)

    # Home drives
    home_drives = drive_stats.rename(columns={'ypd': 'home_ypd', 'scoring_pct': 'home_scoring_pct'})
    games = games.merge(
        home_drives[['game_id', 'team_id', 'home_ypd', 'home_scoring_pct']],
        left_on=['game_id', 'home_team_id'],
        right_on=['game_id', 'team_id'],
        how='left'
    ).drop(columns=['team_id'], errors='ignore')

    # Away drives
    away_drives = drive_stats.rename(columns={'ypd': 'away_ypd', 'scoring_pct': 'away_scoring_pct'})
    games = games.merge(
        away_drives[['game_id', 'team_id', 'away_ypd', 'away_scoring_pct']],
        left_on=['game_id', 'away_team_id'],
        right_on=['game_id', 'team_id'],
        how='left'
    ).drop(columns=['team_id'], errors='ignore')

    return games


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


def binomial_p_value(wins: int, n: int, null_p: float = 0.5) -> float:
    """One-sided binomial test: P(X >= wins) under null."""
    if n == 0:
        return 1.0
    return stats.binom.sf(wins - 1, n, null_p)


def run_walk_forward_analysis():
    """
    Per-season walk-forward with full directional breakdown.

    For each test season N:
    - Create fresh model
    - Feed ALL games from seasons < N (features + updates)
    - For season N: extract features BEFORE updating state (true online)
    - Record predictions vs actuals with metadata
    """
    log.info("=" * 80)
    log.info("NFL RIDGE V2 - DIRECTIONAL PERFORMANCE ANALYSIS")
    log.info("Per-season walk-forward | Home vs Away | Fade strategy test")
    log.info("=" * 80)

    games_df, drives_df = load_all_games()
    games_df = merge_drive_data(games_df, drives_df)

    # Calculate third-down percentage
    games_df['home_3d_pct'] = games_df.apply(
        lambda r: 100 * r['home_3dc'] / r['home_3da']
        if pd.notna(r['home_3dc']) and pd.notna(r['home_3da']) and r['home_3da'] > 0 else None,
        axis=1
    )
    games_df['away_3d_pct'] = games_df.apply(
        lambda r: 100 * r['away_3dc'] / r['away_3da']
        if pd.notna(r['away_3dc']) and pd.notna(r['away_3da']) and r['away_3da'] > 0 else None,
        axis=1
    )

    seasons = sorted(games_df['season'].unique())
    log.info(f"Available seasons: {seasons}")
    log.info(f"Total completed games: {len(games_df)}")

    # We need at least 2 seasons for train. Test on last 3 seasons.
    # With seasons [2021, 2022, 2023, 2024, 2025]:
    #   Test 2023: train on 2021-2022
    #   Test 2024: train on 2021-2023
    #   Test 2025: train on 2021-2024
    test_seasons = [s for s in seasons if s >= 2023]
    log.info(f"Test seasons: {test_seasons}")

    # Collect all test predictions
    all_results = []

    for test_season in test_seasons:
        train_seasons = [s for s in seasons if s < test_season]
        log.info(f"\n{'─' * 60}")
        log.info(f"Testing season {test_season} | Training on {train_seasons}")
        log.info(f"{'─' * 60}")

        # Fresh model for each test season
        model = NFLRidgeV2()

        # ── Phase 1: Process all training seasons (features + updates) ──
        from nfl_ridge_v2 import MIN_GAMES, SRS_RECALC_INTERVAL
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X_spread_train, y_spread_train = [], []

        for season in train_seasons:
            if season > train_seasons[0]:
                model.set_previous_season(season)
                prev_games = games_df[games_df['season'] == season - 1]
                if len(prev_games) > 0:
                    model.league_avg['ppg'] = prev_games['home_score'].mean()
                    model.league_avg['papg'] = prev_games['away_score'].mean()

            season_games = games_df[games_df['season'] == season]
            games_processed = 0

            for _, g in season_games.iterrows():
                hid, aid = g['home_team_id'], g['away_team_id']
                actual_spread = g['away_score'] - g['home_score']
                neutral = g['neutral_site'] == 1
                week = g['week']

                if games_processed > 0 and games_processed % SRS_RECALC_INTERVAL == 0:
                    model.team_srs[season] = model._calculate_srs(season)

                hg = model.team_games[hid][season]['game_count']
                ag = model.team_games[aid][season]['game_count']

                if hg >= MIN_GAMES and ag >= MIN_GAMES:
                    feat = model.extract_spread_features(
                        hid, aid, season, g['date'], week, neutral_site=neutral
                    )
                    if feat is not None and not np.any(np.isnan(feat)):
                        X_spread_train.append(feat)
                        y_spread_train.append(actual_spread)

                # Update team state
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

        # ── Phase 2: Train Ridge model on training data ──
        X_train = np.array(X_spread_train)
        y_train = np.array(y_spread_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)

        model.spread_model = ridge
        model.spread_scaler = scaler

        log.info(f"  Training samples: {len(X_train)}")

        # ── Phase 3: Test on test season (online, game-by-game) ──
        model.set_previous_season(test_season)
        prev_games = games_df[games_df['season'] == test_season - 1]
        if len(prev_games) > 0:
            model.league_avg['ppg'] = prev_games['home_score'].mean()
            model.league_avg['papg'] = prev_games['away_score'].mean()

        test_games = games_df[games_df['season'] == test_season]
        season_predictions = 0
        games_processed = 0

        for _, g in test_games.iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_spread = g['away_score'] - g['home_score']
            vegas_spread = g['vegas_spread'] if pd.notna(g['vegas_spread']) else np.nan
            neutral = g['neutral_site'] == 1
            week = g['week']

            if games_processed > 0 and games_processed % SRS_RECALC_INTERVAL == 0:
                model.team_srs[test_season] = model._calculate_srs(test_season)

            hg = model.team_games[hid][test_season]['game_count']
            ag = model.team_games[aid][test_season]['game_count']

            if hg >= MIN_GAMES and ag >= MIN_GAMES and not np.isnan(vegas_spread):
                feat = model.extract_spread_features(
                    hid, aid, test_season, g['date'], week, neutral_site=neutral
                )
                if feat is not None and not np.any(np.isnan(feat)):
                    feat_scaled = scaler.transform(feat.reshape(1, -1))
                    pred_spread = float(ridge.predict(feat_scaled)[0])

                    edge = pred_spread - vegas_spread
                    actual_vs_vegas = actual_spread - vegas_spread

                    # Determine pick direction
                    # edge < 0 => model predicts LOWER spread => bet HOME
                    # edge > 0 => model predicts HIGHER spread => bet AWAY
                    pick_direction = 'home' if edge < 0 else 'away'

                    # Is home the favorite?
                    home_favored = vegas_spread < 0

                    # Did the pick cover?
                    push = abs(actual_vs_vegas) < 0.5
                    if not push:
                        covered = (edge > 0 and actual_vs_vegas > 0) or \
                                  (edge < 0 and actual_vs_vegas < 0)
                    else:
                        covered = None  # Push

                    # Sub-categories
                    is_road_fav_pick = pick_direction == 'away' and not home_favored
                    is_home_dog_pick = pick_direction == 'home' and not home_favored
                    is_home_fav_pick = pick_direction == 'home' and home_favored
                    is_away_dog_pick = pick_direction == 'away' and home_favored

                    all_results.append({
                        'season': test_season,
                        'week': week,
                        'home_team': g['home_team'],
                        'away_team': g['away_team'],
                        'vegas_spread': vegas_spread,
                        'pred_spread': pred_spread,
                        'actual_spread': actual_spread,
                        'edge': edge,
                        'abs_edge': abs(edge),
                        'pick_direction': pick_direction,
                        'home_favored': home_favored,
                        'neutral': neutral,
                        'covered': covered,
                        'push': push,
                        'is_road_fav_pick': is_road_fav_pick,
                        'is_home_dog_pick': is_home_dog_pick,
                        'is_home_fav_pick': is_home_fav_pick,
                        'is_away_dog_pick': is_away_dog_pick,
                        'actual_vs_vegas': actual_vs_vegas,
                    })
                    season_predictions += 1

            # Always update team state
            model.update_team(
                hid, aid, test_season, g['date'],
                g['home_score'], g['away_score'], is_home=not neutral,
                yards=g['home_yards'], pass_yards=g['home_pass_yards'],
                rush_yards=g['home_rush_yards'], turnovers=g['home_to'],
                third_down_pct=g['home_3d_pct'], first_downs=g['home_fd'],
                scoring_pct=g.get('home_scoring_pct'), ypd=g.get('home_ypd')
            )
            model.update_team(
                aid, hid, test_season, g['date'],
                g['away_score'], g['home_score'], is_home=False,
                yards=g['away_yards'], pass_yards=g['away_pass_yards'],
                rush_yards=g['away_rush_yards'], turnovers=g['away_to'],
                third_down_pct=g['away_3d_pct'], first_downs=g['away_fd'],
                scoring_pct=g.get('away_scoring_pct'), ypd=g.get('away_ypd')
            )
            games_processed += 1

        model.team_srs[test_season] = model._calculate_srs(test_season)
        log.info(f"  Test predictions: {season_predictions}")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(all_results)
    df_valid = df[df['covered'].notna()].copy()  # Exclude pushes

    log.info(f"\n{'═' * 80}")
    log.info(f"ANALYSIS RESULTS")
    log.info(f"Total predictions: {len(df)}, Pushes: {df['push'].sum()}, "
             f"Valid: {len(df_valid)}")
    log.info(f"Test seasons: {sorted(df['season'].unique())}")
    log.info(f"{'═' * 80}")

    # ── 1. Overall ATS by threshold ──────────────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("1. OVERALL ATS BY THRESHOLD")
    log.info(f"{'─' * 80}")
    log.info(f"{'Threshold':<15} {'Record':<12} {'Win%':<8} {'ROI':<10} {'p-value':<10} {'95% CI':<20} {'n'}")
    log.info(f"{'─' * 85}")

    for thresh in [0, 3, 5, 7, 10]:
        subset = df_valid[df_valid['abs_edge'] >= thresh]
        wins = int(subset['covered'].sum())
        n = len(subset)
        if n > 0:
            pct = wins / n * 100
            roi = (wins * 0.91 - (n - wins)) / n * 100
            p = binomial_p_value(wins, n)
            lo, hi = wilson_ci(wins, n)
            log.info(f"{thresh}+ pt edge{'':<6} {wins}-{n - wins:<8} {pct:.1f}%{'':<3} "
                     f"{roi:+.1f}%{'':<5} {p:.4f}{'':<5} [{lo:.1%}, {hi:.1%}]{'':<5} {n}")

    # ── 2. DIRECTIONAL SPLIT: Home vs Away ───────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("2. DIRECTIONAL SPLIT: HOME PICKS vs AWAY PICKS")
    log.info(f"{'─' * 80}")

    for direction in ['home', 'away']:
        log.info(f"\n  === {direction.upper()} PICKS ===")
        dir_subset = df_valid[df_valid['pick_direction'] == direction]
        log.info(f"  Total {direction} picks: {len(dir_subset)}")
        log.info(f"  {'Threshold':<15} {'Record':<12} {'Win%':<8} {'ROI':<10} {'p-value':<10} {'95% CI':<20} {'n'}")
        log.info(f"  {'─' * 83}")

        for thresh in [0, 3, 5, 7, 10]:
            subset = dir_subset[dir_subset['abs_edge'] >= thresh]
            wins = int(subset['covered'].sum())
            n = len(subset)
            if n > 0:
                pct = wins / n * 100
                roi = (wins * 0.91 - (n - wins)) / n * 100
                p = binomial_p_value(wins, n)
                lo, hi = wilson_ci(wins, n)
                log.info(f"  {thresh}+ pt edge{'':<6} {wins}-{n - wins:<8} {pct:.1f}%{'':<3} "
                         f"{roi:+.1f}%{'':<5} {p:.4f}{'':<5} [{lo:.1%}, {hi:.1%}]{'':<5} {n}")

    # ── 3. PER-SEASON CONSISTENCY ────────────────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("3. PER-SEASON CONSISTENCY (Home vs Away)")
    log.info(f"{'─' * 80}")

    for season in sorted(df_valid['season'].unique()):
        s_df = df_valid[df_valid['season'] == season]
        log.info(f"\n  Season {season} ({len(s_df)} games):")
        log.info(f"  {'Category':<25} {'Record':<12} {'Win%':<8} {'n'}")
        log.info(f"  {'─' * 55}")

        for direction in ['home', 'away']:
            for thresh in [0, 3, 5]:
                subset = s_df[(s_df['pick_direction'] == direction) & (s_df['abs_edge'] >= thresh)]
                wins = int(subset['covered'].sum())
                n = len(subset)
                if n > 0:
                    pct = wins / n * 100
                    log.info(f"  {direction.capitalize()} {thresh}+ edge{'':<10} "
                             f"{wins}-{n - wins:<8} {pct:.1f}%{'':<3} {n}")

    # ── 4. DETAILED SUB-CATEGORIES ───────────────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("4. DETAILED SUB-CATEGORIES")
    log.info(f"{'─' * 80}")

    categories = [
        ('Home Fav picks (model=home, vegas=home fav)', 'is_home_fav_pick'),
        ('Home Dog picks (model=home, vegas=away fav)', 'is_home_dog_pick'),
        ('Away Dog picks (model=away, vegas=home fav)', 'is_away_dog_pick'),
        ('Road Fav picks (model=away, vegas=away fav)', 'is_road_fav_pick'),
    ]

    log.info(f"  {'Category':<50} {'Record':<12} {'Win%':<8} {'ROI':<10} {'p-value':<10} {'n'}")
    log.info(f"  {'─' * 90}")

    for label, col in categories:
        for thresh in [0, 3, 5]:
            subset = df_valid[(df_valid[col]) & (df_valid['abs_edge'] >= thresh)]
            wins = int(subset['covered'].sum())
            n = len(subset)
            if n > 0:
                pct = wins / n * 100
                roi = (wins * 0.91 - (n - wins)) / n * 100
                p = binomial_p_value(wins, n)
                log.info(f"  {label} {thresh}+{'':<3} {wins}-{n - wins:<8} {pct:.1f}%{'':<3} "
                         f"{roi:+.1f}%{'':<5} {p:.4f}{'':<5} {n}")
        log.info(f"  {'─' * 90}")

    # Per season for sub-categories
    log.info(f"\n  Per-season breakdown for key categories:")
    for label, col in categories:
        log.info(f"\n  {label}:")
        for season in sorted(df_valid['season'].unique()):
            for thresh in [0, 3]:
                subset = df_valid[(df_valid[col]) & (df_valid['abs_edge'] >= thresh) &
                                  (df_valid['season'] == season)]
                wins = int(subset['covered'].sum())
                n = len(subset)
                if n > 0:
                    pct = wins / n * 100
                    log.info(f"    {season} {thresh}+ edge: {wins}-{n - wins} ({pct:.1f}%) n={n}")

    # ── 5. FADE STRATEGY ANALYSIS ────────────────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("5. FADE STRATEGY: FLIP AWAY PICKS")
    log.info("If away picks are bad, fading = betting HOME instead")
    log.info(f"{'─' * 80}")

    log.info(f"\n  Logic: When model says 'bet AWAY' with edge >= X, instead bet HOME")
    log.info(f"  A fade wins when the ORIGINAL pick LOSES (i.e., model was wrong)\n")

    log.info(f"  {'Strategy':<45} {'Record':<12} {'Win%':<8} {'ROI':<10} {'p-value':<10} {'n'}")
    log.info(f"  {'─' * 85}")

    for thresh in [0, 3, 5, 7]:
        # Away picks at this threshold
        away_subset = df_valid[(df_valid['pick_direction'] == 'away') &
                               (df_valid['abs_edge'] >= thresh)]
        n = len(away_subset)
        original_wins = int(away_subset['covered'].sum())
        fade_wins = n - original_wins  # Fading wins when original loses

        if n > 0:
            orig_pct = original_wins / n * 100
            fade_pct = fade_wins / n * 100
            fade_roi = (fade_wins * 0.91 - (n - fade_wins)) / n * 100
            fade_p = binomial_p_value(fade_wins, n)
            log.info(f"  Fade away {thresh}+ edge{'':<22} "
                     f"{fade_wins}-{n - fade_wins:<8} {fade_pct:.1f}%{'':<3} "
                     f"{fade_roi:+.1f}%{'':<5} {fade_p:.4f}{'':<5} {n}")
            log.info(f"    (original: {original_wins}-{n - original_wins} = {orig_pct:.1f}%)")

    # Fade by sub-category
    log.info(f"\n  Fade by sub-category:")
    for label, col in [('Road Fav picks', 'is_road_fav_pick'),
                       ('Away Dog picks', 'is_away_dog_pick')]:
        for thresh in [0, 3, 5]:
            subset = df_valid[(df_valid[col]) & (df_valid['abs_edge'] >= thresh)]
            n = len(subset)
            original_wins = int(subset['covered'].sum())
            fade_wins = n - original_wins

            if n > 0:
                fade_pct = fade_wins / n * 100
                fade_roi = (fade_wins * 0.91 - (n - fade_wins)) / n * 100
                fade_p = binomial_p_value(fade_wins, n)
                log.info(f"  Fade {label} {thresh}+: "
                         f"{fade_wins}-{n - fade_wins} ({fade_pct:.1f}%) "
                         f"ROI={fade_roi:+.1f}% p={fade_p:.4f} n={n}")

    # Per season fade results
    log.info(f"\n  Per-season fade results (away 3+ edge):")
    for season in sorted(df_valid['season'].unique()):
        subset = df_valid[(df_valid['pick_direction'] == 'away') &
                          (df_valid['abs_edge'] >= 3) &
                          (df_valid['season'] == season)]
        n = len(subset)
        original_wins = int(subset['covered'].sum())
        fade_wins = n - original_wins
        if n > 0:
            log.info(f"    {season}: Fade record {fade_wins}-{n - fade_wins} "
                     f"({fade_wins / n * 100:.1f}%), Original: {original_wins}-{n - original_wins} "
                     f"({original_wins / n * 100:.1f}%), n={n}")

    # ── 6. FISHER EXACT TEST: Home vs Away ───────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("6. STATISTICAL TESTS: Is Home vs Away Split Real?")
    log.info(f"{'─' * 80}")

    for thresh in [0, 3, 5]:
        home = df_valid[(df_valid['pick_direction'] == 'home') & (df_valid['abs_edge'] >= thresh)]
        away = df_valid[(df_valid['pick_direction'] == 'away') & (df_valid['abs_edge'] >= thresh)]

        h_wins = int(home['covered'].sum())
        h_n = len(home)
        a_wins = int(away['covered'].sum())
        a_n = len(away)

        if h_n > 0 and a_n > 0:
            h_pct = h_wins / h_n * 100
            a_pct = a_wins / a_n * 100

            # Fisher exact test: is the difference significant?
            contingency = [[h_wins, h_n - h_wins], [a_wins, a_n - a_wins]]
            _, fisher_p = stats.fisher_exact(contingency, alternative='greater')

            # Effect size (difference in proportions)
            diff = h_pct - a_pct

            log.info(f"\n  At {thresh}+ pt edge:")
            log.info(f"    Home: {h_wins}-{h_n - h_wins} ({h_pct:.1f}%) n={h_n}")
            log.info(f"    Away: {a_wins}-{a_n - a_wins} ({a_pct:.1f}%) n={a_n}")
            log.info(f"    Difference: {diff:+.1f} percentage points")
            log.info(f"    Fisher exact p-value (one-sided, home > away): {fisher_p:.4f}")
            if fisher_p < 0.05:
                log.info(f"    >>> SIGNIFICANT at 5% level")
            elif fisher_p < 0.10:
                log.info(f"    >>> Marginally significant at 10% level")
            else:
                log.info(f"    >>> NOT significant (could be noise)")

    # ── 7. COMBINED STRATEGY: HOME ONLY + FADE AWAY ─────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("7. COMBINED STRATEGY OPTIONS")
    log.info(f"{'─' * 80}")

    log.info(f"\n  Option A: Only take HOME picks (skip away)")
    for thresh in [0, 3, 5, 7]:
        home = df_valid[(df_valid['pick_direction'] == 'home') & (df_valid['abs_edge'] >= thresh)]
        wins = int(home['covered'].sum())
        n = len(home)
        if n > 0:
            pct = wins / n * 100
            roi = (wins * 0.91 - (n - wins)) / n * 100
            p = binomial_p_value(wins, n)
            lo, hi = wilson_ci(wins, n)
            log.info(f"    Home {thresh}+: {wins}-{n - wins} ({pct:.1f}%) ROI={roi:+.1f}% "
                     f"p={p:.4f} CI=[{lo:.1%},{hi:.1%}] n={n}")

    log.info(f"\n  Option B: HOME picks + FADE away (bet home when model says away)")
    for thresh in [0, 3, 5]:
        home = df_valid[(df_valid['pick_direction'] == 'home') & (df_valid['abs_edge'] >= thresh)]
        away = df_valid[(df_valid['pick_direction'] == 'away') & (df_valid['abs_edge'] >= thresh)]

        h_wins = int(home['covered'].sum())
        a_orig_wins = int(away['covered'].sum())
        a_fade_wins = len(away) - a_orig_wins

        total_wins = h_wins + a_fade_wins
        total_n = len(home) + len(away)

        if total_n > 0:
            pct = total_wins / total_n * 100
            roi = (total_wins * 0.91 - (total_n - total_wins)) / total_n * 100
            p = binomial_p_value(total_wins, total_n)
            log.info(f"    Combined {thresh}+: {total_wins}-{total_n - total_wins} ({pct:.1f}%) "
                     f"ROI={roi:+.1f}% p={p:.4f} n={total_n}")

    log.info(f"\n  Option C: HOME picks only at 5+, FADE away at 3+")
    home_5 = df_valid[(df_valid['pick_direction'] == 'home') & (df_valid['abs_edge'] >= 5)]
    away_3 = df_valid[(df_valid['pick_direction'] == 'away') & (df_valid['abs_edge'] >= 3)]
    h_wins = int(home_5['covered'].sum())
    a_fade_wins = len(away_3) - int(away_3['covered'].sum())
    total_wins = h_wins + a_fade_wins
    total_n = len(home_5) + len(away_3)
    if total_n > 0:
        pct = total_wins / total_n * 100
        roi = (total_wins * 0.91 - (total_n - total_wins)) / total_n * 100
        p = binomial_p_value(total_wins, total_n)
        log.info(f"    Home 5+ + Fade away 3+: {total_wins}-{total_n - total_wins} ({pct:.1f}%) "
                 f"ROI={roi:+.1f}% p={p:.4f} n={total_n}")

    # ── 8. COMPARISON WITH NBA PATTERN ───────────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("8. COMPARISON WITH NBA ROAD FAVORITE FADE PATTERN")
    log.info(f"{'─' * 80}")
    log.info("""
  NBA Road Favorite Fade (from CLAUDE.md):
    - Road fav picks: 35% ATS -> fade to 65% ATS
    - 1615 games sample, rule-based confidence system
    - Applied as +1.5 pts penalty to road fav predictions

  NFL Pattern (this analysis):""")

    # Road fav picks in NFL
    rf = df_valid[df_valid['is_road_fav_pick']]
    rf_wins = int(rf['covered'].sum())
    rf_n = len(rf)
    if rf_n > 0:
        log.info(f"    Road fav picks: {rf_wins}-{rf_n - rf_wins} "
                 f"({rf_wins / rf_n * 100:.1f}%) n={rf_n}")
        log.info(f"    Fade road fav:  {rf_n - rf_wins}-{rf_wins} "
                 f"({(rf_n - rf_wins) / rf_n * 100:.1f}%)")
    else:
        log.info(f"    No road fav picks found")

    # All away picks
    aw = df_valid[df_valid['pick_direction'] == 'away']
    aw_wins = int(aw['covered'].sum())
    aw_n = len(aw)
    if aw_n > 0:
        log.info(f"    All away picks: {aw_wins}-{aw_n - aw_wins} "
                 f"({aw_wins / aw_n * 100:.1f}%) n={aw_n}")

    log.info(f"""
  Key differences:
    - NBA has 1615 games (large sample) vs NFL has ~500 (smaller)
    - NBA road fav is a specific sub-category; NFL away picks are broader
    - NBA penalty is structural (travel, crowd) vs NFL may be model bias
    - NFL has fewer away picks (model prefers home) vs NBA more balanced
    """)

    # ── 9. EDGE DISTRIBUTION ANALYSIS ────────────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("9. EDGE DISTRIBUTION: What edges does the model generate?")
    log.info(f"{'─' * 80}")

    home_edges = df_valid[df_valid['pick_direction'] == 'home']['abs_edge']
    away_edges = df_valid[df_valid['pick_direction'] == 'away']['abs_edge']

    log.info(f"\n  Home picks: n={len(home_edges)}, mean edge={home_edges.mean():.1f}, "
             f"median={home_edges.median():.1f}")
    log.info(f"  Away picks: n={len(away_edges)}, mean edge={away_edges.mean():.1f}, "
             f"median={away_edges.median():.1f}")

    log.info(f"\n  Edge distribution (home vs away):")
    log.info(f"  {'Range':<15} {'Home':<10} {'Away':<10} {'Home%':<10} {'Away%'}")
    for lo_edge, hi_edge in [(0, 3), (3, 5), (5, 7), (7, 10), (10, 99)]:
        h_count = ((home_edges >= lo_edge) & (home_edges < hi_edge)).sum()
        a_count = ((away_edges >= lo_edge) & (away_edges < hi_edge)).sum()
        h_pct = h_count / len(home_edges) * 100 if len(home_edges) > 0 else 0
        a_pct = a_count / len(away_edges) * 100 if len(away_edges) > 0 else 0
        label = f"{lo_edge}-{hi_edge}" if hi_edge < 99 else f"{lo_edge}+"
        log.info(f"  {label:<15} {h_count:<10} {a_count:<10} {h_pct:.1f}%{'':<5} {a_pct:.1f}%")

    # ── 10. VEGAS SPREAD SIZE INTERACTION ─────────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("10. VEGAS SPREAD SIZE INTERACTION")
    log.info("Does the away pick underperformance depend on the Vegas spread size?")
    log.info(f"{'─' * 80}")

    for direction in ['home', 'away']:
        dir_df = df_valid[(df_valid['pick_direction'] == direction) & (df_valid['abs_edge'] >= 3)]
        log.info(f"\n  {direction.upper()} picks (3+ edge), by Vegas spread size:")
        log.info(f"  {'Vegas Spread':<20} {'Record':<12} {'Win%':<8} {'n'}")
        log.info(f"  {'─' * 50}")

        abs_vegas = dir_df['vegas_spread'].abs()
        for lo_v, hi_v, label in [(0, 3, 'Close (0-3)'), (3, 7, 'Medium (3-7)'),
                                   (7, 10, 'Big (7-10)'), (10, 99, 'Blowout (10+)')]:
            subset = dir_df[(abs_vegas >= lo_v) & (abs_vegas < hi_v)]
            wins = int(subset['covered'].sum())
            n = len(subset)
            if n > 0:
                pct = wins / n * 100
                log.info(f"  {label:<20} {wins}-{n - wins:<8} {pct:.1f}%{'':<3} {n}")

    # ── 11. SAMPLE GAME EXAMPLES ──────────────────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("11. EXAMPLE GAMES: Away picks that lost (to understand WHY)")
    log.info(f"{'─' * 80}")

    away_losses = df_valid[(df_valid['pick_direction'] == 'away') &
                            (df_valid['covered'] == False) &
                            (df_valid['abs_edge'] >= 3)]
    away_losses = away_losses.sort_values('abs_edge', ascending=False).head(15)

    log.info(f"\n  {'Week':<6} {'Away @ Home':<35} {'Vegas':<8} {'Model':<8} {'Edge':<8} {'Actual':<8}")
    log.info(f"  {'─' * 80}")
    for _, g in away_losses.iterrows():
        matchup = f"{g['away_team']} @ {g['home_team']}"
        log.info(f"  {g['season']}w{g['week']:<3} {matchup:<35} "
                 f"{g['vegas_spread']:+.1f}{'':<3} {g['pred_spread']:+.1f}{'':<3} "
                 f"{g['edge']:+.1f}{'':<3} {g['actual_spread']:+.1f}")

    # ── 12. POWER ANALYSIS ───────────────────────────────────────────────
    log.info(f"\n{'─' * 80}")
    log.info("12. POWER ANALYSIS: How many away picks needed to detect real difference?")
    log.info(f"{'─' * 80}")

    # If true away pick rate is 40% (terrible), how many games to detect at 80% power?
    for true_rate in [0.40, 0.45, 0.48]:
        # For binomial test at 5% significance, 80% power
        # Using normal approximation
        z_alpha = 1.645  # one-sided 5%
        z_beta = 0.842   # 80% power
        p0 = 0.50
        p1 = true_rate
        n_needed = ((z_alpha * np.sqrt(p0 * (1 - p0)) +
                      z_beta * np.sqrt(p1 * (1 - p1))) / (p0 - p1)) ** 2
        log.info(f"  If true away rate = {true_rate:.0%}: need ~{int(n_needed)} games "
                 f"for 80% power at 5% significance")

    away_n_total = len(df_valid[df_valid['pick_direction'] == 'away'])
    log.info(f"\n  Current away pick sample: {away_n_total} games")
    log.info(f"  At 3+ edge: {len(df_valid[(df_valid['pick_direction'] == 'away') & (df_valid['abs_edge'] >= 3)])}")
    log.info(f"  At 5+ edge: {len(df_valid[(df_valid['pick_direction'] == 'away') & (df_valid['abs_edge'] >= 5)])}")

    # ── 13. SUMMARY & RECOMMENDATIONS ────────────────────────────────────
    log.info(f"\n{'═' * 80}")
    log.info("SUMMARY & RECOMMENDATIONS")
    log.info(f"{'═' * 80}")

    # Compute key numbers for summary
    home_all = df_valid[df_valid['pick_direction'] == 'home']
    away_all = df_valid[df_valid['pick_direction'] == 'away']
    h_w, h_n = int(home_all['covered'].sum()), len(home_all)
    a_w, a_n = int(away_all['covered'].sum()), len(away_all)

    home_5 = df_valid[(df_valid['pick_direction'] == 'home') & (df_valid['abs_edge'] >= 5)]
    away_5 = df_valid[(df_valid['pick_direction'] == 'away') & (df_valid['abs_edge'] >= 5)]
    h5_w, h5_n = int(home_5['covered'].sum()), len(home_5)
    a5_w, a5_n = int(away_5['covered'].sum()), len(away_5)

    away_3 = df_valid[(df_valid['pick_direction'] == 'away') & (df_valid['abs_edge'] >= 3)]
    a3_w, a3_n = int(away_3['covered'].sum()), len(away_3)

    log.info(f"""
  KEY FINDINGS:

  1. Home picks overall: {h_w}-{h_n - h_w} ({h_w / h_n * 100:.1f}%) n={h_n}
     Away picks overall: {a_w}-{a_n - a_w} ({a_w / a_n * 100:.1f}%) n={a_n}

  2. Home 5+ edge: {h5_w}-{h5_n - h5_w} ({h5_w / h5_n * 100:.1f}% ATS) n={h5_n}
     Away 5+ edge: {a5_w}-{a5_n - a5_w} ({a5_w / a5_n * 100:.1f}% ATS) n={a5_n}

  3. Away 3+ edge: {a3_w}-{a3_n - a3_w} ({a3_w / a3_n * 100:.1f}% ATS) n={a3_n}
     Fade away 3+: {a3_n - a3_w}-{a3_w} ({(a3_n - a3_w) / a3_n * 100:.1f}% ATS)
""")

    # Check significance
    _, fisher_p_5 = stats.fisher_exact(
        [[h5_w, h5_n - h5_w], [a5_w, a5_n - a5_w]], alternative='greater'
    ) if h5_n > 0 and a5_n > 0 else (None, 1.0)

    home_p_5 = binomial_p_value(h5_w, h5_n)
    away_fade_3_p = binomial_p_value(a3_n - a3_w, a3_n)

    log.info(f"""  STATISTICAL SIGNIFICANCE:
  - Home 5+ ATS > 50%: p={home_p_5:.4f} {'(significant)' if home_p_5 < 0.05 else '(marginal)' if home_p_5 < 0.10 else '(not significant)'}
  - Home > Away at 5+: Fisher p={fisher_p_5:.4f} {'(significant)' if fisher_p_5 < 0.05 else '(marginal)' if fisher_p_5 < 0.10 else '(not significant)'}
  - Fade away 3+ > 50%: p={away_fade_3_p:.4f} {'(significant)' if away_fade_3_p < 0.05 else '(marginal)' if away_fade_3_p < 0.10 else '(not significant)'}
""")

    # Check per-season consistency
    consistent_home = True
    consistent_away_bad = True
    for season in sorted(df_valid['season'].unique()):
        s_home = df_valid[(df_valid['season'] == season) & (df_valid['pick_direction'] == 'home') &
                          (df_valid['abs_edge'] >= 5)]
        s_away = df_valid[(df_valid['season'] == season) & (df_valid['pick_direction'] == 'away') &
                          (df_valid['abs_edge'] >= 3)]
        if len(s_home) > 0:
            if int(s_home['covered'].sum()) / len(s_home) < 0.52:
                consistent_home = False
        if len(s_away) > 0:
            if int(s_away['covered'].sum()) / len(s_away) > 0.52:
                consistent_away_bad = False

    log.info(f"""  PER-SEASON CONSISTENCY:
  - Home 5+ consistently profitable across all test seasons: {'YES' if consistent_home else 'NO'}
  - Away 3+ consistently bad across all test seasons: {'YES' if consistent_away_bad else 'NO'}

  RECOMMENDATIONS:

  A) CONSERVATIVE (if data is marginal/not significant):
     - Keep current approach: 3-star at 5+ edge for ALL picks
     - Note directional split as "interesting but unconfirmed"
     - Wait for more data (need ~100+ away picks for statistical power)

  B) MODERATE (if home picks significant, away marginal):
     - 3-star: Home picks with 5+ edge
     - 2-star: Home picks with 3-5 edge
     - 1-star: Away picks (reduced confidence)
     - No fading (insufficient evidence)

  C) AGGRESSIVE (if both splits are significant and per-season consistent):
     - 3-star: Home picks with 5+ edge
     - 2-star: Home picks with 3-5 edge OR Fade away 3+ edge
     - Skip: Away picks at 0-3 edge
     - Apply +1.5 pt penalty to away picks (like NBA road fav penalty)

  WHICH TO CHOOSE: Look at significance levels above. If Fisher p < 0.05
  and per-season consistent, go with B or C. If p > 0.10, go with A.
""")


if __name__ == '__main__':
    run_walk_forward_analysis()
