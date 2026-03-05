"""
CFB Score Prediction Experiments

Tests 3 alternative prediction architectures against the baseline differential approach:

Baseline: Differential features → predict spread/total directly (current Ridge V2)
Approach A: Asymmetric features (individual team stats, not diffs) → predict spread/total
Approach B: Predict individual scores → derive spread/total (doubled training data)
Approach C: Ensemble of Baseline + Approach B

All approaches share the same CFBRidgeV2 team state (SRS, weighted stats, HCA)
and use identical walk-forward evaluation (train 2022-2023, test 2024-2025).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cfb_ridge_v2 import (
    BASE_HCA,
    DB_PATH,
    FORM_WEIGHT,
    MIN_GAMES,
    SRS_RECALC_INTERVAL,
    CFBRidgeV2,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (same as CFBRidgeV2.train)
# ---------------------------------------------------------------------------

def load_data(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load games with box scores and drive data."""
    conn = sqlite3.connect(str(db_path))

    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.game_date_eastern,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.neutral_site, g.conference_game,
               hs.total_yards as home_yards, hs.passing_yards as home_pass_yards,
               hs.rushing_yards as home_rush_yards, hs.turnovers as home_to,
               hs.first_downs as home_fd,
               aws.total_yards as away_yards, aws.passing_yards as away_pass_yards,
               aws.rushing_yards as away_rush_yards, aws.turnovers as away_to,
               aws.first_downs as away_fd,
               o.latest_spread as vegas_spread, o.latest_total as vegas_total
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id
            AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id
            AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score IS NOT NULL
          AND g.postseason_type IS NULL
        ORDER BY g.date, g.week
    ''', conn)

    drives = pd.read_sql_query('''
        SELECT DISTINCT d.game_id, d.team_id, d.yards, d.is_score
        FROM drives d
        JOIN games g ON d.game_id = g.game_id
        WHERE g.completed = 1
    ''', conn)
    conn.close()

    if not drives.empty:
        drive_stats = drives.groupby(['game_id', 'team_id']).agg(
            total_yards=('yards', 'sum'),
            num_drives=('yards', 'count'),
            scores=('is_score', 'sum'),
        ).reset_index()
        drive_stats['ypd'] = drive_stats['total_yards'] / drive_stats['num_drives'].replace(0, 1)
        drive_stats['scoring_pct'] = drive_stats['scores'] / drive_stats['num_drives'].replace(0, 1)

        for prefix, team_col in [('home', 'home_team_id'), ('away', 'away_team_id')]:
            renamed = drive_stats.rename(columns={
                'ypd': f'{prefix}_ypd', 'scoring_pct': f'{prefix}_scoring_pct'
            })
            games = games.merge(
                renamed[['game_id', 'team_id', f'{prefix}_ypd', f'{prefix}_scoring_pct']],
                left_on=['game_id', team_col],
                right_on=['game_id', 'team_id'],
                how='left'
            ).drop(columns=['team_id'], errors='ignore')

    return games


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def extract_asymmetric_spread(model: CFBRidgeV2, hid: int, aid: int,
                              season: int, date: str, week: int,
                              neutral: bool = False,
                              is_conf: bool = False) -> np.ndarray | None:
    """Approach A: Individual team stats as features (~34), predict spread."""
    hs = model._get_team_stats(hid, season)
    aws = model._get_team_stats(aid, season)

    if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
        return None

    hr = model._get_rest_days(hid, date)
    ar = model._get_rest_days(aid, date)
    home_srs = model._get_team_srs(hid, season)
    away_srs = model._get_team_srs(aid, season)
    hca = 0.0 if neutral else model._get_dynamic_hca(hid, season)

    return np.array([
        # Individual team stats (not diffs)
        hs['ppg'], aws['ppg'],
        hs['papg'], aws['papg'],
        home_srs, away_srs,
        hs['yards'], aws['yards'],
        hs['pass_yards'], aws['pass_yards'],
        hs['rush_yards'], aws['rush_yards'],
        hs['turnovers'], aws['turnovers'],
        hs['first_downs'], aws['first_downs'],
        hs['scoring_pct'], aws['scoring_pct'],
        hs['ypd'], aws['ypd'],
        # Form (individual)
        model._recent_form(hs['margins']), model._recent_form(aws['margins']),
        model._momentum(hs['margins']), model._momentum(aws['margins']),
        model._streak(hs['wins']), model._streak(aws['wins']),
        # Rest (individual)
        min(hr, 14), min(ar, 14),
        1.0 if hr >= 14 else 0.0, 1.0 if ar >= 14 else 0.0,
        # Context
        hca,
        1.0 if is_conf else 0.0,
        min(week / 14.0, 1.0),
        min(hs['games'] / 10.0, 1.0),
        min(aws['games'] / 10.0, 1.0),
    ])


def extract_asymmetric_total(model: CFBRidgeV2, hid: int, aid: int,
                             season: int, date: str, week: int) -> np.ndarray | None:
    """Approach A: Individual team stats for totals (~21)."""
    hs = model._get_team_stats(hid, season)
    aws = model._get_team_stats(aid, season)

    if hs['games'] < MIN_GAMES or aws['games'] < MIN_GAMES:
        return None

    hr = model._get_rest_days(hid, date)
    ar = model._get_rest_days(aid, date)

    return np.array([
        hs['ppg'], aws['ppg'],
        hs['papg'], aws['papg'],
        hs['yards'], aws['yards'],
        hs['turnovers'], aws['turnovers'],
        hs['scoring_pct'], aws['scoring_pct'],
        hs['ypd'], aws['ypd'],
        hs['first_downs'], aws['first_downs'],
        abs(model._recent_form(hs['margins'])),
        abs(model._recent_form(aws['margins'])),
        1.0 if hr >= 14 else 0.0,
        1.0 if ar >= 14 else 0.0,
        min(week / 14.0, 1.0),
        min(hs['games'] / 10.0, 1.0),
        min(aws['games'] / 10.0, 1.0),
    ])


def extract_score_features(model: CFBRidgeV2, team_id: int, opp_id: int,
                           season: int, date: str, week: int,
                           is_home: bool, neutral: bool = False) -> np.ndarray | None:
    """Approach B: Features for predicting one team's score (~20)."""
    ts = model._get_team_stats(team_id, season)
    ops = model._get_team_stats(opp_id, season)

    if ts['games'] < MIN_GAMES or ops['games'] < MIN_GAMES:
        return None

    team_srs = model._get_team_srs(team_id, season)
    opp_srs = model._get_team_srs(opp_id, season)
    rest = model._get_rest_days(team_id, date)

    # HCA: positive for home team, 0 for away/neutral
    if neutral:
        hca_val = 0.0
    elif is_home:
        hca_val = model._get_dynamic_hca(team_id, season)
    else:
        hca_val = 0.0

    return np.array([
        # Team offensive profile
        ts['ppg'],
        ts['yards'], ts['pass_yards'], ts['rush_yards'],
        ts['turnovers'],
        ts['first_downs'],
        ts['scoring_pct'], ts['ypd'],
        # Opponent defensive profile
        ops['papg'],
        ops['turnovers'],        # Opponent's turnovers = their ball security (inverse of takeaway)
        ops['scoring_pct'],      # How efficient opponents are (defensive weakness)
        ops['ypd'],              # How many yards opponents gain per drive
        # Strength ratings
        team_srs, opp_srs,
        # Form
        model._recent_form(ts['margins']),
        model._momentum(ts['margins']),
        model._streak(ts['wins']),
        # Context
        min(rest, 14),
        hca_val,
        min(week / 14.0, 1.0),
    ])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def calc_ats(pred, vegas, actual, mask=None):
    """Calculate ATS record. Returns (wins, total_valid)."""
    if mask is None:
        mask = np.ones(len(pred), dtype=bool)
    edge = pred[mask] - vegas[mask]
    result = actual[mask] - vegas[mask]
    push = np.abs(result) < 0.5
    wins = ((edge > 0) & (result > 0)) | ((edge < 0) & (result < 0))
    wins = wins & ~push
    valid = ~push
    return int(wins.sum()), int(valid.sum())


def calc_ou(pred_total, vegas_total, actual_total, mask=None):
    """Calculate O/U record for model total predictions."""
    if mask is None:
        mask = np.ones(len(pred_total), dtype=bool)
    mt = pred_total[mask]
    vt = vegas_total[mask]
    yt = actual_total[mask]

    # OVER: model > vegas, actual > vegas
    over_pick = mt > vt
    under_pick = mt < vt

    over_correct = int(((yt[over_pick] > vt[over_pick]).sum())) if over_pick.sum() > 0 else 0
    under_correct = int(((yt[under_pick] < vt[under_pick]).sum())) if under_pick.sum() > 0 else 0

    total_correct = over_correct + under_correct
    total_games = int(over_pick.sum() + under_pick.sum())
    return total_correct, total_games


def evaluate_approach(label: str, pred_spread, pred_total,
                      actual_spread, actual_total,
                      vegas_spread, vegas_total,
                      seasons) -> dict:
    """Full evaluation of one approach. Returns summary dict."""
    has_vs = ~np.isnan(vegas_spread)
    has_vt = ~np.isnan(vegas_total)

    results = {'label': label}

    # Spread MAE
    results['spread_mae'] = float(np.abs(pred_spread - actual_spread).mean())
    if has_vs.sum() > 0:
        results['vegas_spread_mae'] = float(
            np.abs(vegas_spread[has_vs] - actual_spread[has_vs]).mean()
        )

    # Total MAE
    results['total_mae'] = float(np.abs(pred_total - actual_total).mean())
    if has_vt.sum() > 0:
        results['vegas_total_mae'] = float(
            np.abs(vegas_total[has_vt] - actual_total[has_vt]).mean()
        )

    # ATS at various thresholds
    ms = pred_spread[has_vs]
    vs = vegas_spread[has_vs]
    ys = actual_spread[has_vs]
    ss = seasons[has_vs]

    for thresh in [0, 3, 5, 7]:
        edge = ms - vs
        mask = np.abs(edge) >= thresh
        w, t = calc_ats(ms, vs, ys, mask)
        key = f'ats_{thresh}'
        results[key] = (w, t, w / t * 100 if t > 0 else 0)

    # Per-season ATS at 3+ and 5+
    for season in sorted(set(ss)):
        for thresh in [3, 5]:
            s_mask = (ss == season) & (np.abs(ms - vs) >= thresh)
            w, t = calc_ats(ms, vs, ys, s_mask)
            results[f'ats_{thresh}_s{season}'] = (w, t, w / t * 100 if t > 0 else 0)

    # O/U at various thresholds
    mt = pred_total[has_vt]
    vt = vegas_total[has_vt]
    yt = actual_total[has_vt]
    ts = seasons[has_vt]

    for thresh in [0, 3, 5]:
        edge = mt - vt
        # OVER picks
        over_mask = edge >= thresh
        if over_mask.sum() > 0:
            oc = int((yt[over_mask] > vt[over_mask]).sum())
            on = int(over_mask.sum())
            results[f'over_{thresh}'] = (oc, on, oc / on * 100 if on > 0 else 0)
        # UNDER picks
        under_mask = edge <= -thresh
        if under_mask.sum() > 0:
            uc = int((yt[under_mask] < vt[under_mask]).sum())
            un = int(under_mask.sum())
            results[f'under_{thresh}'] = (uc, un, uc / un * 100 if un > 0 else 0)

    # Per-season O/U at 0+
    for season in sorted(set(ts)):
        s_mask_t = ts == season
        edge_s = mt[s_mask_t] - vt[s_mask_t]
        over_s = edge_s >= 0
        under_s = edge_s < 0
        if over_s.sum() > 0:
            oc = int((yt[s_mask_t][over_s] > vt[s_mask_t][over_s]).sum())
            results[f'over_0_s{season}'] = (oc, int(over_s.sum()), oc / over_s.sum() * 100)
        if under_s.sum() > 0:
            uc = int((yt[s_mask_t][under_s] < vt[s_mask_t][under_s]).sum())
            results[f'under_0_s{season}'] = (uc, int(under_s.sum()), uc / under_s.sum() * 100)

    return results


def print_comparison(all_results: list[dict]):
    """Print side-by-side comparison of all approaches."""
    log.info("\n" + "=" * 90)
    log.info("COMPARISON TABLE")
    log.info("=" * 90)

    # Header
    labels = [r['label'] for r in all_results]
    header = f"{'Metric':<35}" + "".join(f"{l:<14}" for l in labels)
    log.info(header)
    log.info("-" * (35 + 14 * len(labels)))

    # MAE
    for key, name in [('spread_mae', 'Spread MAE'),
                      ('vegas_spread_mae', 'Vegas Spread MAE'),
                      ('total_mae', 'Total MAE'),
                      ('vegas_total_mae', 'Vegas Total MAE')]:
        vals = []
        for r in all_results:
            v = r.get(key)
            vals.append(f"{v:.2f}" if v is not None else "N/A")
        log.info(f"{name:<35}" + "".join(f"{v:<14}" for v in vals))

    log.info("")

    # ATS
    for thresh in [0, 3, 5, 7]:
        key = f'ats_{thresh}'
        vals = []
        for r in all_results:
            w, t, pct = r.get(key, (0, 0, 0))
            vals.append(f"{pct:.1f}% ({w}/{t})" if t > 0 else "N/A")
        log.info(f"ATS {thresh}+ edge{'':<20}" + "".join(f"{v:<14}" for v in vals))

    log.info("")

    # Per-season ATS
    seasons = set()
    for r in all_results:
        for k in r:
            if k.startswith('ats_3_s'):
                # Key format: ats_3_s2024 — extract season after last 's'
                seasons.add(int(k[len('ats_3_s'):]))
        break

    for season in sorted(seasons):
        for thresh in [3, 5]:
            key = f'ats_{thresh}_s{season}'
            vals = []
            for r in all_results:
                w, t, pct = r.get(key, (0, 0, 0))
                vals.append(f"{pct:.1f}% ({w}/{t})" if t > 0 else "N/A")
            log.info(f"  {season} ATS {thresh}+{'':<18}" + "".join(f"{v:<14}" for v in vals))

    log.info("")

    # O/U
    for thresh in [0, 3, 5]:
        for direction in ['over', 'under']:
            key = f'{direction}_{thresh}'
            vals = []
            for r in all_results:
                data = r.get(key)
                if data:
                    w, t, pct = data
                    vals.append(f"{pct:.1f}% ({w}/{t})")
                else:
                    vals.append("N/A")
            label_str = f"{'OVER' if direction == 'over' else 'UNDER'} {thresh}+"
            log.info(f"{label_str:<35}" + "".join(f"{v:<14}" for v in vals))

    log.info("")

    # Per-season O/U
    for season in sorted(seasons):
        for direction in ['over', 'under']:
            key = f'{direction}_0_s{season}'
            vals = []
            for r in all_results:
                data = r.get(key)
                if data:
                    w, t, pct = data
                    vals.append(f"{pct:.1f}% ({w}/{t})")
                else:
                    vals.append("N/A")
            label_str = f"  {season} {'OVER' if direction == 'over' else 'UNDER'} 0+"
            log.info(f"{label_str:<35}" + "".join(f"{v:<14}" for v in vals))

    # Highlight best
    log.info("\n" + "=" * 90)
    log.info("BEST APPROACH PER METRIC")
    log.info("=" * 90)

    for thresh in [0, 3, 5]:
        key = f'ats_{thresh}'
        best_pct = 0
        best_label = ""
        for r in all_results:
            w, t, pct = r.get(key, (0, 0, 0))
            if t > 0 and pct > best_pct:
                best_pct = pct
                best_label = r['label']
        if best_label:
            w, t, _ = [r for r in all_results if r['label'] == best_label][0].get(key, (0, 0, 0))
            p_val = binomtest(w, t, 0.5, alternative='greater').pvalue if w > t / 2 else 1.0
            log.info(f"ATS {thresh}+: {best_label} at {best_pct:.1f}% (p={p_val:.4f})")

    for key_name, label_name in [('over_5', 'OVER 5+'), ('under_5', 'UNDER 5+')]:
        best_pct = 0
        best_label = ""
        for r in all_results:
            data = r.get(key_name)
            if data and data[1] > 0 and data[2] > best_pct:
                best_pct = data[2]
                best_label = r['label']
        if best_label:
            data = [r for r in all_results if r['label'] == best_label][0].get(key_name)
            if data:
                p_val = binomtest(data[0], data[1], 0.5, alternative='greater').pvalue if data[0] > data[1] / 2 else 1.0
                log.info(f"{label_name}: {best_label} at {best_pct:.1f}% (p={p_val:.4f})")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiments():
    """Run all approaches and compare."""
    log.info("=" * 90)
    log.info("CFB SCORE PREDICTION EXPERIMENTS")
    log.info("Baseline vs Asymmetric vs Score-Based vs Ensemble")
    log.info("=" * 90)

    games = load_data()
    model = CFBRidgeV2()  # Shared team state

    log.info(f"Total games: {len(games)}")
    log.info(f"Vegas spread coverage: {games['vegas_spread'].notna().sum()}")
    log.info(f"Vegas total coverage: {games['vegas_total'].notna().sum()}")

    # Collection arrays for each approach
    # Baseline (differential)
    bl_spread_X, bl_spread_y = [], []
    bl_total_X, bl_total_y = [], []

    # Approach A (asymmetric)
    aa_spread_X, aa_spread_y = [], []
    aa_total_X, aa_total_y = [], []

    # Approach B (score prediction) — doubled data
    sb_X, sb_y = [], []  # Each game adds 2 rows

    # Metadata (aligned with baseline arrays)
    meta_seasons, meta_vegas_s, meta_vegas_t = [], [], []
    meta_actual_spread, meta_actual_total = [], []

    # Score-based metadata (aligned with sb arrays, 2 per game)
    sb_game_idx = []  # Maps each sb row back to game index

    game_idx = 0  # Counter for games with valid features

    seasons = sorted(games['season'].unique())
    log.info(f"Seasons: {seasons}")

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
            actual_spread = g['away_score'] - g['home_score']
            actual_total = g['home_score'] + g['away_score']
            neutral_flag = g['neutral_site'] == 1
            conf_flag = g['conference_game'] == 1
            week = g['week']

            # SRS recalc
            if games_processed > 0 and games_processed % SRS_RECALC_INTERVAL == 0:
                model.team_srs[season] = model._calculate_srs(season)

            home_n = model.team_games[hid][season]['game_count']
            away_n = model.team_games[aid][season]['game_count']

            if home_n >= MIN_GAMES and away_n >= MIN_GAMES:
                # Baseline features
                bl_sf = model.extract_spread_features(
                    hid, aid, season, g['date'], week,
                    neutral_site=neutral_flag, is_conf_game=conf_flag
                )
                bl_tf = model.extract_total_features(
                    hid, aid, season, g['date'], week
                )

                # Approach A features
                aa_sf = extract_asymmetric_spread(
                    model, hid, aid, season, g['date'], week,
                    neutral=neutral_flag, is_conf=conf_flag
                )
                aa_tf = extract_asymmetric_total(
                    model, hid, aid, season, g['date'], week
                )

                # Approach B features (two rows per game)
                sb_home = extract_score_features(
                    model, hid, aid, season, g['date'], week,
                    is_home=not neutral_flag, neutral=neutral_flag
                )
                sb_away = extract_score_features(
                    model, aid, hid, season, g['date'], week,
                    is_home=False, neutral=neutral_flag
                )

                # Only collect if ALL approaches produced valid features
                if all(f is not None for f in [bl_sf, bl_tf, aa_sf, aa_tf, sb_home, sb_away]):
                    bl_spread_X.append(bl_sf)
                    bl_spread_y.append(actual_spread)
                    bl_total_X.append(bl_tf)
                    bl_total_y.append(actual_total)

                    aa_spread_X.append(aa_sf)
                    aa_spread_y.append(actual_spread)
                    aa_total_X.append(aa_tf)
                    aa_total_y.append(actual_total)

                    sb_X.append(sb_home)
                    sb_y.append(g['home_score'])
                    sb_X.append(sb_away)
                    sb_y.append(g['away_score'])
                    sb_game_idx.append(game_idx)
                    sb_game_idx.append(game_idx)

                    meta_seasons.append(season)
                    meta_vegas_s.append(
                        g['vegas_spread'] if pd.notna(g['vegas_spread']) else np.nan
                    )
                    meta_vegas_t.append(
                        g['vegas_total'] if pd.notna(g['vegas_total']) else np.nan
                    )
                    meta_actual_spread.append(actual_spread)
                    meta_actual_total.append(actual_total)
                    game_idx += 1

            # Update team state (always, regardless of feature extraction)
            model.update_team(
                hid, aid, season, g['date'],
                g['home_score'], g['away_score'], is_home=not neutral_flag,
                yards=g['home_yards'], pass_yards=g['home_pass_yards'],
                rush_yards=g['home_rush_yards'], turnovers=g['home_to'],
                first_downs=g['home_fd'],
                scoring_pct=g.get('home_scoring_pct'), ypd=g.get('home_ypd')
            )
            model.update_team(
                aid, hid, season, g['date'],
                g['away_score'], g['home_score'], is_home=False,
                yards=g['away_yards'], pass_yards=g['away_pass_yards'],
                rush_yards=g['away_rush_yards'], turnovers=g['away_to'],
                first_downs=g['away_fd'],
                scoring_pct=g.get('away_scoring_pct'), ypd=g.get('away_ypd')
            )
            games_processed += 1

        model.team_srs[season] = model._calculate_srs(season)

    # Convert to arrays
    bl_spread_X = np.array(bl_spread_X)
    bl_spread_y = np.array(bl_spread_y)
    bl_total_X = np.array(bl_total_X)
    bl_total_y = np.array(bl_total_y)

    aa_spread_X = np.array(aa_spread_X)
    aa_spread_y = np.array(aa_spread_y)
    aa_total_X = np.array(aa_total_X)
    aa_total_y = np.array(aa_total_y)

    sb_X = np.array(sb_X)
    sb_y = np.array(sb_y)
    sb_game_idx = np.array(sb_game_idx)

    meta_seasons = np.array(meta_seasons)
    meta_vegas_s = np.array(meta_vegas_s)
    meta_vegas_t = np.array(meta_vegas_t)
    meta_actual_spread = np.array(meta_actual_spread)
    meta_actual_total = np.array(meta_actual_total)

    # Drop NaN rows
    for name, X_arr, y_arr in [
        ('baseline spread', bl_spread_X, bl_spread_y),
        ('baseline total', bl_total_X, bl_total_y),
        ('asymmetric spread', aa_spread_X, aa_spread_y),
        ('asymmetric total', aa_total_X, aa_total_y),
        ('score', sb_X, sb_y),
    ]:
        nan_mask = np.isnan(X_arr).any(axis=1) | np.isnan(y_arr)
        if nan_mask.sum() > 0:
            log.info(f"WARNING: {nan_mask.sum()} NaN rows in {name}")

    log.info(f"\nTotal aligned games: {game_idx}")
    log.info(f"Baseline features: {bl_spread_X.shape[1]} spread, {bl_total_X.shape[1]} total")
    log.info(f"Asymmetric features: {aa_spread_X.shape[1]} spread, {aa_total_X.shape[1]} total")
    log.info(f"Score features: {sb_X.shape[1]} (x2 rows = {len(sb_X)})")

    # -- Walk-forward split: train on first 2 seasons, test on last 2 --------
    test_seasons = sorted(set(meta_seasons))[-2:]
    train_mask = ~np.isin(meta_seasons, test_seasons)
    test_mask = np.isin(meta_seasons, test_seasons)

    # Score-based split (doubled rows)
    sb_train_mask = np.isin(sb_game_idx, np.where(train_mask)[0])
    sb_test_mask = np.isin(sb_game_idx, np.where(test_mask)[0])

    log.info(f"Train: {train_mask.sum()} games, Test: {test_mask.sum()} games (seasons {test_seasons})")
    log.info(f"Score train: {sb_train_mask.sum()} rows, Score test: {sb_test_mask.sum()} rows")

    all_results = []

    # =====================================================================
    # BASELINE: Differential features → spread/total
    # =====================================================================
    log.info("\n" + "=" * 70)
    log.info("TRAINING: Baseline (Differential)")
    log.info("=" * 70)

    bl_s_scaler = StandardScaler()
    bl_s_train = bl_s_scaler.fit_transform(bl_spread_X[train_mask])
    bl_s_test = bl_s_scaler.transform(bl_spread_X[test_mask])
    bl_s_model = Ridge(alpha=1.0).fit(bl_s_train, bl_spread_y[train_mask])
    bl_spread_pred = bl_s_model.predict(bl_s_test)

    bl_t_scaler = StandardScaler()
    bl_t_train = bl_t_scaler.fit_transform(bl_total_X[train_mask])
    bl_t_test = bl_t_scaler.transform(bl_total_X[test_mask])
    bl_t_model = Ridge(alpha=1.0).fit(bl_t_train, bl_total_y[train_mask])
    bl_total_pred = bl_t_model.predict(bl_t_test)

    all_results.append(evaluate_approach(
        "Baseline", bl_spread_pred, bl_total_pred,
        meta_actual_spread[test_mask], meta_actual_total[test_mask],
        meta_vegas_s[test_mask], meta_vegas_t[test_mask],
        meta_seasons[test_mask]
    ))

    # =====================================================================
    # APPROACH A: Asymmetric features → spread/total
    # =====================================================================
    log.info("\n" + "=" * 70)
    log.info("TRAINING: Approach A (Asymmetric Features)")
    log.info("=" * 70)

    aa_s_scaler = StandardScaler()
    aa_s_train = aa_s_scaler.fit_transform(aa_spread_X[train_mask])
    aa_s_test = aa_s_scaler.transform(aa_spread_X[test_mask])
    aa_s_model = Ridge(alpha=1.0).fit(aa_s_train, aa_spread_y[train_mask])
    aa_spread_pred = aa_s_model.predict(aa_s_test)

    aa_t_scaler = StandardScaler()
    aa_t_train = aa_t_scaler.fit_transform(aa_total_X[train_mask])
    aa_t_test = aa_t_scaler.transform(aa_total_X[test_mask])
    aa_t_model = Ridge(alpha=1.0).fit(aa_t_train, aa_total_y[train_mask])
    aa_total_pred = aa_t_model.predict(aa_t_test)

    all_results.append(evaluate_approach(
        "Asymmetric", aa_spread_pred, aa_total_pred,
        meta_actual_spread[test_mask], meta_actual_total[test_mask],
        meta_vegas_s[test_mask], meta_vegas_t[test_mask],
        meta_seasons[test_mask]
    ))

    # =====================================================================
    # APPROACH B: Score prediction → derive spread/total
    # =====================================================================
    log.info("\n" + "=" * 70)
    log.info("TRAINING: Approach B (Score Prediction)")
    log.info("=" * 70)

    sb_scaler = StandardScaler()
    sb_train_scaled = sb_scaler.fit_transform(sb_X[sb_train_mask])
    sb_test_scaled = sb_scaler.transform(sb_X[sb_test_mask])
    sb_model = Ridge(alpha=1.0).fit(sb_train_scaled, sb_y[sb_train_mask])
    sb_score_pred = sb_model.predict(sb_test_scaled)

    # Map score predictions back to game-level spread/total
    test_game_indices = np.where(test_mask)[0]
    sb_spread_pred = np.zeros(test_mask.sum())
    sb_total_pred = np.zeros(test_mask.sum())

    for i, game_global_idx in enumerate(test_game_indices):
        # Find the two sb rows for this game
        row_indices = np.where(sb_game_idx[sb_test_mask] == game_global_idx)[0]
        if len(row_indices) == 2:
            # First row = home, second row = away (by construction)
            home_pred = sb_score_pred[row_indices[0]]
            away_pred = sb_score_pred[row_indices[1]]
            sb_spread_pred[i] = away_pred - home_pred
            sb_total_pred[i] = home_pred + away_pred
        else:
            sb_spread_pred[i] = np.nan
            sb_total_pred[i] = np.nan

    all_results.append(evaluate_approach(
        "Score-Based", sb_spread_pred, sb_total_pred,
        meta_actual_spread[test_mask], meta_actual_total[test_mask],
        meta_vegas_s[test_mask], meta_vegas_t[test_mask],
        meta_seasons[test_mask]
    ))

    # Print score model coefficients
    score_feat_names = [
        'team_ppg', 'team_yards', 'team_pass_yds', 'team_rush_yds',
        'team_turnovers', 'team_first_downs',
        'team_scoring_pct', 'team_ypd',
        'opp_papg', 'opp_turnovers', 'opp_scoring_pct', 'opp_ypd',
        'team_srs', 'opp_srs',
        'team_form', 'team_momentum', 'team_streak',
        'team_rest', 'is_home_hca', 'season_progress',
    ]
    log.info("\nScore Model Coefficients:")
    coefs = list(zip(score_feat_names, sb_model.coef_))
    coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coefs:
        log.info(f"  {name:<22} {coef:+.4f}")
    log.info(f"  Intercept: {sb_model.intercept_:+.4f}")

    # =====================================================================
    # APPROACH C: Ensemble (Baseline + Score-Based)
    # =====================================================================
    log.info("\n" + "=" * 70)
    log.info("TRAINING: Approach C (Ensemble)")
    log.info("=" * 70)

    for w_bl, w_sb in [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6), (0.7, 0.3)]:
        ens_spread = w_bl * bl_spread_pred + w_sb * sb_spread_pred
        ens_total = w_bl * bl_total_pred + w_sb * sb_total_pred

        label = f"Ens {w_bl:.0%}/{w_sb:.0%}"
        all_results.append(evaluate_approach(
            label, ens_spread, ens_total,
            meta_actual_spread[test_mask], meta_actual_total[test_mask],
            meta_vegas_s[test_mask], meta_vegas_t[test_mask],
            meta_seasons[test_mask]
        ))

    # Also ensemble Asymmetric + Score-Based
    for w_aa, w_sb_w in [(0.5, 0.5)]:
        ens_spread = w_aa * aa_spread_pred + w_sb_w * sb_spread_pred
        ens_total = w_aa * aa_total_pred + w_sb_w * sb_total_pred
        all_results.append(evaluate_approach(
            "Ens A+B 50/50", ens_spread, ens_total,
            meta_actual_spread[test_mask], meta_actual_total[test_mask],
            meta_vegas_s[test_mask], meta_vegas_t[test_mask],
            meta_seasons[test_mask]
        ))

    # =====================================================================
    # COMPARISON
    # =====================================================================
    print_comparison(all_results)


if __name__ == '__main__':
    run_experiments()
