"""Analyze Enhanced Ridge totals performance for NFL and CFB."""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def analyze_nfl_totals():
    """Walk-forward analysis of NFL Enhanced Ridge totals."""
    print("=" * 70)
    print("NFL ENHANCED RIDGE TOTALS ANALYSIS")
    print("=" * 70)

    DECAY = 0.96
    PREV_HALF_LIFE = 4.0
    MIN_GAMES = 2

    conn = sqlite3.connect('nfl_games.db')
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               hs.total_yards as home_yards, hs.turnovers as home_to,
               aws.total_yards as away_yards, aws.turnovers as away_to,
               o.latest_total as vegas_total
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score > 0
        ORDER BY g.date
    ''', conn)
    conn.close()

    print(f"Total games: {len(games)}")
    games = games[games['vegas_total'].notna()].copy()
    print(f"Games with vegas totals: {len(games)}")

    team_stats = defaultdict(lambda: defaultdict(lambda: {
        'ppg': [], 'papg': [], 'wts': [],
        'yards': [], 'yards_wts': [],
        'turnovers': [], 'to_wts': [],
        'margins': [],
    }))
    prev_ratings = {}
    league_avg = {'ppg': 22.0, 'papg': 22.0, 'yards': 330.0, 'turnovers': 1.3}
    total_X, total_y = [], []
    results = []

    def wavg(vals, wts):
        if not vals or not wts:
            return None
        return float(np.average(vals, weights=wts))

    def get_stats(tid, season):
        td = team_stats[tid][season]
        n = len(td['ppg'])
        if n == 0:
            prev = prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', league_avg['ppg']),
                'papg': prev.get('papg', league_avg['papg']),
                'yards': prev.get('yards', league_avg['yards']),
                'turnovers': prev.get('turnovers', league_avg['turnovers']),
                'games': 0, 'margins': [],
            }
        ppg = wavg(td['ppg'], td['wts'])
        papg = wavg(td['papg'], td['wts'])
        yards = wavg(td['yards'], td['yards_wts']) if td['yards'] else league_avg['yards']
        to = wavg(td['turnovers'], td['to_wts']) if td['turnovers'] else league_avg['turnovers']
        prev = prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)
        return {
            'ppg': blend * prev.get('ppg', league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', league_avg['papg']) + (1 - blend) * papg,
            'yards': yards, 'turnovers': to, 'games': n, 'margins': td['margins'][-4:],
        }

    def recent_form(margins):
        return float(np.mean(margins[-4:])) if len(margins) >= 4 else 0.0

    def extract_total_features(hs, aws, week):
        h_form = recent_form(hs.get('margins', []))
        a_form = recent_form(aws.get('margins', []))
        return np.array([
            hs['ppg'] + aws['ppg'], hs['papg'] + aws['papg'],
            hs['yards'] + aws['yards'], hs['turnovers'] + aws['turnovers'],
            3.8, 0.70, abs(h_form) + abs(a_form), 0.0, 0.0, 0.0,
            min(week / 17.0, 1.0), min(hs['games'] / 10.0, 1.0),
            min(aws['games'] / 10.0, 1.0), (hs['games'] + aws['games']) / 34.0,
        ])

    seasons = sorted(games['season'].unique())
    total_model = None
    total_scaler = StandardScaler()

    for season in seasons:
        if season > seasons[0]:
            for tid in team_stats:
                ps = season - 1
                if ps in team_stats[tid]:
                    td = team_stats[tid][ps]
                    if td['ppg']:
                        prev_ratings[tid] = {
                            'ppg': np.mean(td['ppg']), 'papg': np.mean(td['papg']),
                            'yards': np.mean(td['yards']) if td['yards'] else 330.0,
                            'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 1.3,
                        }

        for _, g in games[games['season'] == season].iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_total = g['home_score'] + g['away_score']
            vegas_total = g['vegas_total']
            hs, aws = get_stats(hid, season), get_stats(aid, season)

            if hs['games'] >= MIN_GAMES and aws['games'] >= MIN_GAMES:
                feat = extract_total_features(hs, aws, g['week'])
                total_X.append(feat)
                total_y.append(actual_total)

                if len(total_X) % 50 == 0 and len(total_X) >= 100:
                    X = np.array(total_X)
                    y = np.array(total_y)
                    total_scaler.fit(X)
                    total_model = Ridge(alpha=1.0).fit(total_scaler.transform(X), y)

                if total_model is not None:
                    pred_total = total_model.predict(total_scaler.transform(feat.reshape(1, -1)))[0]
                    edge = pred_total - vegas_total
                    result = actual_total - vegas_total
                    if abs(result) >= 0.5:
                        over_hit = (edge > 0 and result > 0) or (edge < 0 and result < 0)
                        results.append({
                            'season': season, 'week': g['week'], 'vegas_total': vegas_total,
                            'pred_total': pred_total, 'actual_total': actual_total,
                            'edge': edge, 'result': result, 'over_hit': over_hit,
                            'direction': 'OVER' if edge > 0 else 'UNDER',
                        })

            margin = g['home_score'] - g['away_score']
            for tid, pf, pa, yards, to in [
                (hid, g['home_score'], g['away_score'], g['home_yards'], g['home_to']),
                (aid, g['away_score'], g['home_score'], g['away_yards'], g['away_to'])
            ]:
                td = team_stats[tid][season]
                td['wts'] = [w * DECAY for w in td['wts']]
                td['ppg'].append(pf)
                td['papg'].append(pa)
                td['wts'].append(1.0)
                td['margins'].append(pf - pa)
                if pd.notna(yards):
                    td['yards_wts'] = [w * DECAY for w in td['yards_wts']]
                    td['yards'].append(yards)
                    td['yards_wts'].append(1.0)
                if pd.notna(to):
                    td['to_wts'] = [w * DECAY for w in td['to_wts']]
                    td['turnovers'].append(to)
                    td['to_wts'].append(1.0)

    return pd.DataFrame(results)


def analyze_cfb_totals():
    """Walk-forward analysis of CFB Enhanced Ridge totals."""
    print("\n" + "=" * 70)
    print("CFB ENHANCED RIDGE TOTALS ANALYSIS")
    print("=" * 70)

    DECAY = 0.88
    PREV_HALF_LIFE = 5.0
    MIN_GAMES = 2

    conn = sqlite3.connect('cfb_games.db')
    games = pd.read_sql_query('''
        SELECT g.game_id, g.season, g.week, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               hs.total_yards as home_yards, hs.turnovers as home_to,
               aws.total_yards as away_yards, aws.turnovers as away_to,
               o.latest_total as vegas_total
        FROM games g
        LEFT JOIN team_game_stats hs ON g.game_id = hs.game_id AND g.home_team_id = hs.team_id
        LEFT JOIN team_game_stats aws ON g.game_id = aws.game_id AND g.away_team_id = aws.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE g.completed = 1 AND g.home_score IS NOT NULL AND g.postseason_type IS NULL
        ORDER BY g.date
    ''', conn)
    conn.close()

    print(f"Total games: {len(games)}")
    games = games[games['vegas_total'].notna()].copy()
    print(f"Games with vegas totals: {len(games)}")

    team_stats = defaultdict(lambda: defaultdict(lambda: {
        'ppg': [], 'papg': [], 'wts': [],
        'yards': [], 'yards_wts': [],
        'turnovers': [], 'to_wts': [],
        'margins': [],
    }))
    prev_ratings = {}
    league_avg = {'ppg': 28.0, 'papg': 28.0, 'yards': 400.0, 'turnovers': 1.5}
    total_X, total_y = [], []
    results = []

    def wavg(vals, wts):
        if not vals or not wts:
            return None
        return float(np.average(vals, weights=wts))

    def get_stats(tid, season):
        td = team_stats[tid][season]
        n = len(td['ppg'])
        if n == 0:
            prev = prev_ratings.get(tid, {})
            return {
                'ppg': prev.get('ppg', league_avg['ppg']),
                'papg': prev.get('papg', league_avg['papg']),
                'yards': prev.get('yards', league_avg['yards']),
                'turnovers': prev.get('turnovers', league_avg['turnovers']),
                'games': 0, 'margins': [],
            }
        ppg = wavg(td['ppg'], td['wts'])
        papg = wavg(td['papg'], td['wts'])
        yards = wavg(td['yards'], td['yards_wts']) if td['yards'] else league_avg['yards']
        to = wavg(td['turnovers'], td['to_wts']) if td['turnovers'] else league_avg['turnovers']
        prev = prev_ratings.get(tid, {})
        blend = 0.5 ** (n / PREV_HALF_LIFE)
        return {
            'ppg': blend * prev.get('ppg', league_avg['ppg']) + (1 - blend) * ppg,
            'papg': blend * prev.get('papg', league_avg['papg']) + (1 - blend) * papg,
            'yards': yards, 'turnovers': to, 'games': n, 'margins': td['margins'][-4:],
        }

    def recent_form(margins):
        return float(np.mean(margins[-4:])) if len(margins) >= 4 else 0.0

    def extract_total_features(hs, aws, week):
        h_form = recent_form(hs.get('margins', []))
        a_form = recent_form(aws.get('margins', []))
        return np.array([
            hs['ppg'] + aws['ppg'], hs['papg'] + aws['papg'],
            hs['yards'] + aws['yards'], hs['turnovers'] + aws['turnovers'],
            4.4, 64.0, 0.76, abs(h_form) + abs(a_form),
            0.0, min(week / 14.0, 1.0),
            min(hs['games'] / 10.0, 1.0), min(aws['games'] / 10.0, 1.0),
            (hs['games'] + aws['games']) / 24.0, 1.0,
        ])

    seasons = sorted(games['season'].unique())
    total_model = None
    total_scaler = StandardScaler()

    for season in seasons:
        if season > seasons[0]:
            for tid in team_stats:
                ps = season - 1
                if ps in team_stats[tid]:
                    td = team_stats[tid][ps]
                    if td['ppg']:
                        prev_ratings[tid] = {
                            'ppg': np.mean(td['ppg']), 'papg': np.mean(td['papg']),
                            'yards': np.mean(td['yards']) if td['yards'] else 400.0,
                            'turnovers': np.mean(td['turnovers']) if td['turnovers'] else 1.5,
                        }

        for _, g in games[games['season'] == season].iterrows():
            hid, aid = g['home_team_id'], g['away_team_id']
            actual_total = g['home_score'] + g['away_score']
            vegas_total = g['vegas_total']
            hs, aws = get_stats(hid, season), get_stats(aid, season)

            if hs['games'] >= MIN_GAMES and aws['games'] >= MIN_GAMES:
                feat = extract_total_features(hs, aws, g['week'])
                total_X.append(feat)
                total_y.append(actual_total)

                if len(total_X) % 100 == 0 and len(total_X) >= 200:
                    X = np.array(total_X)
                    y = np.array(total_y)
                    total_scaler.fit(X)
                    total_model = Ridge(alpha=1.0).fit(total_scaler.transform(X), y)

                if total_model is not None:
                    pred_total = total_model.predict(total_scaler.transform(feat.reshape(1, -1)))[0]
                    edge = pred_total - vegas_total
                    result = actual_total - vegas_total
                    if abs(result) >= 0.5:
                        over_hit = (edge > 0 and result > 0) or (edge < 0 and result < 0)
                        results.append({
                            'season': season, 'week': g['week'], 'vegas_total': vegas_total,
                            'pred_total': pred_total, 'actual_total': actual_total,
                            'edge': edge, 'result': result, 'over_hit': over_hit,
                            'direction': 'OVER' if edge > 0 else 'UNDER',
                        })

            for tid, pf, pa, yards, to in [
                (hid, g['home_score'], g['away_score'], g['home_yards'], g['home_to']),
                (aid, g['away_score'], g['home_score'], g['away_yards'], g['away_to'])
            ]:
                td = team_stats[tid][season]
                td['wts'] = [w * DECAY for w in td['wts']]
                td['ppg'].append(pf)
                td['papg'].append(pa)
                td['wts'].append(1.0)
                td['margins'].append(pf - pa)
                if pd.notna(yards):
                    td['yards_wts'] = [w * DECAY for w in td['yards_wts']]
                    td['yards'].append(yards)
                    td['yards_wts'].append(1.0)
                if pd.notna(to):
                    td['to_wts'] = [w * DECAY for w in td['to_wts']]
                    td['turnovers'].append(to)
                    td['to_wts'].append(1.0)

    return pd.DataFrame(results)


def print_analysis(df, sport):
    """Print analysis results for a sport."""
    print(f"\nPredictions evaluated: {len(df)}")
    print(f"Overall O/U: {df['over_hit'].mean()*100:.1f}% ({int(df['over_hit'].sum())}/{len(df)})")

    print("\nBY EDGE THRESHOLD:")
    for t in [2, 3, 4, 5, 6, 7, 8]:
        m = df['edge'].abs() >= t
        if m.sum() >= 10:
            print(f"  {t}+ pts: {df.loc[m,'over_hit'].mean()*100:.1f}% ({int(df.loc[m,'over_hit'].sum())}/{m.sum()})")

    print("\nOVER vs UNDER:")
    for d in ['OVER', 'UNDER']:
        m = df['direction'] == d
        if m.sum() > 0:
            print(f"  {d}: {df.loc[m,'over_hit'].mean()*100:.1f}% ({int(df.loc[m,'over_hit'].sum())}/{m.sum()})")

    print("\nOVER by threshold:")
    for t in [4, 5, 6, 7, 8]:
        m = (df['direction'] == 'OVER') & (df['edge'] >= t)
        if m.sum() >= 10:
            print(f"  OVER {t}+: {df.loc[m,'over_hit'].mean()*100:.1f}% ({int(df.loc[m,'over_hit'].sum())}/{m.sum()})")

    print("\nUNDER by threshold:")
    for t in [4, 5, 6, 7, 8]:
        m = (df['direction'] == 'UNDER') & (df['edge'] <= -t)
        if m.sum() >= 10:
            print(f"  UNDER {t}+: {df.loc[m,'over_hit'].mean()*100:.1f}% ({int(df.loc[m,'over_hit'].sum())}/{m.sum()})")

    print("\nBY SEASON:")
    for s in sorted(df['season'].unique()):
        m = df['season'] == s
        print(f"  {s}: {df.loc[m,'over_hit'].mean()*100:.1f}% ({int(df.loc[m,'over_hit'].sum())}/{m.sum()})")


def print_fade_analysis(df, sport):
    """Analyze fade strategies."""
    print(f"\n{'='*60}")
    print(f"{sport} FADE UNDER STRATEGY")
    print("="*60)

    # When model says UNDER, what actually happens?
    under = df[df['direction'] == 'UNDER'].copy()
    under['actual_went_over'] = under['result'] > 0

    print(f"\nWhen model says UNDER ({len(under)} games):")
    over_count = under['actual_went_over'].sum()
    print(f"  Game actually went OVER: {int(over_count)}/{len(under)} = {over_count/len(under)*100:.1f}%")
    print(f"  Game actually went UNDER: {len(under)-int(over_count)}/{len(under)} = {(1-over_count/len(under))*100:.1f}%")

    print("\n" + "-"*60)
    print("FADE UNDER BY THRESHOLD (bet OVER when model says UNDER)")
    print("-"*60)

    for thresh in [2, 3, 4, 5, 6, 7]:
        mask = under['edge'] <= -thresh
        if mask.sum() >= 10:
            subset = under[mask]
            fade_wins = subset['actual_went_over'].sum()
            n = len(subset)
            pct = fade_wins / n * 100
            roi = (pct/100 - 0.524) / 0.524 * 100  # vs 52.4% breakeven
            print(f"  UNDER {thresh}+ edge -> Fade to OVER: {pct:.1f}% ({int(fade_wins)}/{n}) | ROI: {roi:+.1f}%")

    print("\n" + "-"*60)
    print("BY SEASON (Fade UNDER -> bet OVER)")
    print("-"*60)
    for season in sorted(under['season'].unique()):
        mask = under['season'] == season
        if mask.sum() >= 5:
            subset = under[mask]
            pct = subset['actual_went_over'].mean() * 100
            print(f"  {season}: {pct:.1f}% ({int(subset['actual_went_over'].sum())}/{len(subset)})")


if __name__ == '__main__':
    nfl_df = analyze_nfl_totals()
    print_analysis(nfl_df, 'NFL')
    print_fade_analysis(nfl_df, 'NFL')

    cfb_df = analyze_cfb_totals()
    print_analysis(cfb_df, 'CFB')
