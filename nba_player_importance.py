"""
NBA Player Importance Scoring System

Calculates player importance scores and identifies when key players are out.
Key finding: When #2 star is out for AWAY team, back the injured team (57.9% ATS).
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "nba_games.db"


def get_player_importance_scores(season: int | None = None, window_games: int = 20) -> pd.DataFrame:
    """
    Calculate player importance scores based on recent performance.

    Importance = weighted combination of:
    - Minutes share (40%)
    - Points share (30%)
    - Plus/minus impact (15%)
    - Starter status (15%)

    Args:
        season: Season year (e.g., 2025). None = current season
        window_games: Number of recent games to consider

    Returns:
        DataFrame with player importance scores per team
    """
    conn = sqlite3.connect(DB_PATH)

    if season is None:
        season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year

    # Get recent player stats with proper joins
    query = """
    SELECT
        p.name as player_name,
        t.display_name as team,
        g.date as game_date,
        pgs.minutes,
        pgs.points,
        pgs.plus_minus,
        pgs.starter,
        pgs.did_not_play,
        pgs.dnp_reason
    FROM player_game_stats pgs
    JOIN games g ON pgs.game_id = g.game_id
    JOIN players p ON pgs.player_id = p.player_id
    JOIN teams t ON pgs.team_id = t.team_id
    WHERE g.season = ?
      AND pgs.did_not_play = 0
      AND pgs.minutes > 0
    ORDER BY g.date DESC
    """

    df = pd.read_sql_query(query, conn, params=(season,))
    conn.close()

    if df.empty:
        logger.warning(f"No player stats found for season {season}")
        return pd.DataFrame()

    # Get most recent N games per player
    df['game_rank'] = df.groupby(['player_name', 'team']).cumcount() + 1
    df = df[df['game_rank'] <= window_games]

    # Calculate team totals for normalization
    team_totals = df.groupby('team').agg({
        'minutes': 'sum',
        'points': 'sum'
    }).rename(columns={'minutes': 'team_minutes', 'points': 'team_points'})

    # Calculate player averages
    player_stats = df.groupby(['player_name', 'team']).agg({
        'minutes': 'mean',
        'points': 'mean',
        'plus_minus': 'mean',
        'starter': 'mean',
        'game_rank': 'count'  # games played
    }).rename(columns={'game_rank': 'games_played'})

    player_stats = player_stats.reset_index()
    player_stats = player_stats.merge(team_totals, on='team')

    # Calculate shares
    player_stats['minutes_share'] = player_stats['minutes'] / (player_stats['team_minutes'] / player_stats['games_played'] / 5)
    player_stats['points_share'] = player_stats['points'] / (player_stats['team_points'] / player_stats['games_played'] / 5)

    # Normalize plus/minus (z-score within team)
    team_pm_stats = player_stats.groupby('team')['plus_minus'].agg(['mean', 'std'])
    player_stats = player_stats.merge(team_pm_stats, on='team', suffixes=('', '_team'))
    player_stats['pm_zscore'] = (player_stats['plus_minus'] - player_stats['mean']) / player_stats['std'].replace(0, 1)
    player_stats['pm_normalized'] = (player_stats['pm_zscore'] + 3) / 6  # Normalize to 0-1 range
    player_stats['pm_normalized'] = player_stats['pm_normalized'].clip(0, 1)

    # Calculate importance score
    player_stats['importance'] = (
        0.40 * player_stats['minutes_share'].clip(0, 1) +
        0.30 * player_stats['points_share'].clip(0, 1) +
        0.15 * player_stats['pm_normalized'] +
        0.15 * player_stats['starter']
    )

    # Rank within team
    player_stats['team_rank'] = player_stats.groupby('team')['importance'].rank(ascending=False, method='first')

    # Clean up columns
    result = player_stats[[
        'player_name', 'team', 'minutes', 'points', 'plus_minus',
        'starter', 'games_played', 'importance', 'team_rank'
    ]].sort_values(['team', 'team_rank'])

    return result


def get_star_players(season: int | None = None) -> dict[str, list[str]]:
    """
    Get top 3 players (stars) for each team.

    Returns:
        Dict mapping team name to list of [#1 star, #2 star, #3 star]
    """
    importance = get_player_importance_scores(season)

    if importance.empty:
        return {}

    stars = {}
    for team in importance['team'].unique():
        team_players = importance[importance['team'] == team].sort_values('team_rank')
        top_3 = team_players.head(3)['player_name'].tolist()
        stars[team] = top_3

    return stars


def analyze_injury_edge_by_spread(min_games: int = 10) -> pd.DataFrame:
    """
    Analyze the star injury edge broken down by spread ranges.
    """
    conn = sqlite3.connect(DB_PATH)

    query = """
    WITH star_players AS (
        SELECT
            pgs.team_id,
            pgs.player_id,
            p.name as player_name,
            AVG(pgs.minutes) as avg_minutes,
            ROW_NUMBER() OVER (PARTITION BY pgs.team_id ORDER BY AVG(pgs.minutes) DESC) as star_rank
        FROM player_game_stats pgs
        JOIN games g ON pgs.game_id = g.game_id
        JOIN players p ON pgs.player_id = p.player_id
        WHERE g.season >= 2023
          AND pgs.did_not_play = 0
          AND pgs.minutes > 0
        GROUP BY pgs.team_id, pgs.player_id
        HAVING COUNT(*) >= 10
    ),
    star_dnps AS (
        SELECT
            g.game_id,
            g.date,
            ht.display_name as home_team,
            at.display_name as away_team,
            g.home_score,
            g.away_score,
            o.latest_spread as spread,
            t.display_name as injured_team,
            sp.star_rank,
            CASE WHEN pgs.team_id = g.away_team_id THEN 'AWAY' ELSE 'HOME' END as injured_side
        FROM player_game_stats pgs
        JOIN games g ON pgs.game_id = g.game_id
        JOIN teams t ON pgs.team_id = t.team_id
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        JOIN star_players sp ON pgs.player_id = sp.player_id AND pgs.team_id = sp.team_id
        LEFT JOIN odds_and_predictions o ON g.game_id = o.game_id
        WHERE pgs.did_not_play = 1
          AND pgs.dnp_reason NOT IN ("COACH'S DECISION", "NOT WITH TEAM", "REST")
          AND sp.star_rank <= 3
          AND g.home_score IS NOT NULL
          AND o.latest_spread IS NOT NULL
    )
    SELECT
        injured_side,
        star_rank,
        CASE
            WHEN ABS(spread) < 3 THEN 'Close (<3)'
            WHEN ABS(spread) < 7 THEN 'Medium (3-7)'
            ELSE 'Large (7+)'
        END as spread_range,
        COUNT(*) as games,
        SUM(CASE
            WHEN injured_side = 'AWAY' AND (away_score + spread) > home_score THEN 1
            WHEN injured_side = 'HOME' AND (home_score - spread) > away_score THEN 1
            ELSE 0
        END) as injured_team_covers
    FROM star_dnps
    GROUP BY injured_side, star_rank, spread_range
    HAVING COUNT(*) >= ?
    ORDER BY injured_side, star_rank, spread_range
    """

    df = pd.read_sql_query(query, conn, params=(min_games,))
    conn.close()

    if df.empty:
        return df

    df['cover_rate'] = df['injured_team_covers'] / df['games']
    df['roi'] = (df['cover_rate'] * 1.91 - 1) * 100  # At -110 odds

    return df


def get_recent_dnps(team: str, days: int = 7, season: int | None = None) -> pd.DataFrame:
    """
    Get recent DNPs for a team to identify injury patterns.
    """
    conn = sqlite3.connect(DB_PATH)

    if season is None:
        season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year

    query = """
    SELECT
        p.name as player_name,
        g.date as game_date,
        pgs.dnp_reason
    FROM player_game_stats pgs
    JOIN games g ON pgs.game_id = g.game_id
    JOIN players p ON pgs.player_id = p.player_id
    JOIN teams t ON pgs.team_id = t.team_id
    WHERE t.display_name = ?
      AND pgs.did_not_play = 1
      AND pgs.dnp_reason != "COACH'S DECISION"
      AND g.season = ?
    ORDER BY g.date DESC
    LIMIT 50
    """

    df = pd.read_sql_query(query, conn, params=(team, season))
    conn.close()

    return df


def check_star_injury_signal(
    home_team: str,
    away_team: str,
    home_injuries: list[str] | None = None,
    away_injuries: list[str] | None = None,
    season: int | None = None
) -> dict:
    """
    Check if there's a betting signal based on star injuries.

    Key finding: When #2 star is out for AWAY team, back the injured team.
    Win rate: 57.9%, ROI: +10.5%

    Args:
        home_team: Home team display name
        away_team: Away team display name
        home_injuries: List of injured player names for home team
        away_injuries: List of injured player names for away team
        season: Season year

    Returns:
        Dict with signal info
    """
    stars = get_star_players(season)

    result = {
        'home_team': home_team,
        'away_team': away_team,
        'signal': None,
        'signal_strength': 0,
        'signal_reason': None,
        'home_star_injuries': [],
        'away_star_injuries': []
    }

    home_stars = stars.get(home_team, [])
    away_stars = stars.get(away_team, [])

    home_injuries = home_injuries or []
    away_injuries = away_injuries or []

    # Check which stars are injured
    for i, star in enumerate(home_stars):
        if star in home_injuries:
            result['home_star_injuries'].append({'player': star, 'rank': i + 1})

    for i, star in enumerate(away_stars):
        if star in away_injuries:
            result['away_star_injuries'].append({'player': star, 'rank': i + 1})

    # Generate signal based on #2 star rule
    away_injured_ranks = [inj['rank'] for inj in result['away_star_injuries']]
    home_injured_ranks = [inj['rank'] for inj in result['home_star_injuries']]

    if 2 in away_injured_ranks and 1 not in away_injured_ranks:
        # Away #2 star out, #1 still playing - our best signal
        result['signal'] = 'BACK_AWAY'
        result['signal_strength'] = 2
        result['signal_reason'] = "Away #2 star out - historical: 57.9% ATS, +10.5% ROI"

    elif 2 in home_injured_ranks and 1 not in home_injured_ranks:
        # Home #2 star out - fade home (back away)
        result['signal'] = 'BACK_AWAY'
        result['signal_strength'] = 1
        result['signal_reason'] = "Home #2 star out - historical: 55.2% fade rate"

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("NBA Player Importance Analysis")
    print("=" * 60)

    # Get star players
    print("\nCurrent Star Players by Team (2026 season):")
    print("-" * 40)
    stars = get_star_players(season=2026)
    if stars:
        for team in sorted(stars.keys())[:10]:  # Show first 10 teams
            players = stars[team]
            print(f"{team}: {', '.join(players[:3])}")
        print("...")
    else:
        print("No star players found - trying 2025 season...")
        stars = get_star_players(season=2025)
        for team in sorted(stars.keys())[:10]:
            players = stars[team]
            print(f"{team}: {', '.join(players[:3])}")
        print("...")

    # Analyze injury edge by spread
    print("\nInjury Edge Analysis by Spread Range:")
    print("-" * 60)
    edge_df = analyze_injury_edge_by_spread(min_games=10)

    if not edge_df.empty:
        # Focus on the #2 star AWAY edge
        away_2 = edge_df[(edge_df['injured_side'] == 'AWAY') & (edge_df['star_rank'] == 2)]
        if not away_2.empty:
            print("\n#2 Star Out for AWAY Team (Back injured team):")
            for _, row in away_2.iterrows():
                print(f"  {row['spread_range']}: {row['games']} games, "
                      f"{row['cover_rate']:.1%} cover, {row['roi']:+.1f}% ROI")
        else:
            print("No data for #2 star AWAY injuries")

        # Show all combinations
        print("\nAll Star Injury Combinations:")
        for _, row in edge_df.iterrows():
            indicator = " <--" if row['cover_rate'] > 0.55 else ""
            print(f"  {row['injured_side']} #{int(row['star_rank'])} - {row['spread_range']}: "
                  f"{row['games']} games, {row['cover_rate']:.1%} cover, {row['roi']:+.1f}% ROI{indicator}")
    else:
        print("No injury edge data available")

    # Show importance scores for a sample team
    print("\nSample: Lakers Player Importance Scores:")
    print("-" * 40)
    importance = get_player_importance_scores(season=2026)
    if importance.empty:
        importance = get_player_importance_scores(season=2025)

    if not importance.empty:
        lakers = importance[importance['team'] == 'Los Angeles Lakers'].head(8)
        if not lakers.empty:
            for _, row in lakers.iterrows():
                name = row['player_name'][:20]
                print(f"  #{int(row['team_rank'])}: {name:<20} "
                      f"Importance: {row['importance']:.3f} "
                      f"({row['minutes']:.1f} min, {row['points']:.1f} pts)")
        else:
            # Try any team
            any_team = importance['team'].iloc[0]
            team_data = importance[importance['team'] == any_team].head(8)
            print(f"(Showing {any_team} instead)")
            for _, row in team_data.iterrows():
                name = row['player_name'][:20]
                print(f"  #{int(row['team_rank'])}: {name:<20} "
                      f"Importance: {row['importance']:.3f} "
                      f"({row['minutes']:.1f} min, {row['points']:.1f} pts)")
