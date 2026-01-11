"""
System Health Check Module
Provides health status for all sports data and models.
Used by Streamlit dashboard banner and /health API endpoint.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

StatusLevel = Literal["ok", "warning", "error"]


@dataclass
class SportHealth:
    """Health status for a single sport."""
    sport: str
    status: StatusLevel
    last_data_update: datetime | None
    last_prediction_update: datetime | None
    upcoming_games: int
    games_with_predictions: int
    games_with_odds: int
    issues: list[str]

    @property
    def data_age_hours(self) -> float | None:
        if self.last_data_update is None:
            return None
        return (datetime.now() - self.last_data_update).total_seconds() / 3600

    @property
    def prediction_age_hours(self) -> float | None:
        if self.last_prediction_update is None:
            return None
        return (datetime.now() - self.last_prediction_update).total_seconds() / 3600


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: StatusLevel
    sports: dict[str, SportHealth]
    last_daily_update: datetime | None
    daily_update_success: bool | None
    issues: list[str]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict for API."""
        return {
            "status": self.status,
            "last_daily_update": self.last_daily_update.isoformat() if self.last_daily_update else None,
            "daily_update_success": self.daily_update_success,
            "issues": self.issues,
            "sports": {
                sport: {
                    "status": h.status,
                    "last_data_update": h.last_data_update.isoformat() if h.last_data_update else None,
                    "last_prediction_update": h.last_prediction_update.isoformat() if h.last_prediction_update else None,
                    "data_age_hours": round(h.data_age_hours, 1) if h.data_age_hours else None,
                    "prediction_age_hours": round(h.prediction_age_hours, 1) if h.prediction_age_hours else None,
                    "upcoming_games": h.upcoming_games,
                    "games_with_predictions": h.games_with_predictions,
                    "games_with_odds": h.games_with_odds,
                    "issues": h.issues,
                }
                for sport, h in self.sports.items()
            }
        }


def check_sport_health(sport: str) -> SportHealth:
    """Check health status for a single sport."""
    db_path = f"{sport.lower()}_games.db"
    issues = []

    if not os.path.exists(db_path):
        return SportHealth(
            sport=sport.upper(),
            status="error",
            last_data_update=None,
            last_prediction_update=None,
            upcoming_games=0,
            games_with_predictions=0,
            games_with_odds=0,
            issues=[f"Database {db_path} not found"]
        )

    # Get database file modification time as proxy for last update
    db_mtime = datetime.fromtimestamp(os.path.getmtime(db_path))

    # Query database using context manager to ensure connection is closed
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check upcoming games
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM games
                WHERE completed = 0
                AND game_date_eastern >= date('now')
                AND game_date_eastern <= date('now', '+7 days')
            """)
            upcoming_games = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            # Fallback for databases without game_date_eastern
            cursor.execute("""
                SELECT COUNT(*) FROM games
                WHERE completed = 0
                AND date >= date('now')
                AND date <= date('now', '+7 days')
            """)
            upcoming_games = cursor.fetchone()[0]

        # Check games with predictions
        cursor.execute("""
            SELECT COUNT(*) FROM games g
            JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.completed = 0
            AND o.predicted_home_score IS NOT NULL
        """)
        games_with_predictions = cursor.fetchone()[0]

        # Check games with odds
        cursor.execute("""
            SELECT COUNT(*) FROM games g
            JOIN odds_and_predictions o ON g.game_id = o.game_id
            WHERE g.completed = 0
            AND (o.latest_spread IS NOT NULL OR o.opening_spread IS NOT NULL)
        """)
        games_with_odds = cursor.fetchone()[0]

        # Get last prediction timestamp (column may not exist in all DBs)
        last_prediction_update = None
        try:
            cursor.execute("""
                SELECT MAX(prediction_created) FROM odds_and_predictions
                WHERE predicted_home_score IS NOT NULL
            """)
            last_pred_row = cursor.fetchone()[0]
            if last_pred_row:
                try:
                    # Parse ISO datetime, handling Z suffix
                    if last_pred_row.endswith('Z'):
                        last_prediction_update = datetime.fromisoformat(last_pred_row[:-1])
                    else:
                        last_prediction_update = datetime.fromisoformat(last_pred_row)
                except (ValueError, AttributeError):
                    pass
        except sqlite3.OperationalError:
            # Column doesn't exist - use db mtime as fallback
            pass

    # Determine status and issues
    status: StatusLevel = "ok"

    # Check for stale data (> 24 hours)
    data_age_hours = (datetime.now() - db_mtime).total_seconds() / 3600
    if data_age_hours > 48:
        status = "error"
        issues.append(f"Data is {data_age_hours:.0f} hours old")
    elif data_age_hours > 24:
        status = "warning"
        issues.append(f"Data is {data_age_hours:.0f} hours old")

    # Check for missing predictions
    if upcoming_games > 0 and games_with_predictions == 0:
        status = "error"
        issues.append("No predictions for upcoming games")
    elif upcoming_games > 0 and games_with_predictions < upcoming_games * 0.5:
        if status != "error":
            status = "warning"
        issues.append(f"Only {games_with_predictions}/{upcoming_games} games have predictions")

    # Check for missing odds
    if upcoming_games > 0 and games_with_odds == 0:
        if status != "error":
            status = "warning"
        issues.append("No Vegas odds for upcoming games")

    # Check prediction freshness
    if last_prediction_update:
        pred_age_hours = (datetime.now() - last_prediction_update).total_seconds() / 3600
        if pred_age_hours > 24:
            if status != "error":
                status = "warning"
            issues.append(f"Predictions are {pred_age_hours:.0f} hours old")

    return SportHealth(
        sport=sport.upper(),
        status=status,
        last_data_update=db_mtime,
        last_prediction_update=last_prediction_update,
        upcoming_games=upcoming_games,
        games_with_predictions=games_with_predictions,
        games_with_odds=games_with_odds,
        issues=issues
    )


def parse_daily_update_log() -> tuple[datetime | None, bool | None, list[str]]:
    """
    Parse the most recent daily update log to get status.

    Returns:
        Tuple of (last_update_time, success, sport_failures)
    """
    log_dir = Path("logs")
    if not log_dir.exists():
        return None, None, []

    # Find most recent log file
    log_files = sorted(log_dir.glob("daily_update_*.log"), reverse=True)
    if not log_files:
        return None, None, []

    latest_log = log_files[0]
    log_time = datetime.fromtimestamp(latest_log.stat().st_mtime)

    # Parse log for failures using regex
    failures = []
    try:
        with open(latest_log, 'r') as f:
            content = f.read()

        # Look for the SUMMARY section - format is "  NBA: FAILED"
        if "SUMMARY" in content:
            # Use regex to find sport failures: "NBA: FAILED", "CBB: FAILED", etc.
            for match in re.finditer(r'(NBA|CBB|NFL|CFB):\s*FAILED', content):
                sport = match.group(1)
                if sport not in failures:
                    failures.append(sport)

        success = len(failures) == 0
        return log_time, success, failures

    except Exception as e:
        log.warning(f"Failed to parse daily update log: {e}")
        return log_time, None, []


def check_system_health(sports: list[str] | None = None) -> SystemHealth:
    """
    Check overall system health.

    Args:
        sports: List of sports to check. Defaults to ['nfl', 'cfb', 'nba', 'cbb']

    Returns:
        SystemHealth object with overall status and per-sport details
    """
    if sports is None:
        sports = ['nfl', 'cfb', 'nba', 'cbb']

    sport_health = {sport.upper(): check_sport_health(sport) for sport in sports}

    # Parse daily update log
    last_update, update_success, failures = parse_daily_update_log()

    # Determine overall status
    issues = []
    overall_status: StatusLevel = "ok"

    for sport, health in sport_health.items():
        if health.status == "error":
            overall_status = "error"
            issues.append(f"{sport}: {'; '.join(health.issues)}")
        elif health.status == "warning" and overall_status != "error":
            overall_status = "warning"
            issues.append(f"{sport}: {'; '.join(health.issues)}")

    # Add daily update failures
    if failures:
        if overall_status != "error":
            overall_status = "warning"
        issues.append(f"Daily update failed for: {', '.join(failures)}")

    return SystemHealth(
        status=overall_status,
        sports=sport_health,
        last_daily_update=last_update,
        daily_update_success=update_success,
        issues=issues
    )


def get_health_banner_html(health: SystemHealth) -> str:
    """Generate HTML for health status banner in Streamlit."""
    if health.status == "ok":
        bg_color = "#d4edda"
        border_color = "#28a745"
        icon = "&#10003;"  # checkmark
        text = "All systems operational"
    elif health.status == "warning":
        bg_color = "#fff3cd"
        border_color = "#ffc107"
        icon = "&#9888;"  # warning
        text = f"Warning: {'; '.join(health.issues[:2])}"
    else:
        bg_color = "#f8d7da"
        border_color = "#dc3545"
        icon = "&#10007;"  # X mark
        text = f"Error: {'; '.join(health.issues[:2])}"

    # Add last update time
    if health.last_daily_update:
        age_hours = (datetime.now() - health.last_daily_update).total_seconds() / 3600
        if age_hours < 1:
            age_str = f"{int(age_hours * 60)} min ago"
        elif age_hours < 24:
            age_str = f"{int(age_hours)} hours ago"
        else:
            age_str = f"{int(age_hours / 24)} days ago"
        time_text = f" | Last update: {age_str}"
    else:
        time_text = ""

    return f"""
    <div style="
        background-color: {bg_color};
        border-left: 4px solid {border_color};
        padding: 10px 15px;
        margin-bottom: 15px;
        border-radius: 4px;
        font-size: 14px;
    ">
        <span style="font-size: 16px; margin-right: 8px;">{icon}</span>
        <strong>{text}</strong>{time_text}
    </div>
    """


if __name__ == "__main__":
    # Test health check
    health = check_system_health()
    print(f"Overall Status: {health.status}")
    print(f"Last Daily Update: {health.last_daily_update}")
    print(f"Update Success: {health.daily_update_success}")
    print(f"Issues: {health.issues}")
    print()
    for sport, h in health.sports.items():
        print(f"{sport}:")
        print(f"  Status: {h.status}")
        print(f"  Upcoming: {h.upcoming_games}, Predictions: {h.games_with_predictions}, Odds: {h.games_with_odds}")
        if h.issues:
            print(f"  Issues: {h.issues}")
