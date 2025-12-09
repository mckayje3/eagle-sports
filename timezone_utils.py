"""
Timezone Utilities for Sports Prediction System
All times are standardized to US Eastern Time.
"""

from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Fallback for older Python


# Standard timezone for all operations
EASTERN = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')


def now_eastern():
    """Get current datetime in Eastern Time."""
    return datetime.now(EASTERN)


def today_eastern():
    """Get today's date in Eastern Time."""
    return now_eastern().date()


def utc_to_eastern(dt_string):
    """
    Convert UTC datetime string (from ESPN API) to Eastern Time.

    ESPN API returns dates like: "2024-12-08T01:30Z" (UTC)
    This converts to Eastern and returns datetime object.

    Args:
        dt_string: ISO format datetime string, typically ending in 'Z' (UTC)

    Returns:
        datetime object in Eastern Time (timezone-aware)
    """
    if not dt_string:
        return None

    try:
        # Remove 'Z' suffix if present and parse as UTC
        dt_string = dt_string.replace('Z', '+00:00')

        # Handle various ISO formats
        if '+' in dt_string or '-' in dt_string[10:]:  # Has timezone info
            dt = datetime.fromisoformat(dt_string)
        else:
            # No timezone, assume UTC
            dt = datetime.fromisoformat(dt_string).replace(tzinfo=UTC)

        # Convert to Eastern
        return dt.astimezone(EASTERN)
    except (ValueError, TypeError):
        return None


def utc_to_eastern_date(dt_string):
    """
    Convert UTC datetime string to Eastern Time date string (YYYY-MM-DD).

    This is the key function for fixing the late-night game issue.
    A game at 2024-12-09T02:00Z (2 AM UTC) is actually
    2024-12-08 9:00 PM Eastern, so should be dated December 8th.

    Args:
        dt_string: ISO format datetime string from ESPN API

    Returns:
        Date string in YYYY-MM-DD format (Eastern Time)
    """
    eastern_dt = utc_to_eastern(dt_string)
    if eastern_dt:
        return eastern_dt.strftime('%Y-%m-%d')
    return None


def utc_to_eastern_iso(dt_string):
    """
    Convert UTC datetime string to Eastern Time ISO format.
    Preserves full datetime with timezone info.

    Args:
        dt_string: ISO format datetime string from ESPN API

    Returns:
        ISO format datetime string in Eastern Time
    """
    eastern_dt = utc_to_eastern(dt_string)
    if eastern_dt:
        return eastern_dt.isoformat()
    return dt_string  # Return original if conversion fails


def format_game_time(dt_string):
    """
    Format game time for display (e.g., "7:30 PM ET").

    Args:
        dt_string: ISO format datetime string

    Returns:
        Formatted time string like "7:30 PM ET"
    """
    eastern_dt = utc_to_eastern(dt_string)
    if eastern_dt:
        return eastern_dt.strftime('%-I:%M %p ET').replace(' 0', ' ')
    return ''


def format_game_datetime(dt_string):
    """
    Format game date and time for display.

    Args:
        dt_string: ISO format datetime string

    Returns:
        Formatted string like "Sun Dec 8, 7:30 PM ET"
    """
    eastern_dt = utc_to_eastern(dt_string)
    if eastern_dt:
        return eastern_dt.strftime('%a %b %-d, %-I:%M %p ET')
    return ''


def is_game_today(dt_string):
    """Check if a game is scheduled for today (Eastern Time)."""
    game_date = utc_to_eastern_date(dt_string)
    if game_date:
        return game_date == today_eastern().strftime('%Y-%m-%d')
    return False


def is_game_past(dt_string):
    """Check if a game datetime is in the past (Eastern Time)."""
    eastern_dt = utc_to_eastern(dt_string)
    if eastern_dt:
        return eastern_dt < now_eastern()
    return False


def get_eastern_date_for_query(days_offset=0):
    """
    Get a date string for database queries.

    Args:
        days_offset: Number of days from today (negative for past)

    Returns:
        Date string in YYYY-MM-DD format
    """
    target_date = today_eastern() + timedelta(days=days_offset)
    return target_date.strftime('%Y-%m-%d')


# For backwards compatibility - these can be imported by scrapers
def convert_espn_date(espn_date_string):
    """
    Main function for scrapers to convert ESPN API dates.
    Returns the date portion only (YYYY-MM-DD) in Eastern Time.
    """
    return utc_to_eastern_date(espn_date_string)


def convert_espn_datetime(espn_date_string):
    """
    Convert ESPN datetime to full ISO format in Eastern Time.
    """
    return utc_to_eastern_iso(espn_date_string)
