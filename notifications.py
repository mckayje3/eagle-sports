"""
Notification Module for Sports Prediction System
Sends push notifications on update failures using ntfy.sh.

Setup:
1. Install ntfy app on your phone (iOS/Android)
2. Subscribe to your topic (e.g., "eagle-eye-sports-yourname")
3. Set NTFY_TOPIC environment variable or update config below

Usage:
    from notifications import notify_failure, notify_success
    notify_failure("NBA", "Model failed to load: KeyError 'team_stats'")
    notify_success("All sports updated successfully")
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Literal

import requests

log = logging.getLogger(__name__)

# Configuration - set via environment variable or change default
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "eagle-eye-sports")
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
NOTIFICATIONS_ENABLED = os.environ.get("NOTIFICATIONS_ENABLED", "true").lower() == "true"

Priority = Literal["min", "low", "default", "high", "urgent"]


def send_notification(
    title: str,
    message: str,
    priority: Priority = "default",
    tags: list[str] | None = None,
) -> bool:
    """
    Send a push notification via ntfy.sh.

    Args:
        title: Notification title
        message: Notification body
        priority: Priority level (min, low, default, high, urgent)
        tags: Optional emoji tags (e.g., ["warning", "rotating_light"])

    Returns:
        True if sent successfully, False otherwise
    """
    if not NOTIFICATIONS_ENABLED:
        log.debug("Notifications disabled, skipping")
        return True

    if not NTFY_TOPIC:
        log.warning("NTFY_TOPIC not configured, skipping notification")
        return False

    url = f"{NTFY_SERVER}/{NTFY_TOPIC}"

    headers = {
        "Title": title,
        "Priority": priority,
    }

    if tags:
        headers["Tags"] = ",".join(tags)

    try:
        response = requests.post(url, data=message, headers=headers, timeout=10)
        response.raise_for_status()
        log.info(f"Notification sent: {title}")
        return True
    except requests.RequestException as e:
        log.error(f"Failed to send notification: {e}")
        return False


def notify_failure(sport: str, error_message: str) -> bool:
    """
    Send a failure notification for a specific sport update.

    Args:
        sport: Sport name (NFL, NBA, CFB, CBB)
        error_message: Description of the failure

    Returns:
        True if sent successfully
    """
    title = f"Eagle Eye: {sport} Update Failed"
    message = f"{error_message}\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    return send_notification(
        title=title,
        message=message,
        priority="high",
        tags=["warning", "x"],
    )


def notify_partial_failure(failed_sports: list[str], succeeded_sports: list[str]) -> bool:
    """
    Send notification for partial update failure.

    Args:
        failed_sports: List of sports that failed
        succeeded_sports: List of sports that succeeded

    Returns:
        True if sent successfully
    """
    title = "Eagle Eye: Partial Update Failure"
    message = (
        f"Failed: {', '.join(failed_sports)}\n"
        f"Succeeded: {', '.join(succeeded_sports)}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    return send_notification(
        title=title,
        message=message,
        priority="high",
        tags=["warning"],
    )


def notify_success(message: str = "All sports updated successfully") -> bool:
    """
    Send a success notification (optional, disabled by default).

    Args:
        message: Success message

    Returns:
        True if sent successfully
    """
    # Only send success notifications if explicitly enabled
    if os.environ.get("NOTIFY_ON_SUCCESS", "false").lower() != "true":
        return True

    title = "Eagle Eye: Update Complete"

    return send_notification(
        title=title,
        message=f"{message}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        priority="low",
        tags=["white_check_mark"],
    )


def notify_daily_summary(results: dict[str, bool]) -> bool:
    """
    Send daily update summary notification.

    Args:
        results: Dict mapping sport name to success status

    Returns:
        True if sent successfully
    """
    failed = [sport for sport, success in results.items() if not success]
    succeeded = [sport for sport, success in results.items() if success]

    if failed:
        return notify_partial_failure(failed, succeeded)
    else:
        return notify_success(f"All {len(succeeded)} sports updated successfully")


if __name__ == "__main__":
    # Test notification
    print(f"NTFY_TOPIC: {NTFY_TOPIC}")
    print(f"NOTIFICATIONS_ENABLED: {NOTIFICATIONS_ENABLED}")

    if NOTIFICATIONS_ENABLED and NTFY_TOPIC:
        print("\nSending test notification...")
        success = send_notification(
            title="Eagle Eye Test",
            message="This is a test notification from Eagle Eye Sports Tracker",
            priority="low",
            tags=["test", "eagle"],
        )
        print(f"Result: {'Success' if success else 'Failed'}")
    else:
        print("\nNotifications not configured. Set NTFY_TOPIC environment variable.")
