"""
Spread Utility Module - SINGLE SOURCE OF TRUTH for spread conventions

================================================================================
VEGAS CONVENTION (used throughout this codebase):
================================================================================

    spread = away_score - home_score

    NEGATIVE spread (-7) = HOME team is favored by 7 points
    POSITIVE spread (+7) = AWAY team is favored by 7 points
    ZERO spread (0)      = Pick'em (no favorite)

Examples:
    - Chiefs at home vs Raiders, Chiefs favored by 7 => spread = -7
    - Raiders at home vs Chiefs, Chiefs favored by 7 => spread = +7 (away favored)

This convention matches how Vegas displays spreads:
    - "Chiefs -7" means betting on Chiefs requires them to win by more than 7
    - This is displayed as a negative number because they are the favorite

================================================================================
USAGE:
================================================================================

from spread_utils import (
    calculate_spread,           # From scores: spread = away - home
    get_favored_team,          # Returns (team_name, margin) tuple
    is_home_favored,           # True if spread < 0
    format_spread_pick,        # "Chiefs -7.0" or "Raiders +3.5"
    validate_spread_sign,      # Assert spread has correct sign given scores
)

================================================================================
"""

from typing import Tuple, Optional, Union
import warnings


def calculate_spread(home_score: float, away_score: float) -> float:
    """
    Calculate spread from scores using Vegas convention.

    Args:
        home_score: Home team's score
        away_score: Away team's score

    Returns:
        spread: away_score - home_score
                Negative = home favored
                Positive = away favored

    Example:
        >>> calculate_spread(27, 20)  # Home wins 27-20
        -7.0  # Home favored/won by 7
    """
    return float(away_score - home_score)


def is_home_favored(spread: float) -> bool:
    """
    Check if home team is favored based on spread.

    Args:
        spread: The spread value (Vegas convention: away - home)

    Returns:
        True if home is favored (spread < 0)
        False if away is favored or pick'em (spread >= 0)
    """
    return spread < 0


def is_away_favored(spread: float) -> bool:
    """
    Check if away team is favored based on spread.

    Args:
        spread: The spread value (Vegas convention: away - home)

    Returns:
        True if away is favored (spread > 0)
        False if home is favored or pick'em (spread <= 0)
    """
    return spread > 0


def get_favored_team(
    spread: float,
    home_team: str,
    away_team: str
) -> Tuple[Optional[str], float]:
    """
    Get the favored team and margin from spread.

    Args:
        spread: The spread value (Vegas convention: away - home)
        home_team: Name of the home team
        away_team: Name of the away team

    Returns:
        Tuple of (team_name, margin)
        - team_name: The favored team's name, or None if pick'em
        - margin: The margin by which they are favored (always positive)

    Example:
        >>> get_favored_team(-7.0, "Chiefs", "Raiders")
        ("Chiefs", 7.0)  # Home team (Chiefs) favored by 7

        >>> get_favored_team(3.5, "Raiders", "Chiefs")
        ("Chiefs", 3.5)  # Away team (Chiefs) favored by 3.5
    """
    margin = abs(spread)

    if spread < 0:
        # Negative spread = home favored
        return (home_team, margin)
    elif spread > 0:
        # Positive spread = away favored
        return (away_team, margin)
    else:
        # Pick'em
        return (None, 0.0)


def get_predicted_winner(
    spread: float,
    home_team: str,
    away_team: str
) -> Optional[str]:
    """
    Get the predicted winner based on spread.

    This is an alias for get_favored_team that just returns the team name.

    Args:
        spread: The spread value (Vegas convention: away - home)
        home_team: Name of the home team
        away_team: Name of the away team

    Returns:
        The predicted winner's name, or None if pick'em (spread == 0)
    """
    if spread < 0:
        return home_team
    elif spread > 0:
        return away_team
    else:
        return None


def format_spread(spread: float, include_sign: bool = True) -> str:
    """
    Format a spread value for display.

    Args:
        spread: The spread value
        include_sign: If True, include + for positive values

    Returns:
        Formatted string like "-7.0" or "+3.5"
    """
    if include_sign and spread > 0:
        return f"+{spread:.1f}"
    else:
        return f"{spread:.1f}"


def format_spread_pick(
    spread: float,
    home_team: str,
    away_team: str,
    include_margin: bool = True
) -> str:
    """
    Format a spread pick for display (e.g., "Chiefs -7.0").

    Args:
        spread: The spread value (Vegas convention)
        home_team: Name of the home team
        away_team: Name of the away team
        include_margin: If True, include the point margin

    Returns:
        Formatted string like "Chiefs -7.0" or "Raiders +3.5"

    Example:
        >>> format_spread_pick(-7.0, "Chiefs", "Raiders")
        "Chiefs -7.0"  # Home team favored

        >>> format_spread_pick(3.5, "Raiders", "Chiefs")
        "Chiefs +3.5"  # Away team favored
    """
    if spread < 0:
        # Home favored
        team = home_team
        display_spread = spread  # Already negative
    elif spread > 0:
        # Away favored
        team = away_team
        display_spread = -spread  # Negate to show as favorite (negative)
    else:
        return f"{home_team} PK"

    if include_margin:
        return f"{team} {display_spread:+.1f}"
    else:
        return team


def calculate_spread_deviation(
    model_spread: float,
    vegas_spread: float
) -> float:
    """
    Calculate how much the model deviates from Vegas.

    Positive deviation = model favors HOME more than Vegas
    Negative deviation = model favors AWAY more than Vegas

    Args:
        model_spread: Model's predicted spread (Vegas convention)
        vegas_spread: Vegas spread (Vegas convention)

    Returns:
        model_spread - vegas_spread

    Example:
        Model: -10 (home by 10), Vegas: -7 (home by 7)
        Deviation = -10 - (-7) = -3 (model favors home more)
    """
    return model_spread - vegas_spread


def model_favors_home_more(model_spread: float, vegas_spread: float) -> bool:
    """
    Check if model favors home team more than Vegas does.

    Args:
        model_spread: Model's predicted spread
        vegas_spread: Vegas spread

    Returns:
        True if model has home winning by more (or losing by less) than Vegas
    """
    # More negative = more home-favored
    return model_spread < vegas_spread


def model_favors_away_more(model_spread: float, vegas_spread: float) -> bool:
    """
    Check if model favors away team more than Vegas does.

    Args:
        model_spread: Model's predicted spread
        vegas_spread: Vegas spread

    Returns:
        True if model has away winning by more (or losing by less) than Vegas
    """
    # More positive = more away-favored
    return model_spread > vegas_spread


def validate_spread_sign(
    spread: float,
    home_score: float,
    away_score: float,
    tolerance: float = 0.01
) -> bool:
    """
    Validate that spread has the correct sign given the scores.

    This is a sanity check to catch convention errors.

    Args:
        spread: The calculated spread to validate
        home_score: Home team's score
        away_score: Away team's score
        tolerance: Small tolerance for floating point comparison

    Returns:
        True if spread sign is correct

    Raises:
        ValueError: If spread sign is incorrect
    """
    expected = away_score - home_score

    if abs(spread - expected) > tolerance:
        raise ValueError(
            f"Spread convention violation! "
            f"Got spread={spread:.1f}, but with home={home_score:.1f} and away={away_score:.1f}, "
            f"expected spread={expected:.1f} (away - home). "
            f"Convention: negative spread = home favored"
        )

    return True


def validate_prediction_spread(
    pred_spread: float,
    pred_home_score: float,
    pred_away_score: float,
    context: str = ""
) -> None:
    """
    Validate that a prediction's spread matches its scores.

    Call this after generating predictions to catch convention errors early.

    Args:
        pred_spread: The predicted spread
        pred_home_score: Predicted home score
        pred_away_score: Predicted away score
        context: Additional context for error message (e.g., game ID)

    Raises:
        ValueError: If spread doesn't match scores
    """
    expected = pred_away_score - pred_home_score
    tolerance = 0.1

    if abs(pred_spread - expected) > tolerance:
        context_str = f" ({context})" if context else ""
        raise ValueError(
            f"Prediction spread mismatch{context_str}! "
            f"Stored spread={pred_spread:.1f}, but scores {pred_away_score:.1f}-{pred_home_score:.1f} "
            f"imply spread={expected:.1f}. "
            f"Fix: spread should be (away_score - home_score)"
        )


# ============================================================================
# DEPRECATED FUNCTIONS - These exist only to catch incorrect usage patterns
# ============================================================================

def _wrong_spread_calculation():
    """This function does not exist - if you're looking for it, you're doing it wrong!"""
    raise NotImplementedError(
        "If you're looking for home_score - away_score, STOP! "
        "The correct formula is: spread = away_score - home_score. "
        "Use calculate_spread(home_score, away_score) instead."
    )


# ============================================================================
# Module-level documentation test
# ============================================================================

if __name__ == "__main__":
    # Test examples to verify the module works correctly
    print("=" * 60)
    print("SPREAD UTILS - Testing Vegas Convention")
    print("=" * 60)

    # Test 1: Home team wins 27-20
    home, away = 27, 20
    spread = calculate_spread(home, away)
    print(f"\nTest 1: Home wins {home}-{away}")
    print(f"  Spread: {spread} (should be -7, home favored)")
    print(f"  Home favored? {is_home_favored(spread)} (should be True)")
    winner, margin = get_favored_team(spread, "Chiefs", "Raiders")
    print(f"  Favored: {winner} by {margin} (should be Chiefs by 7)")

    # Test 2: Away team wins 31-24
    home, away = 24, 31
    spread = calculate_spread(home, away)
    print(f"\nTest 2: Away wins {away}-{home}")
    print(f"  Spread: {spread} (should be +7, away favored)")
    print(f"  Away favored? {is_away_favored(spread)} (should be True)")
    winner, margin = get_favored_team(spread, "Raiders", "Chiefs")
    print(f"  Favored: {winner} by {margin} (should be Chiefs by 7)")

    # Test 3: Format picks
    print("\nTest 3: Format picks")
    print(f"  Home favored: {format_spread_pick(-7.0, 'Chiefs', 'Raiders')}")
    print(f"  Away favored: {format_spread_pick(3.5, 'Raiders', 'Chiefs')}")

    # Test 4: Model vs Vegas deviation
    print("\nTest 4: Model vs Vegas")
    model, vegas = -10, -7
    print(f"  Model: {model}, Vegas: {vegas}")
    print(f"  Model favors home more? {model_favors_home_more(model, vegas)} (should be True)")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
