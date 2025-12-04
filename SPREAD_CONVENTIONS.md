# Spread Conventions Guide

This document explains how spreads are stored, calculated, and displayed in Eagle Eye Sports Tracker.

## Standard Betting Notation

In sports betting, spreads are always shown from the **favorite's perspective** with a **negative number**:
- **Favorite**: Shown with negative spread (they "give" points)
- **Underdog**: Shown with positive spread (they "receive" points)

### Examples:
- "Chiefs -7.0" = Chiefs are 7-point favorites (expected to win by 7)
- "Broncos +3.0" = Broncos are 3-point underdogs (expected to lose by 3)

## Internal Data Storage

### Predicted Spread (`predicted_spread`)
```
predicted_spread = predicted_home_score - predicted_away_score
```
- **Positive value** = Home team predicted to win (home is favorite)
- **Negative value** = Away team predicted to win (away is favorite)

### Vegas Spread (`vegas_spread`)
Stored as the **home team's spread**:
- **Negative value** = Home team is favorite
- **Positive value** = Away team is favorite

## Display Logic

When displaying spreads to users, convert internal values to standard betting notation:

### Our Predictions
```python
spread = row['predicted_spread']  # home_score - away_score

if spread > 0:
    # Home team is favorite - show as negative
    spread_text = f"{home_team} {-spread:.1f}"  # "Eagles -7.0"
else:
    # Away team is favorite - show as negative
    spread_text = f"{away_team} {spread:.1f}"   # "Vikings -3.0"
```

### Vegas Lines
```python
vegas_spread = row['vegas_spread']  # stored as home team spread

if vegas_spread < 0:
    # Home team is favorite
    vegas_text = f"{home_team} {vegas_spread:.1f}"   # "Eagles -7.0"
else:
    # Away team is favorite
    vegas_text = f"{away_team} {-vegas_spread:.1f}"  # "Vikings -3.0"
```

## Key Rules

1. **Favorites always get negative numbers** when displayed
2. **Internal storage**: `home_score - away_score` (positive = home wins)
3. **Vegas storage**: Home team spread (negative = home favorite)
4. **Never show double negatives** like "Team --7.0"
5. **Always show the favorite's name** with the spread, not the underdog

## Adding a New Sport

When adding predictions for a new sport (NHL, MLB, etc.):

1. **Calculate spread** as `home_score - away_score`
2. **Store in database** with positive = home team advantage
3. **Display using standard notation**: favorite with negative number
4. **Test with examples**:
   - If home team wins by 5: `spread = 5`, display as "{Home} -5.0"
   - If away team wins by 3: `spread = -3`, display as "{Away} -3.0"

## Spread Ranges

For confidence intervals, the spread range should be symmetric around the predicted spread:
```python
spread_low = predicted_spread - margin  # More favorable to away team
spread_high = predicted_spread + margin  # More favorable to home team
```

Display these using the same convention (positive = home advantage).

## Quick Reference

| Scenario | Internal Value | Display |
|----------|---------------|---------|
| Home favored by 7 | `+7.0` | "Home Team -7.0" |
| Away favored by 3 | `-3.0` | "Away Team -3.0" |
| Pick 'em | `0.0` | "Pick 'em" or "Home Team -0.0" |
| Vegas: Home -7 | `-7.0` | "Home Team -7.0" |
| Vegas: Away -3 | `+3.0` | "Away Team -3.0" |
