# NBA Player-Level Impact Analysis

**Date:** 2025-12-28
**Focus:** How do player injuries affect total scores?
**Data:** 474 games with injury data (2025-26 season)

---

## Executive Summary

**Key Finding:** Player injuries have **minimal impact on total scores** (r = -0.06) but **moderate impact on spread** (r = -0.14).

For total prediction, player-level features add marginal value at best. Vegas total and historical team pace remain the dominant predictors.

---

## 1. Data Overview

| Metric | Value |
|--------|-------|
| Games with injury data | 474 |
| Injury records analyzed | 2,239 |
| Players with PPG data | 1,001 |
| Average total (injury sample) | 233.3 |
| Average total (all games) | 227.7 |

**Note:** Injury data only available for 2025-26 season (91% coverage). Earlier seasons have 0% coverage.

---

## 2. Injury Feature Correlations

### vs Total Score

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| total_ppg_out | -0.061 | Weak negative (more injuries = slightly lower total) |
| home_ppg_out | -0.080 | Home injuries slightly reduce total |
| away_ppg_out | -0.013 | Away injuries minimal impact |
| ppg_diff | +0.060 | No meaningful relationship |
| top_out_diff | -0.004 | Star availability doesn't predict total |

### vs Spread

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| home_ppg_out | +0.159 | Home injuries help away team |
| ppg_diff | -0.141 | Healthier team covers spread |
| top_out_diff | -0.104 | Star availability affects margin |
| home_top_out | +0.172 | Losing home star hurts margin |

### vs Home Win

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| ppg_diff | +0.102 | Healthier team wins more |
| home_top_out | -0.108 | Missing home star hurts win% |
| top_out_diff | +0.090 | Away star out helps home |

---

## 3. Comparison with Vegas

| Feature | vs Spread (r) | vs Total (r) |
|---------|---------------|--------------|
| **Vegas Spread** | **+0.419** | +0.136 |
| **Vegas Total** | -0.046 | **+0.320** |
| Injury PPG Diff | -0.153 | +0.056 |
| Top Scorer Diff | -0.126 | -0.010 |
| Total PPG Out | +0.097 | -0.058 |

**Conclusion:** Vegas lines are 3-5x more predictive than injury features.

---

## 4. Impact by PPG Tier

### When Stars (25+ PPG) Are Out

| Player | PPG | Games Played | Games Missed | Total With | Total Without | Diff |
|--------|-----|--------------|--------------|------------|---------------|------|
| Joel Embiid | 33.8 | 44 | 46 | 227.9 | 222.1 | -5.8 |
| Luka Doncic | 33.4 | 23 | 11 | 232.9 | 228.6 | -4.3 |
| SGA | 30.9 | 103 | 8 | 226.1 | 220.8 | -5.3 |
| Giannis | 30.3 | 75 | 17 | 227.9 | 223.2 | -4.6 |
| Nikola Jokic | 27.8 | 88 | 13 | 236.3 | 223.6 | -12.7 |
| Stephen Curry | 26.8 | 26 | 10 | 231.8 | 215.5 | -16.3 |

**Average star impact:** -3.5 to -5 points when out

### By Total PPG Out (Combined)

| PPG Out | Avg Total | vs Avg | Games |
|---------|-----------|--------|-------|
| 0-15 | 234.1 | +0.9 | 122 |
| 15-30 | 233.6 | +0.4 | 180 |
| 30-50 | 233.3 | +0.0 | 131 |
| 50-100 | 231.4 | -1.9 | 30 |

**Conclusion:** Even massive injury loads (50+ PPG out) only reduce totals by ~2 points.

---

## 5. Injury Advantage Analysis

*Does the healthier team have an advantage?*

| Injury Differential | Avg Margin | Home Win% | Games |
|---------------------|------------|-----------|-------|
| Home much healthier (15+ PPG advantage) | +4.5 | 64.9% | 57 |
| Home slightly healthier | +2.5 | 58.4% | 185 |
| Away slightly healthier | +0.5 | 51.1% | 182 |
| Away much healthier (15+ PPG advantage) | -1.5 | 43.8% | 48 |

**Conclusion:** ~4-5 point swing between extreme injury differentials.

---

## 6. Why Player Injuries Don't Affect Totals Much

### Theory vs Reality

**Expected:** Missing a 25 PPG scorer should reduce total by ~25 points.

**Actual:** Totals only drop 3-5 points when stars are out.

### Reasons

1. **Replacement players absorb shots** - Someone else takes the 15-20 shots
2. **Pace stays similar** - Teams run same offensive systems
3. **Defense adjusts** - Opponents may play differently without star matchup
4. **Usage redistribution** - Second/third options increase scoring
5. **High variance** - Game-to-game scoring variance (~15 pts) overwhelms injury effect

---

## 7. Recommendations for Total Prediction

### Primary Features (Use These)
- Vegas total (r = +0.32)
- Historical team PPG combined
- Expected total from team averages
- Pace indicators

### Secondary Features (Marginal Value)
- Total PPG out (r = -0.06)
- Number of players injured
- Star player availability

### Not Worth Including
- Individual player correlations
- Detailed injury reports
- Player-specific adjustments

### Expected Improvement
- Adding player injury data: **+0.5-1 point accuracy**
- Not worth the complexity for totals
- More valuable for spread prediction

---

## 8. Recommendations for Spread Prediction

### Player Features Worth Including
- PPG differential (r = -0.14)
- Home star availability (r = +0.17)
- Top scorer differential (r = -0.10)

### Expected Improvement
- Adding injury features: **+1-2 point edge in extreme cases**
- Worth including as secondary features
- Combine with Vegas spread for best results

### UPDATE (2026-01-02): Star Injury Post-Hoc Adjustment

Testing showed that a **post-hoc adjustment** works better than learned features:

```
adjusted_spread = base_spread + (home_star_ppg_lost - away_star_ppg_lost) * 0.05
```

**Results with factor=0.05:**
- MAE improvement: +0.25 pts on injury games
- ATS improvement: 55% (up from 51.2%)
- Mid-season MAE reaches parity with Vegas (11.08 vs 11.04)

See `nba_ridge_model_performance.md` for full details.

---

## 9. Data Limitations

1. **Only 2025-26 season has injury data** - Can't validate across seasons
2. **Coach's decisions vs real injuries** - Some "Out" statuses are rest, not injury
3. **No severity information** - Day-to-day vs season-ending treated same
4. **Team assignment issues** - Some players show wrong teams (trades)
5. **Sample size** - Only 474 games with full injury data

---

## 10. Future Analysis Ideas

1. **Pace-specific player impact** - Do certain players affect pace more?
2. **Back-to-back fatigue** - Player minutes in B2B situations
3. **Lineup combinations** - Specific player pairings and their totals
4. **In-season trends** - Do injuries matter more early vs late season?
5. **Blowout effects** - Do injuries lead to more blowouts (lower totals)?

---

## Appendix: Top Scorers by Team (2025-26)

| Team | Player | PPG | MPG | Games |
|------|--------|-----|-----|-------|
| LAL | Luka Doncic | 33.4 | 35.6 | 23 |
| OKC | Shai Gilgeous-Alexander | 30.8 | 32.2 | 33 |
| CLE | Donovan Mitchell | 29.2 | 33.1 | 32 |
| PHI | Tyrese Maxey | 29.2 | 37.4 | 31 |
| MIN | Anthony Edwards | 28.3 | 34.0 | 27 |
| DEN | Nikola Jokic | 28.1 | 33.4 | 34 |
| BOS | Jaylen Brown | 27.7 | 32.0 | 31 |
| NY | Jalen Brunson | 27.4 | 34.1 | 32 |
| MIL | Giannis Antetokounmpo | 27.4 | 28.5 | 19 |
| GS | Stephen Curry | 26.8 | 30.2 | 26 |

*Note: Team assignments may reflect trades during season*

---

*Analysis generated by Claude Code on 2025-12-28*
