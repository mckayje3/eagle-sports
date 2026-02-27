# NBA Feature Correlation Analysis

**Date:** 2025-12-28
**Data:** 3,326 games across 2024-2026 seasons
**Database:** nba_games.db

---

## Executive Summary

- **FG% differential is the strongest predictor** of game outcome (r = -0.79)
- Vegas lines predict winners 66.4% of the time, but offer no ATS edge
- All key correlations are stable across seasons (good for model generalization)
- Home court advantage is ~2 points (55% home win rate)

---

## 1. Correlation with Spread (Game Outcome)

*Positive correlation = away team advantage*

| Feature | Correlation | Visual |
|---------|-------------|--------|
| fg_pct_diff | -0.790 | ################################ |
| ast_diff | -0.620 | ######################### |
| 3p_pct_diff | -0.601 | ######################## |
| away_fg_pct | +0.566 | ####################### |
| home_fg_pct | -0.558 | ####################### |
| reb_diff | -0.499 | #################### |
| vegas_spread | +0.455 | ################## |
| away_3p_pct | +0.428 | ################# |
| home_ast | -0.419 | ################# |
| home_3p_pct | -0.411 | ################ |
| away_ast | +0.406 | ################ |
| paint_diff | -0.394 | ################ |
| fastbreak_diff | -0.385 | ############### |
| away_reb | +0.358 | ############## |
| home_reb | -0.352 | ############## |
| tov_diff | +0.307 | ############ |
| stl_diff | -0.301 | ############ |

---

## 2. Correlation with Total Points

| Feature | Correlation | Visual |
|---------|-------------|--------|
| combined_fg_pct | +0.649 | ########################## |
| combined_ast | +0.634 | ######################### |
| combined_paint | +0.500 | #################### |
| combined_3p_pct | +0.498 | #################### |
| away_ast | +0.482 | ################### |
| home_ast | +0.465 | ################### |
| away_fg_pct | +0.463 | ################### |
| home_fg_pct | +0.449 | ################## |
| vegas_total | +0.445 | ################## |
| away_paint | +0.385 | ############### |
| home_3p_pct | +0.367 | ############### |
| home_paint | +0.357 | ############## |
| away_3p_pct | +0.346 | ############## |
| combined_fastbreak | +0.246 | ########## |

---

## 3. Correlation with Home Win

| Feature | Correlation | Visual |
|---------|-------------|--------|
| fg_pct_diff | +0.642 | ########################## |
| 3p_pct_diff | +0.484 | ################### |
| ast_diff | +0.470 | ################### |
| home_fg_pct | +0.459 | ################## |
| away_fg_pct | -0.455 | ################## |
| vegas_spread | -0.407 | ################ |
| reb_diff | +0.400 | ################ |
| away_3p_pct | -0.339 | ############## |
| home_3p_pct | +0.337 | ############# |
| home_ast | +0.321 | ############# |
| away_ast | -0.304 | ############ |
| paint_diff | +0.294 | ############ |
| away_reb | -0.284 | ########### |
| home_reb | +0.284 | ########### |
| fastbreak_diff | +0.265 | ########### |
| blk_diff | +0.247 | ########## |

---

## 4. Vegas Lines Predictive Analysis

**Games analyzed:** 1,873 (56.3% of total)

### Spread Prediction
- Mean Absolute Error: **10.66 points**
- RMSE: 13.79 points
- Correlation with actual: 0.455
- ATS Accuracy: **47.5%** (no edge)

### Total Prediction
- Mean Absolute Error: **14.43 points**
- RMSE: 18.10 points
- Correlation with actual: 0.444
- Over hit rate: **50.5%** (no edge)

### Moneyline (Straight Up)
- Winner prediction accuracy: **66.4%**
- Favorite win rate: 66.4%

---

## 5. Feature Multicollinearity

**High correlations between features (potential redundancy):**

| Feature 1 | Feature 2 | Correlation | Recommendation |
|-----------|-----------|-------------|----------------|
| home_dreb | home_reb | 0.814 | Use total reb only |
| away_dreb | away_reb | 0.807 | Use total reb only |
| home_tov | away_stl | 0.793 | Same info, use one |
| home_stl | away_tov | 0.787 | Same info, use one |
| home_dreb | away_fg_pct | -0.652 | Defensive reb limits opponent |
| home_fg_pct | away_dreb | -0.636 | Offensive efficiency vs defense |
| away_oreb | away_reb | 0.596 | Use total reb only |
| home_oreb | home_reb | 0.587 | Use total reb only |
| home_fg_pct | home_3p_pct | 0.586 | Moderate overlap, keep both |
| home_fg_pct | home_ast | 0.566 | Assists create easy shots |

---

## 6. Opponent Stat Relationships

| Relationship | Correlation | Interpretation |
|--------------|-------------|----------------|
| Home steals vs Away turnovers | +0.787 | Same event, different POV |
| Away steals vs Home turnovers | +0.793 | Same event, different POV |
| Home blocks vs Away FG% | -0.345 | Blocks reduce opponent efficiency |
| Away blocks vs Home FG% | -0.309 | Blocks reduce opponent efficiency |
| Home OREB vs Away DREB | -0.066 | Weak relationship |
| Away OREB vs Home DREB | -0.072 | Weak relationship |
| Home assists vs Home FG% | +0.566 | Ball movement = better shots |
| Away assists vs Away FG% | +0.541 | Ball movement = better shots |

---

## 7. Correlation Stability Across Seasons

| Feature | 2024 | 2025 | 2026 | Std Dev | Stable? |
|---------|------|------|------|---------|---------|
| fg_pct_diff | -0.807 | -0.774 | -0.790 | 0.017 | YES |
| 3p_pct_diff | -0.621 | -0.608 | -0.525 | 0.052 | YES |
| ast_diff | -0.612 | -0.620 | -0.643 | 0.016 | YES |
| reb_diff | -0.523 | -0.486 | -0.471 | 0.027 | YES |
| home_fg_pct | -0.559 | -0.553 | -0.571 | 0.009 | YES |
| away_fg_pct | +0.601 | +0.541 | +0.553 | 0.032 | YES |

*Stable = standard deviation < 0.1*

---

## 8. Home Court Advantage

| Season | Home Win Rate | Avg Home Margin | Games |
|--------|---------------|-----------------|-------|
| 2024 | 55.3% | +2.5 pts | 1,393 |
| 2025 | 55.0% | +2.1 pts | 1,397 |
| 2026 | 55.0% | +1.8 pts | 536 |

*Slight declining trend in home margin*

---

## 9. Feature Tier Recommendations

### Tier 1 - Essential (|r| > 0.5)
- FG% differential
- Assist differential
- 3P% differential
- Rebound differential
- Vegas spread (if available)

### Tier 2 - Important (|r| > 0.3)
- Paint points differential
- Fastbreak points differential
- Turnover differential
- Steal differential
- Vegas total (for total prediction)

### Tier 3 - Avoid/Redundant
- Offensive + Defensive rebounds separately (use total only)
- Individual team stats without differential
- Opponent turnovers when you have steals

---

## 10. Important Caveats

1. **These are IN-GAME correlations** - They show what stats correlate with outcomes within the same game.

2. **For predictions, use HISTORICAL averages** - Pre-game features based on past performance will have lower correlations.

3. **Correlation ≠ Causation** - High FG% correlates with winning, but you can't predict FG% before the game.

4. **Vegas lines are efficient** - 66% winner accuracy is hard to beat, and ATS is essentially random.

---

## Next Steps

1. Analyze pre-game historical averages vs outcomes
2. Build rolling averages for team stats
3. Test predictive power of historical features
4. Compare model performance to Vegas baseline

---

## 11. Pre-Game Historical Features Analysis

**Key Question:** How well do historical averages predict outcomes?

### Pre-Game Features vs Spread

*Using 10-game rolling averages computed BEFORE each game*

| Feature | Correlation | Notes |
|---------|-------------|-------|
| vegas_spread | +0.455 | Best predictor (already pre-game) |
| hist_net_rating_diff | -0.373 | Best non-Vegas feature |
| expected_spread | +0.373 | (home_ppg + away_papg) - (away_ppg + home_papg) |
| hist_net_rating_home | -0.270 | Home team PPG - PAPG |
| hist_net_rating_away | +0.265 | Away team PPG - PAPG |
| hist_ppg_diff | -0.244 | Home PPG - Away PPG |
| hist_papg_diff | +0.241 | Home PAPG - Away PAPG |
| hist_fg_pct_diff | -0.221 | Historical FG% differential |
| hist_tov_diff | +0.207 | Turnover tendency |
| hist_3p_pct_diff | -0.156 | Historical 3P% differential |
| hist_reb_diff | -0.121 | Historical rebound differential |
| hist_ast_diff | -0.090 | Historical assist differential |

### Pre-Game Features vs Total

| Feature | Correlation | Notes |
|---------|-------------|-------|
| vegas_total | +0.445 | Best predictor |
| expected_total | +0.389 | Combined historical PPG/PAPG |
| combined_hist_papg | +0.332 | Both teams' points allowed |
| combined_hist_ppg | +0.319 | Both teams' PPG |
| home_hist_papg | +0.267 | Home team defense (inverted) |
| home_hist_ppg | +0.242 | Home team offense |
| away_hist_ppg | +0.242 | Away team offense |
| away_hist_papg | +0.233 | Away team defense (inverted) |

### Pre-Game Features vs Home Win

| Feature | Correlation | Notes |
|---------|-------------|-------|
| vegas_spread | -0.407 | Best predictor |
| hist_net_rating_diff | +0.313 | Net rating differential |
| home_hist_ppg | +0.139 | Home team scoring |
| away_hist_fg_pct | -0.135 | Away team efficiency |
| home_hist_papg | -0.131 | Home team defense |
| away_hist_ppg | -0.131 | Away team scoring |

---

## 12. In-Game vs Pre-Game Correlation Comparison

**Critical Insight:** Pre-game features are 3-4x weaker than in-game stats

| Feature | In-Game r | Pre-Game r | Drop |
|---------|-----------|------------|------|
| FG% differential | -0.790 | -0.221 | -0.57 (3.6x weaker) |
| Assist differential | -0.620 | -0.090 | -0.53 (6.9x weaker) |
| 3P% differential | -0.601 | -0.156 | -0.44 (3.9x weaker) |
| Rebound differential | -0.499 | -0.121 | -0.38 (4.1x weaker) |
| Vegas spread | +0.455 | +0.455 | 0.00 (unchanged) |
| Net Rating diff | N/A | -0.373 | Best pre-game stat |

**Why the drop?**
- In-game stats measure what actually happened
- Pre-game stats are historical averages with high variance
- Shooting % is highly variable game-to-game
- Vegas already incorporates most available information

---

## 13. Model Strategy Recommendations

### For Spread Prediction
1. **Use Vegas spread as baseline** (r = 0.455)
2. **Add Net Rating differential** (r = -0.373) - may add marginal value
3. **PPG/PAPG differentials** - secondary features
4. **Historical FG%** - small additional signal

### For Total Prediction
1. **Use Vegas total as baseline** (r = 0.445)
2. **Expected total from PPG/PAPG** (r = 0.389)
3. **Combined historical PPG** - proxy for pace

### Realistic Expectations
- **Don't expect to beat Vegas significantly**
- **Focus on edge cases** where model disagrees strongly
- **ATS is essentially random** (47.5%) - market is efficient
- **Value may exist in totals** (50.5% over rate)

### Feature Engineering Priorities
1. Net Rating (PPG - PAPG) is more predictive than raw PPG
2. Differentials are more predictive than individual team stats
3. 10-game rolling window is reasonable for recency
4. Consider home/away splits for more signal

---

## 14. Data Quality Notes

- **3,326 games** with complete box scores
- **1,873 games** (56.3%) with Vegas odds
- **3,184 games** with sufficient history for rolling averages
- All correlations stable across 3 seasons (2024-2026)

---

*Analysis generated by Claude Code on 2025-12-28*
*Updated with pre-game historical analysis*
