# Sports Prediction Models - Parameter Reference

## Overview

Walk-forward Ridge regression models with EWMA decay. Updated Jan 2026.

---

## NFL Model Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| DECAY | 0.96 | Recent games weighted heavily |
| MIN_GAMES | 2 | Minimum games before predictions |
| PREV_HALF_LIFE | 4.0 | Blend with previous season stats |

**2024 Playoff Performance:**
- Spreads: 7-4 (63.6% ATS), +23.6% ROI
- Totals: 5-7 (41.7%), -22.5% ROI

**Key Files:** `nfl_predictor.py`, `predict_nfl_playoffs.py`, `predict_nfl_totals.py`

---

## CFB Model Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| DECAY | 0.88 | More decay than NFL |
| MIN_GAMES | 2 | Minimum games before predictions |
| PREV_HALF_LIFE | 5.0 | Blend with previous season |

**Performance:**
- Simple model: 50.9% ATS overall, 52.0% conference games
- Enhanced model: 49.1% ATS (worse - don't use)
- Edge classifier BET_WITH @ 0.5: 55.8% WR, +7.2% ROI
- 2024 playoff final rounds: 1-2 (33.3% ATS)

**Key Insight:** Conference games more predictable (52% vs 47.5%)

**Neutral Site:** Set `neutral=True` in `extract_spread_features()` for bowl/playoff games.

**Key Files:** `cfb_predictor.py`, `cfb_simple_model.py`, `predict_cfb_playoffs.py`

---

## NBA Model Parameters

Deep Eagle neural network with post-prediction adjustments.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | 100 → 256 → 128 → 64 → heads | Deep Eagle NN |
| BIG_FAVORITE_THRESHOLD | 10.0 | Trigger big underdog adjustment |
| UNDERDOG_ADJUSTMENT | 1.5 | Points added toward 10+ underdogs |
| STRUGGLING_HOME_MARGIN | -3.0 | Avg margin threshold for struggling team |
| STRUGGLING_HOME_ADJUSTMENT | 1.5 | Points toward road fav vs struggling home |
| MIDDLE_SPREAD_EDGE | 2-6 pts | Range for spread fade |
| MIDDLE_TOTAL_EDGE | 4-6 pts | Range for total fade |

**Post-Prediction Adjustments (applied in order):**
1. Big underdog: +1.5 pts toward teams that are 10+ point underdogs
2. Struggling home: +1.5 pts toward road favorite when home team struggling
3. Fade middle-edge favorites: Flip 2-6pt edges toward Vegas favorite
4. Fade middle-edge totals: Flip 4-6pt edges in either direction

**Backtest Performance (644 games, 2026 season):**
- Overall: Spreads 53.3%, Totals 54.5%
- Fade adjustments added +41 games (spreads +23, totals +18)

| Edge Range | Spreads | Totals | Profitable? |
|------------|---------|--------|-------------|
| Small (<2/4 pts) | 52.0% | 52.1% | No |
| Middle (2-6/4-6 pts) | 51.1% | 58.8% | Totals yes |
| Large (6+ pts) | 62.7% | 55.0% | Yes |

**Star System (profitability-based):**
- ⭐⭐⭐ (>58%): Spread 6+ pts, Total 4-6 pts
- ⭐⭐ (53-58%): Total 6+ pts
- ⭐ (<53%): Everything else (below breakeven)

**Key Insight:** Middle-edge favorites (2-6 pts) were 35% ATS before fading. Flipping these to underdog bets yields 65% ATS.

**Key Files:** `nba_predictor.py`, `train_deep_eagle_nba.py`

---

## CBB Model Parameters

Deep Eagle neural network with post-prediction adjustments.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | 79 → 256 → 128 → 64 → heads | Deep Eagle NN |
| BIG_DOG_THRESHOLD | 10.0 | Vegas spread threshold for big underdog |
| BIG_DOG_ADJUSTMENT | 1.5 | Points to add toward big underdogs |
| FADE_EDGE_MIN | 6.0 | Lower bound for fade range |
| FADE_EDGE_MAX | 8.0 | Upper bound for fade range |
| CLOSE_GAME_THRESHOLD | 5.0 | Vegas spread threshold for "close game" |

**Post-Prediction Adjustments (applied in order):**
1. Big underdog: +1.5 pts toward teams getting 10+ points
2. Fade 6-8 pt edges in close games: When |edge| is 6-8 pts AND |vegas_spread| < 5, flip the bet

**Backtest Performance (12,447 games, 2024-2025):**
- Before adjustments: 55.8% ATS overall
- After adjustments: 56.4% ATS overall (+39 games)

| Edge Range | ATS % | Profitable? |
|------------|-------|-------------|
| < 2 pts | 50.8% | No |
| 2-4 pts | 52.0% | No |
| 4-6 pts | 50.9% | No |
| 6+ pts | 71.9% | Yes (very!) |

**Star System (profitability-based):**
- ⭐⭐⭐ (>58%): Spread 6+ pts edge only (71.9% after adjustments)
- ⭐ (<53%): Everything else (below breakeven)

**Totals:** ~50% overall, no profitable patterns - all 1 star

**Key Insights:**
- 6-8 pt edges in close games hit 45.8% before fade, 54.2% after
- 10+ pt edges hit 79.9% (extremely profitable)
- Model much stronger on non-conference games (61% vs 51.5%)

**Key Files:** `cbb_predictor.py`, `cbb_espn_scraper.py`, `train_deep_eagle_cbb.py`

---

## Spread Convention

`spread = away_score - home_score`

| Vegas Line | Meaning |
|------------|---------|
| -3.5 | Home favored by 3.5 |
| +3.5 | Away favored by 3.5 |

**Edge:** `model_spread - vegas_spread`

---

## Totals vs Spreads

Spread edges more reliable. Rule of thumb: 4pt spread edge ~ 6pt totals edge.

---

## Performance Summary (Jan 2026)

| Sport | Spread ATS | Totals | Notes |
|-------|------------|--------|-------|
| NFL | ~54% | ~50% | Strong playoff performer (63.6%) |
| CFB | 50.9% | N/A | Conf games better (52%) |
| NBA | 53.3% | 54.5% | With fade adjustments (+41 games) |
| CBB | 56.4% | ~50% | With adjustments (+39 games), 6+ edge = 71.9% |

---

## NBA Betting Guidelines

Based on backtest validation (644 games):

| Pick Type | Win % | Stars | Action |
|-----------|-------|-------|--------|
| Spread 6+ edge | 62.7% | ⭐⭐⭐ | Strong bet |
| Total 4-6 edge | 58.8% | ⭐⭐⭐ | Strong bet |
| Total 6+ edge | 55.0% | ⭐⭐ | Standard bet |
| Everything else | 51-52% | ⭐ | Skip or small |

**Breakeven at -110 vig: 52.4%**

**Avoid:** Spread picks <6 pts edge, Blowout games (12+ pt spreads at 47.1%)

---

## CBB Betting Guidelines

Based on backtest validation (12,447 games):

| Pick Type | Win % | Stars | Action |
|-----------|-------|-------|--------|
| Spread 6+ edge | 71.9% | ⭐⭐⭐ | Strong bet |
| Spread 10+ edge | 79.9% | ⭐⭐⭐ | Very strong bet |
| Everything else | 50-52% | ⭐ | Skip |
| Totals (all) | ~50% | ⭐ | Skip |

**Breakeven at -110 vig: 52.4%**

**Only bet:** Spread picks with 6+ pts edge (after adjustments applied)

**Bonus:** Non-conference games with 6+ edge hit 72.9%
