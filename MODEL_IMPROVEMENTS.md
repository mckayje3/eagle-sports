# Model Improvements: Applying Basketball Lessons to Other Sports

## Summary of Edge Analysis Results

### Basketball (NBA/CBB)
| Model | Bet Type | Threshold | Win % | Sample | Status |
|-------|----------|-----------|-------|--------|--------|
| **NBA Ridge V2 + Rules** | Spread | **5+ pts, 2+ stars** | **64.3% ATS** | 196 | **✅ BEST** |
| **NBA Ridge V2 Totals** | **Total** | **Fade UNDER 7+** | **59% O/U** | 134 | **✅ PROFITABLE** |
| NBA Ridge V2 (base) | Spread | 5+ pts | 59.0% ATS | 522 | Profitable |
| NBA Deep Eagle (with fades) | Spread | 6+ pts | 62.7% ATS | - | Profitable |
| NBA Enhanced Ridge | Spread | Any | 51.4% ATS | - | Not profitable |
| **CBB Enhanced Ridge** | Spread | **Any** | **~50% ATS** | **11,066** | **❌ NOT PROFITABLE** |

**Key Findings:**
- NBA Ridge V2 with rule-based filtering achieves 64.3% ATS - the best NBA result
- **NBA Totals: Fade UNDER 7+ = 59% win rate** (model has OVER bias)
- CBB does NOT beat Vegas at any threshold - previous 62.9% claim was small sample noise (70 games)

### Football (NFL/CFB) - Updated Analysis (2026-01-15)
| Model | 5+ pt Edge ATS | 7+ pt Edge ATS | Status |
|-------|---------------|----------------|--------|
| **NFL Deep Eagle MLP** | **58.4%** | **63.0%** | **✅ Profitable!** |
| NFL Ridge (fixed) | 53.8% | 57.1% | Marginal |
| CFB Deep Eagle MLP | 50.5% | ~50% | Not profitable |
| CFB Ridge (fixed) | 51.3% | ~52% | Breakeven |

**Key Finding:** NFL Deep Eagle significantly outperforms Ridge and is profitable at 5+ pt edges.

### Hockey (NHL) - Updated Analysis (2026-01-17) ✅ MONEYLINE STRATEGY

**IMPORTANT CORRECTION:** Puck lines (±1.5) have lopsided juice (+200/-200 range), NOT -110 like NFL/NBA spreads. Our previous puck line analysis was misleading because we didn't account for the actual odds.

**MONEYLINE Strategy (707 games with moneylines):**
| Strategy | Win Rate | ROI | Sample | Significance |
|----------|----------|-----|--------|--------------|
| **Home Underdogs (all)** | **52.0%** | **+17.5%** | 223 | **p=0.009** ✅ |
| Home dogs +100-120 | **59.6%** | **+26.1%** | 94 | Sweet spot! |
| Home dogs +121-180 | ~55% | +20%+ | 43 | Good range |
| Away underdogs | 39.5% | -7.2% | 484 | **AVOID** |

**Key Findings:**
- HOME UNDERDOGS win 52% of games outright (statistically significant, p=0.009)
- Best range: +100 to +120 moneyline (59.6% win rate)
- Away underdogs underperform significantly (39.5% win rate)
- Puck line analysis was misleading due to lopsided juice

**Strategy: BET HOME UNDERDOGS ON THE MONEYLINE**
- 3-star plays: Home dog +100 to +120 (59.6% win rate, 26.1% ROI)
- 2-star plays: Home dog +121 to +180 (~55% win rate)

---

## Key Lessons from Basketball Models

### 1. Deep Eagle > Ridge for Finding Edges
Ridge models are more accurate overall (lower MAE) but have NO profitable ATS thresholds. Deep Eagle models are less accurate overall but identify HIGH-VALUE spots:
- Ridge hits ~50% ATS at all edge thresholds
- Deep Eagle hits 62-72% ATS at 6+ point edges

**Why?** Ridge is constrained to linear relationships. Deep Eagle can learn complex patterns like:
- Momentum/streaks effects
- Bounce-back after big losses
- Team-specific adjustments
- Situational factors (rest, travel, etc.)

### 2. Post-Prediction Adjustments Are Critical
The basketball models apply these adjustments AFTER the Deep Eagle prediction:

**NBA Adjustments:**
```
1. Big Underdog (+1.5 pts): When team is 10+ point dog
2. Struggling Home (+1.5 pts): Road fav vs struggling home (margin < -3)
3. Fade Middle-Edge Favorites: 2-6pt edges toward favorite → flip to dog
4. Fade Middle-Edge Totals: 4-6pt edges → flip direction
```

**CBB Adjustments:**
```
1. Big Underdog (+1.5 pts): When team is 10+ point dog
2. Fade Close Game Edges: 6-8pt edges in close games (Vegas < 5) → flip
```

These adjustments corrected known model biases found through backtesting.

### 3. Sport-Specific Thresholds Matter
- NBA: Only 6+ pt edges are profitable (3-star plays)
- CBB: Only 6+ pt edges are profitable (3-star plays)
- NFL/CFB: Ridge shows no profitable threshold at any level

---

## Recommended Improvements by Sport

### NFL (✅ COMPLETED - 2026-01-15)

**Problem:** Ridge model was 45% ATS even at high edges (worse than random).

**Solution Applied:**

1. ✅ **Fixed Ridge model feature signs** - Now 53.8% at 5+ pt edges
2. ✅ **Ran Deep Eagle MLP edge analysis** - 58.4% at 5+ pt edges, 63.0% at 7+ pt edges
3. ✅ **Added post-prediction adjustments to NFL predictor:**
   ```python
   # NFL adjustments implemented in nfl_predictor.py:
   BIG_UNDERDOG_THRESHOLD = 7.0  # Teams that are 7+ point underdogs
   UNDERDOG_ADJUSTMENT = 1.0     # Points added toward underdog

   REST_ADVANTAGE_THRESHOLD = 3  # Days of rest advantage
   REST_ADJUSTMENT = 0.5         # Points for significant rest advantage

   POST_BYE_ADJUSTMENT = 1.0     # Points for team coming off bye
   ```
4. ✅ **Updated star ratings** - 5+ pt edges = 3 stars (profitable)
5. ✅ **Created `nfl_deep_eagle_edge_analysis.py`** for future backtesting

**Result:** NFL is now profitable at 5+ pt edges (58.4% ATS). Use 3-star plays only.

### CFB (Medium Priority)

**Problem:**
- Ridge model is 52% ATS (breakeven, not profitable at -110 vig)
- Model severely struggles with neutral site games
- Model can't handle outlier dominant teams (e.g., Indiana 2025)

**Recommended Changes:**

1. **Fix Neutral Site Handling**
   - Only 7.8% of games are neutral site (bowl games, playoffs)
   - Model trained mostly on home games doesn't transfer well
   - **Solution:** Train separate model or use different HCA weight for neutral sites

2. **Add Team-Tier Awareness**
   - CFB has massive skill gaps (unlike NFL parity)
   - Model struggles when dominant team faces weak opponent
   - **Solution:** Add feature for team tier (CFP teams, Power 5, Group of 5, FCS)

3. **Use Deep Eagle with Adjustments**
   ```python
   # Suggested CFB adjustments to backtest:
   BIG_UNDERDOG_THRESHOLD = 14.0  # Higher due to larger spreads
   UNDERDOG_ADJUSTMENT = 2.0      # Larger adjustment for CFB

   # Neutral site adjustment
   NEUTRAL_SITE_HCA_ZERO = True   # Set HCA to 0 for bowl games

   # Conference game adjustment
   CONFERENCE_GAME_ADJUSTMENT = 1.0  # Conference games more predictable
   ```

4. **Bowl Game Specific Model**
   - Teams have 3-4 weeks to prepare
   - Motivation varies (opt-outs, transfers)
   - Consider separate bowl game analysis

### NBA (✅ COMPLETED - 2026-01-21)

**Problem:** Enhanced Ridge and Deep Eagle models were complex but only ~51-56% ATS.

**Solution Applied - Ridge V2 + Rule-Based Confidence:**

1. ✅ **Built Ridge V2** - Pure model (no Vegas blend) with:
   - SRS opponent-adjusted ratings
   - Reduced HCA (1.5 pts vs traditional 2.2)
   - Road favorite penalty (+1.5 pts)
   - Dampened form features (30% weight)

2. ✅ **Discovered road fav fade** - Model picks road favorites at 35% ATS
   - When faded (bet home dog instead): **65% ATS**

3. ✅ **Rule-based confidence scoring:**
   ```python
   # Requires 5+ pt edge to qualify, then:
   - FADE road fav picks (35% ATS -> 65% when faded)
   - +1 for home picks (62.5% vs 55.4%)
   - +1 for home favorites (67.2% ATS)
   - +1 for close games (Vegas < 4 pts)
   - -1 for blowout spreads (Vegas 10+ pts)
   - +1 for big edges (7+ pts)
   # 2+ score = 64.3% ATS (126/196 games)
   ```

4. ✅ **Updated dashboard** with rule-based stars and FADE indicator
5. ✅ **Updated all documentation**

**Walk-Forward Backtest Results (1615 games, 2025-2026 combined):**
| Filter | Record | ATS % | Notes |
|--------|--------|-------|-------|
| Base 5+ pt edges | 308/522 | 59.0% | Good baseline |
| **2+ stars (rule filter)** | **126/196** | **64.3%** | **BEST** |
| 3+ stars | 41/65 | 63.1% | Also strong |
| Totals (any) | ~50% | - | Not profitable |

**Key Insight:** Simple rules outperform complex neural nets. The road fav fade alone is worth ~30% of the edge improvement.

### NHL (✅ CONFIRMED PROFITABLE - 2026-01-17)

**IMPORTANT CORRECTION:** Puck lines have lopsided juice (+200/-200), NOT -110 like NFL/NBA. Use MONEYLINES instead.

**Solution Applied:**

1. ✅ **Backfilled moneyline data** - 707 games with moneylines from 2023-2025
2. ✅ **Discovered profitable strategy** - HOME UNDERDOGS on moneyline
3. ✅ **Updated dashboard** - Shows moneyline plays, not puck line picks
4. ✅ **Updated star ratings** - Based on moneyline range, not spread edge

**Moneyline Backtest Results (707 games - statistically significant):**
- **Home underdogs: 52.0% win rate, +17.5% ROI (p=0.009)**
- Best range +100-120: 59.6% win rate, +26.1% ROI
- Away underdogs: 39.5% win rate, -7.2% ROI (AVOID)

---

## Action Items

### ✅ Completed (2026-01-15)
- [x] Create `nfl_deep_eagle_edge_analysis.py` to test Deep Eagle thresholds
- [x] Add post-prediction adjustments to NFL predictor
- [x] Update NFL star ratings to 5+ pts = 3 stars
- [x] Create `cfb_deep_eagle_edge_analysis.py` for testing
- [x] Create `nhl_deep_eagle_edge_analysis.py` for testing
- [x] Add underdog adjustment to NHL predictor
- [x] Update NHL star ratings to 1+ goal edge = 3 stars

### Remaining (CFB)
- [ ] Fix neutral site handling in CFB model
- [ ] Backtest CFB adjustments similar to basketball
- [ ] Train CFB model with team-tier features
- [ ] Create bowl-game specific predictions

### ✅ Completed (NHL - 2026-01-17)
- [x] Backfill NHL historical odds data with MONEYLINES (707 games)
- [x] Discovered puck line has lopsided juice - use moneyline instead
- [x] Validated HOME UNDERDOG moneyline strategy (52% win, +17.5% ROI, p=0.009)
- [x] Updated dashboard to show moneyline plays
- [x] Updated all documentation

### ✅ Completed (NBA Ridge V2 - 2026-01-21)
- [x] Built Ridge V2 model (pure model, no Vegas blend, SRS ratings, reduced HCA)
- [x] Discovered road fav fade (35% ATS -> 65% when faded)
- [x] Implemented rule-based confidence scoring (64.3% ATS at 2+ stars)
- [x] Walk-forward validated on 1615 games (2025-2026 combined)
- [x] Updated `nba_predictor.py` with `calculate_spread_confidence()` method
- [x] Updated `streamlit_app.py` with FADE indicator and rule-based stars
- [x] Updated all documentation (CLAUDE.md, MODEL_IMPROVEMENTS.md)

### ✅ Completed (CBB Analysis - 2026-01-22)
- [x] Created `cbb_full_walkforward.py` for true online learning backtest
- [x] Ran full walk-forward on 11,066 games (2024-2026)
- [x] **CONCLUSION: CBB is NOT profitable at any threshold**
- [x] Previous 62.9% claim (70 games) was small sample noise - true rate is ~49%
- [x] No rule-based filters improve results (unlike NBA)
- [x] Updated documentation to reflect CBB as "entertainment only"

### ✅ Completed (NBA Totals Fade - 2026-01-22)
- [x] Created `nba_totals_analysis.py` for totals walk-forward analysis
- [x] Ran full walk-forward on 1615 games (2025-2026 combined)
- [x] **DISCOVERY: Fade UNDER 7+ = 59% win rate (79-55 record)**
- [x] Model has +1.01 OVER bias, so when it predicts UNDER strongly, bet OVER
- [x] Added `calculate_total_confidence()` method to `nba_predictor.py`
- [x] Updated `_apply_total_adjustments()` with fade UNDER 7+ logic
- [x] Updated `streamlit_app.py` with total fade indicator and 2-star rating
- [x] Updated all documentation (CLAUDE.md, MODEL_IMPROVEMENTS.md)

### Longer Term
- [ ] Unify all predictors with consistent adjustment framework
- [ ] Automated backtest framework for testing new adjustments

### 2026 Offseason (NFL/CFB Model Optimization)
Priority project for Feb-Aug 2026 before next season starts:

**NFL Enhanced Ridge:**
- Current spreads: ~53% ATS (marginal)
- Current totals: 50.3% O/U (not useful)
- UNDER only 41.7% - model has OVER bias that doesn't work
- Needs: Feature analysis, drive efficiency tuning, weather/dome factors
- Challenge: Fewer games per season, need multi-year training

**CFB Enhanced Ridge:**
- Current spreads: ~50-52% ATS (breakeven)
- Current totals: 52.7% O/U overall, but **UNDER at 5+ pts = 56.7%**
- Model overpredicts scoring (same pattern as NFL Deep Eagle)
- Needs: Neutral site handling, team tiers, conference awareness
- Potential: UNDER bets at high edges may be actionable

**Approach:**
1. Feature correlation analysis (like basketball models)
2. Walk-forward backtesting at multiple thresholds
3. Discover adjustment patterns (underdog, situational)
4. Tune decay rates, HCA, form windows

---

## Enhanced Ridge Totals Analysis (2026-01-18)

### NFL Ridge Totals - FADE UNDER STRATEGY IMPLEMENTED
| Threshold | Record | Win % | Notes |
|-----------|--------|-------|-------|
| Overall | 184/366 | 50.3% | Random |
| OVER | 134/246 | 54.5% | Slight edge |
| UNDER | 50/120 | 41.7% | **Terrible - FADE IT** |

**Fade UNDER Strategy (bet OVER when model says UNDER):**
| Threshold | Record | Win % | ROI |
|-----------|--------|-------|-----|
| All UNDER | 70/120 | 58.3% | +11.2% |
| UNDER 4+ | 22/39 | 56.4% | +7.7% |
| **UNDER 5+** | **13/21** | **61.9%** | **+18.1%** |
| UNDER 6+ | 9/15 | 60.0% | +14.5% |

**Implementation (2026-01-18):** Added `_apply_total_adjustments()` to NFL predictor.
When Ridge model predicts UNDER by 5+ points, automatically flips to OVER (+10 pts).
Stored in `pred_total` (adjusted) vs `pred_total_base` (raw model).

### CFB Ridge Totals - UNDER Shows Promise
| Threshold | Record | Win % | Notes |
|-----------|--------|-------|-------|
| Overall | 746/1416 | 52.7% | Marginal |
| OVER | 486/891 | 54.5% | Slight edge |
| UNDER overall | 260/525 | 49.5% | But improves... |
| **UNDER 5+ pts** | **93/164** | **56.7%** | **Promising** |
| **UNDER 6+ pts** | **67/119** | **56.3%** | **Good** |
| **UNDER 7+ pts** | **50/87** | **57.5%** | **Profitable!** |

**Season trend:** 45% (2022) → 51% (2023) → 55% (2024) → 54% (2025) - improving!

**Conclusion:** CFB Ridge (like NFL Deep Eagle) overpredicts scoring. UNDER bets at 5+ pt edges may be actionable. Monitor in offseason optimization.

---

## Edge Threshold Recommendations

Based on backtesting, here are the recommended star ratings:

| Sport | Bet Type | 3-Star Threshold | Win/ATS % | Sample | Notes |
|-------|----------|------------------|-----------|--------|-------|
| **NBA** | Spread | **5+ pts, 2+ stars** | **64.3% ATS** | 196 | **Ridge V2 + rule filter** |
| **NFL** | Spread | **>= 5 pts** | **58.4% ATS** | - | **Deep Eagle + adjustments** |
| **NHL** | **MONEYLINE** | **Home dog +100-120** | **59.6% win** | 94 | **NOT puck line (juice issue)** |
| CFB | Spread | N/A | ~50% | - | Not profitable at any threshold |
| CBB | Spread | N/A | ~50% | 11,066 | **NOT profitable - entertainment only** |

**Profitable sports:** NBA, NFL, NHL
**Not profitable:** CFB, CBB

**IMPORTANT:** NHL uses MONEYLINE (not puck line/spread) because puck lines have lopsided juice (+200/-200), making the math completely different from -110 spreads.

---

*Last Updated: 2026-01-22 (CBB analysis added)*
