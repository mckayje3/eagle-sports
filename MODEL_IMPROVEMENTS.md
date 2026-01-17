# Model Improvements: Applying Basketball Lessons to Other Sports

## Summary of Edge Analysis Results

### Basketball (NBA/CBB) - Working Well
| Model | 6+ pt Edge ATS | Status |
|-------|---------------|--------|
| NBA Deep Eagle (with fades) | 62.7% | Profitable |
| CBB Deep Eagle (with fades) | 71.9% | Profitable |
| NBA Enhanced Ridge | 51.4% | Not profitable |
| CBB Enhanced Ridge | 49.8% | Not profitable |

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

### Longer Term
- [ ] Unify all predictors with consistent adjustment framework
- [ ] Automated backtest framework for testing new adjustments

---

## Edge Threshold Recommendations

Based on backtesting, here are the recommended star ratings:

| Sport | Bet Type | 3-Star Threshold | Win/ATS % | Notes |
|-------|----------|------------------|-----------|-------|
| NBA | Spread | >= 5 pts | 56.1% ATS | Deep Eagle + fades |
| CBB | Spread | 2-4 pts | 62.9% ATS | Deep Eagle + fades (odd pattern) |
| **NFL** | Spread | **>= 5 pts** | **58.4%** ATS | **Deep Eagle + adjustments** |
| CFB | Spread | N/A | ~50% | Not profitable at any threshold |
| **NHL** | **MONEYLINE** | **Home dog +100-120** | **59.6% win** | **NOT puck line (juice issue)** |

**Profitable sports:** NBA, CBB, NFL, NHL
**Not profitable:** CFB

**IMPORTANT:** NHL uses MONEYLINE (not puck line/spread) because puck lines have lopsided juice (+200/-200), making the math completely different from -110 spreads.

---

*Last Updated: 2026-01-17*
