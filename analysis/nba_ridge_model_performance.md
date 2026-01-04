# NBA Enhanced Ridge Model Performance Report

**Date:** 2026-01-04 (updated)
**Model:** Enhanced Ridge with Dynamic HCA and Star Injury Adjustment
**Code:** `nba_enhanced_ridge.py`

> **See also:** [NBA Simple Model](nba_simple_model.md) for the baseline 12-feature model without injuries or dynamic HCA.

---

## Executive Summary

The Enhanced model uses 17 features including dynamic per-team HCA, DNP-based star injury adjustments, and momentum indicators. Despite the added complexity, it currently performs **on par with the simpler 12-feature model** (MAE: 11.08 vs 11.09).

**Current Status:** The extra features haven't yet translated to measurable improvement. The hypothesis is that value may emerge in specific scenarios (injury games, mid-season, etc.) that warrant continued development.

---

## 1. Model Architecture

### Features (17)
| # | Feature | Coefficient | Notes |
|---|---------|-------------|-------|
| 0 | PPG diff | -1.59 | |
| 1 | PAPG diff | +2.04 | Defense matters more |
| 2 | Net rating diff | -2.99 | **Strongest** |
| 3 | Recent form (L5) | +1.02 | Last 5 games margin |
| 4 | Momentum | -0.01 | Weak signal |
| 5 | Streak diff | +0.08 | Weak signal |
| 6 | Rest diff | -0.49 | |
| 7 | Home B2B | +0.78 | Fatigue penalty |
| 8 | Away B2B | -0.88 | Fatigue penalty |
| 9 | Dynamic HCA | -0.67 | Per-team, blended |
| 10 | Home reliability | +0.38 | Games played proxy |
| 11 | Away reliability | -0.25 | |
| 12 | Injury adj | -0.59 | DNP-based stars |
| 13 | Season progress | +0.35 | |
| 14 | FG% diff | -0.53 | |
| 15 | Reb diff | -0.32 | |
| 16 | TOV diff | +0.73 | |

### Star Injury Adjustment (DNP-Based)
- **Data Source:** `player_game_stats.did_not_play` (available all seasons)
- **Factor:** 0.05 per star PPG lost
- **Importance Threshold:** 0.35 (top ~3 players per team)
- **Importance Formula:** 40% minutes share + 30% points share + 15% plus/minus + 15% starter rate
- **Non-Injury Exclusions:** COACH'S DECISION, NOT WITH TEAM, REST, G LEAGUE, PERSONAL
- **Coefficient:** -0.591 (9th strongest feature)

---

## 2. Performance by Season Segment (2026 Season)

| Segment | Games | Model MAE | Vegas MAE | Difference | Model ATS |
|---------|-------|-----------|-----------|------------|-----------|
| **Early** (0-300) | 293 | 10.89 | 10.19 | +0.71 | 46.1% |
| **Mid** (300-800) | 263 | 11.08 | 11.04 | **+0.04** | **56.3%** |
| Late (800+) | - | - | - | - | - |
| **OVERALL** | 557 | 10.97 | 10.59 | +0.38 | 51.0% |

### Key Findings

1. **Mid Season is strongest** - Model reaches Vegas-level MAE (11.08 vs 11.04)
2. **Early season struggles** - Less historical data affects accuracy
3. **ATS peaks mid-season** - 56.3% is profitable territory

---

## 3. Star Injury Adjustment Impact

### Testing Results (factor=0.05)
- **MAE improvement:** +0.25 points on injury games
- **ATS improvement:** 55% (up from 51.2%)
- **Games affected:** ~170 per season (15% of games)

### Factor Comparison
| Factor | MAE Change | ATS |
|--------|------------|-----|
| 0.00 (no adjustment) | baseline | 51.2% |
| 0.05 (optimal for MAE) | **+0.25** | 55.0% |
| 0.10 | -0.05 | 57.0% |
| 0.12 | -0.15 | **60.0%** |
| 0.15 | -0.30 | 58.0% |

**Trade-off:** Higher factors improve ATS but hurt overall MAE. Factor=0.05 optimizes MAE while providing solid ATS improvement.

---

## 4. Historical Context

### Vegas MAE Comparison
| Season | Model MAE | Vegas MAE | Gap |
|--------|-----------|-----------|-----|
| 2025 | 11.02 | 10.71 | +0.32 |
| 2026 (YTD) | 10.97 | 10.59 | +0.38 |
| 2026 Mid Only | 11.08 | 11.04 | +0.04 |

### ATS Performance History
| Season | Model ATS | Vegas Efficiency |
|--------|-----------|------------------|
| 2025 | 50.6% | 47.5% (essentially random) |
| 2026 (YTD) | 51.0% | - |
| 2026 Mid Only | **56.3%** | - |

---

## 5. Feature Analysis

### Strong Signals (|coef| > 0.7)
| Feature | Coefficient | Notes |
|---------|-------------|-------|
| Net rating diff | -2.99 | By far the strongest |
| PAPG diff | +2.04 | Defense predicts better |
| PPG diff | -1.59 | |
| Recent form (L5) | +1.02 | Last 5 games margin |
| Away B2B | -0.88 | |
| Home B2B | +0.78 | |
| TOV diff | +0.73 | |

### Weak/Negligible Signals (|coef| < 0.1)
| Feature | Coefficient | Notes |
|---------|-------------|-------|
| Momentum | -0.01 | Essentially zero |
| Streak diff | +0.08 | Minimal impact |

**Observation:** Momentum and streak features add no value. Consider removing in future versions.

---

## 6. Injury Feature Correlations

From earlier analysis (`nba_player_impact_analysis.md`):

| Feature | vs Spread (r) | vs Total (r) |
|---------|---------------|--------------|
| Vegas Spread | +0.42 | +0.14 |
| PPG Diff (injury) | -0.14 | +0.06 |
| Home Star Out | +0.17 | -0.08 |
| Top Scorer Diff | -0.10 | -0.01 |

**Key Insight:** Injury features are weak (r=0.14-0.17) compared to Vegas (r=0.42), but the star injury adjustment adds marginal value without hurting MAE.

---

## 7. Implementation Details

### Constants
```python
STAR_INJURY_FACTOR = 0.05        # Optimal for MAE
STAR_IMPORTANCE_THRESHOLD = 0.35 # Top ~3 players per team

# Non-injury DNP reasons (filtered out)
NON_INJURY_REASONS = ["COACH'S DECISION", "NOT WITH TEAM", "REST",
                      "G LEAGUE - TWO-WAY", "G LEAGUE", "PERSONAL"]
```

### Key Methods
- `load_player_rankings()` - Calculates importance scores (mins/pts/pm/starter)
- `get_injury_adjustment(team_id, game_id)` - Uses DNP data from player_game_stats
- `predict()` - Returns spread with injury adjustment baked in

### Output Columns
- `pred_spread` - Final prediction (base + adjustment)
- `pred_spread_base` - Base model prediction
- `star_injury_adj` - Star injury adjustment applied

---

## 8. Usage

### Training
```bash
python nba_ridge_model.py
```

### Predictions Output
- Saved to: `nba_current_predictions.csv`
- Includes: `pred_spread`, `pred_spread_base`, `star_injury_adj`, `pred_total`

### Model Artifacts
- Model: `models/nba_ridge_enhanced.pkl`
- Contains: Ridge models, scalers, team stats, player stats

---

## 9. Dynamic Per-Team HCA (Added 2026-01-03)

The model now tracks home court advantage dynamically throughout the season.

### How It Works
```python
# Track home and away margins separately with decay (0.97)
home_margin_wavg = weighted_avg(home_margins, decay_weights)
away_margin_wavg = weighted_avg(away_margins, decay_weights)

# Current season HCA (raw)
current_hca = home_margin_wavg - away_margin_wavg

# Blend with previous season based on sample size
blend_weight = min(total_games / 40, 1.0)
dynamic_hca = blend_weight * current + (1 - blend_weight) * prev_season
```

### Blending Schedule
| Games Played | Current Season | Previous Season |
|--------------|----------------|-----------------|
| 0-5 | 0% | 100% |
| 10 | 25% | 75% |
| 20 | 50% | 50% |
| 40+ | 100% | 0% |

### 2026 Season HCA Distribution
- Mean: 1.41 pts
- Std: 0.69 pts
- Range: -0.49 to 2.79 pts

### Why Dynamic HCA?
- Captures teams whose home advantage is changing mid-season
- New arena energy, roster changes, injury impacts
- Uses same decay as PPG/PAPG for consistency

---

## 10. Future Improvements

### Potential Enhancements
1. **Segment-specific models** - Train separate models for early/mid/late season
2. **Dynamic injury factor** - Adjust factor based on spread size
3. **Multiple injury tiers** - Different factors for #1 vs #2 vs #3 stars
4. **Recent injury trends** - Track if player just returned vs long-term out

### Not Recommended
- Higher injury factors (hurts MAE)
- Complex injury features as learned parameters (Ridge doesn't learn them well)
- Total-specific injury adjustments (minimal correlation)

---

## 11. Comparison with Other Approaches

| Approach | Spread MAE | Notes |
|----------|------------|-------|
| Vegas baseline | 10.59 | Market efficiency |
| Ridge (no injury) | 11.25 | Base model |
| Ridge (importance features) | 11.02 | Learned injury features |
| **Ridge (star adjustment)** | **10.97** | Post-hoc 0.05 factor |

---

## 12. Edge Classifier Integration

The neural network edge classifier has been updated to include star injury features.

### New Features Added
- `home_star_ppg_out` - PPG of home stars out
- `away_star_ppg_out` - PPG of away stars out
- `star_injury_adjustment` - The 0.05 factor adjustment
- `has_star_injury` - Binary flag
- `star_ppg_diff` - Differential between teams
- `total_star_ppg_out` - Combined PPG out

### Star Injury Edge Pattern
| Scenario | Games | With Model WR |
|----------|-------|---------------|
| **With star injuries** | 516 | **50.4%** |
| No star injuries | 3,337 | 47.1% |
| Home star out (15+ PPG) | 272 | 51.5% |
| Away star out (15+ PPG) | 266 | 48.5% |

### Best Betting Strategies (2026 Test)

**Spread:**
| Threshold | Action | Record | Win% | ROI |
|-----------|--------|--------|------|-----|
| >= 0.50 | BET_WITH | 127-106 | 54.5% | +4.1% |
| >= 0.55 | BET_WITH | 84-65 | 56.4% | +7.6% |
| **>= 0.60** | **BET_WITH** | **45-29** | **60.8%** | **+16.1%** |

**Total:**
| Threshold | Action | Record | Win% | ROI |
|-----------|--------|--------|------|-----|
| >= 0.60 | BET_WITH | 45-34 | 57.0% | +8.7% |
| **>= 0.70** | **BET_WITH** | **18-11** | **62.1%** | **+18.5%** |

### Other Notable Patterns
- Close/small favorite + Model likes away (+5): **55.1%** WR
- Big spread edge (7+) + away bias: **53.8%** WR
- Mid season has best ATS: **49.3%** (vs 46.6% early, 45.8% late)

---

## 13. Model Comparison: Simple vs Enhanced (Fair Comparison)

Test set: 867 games (2025 season with Vegas lines)

| Aspect | Simple Model | Enhanced Model |
|--------|--------------|----------------|
| **Spread Features** | 12 | 17 |
| **Total Features** | 6 | 15 |
| **HCA** | Flat (intercept) | Dynamic per-team |
| **Injuries** | None | DNP-based star adjustment |
| **Box scores** | FG%, Reb, TOV | FG%, Reb, TOV |
| **Decay** | 0.97 | 0.93 |
| **Spread MAE** | 11.05 | 11.09 |
| **Total MAE** | 14.78 | 14.86 |
| **Winner Acc** | 67.1% | 66.7% |
| **Vegas Spread MAE** | 10.75 | 10.75 |
| **Vegas Total MAE** | 14.25 | 14.25 |

### Enhanced Total Features (15)

| # | Feature | Description |
|---|---------|-------------|
| 0 | Combined PPG | Home + Away PPG |
| 1 | Combined PAPG | Home + Away PAPG |
| 2 | Home pace proxy | (PPG + PAPG) / 2 |
| 3 | Away pace proxy | (PPG + PAPG) / 2 |
| 4 | Home B2B | Back-to-back indicator |
| 5 | Away B2B | Back-to-back indicator |
| 6 | Home reliability | Games played / 30 |
| 7 | Away reliability | Games played / 30 |
| 8 | Combined FG% | Home + Away FG% |
| 9 | Combined Reb | Home + Away rebounds |
| 10 | Combined TOV | Home + Away turnovers |
| 11 | Recent intensity | Sum of absolute margins (L5) |
| 12 | Combined momentum | Sum of absolute momentum |
| 13 | Season progress | Total games / 164 |
| 14 | Injury total adj | Star PPG out (lowers total) |

### Current Reality

**Simple model slightly wins.** Enhanced total improved from 15.00 (8 feat) to 14.86 (15 feat) but still trails Simple (14.78). Vegas beats both.

### Why Keep the Enhanced Model?

1. **Injury tracking available** - Coefficient shows model learns from star injuries
2. **Total improved** - 15 features better than old 8 features
3. **Edge classifier input** - May provide different signals for identifying betting edges
4. **Different decay** - 0.93 vs 0.97 captures different dynamics

### When to Use Each

- **Simple Model**: Primary model. Clean, interpretable, slightly better MAE.
- **Enhanced Model**: Use for injury-aware predictions and edge classifier experiments.

### Code Files
- Simple: `nba_simple_model.py` → `models/nba_simple_model.pkl`
- Enhanced: `nba_enhanced_ridge.py` → `models/nba_ridge_enhanced.pkl`

---

## 14. Future Research Directions

The enhanced model provides infrastructure for exploring:

1. **Injury-specific edge** - Evaluate ATS on games with star injuries vs without
2. **Dynamic HCA edge** - Do teams with unusual HCA values present betting opportunities?
3. **Momentum signals** - Currently weak (coef -0.01) - explore different windows or definitions
4. **Segment-specific models** - Train separate models for early/mid/late season
5. **Non-linear injury effects** - #1 star out may have different impact than #2/#3

---

*Last updated: 2026-01-04*
*Models: nba_simple_model.py, nba_enhanced_ridge.py*
*Edge Classifier: nba_edge_classifier.py*
*Injury Data: player_game_stats.did_not_play (all seasons)*
