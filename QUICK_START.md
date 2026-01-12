# Quick Start

## Launch Dashboard

**Easiest:** Double-click `launch_dashboard.bat`

**Or from terminal:**
```bash
streamlit run streamlit_app.py
```

Opens at http://localhost:8501

---

## Daily Workflow

### Morning (predictions ready)
1. Open dashboard
2. Select sport (NFL, CFB, NBA, CBB)
3. Look for games with ⭐⭐⭐ (high confidence)
4. Check both spread and total edges

### Finding Best Bets
- Games sorted by edge magnitude
- ⭐⭐⭐ = 5+ point edge on spreads
- Look for OVER/UNDER with 8+ point edge

### After Games
Results update automatically with next daily run.

---

## Key Commands

```bash
# Update everything (runs daily at 9am)
python daily_update.py

# NFL playoffs
python predict_nfl_playoffs.py

# Sync to cloud
python push_databases.py
```

---

## Reading the Cards

```
Bills @ Jaguars              | 2026-01-11 SAT 1:00 PM |
Spread | Vegas: +1.5 | Model: +0.5 | Edge: -1.0 → Jaguars +1.5 | ⭐
Total  | Vegas: 52.5 | Model: 56.5 | Edge: +4.0 → OVER 52.5   | ⭐⭐
```

- **Edge negative** → bet HOME team
- **Edge positive** → bet AWAY team
- **Total edge positive** → OVER
- **Total edge negative** → UNDER

---

## Cloud App

Same data, hosted online. Syncs with daily update.

Manual sync: `python push_databases.py`
