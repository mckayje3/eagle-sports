"""
Analyze game dates to plan efficient historical odds fetching
"""
import sqlite3
from datetime import datetime

conn = sqlite3.connect('cfb_games.db')
cursor = conn.cursor()

print("=" * 80)
print("2025 SEASON GAME DATE DISTRIBUTION")
print("=" * 80)

# Get all game dates for weeks 1-14
cursor.execute('''
    SELECT week, DATE(date) as game_date, COUNT(*) as num_games
    FROM games
    WHERE season = 2025 AND week BETWEEN 1 AND 14
    GROUP BY week, DATE(date)
    ORDER BY week, game_date
''')

current_week = None
week_totals = {}
all_dates = []

print("\nWeek-by-week breakdown:\n")

for week, date_str, count in cursor.fetchall():
    if week != current_week:
        if current_week is not None:
            print(f"  Week {current_week} total: {week_totals[current_week]} games\n")
        current_week = week
        week_totals[week] = 0
        print(f"Week {week}:")

    week_totals[week] += count
    all_dates.append((week, date_str, count))

    # Parse date to show day of week
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day_name = date_obj.strftime('%A')
        print(f"  {date_str} ({day_name}): {count} games")
    except:
        print(f"  {date_str}: {count} games")

if current_week:
    print(f"  Week {current_week} total: {week_totals[current_week]} games\n")

print("=" * 80)
print("COST ANALYSIS")
print("=" * 80)

# Find Saturdays only
saturdays = [d for d in all_dates if datetime.strptime(d[1], '%Y-%m-%d').weekday() == 5]
saturday_games = sum(d[2] for d in saturdays)

# All dates
total_dates = len(all_dates)
total_games = sum(d[2] for d in all_dates)

print(f"\nOption 1: ALL GAME DATES")
print(f"  Total dates: {total_dates}")
print(f"  Total games: {total_games}")
print(f"  API cost (spreads + totals): {total_dates * 20} requests")
print(f"  Coverage: 100%")

print(f"\nOption 2: SATURDAYS ONLY")
print(f"  Total dates: {len(saturdays)}")
print(f"  Total games: {saturday_games}")
print(f"  API cost (spreads + totals): {len(saturdays) * 20} requests")
print(f"  Coverage: {saturday_games/total_games*100:.1f}%")

print(f"\nOption 3: HEAVY GAME DAYS (20+ games)")
heavy_days = [d for d in all_dates if d[2] >= 20]
heavy_games = sum(d[2] for d in heavy_days)
print(f"  Total dates: {len(heavy_days)}")
print(f"  Total games: {heavy_games}")
print(f"  API cost (spreads + totals): {len(heavy_days) * 20} requests")
print(f"  Coverage: {heavy_games/total_games*100:.1f}%")

print("\n" + "=" * 80)
print(f"You have 491 requests remaining")
print("=" * 80)

conn.close()
