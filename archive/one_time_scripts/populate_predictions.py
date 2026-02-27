"""
Populate the PredictionCache with sample predictions
This creates some sample predictions for the beta users to view
"""
from sqlalchemy.orm import Session
from api.database import SessionLocal, PredictionCache
from datetime import datetime, timedelta
import random

def create_sample_predictions(week: int = 14, season: int = 2024):
    """Create sample predictions for a given week"""
    db = SessionLocal()

    # Sample matchups (some real CFB teams)
    matchups = [
        ("Georgia", "Alabama"),
        ("Ohio State", "Michigan"),
        ("Texas", "Oklahoma"),
        ("Oregon", "Washington"),
        ("USC", "Notre Dame"),
        ("Florida State", "Clemson"),
        ("LSU", "Texas A&M"),
        ("Penn State", "Wisconsin"),
        ("Utah", "Colorado"),
        ("Tennessee", "Kentucky"),
    ]

    print(f"Creating sample predictions for Week {week}, Season {season}")
    print("=" * 60)

    for i, (away_team, home_team) in enumerate(matchups, 1):
        # Generate realistic predictions
        home_score = random.randint(20, 45)
        away_score = random.randint(17, 42)

        # Ensure some variance
        if random.random() > 0.5:
            home_score, away_score = max(home_score, away_score), min(home_score, away_score)

        spread = home_score - away_score
        total = home_score + away_score
        win_prob = 0.5 + (spread / 40.0)  # Simple win probability
        win_prob = max(0.1, min(0.9, win_prob))  # Clamp between 10% and 90%

        game_id = f"{season}-{week}-{i}"

        # Check if prediction already exists
        existing = db.query(PredictionCache).filter(
            PredictionCache.game_id == game_id
        ).first()

        if existing:
            print(f"[SKIP] {away_team} @ {home_team} - already exists")
            continue

        prediction = PredictionCache(
            game_id=game_id,
            sport="cfb",
            season=season,
            week=week,
            home_team=home_team,
            away_team=away_team,
            game_date=datetime.now() + timedelta(days=7-i),  # Spread over next week
            predicted_home_score=home_score,
            predicted_away_score=away_score,
            predicted_spread=spread,
            predicted_total=total,
            home_win_probability=win_prob,
            actual_home_score=None,
            actual_away_score=None,
            game_completed=False
        )

        db.add(prediction)
        print(f"[OK] {away_team} @ {home_team}: {away_score}-{home_score} (Spread: {spread:+.1f}, Total: {total})")

    db.commit()
    print("=" * 60)
    print(f"Sample predictions created successfully!")
    print(f"\nYou can now view these predictions in the Streamlit app.")
    db.close()

if __name__ == "__main__":
    create_sample_predictions(week=14, season=2024)
