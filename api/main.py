"""
Main FastAPI application for Sports Prediction API
"""
import logging
from collections import defaultdict
from time import time
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Simple Rate Limiting (in-memory, for single instance)
# =============================================================================
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # window in seconds

# Store: {ip: [(timestamp, count)]}
rate_limit_store = defaultdict(list)


def check_rate_limit(request: Request) -> bool:
    """Check if request exceeds rate limit. Returns True if allowed."""
    client_ip = request.client.host if request.client else "unknown"
    now = time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries and count recent requests
    rate_limit_store[client_ip] = [
        ts for ts in rate_limit_store[client_ip] if ts > window_start
    ]

    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return False

    rate_limit_store[client_ip].append(now)
    return True


async def rate_limit_dependency(request: Request):
    """Dependency that enforces rate limiting"""
    if not check_rate_limit(request):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )

# Add parent directory to path to import prediction modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database import get_db, init_db, User, PredictionView, PredictionCache
from api.auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    create_user,
    get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from api.models import (
    UserCreate,
    UserResponse,
    Token,
    PredictionResponse,
    PredictionListResponse,
    UserStatsResponse,
    HealthResponse
)

# Initialize database on startup
init_db()

# Create FastAPI app
app = FastAPI(
    title="Sports Prediction API",
    description="API for college football and sports predictions using deep learning",
    version="1.0.0"
)

# CORS middleware - configure allowed origins from environment
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check with per-sport status"""
    try:
        from system_health import check_system_health
        health = check_system_health()
        return {
            "status": health.status,
            "timestamp": datetime.utcnow(),
            "version": "1.0.0",
            "last_daily_update": health.last_daily_update,
            "daily_update_success": health.daily_update_success,
            "issues": health.issues,
            "sports": {
                sport: {
                    "status": h.status,
                    "last_data_update": h.last_data_update,
                    "last_prediction_update": h.last_prediction_update,
                    "data_age_hours": round(h.data_age_hours, 1) if h.data_age_hours else None,
                    "prediction_age_hours": round(h.prediction_age_hours, 1) if h.prediction_age_hours else None,
                    "upcoming_games": h.upcoming_games,
                    "games_with_predictions": h.games_with_predictions,
                    "games_with_odds": h.games_with_odds,
                    "issues": h.issues,
                }
                for sport, h in health.sports.items()
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow(),
            "version": "1.0.0",
            "issues": [str(e)],
        }


@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user

    Note: For beta, you'll create users manually. This endpoint can be disabled in production.
    """
    try:
        db_user = create_user(
            db=db,
            email=user.email,
            username=user.username,
            password=user.password,
            full_name=user.full_name
        )
        return db_user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/token", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    _: None = Depends(rate_limit_dependency)
):
    """
    Login endpoint - returns JWT token

    Use username or email in the 'username' field
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    logger.info(f"Successful login for user: {user.username}")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user profile"""
    return current_user


@app.get("/users/me/stats", response_model=UserStatsResponse)
async def get_user_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user prediction statistics"""
    # Get all predictions viewed by user
    views = db.query(PredictionView).filter(PredictionView.user_id == current_user.id).all()

    # Calculate statistics
    total_viewed = len(views)

    # Current week views - dynamically determine current week
    current_week = get_current_week_from_db(db)
    this_week_views = len([v for v in views if v.week == current_week])

    # Calculate accuracy on completed predictions
    completed = [v for v in views if v.prediction_correct is not None]
    overall_accuracy = None
    if completed:
        correct = sum(1 for v in completed if v.prediction_correct)
        overall_accuracy = correct / len(completed) * 100

    # Find favorite teams
    team_counts = {}
    for view in views:
        for team in [view.home_team, view.away_team]:
            team_counts[team] = team_counts.get(team, 0) + 1

    favorite_teams = sorted(team_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    favorite_teams = [team for team, _ in favorite_teams]

    return {
        "total_predictions_viewed": total_viewed,
        "predictions_this_week": this_week_views,
        "overall_accuracy": overall_accuracy,
        "spread_accuracy": None,  # TODO: Calculate
        "total_accuracy": None,    # TODO: Calculate
        "favorite_teams": favorite_teams
    }


@app.get("/predictions/cfb/week/{week}", response_model=PredictionListResponse)
async def get_week_predictions(
    week: int,
    season: int = 2024,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Input validation
    if not (1 <= week <= 18):
        raise HTTPException(status_code=400, detail="Week must be between 1 and 18")
    if not (2020 <= season <= 2030):
        raise HTTPException(status_code=400, detail="Season must be between 2020 and 2030")
    """
    Get predictions for a specific week

    This will return cached predictions from the database
    """
    # Get predictions from cache
    predictions = db.query(PredictionCache).filter(
        PredictionCache.season == season,
        PredictionCache.week == week,
        PredictionCache.sport == "cfb"
    ).all()

    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for week {week}"
        )

    # Log that user viewed these predictions
    for pred in predictions:
        view = PredictionView(
            user_id=current_user.id,
            game_id=pred.game_id,
            sport=pred.sport,
            season=pred.season,
            week=pred.week,
            home_team=pred.home_team,
            away_team=pred.away_team,
            predicted_home_score=pred.predicted_home_score,
            predicted_away_score=pred.predicted_away_score,
            predicted_spread=pred.predicted_spread,
            predicted_total=pred.predicted_total,
            home_win_probability=pred.home_win_probability,
            actual_home_score=pred.actual_home_score,
            actual_away_score=pred.actual_away_score,
            prediction_correct=None if not pred.game_completed else (
                (pred.predicted_home_score > pred.predicted_away_score) ==
                (pred.actual_home_score > pred.actual_away_score)
            )
        )
        db.add(view)

    db.commit()

    return {
        "predictions": predictions,
        "total_count": len(predictions),
        "week": week,
        "season": season
    }


def get_current_week_from_db(db: Session) -> int:
    """Dynamically determine the current week from database"""
    from sqlalchemy import text

    try:
        # Get the most recent incomplete game's week
        result = db.execute(text("""
            SELECT week FROM prediction_cache
            WHERE game_completed = 0
            ORDER BY game_date ASC
            LIMIT 1
        """)).fetchone()

        if result:
            return result[0]

        # Fallback: get the latest week with predictions
        result = db.execute(text("""
            SELECT MAX(week) FROM prediction_cache
        """)).fetchone()

        if result and result[0]:
            return result[0]

    except Exception:
        pass

    # Final fallback based on current date (rough estimate)
    from datetime import datetime
    now = datetime.now()
    # CFB season typically starts late August/early September
    if now.month >= 8:
        weeks_since_august = (now.month - 8) * 4 + (now.day // 7)
        return min(max(1, weeks_since_august), 15)

    return 1  # Default to week 1


@app.get("/predictions/cfb/current", response_model=PredictionListResponse)
async def get_current_week_predictions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get predictions for the current week"""
    current_week = get_current_week_from_db(db)
    return await get_week_predictions(
        week=current_week,
        current_user=current_user,
        db=db
    )


@app.get("/predictions/history", response_model=PredictionListResponse)
async def get_prediction_history(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 50
):
    # Input validation
    if not (1 <= limit <= 500):
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 500")
    """Get user's prediction viewing history"""
    views = db.query(PredictionView).filter(
        PredictionView.user_id == current_user.id
    ).order_by(
        PredictionView.viewed_at.desc()
    ).limit(limit).all()

    # Convert to PredictionResponse format
    predictions = []
    for view in views:
        predictions.append({
            "game_id": view.game_id,
            "sport": view.sport,
            "season": view.season,
            "week": view.week,
            "home_team": view.home_team,
            "away_team": view.away_team,
            "game_date": view.viewed_at,  # Using viewed_at as proxy
            "predicted_home_score": view.predicted_home_score,
            "predicted_away_score": view.predicted_away_score,
            "predicted_spread": view.predicted_spread,
            "predicted_total": view.predicted_total,
            "home_win_probability": view.home_win_probability,
            "actual_home_score": view.actual_home_score,
            "actual_away_score": view.actual_away_score,
            "game_completed": view.prediction_correct is not None
        })

    return {
        "predictions": predictions,
        "total_count": len(predictions),
        "week": 0,  # Mixed weeks
        "season": 2024
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
