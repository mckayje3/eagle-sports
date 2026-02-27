"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List
from datetime import datetime

# =============================================================================
# Constants for validation
# =============================================================================
MIN_WEEK = 1
MAX_WEEK = 18  # NFL/CFB max weeks
MIN_SEASON = 2020
MAX_SEASON = 2030
MIN_LIMIT = 1
MAX_LIMIT = 500


class UserCreate(BaseModel):
    """Schema for creating a new user"""
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None

    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'Username must be alphanumeric'
        assert len(v) >= 3, 'Username must be at least 3 characters'
        return v

    @validator('password')
    def password_strength(cls, v):
        assert len(v) >= 6, 'Password must be at least 6 characters'
        return v


class UserResponse(BaseModel):
    """Schema for user response (without password)"""
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Schema for authentication token"""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Schema for token payload"""
    username: Optional[str] = None


class PredictionResponse(BaseModel):
    """Schema for a single game prediction"""
    game_id: str
    sport: str
    season: int
    week: int
    home_team: str
    away_team: str
    game_date: datetime
    predicted_home_score: int
    predicted_away_score: int
    predicted_spread: float
    predicted_total: float
    home_win_probability: float
    actual_home_score: Optional[int] = None
    actual_away_score: Optional[int] = None
    game_completed: bool = False

    class Config:
        from_attributes = True


class PredictionListResponse(BaseModel):
    """Schema for list of predictions"""
    predictions: List[PredictionResponse]
    total_count: int
    week: int
    season: int


class WeekRequest(BaseModel):
    """Schema for week-based requests with validation"""
    week: int = Field(ge=MIN_WEEK, le=MAX_WEEK, description="Week number (1-18)")
    season: int = Field(ge=MIN_SEASON, le=MAX_SEASON, default=2024, description="Season year")


class PaginationParams(BaseModel):
    """Schema for pagination parameters"""
    limit: int = Field(ge=MIN_LIMIT, le=MAX_LIMIT, default=50, description="Max results to return")
    offset: int = Field(ge=0, default=0, description="Results offset")


class UserStatsResponse(BaseModel):
    """Schema for user prediction statistics"""
    total_predictions_viewed: int
    predictions_this_week: int
    overall_accuracy: Optional[float] = None
    spread_accuracy: Optional[float] = None
    total_accuracy: Optional[float] = None
    favorite_teams: List[str] = []


class SportHealthResponse(BaseModel):
    """Health status for a single sport"""
    status: str
    last_data_update: Optional[datetime] = None
    last_prediction_update: Optional[datetime] = None
    data_age_hours: Optional[float] = None
    prediction_age_hours: Optional[float] = None
    upcoming_games: int = 0
    games_with_predictions: int = 0
    games_with_odds: int = 0
    issues: List[str] = []


class HealthResponse(BaseModel):
    """Schema for health check"""
    status: str
    timestamp: datetime
    version: str
    last_daily_update: Optional[datetime] = None
    daily_update_success: Optional[bool] = None
    issues: List[str] = []
    sports: Optional[dict[str, SportHealthResponse]] = None
