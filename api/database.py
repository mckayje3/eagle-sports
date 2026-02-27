"""
Database models and connection for the Sports Prediction API
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./users.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """User account model"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    predictions_viewed = relationship("PredictionView", back_populates="user")


class PredictionView(Base):
    """Track which predictions a user has viewed"""
    __tablename__ = "prediction_views"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    game_id = Column(String, index=True)  # ESPN game ID
    sport = Column(String, default="cfb")
    season = Column(Integer)
    week = Column(Integer)
    viewed_at = Column(DateTime, default=datetime.utcnow)

    # Store the prediction that was shown
    home_team = Column(String)
    away_team = Column(String)
    predicted_home_score = Column(Integer)
    predicted_away_score = Column(Integer)
    predicted_spread = Column(Float)
    predicted_total = Column(Float)
    home_win_probability = Column(Float)

    # Store actual results when available
    actual_home_score = Column(Integer, nullable=True)
    actual_away_score = Column(Integer, nullable=True)
    prediction_correct = Column(Boolean, nullable=True)

    # Relationships
    user = relationship("User", back_populates="predictions_viewed")


class PredictionCache(Base):
    """Cache predictions to avoid regenerating"""
    __tablename__ = "prediction_cache"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(String, unique=True, index=True)
    sport = Column(String, default="cfb")
    season = Column(Integer)
    week = Column(Integer)

    home_team = Column(String)
    away_team = Column(String)
    game_date = Column(DateTime)

    # Predictions
    predicted_home_score = Column(Integer)
    predicted_away_score = Column(Integer)
    predicted_spread = Column(Float)
    predicted_total = Column(Float)
    home_win_probability = Column(Float)

    # Actual results (filled in after game)
    actual_home_score = Column(Integer, nullable=True)
    actual_away_score = Column(Integer, nullable=True)
    game_completed = Column(Boolean, default=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


if __name__ == "__main__":
    # Create tables when run directly
    init_db()
