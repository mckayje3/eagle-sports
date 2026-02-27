# Sports Prediction API - Beta Setup Guide

## What's Been Built

A complete authentication and API system for your 5 beta testers, including:

1. **FastAPI Backend** - RESTful API with JWT authentication
2. **SQLite Database** - User accounts and prediction storage
3. **Streamlit Web UI** - User-friendly interface for viewing predictions
4. **Sample Data** - 10 sample CFB predictions for Week 14

---

## Current Status

### Running Services

Both services are currently running:

- **API Server**: http://localhost:8000
  - Interactive API docs: http://localhost:8000/docs
- **Streamlit App**: http://localhost:8503
  - This is what your beta users will use

### Beta User Accounts

6 accounts have been created (5 beta users + 1 admin):

| Username | Email | Password | Admin |
|----------|-------|----------|-------|
| user1 | user1@example.com | password123 | No |
| user2 | user2@example.com | password123 | No |
| user3 | user3@example.com | password123 | No |
| user4 | user4@example.com | password123 | No |
| user5 | user5@example.com | password123 | No |
| admin | admin@example.com | admin123 | Yes |

**IMPORTANT**: Edit `create_users.py` to customize these accounts with:
- Real names for your friends/family
- Their actual email addresses
- Secure passwords

Then run: `py create_users.py` (will skip existing users)

---

## How to Start the Services

### 1. Start the API Server

```bash
py -m uvicorn api.main:app --reload
```

This starts the backend API on http://localhost:8000

### 2. Start the Streamlit App

```bash
py -m streamlit run streamlit_app.py
```

This starts the user interface on http://localhost:8503

---

## API Endpoints

### Authentication
- `POST /token` - Login (get JWT token)
- `POST /register` - Register new user
- `GET /users/me` - Get current user profile

### Predictions
- `GET /predictions/cfb/current` - Current week predictions
- `GET /predictions/cfb/week/{week}` - Specific week predictions
- `GET /predictions/history` - User's viewing history

### Stats
- `GET /users/me/stats` - User statistics and accuracy

### Health
- `GET /` - Basic health check
- `GET /health` - Detailed health check

**View interactive API documentation**: http://localhost:8000/docs

---

## Database

Location: `./sports_predictions.db` (SQLite)

Tables:
- `users` - User accounts
- `prediction_cache` - Cached predictions
- `prediction_views` - User viewing history

---

## Adding Real Predictions

Currently there are only sample predictions. To add real predictions from your model:

### Option 1: Create a script to populate from your trained model

```python
from api.database import SessionLocal, PredictionCache
from your_predictor import load_model, make_predictions

# Load your trained model
predictor = load_model('models/cfb_predictor.pth')

# Make predictions
predictions = predictor.predict_week(week=14, season=2024)

# Save to database
db = SessionLocal()
for pred in predictions:
    cache_entry = PredictionCache(
        game_id=pred['game_id'],
        sport='cfb',
        season=2024,
        week=14,
        home_team=pred['home_team'],
        away_team=pred['away_team'],
        # ... other fields
    )
    db.add(cache_entry)
db.commit()
```

### Option 2: Create an admin endpoint

Add an endpoint to the API that accepts predictions and saves them to the cache.

---

## Customizing for Your Beta Users

### 1. Update User Accounts

Edit `create_users.py` with your friends' real information:

```python
beta_users = [
    {
        "username": "john_smith",
        "email": "john@example.com",
        "password": "SecurePass123!",
        "full_name": "John Smith",
        "is_admin": False
    },
    # ... 4 more
]
```

### 2. Share Access

Send your beta users:
- The Streamlit app URL (http://localhost:8503 or your server's URL)
- Their username and password
- Brief instructions on how to use it

### 3. Monitor Usage

Check the database to see:
- Login activity (last_login in users table)
- What predictions users view (prediction_views table)
- User statistics via the API

---

## Security Notes

### Current Setup (Development)
- Secret key is hardcoded (OK for beta)
- SQLite database (OK for 5 users)
- Passwords are bcrypt hashed (Good!)
- JWT tokens expire after 7 days (Reasonable)
- CORS allows all origins (OK for local testing)

### Before Production
- [ ] Set SECRET_KEY via environment variable
- [ ] Switch to PostgreSQL
- [ ] Restrict CORS to your domain only
- [ ] Add rate limiting
- [ ] Use HTTPS
- [ ] Add input validation
- [ ] Implement password reset flow

---

## Files Created

```
api/
├── __init__.py
├── main.py           # FastAPI application
├── database.py       # Database models and connection
├── auth.py          # Authentication logic
└── models.py        # Pydantic schemas

create_users.py       # Script to create beta users
populate_predictions.py  # Script to add sample predictions
streamlit_app.py     # Streamlit web interface
requirements_api.txt # API dependencies
sports_predictions.db # SQLite database (created automatically)
```

---

## Next Steps

1. **Customize user accounts** - Edit create_users.py with real info
2. **Add real predictions** - Connect your trained model to populate predictions
3. **Test with beta users** - Share the Streamlit URL and credentials
4. **Collect feedback** - Monitor usage and gather user feedback
5. **Iterate** - Improve based on what you learn

---

## Troubleshooting

### API won't start
- Check if port 8000 is available
- Ensure all dependencies are installed: `py -m pip install -r requirements_api.txt`

### Streamlit won't start
- Check if port 8503 is available
- Ensure streamlit is installed: `py -m pip install streamlit`

### Login fails
- Verify the API is running at http://localhost:8000
- Check username/password in the database
- View API logs for errors

### No predictions show up
- Run `populate_predictions.py` to add sample data
- Check the database for entries in prediction_cache table
- Verify the week number matches what's in the database

---

## Support

For issues or questions:
- Check the API docs: http://localhost:8000/docs
- View API logs in the terminal running uvicorn
- View Streamlit logs in the terminal running streamlit
- Inspect the database: Use DB Browser for SQLite or similar tool

---

**Built with:**
- FastAPI 0.121.3
- Streamlit 1.51.0
- SQLAlchemy 2.0.44
- JWT Authentication
- bcrypt password hashing
