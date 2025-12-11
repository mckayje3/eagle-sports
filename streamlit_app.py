"""
Eagle Eye Sports Tracker - Main Streamlit Application
Works with both local development and Streamlit Cloud deployment.
Self-contained with built-in authentication (no separate API required).
"""
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime
import hashlib
import subprocess
import os
import requests
from timezone_utils import now_eastern, today_eastern

# Try to import feedparser for RSS feeds
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# Try to import deep-eagle module (with fallback)
DEEP_EAGLE_AVAILABLE = False
try:
    import deep_eagle
    DEEP_EAGLE_AVAILABLE = True
except ImportError:
    # Graceful fallback - app still works with statistical predictions
    pass

# Page configuration
st.set_page_config(
    page_title="Eagle Eye Sports Tracker",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .app-header {
        font-size: 3rem;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .app-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Simple Authentication (Built-in)
# ============================================================================

# Beta user credentials (hashed passwords)
USERS = {
    "user1": {
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "full_name": "Beta User 1",
        "email": "user1@example.com"
    },
    "user2": {
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "full_name": "Beta User 2",
        "email": "user2@example.com"
    },
    "user3": {
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "full_name": "Beta User 3",
        "email": "user3@example.com"
    },
    "user4": {
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "full_name": "Beta User 4",
        "email": "user4@example.com"
    },
    "user5": {
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "full_name": "Beta User 5",
        "email": "user5@example.com"
    },
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "full_name": "Admin User",
        "email": "admin@example.com"
    }
}

def verify_password(username: str, password: str) -> bool:
    """Verify username and password"""
    if username not in USERS:
        return False
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == USERS[username]["password_hash"]

def get_user_info(username: str) -> dict:
    """Get user information"""
    if username in USERS:
        return {
            "username": username,
            "full_name": USERS[username]["full_name"],
            "email": USERS[username]["email"]
        }
    return None

# ============================================================================
# Database Migration & Setup
# ============================================================================

def migrate_database():
    """Ensure database schema is up to date"""
    try:
        conn = sqlite3.connect('cfb_games.db')
        cursor = conn.cursor()

        # Check if school_name column exists
        cursor.execute("PRAGMA table_info(teams)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'school_name' not in columns:
            # Add school_name column
            cursor.execute("ALTER TABLE teams ADD COLUMN school_name TEXT")

            # Populate school_name by extracting from display_name
            cursor.execute("SELECT team_id, name, display_name FROM teams")
            teams = cursor.fetchall()

            for team_id, mascot, display_name in teams:
                if display_name and mascot:
                    school_name = display_name.replace(mascot, "").strip()
                else:
                    school_name = display_name
                cursor.execute("UPDATE teams SET school_name = ? WHERE team_id = ?",
                             (school_name, team_id))

            conn.commit()

        conn.close()
    except Exception as e:
        # Silently fail - app will still work with display_name fallback
        pass

# Run migration on app startup
migrate_database()

# ============================================================================
# Database Functions
# ============================================================================

@st.cache_resource
def get_db_connection():
    """Create cached database connection"""
    return sqlite3.connect('cfb_games.db', check_same_thread=False)

@st.cache_data(ttl=3600)
def check_schema():
    """Check which columns exist in teams table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(teams)")
    columns = [col[1] for col in cursor.fetchall()]
    return {
        'has_school_name': 'school_name' in columns,
        'has_display_name': 'display_name' in columns
    }

def get_team_name_field():
    """Get the appropriate field name for team names"""
    schema = check_schema()
    if schema['has_school_name']:
        return 'school_name'
    elif schema['has_display_name']:
        return 'display_name'
    else:
        return 'name'

@st.cache_data(ttl=300)
def load_database_stats():
    """Load summary statistics from database"""
    conn = get_db_connection()
    stats = {}

    stats['total_games'] = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM games WHERE season = 2024", conn
    ).iloc[0]['count']

    stats['completed_games'] = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM games WHERE completed = 1 AND season = 2024", conn
    ).iloc[0]['count']

    stats['total_teams'] = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM teams", conn
    ).iloc[0]['count']

    try:
        stats['games_with_odds'] = pd.read_sql_query(
            "SELECT COUNT(DISTINCT game_id) as count FROM game_odds", conn
        ).iloc[0]['count']
    except:
        stats['games_with_odds'] = 0

    return stats

@st.cache_data(ttl=60)
def get_predictions_from_db(week: int, sport: str = 'CFB', season: int = 2025):
    """Get predictions from users.db"""
    try:
        conn = sqlite3.connect('users.db')
        query = """
            SELECT * FROM prediction_cache
            WHERE week = ? AND sport = ? AND season = ?
            ORDER BY game_date
        """
        df = pd.read_sql_query(query, conn, params=(week, sport.upper(), season))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_nfl_stats():
    """Load NFL database statistics"""
    try:
        conn = sqlite3.connect('nfl_games.db', check_same_thread=False)
        stats = {}
        stats['total_games'] = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM games WHERE season = 2025", conn
        ).iloc[0]['count']
        stats['completed_games'] = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM games WHERE completed = 1 AND season = 2025", conn
        ).iloc[0]['count']
        stats['total_teams'] = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM teams", conn
        ).iloc[0]['count']
        try:
            stats['games_with_odds'] = pd.read_sql_query(
                "SELECT COUNT(DISTINCT game_id) as count FROM game_odds", conn
            ).iloc[0]['count']
        except:
            stats['games_with_odds'] = 0
        conn.close()
        return stats
    except:
        return {'total_games': 0, 'completed_games': 0, 'total_teams': 32, 'games_with_odds': 0}

# ============================================================================
# Login Page
# ============================================================================

def login_page():
    """Display login page"""
    st.markdown('<h1 class="app-header">🦅 Eagle Eye Sports Tracker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">AI-Powered College Football Predictions</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🔐 Beta Access Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
            elif verify_password(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.markdown("---")
    st.info("""
    **Beta Testing**

    Welcome to the Sports Prediction App beta!

    **Test Credentials:**
    - Username: user1, user2, user3, user4, or user5
    - Password: password123

    (Admin: admin / admin123)
    """)

# ============================================================================
# Main Dashboard Pages
# ============================================================================

def show_overview(user, db_stats):
    """Show overview/home page"""
    st.markdown('<p class="main-header">Sports Predictions Dashboard</p>', unsafe_allow_html=True)
    st.markdown(f"### Welcome back, {user['full_name']}!")

    # CFB Stats
    st.markdown("### College Football (2025)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("CFB Games", f"{db_stats['total_games']:,}", delta="2025 Season")

    with col2:
        completion_pct = db_stats['completed_games']/max(db_stats['total_games'],1)*100
        st.metric("Completed", f"{db_stats['completed_games']:,}",
                 delta=f"{completion_pct:.0f}% complete")

    with col3:
        st.metric("Teams", f"{db_stats['total_teams']}", delta="FBS Teams")

    with col4:
        odds_pct = db_stats['games_with_odds']/max(db_stats['total_games'],1)*100
        st.metric("Games w/ Odds", f"{db_stats['games_with_odds']:,}",
                 delta=f"{odds_pct:.0f}% coverage")

    # NFL Stats
    nfl_stats = load_nfl_stats()
    st.markdown("### NFL (2025)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("NFL Games", f"{nfl_stats['total_games']:,}", delta="2025 Season")

    with col2:
        nfl_completion_pct = nfl_stats['completed_games']/max(nfl_stats['total_games'],1)*100
        st.metric("Completed", f"{nfl_stats['completed_games']:,}",
                 delta=f"{nfl_completion_pct:.0f}% complete")

    with col3:
        st.metric("Teams", f"{nfl_stats['total_teams']}", delta="NFL Teams")

    with col4:
        nfl_odds_pct = nfl_stats['games_with_odds']/max(nfl_stats['total_games'],1)*100
        st.metric("Games w/ Odds", f"{nfl_stats['games_with_odds']:,}",
                 delta=f"{nfl_odds_pct:.0f}% coverage")

    st.markdown("---")

    # Recent games
    st.markdown("### 📅 Recent Games")
    conn = get_db_connection()
    team_field = get_team_name_field()
    recent_games = pd.read_sql_query(f"""
        SELECT
            g.week,
            g.date as game_date,
            ht.{team_field} as home_team,
            at.{team_field} as away_team,
            g.home_score,
            g.away_score,
            g.completed
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.season = 2024
        AND g.home_team_id > 0
        AND (g.home_score > 0 OR g.away_score > 0)
        ORDER BY g.date DESC
        LIMIT 10
    """, conn)

    recent_games['result'] = recent_games.apply(
        lambda x: f"{x['away_score']:.0f}-{x['home_score']:.0f}" if x['completed'] else "Pending",
        axis=1
    )

    display_df = recent_games[['week', 'away_team', 'home_team', 'result']]
    display_df.columns = ['Week', 'Away Team', 'Home Team', 'Result']

    st.dataframe(display_df, use_container_width=True, hide_index=True)

def show_predictions():
    """Show predictions page with CFB, NFL, and NBA tabs"""
    st.markdown('<p class="main-header">View Predictions</p>', unsafe_allow_html=True)

    # Initialize sport selection in session state if not exists
    if 'selected_sport' not in st.session_state:
        st.session_state.selected_sport = "College Football"

    # Sport selector that persists across reruns
    sport_options = ["College Football", "NFL", "NBA", "College Basketball"]
    selected_sport = st.radio(
        "Select Sport",
        sport_options,
        index=sport_options.index(st.session_state.selected_sport) if st.session_state.selected_sport in sport_options else 0,
        horizontal=True,
        key="sport_selector"
    )

    # Update session state
    st.session_state.selected_sport = selected_sport

    st.divider()

    # Show predictions for selected sport
    if selected_sport == "College Football":
        show_sport_predictions('CFB', max_week=15, default_week=14)
    elif selected_sport == "NFL":
        show_sport_predictions('NFL', max_week=18, default_week=14)
    elif selected_sport == "NBA":
        show_nba_predictions_live()
    else:
        show_cbb_predictions_live()


def run_nfl_predictions_update(week: int = None):
    """Run the NFL predictions update - calls module directly"""
    try:
        from update_predictions import update_predictions
        success, predictions_df = update_predictions(force_week=week)
        if success:
            return True, "NFL predictions updated successfully!"
        else:
            return False, "Failed to generate NFL predictions"
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Error running update: {str(e)}"


def run_cfb_predictions_update(week: int = None):
    """Run the CFB predictions update - calls module directly"""
    try:
        from update_predictions_cfb import update_predictions
        success, predictions_df = update_predictions(force_week=week)
        if success:
            return True, "CFB predictions updated successfully!"
        else:
            return False, "Failed to generate CFB predictions"
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Error running update: {str(e)}"


def run_nba_predictions_update(days: int = 7):
    """Run the NBA predictions update - calls module directly"""
    try:
        from update_predictions_nba import update_predictions
        success, predictions_df = update_predictions(days=days)
        if success:
            return True, "NBA predictions updated successfully!"
        else:
            return False, "Failed to generate NBA predictions"
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Error running update: {str(e)}"


def run_cbb_predictions_update(days: int = 7):
    """Run the CBB predictions update - calls module directly"""
    try:
        from update_predictions_cbb import update_predictions
        success, predictions_df = update_predictions(days=days)
        if success:
            return True, "CBB predictions updated successfully!"
        else:
            return False, "Failed to generate CBB predictions"
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Error running update: {str(e)}"


def sync_nfl_predictions_to_cache():
    """Sync NFL predictions from nfl_games.db to users.db prediction_cache"""
    try:
        # Read from nfl_games.db predictions table
        nfl_conn = sqlite3.connect('nfl_games.db')
        predictions_query = '''
            SELECT
                p.game_id,
                g.season,
                g.week,
                g.date as game_date,
                ht.display_name as home_team,
                at.display_name as away_team,
                p.pred_spread as predicted_spread,
                p.pred_total as predicted_total,
                p.pred_home_score as predicted_home_score,
                p.pred_away_score as predicted_away_score,
                p.pred_home_win_prob as home_win_probability,
                p.pred_home_win_prob as confidence,
                p.vegas_spread,
                p.vegas_total,
                p.spread_edge,
                g.completed as game_completed,
                g.home_score as actual_home_score,
                g.away_score as actual_away_score
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.date >= date('now', '-7 days')
            ORDER BY g.week, g.date
        '''
        predictions_df = pd.read_sql_query(predictions_query, nfl_conn)
        nfl_conn.close()

        if predictions_df.empty:
            return False, "No predictions found in NFL database"

        # Add sport column
        predictions_df['sport'] = 'NFL'

        # Write to users.db prediction_cache
        users_conn = sqlite3.connect('users.db')
        cursor = users_conn.cursor()

        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_cache (
                game_id INTEGER,
                season INTEGER,
                week INTEGER,
                sport TEXT,
                game_date TEXT,
                home_team TEXT,
                away_team TEXT,
                predicted_spread REAL,
                predicted_total REAL,
                predicted_home_score REAL,
                predicted_away_score REAL,
                home_win_probability REAL,
                confidence REAL,
                vegas_spread REAL,
                vegas_total REAL,
                spread_edge REAL,
                game_completed INTEGER,
                actual_home_score REAL,
                actual_away_score REAL,
                PRIMARY KEY (game_id, sport)
            )
        ''')

        # Delete existing NFL predictions and insert new
        cursor.execute("DELETE FROM prediction_cache WHERE sport = 'NFL'")

        for _, row in predictions_df.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO prediction_cache
                (game_id, season, week, sport, game_date, home_team, away_team,
                 predicted_spread, predicted_total, predicted_home_score, predicted_away_score,
                 home_win_probability, confidence, vegas_spread, vegas_total,
                 game_completed, actual_home_score, actual_away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['game_id'], row['season'], row['week'], row['sport'],
                row['game_date'], row['home_team'], row['away_team'],
                row['predicted_spread'], row['predicted_total'],
                row.get('predicted_home_score'), row.get('predicted_away_score'),
                row.get('home_win_probability'), row.get('confidence'),
                row.get('vegas_spread'), row.get('vegas_total'),
                row['game_completed'], row.get('actual_home_score'), row.get('actual_away_score')
            ))

        users_conn.commit()
        users_conn.close()

        return True, f"Synced {len(predictions_df)} NFL predictions to dashboard"

    except Exception as e:
        return False, f"Sync error: {str(e)}"


def is_fallback_predictions(df):
    """
    Detect if predictions are fallback (not from Deep Eagle).
    Fallback predictions have identical spread/total values for all games.
    Returns True if predictions appear to be fallback values.
    """
    if df.empty or len(df) < 3:
        return False

    # Check if all spreads are identical (strong indicator of fallback)
    spreads = df['predicted_spread'].dropna().unique()
    if len(spreads) == 1:
        return True

    # Check if spread variance is very low (< 0.1) across many games
    spread_std = df['predicted_spread'].std()
    if spread_std < 0.1 and len(df) > 5:
        return True

    return False


def display_game_card(row, sport_emoji="🏈"):
    """
    Unified game display component for all sports.
    Shows: Metric | Model | Vegas | Result columns with comparison tags.

    row: DataFrame row with prediction data
    sport_emoji: emoji for the sport (🏈 for NFL/CFB, 🏀 for NBA)
    """
    home_team = row['home_team']
    away_team = row['away_team']

    # Model predictions
    model_home_score = row.get('predicted_home_score')
    model_away_score = row.get('predicted_away_score')
    model_spread = row.get('predicted_spread', 0)  # home - away, positive = home favorite
    model_total = row.get('predicted_total', 0)
    model_winner = home_team if model_spread > 0 else away_team
    home_win_prob = row.get('home_win_probability', 0.5)
    confidence = row.get('confidence', 0.5)

    # Vegas lines
    vegas_spread = row.get('vegas_spread')  # home team spread, negative = home favorite
    vegas_total = row.get('vegas_total')
    has_vegas = vegas_spread is not None and not pd.isna(vegas_spread) and vegas_spread != 0

    # Calculate Vegas implied scores if available
    vegas_home_score = None
    vegas_away_score = None
    vegas_winner = None
    if has_vegas and vegas_total and not pd.isna(vegas_total):
        vegas_home_score = vegas_total / 2 - vegas_spread / 2
        vegas_away_score = vegas_total / 2 + vegas_spread / 2
        vegas_winner = home_team if vegas_spread < 0 else away_team

    # Actual results
    game_completed = row.get('game_completed', 0)
    actual_home = row.get('actual_home_score')
    actual_away = row.get('actual_away_score')
    has_result = game_completed and actual_home is not None and actual_away is not None

    if has_result:
        actual_spread = actual_home - actual_away
        actual_total = actual_home + actual_away
        actual_winner = home_team if actual_home > actual_away else (away_team if actual_away > actual_home else "Tie")

    # Format spread for display
    # Model spread stored as: home - away (positive = home wins, negative = home loses)
    # Vegas convention: negative = home favored, positive = away favored
    # So we need to NEGATE the model spread to convert to Vegas convention
    def format_spread(spread_val):
        """Convert model spread to Vegas convention.
        Model: home - away (negative = home loses)
        Vegas: negative = home favored
        So negate: if home loses by 11 (model = -11), display as +11 (home is underdog)"""
        if spread_val is None or pd.isna(spread_val):
            return "NL"
        # Negate to convert to Vegas convention
        vegas_format = -spread_val
        return f"{vegas_format:+.1f}"

    def format_vegas_spread(spread_val):
        """Vegas spread already in correct convention.
        Negative = home team favored, Positive = away team favored."""
        if spread_val is None or pd.isna(spread_val):
            return "NL"
        return f"{spread_val:+.1f}"

    # Build the expander title
    conf_pct = (confidence if confidence else 0.5) * 100
    title = f"{sport_emoji} {away_team} @ {home_team}"

    with st.expander(title, expanded=False):
        # Create 4 columns: Metric | Model | Vegas | Result
        col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

        with col1:
            st.markdown("**Metric**")
            st.write(f"✈️ {away_team}")
            st.write(f"🏠 {home_team}")
            st.write("Spread")
            st.write("Total")
            st.write("Winner")

        with col2:
            st.markdown("**Model**")
            if model_home_score is not None:
                st.write(f"{model_away_score:.0f}")
                st.write(f"{model_home_score:.0f}")
            else:
                st.write("-")
                st.write("-")
            st.write(format_spread(model_spread))
            st.write(f"{model_total:.1f}" if model_total else "-")
            st.write(f"{model_winner} ({conf_pct:.0f}%)")

        with col3:
            st.markdown("**Vegas**")
            if has_vegas:
                if vegas_home_score is not None:
                    st.write(f"{vegas_away_score:.0f}")
                    st.write(f"{vegas_home_score:.0f}")
                else:
                    st.write("-")
                    st.write("-")
                st.write(format_vegas_spread(vegas_spread))
                st.write(f"{vegas_total:.1f}" if vegas_total else "-")
                st.write(vegas_winner if vegas_winner else "-")
            else:
                st.write("NL")
                st.write("NL")
                st.write("NL")
                st.write("NL")
                st.write("NL")

        with col4:
            st.markdown("**Result**")
            if has_result:
                st.write(f"{actual_away:.0f}")
                st.write(f"{actual_home:.0f}")
                st.write(format_spread(actual_spread))
                st.write(f"{actual_total:.0f}")
                st.write(actual_winner)
            else:
                st.write("--")
                st.write("--")
                st.write("--")
                st.write("--")
                st.write("Pending")

        # Comparison tags (only for completed games)
        if has_result:
            st.markdown("---")
            tags = []

            # Winner comparison
            model_winner_correct = (model_winner == actual_winner)

            if has_vegas:
                vegas_winner_correct = (vegas_winner == actual_winner)

                if model_winner_correct and vegas_winner_correct:
                    tags.append("🎯 Both Got Winner")
                elif model_winner_correct and not vegas_winner_correct:
                    tags.append("✅ Model Beat Vegas (Winner)")
                elif not model_winner_correct and vegas_winner_correct:
                    tags.append("❌ Vegas Beat Model (Winner)")
                else:
                    tags.append("😬 Both Wrong (Winner)")
            else:
                # No Vegas lines - just show if model was correct
                if model_winner_correct:
                    tags.append("✅ Model Got Winner")
                else:
                    tags.append("❌ Model Wrong (Winner)")

            if has_vegas:
                # Spread comparison: who was closer to actual spread?
                # model_spread = home_score - away_score (positive = home wins)
                # vegas_spread = home team's line (negative = home favored)
                # To convert vegas to same format: if vegas_spread = -7, home expected to win by 7
                # So vegas_implied_spread = -vegas_spread
                vegas_implied_spread = -vegas_spread

                model_spread_diff = abs(actual_spread - model_spread)
                vegas_spread_diff = abs(actual_spread - vegas_implied_spread)

                if model_spread_diff < vegas_spread_diff - 0.5:
                    tags.append("✅ Model Beat Vegas (Spread)")
                elif vegas_spread_diff < model_spread_diff - 0.5:
                    tags.append("❌ Vegas Beat Model (Spread)")
                else:
                    tags.append("🤝 Spread Tie")

                # Total comparison
                if vegas_total and not pd.isna(vegas_total):
                    model_total_diff = abs(actual_total - model_total)
                    vegas_total_diff = abs(actual_total - vegas_total)

                    if model_total_diff < vegas_total_diff - 0.5:
                        tags.append("✅ Model Beat Vegas (Total)")
                    elif vegas_total_diff < model_total_diff - 0.5:
                        tags.append("❌ Vegas Beat Model (Total)")
                    else:
                        tags.append("🤝 Total Tie")

            # Display tags
            if tags:
                st.write(" | ".join(tags))


def display_prediction_freshness(predictions_df, game_dates=None, is_past_games=False, sport=None):
    """
    Display when predictions were generated and warn if stale (only for upcoming games).

    Args:
        predictions_df: DataFrame with predictions (should have 'created_at' column)
        game_dates: Optional list/series of game dates for comparison
        is_past_games: If True, games are in the past so don't warn about staleness
        sport: Optional sport identifier for session state timestamp lookup
    """
    from datetime import datetime, timedelta

    latest_created = None

    # First check session state for recent update timestamp (persists across reruns)
    if sport:
        session_key = f'{sport.lower()}_last_update'
        if session_key in st.session_state:
            session_timestamp = st.session_state[session_key]
            latest_created = pd.to_datetime(session_timestamp)

    # Fall back to database timestamp if no session state
    if latest_created is None:
        # Check if created_at column exists and has data
        if 'created_at' not in predictions_df.columns:
            st.caption("Prediction timestamp not available")
            return

        # Get the most recent prediction timestamp
        created_times = predictions_df['created_at'].dropna()
        if created_times.empty:
            st.caption("Prediction timestamp not available")
            return

        # Parse timestamps - handle various formats
        try:
            # Try to get the latest created_at time
            latest_created = pd.to_datetime(created_times).max()
            if pd.isna(latest_created):
                st.caption("Prediction timestamp not available")
                return
        except Exception:
            st.caption("Prediction timestamp not available")
            return

    now = now_eastern()
    # Convert to naive datetime for comparison (both are effectively Eastern Time)
    latest_dt = latest_created.to_pydatetime()
    if latest_dt.tzinfo is None:
        # Naive datetime - compare with naive now
        age = now.replace(tzinfo=None) - latest_dt
    else:
        age = now - latest_dt
    hours_old = age.total_seconds() / 3600

    # Format the timestamp
    created_str = latest_created.strftime('%a %b %d, %I:%M %p')

    # For past games, just show info without stale warnings
    if is_past_games:
        st.info(f"Predictions generated: **{created_str}**")
        return

    # Determine freshness status (only warn for upcoming games)
    if hours_old < 12:
        # Fresh - generated within 12 hours
        st.success(f"Predictions generated: **{created_str}** ({hours_old:.0f}h ago)")
    elif hours_old < 24:
        # Getting stale - 12-24 hours old
        st.warning(f"Predictions generated: **{created_str}** ({hours_old:.0f}h ago) - Consider refreshing")
    elif hours_old < 48:
        # Stale - 1-2 days old
        st.warning(f"Predictions generated: **{created_str}** ({hours_old/24:.1f} days ago) - Predictions may be stale")
    else:
        # Very stale - over 2 days old
        days_old = hours_old / 24
        st.error(f"Predictions generated: **{created_str}** ({days_old:.0f} days ago) - Predictions are stale, please refresh!")


def display_top_picks(predictions_df, sport_emoji="🏈", max_picks=5, min_disagreement=5.0):
    """
    Display top picks where model disagrees most with Vegas.

    Args:
        predictions_df: DataFrame with predictions and Vegas lines
        sport_emoji: Emoji for the sport
        max_picks: Maximum number of picks to show
        min_disagreement: Minimum disagreement threshold in points
    """
    # Get column names based on sport (CBB uses different column names)
    spread_col = 'pred_spread' if 'pred_spread' in predictions_df.columns else 'predicted_spread'
    total_col = 'pred_total' if 'pred_total' in predictions_df.columns else 'predicted_total'

    # Filter to games with Vegas odds
    df = predictions_df.copy()
    df = df[df['vegas_spread'].notna() & df['vegas_total'].notna()]

    if df.empty:
        return

    # Calculate disagreement
    # Model spread: home - away (positive = home wins)
    # Vegas spread: negative = home favored
    # To compare: convert model to Vegas convention by negating
    df['spread_disagreement'] = abs((-df[spread_col]) - df['vegas_spread'])
    df['total_disagreement'] = abs(df[total_col] - df['vegas_total'])
    df['max_disagreement'] = df[['spread_disagreement', 'total_disagreement']].max(axis=1)

    # Filter by minimum threshold and sort
    top_picks = df[df['max_disagreement'] >= min_disagreement].nlargest(max_picks, 'max_disagreement')

    if top_picks.empty:
        return

    st.markdown("### 🎯 Top Picks - Model vs Vegas Disagreements")
    st.caption(f"Games where model disagrees with Vegas by {min_disagreement}+ points")

    for idx, row in top_picks.iterrows():
        home_team = row.get('home_team', 'Home')
        away_team = row.get('away_team', 'Away')
        model_spread = row[spread_col]
        vegas_spread = row['vegas_spread']
        model_total = row[total_col]
        vegas_total = row['vegas_total']
        spread_diff = row['spread_disagreement']
        total_diff = row['total_disagreement']

        # Determine which disagreement is larger
        if spread_diff >= total_diff:
            # Spread is the bigger disagreement
            # Convert model spread to Vegas convention
            model_vegas_spread = -model_spread
            if model_vegas_spread < vegas_spread:
                # Model likes home more than Vegas
                pick_team = home_team
                pick_desc = f"Model: {model_vegas_spread:+.1f} vs Vegas: {vegas_spread:+.1f}"
            else:
                # Model likes away more than Vegas
                pick_team = away_team
                pick_desc = f"Model: {model_vegas_spread:+.1f} vs Vegas: {vegas_spread:+.1f}"
            disagreement_type = "Spread"
            disagreement_val = spread_diff
        else:
            # Total is the bigger disagreement
            if model_total > vegas_total:
                pick_desc = f"OVER {vegas_total:.1f} (Model: {model_total:.1f})"
            else:
                pick_desc = f"UNDER {vegas_total:.1f} (Model: {model_total:.1f})"
            pick_team = ""
            disagreement_type = "Total"
            disagreement_val = total_diff

        # Display the pick
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            st.write(f"{sport_emoji} **{away_team}** @ **{home_team}**")
        with col2:
            if disagreement_type == "Spread":
                st.write(f"Spread: {pick_desc}")
            else:
                st.write(f"{pick_desc}")
        with col3:
            st.write(f"**{disagreement_val:.1f}** pts")

    st.markdown("---")


def show_nba_predictions():
    """Show NBA predictions - coming soon placeholder (legacy function)"""
    show_nba_predictions_live()


def show_nba_predictions_live():
    """Show live NBA predictions for 2025-26 season"""
    st.markdown("### 🏀 NBA 2024-25 Predictions")

    # NBA uses season=2025 for 2024-25 season
    season = 2025

    from datetime import timedelta

    # Calendar-style date picker for basketball (3 days before, today, 3 days after)
    today = today_eastern()

    # Create a row of date buttons: 3 before + today + 3 after = 7 days
    st.markdown("**Select Date:**")
    date_cols = st.columns(7)

    # Initialize selected date in session state
    if 'nba_selected_date' not in st.session_state:
        st.session_state.nba_selected_date = today

    for i, col in enumerate(date_cols):
        # i=0 is 3 days ago, i=3 is today, i=6 is 3 days from now
        date = today + timedelta(days=i - 3)
        day_name = date.strftime('%a')  # Mon, Tue, etc.
        day_num = date.strftime('%d')   # 01, 02, etc.
        month = date.strftime('%b')     # Dec, Jan, etc.

        # Highlight today differently
        if date == today:
            label = f"Today\n{month} {day_num}"
        else:
            label = f"{day_name}\n{month} {day_num}"

        with col:
            # Use different styling for selected date
            is_selected = st.session_state.nba_selected_date == date
            if st.button(label, key=f"nba_date_{i}", use_container_width=True,
                        type="primary" if is_selected else "secondary"):
                st.session_state.nba_selected_date = date
                st.rerun()

    selected_date = st.session_state.nba_selected_date

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔄 Refresh", use_container_width=True, key="nba_refresh"):
            st.cache_data.clear()
            st.rerun()

    with col2:
        if st.button("📊 Update Predictions", use_container_width=True, key="nba_update"):
            with st.spinner("Fetching latest odds and updating predictions..."):
                success, msg = run_nba_predictions_update(days=7)
                if success:
                    st.session_state['nba_last_update'] = datetime.now().isoformat()
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(msg)

    # Format selected date for SQL query (YYYY-MM-DD)
    date_str = selected_date.strftime('%Y-%m-%d')

    # Fetch NBA predictions for the selected date
    try:
        conn = sqlite3.connect('users.db')
        query = """
            SELECT * FROM prediction_cache
            WHERE sport = 'NBA' AND season = ?
            AND date(game_date) = date(?)
            ORDER BY game_date
        """
        predictions_df = pd.read_sql_query(query, conn, params=(season, date_str))
        conn.close()
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        predictions_df = pd.DataFrame()

    if predictions_df.empty:
        st.info(f"No games scheduled for {selected_date.strftime('%B %d, %Y')}")

        # Show games on other days
        try:
            conn = sqlite3.connect('users.db')
            other_games_query = """
                SELECT date(game_date) as game_day, COUNT(*) as count
                FROM prediction_cache
                WHERE sport = 'NBA' AND season = ?
                GROUP BY date(game_date)
                ORDER BY game_day DESC
                LIMIT 14
            """
            other_df = pd.read_sql_query(other_games_query, conn, params=(season,))
            conn.close()

            if not other_df.empty:
                st.markdown("**Games available on other days:**")
                for _, row in other_df.iterrows():
                    game_day = pd.to_datetime(row['game_day']).strftime('%a %b %d')
                    st.write(f"• {game_day}: {row['count']} games")
        except Exception as e:
            st.warning(f"Could not load other games: {e}")
        return

    # Header with date
    st.markdown(f"### 🏀 Games for {selected_date.strftime('%A, %B %d, %Y')}")
    st.info(f"**{len(predictions_df)} games** scheduled")

    # Check if viewing past games
    is_past_games = selected_date < today_eastern()

    # Show prediction freshness
    display_prediction_freshness(predictions_df, is_past_games=is_past_games, sport='NBA')

    # Check if predictions are fallback (not from Deep Eagle)
    if is_fallback_predictions(predictions_df):
        st.warning("These predictions are using fallback values (not Deep Eagle). "
                   "Run populate_nba_predictions.py to regenerate with Deep Eagle.")

    # Check if confidence column exists, add if not
    if 'confidence' not in predictions_df.columns:
        predictions_df['confidence'] = 0.85
    else:
        predictions_df['confidence'] = predictions_df['confidence'].fillna(0.85)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Total Points", f"{predictions_df['predicted_total'].mean():.1f}")

    with col2:
        st.metric("Avg Spread", f"{predictions_df['predicted_spread'].abs().mean():.1f} pts")

    with col3:
        close_games = (predictions_df['predicted_spread'].abs() < 5).sum()
        st.metric("Close Games (<5)", close_games)

    with col4:
        avg_conf = predictions_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.0%}")

    st.markdown("---")

    # Top Picks - Model vs Vegas Disagreements
    display_top_picks(predictions_df, sport_emoji="🏀", max_picks=5, min_disagreement=5.0)

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox("Status", ["All", "Pending", "Completed"], key="nba_status")

    with col2:
        spread_filter = st.selectbox("Spread Size",
                                     ["All", "Close (<5)", "Medium (5-10)", "Large (>10)"], key="nba_spread_filter")

    with col3:
        conf_filter = st.selectbox("Confidence",
                                   ["All", "High (88%+)", "Medium (80-88%)", "Low (<80%)"], key="nba_conf_filter")

    # Apply filters
    filtered_df = predictions_df.copy()

    if status_filter == "Pending":
        filtered_df = filtered_df[filtered_df['game_completed'] == 0]
    elif status_filter == "Completed":
        filtered_df = filtered_df[filtered_df['game_completed'] == 1]

    if spread_filter == "Close (<5)":
        filtered_df = filtered_df[filtered_df['predicted_spread'].abs() < 5]
    elif spread_filter == "Medium (5-10)":
        filtered_df = filtered_df[(filtered_df['predicted_spread'].abs() >= 5) &
                                  (filtered_df['predicted_spread'].abs() <= 10)]
    elif spread_filter == "Large (>10)":
        filtered_df = filtered_df[filtered_df['predicted_spread'].abs() > 10]

    if conf_filter == "High (88%+)":
        filtered_df = filtered_df[filtered_df['confidence'] >= 0.88]
    elif conf_filter == "Medium (80-88%)":
        filtered_df = filtered_df[(filtered_df['confidence'] >= 0.80) &
                                  (filtered_df['confidence'] < 0.88)]
    elif conf_filter == "Low (<80%)":
        filtered_df = filtered_df[filtered_df['confidence'] < 0.80]

    st.markdown(f"### Showing {len(filtered_df)} predictions")

    # Display predictions using unified display component
    for idx, row in filtered_df.iterrows():
        display_game_card(row, sport_emoji="🏀")


def show_cbb_predictions_live():
    """Show live College Basketball predictions from cbb_games.db"""
    st.markdown("### 🏀 Men's College Basketball Predictions")

    # CBB uses season=2025 for 2024-25 season
    season = 2025

    from datetime import timedelta

    # Calendar-style date picker for basketball (3 days before, today, 3 days after)
    today = today_eastern()

    # Create a row of date buttons: 3 before + today + 3 after = 7 days
    st.markdown("**Select Date:**")
    date_cols = st.columns(7)

    # Initialize selected date in session state
    if 'cbb_selected_date' not in st.session_state:
        st.session_state.cbb_selected_date = today

    for i, col in enumerate(date_cols):
        # i=0 is 3 days ago, i=3 is today, i=6 is 3 days from now
        date = today + timedelta(days=i - 3)
        day_name = date.strftime('%a')  # Mon, Tue, etc.
        day_num = date.strftime('%d')   # 01, 02, etc.
        month = date.strftime('%b')     # Dec, Jan, etc.

        # Highlight today differently
        if date == today:
            label = f"Today\n{month} {day_num}"
        else:
            label = f"{day_name}\n{month} {day_num}"

        with col:
            # Use different styling for selected date
            is_selected = st.session_state.cbb_selected_date == date
            if st.button(label, key=f"cbb_date_{i}", use_container_width=True,
                        type="primary" if is_selected else "secondary"):
                st.session_state.cbb_selected_date = date
                st.rerun()

    selected_date = st.session_state.cbb_selected_date

    # Action buttons row
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔄 Refresh", use_container_width=True, key="cbb_refresh"):
            st.cache_data.clear()
            st.rerun()

    with col2:
        if st.button("📊 Update Predictions", use_container_width=True, key="cbb_update"):
            with st.spinner("Fetching latest odds and updating predictions..."):
                success, msg = run_cbb_predictions_update(days=7)
                if success:
                    st.session_state['cbb_last_update'] = datetime.now().isoformat()
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(msg)

    # Load predictions from CSV file
    try:
        predictions_df = pd.read_csv('cbb_predictions.csv')
    except FileNotFoundError:
        st.warning("No predictions file found. Click 'Update Predictions' to create predictions.")
        return
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return

    if predictions_df.empty:
        st.warning("No predictions available.")
        return

    # Enrich predictions with Vegas odds and results from database
    try:
        conn = sqlite3.connect('cbb_games.db')

        # Get game results and odds
        enrichment_query = """
            SELECT
                g.game_id,
                g.completed,
                g.home_score as actual_home_score,
                g.away_score as actual_away_score,
                o.closing_spread_home as vegas_spread,
                o.closing_total as vegas_total
            FROM games g
            LEFT JOIN game_odds o ON g.game_id = o.game_id
        """
        enrichment_df = pd.read_sql_query(enrichment_query, conn)
        conn.close()

        # Merge with predictions
        predictions_df['game_id'] = predictions_df['game_id'].astype(str)
        enrichment_df['game_id'] = enrichment_df['game_id'].astype(str)
        predictions_df = predictions_df.merge(enrichment_df, on='game_id', how='left')

    except Exception as e:
        st.warning(f"Could not load odds/results from database: {e}")

    # Filter by selected date - handle mixed date formats (YYYY-MM-DD and ISO with time)
    predictions_df['game_date'] = pd.to_datetime(predictions_df['date'], format='mixed', utc=True).dt.date
    filtered_df = predictions_df[predictions_df['game_date'] == selected_date]

    if filtered_df.empty:
        st.info(f"No games scheduled for {selected_date.strftime('%B %d, %Y')}")
        # Show how many games on other days
        games_by_date = predictions_df.groupby('game_date').size()
        if not games_by_date.empty:
            st.markdown("**Games available on other days:**")
            for date, count in games_by_date.items():
                st.write(f"• {date.strftime('%a %b %d')}: {count} games")
        return

    # Header with date
    st.markdown(f"### 🏀 Games for {selected_date.strftime('%A, %B %d, %Y')}")
    st.info(f"**{len(filtered_df)} games** scheduled")

    # Determine if viewing past games (for freshness display)
    is_past_games = False
    if 'date' in filtered_df.columns and not filtered_df.empty:
        try:
            latest_game_date = pd.to_datetime(filtered_df['date']).max()
            if pd.notna(latest_game_date) and latest_game_date.date() < today_eastern():
                is_past_games = True
        except Exception:
            pass

    # Show prediction freshness (CBB - check session state first, then CSV file)
    try:
        import os
        now = datetime.now()  # Use naive datetime for comparison
        mod_time = None

        # Check session state for recent update timestamp (persists across reruns)
        if 'cbb_last_update' in st.session_state:
            mod_time = datetime.fromisoformat(st.session_state['cbb_last_update'])
            # Remove timezone info if present for comparison
            if mod_time.tzinfo is not None:
                mod_time = mod_time.replace(tzinfo=None)
        else:
            # Fall back to CSV file modification time
            csv_path = 'cbb_predictions.csv'
            if os.path.exists(csv_path):
                mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path))

        if mod_time:
            age = now - mod_time
            hours_old = age.total_seconds() / 3600
            mod_str = mod_time.strftime('%a %b %d, %I:%M %p')

            # For past games, just show info without stale warnings
            if is_past_games:
                st.info(f"Predictions generated: **{mod_str}**")
            elif hours_old < 12:
                st.success(f"Predictions generated: **{mod_str}** ({hours_old:.0f}h ago)")
            elif hours_old < 24:
                st.warning(f"Predictions generated: **{mod_str}** ({hours_old:.0f}h ago) - Consider refreshing")
            elif hours_old < 48:
                st.warning(f"Predictions generated: **{mod_str}** ({hours_old/24:.1f} days ago) - Predictions may be stale")
            else:
                st.error(f"Predictions generated: **{mod_str}** ({hours_old/24:.0f} days ago) - Predictions are stale, please refresh!")
    except Exception:
        pass

    # Use filtered_df for the rest of the function
    predictions_df = filtered_df

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Total Points", f"{predictions_df['pred_total'].mean():.1f}")

    with col2:
        st.metric("Avg Spread", f"{predictions_df['pred_spread'].abs().mean():.1f} pts")

    with col3:
        close_games = (predictions_df['pred_spread'].abs() < 5).sum()
        st.metric("Close Games (<5)", close_games)

    with col4:
        avg_conf = predictions_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.0%}")

    st.markdown("---")

    # Top Picks - Model vs Vegas Disagreements (CBB uses pred_spread/pred_total columns)
    display_top_picks(predictions_df, sport_emoji="🏀", max_picks=5, min_disagreement=5.0)

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox("Status", ["All", "Pending", "Completed"], key="cbb_status")

    with col2:
        spread_filter = st.selectbox("Spread Size",
                                     ["All", "Close (<5)", "Medium (5-10)", "Large (>10)"], key="cbb_spread_filter")

    with col3:
        conf_filter = st.selectbox("Confidence",
                                   ["All", "High (88%+)", "Medium (80-88%)", "Low (<80%)"], key="cbb_conf_filter")

    # Apply filters
    filtered_df = predictions_df.copy()

    if status_filter == "Pending":
        filtered_df = filtered_df[(filtered_df['completed'] == 0) | (filtered_df['completed'].isna())]
    elif status_filter == "Completed":
        filtered_df = filtered_df[filtered_df['completed'] == 1]

    if spread_filter == "Close (<5)":
        filtered_df = filtered_df[filtered_df['pred_spread'].abs() < 5]
    elif spread_filter == "Medium (5-10)":
        filtered_df = filtered_df[(filtered_df['pred_spread'].abs() >= 5) &
                                  (filtered_df['pred_spread'].abs() <= 10)]
    elif spread_filter == "Large (>10)":
        filtered_df = filtered_df[filtered_df['pred_spread'].abs() > 10]

    if conf_filter == "High (88%+)":
        filtered_df = filtered_df[filtered_df['confidence'] >= 0.88]
    elif conf_filter == "Medium (80-88%)":
        filtered_df = filtered_df[(filtered_df['confidence'] >= 0.80) &
                                  (filtered_df['confidence'] < 0.88)]
    elif conf_filter == "Low (<80%)":
        filtered_df = filtered_df[filtered_df['confidence'] < 0.80]

    st.markdown(f"### Showing {len(filtered_df)} predictions")

    # Rename columns to match display_game_card expectations
    display_df = filtered_df.rename(columns={
        'pred_home_score': 'predicted_home_score',
        'pred_away_score': 'predicted_away_score',
        'pred_spread': 'predicted_spread',
        'pred_total': 'predicted_total',
        'completed': 'game_completed'
    })

    # Display predictions using unified display component
    for idx, row in display_df.iterrows():
        display_game_card(row, sport_emoji="🏀")


def show_sport_predictions(sport: str, max_week: int, default_week: int):
    """Show predictions for a specific sport with confidence intervals"""
    sport_emoji = "🏈" if sport == "CFB" else "🏟️"
    st.markdown(f"### {sport_emoji} {sport} Week {default_week} Predictions")

    # Week selector and action buttons - all sports get 3 columns now
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        week = st.number_input(f"Select {sport} Week", min_value=1, max_value=max_week, value=default_week, key=f"{sport}_week")

    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key=f"{sport}_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Update Predictions button for NFL and CFB
    with col3:
        if st.button("📊 Update Predictions", use_container_width=True, key=f"{sport}_update"):
            with st.spinner("Fetching latest odds and updating predictions..."):
                if sport == "NFL":
                    # Run NFL update script
                    success, msg = run_nfl_predictions_update(week)
                    if success:
                        sync_success, sync_msg = sync_nfl_predictions_to_cache()
                        if sync_success:
                            st.session_state['nfl_last_update'] = datetime.now().isoformat()
                            st.success(f"Updated! {sync_msg}")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.warning(f"Predictions updated but sync failed: {sync_msg}")
                    else:
                        st.error(msg)
                elif sport == "CFB":
                    # Run CFB update script
                    success, msg = run_cfb_predictions_update(week)
                    if success:
                        st.session_state['cfb_last_update'] = datetime.now().isoformat()
                        st.success(msg)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(msg)

    # Fetch predictions
    predictions_df = get_predictions_from_db(week, sport=sport, season=2025)

    if predictions_df.empty:
        st.warning(f"No predictions available for Week {week}")
        return

    st.success(f"Found {len(predictions_df)} games for Week {week}")

    # Determine if viewing past games
    is_past_games = False
    if 'date' in predictions_df.columns and not predictions_df.empty:
        try:
            latest_game_date = pd.to_datetime(predictions_df['date']).max()
            if pd.notna(latest_game_date) and latest_game_date.date() < today_eastern():
                is_past_games = True
        except Exception:
            pass

    # Show prediction freshness
    display_prediction_freshness(predictions_df, is_past_games=is_past_games, sport=sport)

    # Check if predictions are fallback (not from Deep Eagle)
    if is_fallback_predictions(predictions_df):
        st.warning("These predictions are using fallback values (not Deep Eagle). "
                   "This usually means games weren't scraped for this week. "
                   "Use 'Update Predictions' to regenerate with Deep Eagle.")

    # Check if confidence column exists, add if not; also fill NaN values
    if 'confidence' not in predictions_df.columns:
        predictions_df['confidence'] = 0.85  # Default confidence
    else:
        predictions_df['confidence'] = predictions_df['confidence'].fillna(0.85)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Total Points", f"{predictions_df['predicted_total'].mean():.1f}")

    with col2:
        st.metric("Avg Spread", f"{predictions_df['predicted_spread'].abs().mean():.1f} pts")

    with col3:
        close_games = (predictions_df['predicted_spread'].abs() < 7).sum()
        st.metric("Close Games (<7)", close_games)

    with col4:
        avg_conf = predictions_df['confidence'].mean() if 'confidence' in predictions_df.columns else 0.85
        st.metric("Avg Confidence", f"{avg_conf:.0%}")

    st.markdown("---")

    # Top Picks - Model vs Vegas Disagreements
    sport_emoji_top = "🏈" if sport == "NFL" else "🏟️"
    display_top_picks(predictions_df, sport_emoji=sport_emoji_top, max_picks=5, min_disagreement=5.0)

    # High Confidence Picks Section - fill NaN confidence with 0.85
    if 'confidence' in predictions_df.columns:
        predictions_df['confidence'] = predictions_df['confidence'].fillna(0.85)
        high_conf = predictions_df[predictions_df['confidence'] >= 0.90]
    else:
        high_conf = predictions_df.head(0)
    if len(high_conf) > 0:
        st.markdown("### ⭐ High Confidence Picks (90%+)")
        for _, row in high_conf.head(5).iterrows():
            conf_val = row.get('confidence') if row.get('confidence') is not None else 0.85
            conf_pct = conf_val * 100
            spread = row['predicted_spread']
            if spread > 0:
                pick = f"{row['home_team']} {spread:+.1f}"
            else:
                pick = f"{row['away_team']} {abs(spread):+.1f}"

            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{row['away_team']}** @ **{row['home_team']}**")
            with col2:
                st.write(f"Pick: {pick}")
            with col3:
                st.write(f"🎯 {conf_pct:.0f}%")
        st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox("Status", ["All", "Pending", "Completed"], key=f"{sport}_status")

    with col2:
        spread_filter = st.selectbox("Spread Size",
                                     ["All", "Close (<7)", "Medium (7-14)", "Large (>14)"], key=f"{sport}_spread_filter")

    with col3:
        conf_filter = st.selectbox("Confidence",
                                   ["All", "High (90%+)", "Medium (80-90%)", "Low (<80%)"], key=f"{sport}_conf_filter")

    # Apply filters
    filtered_df = predictions_df.copy()

    if status_filter == "Pending":
        filtered_df = filtered_df[filtered_df['game_completed'] == 0]
    elif status_filter == "Completed":
        filtered_df = filtered_df[filtered_df['game_completed'] == 1]

    if spread_filter == "Close (<7)":
        filtered_df = filtered_df[filtered_df['predicted_spread'].abs() < 7]
    elif spread_filter == "Medium (7-14)":
        filtered_df = filtered_df[(filtered_df['predicted_spread'].abs() >= 7) &
                                  (filtered_df['predicted_spread'].abs() <= 14)]
    elif spread_filter == "Large (>14)":
        filtered_df = filtered_df[filtered_df['predicted_spread'].abs() > 14]

    if 'confidence' in filtered_df.columns:
        if conf_filter == "High (90%+)":
            filtered_df = filtered_df[filtered_df['confidence'] >= 0.90]
        elif conf_filter == "Medium (80-90%)":
            filtered_df = filtered_df[(filtered_df['confidence'] >= 0.80) &
                                      (filtered_df['confidence'] < 0.90)]
        elif conf_filter == "Low (<80%)":
            filtered_df = filtered_df[filtered_df['confidence'] < 0.80]

    st.markdown(f"### Showing {len(filtered_df)} predictions")

    # Display predictions using unified display component
    sport_emoji = "🏈" if sport == "NFL" else "🏟️"
    for idx, row in filtered_df.iterrows():
        display_game_card(row, sport_emoji=sport_emoji)


# ============================================================================
# News Functions
# ============================================================================

@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_rss_feed(url, source_name, max_items=5):
    """Fetch and parse an RSS feed"""
    if not FEEDPARSER_AVAILABLE:
        return []
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_items]:
            articles.append({
                'title': entry.get('title', 'No title'),
                'link': entry.get('link', '#'),
                'published': entry.get('published', ''),
                'source': source_name
            })
        return articles
    except Exception as e:
        return []


@st.cache_data(ttl=600)
def get_all_news():
    """Aggregate news from multiple sources"""
    all_articles = []

    # ESPN CFB
    espn_cfb = fetch_rss_feed(
        'https://www.espn.com/espn/rss/ncf/news',
        'ESPN CFB',
        max_items=8
    )
    all_articles.extend(espn_cfb)

    # ESPN NFL
    espn_nfl = fetch_rss_feed(
        'https://www.espn.com/espn/rss/nfl/news',
        'ESPN NFL',
        max_items=8
    )
    all_articles.extend(espn_nfl)

    # CBS Sports CFB
    cbs_cfb = fetch_rss_feed(
        'https://www.cbssports.com/rss/headlines/college-football/',
        'CBS Sports CFB',
        max_items=6
    )
    all_articles.extend(cbs_cfb)

    # CBS Sports NFL
    cbs_nfl = fetch_rss_feed(
        'https://www.cbssports.com/rss/headlines/nfl/',
        'CBS Sports NFL',
        max_items=6
    )
    all_articles.extend(cbs_nfl)

    # Yahoo CFB
    yahoo_cfb = fetch_rss_feed(
        'https://sports.yahoo.com/college-football/rss/',
        'Yahoo CFB',
        max_items=5
    )
    all_articles.extend(yahoo_cfb)

    # Bleacher Report
    br_feed = fetch_rss_feed(
        'https://bleacherreport.com/articles/feed',
        'Bleacher Report',
        max_items=5
    )
    all_articles.extend(br_feed)

    # Pro Football Talk (NBC Sports)
    pft_feed = fetch_rss_feed(
        'https://profootballtalk.nbcsports.com/feed/',
        'NBC PFT',
        max_items=6
    )
    all_articles.extend(pft_feed)

    # The Athletic
    athletic_feed = fetch_rss_feed(
        'https://theathletic.com/feeds/rss/news/',
        'The Athletic',
        max_items=6
    )
    all_articles.extend(athletic_feed)

    return all_articles


@st.cache_data(ttl=300)
def fetch_espn_injuries(sport='nfl'):
    """Fetch injury data from ESPN API"""
    try:
        if sport == 'nfl':
            url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries'
        else:
            url = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/injuries'

        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        return None


def get_upcoming_games_with_odds():
    """Get upcoming CFB games with Vegas odds from database"""
    try:
        conn = get_db_connection()

        query = '''
        SELECT
            g.game_id, g.season, g.week, g.date,
            ht.name as home_team, at.name as away_team,
            go.latest_line as spread,
            go.latest_total_line as total
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN game_odds go ON g.game_id = go.game_id
        WHERE g.completed = 0 AND g.season = 2025
        ORDER BY g.date
        LIMIT 20
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.to_dict('records') if not df.empty else []
    except Exception as e:
        return []


def get_nfl_upcoming_games_with_odds():
    """Get upcoming NFL games with Vegas odds from database"""
    try:
        conn = sqlite3.connect('nfl_games.db')

        query = '''
        SELECT
            g.game_id, g.season, g.week, g.date,
            ht.name as home_team, at.name as away_team,
            go.latest_line as spread,
            go.latest_total_line as total
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN game_odds go ON g.game_id = go.game_id
        WHERE g.completed = 0 AND g.season = 2025
        ORDER BY g.date
        LIMIT 20
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.to_dict('records') if not df.empty else []
    except Exception as e:
        return []


def show_news():
    """Display news aggregator page"""
    st.markdown('<p class="main-header">Sports News</p>', unsafe_allow_html=True)

    # Quick links section
    st.markdown("### Quick Links")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        **ESPN**
        - [College Football](https://www.espn.com/college-football/)
        - [NFL](https://www.espn.com/nfl/)
        - [CFB Scoreboard](https://www.espn.com/college-football/scoreboard)
        - [NFL Scoreboard](https://www.espn.com/nfl/scoreboard)
        """)

    with col2:
        st.markdown("""
        **Vegas Insider**
        - [CFB Odds](https://www.vegasinsider.com/college-football/odds/las-vegas/)
        - [NFL Odds](https://www.vegasinsider.com/nfl/odds/las-vegas/)
        - [CFB Matchups](https://www.vegasinsider.com/college-football/matchups/)
        - [NFL Matchups](https://www.vegasinsider.com/nfl/matchups/)
        """)

    with col3:
        st.markdown("""
        **Betting Resources**
        - [Action Network CFB](https://www.actionnetwork.com/ncaaf)
        - [Action Network NFL](https://www.actionnetwork.com/nfl)
        - [Covers CFB](https://www.covers.com/ncaaf)
        - [Covers NFL](https://www.covers.com/nfl)
        """)

    with col4:
        st.markdown("""
        **Analysis**
        - [CBS Sports CFB](https://www.cbssports.com/college-football/)
        - [CBS Sports NFL](https://www.cbssports.com/nfl/)
        - [247 Sports](https://247sports.com/)
        - [The Athletic](https://theathletic.com/college-football/)
        """)

    st.markdown("---")

    # News feed section
    st.markdown("### Latest Headlines")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Refresh News"):
            st.cache_data.clear()
            st.rerun()

    # Filter options
    source_filter = st.selectbox(
        "Filter by Source",
        ["All Sources", "ESPN CFB", "ESPN NFL", "CBS Sports CFB", "CBS Sports NFL", "Yahoo CFB", "Bleacher Report", "NBC PFT", "The Athletic"]
    )

    # Fetch all news
    with st.spinner("Loading news..."):
        articles = get_all_news()

    if not articles:
        st.warning("Unable to fetch news. Please try again later or check if feedparser is installed.")
    else:
        # Apply filter
        if source_filter != "All Sources":
            articles = [a for a in articles if a['source'] == source_filter]

        st.success(f"Found {len(articles)} articles")

        # Display articles in two columns
        col1, col2 = st.columns(2)

        for i, article in enumerate(articles):
            with col1 if i % 2 == 0 else col2:
                source_colors = {
                    'ESPN CFB': '#cc0000',
                    'ESPN NFL': '#cc0000',
                    'CBS Sports CFB': '#0066cc',
                    'CBS Sports NFL': '#0066cc',
                    'Yahoo CFB': '#6001d2',
                    'Bleacher Report': '#00b2a9',
                    'NBC PFT': '#006699',
                    'The Athletic': '#d63a3a'
                }
                color = source_colors.get(article['source'], '#666666')

                st.markdown(f"""
                <div style="padding: 10px; margin: 5px 0; border-left: 3px solid {color}; background-color: #f8f9fa;">
                    <small style="color: {color}; font-weight: bold;">{article['source']}</small>
                    <br>
                    <a href="{article['link']}" target="_blank" style="color: #1f77b4; text-decoration: none; font-weight: 500;">
                        {article['title']}
                    </a>
                    <br>
                    <small style="color: #888;">{article['published'][:25] if article['published'] else ''}</small>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Current Vegas Odds Section
    st.markdown("### Current Vegas Odds")

    odds_tab1, odds_tab2 = st.tabs(["CFB Odds", "NFL Odds"])

    with odds_tab1:
        cfb_odds = get_upcoming_games_with_odds()
        if cfb_odds:
            for game in cfb_odds[:10]:
                spread_str = f"Spread: {game['spread']:+.1f}" if game['spread'] else "Spread: N/A"
                total_str = f"O/U: {game['total']:.1f}" if game['total'] else "O/U: N/A"
                st.markdown(f"""
                <div style="padding: 8px; margin: 3px 0; border-left: 3px solid #cc0000; background-color: #f8f9fa;">
                    <strong>{game['away_team']} @ {game['home_team']}</strong><br>
                    <small>Week {game['week']} | {spread_str} | {total_str}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No upcoming CFB games with odds found.")

    with odds_tab2:
        nfl_odds = get_nfl_upcoming_games_with_odds()
        if nfl_odds:
            for game in nfl_odds[:10]:
                spread_str = f"Spread: {game['spread']:+.1f}" if game['spread'] else "Spread: N/A"
                total_str = f"O/U: {game['total']:.1f}" if game['total'] else "O/U: N/A"
                st.markdown(f"""
                <div style="padding: 8px; margin: 3px 0; border-left: 3px solid #0066cc; background-color: #f8f9fa;">
                    <strong>{game['away_team']} @ {game['home_team']}</strong><br>
                    <small>Week {game['week']} | {spread_str} | {total_str}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No upcoming NFL games with odds found.")

    st.markdown("---")

    # Injury Reports Section
    st.markdown("### Injury Reports")

    inj_tab1, inj_tab2 = st.tabs(["NFL Injuries", "CFB Injuries"])

    with inj_tab1:
        nfl_injuries = fetch_espn_injuries('nfl')
        if nfl_injuries and 'injuries' in nfl_injuries:
            teams_shown = 0
            for team_data in nfl_injuries['injuries'][:8]:
                team_name = team_data.get('team', {}).get('displayName', 'Unknown')
                injuries = team_data.get('injuries', [])
                if injuries:
                    teams_shown += 1
                    with st.expander(f"🏈 {team_name} ({len(injuries)} injuries)"):
                        for inj in injuries[:5]:
                            athlete = inj.get('athlete', {})
                            name = athlete.get('displayName', 'Unknown')
                            position = athlete.get('position', {}).get('abbreviation', '')
                            status = inj.get('status', 'Unknown')
                            st.markdown(f"- **{name}** ({position}) - {status}")
            if teams_shown == 0:
                st.info("No significant NFL injuries reported.")
        else:
            st.info("Unable to fetch NFL injury data.")

    with inj_tab2:
        cfb_injuries = fetch_espn_injuries('cfb')
        if cfb_injuries and 'injuries' in cfb_injuries:
            teams_shown = 0
            for team_data in cfb_injuries['injuries'][:8]:
                team_name = team_data.get('team', {}).get('displayName', 'Unknown')
                injuries = team_data.get('injuries', [])
                if injuries:
                    teams_shown += 1
                    with st.expander(f"🏈 {team_name} ({len(injuries)} injuries)"):
                        for inj in injuries[:5]:
                            athlete = inj.get('athlete', {})
                            name = athlete.get('displayName', 'Unknown')
                            position = athlete.get('position', {}).get('abbreviation', '')
                            status = inj.get('status', 'Unknown')
                            st.markdown(f"- **{name}** ({position}) - {status}")
            if teams_shown == 0:
                st.info("No significant CFB injuries reported.")
        else:
            st.info("Unable to fetch CFB injury data.")

    st.markdown("---")

    # Betting Resources
    st.markdown("### Betting Resources")

    st.info("""
    **Popular Sportsbooks:**
    - [DraftKings](https://sportsbook.draftkings.com/)
    - [FanDuel](https://sportsbook.fanduel.com/)
    - [BetMGM](https://sports.betmgm.com/)
    - [Caesars](https://www.caesars.com/sportsbook-and-casino)

    **Odds Comparison:**
    - [OddsShark](https://www.oddsshark.com/)
    - [The Odds API](https://the-odds-api.com/)
    - [Odds Portal](https://www.oddsportal.com/)
    """)


def show_database_explorer():
    """Database exploration interface"""
    st.markdown('<p class="main-header">Database Explorer</p>', unsafe_allow_html=True)

    conn = get_db_connection()

    table = st.selectbox("Select Table", ["games", "teams", "team_game_stats"])

    st.markdown(f"### {table.upper()} Table")

    team_field = get_team_name_field()

    if table == "games":
        query = f"""
            SELECT
                g.game_id, g.week, g.date as game_date,
                ht.{team_field} as home_team,
                at.{team_field} as away_team,
                g.home_score, g.away_score, g.completed
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.season = 2024
            AND g.home_team_id > 0
            AND g.away_team_id > 0
            ORDER BY g.date DESC
            LIMIT 100
        """
    elif table == "teams":
        query = f"SELECT * FROM teams WHERE team_id > 0 ORDER BY {team_field} LIMIT 100"
    elif table == "team_game_stats":
        query = f"""
            SELECT tgs.*, t.{team_field} as team_name
            FROM team_game_stats tgs
            JOIN teams t ON tgs.team_id = t.team_id
            ORDER BY tgs.game_id DESC
            LIMIT 100
        """

    try:
        df = pd.read_sql_query(query, conn)
        st.markdown(f"Showing {len(df)} records")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name=f"{table}_export.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error querying database: {e}")

# ============================================================================
# Model Insights Page
# ============================================================================

def show_model_insights():
    """Display Deep Eagle model transparency and explainability information"""
    st.markdown('<p class="main-header">Deep Eagle Model Insights</p>', unsafe_allow_html=True)

    st.markdown("""
    This page provides transparency into how the Deep Eagle prediction models work for each sport,
    including the features used, architecture, and performance metrics.
    """)

    # Sport-specific tabs
    sport_tab = st.tabs(["NFL", "College Football", "NBA", "College Basketball"])

    # =========================================================================
    # NFL TAB
    # =========================================================================
    with sport_tab[0]:
        st.markdown("## NFL Model: Deep Eagle v2.0")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model File", "deep_eagle_nfl_2025.pt")
        with col2:
            st.metric("Total Features", "78")
        with col3:
            st.metric("Training Games", "~500")
        with col4:
            st.metric("Seasons", "2024-2025")

        st.markdown("""
        ### Architecture
        - **Type:** Deep Neural Network (PyTorch)
        - **Hidden Layers:** 256 → 128 → 64 neurons
        - **Activation:** ReLU with Batch Normalization
        - **Regularization:** Dropout (0.3), Early Stopping
        - **Output:** Dual heads for home/away score prediction
        """)

        st.markdown("### Feature Categories (78 total)")
        nfl_features = {
            "Vegas Odds": {
                "features": ["Opening Spread", "Latest Spread", "Opening Total", "Latest Total", "Moneylines"],
                "count": 8,
                "description": "Market consensus from major sportsbooks"
            },
            "Historical Team Stats": {
                "features": ["PPG", "Points Allowed", "Yards/Game", "Turnovers/Game", "Win %"],
                "count": 12,
                "description": "Season-to-date averages (home and away teams)"
            },
            "Home/Away Splits": {
                "features": ["Home PPG", "Away PPG", "Home Win %", "Away Win %", "Home/Away PPG Differential"],
                "count": 18,
                "description": "Venue-specific performance metrics"
            },
            "Drive Efficiency": {
                "features": ["Points/Drive", "Yards/Drive", "Scoring %", "3-and-Out %", "Red Zone %"],
                "count": 26,
                "description": "Offensive and defensive efficiency per possession"
            },
            "Game Context": {
                "features": ["Week", "Neutral Site", "Conference Game", "Dome/Outdoor"],
                "count": 7,
                "description": "Situational factors"
            },
            "Matchup Differentials": {
                "features": ["PPG Diff", "Win % Diff", "PPD Diff", "Venue PPG Diff"],
                "count": 7,
                "description": "Head-to-head statistical comparisons"
            }
        }

        for category, info in nfl_features.items():
            with st.expander(f"**{category}** ({info['count']} features)"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Key Features:** {', '.join(info['features'])}")

        st.markdown("### Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spread MAE", "~11 points")
            st.metric("Total MAE", "~14 points")
        with col2:
            st.metric("Home Score MAE", "~10 points")
            st.metric("Away Score MAE", "~10 points")

    # =========================================================================
    # CFB TAB
    # =========================================================================
    with sport_tab[1]:
        st.markdown("## College Football Model: Deep Eagle v3")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model File", "deep_eagle_cfb_2025_v3.pt")
        with col2:
            st.metric("Total Features", "79")
        with col3:
            st.metric("Training Games", "1,777")
        with col4:
            st.metric("Seasons", "2024-2025")

        st.markdown("""
        ### Architecture
        - **Type:** Deep Neural Network (PyTorch)
        - **Hidden Layers:** 256 → 128 → 64 neurons
        - **Activation:** ReLU with Batch Normalization
        - **Regularization:** Dropout (0.3), Early Stopping (patience=20)
        - **Output:** Dual heads for home/away score prediction
        - **No Target Leak:** Model does NOT use actual game outcomes as inputs
        """)

        st.markdown("### Feature Categories (79 total)")
        cfb_features = {
            "Vegas Odds": {
                "features": ["Opening Spread", "Latest Spread", "Opening Total", "Latest Total", "Home/Away Moneylines"],
                "count": 8,
                "description": "Market consensus - opening and current lines from The Odds API"
            },
            "Historical Team Stats": {
                "features": ["Games Played", "PPG", "Points Allowed/Game", "Yards/Game", "Turnovers/Game", "Win %"],
                "count": 12,
                "description": "Season-to-date averages calculated BEFORE each game (no lookahead)"
            },
            "Home/Away Splits": {
                "features": ["Home Games", "Home PPG", "Home Points Allowed", "Home Win %", "Away equivalents", "Home/Away PPG Diff"],
                "count": 18,
                "description": "Venue-specific performance - captures home field advantage"
            },
            "Drive Efficiency (Offense)": {
                "features": ["Points/Drive", "Yards/Drive", "Plays/Drive", "Seconds/Drive", "Scoring %", "Red Zone %", "3-and-Out %", "Explosive Drive %"],
                "count": 18,
                "description": "How efficiently each team moves the ball"
            },
            "Drive Efficiency (Defense)": {
                "features": ["Defensive PPD Allowed", "Defensive YPD Allowed", "Defensive Scoring % Allowed", "3-and-Outs Forced"],
                "count": 8,
                "description": "How well defense stops opposing drives"
            },
            "Game Context": {
                "features": ["Week", "Week Normalized (0-1)", "Neutral Site", "Conference Game", "Temperature", "Wind Speed", "Is Dome"],
                "count": 7,
                "description": "Situational and environmental factors"
            },
            "Matchup Differentials": {
                "features": ["PPG Differential", "Points Allowed Differential", "Win % Differential", "PPD Differential", "Scoring % Differential", "Venue PPG Differential", "Venue Win % Differential", "Combined Home Advantage"],
                "count": 8,
                "description": "Head-to-head comparisons between teams"
            }
        }

        for category, info in cfb_features.items():
            with st.expander(f"**{category}** ({info['count']} features)"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Key Features:** {', '.join(info['features'])}")

        st.markdown("### Performance (Validation Set)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spread MAE", "14.5 points")
            st.metric("Total MAE", "14.7 points")
        with col2:
            st.metric("Home Score MAE", "10.8 points")
            st.metric("Away Score MAE", "9.9 points")

        st.markdown("""
        ### Key Design Decisions
        1. **No Target Leak:** Unlike earlier versions, v3 excludes `point_spread`, `total_points`, and `home_win` from input features
        2. **Odds as Features:** Vegas lines (opening and latest) ARE used as inputs since they're available at prediction time
        3. **Home/Away Splits:** Critical for capturing home field advantage in college football
        4. **Drive Data:** Provides more signal than raw stats about team quality
        """)

    # =========================================================================
    # NBA TAB
    # =========================================================================
    with sport_tab[2]:
        st.markdown("## NBA Model: Deep Eagle v1.0")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model File", "deep_eagle_nba_2024.pt")
        with col2:
            st.metric("Total Features", "~50")
        with col3:
            st.metric("Training Games", "~1,200")
        with col4:
            st.metric("Seasons", "2023-2024")

        st.markdown("""
        ### Architecture
        - **Type:** Deep Neural Network (PyTorch)
        - **Hidden Layers:** 256 → 128 → 64 neurons
        - **Activation:** ReLU with Batch Normalization
        - **Regularization:** Dropout (0.3)
        - **Output:** Dual heads for home/away score prediction
        """)

        st.markdown("### Feature Categories")
        nba_features = {
            "Vegas Odds": {
                "features": ["Opening Spread", "Latest Spread", "Opening Total", "Latest Total"],
                "description": "Market consensus from sportsbooks"
            },
            "Team Statistics": {
                "features": ["PPG", "Points Allowed", "FG%", "3P%", "Rebounds", "Assists", "Turnovers"],
                "description": "Season-to-date team averages"
            },
            "Home/Away Performance": {
                "features": ["Home Record", "Away Record", "Home PPG", "Away PPG"],
                "description": "Venue-specific performance splits"
            },
            "Recent Form": {
                "features": ["Last 5 Games PPG", "Last 10 Games Win %"],
                "description": "Recent performance momentum"
            }
        }

        for category, info in nba_features.items():
            with st.expander(f"**{category}**"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Key Features:** {', '.join(info['features'])}")

        st.markdown("### Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spread MAE", "~10 points")
            st.metric("Total MAE", "~15 points")
        with col2:
            st.metric("Status", "Active")
            st.metric("Last Updated", "December 2025")

        st.info("NBA model is actively being improved with additional historical data and features.")

    # =========================================================================
    # CBB TAB
    # =========================================================================
    with sport_tab[3]:
        st.markdown("## College Basketball Model: Deep Eagle v1.0")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model File", "deep_eagle_cbb_2025.pt")
        with col2:
            st.metric("Total Features", "78")
        with col3:
            st.metric("Training Games", "~5,300")
        with col4:
            st.metric("Seasons", "2024-2025")

        st.markdown("""
        ### Architecture
        - **Type:** Deep Neural Network (PyTorch)
        - **Hidden Layers:** 256 -> 128 -> 64 neurons
        - **Activation:** ReLU with Batch Normalization
        - **Regularization:** Dropout (0.3), Early Stopping (patience=20)
        - **Output:** Dual heads for home/away score prediction
        """)

        st.markdown("### Feature Categories (78 total)")
        cbb_features = {
            "Vegas Odds": {
                "features": ["Opening Spread", "Latest Spread", "Opening Total", "Latest Total", "Home/Away Moneylines"],
                "count": 8,
                "description": "Market consensus from major sportsbooks via ESPN"
            },
            "Historical Team Stats": {
                "features": ["Games Played", "PPG", "Points Allowed/Game", "FG%", "3P%", "FT%", "Rebounds", "Assists"],
                "count": 14,
                "description": "Season-to-date team averages calculated BEFORE each game"
            },
            "Home/Away Splits": {
                "features": ["Home Games", "Home PPG", "Home Points Allowed", "Home Win %", "Away equivalents", "Venue PPG Diff"],
                "count": 18,
                "description": "Venue-specific performance - captures home court advantage"
            },
            "Shooting Efficiency": {
                "features": ["eFG%", "True Shooting %", "3P Attempt Rate", "FT Rate", "Offensive Rating", "Defensive Rating"],
                "count": 12,
                "description": "Advanced shooting and efficiency metrics"
            },
            "Rebounding & Turnovers": {
                "features": ["Offensive Rebounds", "Defensive Rebounds", "Total Rebounds", "Turnovers", "Steals", "Blocks"],
                "count": 12,
                "description": "Possession-related statistics"
            },
            "Game Context": {
                "features": ["Week", "Week Normalized (0-1)", "Neutral Site", "Conference Game"],
                "count": 6,
                "description": "Situational factors affecting game outcome"
            },
            "Matchup Differentials": {
                "features": ["PPG Differential", "Points Allowed Differential", "Win % Differential", "Efficiency Differential", "Venue PPG Differential"],
                "count": 8,
                "description": "Head-to-head comparisons between teams"
            }
        }

        for category, info in cbb_features.items():
            with st.expander(f"**{category}** ({info['count']} features)"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Key Features:** {', '.join(info['features'])}")

        st.markdown("### Performance (Validation Set)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Winner Accuracy", "64.9%")
            st.metric("Spread MAE", "9.5 points")
        with col2:
            st.metric("Home Score MAE", "8.5 points")
            st.metric("Away Score MAE", "8.6 points")

        st.markdown("""
        ### Key Characteristics
        1. **High Volume:** ~6,300 games per season across 350+ teams
        2. **Conference Diversity:** Big differences between Power 5 and smaller conferences
        3. **Home Court Advantage:** Stronger effect than professional basketball
        4. **March Madness:** Tournament games have different dynamics (neutral sites)
        """)

        st.success("CBB model trained on 2024-25 season data with 12,500+ total games across both seasons.")

    # =========================================================================
    # COMMON METHODOLOGY SECTION
    # =========================================================================
    st.markdown("---")
    st.markdown("## Common Methodology")

    st.markdown("""
    ### Data Collection
    - **Game Data:** ESPN API (scores, team stats, drive data where available)
    - **Odds Data:** The Odds API (opening and current lines from major sportsbooks)
    - **Update Frequency:** Daily during active seasons

    ### Feature Engineering Principles
    1. **No Lookahead:** Historical stats only use games BEFORE the prediction target
    2. **Odds Integration:** Vegas lines used as features (available at prediction time)
    3. **Home/Away Splits:** Capture venue-specific performance differences
    4. **Standardization:** All features scaled using StandardScaler

    ### Model Training
    - **Framework:** PyTorch
    - **Optimizer:** Adam with learning rate scheduling (ReduceLROnPlateau)
    - **Loss Function:** Mean Squared Error (MSE)
    - **Regularization:** Dropout (0.3), BatchNorm, Early Stopping

    ### Known Limitations
    1. Models tend to predict toward the mean (may miss blowouts)
    2. No injury data incorporated
    3. Weather only partially captured (dome indicator, temperature)
    4. Vegas incorporates information we don't have access to
    """)


# ============================================================================
# Main App Logic
# ============================================================================

def main_page():
    """Display main application page"""
    user = get_user_info(st.session_state.username)

    # Sidebar
    with st.sidebar:
        st.title(f"👤 {user['username']}")
        st.write(f"**{user['full_name']}**")
        st.write(f"_{user['email']}_")

        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("# 🦅 Sports Predictions")
        st.markdown("---")

        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Overview"

        page = st.radio("Navigate", ["Overview", "View Predictions", "News", "Model Insights", "Database Explorer"])
        st.session_state.current_page = page

        st.markdown("---")
        # Prediction engine status
        st.success("🚀 Deep Eagle v2.0")
        st.caption("LSTM Models Active")

    # Load database stats
    db_stats = load_database_stats()
    nfl_stats = load_nfl_stats()

    # App Header
    st.markdown('<h1 class="app-header">🦅 Eagle Eye Sports Tracker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">AI-Powered CFB & NFL Predictions</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Main content
    if st.session_state.current_page == "Overview":
        show_overview(user, db_stats)
    elif st.session_state.current_page == "View Predictions":
        show_predictions()
    elif st.session_state.current_page == "News":
        show_news()
    elif st.session_state.current_page == "Model Insights":
        show_model_insights()
    elif st.session_state.current_page == "Database Explorer":
        show_database_explorer()

# ============================================================================
# Entry Point
# ============================================================================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_page()
else:
    login_page()
