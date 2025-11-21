"""
Sports Prediction App - Standalone Version for Streamlit Cloud
No API required - authentication built-in
"""
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime
import hashlib

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
    page_title="Sports Predictions",
    page_icon="🏈",
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
def get_predictions_from_db(week: int):
    """Get predictions from users.db"""
    try:
        conn = sqlite3.connect('users.db')
        query = """
            SELECT * FROM prediction_cache
            WHERE week = ? AND season = 2024
            ORDER BY game_date
        """
        df = pd.read_sql_query(query, conn, params=(week,))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()

# ============================================================================
# Login Page
# ============================================================================

def login_page():
    """Display login page"""
    st.title("🏈 Sports Predictions")
    st.subheader("Beta Access Login")

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
    st.markdown('<p class="main-header">College Football Predictions</p>', unsafe_allow_html=True)
    st.markdown(f"### Welcome back, {user['full_name']}!")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Games", f"{db_stats['total_games']:,}", delta="2024 Season")

    with col2:
        completion_pct = db_stats['completed_games']/max(db_stats['total_games'],1)*100
        st.metric("Completed Games", f"{db_stats['completed_games']:,}",
                 delta=f"{completion_pct:.0f}% complete")

    with col3:
        st.metric("Teams Tracked", f"{db_stats['total_teams']}", delta="FBS Teams")

    with col4:
        odds_pct = db_stats['games_with_odds']/max(db_stats['total_games'],1)*100
        st.metric("Games w/ Odds", f"{db_stats['games_with_odds']:,}",
                 delta=f"{odds_pct:.0f}% coverage")

    st.markdown("---")

    # Recent games
    st.markdown("### 📅 Recent Games")
    conn = get_db_connection()
    recent_games = pd.read_sql_query("""
        SELECT
            g.week,
            g.date as game_date,
            ht.school_name as home_team,
            at.school_name as away_team,
            g.home_score,
            g.away_score,
            g.completed
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.season = 2024
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
    """Show predictions page"""
    st.markdown('<p class="main-header">View Predictions</p>', unsafe_allow_html=True)

    # Week selector
    col1, col2 = st.columns([3, 1])

    with col1:
        week = st.number_input("Select Week", min_value=1, max_value=15, value=13)

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Fetch predictions
    predictions_df = get_predictions_from_db(week)

    if predictions_df.empty:
        st.warning(f"No predictions available for Week {week}")
        return

    st.success(f"Found {len(predictions_df)} games for Week {week}")

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
        completed = predictions_df['game_completed'].sum()
        st.metric("Completed", int(completed))

    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        status_filter = st.selectbox("Status", ["All", "Pending", "Completed"])

    with col2:
        spread_filter = st.selectbox("Spread Size",
                                     ["All", "Close (<7)", "Medium (7-14)", "Large (>14)"])

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

    st.markdown(f"### Showing {len(filtered_df)} predictions")

    # Display predictions
    for idx, row in filtered_df.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"

        with st.expander(matchup):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Predicted Score**")
                st.markdown(f"**{row['predicted_away_score']} - {row['predicted_home_score']}**")
                st.markdown(f"{row['away_team']}")
                st.markdown(f"{row['home_team']}")
                st.markdown(f"Win Probability: {row['home_win_probability']:.1%}")

            with col2:
                st.markdown("**Spread & Total**")
                spread = row['predicted_spread']
                if spread > 0:
                    spread_text = f"{row['home_team']} {spread:+.1f}"
                else:
                    spread_text = f"{row['away_team']} {abs(spread):+.1f}"

                st.markdown(f"Spread: **{spread_text}**")
                st.markdown(f"Total: **{row['predicted_total']:.1f}** (O/U)")

            with col3:
                st.markdown("**Result**")
                # Check if game is actually completed with valid scores
                actual_away = row['actual_away_score']
                actual_home = row['actual_home_score']

                # Only show results if we have valid actual scores (not None and not both zero)
                has_valid_scores = (actual_away is not None and actual_home is not None and
                                   not (actual_away == 0 and actual_home == 0))

                if row['game_completed'] and has_valid_scores:
                    st.markdown(f"**{actual_away:.0f} - {actual_home:.0f}**")

                    # Check prediction accuracy
                    predicted_winner = row['home_team'] if row['predicted_home_score'] > row['predicted_away_score'] else row['away_team']
                    actual_winner = row['home_team'] if actual_home > actual_away else row['away_team']

                    if predicted_winner == actual_winner:
                        st.success("✓ Correct prediction!")
                    else:
                        st.error("✗ Incorrect prediction")
                else:
                    st.warning("⏳ Pending")

def show_database_explorer():
    """Database exploration interface"""
    st.markdown('<p class="main-header">Database Explorer</p>', unsafe_allow_html=True)

    conn = get_db_connection()

    table = st.selectbox("Select Table", ["games", "teams", "team_game_stats"])

    st.markdown(f"### {table.upper()} Table")

    if table == "games":
        query = """
            SELECT
                g.game_id, g.week, g.date as game_date,
                ht.school_name as home_team, at.school_name as away_team,
                g.home_score, g.away_score, g.completed
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.season = 2024
            ORDER BY g.date DESC
            LIMIT 100
        """
    elif table == "teams":
        query = "SELECT * FROM teams ORDER BY school_name LIMIT 100"
    elif table == "team_game_stats":
        query = """
            SELECT tgs.*, t.school_name as team_name
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
        st.markdown("# 🏈 CFB Predictions")
        st.markdown("---")

        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Overview"

        page = st.radio("Navigate", ["Overview", "View Predictions", "Database Explorer"])
        st.session_state.current_page = page

        st.markdown("---")
        # Prediction engine status
        if DEEP_EAGLE_AVAILABLE:
            st.success("🚀 Deep Eagle Active")
        else:
            st.info("📊 Statistical Mode")

    # Load database stats
    db_stats = load_database_stats()

    # Main content
    if st.session_state.current_page == "Overview":
        show_overview(user, db_stats)
    elif st.session_state.current_page == "View Predictions":
        show_predictions()
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
