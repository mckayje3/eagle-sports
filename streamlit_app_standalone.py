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
    """Show predictions page with CFB and NFL tabs"""
    st.markdown('<p class="main-header">View Predictions</p>', unsafe_allow_html=True)

    # Sport tabs
    sport_tab = st.tabs(["College Football", "NFL"])

    with sport_tab[0]:
        show_sport_predictions('CFB', max_week=15, default_week=13)

    with sport_tab[1]:
        show_sport_predictions('NFL', max_week=18, default_week=12)


def show_sport_predictions(sport: str, max_week: int, default_week: int):
    """Show predictions for a specific sport with confidence intervals"""
    sport_emoji = "🏈" if sport == "CFB" else "🏟️"
    st.markdown(f"### {sport_emoji} {sport} Week {default_week} Predictions")

    # Week selector
    col1, col2 = st.columns([3, 1])

    with col1:
        week = st.number_input(f"Select {sport} Week", min_value=1, max_value=max_week, value=default_week, key=f"{sport}_week")

    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key=f"{sport}_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Fetch predictions
    predictions_df = get_predictions_from_db(week, sport=sport, season=2025)

    if predictions_df.empty:
        st.warning(f"No predictions available for Week {week}")
        return

    st.success(f"Found {len(predictions_df)} games for Week {week}")

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

    # Display predictions
    for idx, row in filtered_df.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        conf_val = row.get('confidence') if row.get('confidence') is not None else 0.85
        conf_pct = conf_val * 100
        conf_indicator = "🟢" if conf_pct >= 90 else "🟡" if conf_pct >= 80 else "🔴"

        with st.expander(f"{conf_indicator} {matchup} ({conf_pct:.0f}% confidence)"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Predicted Score**")
                st.markdown(f"**{row['predicted_away_score']:.1f} - {row['predicted_home_score']:.1f}**")
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

                # Confidence interval display
                st.markdown("---")
                st.markdown(f"**Confidence: {conf_pct:.0f}%**")

                # Show spread range if available
                spread_low = row.get('spread_low') if row.get('spread_low') is not None else spread - 2
                spread_high = row.get('spread_high') if row.get('spread_high') is not None else spread + 2
                total_low = row.get('total_low') if row.get('total_low') is not None else row['predicted_total'] - 3
                total_high = row.get('total_high') if row.get('total_high') is not None else row['predicted_total'] + 3

                st.caption(f"Spread range: {spread_low:+.1f} to {spread_high:+.1f}")
                st.caption(f"Total range: {total_low:.1f} to {total_high:.1f}")

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

                    # Show if spread was covered
                    actual_spread = actual_home - actual_away
                    if abs(actual_spread - spread) < 3:
                        st.info("📊 Spread within 3 pts")
                else:
                    st.warning("⏳ Pending")

                # Vegas comparison
                vegas_spread = row.get('vegas_spread')
                vegas_total = row.get('vegas_total')
                if vegas_spread and not pd.isna(vegas_spread):
                    st.markdown("---")
                    st.markdown("**Vegas Lines**")
                    st.caption(f"Spread: {vegas_spread:+.1f}")
                    if vegas_total and not pd.isna(vegas_total):
                        st.caption(f"Total: {vegas_total:.1f}")

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

        page = st.radio("Navigate", ["Overview", "View Predictions", "Database Explorer"])
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
