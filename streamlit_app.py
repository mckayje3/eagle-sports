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

    # Sport tabs
    sport_tab = st.tabs(["College Football", "NFL", "NBA"])

    with sport_tab[0]:
        show_sport_predictions('CFB', max_week=15, default_week=14)

    with sport_tab[1]:
        show_sport_predictions('NFL', max_week=18, default_week=14)

    with sport_tab[2]:
        show_nba_predictions_live()


def run_nfl_predictions_update(week: int = None):
    """Run the NFL predictions update script"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'update_predictions.py')
        cmd = ['py', script_path]
        if week:
            cmd.extend(['--week', str(week)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=os.path.dirname(__file__)
        )

        if result.returncode == 0:
            return True, "Predictions updated successfully!"
        else:
            return False, f"Update failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Update timed out (>2 minutes)"
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
            WHERE g.season = 2025
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
                 home_win_probability, confidence, vegas_spread, vegas_total, spread_edge,
                 game_completed, actual_home_score, actual_away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['game_id'], row['season'], row['week'], row['sport'],
                row['game_date'], row['home_team'], row['away_team'],
                row['predicted_spread'], row['predicted_total'],
                row.get('predicted_home_score'), row.get('predicted_away_score'),
                row.get('home_win_probability'), row.get('confidence'),
                row.get('vegas_spread'), row.get('vegas_total'), row.get('spread_edge'),
                row['game_completed'], row.get('actual_home_score'), row.get('actual_away_score')
            ))

        users_conn.commit()
        users_conn.close()

        return True, f"Synced {len(predictions_df)} NFL predictions to dashboard"

    except Exception as e:
        return False, f"Sync error: {str(e)}"


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


def show_nba_predictions():
    """Show NBA predictions - coming soon placeholder (legacy function)"""
    show_nba_predictions_live()


def show_nba_predictions_live():
    """Show live NBA predictions for 2025-26 season"""
    st.markdown("### 🏀 NBA 2025-26 Predictions")

    # NBA uses season=2025 for 2025-26 season
    season = 2025

    # Get current "week" (weeks since Oct 15, 2025)
    from datetime import datetime
    season_start = datetime(2025, 10, 15)
    current_week = max(0, (datetime.now() - season_start).days // 7)

    col1, col2 = st.columns([3, 1])

    with col1:
        week = st.number_input("Select Week (since season start)", min_value=0, max_value=50, value=min(current_week, 7), key="nba_week")

    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key="nba_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Fetch NBA predictions
    try:
        conn = sqlite3.connect('users.db')
        query = """
            SELECT * FROM prediction_cache
            WHERE week = ? AND sport = 'NBA' AND season = ?
            ORDER BY game_date
        """
        predictions_df = pd.read_sql_query(query, conn, params=(week, season))
        conn.close()
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        predictions_df = pd.DataFrame()

    if predictions_df.empty:
        st.warning(f"No predictions available for Week {week}")

        # Show all upcoming games
        st.markdown("#### Upcoming Games")
        try:
            conn = sqlite3.connect('users.db')
            upcoming_query = """
                SELECT * FROM prediction_cache
                WHERE sport = 'NBA' AND season = ? AND game_completed = 0
                ORDER BY game_date
                LIMIT 20
            """
            upcoming_df = pd.read_sql_query(upcoming_query, conn, params=(season,))
            conn.close()

            if not upcoming_df.empty:
                st.success(f"Found {len(upcoming_df)} upcoming games")
                for _, row in upcoming_df.iterrows():
                    spread = row['predicted_spread']
                    if spread > 0:
                        pick = f"{row['home_team']} by {spread:.1f}"
                    else:
                        pick = f"{row['away_team']} by {abs(spread):.1f}"
                    st.write(f"**{row['away_team']}** @ **{row['home_team']}** - Pick: {pick}")
        except Exception as e:
            st.warning(f"Could not load upcoming games: {e}")
        return

    st.success(f"Found {len(predictions_df)} games for Week {week}")

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

    # High Confidence Picks Section
    high_conf = predictions_df[predictions_df['confidence'] >= 0.88]
    if len(high_conf) > 0:
        st.markdown("### ⭐ High Confidence Picks (88%+)")
        for _, row in high_conf.head(5).iterrows():
            conf_pct = row['confidence'] * 100
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


def show_sport_predictions(sport: str, max_week: int, default_week: int):
    """Show predictions for a specific sport with confidence intervals"""
    sport_emoji = "🏈" if sport == "CFB" else "🏟️"
    st.markdown(f"### {sport_emoji} {sport} Week {default_week} Predictions")

    # Week selector and action buttons
    if sport == "NFL":
        col1, col2, col3 = st.columns([3, 1, 1])
    else:
        col1, col2 = st.columns([3, 1])

    with col1:
        week = st.number_input(f"Select {sport} Week", min_value=1, max_value=max_week, value=default_week, key=f"{sport}_week")

    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key=f"{sport}_refresh"):
            st.cache_data.clear()
            st.rerun()

    # NFL-specific: Update Predictions button
    if sport == "NFL":
        with col3:
            if st.button("📊 Update Odds", use_container_width=True, key=f"{sport}_update"):
                with st.spinner("Fetching latest odds and updating predictions..."):
                    # Step 1: Run update script
                    success, msg = run_nfl_predictions_update(week)
                    if success:
                        # Step 2: Sync to dashboard cache
                        sync_success, sync_msg = sync_nfl_predictions_to_cache()
                        if sync_success:
                            st.success(f"Updated! {sync_msg}")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.warning(f"Predictions updated but sync failed: {sync_msg}")
                    else:
                        st.error(msg)

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
    This page provides transparency into how the Deep Eagle prediction model works,
    including the factors it considers and their relative importance.
    """)

    # Model Overview
    st.markdown("## Model Architecture")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", "Deep Neural Network")
    with col2:
        st.metric("Total Features", "57")
    with col3:
        st.metric("Training Games", "452")

    st.markdown("""
    **Deep Eagle v3** uses a multi-layer neural network with:
    - 3 hidden layers (256 → 128 → 64 neurons)
    - Batch normalization and dropout for regularization
    - Separate prediction heads for home and away scores
    - Time-based train/test split (no future data leakage)
    """)

    st.markdown("---")

    # Feature Categories
    st.markdown("## Feature Categories")

    feature_categories = {
        "Vegas Odds (Latest Available)": {
            "features": ["Latest Spread", "Latest Total"],
            "description": "Current betting lines from major sportsbooks. These capture market consensus and are strong predictors.",
            "importance": "15.5%",
            "correlation": "r = -0.43 (spread)"
        },
        "Team Performance Differentials": {
            "features": ["PPG Differential", "Win % Differential", "Points Allowed Differential", "Points Per Drive Differential"],
            "description": "Difference between home and away team season averages. Positive = home team advantage.",
            "importance": "11.0%",
            "correlation": "r = 0.30 (PPG diff)"
        },
        "Drive Efficiency Metrics": {
            "features": ["Points Per Drive", "Yards Per Drive", "Scoring %", "3-and-Out %", "Explosive Drive %"],
            "description": "Advanced metrics measuring offensive and defensive efficiency per possession.",
            "importance": "25.8%",
            "correlation": "r = 0.28 (scoring %)"
        },
        "Historical Team Stats": {
            "features": ["Games Played", "PPG", "Points Allowed", "Yards Per Game", "Turnovers Per Game", "Win %"],
            "description": "Season-to-date averages for each team entering the game.",
            "importance": "22.5%",
            "correlation": "r = 0.20 (win %)"
        },
        "Game Context": {
            "features": ["Week (normalized)", "Home/Away", "Dome/Outdoor", "Conference Game"],
            "description": "Situational factors that may influence game outcomes.",
            "importance": "7.6%",
            "correlation": "varies"
        },
        "Defensive Metrics": {
            "features": ["Defensive PPD Allowed", "Defensive YPD Allowed", "3-and-Outs Forced"],
            "description": "How well each team's defense performs against opposing offenses.",
            "importance": "17.6%",
            "correlation": "r = -0.07"
        }
    }

    for category, info in feature_categories.items():
        with st.expander(f"**{category}** — {info['importance']} of model importance"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Key Features:** {', '.join(info['features'])}")
            st.markdown(f"**Correlation with Spread:** {info['correlation']}")

    st.markdown("---")

    # Top Features by Importance
    st.markdown("## Top 20 Features by Importance")

    st.markdown("""
    Feature importance is measured using **permutation importance**: we shuffle each feature
    and measure how much prediction accuracy decreases. Higher values = more important.
    """)

    # Feature importance data
    importance_data = pd.DataFrame({
        'Feature': [
            'Latest Vegas Spread', 'Latest Vegas Total', 'Away Yards/Game',
            'Is Dome Game', 'Home 3-and-Out %', 'Away Games Played',
            'Points Allowed Differential', 'Away Total Drives', 'Home Total Drives',
            'Home Def Yards/Drive', 'Home Points Allowed', 'Away Explosive Drive %',
            'Away Turnovers/Game', 'PPG Differential', 'Home Def 3-and-Outs Forced',
            'Away Def Yards/Drive', 'Win % Differential', 'Away Yards/Drive',
            'Home PPG', 'Home Turnovers/Game'
        ],
        'Importance': [
            0.634, 0.261, 0.226, 0.225, 0.213, 0.203, 0.193, 0.186, 0.180,
            0.172, 0.164, 0.159, 0.159, 0.159, 0.134, 0.130, 0.129, 0.124, 0.123, 0.117
        ],
        'Category': [
            'Odds', 'Odds', 'Team Stats', 'Context', 'Drive Efficiency', 'Team Stats',
            'Differential', 'Drive Efficiency', 'Drive Efficiency', 'Defense',
            'Team Stats', 'Drive Efficiency', 'Team Stats', 'Differential', 'Defense',
            'Defense', 'Differential', 'Drive Efficiency', 'Team Stats', 'Team Stats'
        ]
    })
    importance_data['Pct'] = (importance_data['Importance'] / importance_data['Importance'].sum() * 100).round(1)

    # Create bar chart
    import plotly.express as px
    fig = px.bar(
        importance_data.head(15),
        x='Importance',
        y='Feature',
        color='Category',
        orientation='h',
        title='Top 15 Features by Permutation Importance',
        color_discrete_map={
            'Odds': '#2ecc71',
            'Differential': '#3498db',
            'Drive Efficiency': '#9b59b6',
            'Team Stats': '#e74c3c',
            'Defense': '#f39c12',
            'Context': '#1abc9c'
        }
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.dataframe(
        importance_data.style.format({'Importance': '{:.3f}', 'Pct': '{:.1f}%'}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Correlation Analysis
    st.markdown("## Feature Correlations with Game Spread")

    st.markdown("""
    **Correlation (r)** measures linear relationship between each feature and the actual game spread.
    - Positive r: Higher feature value → home team wins by more
    - Negative r: Higher feature value → away team wins by more
    - |r| > 0.3 is considered moderate, |r| > 0.5 is strong
    """)

    correlation_data = pd.DataFrame({
        'Feature': [
            'Latest Vegas Spread', 'PPD Differential', 'PPG Differential',
            'Win % Differential', 'Scoring % Differential', 'Away Win %',
            'Points Allowed Differential', 'Away PPD', 'Away Scoring %', 'Away PPG'
        ],
        'Correlation (r)': [-0.425, 0.300, 0.297, 0.293, 0.276, -0.200, -0.188, -0.166, -0.164, -0.163],
        'R-squared': [0.181, 0.090, 0.088, 0.086, 0.076, 0.040, 0.035, 0.028, 0.027, 0.027],
        'Interpretation': [
            'Moderate: Vegas line is predictive',
            'Moderate: Better offense helps home team',
            'Moderate: Higher-scoring teams win',
            'Moderate: Better record predicts wins',
            'Moderate: Red zone efficiency matters',
            'Weak: Better away teams win on road',
            'Weak: Allowing fewer points helps',
            'Weak: Away offensive efficiency',
            'Weak: Away scoring drives',
            'Weak: Away team scoring average'
        ]
    })

    st.dataframe(
        correlation_data.style.format({'Correlation (r)': '{:+.3f}', 'R-squared': '{:.3f}'}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Model Performance
    st.markdown("## Model Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Training Performance (Weeks 1-10)")
        st.metric("Spread MAE", "7.7 points")
        st.metric("Winner Accuracy", "67.1%")
        st.metric("Total Points MAE", "13.5 points")

    with col2:
        st.markdown("### Week 13 Actual Results")
        st.metric("Spread MAE", "11.0 points")
        st.metric("Winner Accuracy", "58.3% (7/12)")
        st.metric("vs Vegas", "Vegas: 9/12 (75%)")

    st.markdown("""
    **Note:** Training performance is measured on held-out weeks 11-13 during model development.
    Real-world performance (Week 13 actual) shows the model is competitive but doesn't beat Vegas.
    """)

    st.markdown("---")

    # Methodology
    st.markdown("## Methodology & Data Sources")

    st.markdown("""
    ### Data Collection
    - **Game Data:** ESPN API (scores, team stats, drive data)
    - **Odds Data:** The Odds API (opening and current lines from major books)
    - **Seasons Covered:** 2024-2025 NFL (452 completed games)

    ### Feature Engineering
    - Historical stats calculated using only games *before* prediction target (no lookahead)
    - Drive efficiency metrics aggregated per-team per-season
    - Odds captured as "latest available" to match prediction-time data

    ### Model Training
    - **Framework:** PyTorch
    - **Split:** Time-based (train on weeks 1-10, test on weeks 11+)
    - **Optimization:** Adam optimizer with learning rate scheduling
    - **Regularization:** Dropout (0.3), early stopping (patience=20)

    ### Known Limitations
    1. Model tends to predict spreads closer to the mean (misses blowouts)
    2. Does not account for injuries or weather (beyond dome indicator)
    3. Limited to 2 seasons of training data
    4. Vegas lines incorporate information we don't have access to
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
