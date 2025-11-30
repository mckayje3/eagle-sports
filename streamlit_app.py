"""
Sports Prediction App with Authentication
Combines authentication system with full dashboard functionality
"""
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import feedparser
from bs4 import BeautifulSoup
import re

# API Configuration
API_BASE_URL = "http://localhost:8000"

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Authentication Functions
# ============================================================================

def login(username: str, password: str) -> dict:
    """Login and get JWT token"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/token",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


def get_current_user(token: str) -> dict:
    """Get current user profile"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching user profile: {e}")
        return None


def get_user_stats(token: str) -> dict:
    """Get user statistics"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/users/me/stats",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


def get_predictions_api(token: str, week: int = None) -> dict:
    """Get predictions from API"""
    try:
        if week:
            url = f"{API_BASE_URL}/predictions/cfb/week/{week}"
        else:
            url = f"{API_BASE_URL}/predictions/cfb/current"

        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"predictions": [], "total_count": 0}
        return None
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return None


# ============================================================================
# Database Functions
# ============================================================================

def get_db_connection():
    """Create database connection"""
    return sqlite3.connect('cfb_games.db')


def load_database_stats():
    """Load summary statistics from database"""
    conn = get_db_connection()
    stats = {}

    # Total games
    stats['total_games'] = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM games", conn
    ).iloc[0]['count']

    # Completed games
    stats['completed_games'] = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM games WHERE completed = 1", conn
    ).iloc[0]['count']

    # Total teams
    stats['total_teams'] = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM teams", conn
    ).iloc[0]['count']

    # Games with odds
    try:
        stats['games_with_odds'] = pd.read_sql_query(
            "SELECT COUNT(DISTINCT game_id) as count FROM game_odds", conn
        ).iloc[0]['count']
    except:
        stats['games_with_odds'] = 0

    conn.close()
    return stats


# ============================================================================
# Login Page
# ============================================================================

def login_page():
    """Display login page"""
    st.title("🏈 Sports Predictions")
    st.subheader("Beta Access Login")

    with st.form("login_form"):
        username = st.text_input("Username or Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                token_data = login(username, password)
                if token_data:
                    st.session_state.token = token_data["access_token"]
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    st.markdown("---")
    st.info("""
    **Beta Testing**

    Welcome to the Sports Prediction App beta! Use your provided credentials to login.

    If you don't have credentials, please contact the administrator.
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
        st.metric(
            label="Total Games",
            value=f"{db_stats['total_games']:,}",
            delta="2024 Season"
        )

    with col2:
        st.metric(
            label="Completed Games",
            value=f"{db_stats['completed_games']:,}",
            delta=f"{db_stats['completed_games']/db_stats['total_games']*100:.0f}% complete"
        )

    with col3:
        st.metric(
            label="Teams Tracked",
            value=f"{db_stats['total_teams']}",
            delta="FBS Teams"
        )

    with col4:
        st.metric(
            label="Games w/ Odds",
            value=f"{db_stats['games_with_odds']:,}",
            delta=f"{db_stats['games_with_odds']/max(db_stats['total_games'],1)*100:.0f}% coverage"
        )

    st.markdown("---")

    # Your Stats
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Your Statistics")
        stats = get_user_stats(st.session_state.token)
        if stats:
            st.metric("Total Predictions Viewed", stats['total_predictions_viewed'])
            st.metric("This Week", stats['predictions_this_week'])
            if stats['overall_accuracy'] is not None:
                st.metric("Overall Accuracy", f"{stats['overall_accuracy']:.1f}%")

            if stats['favorite_teams']:
                st.write("**Favorite Teams:**")
                for team in stats['favorite_teams'][:5]:
                    st.write(f"- {team}")

    with col2:
        st.markdown("### 📅 Recent Games")
        conn = get_db_connection()
        recent_games = pd.read_sql_query("""
            SELECT
                g.week,
                g.date as game_date,
                ht.name as home_team,
                at.name as away_team,
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
        conn.close()

        recent_games['result'] = recent_games.apply(
            lambda x: f"{x['away_score']:.0f}-{x['home_score']:.0f}" if x['completed'] else "Pending",
            axis=1
        )

        display_df = recent_games[['week', 'away_team', 'home_team', 'result']]
        display_df.columns = ['Week', 'Away Team', 'Home Team', 'Result']

        st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_predictions():
    """Show predictions page with filters and details"""
    st.markdown('<p class="main-header">View Predictions</p>', unsafe_allow_html=True)

    # Week selector
    col1, col2 = st.columns([3, 1])

    with col1:
        week = st.number_input("Select Week", min_value=1, max_value=15, value=13)

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Fetch predictions
    predictions_data = get_predictions_api(st.session_state.token, week=week)

    if not predictions_data or predictions_data['total_count'] == 0:
        st.warning(f"No predictions available for Week {week}")
        return

    predictions = predictions_data['predictions']
    st.success(f"Found {len(predictions)} games for Week {week}")

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    # Calculate stats
    completed_games = [p for p in predictions if p.get('game_completed')]
    avg_total = sum(p['predicted_total'] for p in predictions) / len(predictions)
    avg_spread = sum(abs(p['predicted_spread']) for p in predictions) / len(predictions)
    close_games = sum(1 for p in predictions if abs(p['predicted_spread']) < 7)

    with col1:
        st.metric("Avg Total Points", f"{avg_total:.1f}")

    with col2:
        st.metric("Avg Spread", f"{avg_spread:.1f} pts")

    with col3:
        st.metric("Close Games (<7)", close_games)

    with col4:
        st.metric("Completed", len(completed_games))

    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        status_filter = st.selectbox(
            "Status",
            ["All", "Pending", "Completed"]
        )

    with col2:
        spread_filter = st.selectbox(
            "Spread Size",
            ["All", "Close (<7)", "Medium (7-14)", "Large (>14)"]
        )

    # Apply filters
    filtered = predictions

    if status_filter == "Pending":
        filtered = [p for p in filtered if not p.get('game_completed')]
    elif status_filter == "Completed":
        filtered = [p for p in filtered if p.get('game_completed')]

    if spread_filter == "Close (<7)":
        filtered = [p for p in filtered if abs(p['predicted_spread']) < 7]
    elif spread_filter == "Medium (7-14)":
        filtered = [p for p in filtered if 7 <= abs(p['predicted_spread']) <= 14]
    elif spread_filter == "Large (>14)":
        filtered = [p for p in filtered if abs(p['predicted_spread']) > 14]

    st.markdown(f"### Showing {len(filtered)} predictions")

    # Display predictions
    for pred in filtered:
        neutral = " (Neutral)" if pred.get('neutral_site', 0) == 1 else ""
        matchup = f"{pred['away_team']} @ {pred['home_team']}{neutral}"

        with st.expander(matchup):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Predicted Score**")
                st.markdown(f"**{pred['predicted_away_score']} - {pred['predicted_home_score']}**")
                st.markdown(f"{pred['away_team']}")
                st.markdown(f"{pred['home_team']}")
                st.markdown(f"Win Probability: {pred['home_win_probability']:.1%}")

            with col2:
                st.markdown("**Spread & Total**")
                spread = pred['predicted_spread']
                if spread > 0:
                    spread_text = f"{pred['home_team']} {spread:+.1f}"
                else:
                    spread_text = f"{pred['away_team']} {abs(spread):+.1f}"

                st.markdown(f"Spread: **{spread_text}**")
                st.markdown(f"Total: **{pred['predicted_total']:.1f}** (O/U)")

            with col3:
                st.markdown("**Result**")
                if pred.get('game_completed'):
                    actual_away = pred['actual_away_score']
                    actual_home = pred['actual_home_score']
                    st.markdown(f"**{actual_away} - {actual_home}**")

                    # Check if prediction was correct
                    predicted_winner = pred['home_team'] if pred['predicted_home_score'] > pred['predicted_away_score'] else pred['away_team']
                    actual_winner = pred['home_team'] if actual_home > actual_away else pred['away_team']

                    if predicted_winner == actual_winner:
                        st.success("✓ Correct prediction!")
                    else:
                        st.error("✗ Incorrect prediction")
                else:
                    st.info("Game not yet played")


def show_database_explorer():
    """Database exploration interface"""
    st.markdown('<p class="main-header">Database Explorer</p>', unsafe_allow_html=True)

    conn = get_db_connection()

    # Table selector
    table = st.selectbox(
        "Select Table",
        ["games", "teams", "team_game_stats", "game_odds"]
    )

    # Query builder
    st.markdown(f"### {table.upper()} Table")

    if table == "games":
        query = """
            SELECT
                g.game_id,
                g.week,
                g.date as game_date,
                ht.name as home_team,
                at.name as away_team,
                g.home_score,
                g.away_score,
                g.completed
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.season = 2024
            ORDER BY g.date DESC
            LIMIT 100
        """
    elif table == "teams":
        query = "SELECT * FROM teams ORDER BY name LIMIT 100"
    elif table == "team_game_stats":
        query = """
            SELECT
                tgs.*,
                t.name as team_name
            FROM team_game_stats tgs
            JOIN teams t ON tgs.team_id = t.team_id
            ORDER BY tgs.game_id DESC
            LIMIT 100
        """
    elif table == "game_odds":
        query = """
            SELECT
                go.*,
                ht.name as home_team,
                at.name as away_team
            FROM game_odds go
            JOIN games g ON go.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            ORDER BY go.odds_id DESC
            LIMIT 100
        """

    try:
        df = pd.read_sql_query(query, conn)
        st.markdown(f"Showing {len(df)} records")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{table}_export.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error querying database: {e}")

    conn.close()


# ============================================================================
# News Aggregator
# ============================================================================

@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_rss_feed(url, source_name, max_items=10):
    """Fetch and parse RSS feed"""
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_items]:
            articles.append({
                'title': entry.get('title', 'No title'),
                'link': entry.get('link', '#'),
                'published': entry.get('published', entry.get('updated', '')),
                'source': source_name,
                'summary': entry.get('summary', '')[:200] + '...' if entry.get('summary') else ''
            })
        return articles
    except Exception as e:
        return []


@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_web_headlines(url, source_name, selector, link_selector=None, max_items=10):
    """Fetch headlines from a webpage using CSS selectors"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = []
        elements = soup.select(selector)[:max_items]

        for elem in elements:
            if link_selector:
                link_elem = elem.select_one(link_selector)
                link = link_elem.get('href', '#') if link_elem else '#'
            else:
                link = elem.get('href', '#') if elem.name == 'a' else elem.find('a').get('href', '#') if elem.find('a') else '#'

            title = elem.get_text(strip=True)

            # Make relative links absolute
            if link.startswith('/'):
                from urllib.parse import urljoin
                link = urljoin(url, link)

            if title and len(title) > 10:  # Filter out empty/short entries
                articles.append({
                    'title': title[:150],
                    'link': link,
                    'source': source_name,
                    'published': '',
                    'summary': ''
                })

        return articles
    except Exception as e:
        return []


def get_all_news():
    """Aggregate news from all sources"""
    all_articles = []

    # ESPN College Football RSS
    espn_cfb = fetch_rss_feed(
        'https://www.espn.com/espn/rss/ncf/news',
        'ESPN CFB',
        max_items=8
    )
    all_articles.extend(espn_cfb)

    # ESPN NFL RSS
    espn_nfl = fetch_rss_feed(
        'https://www.espn.com/espn/rss/nfl/news',
        'ESPN NFL',
        max_items=8
    )
    all_articles.extend(espn_nfl)

    # CBS Sports CFB RSS
    cbs_cfb = fetch_rss_feed(
        'https://www.cbssports.com/rss/headlines/college-football/',
        'CBS Sports CFB',
        max_items=6
    )
    all_articles.extend(cbs_cfb)

    # CBS Sports NFL RSS
    cbs_nfl = fetch_rss_feed(
        'https://www.cbssports.com/rss/headlines/nfl/',
        'CBS Sports NFL',
        max_items=6
    )
    all_articles.extend(cbs_nfl)

    # Yahoo Sports CFB
    yahoo_cfb = fetch_rss_feed(
        'https://sports.yahoo.com/college-football/rss/',
        'Yahoo CFB',
        max_items=5
    )
    all_articles.extend(yahoo_cfb)

    # Bleacher Report (general)
    br_feed = fetch_rss_feed(
        'https://bleacherreport.com/articles/feed',
        'Bleacher Report',
        max_items=5
    )
    all_articles.extend(br_feed)

    return all_articles


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
        ["All Sources", "ESPN CFB", "ESPN NFL", "CBS Sports CFB", "CBS Sports NFL", "Yahoo CFB", "Bleacher Report"]
    )

    # Fetch all news
    with st.spinner("Loading news..."):
        articles = get_all_news()

    if not articles:
        st.warning("Unable to fetch news. Please try again later.")
        return

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
                'Bleacher Report': '#00b2a9'
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

    # Betting lines quick view
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


# ============================================================================
# Main App Logic
# ============================================================================

def main_page():
    """Display main application page"""
    token = st.session_state.token

    # Get user profile
    user = get_current_user(token)
    if not user:
        st.error("Session expired. Please login again.")
        st.session_state.clear()
        st.rerun()
        return

    # Sidebar
    with st.sidebar:
        st.title(f"👤 {user['username']}")
        st.write(f"**{user['full_name']}**")
        st.write(f"_{user['email']}_")

        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")

        # Navigation
        st.markdown("# 🏈 CFB Predictions")
        st.markdown("---")

        # Initialize page in session state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Overview"

        page = st.radio(
            "Navigate",
            ["Overview", "View Predictions", "News", "Database Explorer"]
        )
        st.session_state.current_page = page

        st.markdown("---")

        # User stats
        st.subheader("📊 Quick Stats")
        stats = get_user_stats(token)
        if stats:
            st.metric("Predictions Viewed", stats['total_predictions_viewed'])
            st.metric("This Week", stats['predictions_this_week'])
            if stats['overall_accuracy'] is not None:
                st.metric("Your Accuracy", f"{stats['overall_accuracy']:.1f}%")

    # Load database stats once
    db_stats = load_database_stats()

    # Main content area
    if st.session_state.current_page == "Overview":
        show_overview(user, db_stats)
    elif st.session_state.current_page == "View Predictions":
        show_predictions()
    elif st.session_state.current_page == "News":
        show_news()
    elif st.session_state.current_page == "Database Explorer":
        show_database_explorer()


# ============================================================================
# Entry Point
# ============================================================================

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Main app logic
if st.session_state.logged_in:
    main_page()
else:
    login_page()
