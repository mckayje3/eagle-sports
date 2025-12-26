"""
College Football Prediction Dashboard
Streamlit web interface for viewing predictions, tracking accuracy, and analyzing data
"""
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Page config
st.set_page_config(
    page_title="CFB Prediction Dashboard",
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
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def get_db_connection():
    """Create database connection"""
    return sqlite3.connect('cfb_games.db')


def load_predictions():
    """Load predictions from CSV - prefer results if available"""
    # List of prediction files to try in order of preference
    prediction_files = [
        'predictions_with_results.csv',
        'predicted_scores.csv',
        'enhanced_predictions_week_13.csv',
        'predictions_log.csv'
    ]

    for filename in prediction_files:
        try:
            df = pd.read_csv(filename)
            # Add confidence column from win_probability if not present
            if 'confidence' not in df.columns and 'win_probability' in df.columns:
                df['confidence'] = df['win_probability']
            return df
        except FileNotFoundError:
            continue
        except pd.errors.EmptyDataError:
            continue

    return None


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
    stats['games_with_odds'] = pd.read_sql_query(
        "SELECT COUNT(DISTINCT game_id) as count FROM odds_and_predictions WHERE latest_spread IS NOT NULL", conn
    ).iloc[0]['count']

    conn.close()
    return stats


def main():
    """Main dashboard page"""

    # Initialize session state for navigation
    if 'nav_to' not in st.session_state:
        st.session_state['nav_to'] = None

    # Sidebar navigation
    st.sidebar.markdown("# 🏈 CFB Predictions")
    st.sidebar.markdown("---")

    # Check if we need to navigate to a specific page
    default_page = 0
    pages = ["Overview", "View Predictions", "Make Predictions", "Prediction Tracker",
             "Database Explorer", "Model Performance"]

    if st.session_state['nav_to'] in pages:
        default_page = pages.index(st.session_state['nav_to'])
        st.session_state['nav_to'] = None  # Reset after using

    page = st.sidebar.radio(
        "Navigate",
        pages,
        index=default_page
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")

    # Load quick stats for sidebar
    db_stats = load_database_stats()
    st.sidebar.metric("Games in Database", f"{db_stats['total_games']:,}")
    st.sidebar.metric("Completed Games", f"{db_stats['completed_games']:,}")
    st.sidebar.metric("Teams Tracked", f"{db_stats['total_teams']}")

    # Main content area
    if page == "Overview":
        show_overview(db_stats)
    elif page == "View Predictions":
        show_predictions()
    elif page == "Make Predictions":
        show_make_predictions()
    elif page == "Prediction Tracker":
        show_tracker()
    elif page == "Database Explorer":
        show_database_explorer()
    elif page == "Model Performance":
        show_model_performance()


def show_overview(db_stats):
    """Show overview/home page"""
    st.markdown('<p class="main-header">College Football Prediction System</p>', unsafe_allow_html=True)
    st.markdown("### Dashboard Overview")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Games",
            value=f"{db_stats['total_games']:,}",
            delta="2025 Season"
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
            delta=f"{db_stats['games_with_odds']/db_stats['total_games']*100:.0f}% coverage"
        )

    st.markdown("---")

    # Recent activity
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 System Features")
        st.markdown("""
        **Data Collection:**
        - ESPN game data (868 games)
        - VegasInsider betting odds
        - The Odds API integration
        - Automated daily updates

        **Machine Learning:**
        - Win/Loss Classifier (65% accuracy)
        - Point Spread Predictor (MAE 15.12)
        - 55 enhanced features
        - Prediction tracking system

        **Analysis:**
        - Team performance metrics
        - Against The Spread (ATS) tracking
        - Confidence level evaluation
        - Historical accuracy trends
        """)

    with col2:
        st.markdown("### 🎯 Your Predictions")
        predictions_df = load_predictions()

        if predictions_df is not None and len(predictions_df) > 0:
            total_preds = len(predictions_df)

            # Check if actual_winner column exists (games have been played)
            if 'actual_winner' in predictions_df.columns:
                completed = predictions_df['actual_winner'].notna().sum()
                pending = total_preds - completed

                st.metric("Total Predictions", total_preds)
                st.metric("Completed", completed)
                st.metric("Pending", pending)

                if completed > 0:
                    correct = predictions_df['correct_winner'].sum()
                    accuracy = correct / completed * 100
                    st.metric("Accuracy", f"{accuracy:.1f}%", f"{int(correct)}/{completed} correct")
            else:
                # All predictions are pending (no results yet)
                st.metric("Total Predictions", total_preds)
                st.metric("Completed", 0)
                st.metric("Pending", total_preds)
        else:
            st.info("No predictions logged yet. Go to 'Make Predictions' to start!")

    st.markdown("---")

    # Recent games
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
        ORDER BY g.date DESC
        LIMIT 10
    """, conn)
    conn.close()

    recent_games['result'] = recent_games.apply(
        lambda x: f"{x['home_score']:.0f}-{x['away_score']:.0f}" if x['completed'] else "Pending",
        axis=1
    )

    display_df = recent_games[['week', 'game_date', 'away_team', 'home_team', 'result']]
    display_df.columns = ['Week', 'Date', 'Away Team', 'Home Team', 'Result']

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_predictions():
    """Show all predictions"""
    st.markdown('<p class="main-header">View Predictions</p>', unsafe_allow_html=True)

    # Add refresh button
    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    predictions_df = load_predictions()

    if predictions_df is None or len(predictions_df) == 0:
        st.warning("No predictions found. Run `py predict_scores.py` to generate predictions!")
        return

    # Check which prediction file we loaded
    has_results = 'actual_home_score' in predictions_df.columns
    completed_count = predictions_df['completed'].sum() if 'completed' in predictions_df.columns else 0

    prediction_source = "Deep Learning Score Predictor (PyTorch)"
    if has_results:
        prediction_source = f"Deep Learning Score Predictor with Results ({completed_count} games completed)"

    st.info(f"📊 Showing {len(predictions_df)} predictions from **{prediction_source}**")

    # Show performance summary if results are available
    if has_results and completed_count > 0:
        completed_mask = predictions_df['completed'] == 1

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            win_accuracy = predictions_df.loc[completed_mask, 'correct_winner'].mean()
            st.metric("Win Accuracy", f"{win_accuracy*100:.1f}%")

        with col_b:
            spread_error = predictions_df.loc[completed_mask, 'spread_error'].mean()
            st.metric("Avg Spread Error", f"{spread_error:.1f} pts")

        with col_c:
            total_error = predictions_df.loc[completed_mask, 'total_error'].mean()
            st.metric("Avg Total Error", f"{total_error:.1f} pts")

        with col_d:
            # Compare to Vegas if available
            vegas_mask = completed_mask & predictions_df['vegas_spread'].notna()
            if vegas_mask.sum() > 0:
                vegas_spread_error = predictions_df.loc[vegas_mask, 'vegas_spread_error'].mean()
                vs_vegas = spread_error - vegas_spread_error
                st.metric("vs Vegas Spread", f"{vs_vegas:+.1f} pts",
                         delta_color="inverse")  # Lower is better
            else:
                st.metric("Vegas Data", "N/A")

    # Show summary statistics for score predictions
    if 'predicted_away_score' in predictions_df.columns:
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            avg_total = predictions_df['predicted_total'].mean()
            st.metric("Avg Total Points", f"{avg_total:.1f}")

        with col_b:
            avg_spread = predictions_df['predicted_spread'].abs().mean()
            st.metric("Avg Spread", f"{avg_spread:.1f} pts")

        with col_c:
            close_games = (predictions_df['predicted_spread'].abs() < 7).sum()
            st.metric("Close Games (<7)", close_games)

        with col_d:
            blowouts = (predictions_df['predicted_spread'].abs() > 21).sum()
            st.metric("Blowouts (>21)", blowouts)

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox(
            "Status",
            ["All", "Pending", "Completed"]
        )

    with col2:
        if 'week' in predictions_df.columns:
            weeks = ["All"] + sorted(predictions_df['week'].dropna().unique().tolist())
            week_filter = st.selectbox("Week", weeks)

    with col3:
        if 'confidence' in predictions_df.columns:
            conf_filter = st.selectbox(
                "Confidence",
                ["All", "High (>75%)", "Medium (65-75%)", "Low (<65%)"]
            )

    # Apply filters
    filtered_df = predictions_df.copy()

    if 'actual_winner' in predictions_df.columns:
        if status_filter == "Pending":
            filtered_df = filtered_df[filtered_df['actual_winner'].isna()]
        elif status_filter == "Completed":
            filtered_df = filtered_df[filtered_df['actual_winner'].notna()]
    else:
        # No results yet, all are pending
        if status_filter == "Completed":
            filtered_df = filtered_df.iloc[0:0]  # Empty dataframe

    if week_filter != "All":
        filtered_df = filtered_df[filtered_df['week'] == week_filter]

    if conf_filter == "High (>75%)":
        filtered_df = filtered_df[filtered_df['confidence'] >= 0.75]
    elif conf_filter == "Medium (65-75%)":
        filtered_df = filtered_df[(filtered_df['confidence'] >= 0.65) & (filtered_df['confidence'] < 0.75)]
    elif conf_filter == "Low (<65%)":
        filtered_df = filtered_df[filtered_df['confidence'] < 0.65]

    st.markdown(f"### Showing {len(filtered_df)} predictions")

    # Display predictions
    for idx, pred in filtered_df.iterrows():
        # Format the matchup with neutral site indicator
        neutral = " (Neutral)" if pred.get('neutral_site', 0) == 1 else ""
        matchup = f"Week {pred.get('week', 'N/A')}: {pred['away_team']} @ {pred['home_team']}{neutral}"

        with st.expander(matchup):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Predicted Score**")
                # Show predicted scores if available
                if 'predicted_away_score' in pred and 'predicted_home_score' in pred:
                    away_score = int(pred['predicted_away_score'])
                    home_score = int(pred['predicted_home_score'])
                    st.markdown(f"**{away_score} - {home_score}**")
                    st.markdown(f"{pred['away_team'][:20]}")
                    st.markdown(f"{pred['home_team'][:20]}")
                else:
                    st.markdown(f"Winner: **{pred['predicted_winner']}**")

                # Show confidence
                if 'confidence' in pred:
                    st.markdown(f"Confidence: {pred['confidence']:.1%}")
                if 'home_win_prob' in pred:
                    st.markdown(f"Home Win Prob: {pred['home_win_prob']:.1%}")

            with col2:
                st.markdown("**Spread & Total**")
                # Show predicted spread with margin of error
                if 'predicted_spread' in pred and pd.notna(pred['predicted_spread']):
                    spread = pred['predicted_spread']
                    if spread > 0:
                        spread_text = f"**{pred['home_team'][:20]} {spread:+.1f}**"
                    else:
                        spread_text = f"**{pred['away_team'][:20]} {abs(spread):+.1f}**"

                    # Add margin of error if available
                    if 'spread_margin_error' in pred and pd.notna(pred['spread_margin_error']):
                        margin = pred['spread_margin_error']
                        st.markdown(f"Spread: {spread_text}")
                        st.markdown(f"_Margin: ±{margin:.1f} pts_")
                        # Show confidence interval
                        if 'spread_confidence_low' in pred and 'spread_confidence_high' in pred:
                            low = pred['spread_confidence_low']
                            high = pred['spread_confidence_high']
                            st.markdown(f"_Range: {low:.1f} to {high:+.1f}_")
                    else:
                        st.markdown(f"Spread: {spread_text}")
                elif pd.notna(pred.get('spread')):
                    st.markdown(f"Vegas Spread: {pred['spread']}")

                st.markdown("")  # Add spacing

                # Show predicted total with margin of error
                if 'predicted_total' in pred and pd.notna(pred['predicted_total']):
                    total = pred['predicted_total']
                    st.markdown(f"Total: **{total:.1f}** (O/U)")

                    # Add margin of error if available
                    if 'total_margin_error' in pred and pd.notna(pred['total_margin_error']):
                        margin = pred['total_margin_error']
                        st.markdown(f"_Margin: ±{margin:.1f} pts_")
                        # Show confidence interval
                        if 'total_confidence_low' in pred and 'total_confidence_high' in pred:
                            low = pred['total_confidence_low']
                            high = pred['total_confidence_high']
                            st.markdown(f"_Range: {low:.1f} to {high:.1f}_")

                # Show implied spread if available
                if pd.notna(pred.get('implied_spread')):
                    st.markdown(f"Model Spread: {pred['implied_spread']:.1f}")

            with col3:
                st.markdown("**Result**")
                if pd.notna(pred.get('actual_winner')):
                    # Game has been played
                    result = "✓ CORRECT" if pred.get('correct_winner') == 1 else "✗ WRONG"
                    st.markdown(f"Actual: **{pred['actual_winner']}**")

                    # Show actual scores
                    if pd.notna(pred.get('actual_home_score')) and pd.notna(pred.get('actual_away_score')):
                        actual_away = int(pred['actual_away_score'])
                        actual_home = int(pred['actual_home_score'])
                        st.markdown(f"**{actual_away} - {actual_home}**")

                    # Show prediction accuracy
                    if pred.get('correct_winner') == 1:
                        st.success(result)
                    else:
                        st.error(result)

                    # Show errors
                    if pd.notna(pred.get('spread_error')):
                        st.markdown(f"_Spread Error: {pred['spread_error']:.1f} pts_")
                    if pd.notna(pred.get('total_error')):
                        st.markdown(f"_Total Error: {pred['total_error']:.1f} pts_")

                    # Compare to Vegas if available
                    if pd.notna(pred.get('vegas_spread')):
                        st.markdown("")
                        st.markdown("**vs Vegas:**")
                        if pd.notna(pred.get('vegas_spread_error')):
                            model_better = pred['spread_error'] < pred['vegas_spread_error']
                            indicator = "✓" if model_better else "✗"
                            st.markdown(f"{indicator} Model: {pred['spread_error']:.1f} pts")
                            st.markdown(f"Vegas: {pred['vegas_spread_error']:.1f} pts")
                else:
                    st.info("Pending")


def show_make_predictions():
    """Interface for making new predictions"""
    st.markdown('<p class="main-header">Make New Predictions</p>', unsafe_allow_html=True)

    st.info("This page allows you to predict upcoming games. The models will be loaded and predictions generated.")

    # Show upcoming games
    conn = get_db_connection()
    upcoming_games = pd.read_sql_query("""
        SELECT
            g.game_id,
            g.week,
            g.date as game_date,
            ht.name as home_team,
            at.name as away_team,
            g.neutral_site
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.completed = 0
        ORDER BY g.date, g.game_id
        LIMIT 50
    """, conn)
    conn.close()

    if len(upcoming_games) == 0:
        st.warning("No upcoming games found in database.")
        return

    st.markdown(f"### {len(upcoming_games)} Upcoming Games")

    # Week filter
    weeks = sorted(upcoming_games['week'].unique())
    selected_week = st.selectbox("Select Week", weeks)

    week_games = upcoming_games[upcoming_games['week'] == selected_week]

    st.dataframe(
        week_games[['game_date', 'away_team', 'home_team', 'neutral_site']],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Buttons for prediction
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎯 Predict All Week Games", use_container_width=True):
            with st.spinner(f'Generating predictions for Week {selected_week}...'):
                try:
                    # Import and run prediction system
                    import sys
                    sys.path.insert(0, '.')
                    from predict_with_enhanced_models import EnhancedPredictionSystem

                    system = EnhancedPredictionSystem()
                    predictions = system.predict_week(selected_week)

                    if predictions is not None and len(predictions) > 0:
                        st.success(f"✅ Generated {len(predictions)} predictions for Week {selected_week}!")
                        st.dataframe(predictions[['home_team', 'away_team', 'predicted_winner', 'confidence', 'predicted_spread']],
                                   use_container_width=True, hide_index=True)

                        # Save predictions
                        output_file = f'enhanced_predictions_week_{selected_week}.csv'
                        predictions.to_csv(output_file, index=False)
                        st.info(f"Saved to {output_file}")
                    else:
                        st.warning("No predictions generated")
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    st.info("Make sure models are trained. Run: py cfb_predictor_v2.py")

    with col2:
        if st.button("📊 View Model Details", use_container_width=True):
            st.session_state['nav_to'] = 'Model Performance'
            st.rerun()


def show_tracker():
    """Show prediction tracking and accuracy"""
    st.markdown('<p class="main-header">Prediction Tracker</p>', unsafe_allow_html=True)

    predictions_df = load_predictions()

    if predictions_df is None or len(predictions_df) == 0:
        st.warning("No predictions to track yet.")
        return

    # Calculate statistics
    if 'actual_winner' in predictions_df.columns:
        completed_df = predictions_df[predictions_df['actual_winner'].notna()].copy()
    else:
        completed_df = pd.DataFrame()  # Empty dataframe if no results yet

    if len(completed_df) == 0:
        st.info("No completed games yet. Predictions will be evaluated after games finish.")

        # Show pending predictions
        if 'actual_winner' in predictions_df.columns:
            pending_df = predictions_df[predictions_df['actual_winner'].isna()]
        else:
            pending_df = predictions_df  # All predictions are pending
        st.markdown(f"### {len(pending_df)} Pending Predictions")
        st.dataframe(
            pending_df[['week', 'away_team', 'home_team', 'predicted_winner', 'confidence']],
            use_container_width=True,
            hide_index=True
        )
        return

    # Overall stats
    total_preds = len(predictions_df)
    completed = len(completed_df)
    pending = total_preds - completed
    correct = completed_df['correct_winner'].sum()
    accuracy = correct / completed * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", total_preds)
    with col2:
        st.metric("Completed", completed)
    with col3:
        st.metric("Correct", int(correct))
    with col4:
        st.metric("Accuracy", f"{accuracy:.1f}%")

    st.markdown("---")

    # Accuracy by confidence level
    if 'confidence' in completed_df.columns:
        st.markdown("### Accuracy by Confidence Level")

        high_conf = completed_df[completed_df['confidence'] >= 0.75]
        med_conf = completed_df[(completed_df['confidence'] >= 0.65) & (completed_df['confidence'] < 0.75)]
        low_conf = completed_df[completed_df['confidence'] < 0.65]

        conf_data = []
        for name, df_subset in [("High (>75%)", high_conf), ("Medium (65-75%)", med_conf), ("Low (<65%)", low_conf)]:
            if len(df_subset) > 0:
                acc = df_subset['correct_winner'].mean() * 100
                conf_data.append({
                    'Confidence Level': name,
                    'Games': len(df_subset),
                    'Correct': int(df_subset['correct_winner'].sum()),
                    'Accuracy': f"{acc:.1f}%"
                })

        if conf_data:
            conf_df = pd.DataFrame(conf_data)
            st.dataframe(conf_df, use_container_width=True, hide_index=True)

            # Chart
            fig = px.bar(
                conf_df,
                x='Confidence Level',
                y='Games',
                color='Accuracy',
                title='Predictions by Confidence Level'
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Home vs Away accuracy
    st.markdown("### Home vs Away Predictions")

    col1, col2 = st.columns(2)

    home_preds = completed_df[completed_df['predicted_winner'] == completed_df['home_team']]
    away_preds = completed_df[completed_df['predicted_winner'] == completed_df['away_team']]

    with col1:
        st.markdown("**Home Team Predicted**")
        if len(home_preds) > 0:
            home_acc = home_preds['correct_winner'].mean() * 100
            st.metric("Games", len(home_preds))
            st.metric("Accuracy", f"{home_acc:.1f}%")

    with col2:
        st.markdown("**Away Team Predicted**")
        if len(away_preds) > 0:
            away_acc = away_preds['correct_winner'].mean() * 100
            st.metric("Games", len(away_preds))
            st.metric("Accuracy", f"{away_acc:.1f}%")


def show_database_explorer():
    """Database exploration interface"""
    st.markdown('<p class="main-header">Database Explorer</p>', unsafe_allow_html=True)

    conn = get_db_connection()

    # Table selector
    table = st.selectbox(
        "Select Table",
        ["games", "teams", "team_game_stats", "odds_and_predictions"]
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
    elif table == "odds_and_predictions":
        query = """
            SELECT
                op.game_id,
                op.source,
                op.opening_spread,
                op.latest_spread,
                op.opening_total,
                op.latest_total,
                op.predicted_home_score,
                op.predicted_away_score,
                ht.name as home_team,
                at.name as away_team
            FROM odds_and_predictions op
            JOIN games g ON op.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            ORDER BY op.game_id DESC
            LIMIT 100
        """

    df = pd.read_sql_query(query, conn)
    conn.close()

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


def show_model_performance():
    """Show model performance metrics and visualizations"""
    st.markdown('<p class="main-header">Model Performance</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Win/Loss Classifier")
        st.markdown("""
        **Architecture:**
        - Input: 55 features (enhanced)
        - Hidden: 64 → 32 → 16 neurons
        - Output: Binary classification (sigmoid)

        **Performance:**
        - Training Accuracy: ~65%
        - Features: Team stats, form, head-to-head

        **Model File:** `cfb_model.keras`
        """)

        # Check if training history image exists
        if os.path.exists('training_history.png'):
            st.image('training_history.png', caption='Win/Loss Model Training')

    with col2:
        st.markdown("### Point Spread Predictor")
        st.markdown("""
        **Architecture:**
        - Input: 26 key features
        - Hidden: 128 → 64 → 32 → 16 neurons
        - Output: Point differential (linear)

        **Performance:**
        - Mean Absolute Error: 15.12 points
        - Within 7 points: 33%
        - Within 14 points: 55%

        **Model File:** `spread_model.keras`
        """)

        # Check if spread images exist
        if os.path.exists('spread_training_history.png'):
            st.image('spread_training_history.png', caption='Spread Model Training')

    st.markdown("---")

    # Feature importance (if we have it)
    st.markdown("### Feature Set")

    features_info = {
        'Category': ['Basic', 'Scoring', 'Yards', 'Efficiency', 'Form'],
        'Features': [
            'Week, Neutral Site, H2H',
            'Points Scored/Allowed, Point Differential',
            'Total/Passing/Rushing Yards',
            'Turnovers, Third Down %, Penalties',
            'Win %, Recent Form'
        ],
        'Count': [3, 12, 15, 10, 15]
    }

    features_df = pd.DataFrame(features_info)
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    # Show spread prediction visualization if available
    if os.path.exists('spread_predictions.png'):
        st.markdown("---")
        st.markdown("### Spread Prediction Accuracy")
        st.image('spread_predictions.png', caption='Predicted vs Actual Spreads', use_column_width=True)


if __name__ == "__main__":
    main()
