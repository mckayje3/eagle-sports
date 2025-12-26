"""
Compare Deep-Eagle Model Predictions to Vegas Lines
Identifies games where our model disagrees with Vegas and evaluates accuracy
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, r'C:\Users\jbeast\documents\coding\deep')

import sqlite3
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from core import LSTMModel


class VegasComparison:
    """Compare Deep-Eagle predictions to Vegas lines"""

    def __init__(self, db_path='cfb_games.db', models_dir='models'):
        self.db_path = db_path
        self.models_dir = Path(models_dir)

        self.spread_model = None
        self.total_model = None
        self.scaler = None
        self.feature_columns = None

        self.load_models()

    def load_models(self):
        """Load trained models and scaler"""
        print("Loading Deep-Eagle models...")

        # Load scaler and feature columns
        with open(self.models_dir / 'cfb_scaler.pkl', 'rb') as f:
            scaler_data = pickle.load(f)
            self.scaler = scaler_data['scaler']
            self.feature_columns = scaler_data['feature_columns']

        # Load spread model
        self.spread_model = LSTMModel(
            input_dim=len(self.feature_columns),
            hidden_dim=128,
            output_dim=1,
            num_layers=2,
            dropout=0.2
        )
        spread_checkpoint = torch.load(self.models_dir / 'cfb_spread_best.pth', weights_only=True)
        if 'model_state_dict' in spread_checkpoint:
            self.spread_model.load_state_dict(spread_checkpoint['model_state_dict'])
        else:
            self.spread_model.load_state_dict(spread_checkpoint)
        self.spread_model.eval()

        # Load total model
        self.total_model = LSTMModel(
            input_dim=len(self.feature_columns),
            hidden_dim=128,
            output_dim=1,
            num_layers=2,
            dropout=0.2
        )
        total_checkpoint = torch.load(self.models_dir / 'cfb_total_best.pth', weights_only=True)
        if 'model_state_dict' in total_checkpoint:
            self.total_model.load_state_dict(total_checkpoint['model_state_dict'])
        else:
            self.total_model.load_state_dict(total_checkpoint)
        self.total_model.eval()

        print(f"✅ Models loaded successfully")
        print(f"   Features: {len(self.feature_columns)}")

    def get_games_with_odds(self, season=2025, weeks=None):
        """
        Get games that have both model predictions available and Vegas odds

        Args:
            season: Season year
            weeks: List of weeks to analyze (default: all completed weeks)

        Returns:
            DataFrame with games, odds, and actuals
        """
        conn = sqlite3.connect(self.db_path)

        week_filter = ""
        if weeks:
            week_list = ','.join(map(str, weeks))
            week_filter = f"AND g.week IN ({week_list})"

        query = f'''
            SELECT
                g.game_id,
                g.season,
                g.week,
                g.date,
                g.home_team_id,
                g.away_team_id,
                ht.name as home_team,
                at.name as away_team,
                g.home_score,
                g.away_score,
                (g.home_score - g.away_score) as actual_spread,
                (g.home_score + g.away_score) as actual_total,
                op.latest_spread as vegas_spread,
                op.current_total as vegas_total,
                op.current_moneyline_home as home_ml,
                op.current_moneyline_away as away_ml
            FROM games g
            LEFT JOIN teams ht ON g.home_team_id = ht.team_id
            LEFT JOIN teams at ON g.away_team_id = at.team_id
            LEFT JOIN odds_and_predictions op ON g.game_id = op.game_id
            WHERE g.season = {season}
                AND g.completed = 1
                {week_filter}
                AND op.latest_spread IS NOT NULL
            ORDER BY g.week, g.date
        '''

        df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"\\n📊 Found {len(df)} games with Vegas lines")
        return df

    def compare_predictions(self, weeks=None):
        """
        Compare Deep-Eagle predictions to Vegas lines for completed games

        Args:
            weeks: List of weeks to analyze (default: weeks 11-13 validation set)

        Returns:
            DataFrame with predictions and comparisons
        """
        if weeks is None:
            weeks = [11, 12, 13]  # Validation weeks

        # Get games with odds
        games_df = self.get_games_with_odds(season=2025, weeks=weeks)

        if len(games_df) == 0:
            print("⚠️  No games found with Vegas lines")
            return None

        print(f"\\n🔮 Comparing predictions for weeks {weeks}...")

        # TODO: Load team features and generate predictions
        # For now, return the games dataframe
        # This will be completed once we have the feature engineering pipeline

        return games_df

    def analyze_performance(self, df):
        """
        Analyze how well predictions performed vs Vegas

        Args:
            df: DataFrame with predictions and actuals

        Returns:
            Performance metrics dictionary
        """
        metrics = {}

        # Spread accuracy
        if 'predicted_spread' in df.columns:
            spread_mae = np.abs(df['predicted_spread'] - df['actual_spread']).mean()
            vegas_spread_mae = np.abs(df['vegas_spread'] - df['actual_spread']).mean()

            metrics['spread_mae'] = spread_mae
            metrics['vegas_spread_mae'] = vegas_spread_mae
            metrics['spread_improvement'] = vegas_spread_mae - spread_mae

        # Total accuracy
        if 'predicted_total' in df.columns:
            total_mae = np.abs(df['predicted_total'] - df['actual_total']).mean()
            vegas_total_mae = np.abs(df['vegas_total'] - df['actual_total']).mean()

            metrics['total_mae'] = total_mae
            metrics['vegas_total_mae'] = vegas_total_mae
            metrics['total_improvement'] = vegas_total_mae - total_mae

        # Against the spread (ATS) performance
        if 'predicted_spread' in df.columns:
            df['model_ats'] = (df['predicted_spread'] > df['vegas_spread']) == (df['actual_spread'] > df['vegas_spread'])
            metrics['model_ats_pct'] = df['model_ats'].mean()

        return metrics

    def print_summary(self, metrics):
        """Print comparison summary"""
        print("\\n" + "=" * 80)
        print("DEEP-EAGLE VS VEGAS COMPARISON")
        print("=" * 80)

        if 'spread_mae' in metrics:
            print(f"\\n📏 Spread Accuracy:")
            print(f"   Deep-Eagle MAE: {metrics['spread_mae']:.2f} points")
            print(f"   Vegas MAE: {metrics['vegas_spread_mae']:.2f} points")
            if metrics['spread_improvement'] > 0:
                print(f"   ✅ Deep-Eagle better by {metrics['spread_improvement']:.2f} points!")
            else:
                print(f"   ⚠️  Vegas better by {abs(metrics['spread_improvement']):.2f} points")

        if 'total_mae' in metrics:
            print(f"\\n🎯 Total Accuracy:")
            print(f"   Deep-Eagle MAE: {metrics['total_mae']:.2f} points")
            print(f"   Vegas MAE: {metrics['vegas_total_mae']:.2f} points")
            if metrics['total_improvement'] > 0:
                print(f"   ✅ Deep-Eagle better by {metrics['total_improvement']:.2f} points!")
            else:
                print(f"   ⚠️  Vegas better by {abs(metrics['total_improvement']):.2f} points")

        if 'model_ats_pct' in metrics:
            print(f"\\n💰 Against The Spread (ATS):")
            print(f"   Model ATS: {metrics['model_ats_pct']*100:.1f}%")
            print(f"   Break-even: 52.4% (needed to profit)")
            if metrics['model_ats_pct'] > 0.524:
                print(f"   ✅ Profitable!")
            else:
                print(f"   ⚠️  Below break-even")

        print("=" * 80)


def main():
    """Compare Deep-Eagle to Vegas lines"""
    print("=" * 80)
    print("DEEP-EAGLE VS VEGAS LINE COMPARISON")
    print("=" * 80)

    # Check if models exist
    if not Path('models/cfb_spread_best.pth').exists():
        print("\\n❌ Models not found. Please train models first:")
        print("   py train_deep_eagle_cfb.py")
        return

    # Create comparison
    comparison = VegasComparison()

    # Compare validation weeks
    games_df = comparison.compare_predictions(weeks=[11, 12, 13])

    if games_df is not None:
        print(f"\\n✅ Loaded {len(games_df)} games")
        print(f"\\nSample games:")
        print(games_df[['week', 'home_team', 'away_team', 'vegas_spread', 'actual_spread']].head(10))

        # TODO: Add predictions and full analysis
        # This requires integrating with the feature engineering pipeline
        print(f"\\n⚠️  Full comparison implementation pending")
        print(f"   Need to integrate feature engineering for predictions")


if __name__ == '__main__':
    main()
