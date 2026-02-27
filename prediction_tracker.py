"""
Prediction Tracking System
Compares predictions against actual results and calculates accuracy
"""
import pandas as pd
import sqlite3
from datetime import datetime
import json


class PredictionTracker:
    """Track and evaluate prediction accuracy"""

    def __init__(self, db_path='cfb_games.db'):
        self.db_path = db_path

    def save_prediction(self, prediction_data, filename='predictions_log.csv'):
        """
        Save a prediction to tracking file

        Args:
            prediction_data: Dict with prediction details
            filename: CSV file to append to
        """
        df_new = pd.DataFrame([prediction_data])

        try:
            # Append to existing file
            df_existing = pd.read_csv(filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(filename, index=False)
        except FileNotFoundError:
            # Create new file
            df_new.to_csv(filename, index=False)

        print(f"Prediction logged to {filename}")

    def save_predictions_batch(self, predictions_list, filename='predictions_log.csv'):
        """Save multiple predictions at once"""
        df_new = pd.DataFrame(predictions_list)

        try:
            df_existing = pd.read_csv(filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Remove duplicates based on game_id
            df_combined = df_combined.drop_duplicates(subset=['game_id'], keep='last')
            df_combined.to_csv(filename, index=False)
        except FileNotFoundError:
            df_new.to_csv(filename, index=False)

        print(f"Saved {len(predictions_list)} predictions to {filename}")

    def update_with_results(self, filename='predictions_log.csv'):
        """
        Update predictions with actual game results from database

        Returns:
            DataFrame with updated results
        """
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"No predictions file found: {filename}")
            return None

        print(f"Loaded {len(df)} predictions from {filename}")

        conn = sqlite3.connect(self.db_path)

        # Get actual results for each game
        for idx, pred in df.iterrows():
            if pd.notna(pred.get('actual_home_score')):
                # Already has results
                continue

            game_id = pred['game_id']

            query = """
                SELECT
                    home_score,
                    away_score,
                    completed
                FROM games
                WHERE game_id = ?
            """

            result = pd.read_sql_query(query, conn, params=[game_id])

            if not result.empty and result.iloc[0]['completed']:
                actual_home = result.iloc[0]['home_score']
                actual_away = result.iloc[0]['away_score']

                df.at[idx, 'actual_home_score'] = actual_home
                df.at[idx, 'actual_away_score'] = actual_away
                df.at[idx, 'actual_winner'] = pred['home_team'] if actual_home > actual_away else pred['away_team']
                df.at[idx, 'actual_point_diff'] = actual_home - actual_away

                # Check if prediction was correct
                df.at[idx, 'correct_winner'] = 1 if pred['predicted_winner'] == df.at[idx, 'actual_winner'] else 0

                # Check spread if available
                if pd.notna(pred.get('spread')):
                    actual_vs_spread = actual_home - actual_away + pred['spread']
                    df.at[idx, 'covered_spread'] = 1 if actual_vs_spread > 0 else 0

        conn.close()

        # Save updated file
        df.to_csv(filename, index=False)
        print(f"Updated results saved to {filename}")

        return df

    def calculate_accuracy(self, filename='predictions_log.csv'):
        """
        Calculate prediction accuracy metrics

        Returns:
            Dictionary with accuracy statistics
        """
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"No predictions file found: {filename}")
            return None

        # Filter to games with results
        df_completed = df[pd.notna(df['actual_winner'])].copy()

        if df_completed.empty:
            print("No completed games yet to evaluate")
            return None

        stats = {
            'total_predictions': len(df),
            'completed_games': len(df_completed),
            'pending_games': len(df) - len(df_completed),
        }

        # Win/Loss accuracy
        correct_winners = df_completed['correct_winner'].sum()
        stats['winner_accuracy'] = correct_winners / len(df_completed)
        stats['winner_correct'] = int(correct_winners)
        stats['winner_incorrect'] = len(df_completed) - int(correct_winners)

        # Confidence levels
        if 'confidence' in df_completed.columns:
            high_conf = df_completed[df_completed['confidence'] >= 0.75]
            med_conf = df_completed[(df_completed['confidence'] >= 0.65) & (df_completed['confidence'] < 0.75)]
            low_conf = df_completed[df_completed['confidence'] < 0.65]

            stats['high_confidence_count'] = len(high_conf)
            stats['high_confidence_accuracy'] = high_conf['correct_winner'].mean() if len(high_conf) > 0 else 0

            stats['med_confidence_count'] = len(med_conf)
            stats['med_confidence_accuracy'] = med_conf['correct_winner'].mean() if len(med_conf) > 0 else 0

            stats['low_confidence_count'] = len(low_conf)
            stats['low_confidence_accuracy'] = low_conf['correct_winner'].mean() if len(low_conf) > 0 else 0

        # Spread accuracy (if applicable)
        df_with_spread = df_completed[pd.notna(df_completed.get('spread'))].copy()
        if len(df_with_spread) > 0:
            stats['ats_predictions'] = len(df_with_spread)
            stats['ats_correct'] = int(df_with_spread['covered_spread'].sum()) if 'covered_spread' in df_with_spread.columns else 0
            stats['ats_accuracy'] = stats['ats_correct'] / stats['ats_predictions']

        # Home/Away breakdown
        home_predicted = df_completed[df_completed['predicted_winner'] == df_completed['home_team']]
        away_predicted = df_completed[df_completed['predicted_winner'] == df_completed['away_team']]

        stats['home_predictions'] = len(home_predicted)
        stats['home_accuracy'] = home_predicted['correct_winner'].mean() if len(home_predicted) > 0 else 0

        stats['away_predictions'] = len(away_predicted)
        stats['away_accuracy'] = away_predicted['correct_winner'].mean() if len(away_predicted) > 0 else 0

        return stats

    def print_report(self, filename='predictions_log.csv'):
        """Print detailed accuracy report"""
        # Update with latest results
        df = self.update_with_results(filename)

        if df is None:
            return

        # Calculate stats
        stats = self.calculate_accuracy(filename)

        if stats is None:
            return

        print("\n" + "="*80)
        print("PREDICTION ACCURACY REPORT")
        print("="*80 + "\n")

        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"  Completed: {stats['completed_games']}")
        print(f"  Pending: {stats['pending_games']}\n")

        print("WINNER PREDICTION ACCURACY:")
        print(f"  Correct: {stats['winner_correct']}/{stats['completed_games']}")
        print(f"  Accuracy: {stats['winner_accuracy']:.1%}\n")

        if 'high_confidence_count' in stats:
            print("ACCURACY BY CONFIDENCE LEVEL:")
            print(f"  High (>75%): {stats['high_confidence_accuracy']:.1%} ({stats['high_confidence_count']} games)")
            print(f"  Medium (65-75%): {stats['med_confidence_accuracy']:.1%} ({stats['med_confidence_count']} games)")
            print(f"  Low (<65%): {stats['low_confidence_accuracy']:.1%} ({stats['low_confidence_count']} games)\n")

        print("HOME/AWAY BREAKDOWN:")
        print(f"  Home team predicted: {stats['home_accuracy']:.1%} ({stats['home_predictions']} games)")
        print(f"  Away team predicted: {stats['away_accuracy']:.1%} ({stats['away_predictions']} games)\n")

        if 'ats_predictions' in stats:
            print("AGAINST THE SPREAD (ATS):")
            print(f"  Correct: {stats['ats_correct']}/{stats['ats_predictions']}")
            print(f"  ATS Accuracy: {stats['ats_accuracy']:.1%}\n")

        # Show recent predictions
        df_recent = df.tail(10)
        print("="*80)
        print("RECENT PREDICTIONS:")
        print("="*80 + "\n")

        for _, pred in df_recent.iterrows():
            print(f"{pred['away_team']} @ {pred['home_team']}")
            print(f"  Predicted: {pred['predicted_winner']} ({pred.get('confidence', 0):.1%})")

            if pd.notna(pred.get('actual_winner')):
                result = "CORRECT" if pred['correct_winner'] == 1 else "WRONG"
                print(f"  Actual: {pred['actual_winner']} - {result}")
                print(f"  Score: {pred['actual_home_score']:.0f}-{pred['actual_away_score']:.0f}")
            else:
                print(f"  Status: Pending")
            print()

        print("="*80 + "\n")

        # Save summary to JSON
        with open('prediction_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print("Statistics saved to prediction_stats.json\n")


def log_predictions_from_csv(input_file, output_file='predictions_log.csv'):
    """
    Convert prediction CSV to tracking format

    Args:
        input_file: CSV with predictions (e.g., specific_game_predictions.csv)
        output_file: Tracking log file
    """
    tracker = PredictionTracker()

    df = pd.read_csv(input_file)

    predictions = []
    for _, pred in df.iterrows():
        prediction_data = {
            'game_id': pred.get('game_id'),
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'week': pred.get('week'),
            'home_team': pred.get('home_team'),
            'away_team': pred.get('away_team'),
            'predicted_winner': pred.get('predicted_winner'),
            'home_win_prob': pred.get('home_win_prob'),
            'confidence': pred.get('confidence'),
            'spread': pred.get('spread'),
            'implied_spread': pred.get('implied_spread'),
            'model_version': 'v1.0'
        }
        predictions.append(prediction_data)

    tracker.save_predictions_batch(predictions, output_file)
    print(f"Logged {len(predictions)} predictions")


if __name__ == '__main__':
    # Example: Log recent predictions and check accuracy
    tracker = PredictionTracker()

    # If you have predictions from earlier
    try:
        print("Logging predictions from specific_game_predictions.csv...")
        log_predictions_from_csv('specific_game_predictions.csv')
    except FileNotFoundError:
        print("No specific_game_predictions.csv found")

    # Generate report
    tracker.print_report()
