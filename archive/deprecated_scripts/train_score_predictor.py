"""
Train Comprehensive Score Predictor using Deep Learning

This script trains three models to predict:
1. Win/Loss (classification)
2. Point Differential / Spread (regression)
3. Total Points / Over-Under (regression)

Models are combined to predict actual game scores.
"""

from cfb_score_predictor import CFBScorePredictor
from core.utils import set_seed
import matplotlib.pyplot as plt
import numpy as np


def plot_training_histories(histories, save_dir='models'):
    """Plot training histories for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Win/Loss Classifier
    ax = axes[0]
    ax.plot(histories['win']['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(histories['win']['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Win/Loss Classifier', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Spread Predictor
    ax = axes[1]
    ax.plot(histories['spread']['train_loss'], label='Train Loss (MSE)', linewidth=2)
    ax.plot(histories['spread']['val_loss'], label='Val Loss (MSE)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Point Differential (Spread) Predictor', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Total Points Predictor
    ax = axes[2]
    ax.plot(histories['total']['train_loss'], label='Train Loss (MSE)', linewidth=2)
    ax.plot(histories['total']['val_loss'], label='Val Loss (MSE)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Total Points (Over/Under) Predictor', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'{save_dir}/training_histories.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining histories plot saved to {save_path}")
    plt.show()


def evaluate_predictions(predictor, X, y_win, y_spread, y_total, df):
    """Evaluate predictions on validation data"""
    print("\n" + "="*80)
    print("EVALUATING PREDICTIONS ON VALIDATION DATA")
    print("="*80 + "\n")

    # Make predictions
    predictions = predictor.predict_scores(X)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

    # Win/Loss accuracy
    win_pred = (predictions['home_win_prob'] > 0.5).astype(int)
    win_accuracy = accuracy_score(y_win, win_pred)

    # Spread MAE
    spread_mae = mean_absolute_error(y_spread, predictions['spread'])
    spread_rmse = np.sqrt(mean_squared_error(y_spread, predictions['spread']))

    # Total MAE
    total_mae = mean_absolute_error(y_total, predictions['total'])
    total_rmse = np.sqrt(mean_squared_error(y_total, predictions['total']))

    # Actual scores MAE
    home_score_mae = mean_absolute_error(df['home_score'].values, predictions['home_score'])
    away_score_mae = mean_absolute_error(df['away_score'].values, predictions['away_score'])

    print("Performance Metrics:")
    print("-" * 80)
    print(f"Win/Loss Accuracy:        {win_accuracy*100:.2f}%")
    print(f"\nSpread Prediction:")
    print(f"  MAE:                    {spread_mae:.2f} points")
    print(f"  RMSE:                   {spread_rmse:.2f} points")
    print(f"\nTotal Points Prediction:")
    print(f"  MAE:                    {total_mae:.2f} points")
    print(f"  RMSE:                   {total_rmse:.2f} points")
    print(f"\nActual Score Prediction:")
    print(f"  Home Score MAE:         {home_score_mae:.2f} points")
    print(f"  Away Score MAE:         {away_score_mae:.2f} points")
    print(f"  Combined Score MAE:     {(home_score_mae + away_score_mae)/2:.2f} points")
    print("-" * 80)

    # Show sample predictions
    print("\nSample Predictions (First 10 games):")
    print("-" * 80)
    print(f"{'Game':<30} {'Actual':<15} {'Predicted':<15} {'Spread':<10} {'Total':<10}")
    print("-" * 80)

    for i in range(min(10, len(df))):
        game_str = f"{df.iloc[i].get('away_team', 'Away')} @ {df.iloc[i].get('home_team', 'Home')}"[:30]
        actual_str = f"{int(df.iloc[i]['away_score'])}-{int(df.iloc[i]['home_score'])}"
        predicted_str = f"{predictions['away_score'][i]}-{predictions['home_score'][i]}"
        spread_str = f"{predictions['spread'][i]:+.1f}"
        total_str = f"{predictions['total'][i]:.1f}"

        print(f"{game_str:<30} {actual_str:<15} {predicted_str:<15} {spread_str:<10} {total_str:<10}")

    print("-" * 80 + "\n")


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("COLLEGE FOOTBALL SCORE PREDICTOR - TRAINING PIPELINE")
    print("Using Deep Learning to Predict Actual Game Scores")
    print("="*80 + "\n")

    # Set random seed for reproducibility
    set_seed(42)

    # Create predictor
    predictor = CFBScorePredictor()

    # Load data
    X, y_win, y_spread, y_total, df = predictor.load_data(
        csv_file='ml_features_v2_2024.csv',
        min_week=4
    )

    # Train all models
    histories = predictor.train_all_models(
        X=X,
        y_win=y_win,
        y_spread=y_spread,
        y_total=y_total,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        patience=15,
        save_dir='models',
        verbose=True
    )

    # Save models
    predictor.save('models')

    # Plot training histories
    plot_training_histories(histories, save_dir='models')

    # Evaluate on training data (for demonstration)
    evaluate_predictions(predictor, X, y_win, y_spread, y_total, df)

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nModels saved to:")
    print("  - models/win_model.pt")
    print("  - models/spread_model.pt")
    print("  - models/total_model.pt")
    print("  - models/score_predictor_data.pkl")
    print("\nTraining plot saved to:")
    print("  - models/training_histories.png")
    print("\nYou can now use these models to predict actual game scores!")
    print("Run: py predict_scores.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
