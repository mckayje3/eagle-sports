"""
Train College Football Predictor using PyTorch and Deep Framework

This script demonstrates the new architecture using the core deep learning module.
"""

from cfb_predictor_pytorch import CFBPredictorPyTorch
from core.utils import set_seed
import matplotlib.pyplot as plt


def train_win_predictor():
    """Train a win/loss classifier"""
    print("\n" + "="*60)
    print("Training Win/Loss Classifier")
    print("="*60 + "\n")

    # Set random seed for reproducibility
    set_seed(42)

    # Create predictor
    predictor = CFBPredictorPyTorch(
        model_type='feedforward',
        task='classification'
    )

    # Load data
    X, y_win, y_spread, df = predictor.load_data(
        csv_file='ml_features_v2_2025.csv',
        min_week=4
    )

    # Train model
    history = predictor.train(
        X=X,
        y=y_win,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        patience=15,
        save_path='models/cfb_win_predictor_pytorch.pt',
        verbose=True
    )

    # Save model
    predictor.save('models/cfb_win_predictor_pytorch.pt')

    print("\n" + "="*60)
    print("Win/Loss Classifier Training Complete!")
    print("="*60 + "\n")

    return predictor, history


def train_spread_predictor():
    """Train a point spread regressor"""
    print("\n" + "="*60)
    print("Training Point Spread Regressor")
    print("="*60 + "\n")

    # Set random seed for reproducibility
    set_seed(42)

    # Create predictor
    predictor = CFBPredictorPyTorch(
        model_type='feedforward',
        task='regression'
    )

    # Load data
    X, y_win, y_spread, df = predictor.load_data(
        csv_file='ml_features_v2_2025.csv',
        min_week=4
    )

    # Train model
    history = predictor.train(
        X=X,
        y=y_spread,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        patience=15,
        save_path='models/cfb_spread_predictor_pytorch.pt',
        verbose=True
    )

    # Save model
    predictor.save('models/cfb_spread_predictor_pytorch.pt')

    print("\n" + "="*60)
    print("Point Spread Regressor Training Complete!")
    print("="*60 + "\n")

    return predictor, history


def plot_training_history(history, title, save_path=None):
    """Plot training history"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot loss
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    """Main training pipeline"""
    import os

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Train win/loss classifier
    win_predictor, win_history = train_win_predictor()

    # Plot training history
    plot_training_history(
        win_history,
        'CFB Win/Loss Classifier - Training History',
        save_path='models/win_predictor_history.png'
    )

    # Train spread regressor
    spread_predictor, spread_history = train_spread_predictor()

    # Plot training history
    plot_training_history(
        spread_history,
        'CFB Point Spread Regressor - Training History',
        save_path='models/spread_predictor_history.png'
    )

    print("\n" + "="*60)
    print("All Models Trained Successfully!")
    print("="*60)
    print("\nModels saved to:")
    print("  - models/cfb_win_predictor_pytorch.pt")
    print("  - models/cfb_spread_predictor_pytorch.pt")
    print("\nTraining plots saved to:")
    print("  - models/win_predictor_history.png")
    print("  - models/spread_predictor_history.png")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
