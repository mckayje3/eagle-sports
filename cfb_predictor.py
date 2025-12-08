"""
College Football Game Outcome Predictor
Uses deep learning to predict game winners and point spreads
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CFBPredictor:
    """Deep learning model for college football predictions"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.history = None

    def load_data(self, csv_file='ml_features_2025.csv', min_week=3):
        """
        Load and prepare data for training

        Args:
            csv_file: Path to features CSV
            min_week: Minimum week to include (teams need some games for stats)

        Returns:
            X, y_win, y_spread for training
        """
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)

        print(f"Initial data shape: {df.shape}")
        print(f"Weeks available: {df['week'].min()} to {df['week'].max()}")

        # Filter to games where teams have played some games
        df = df[df['week'] >= min_week].copy()
        print(f"After filtering to week >= {min_week}: {df.shape}")

        # Remove rows with missing outcomes
        df = df.dropna(subset=['home_win', 'point_differential'])
        print(f"After removing missing outcomes: {df.shape}")

        # Select features
        feature_cols = [
            'week',
            'neutral_site',
            'h2h_games',
            'h2h_win_pct',
            'home_recent_win_pct',
            'away_recent_win_pct',
            'recent_form_diff'
        ]

        # Check which features actually exist
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"\nAvailable features: {len(available_features)}")
        print(available_features)

        self.feature_columns = available_features

        X = df[self.feature_columns].values
        y_win = df['home_win'].values
        y_spread = df['point_differential'].values

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Win labels shape: {y_win.shape}")
        print(f"Spread labels shape: {y_spread.shape}")
        print(f"Home team wins: {y_win.sum()}/{len(y_win)} ({100*y_win.mean():.1f}%)")

        return X, y_win, y_spread, df

    def build_model(self, input_dim, model_type='classification'):
        """
        Build neural network model

        Args:
            input_dim: Number of input features
            model_type: 'classification' for win/loss or 'regression' for spread
        """
        print(f"\nBuilding {model_type} model with {input_dim} inputs...")

        model = keras.Sequential([
            # Input layer
            keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),

            # Hidden layers
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(16, activation='relu'),

            # Output layer
            keras.layers.Dense(1, activation='sigmoid' if model_type == 'classification' else 'linear')
        ])

        if model_type == 'classification':
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
        else:
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )

        print(f"Model built successfully!")
        print(f"Total parameters: {model.count_params():,}")

        return model

    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        """
        Train the model

        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build model if not exists
        if self.model is None:
            model_type = 'classification' if len(np.unique(y)) == 2 else 'regression'
            self.model = self.build_model(X.shape[1], model_type=model_type)

        # Early stopping callback
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        print(f"\nTraining model...")
        print(f"Training samples: {int(len(X) * (1 - validation_split))}")
        print(f"Validation samples: {int(len(X) * validation_split)}")

        # Train
        self.history = self.model.fit(
            X_scaled, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=verbose
        )

        print(f"\nTraining complete!")
        print(f"Best epoch: {len(self.history.history['loss']) - 15}")

        return self.history

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        X_scaled = self.scaler.transform(X_test)
        results = self.model.evaluate(X_scaled, y_test, verbose=0)

        print("\nTest Set Performance:")
        for i, metric_name in enumerate(self.model.metrics_names):
            print(f"  {metric_name}: {results[i]:.4f}")

        return results

    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0)

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy or MAE
        metric_key = 'accuracy' if 'accuracy' in self.history.history else 'mae'
        metric_name = 'Accuracy' if metric_key == 'accuracy' else 'MAE'

        axes[1].plot(self.history.history[metric_key], label=f'Training {metric_name}')
        axes[1].plot(self.history.history[f'val_{metric_key}'], label=f'Validation {metric_name}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'Model {metric_name} Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("\nSaved training history plot to training_history.png")
        plt.close()

    def save_model(self, filepath='cfb_model.keras'):
        """Save model to file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='cfb_model.keras'):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """Main training script"""
    print("="*80)
    print("COLLEGE FOOTBALL OUTCOME PREDICTOR")
    print("="*80)

    # Initialize predictor
    predictor = CFBPredictor()

    # Load data
    X, y_win, y_spread, df = predictor.load_data(min_week=1)

    if len(X) < 50:
        print("\nWARNING: Not enough data for training!")
        print("Need more completed games with team statistics.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_win, test_size=0.2, random_state=42
    )

    print(f"\nTraining set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games")

    # Train win/loss classifier
    print("\n" + "="*80)
    print("TRAINING WIN/LOSS CLASSIFIER")
    print("="*80)

    predictor.train(X_train, y_train, epochs=150, batch_size=32)

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    predictor.evaluate(X_test, y_test)

    # Plot training history
    predictor.plot_training_history()

    # Test predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)

    predictions = predictor.predict(X_test[:10])
    print("\nFirst 10 test predictions:")
    print(f"{'Predicted':>12} {'Actual':>8} {'Correct':>10}")
    print("-" * 35)

    for i in range(10):
        pred_win = predictions[i][0]
        actual_win = y_test[i]
        correct = (pred_win > 0.5) == (actual_win == 1)
        check_mark = 'YES' if correct else 'NO'
        print(f"{pred_win:>12.1%} {actual_win:>8.0f} {check_mark:>10}")

    # Save model
    predictor.save_model()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nModel saved and ready for predictions!")
    print("Use the model to predict upcoming games.")


if __name__ == '__main__':
    # Check if TensorFlow can use GPU
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))
    print()

    main()
