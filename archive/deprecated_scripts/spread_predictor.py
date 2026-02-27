"""
Point Spread Predictor
Predicts point differentials (spreads) using regression
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SpreadPredictor:
    """Neural network for predicting point spreads"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.history = None

    def load_data(self, csv_file='ml_features_v2_2025.csv', min_week=4):
        """
        Load enhanced features for spread prediction

        Args:
            csv_file: Features CSV (with team stats)
            min_week: Minimum week (need stats)
        """
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)

        print(f"Initial data shape: {df.shape}")

        # Filter to games with team stats
        df = df[df['week'] >= min_week].copy()
        print(f"After filtering to week >= {min_week}: {df.shape}")

        # Remove rows with missing outcomes
        df = df.dropna(subset=['point_differential'])
        print(f"After removing missing outcomes: {df.shape}")

        # Select best features for spread prediction
        feature_cols = [
            # Basic
            'week',
            'neutral_site',

            # Win percentage
            'home_win_pct',
            'away_win_pct',
            'win_pct_diff',

            # Scoring
            'home_points_scored_avg',
            'away_points_scored_avg',
            'points_scored_diff',
            'home_points_allowed_avg',
            'away_points_allowed_avg',
            'points_allowed_diff',
            'home_point_differential_avg',
            'away_point_differential_avg',
            'point_differential_diff',

            # Yards
            'home_total_yards_avg',
            'away_total_yards_avg',
            'yards_diff',
            'passing_yards_diff',
            'rushing_yards_diff',

            # Other
            'turnovers_diff',
            'home_third_down_pct',
            'away_third_down_pct',

            # Form
            'home_recent_win_pct',
            'away_recent_win_pct',
            'recent_form_diff',

            # H2H
            'h2h_win_pct'
        ]

        # Check which features exist
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"\nAvailable features: {len(available_features)}")

        self.feature_columns = available_features

        X = df[self.feature_columns].values
        y_spread = df['point_differential'].values

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Spread labels shape: {y_spread.shape}")
        print(f"Average point differential: {y_spread.mean():.2f}")
        print(f"Std dev: {y_spread.std():.2f}")
        print(f"Range: [{y_spread.min():.0f}, {y_spread.max():.0f}]")

        return X, y_spread, df

    def build_model(self, input_dim):
        """Build regression model for spread prediction"""
        print(f"\nBuilding spread prediction model with {input_dim} inputs...")

        model = keras.Sequential([
            # Input layer
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),

            # Hidden layers
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),

            keras.layers.Dense(16, activation='relu'),

            # Output layer (linear for regression)
            keras.layers.Dense(1, activation='linear')
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )

        print(f"Model built!")
        print(f"Total parameters: {model.count_params():,}")

        return model

    def train(self, X, y, validation_split=0.2, epochs=150, batch_size=32, verbose=1):
        """Train the spread prediction model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build model if not exists
        if self.model is None:
            self.model = self.build_model(X.shape[1])

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
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
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        print(f"\nTraining complete!")
        return self.history

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        X_scaled = self.scaler.transform(X_test)
        results = self.model.evaluate(X_scaled, y_test, verbose=0)

        print("\nTest Set Performance:")
        for i, metric_name in enumerate(self.model.metrics_names):
            print(f"  {metric_name}: {results[i]:.4f}")

        # Additional metrics
        predictions = self.model.predict(X_scaled, verbose=0)
        errors = predictions.flatten() - y_test
        abs_errors = np.abs(errors)

        print(f"\nAdditional Metrics:")
        print(f"  Mean Absolute Error: {abs_errors.mean():.2f} points")
        print(f"  Median Absolute Error: {np.median(abs_errors):.2f} points")
        print(f"  Within 3 points: {(abs_errors <= 3).mean():.1%}")
        print(f"  Within 7 points: {(abs_errors <= 7).mean():.1%}")
        print(f"  Within 14 points: {(abs_errors <= 14).mean():.1%}")

        return results

    def predict(self, X):
        """Make spread predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0)

    def plot_training_history(self):
        """Plot training curves"""
        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss (MSE)')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss (MSE)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Model Loss Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Absolute Error (points)')
        axes[1].set_title('Prediction Error Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('spread_training_history.png', dpi=150)
        print("\nSaved training history to spread_training_history.png")
        plt.close()

    def plot_predictions(self, X_test, y_test):
        """Plot predicted vs actual spreads"""
        predictions = self.predict(X_test).flatten()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Scatter plot
        axes[0].scatter(y_test, predictions, alpha=0.5)
        axes[0].plot([-60, 60], [-60, 60], 'r--', label='Perfect prediction')
        axes[0].set_xlabel('Actual Point Differential')
        axes[0].set_ylabel('Predicted Point Differential')
        axes[0].set_title('Predicted vs Actual Spreads')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Error distribution
        errors = predictions - y_test
        axes[1].hist(errors, bins=30, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--', label='Perfect prediction')
        axes[1].set_xlabel('Prediction Error (points)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('spread_predictions.png', dpi=150)
        print("Saved prediction plot to spread_predictions.png")
        plt.close()

    def save_model(self, filepath='spread_model.keras'):
        """Save model and scaler to file"""
        self.model.save(filepath)

        # Save scaler and feature columns
        scaler_file = filepath.replace('.keras', '_scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)

        print(f"Model saved to {filepath}")
        print(f"Scaler saved to {scaler_file}")

    def load_model(self, filepath='spread_model.keras'):
        """Load model and scaler from file"""
        self.model = keras.models.load_model(filepath)

        # Load scaler and feature columns
        scaler_file = filepath.replace('.keras', '_scaler.pkl')
        try:
            with open(scaler_file, 'rb') as f:
                data = pickle.load(f)
                self.scaler = data['scaler']
                self.feature_columns = data['feature_columns']
            print(f"Model and scaler loaded from {filepath}")
        except FileNotFoundError:
            print(f"Warning: Scaler file not found ({scaler_file})")
            print("Model loaded but predictions may not work without scaler")


def main():
    """Main training script"""
    print("="*80)
    print("POINT SPREAD PREDICTOR")
    print("="*80)

    # Initialize predictor
    predictor = SpreadPredictor()

    # Load data
    X, y_spread, df = predictor.load_data(min_week=4)

    if len(X) < 50:
        print("\nWARNING: Not enough data for training!")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_spread, test_size=0.2, random_state=42
    )

    print(f"\nTraining set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games")

    # Train
    print("\n" + "="*80)
    print("TRAINING SPREAD PREDICTOR")
    print("="*80)

    predictor.train(X_train, y_train, epochs=200, batch_size=32)

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    predictor.evaluate(X_test, y_test)

    # Plot results
    predictor.plot_training_history()
    predictor.plot_predictions(X_test, y_test)

    # Sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)

    predictions = predictor.predict(X_test[:10]).flatten()
    print(f"\n{'Predicted':>12} {'Actual':>10} {'Error':>10}")
    print("-" * 35)

    for i in range(10):
        pred = predictions[i]
        actual = y_test[i]
        error = pred - actual
        print(f"{pred:>12.1f} {actual:>10.1f} {error:>10.1f}")

    # Save model
    predictor.save_model()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nSpread prediction model ready!")
    print("Use this model to predict point differentials.")


if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    print()
    main()
