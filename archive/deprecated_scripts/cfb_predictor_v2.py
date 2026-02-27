"""
College Football Game Outcome Predictor V2
Enhanced version using 55 features for better team differentiation
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


class CFBPredictorV2:
    """Enhanced deep learning model with comprehensive team statistics"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.history = None

    def load_data(self, csv_file='ml_features_v2_2025.csv', min_week=3):
        """
        Load enhanced features for training

        Args:
            csv_file: Features CSV with 55 features
            min_week: Minimum week (need stats)
        """
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)

        print(f"Initial data shape: {df.shape}")
        print(f"Weeks available: {df['week'].min()} to {df['week'].max()}")

        # Filter to games with team stats
        df = df[df['week'] >= min_week].copy()
        print(f"After filtering to week >= {min_week}: {df.shape}")

        # Remove rows with missing outcomes
        df = df.dropna(subset=['home_win', 'point_differential'])
        print(f"After removing missing outcomes: {df.shape}")

        # Select all available features (excluding outcomes and IDs)
        exclude_cols = [
            'game_id', 'home_team_id', 'away_team_id',
            'home_win', 'point_differential',
            'home_team', 'away_team', 'season'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Check which features exist
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"\nAvailable features: {len(available_features)}")

        self.feature_columns = available_features

        X = df[self.feature_columns].values
        y_win = df['home_win'].values

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Win labels shape: {y_win.shape}")
        print(f"Home team wins: {y_win.sum()}/{len(y_win)} ({100*y_win.mean():.1f}%)")

        # Show feature breakdown
        print(f"\nFeature categories:")
        basic_features = [f for f in available_features if f in ['week', 'neutral_site', 'h2h_games', 'h2h_win_pct']]
        print(f"  Basic: {len(basic_features)}")

        scoring_features = [f for f in available_features if 'points' in f or 'point_differential' in f]
        print(f"  Scoring: {len(scoring_features)}")

        yards_features = [f for f in available_features if 'yards' in f]
        print(f"  Yards: {len(yards_features)}")

        efficiency_features = [f for f in available_features if any(x in f for x in ['turnover', 'third_down', 'penalty'])]
        print(f"  Efficiency: {len(efficiency_features)}")

        form_features = [f for f in available_features if 'win_pct' in f or 'form' in f or 'recent' in f]
        print(f"  Form/Record: {len(form_features)}")

        return X, y_win, df

    def build_model(self, input_dim):
        """Build classification model for win/loss prediction"""
        print(f"\nBuilding enhanced win/loss classifier with {input_dim} inputs...")

        model = keras.Sequential([
            # Input layer - larger for more features
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),

            # Hidden layers
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(16, activation='relu'),

            # Output layer (binary classification)
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        print(f"Model built!")
        print(f"Total parameters: {model.count_params():,}")

        return model

    def train(self, X, y, validation_split=0.2, epochs=150, batch_size=32, verbose=1):
        """Train the enhanced model"""
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

        print(f"\nTraining enhanced model...")
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
        predictions = (self.model.predict(X_scaled, verbose=0) > 0.5).astype(int)
        accuracy = (predictions.flatten() == y_test).mean()

        print(f"\nAccuracy: {accuracy:.1%}")

        return results

    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0)

    def predict_proba(self, X):
        """Get win probabilities"""
        return self.predict(X)

    def plot_training_history(self):
        """Plot training curves"""
        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # AUC
        axes[2].plot(self.history.history['auc'], label='Training AUC')
        axes[2].plot(self.history.history['val_auc'], label='Validation AUC')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('Model AUC Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history_v2.png', dpi=150)
        print("\nSaved training history to training_history_v2.png")
        plt.close()

    def save_model(self, filepath='cfb_model_v2.keras'):
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

    def load_model(self, filepath='cfb_model_v2.keras'):
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
    print("COLLEGE FOOTBALL PREDICTOR V2 - ENHANCED FEATURES")
    print("="*80)

    # Initialize predictor
    predictor = CFBPredictorV2()

    # Load enhanced data
    X, y_win, df = predictor.load_data(min_week=3)

    if len(X) < 50:
        print("\nWARNING: Not enough data for training!")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_win, test_size=0.2, random_state=42
    )

    print(f"\nTraining set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games")

    # Train
    print("\n" + "="*80)
    print("TRAINING WIN/LOSS CLASSIFIER")
    print("="*80)

    predictor.train(X_train, y_train, epochs=150, batch_size=32)

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    predictor.evaluate(X_test, y_test)

    # Plot results
    predictor.plot_training_history()

    # Sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)

    predictions = predictor.predict_proba(X_test[:10]).flatten()
    print(f"\n{'Home Win Prob':>15} {'Actual':>10}")
    print("-" * 27)

    for i in range(10):
        prob = predictions[i]
        actual = "Home Win" if y_test[i] == 1 else "Away Win"
        print(f"{prob:>15.1%} {actual:>10}")

    # Save model
    predictor.save_model()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nEnhanced win/loss model ready!")
    print("This model uses 55 features for better team differentiation.")
    print("\nKey improvements over v1:")
    print("  - 55 features vs 7 (team stats, scoring, efficiency)")
    print("  - Better differentiation between strong/weak teams")
    print("  - More nuanced predictions beyond home field advantage")


if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    print()
    main()
