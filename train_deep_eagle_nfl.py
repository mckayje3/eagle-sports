"""
Train Deep-Eagle LSTM Model for NFL Spread and Total Prediction
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, r'C:\Users\jbeast\documents\coding\deep')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from pathlib import Path

from core import (
    TimeSeriesDataset,
    TimeSeriesDataLoader,
    LSTMModel,
    Trainer
)
from core.training.callbacks import EarlyStopping, ModelCheckpoint


class NFLPredictor:
    """Deep-Eagle LSTM predictor for NFL spreads and totals"""

    def __init__(self, sequence_length=10, hidden_dim=128, num_layers=2, dropout=0.2):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.spread_model = None
        self.total_model = None
        self.scaler = None
        self.feature_columns = None

    def prepare_features(self, df):
        """
        Select and prepare features for training

        Returns:
            features (np.array), spread_targets (np.array), total_targets (np.array)
        """
        # Select feature columns (exclude identifiers and targets)
        exclude_cols = [
            'game_id', 'season', 'week', 'date', 'team_id',
            'home_team_id', 'away_team_id', 'completed',
            'home_score', 'away_score', 'points_scored', 'points_allowed',
            'point_differential', 'win',  # These are targets or derived from targets
            # Exclude current game Vegas lines (would be data leakage)
            'vegas_spread_home', 'vegas_spread', 'vegas_total',
            'covered_spread', 'went_over'  # These are outcomes, not features
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols

        print(f"\nFeature Selection:")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Feature columns: {len(feature_cols)}")
        print(f"   Excluded columns: {len(exclude_cols)}")

        # Extract features
        features = df[feature_cols].values

        # Extract targets
        spread_targets = df['point_differential'].values  # Home team perspective
        total_targets = (df['points_scored'] + df['points_allowed']).values

        # Handle any remaining NaN values
        features = np.nan_to_num(features, nan=0.0)

        return features, spread_targets, total_targets

    def split_data(self, df, train_weeks=None, val_weeks=None):
        """
        Split data by week for temporal validation

        Args:
            df: Full dataset
            train_weeks: Weeks for training (default: 3-10)
            val_weeks: Weeks for validation (default: 11-12)

        Returns:
            train_df, val_df
        """
        if train_weeks is None:
            train_weeks = list(range(3, 11))  # Weeks 3-10
        if val_weeks is None:
            val_weeks = [11, 12]  # Weeks 11-12

        train_df = df[df['week'].isin(train_weeks)].copy()
        val_df = df[df['week'].isin(val_weeks)].copy()

        print(f"\nTrain/Val Split:")
        print(f"   Train weeks: {min(train_weeks)}-{max(train_weeks)}")
        print(f"   Train games: {len(train_df)}")
        val_week_str = str(val_weeks[0]) if len(val_weeks) == 1 else f"{min(val_weeks)}-{max(val_weeks)}"
        print(f"   Val weeks: {val_week_str}")
        print(f"   Val games: {len(val_df)}")

        return train_df, val_df

    def train_model(self, train_features, train_targets, val_features, val_targets,
                   model_name='spread', epochs=100):
        """
        Train a single LSTM model

        Args:
            train_features: Training features
            train_targets: Training targets
            val_features: Validation features
            val_targets: Validation targets
            model_name: Name for this model (spread or total)
            epochs: Number of training epochs

        Returns:
            Trained model, training history
        """
        print(f"\n{'=' * 80}")
        print(f"TRAINING NFL {model_name.upper()} MODEL")
        print(f"{'=' * 80}")

        # Create datasets
        train_dataset = TimeSeriesDataset(
            data=train_features,
            targets=train_targets,
            sequence_length=self.sequence_length,
            forecast_horizon=1
        )

        val_dataset = TimeSeriesDataset(
            data=val_features,
            targets=val_targets,
            sequence_length=self.sequence_length,
            forecast_horizon=1
        )

        print(f"\nDataset Creation:")
        print(f"   Train sequences: {len(train_dataset)}")
        print(f"   Val sequences: {len(val_dataset)}")

        # Create data loaders
        batch_size = 32
        train_loader = TimeSeriesDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TimeSeriesDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        input_dim = train_features.shape[1]
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        print(f"\nModel Architecture:")
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden dim: {self.hidden_dim}")
        print(f"   Num layers: {self.num_layers}")
        print(f"   Dropout: {self.dropout}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Setup training
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, min_delta=0.0001),
            ModelCheckpoint(
                filepath=f'models/nfl_{model_name}_best.pth',
                monitor='val_loss',
                save_best_only=True
            )
        ]

        # Create trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            callbacks=callbacks
        )

        # Train
        print(f"\nTraining...")
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs
        )

        print(f"\nTraining Complete!")
        print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"   Best val loss: {min(history['val_loss']):.4f}")

        return model, history

    def evaluate_model(self, model, features, targets, target_name='spread'):
        """
        Evaluate model performance

        Args:
            model: Trained model
            features: Test features
            targets: True targets
            target_name: Name of target (for display)

        Returns:
            predictions, metrics dict
        """
        # Create dataset
        dataset = TimeSeriesDataset(
            data=features,
            targets=targets,
            sequence_length=self.sequence_length,
            forecast_horizon=1
        )

        loader = TimeSeriesDataLoader(dataset, batch_size=32, shuffle=False)

        # Get predictions
        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                pred = model(batch_x)
                predictions.extend(pred.cpu().numpy().flatten())
                actuals.extend(batch_y.cpu().numpy().flatten())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

        print(f"\n{target_name.upper()} Evaluation:")
        print(f"   MSE:  {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")

        # Vegas benchmark comparison (NFL benchmarks)
        if target_name == 'spread':
            print(f"\n   Vegas Benchmark MAE: ~5.5 points (NFL)")
            if mae < 5.5:
                print(f"   BEATING VEGAS TARGET!")
            elif mae < 7.0:
                print(f"   Close to Vegas performance")
            else:
                print(f"   Below Vegas performance")

        elif target_name == 'total':
            print(f"\n   Vegas Benchmark MAE: ~7.0 points (NFL)")
            if mae < 7.0:
                print(f"   BEATING VEGAS TARGET!")
            elif mae < 9.0:
                print(f"   Close to Vegas performance")
            else:
                print(f"   Below Vegas performance")

        return predictions, metrics

    def save_models(self, save_dir='models'):
        """Save trained models and scaler"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Save spread model
        if self.spread_model is not None:
            torch.save({
                'model_state_dict': self.spread_model.state_dict(),
                'input_dim': len(self.feature_columns),
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }, save_dir / 'nfl_spread_best.pth')
            print(f"   Saved spread model")

        # Save total model
        if self.total_model is not None:
            torch.save({
                'model_state_dict': self.total_model.state_dict(),
                'input_dim': len(self.feature_columns),
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }, save_dir / 'nfl_total_best.pth')
            print(f"   Saved total model")

        # Save scaler
        scaler_data = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        with open(save_dir / 'nfl_scaler.pkl', 'wb') as f:
            pickle.dump(scaler_data, f)

        print(f"\nModels saved to: {save_dir}")
        print(f"   - nfl_spread_best.pth")
        print(f"   - nfl_total_best.pth")
        print(f"   - nfl_scaler.pkl")


def main():
    """Main training pipeline"""
    print("=" * 80)
    print("DEEP-EAGLE NFL TRAINING PIPELINE")
    print("=" * 80)

    # Check for prepared data
    data_file = 'nfl_training_data.csv'
    if not Path(data_file).exists():
        print(f"\nData file not found: {data_file}")
        print(f"Please run: py nfl_data_preparation.py")
        return

    # Load data
    print(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df)} games")

    # Initialize predictor
    predictor = NFLPredictor(
        sequence_length=10,  # Last 10 games
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    )

    # Split by weeks (train on early season, validate on late season)
    train_df, val_df = predictor.split_data(df)

    # Prepare features
    print(f"\nPreparing features...")
    train_features, train_spread, train_total = predictor.prepare_features(train_df)
    val_features, val_spread, val_total = predictor.prepare_features(val_df)

    # Scale features
    print(f"\nScaling features...")
    predictor.scaler = StandardScaler()
    train_features_scaled = predictor.scaler.fit_transform(train_features)
    val_features_scaled = predictor.scaler.transform(val_features)
    print(f"   Features scaled")

    # Train spread model
    spread_model, spread_history = predictor.train_model(
        train_features_scaled, train_spread,
        val_features_scaled, val_spread,
        model_name='spread',
        epochs=100
    )
    predictor.spread_model = spread_model

    # Evaluate spread model
    spread_preds, spread_metrics = predictor.evaluate_model(
        spread_model, val_features_scaled, val_spread, target_name='spread'
    )

    # Train total model
    total_model, total_history = predictor.train_model(
        train_features_scaled, train_total,
        val_features_scaled, val_total,
        model_name='total',
        epochs=100
    )
    predictor.total_model = total_model

    # Evaluate total model
    total_preds, total_metrics = predictor.evaluate_model(
        total_model, val_features_scaled, val_total, target_name='total'
    )

    # Save models
    predictor.save_models()

    # Final summary
    print("\n" + "=" * 80)
    print("NFL TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nSpread Model:")
    print(f"   MAE: {spread_metrics['mae']:.2f} points")
    print(f"   RMSE: {spread_metrics['rmse']:.2f} points")

    print(f"\nTotal Model:")
    print(f"   MAE: {total_metrics['mae']:.2f} points")
    print(f"   RMSE: {total_metrics['rmse']:.2f} points")

    print(f"\nModels saved to: models/")
    print(f"  - nfl_spread_best.pth")
    print(f"  - nfl_total_best.pth")
    print(f"  - nfl_scaler.pkl")
    print("=" * 80)


if __name__ == '__main__':
    main()
