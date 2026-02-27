# PyTorch Migration Guide

This document describes the migration from TensorFlow to PyTorch using the shared deep learning framework.

## Overview

The sports prediction system has been refactored to use the core deep learning module from `documents/coding/deep`. This provides:

- ✅ **Shared 80% core functionality** across projects
- ✅ **Modular, production-ready** training pipeline
- ✅ **Better model management** with checkpointing and callbacks
- ✅ **Consistent architecture** using PyTorch instead of TensorFlow
- ✅ **Extensible framework** for future enhancements

## Architecture Changes

### Before (TensorFlow)
```
sports/
├── cfb_predictor.py          # TensorFlow/Keras predictor
├── cfb_predictor_v2.py        # Enhanced TensorFlow predictor
├── spread_predictor.py        # TensorFlow spread predictor
└── predict_upcoming_games.py  # Prediction script
```

### After (PyTorch + Deep Framework)
```
sports/
├── cfb_models.py                    # PyTorch models (NEW)
├── cfb_predictor_pytorch.py         # PyTorch predictor wrapper (NEW)
├── train_pytorch_model.py           # Training script (NEW)
├── predict_upcoming_pytorch.py      # Prediction script (NEW)
└── [old TensorFlow files...]        # Keep for reference
```

## Installation

### 1. Install Deep Framework

The deep learning framework must be installed as an editable package:

```bash
pip install -e C:\Users\jbeast\documents\coding\deep
```

This installs:
- PyTorch (`torch>=2.0.0`)
- NumPy, Pandas, scikit-learn
- PyYAML, tqdm
- The core deep learning framework

### 2. Install Other Dependencies

```bash
pip install -r requirements.txt
```

## New Components

### 1. cfb_models.py

Defines PyTorch models that extend the deep framework's `BaseTimeSeriesModel`:

- **CFBFeedForwardModel**: Feedforward neural network for tabular game data
- **CFBLSTMModel**: LSTM model for sequential game history (future use)

```python
from cfb_models import CFBFeedForwardModel

model = CFBFeedForwardModel(
    input_dim=50,
    hidden_dim=64,
    output_dim=1,
    num_layers=3,
    dropout=0.3,
    task_type='classification'
)
```

### 2. cfb_predictor_pytorch.py

Wrapper class that provides a similar interface to the old TensorFlow predictors:

```python
from cfb_predictor_pytorch import CFBPredictorPyTorch

# Create predictor
predictor = CFBPredictorPyTorch(
    model_type='feedforward',
    task='classification'  # or 'regression'
)

# Load data
X, y_win, y_spread, df = predictor.load_data('ml_features_v2_2025.csv')

# Train
predictor.train(X, y_win, epochs=100, batch_size=32)

# Predict
predictions = predictor.predict(X_new)

# Save/load
predictor.save('models/cfb_model_pytorch.pt')
predictor.load('models/cfb_model_pytorch.pt')
```

### 3. train_pytorch_model.py

Complete training pipeline that:
- Trains both win/loss classifier and spread regressor
- Uses deep framework's Trainer with callbacks
- Saves models and training plots
- Leverages early stopping and model checkpointing

```bash
python train_pytorch_model.py
```

### 4. predict_upcoming_pytorch.py

Makes predictions on upcoming games using trained PyTorch models:

```bash
python predict_upcoming_pytorch.py
```

## Key Features from Deep Framework

### 1. Trainer with Callbacks

The deep framework's `Trainer` provides:

```python
from core.training import Trainer, EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=15, verbose=True),
    ModelCheckpoint(filepath='best_model.pt', verbose=True)
]

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda',
    callbacks=callbacks
)

history = trainer.fit(train_loader, val_loader, epochs=100)
```

### 2. Model Checkpointing

Automatic saving of best model:
- Monitors validation loss
- Saves only when improvement detected
- Restores best weights at end of training

### 3. Early Stopping

Prevents overfitting:
- Monitors validation loss
- Stops training if no improvement for `patience` epochs
- Restores best weights

### 4. Device Management

Automatic GPU detection:
```python
from core.utils import get_device

device = get_device()  # Returns 'cuda' if available, else 'cpu'
```

### 5. Reproducibility

Set random seeds across all libraries:
```python
from core.utils import set_seed

set_seed(42)
```

## Usage Examples

### Training Models

```python
from cfb_predictor_pytorch import CFBPredictorPyTorch
from core.utils import set_seed

# Set seed for reproducibility
set_seed(42)

# Train win predictor
win_predictor = CFBPredictorPyTorch(
    model_type='feedforward',
    task='classification'
)
X, y_win, _, _ = win_predictor.load_data('ml_features_v2_2025.csv', min_week=4)
win_predictor.train(X, y_win, epochs=100, patience=15)
win_predictor.save('models/win_model.pt')

# Train spread predictor
spread_predictor = CFBPredictorPyTorch(
    model_type='feedforward',
    task='regression'
)
X, _, y_spread, _ = spread_predictor.load_data('ml_features_v2_2025.csv', min_week=4)
spread_predictor.train(X, y_spread, epochs=100, patience=15)
spread_predictor.save('models/spread_model.pt')
```

### Making Predictions

```python
from cfb_predictor_pytorch import CFBPredictorPyTorch
import pandas as pd

# Load model
predictor = CFBPredictorPyTorch(task='classification')
predictor.load('models/win_model.pt')

# Prepare features
features_df = pd.read_csv('game_features.csv')
X = features_df[predictor.feature_columns].values

# Predict
predictions = predictor.predict(X)
print(f"Home team win probability: {predictions[0]:.1%}")
```

## Comparison: TensorFlow vs PyTorch

| Feature | TensorFlow (Old) | PyTorch (New) |
|---------|------------------|---------------|
| Framework | Keras/TensorFlow | PyTorch + Deep Framework |
| Model Definition | Sequential API | nn.Module classes |
| Training Loop | `model.fit()` | `Trainer.fit()` with callbacks |
| Callbacks | Keras callbacks | Deep framework callbacks |
| Model Saving | `.keras` files | `.pt` checkpoints |
| Scaler Saving | Manual | Automatic with model |
| Device Management | Manual | Automatic (`get_device()`) |
| Early Stopping | Keras callback | Framework callback |
| Code Sharing | None | 80% shared with deep framework |

## Migration Checklist

- [x] Install deep framework as editable package
- [x] Create PyTorch model classes (cfb_models.py)
- [x] Create PyTorch predictor wrapper (cfb_predictor_pytorch.py)
- [x] Create training script (train_pytorch_model.py)
- [x] Create prediction script (predict_upcoming_pytorch.py)
- [x] Update requirements.txt
- [ ] Train new PyTorch models
- [ ] Validate predictions match or improve on TensorFlow
- [ ] Update dashboard to use PyTorch models (if applicable)
- [ ] Deprecate TensorFlow files

## Future Enhancements

With the deep framework integration, you can now easily:

1. **Use LSTM models** for sequence-based predictions:
   ```python
   predictor = CFBPredictorPyTorch(model_type='lstm', task='classification')
   ```

2. **Add custom features** using framework's FeatureEngine:
   ```python
   from core import FeatureEngine
   from cfb_custom_features import TeamMomentumFeatures

   engine = FeatureEngine(
       transformers=[TeamMomentumFeatures()],
       scaler='standard'
   )
   ```

3. **Time-series validation** with walk-forward splits:
   ```python
   from core import WalkForwardSplit

   splitter = WalkForwardSplit(n_splits=5)
   ```

4. **Advanced models** (GRU, Transformer) from deep framework:
   ```python
   from core.models import TransformerModel
   ```

## Troubleshooting

### Import Errors

If you get import errors from `core`:
```bash
# Reinstall deep framework
pip install -e C:\Users\jbeast\documents\coding\deep
```

### CUDA/GPU Issues

To force CPU usage:
```python
predictor = CFBPredictorPyTorch(...)
predictor.device = 'cpu'
```

### Model Performance

If models underperform:
1. Check feature scaling is working
2. Increase `hidden_dim` or `num_layers`
3. Adjust learning rate
4. Increase training epochs
5. Check for data quality issues

## Questions?

See the deep framework documentation:
- `C:\Users\jbeast\documents\coding\deep\README.md`
- `C:\Users\jbeast\documents\coding\deep\QUICKSTART.md`
- `C:\Users\jbeast\documents\coding\deep\examples\sports_analytics\`
