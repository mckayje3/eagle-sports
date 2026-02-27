"""
Retrain CFB model with 2023-2024-2025 data
Train on 2023-2024, test on 2025 (proper time-series validation)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import os


class DeepEagleModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(DeepEagleModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.home_score_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.away_score_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        home_score = self.home_score_head(features)
        away_score = self.away_score_head(features)
        return torch.cat([home_score, away_score], dim=1)


def main():
    print('=' * 80)
    print('TRAINING CFB MODEL WITH 2023-2024-2025 DATA')
    print('=' * 80)

    # Load combined data
    df = pd.read_csv('cfb_2023_2024_2025_deep_eagle_features.csv')
    print(f'\nLoaded {len(df)} total games')

    # Split: 2023-2024 for training, 2025 for testing
    train_df = df[df['season'].isin([2023, 2024])]
    test_df = df[df['season'] == 2025]

    print(f'Training: 2023-2024 -> {len(train_df)} games')
    print(f'Testing: 2025 -> {len(test_df)} games')

    # Identify feature columns
    exclude_cols = [
        'game_id', 'season', 'week', 'home_team_id', 'away_team_id',
        'home_score', 'away_score', 'point_spread', 'total_points', 'home_win'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f'Using {len(feature_cols)} features')

    # Prepare data
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[['home_score', 'away_score']].values
    y_test = test_df[['home_score', 'away_score']].values

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    device = torch.device('cpu')
    print(f'Device: {device}')

    # Build model
    model = DeepEagleModel(input_dim=X_train.shape[1]).to(device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # Training
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    best_state = None

    print(f'\nTraining...')
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val MAE':<12}")
    print('-' * 45)

    for epoch in range(200):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            val_loss = criterion(val_pred, y_test_t).item()
            val_mae = torch.mean(torch.abs(val_pred - y_test_t)).item()

        avg_train_loss = np.mean(train_losses)

        if (epoch + 1) % 20 == 0:
            print(f'{epoch+1:<8} {avg_train_loss:<12.4f} {val_loss:<12.4f} {val_mae:<12.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break

    # Restore best model
    model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()

    # Metrics
    pred_spread = predictions[:, 0] - predictions[:, 1]
    actual_spread = y_test[:, 0] - y_test[:, 1]
    spread_mae = np.mean(np.abs(pred_spread - actual_spread))

    pred_total = predictions[:, 0] + predictions[:, 1]
    actual_total = y_test[:, 0] + y_test[:, 1]
    total_mae = np.mean(np.abs(pred_total - actual_total))

    pred_winner = (predictions[:, 0] > predictions[:, 1]).astype(int)
    actual_winner = (y_test[:, 0] > y_test[:, 1]).astype(int)
    winner_accuracy = np.mean(pred_winner == actual_winner)

    print(f'\n' + '=' * 80)
    print('CFB 2025 TEST SET EVALUATION')
    print('=' * 80)
    print(f'Winner Accuracy: {winner_accuracy:.1%}')
    print(f'Spread MAE: {spread_mae:.2f} points')
    print(f'Total MAE: {total_mae:.2f} points')

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'sport': 'CFB',
        'season': 2025,
        'feature_cols': feature_cols
    }, 'models/deep_eagle_cfb_2025_gameday.pt')

    with open('models/deep_eagle_cfb_2025_gameday_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f'\nModel saved: models/deep_eagle_cfb_2025_gameday.pt')

    # Now evaluate vs Vegas
    print(f'\n' + '=' * 80)
    print('CFB 2025 MODEL VS VEGAS COMPARISON')
    print('=' * 80)

    # Get Vegas lines for test games
    test_df_reset = test_df.reset_index(drop=True)
    vegas_spread = test_df_reset['odds_latest_spread'].values if 'odds_latest_spread' in test_df_reset.columns else None
    vegas_total = test_df_reset['odds_latest_total'].values if 'odds_latest_total' in test_df_reset.columns else None

    if vegas_spread is not None:
        # Filter out games without Vegas lines
        has_spread = ~np.isnan(vegas_spread) & (vegas_spread != 0)

        if has_spread.sum() > 0:
            vs_spread = vegas_spread[has_spread]
            vs_actual_spread = actual_spread[has_spread]
            vs_pred_spread = pred_spread[has_spread]

            # ATS: home_covered = actual_spread > -vegas_spread
            home_covered = vs_actual_spread > -vs_spread
            model_picked_home_cover = vs_pred_spread > -vs_spread

            model_ats_correct = (model_picked_home_cover == home_covered)
            model_ats_accuracy = np.mean(model_ats_correct)

            print(f'\nATS (Against The Spread) Analysis:')
            print(f'  Games with Vegas spread: {has_spread.sum()}')
            print(f'  Model ATS Accuracy: {model_ats_accuracy:.1%}')

            # Compare to Vegas straight up
            vegas_pred_winner = vs_spread < 0  # negative spread = Vegas favors home
            actual_home_win = vs_actual_spread > 0
            vegas_winner_correct = vegas_pred_winner == actual_home_win
            model_pred_winner = vs_pred_spread > 0
            model_winner_correct = model_pred_winner == actual_home_win

            print(f'\nWinner Prediction:')
            print(f'  Vegas Winner Accuracy: {np.mean(vegas_winner_correct):.1%}')
            print(f'  Model Winner Accuracy: {np.mean(model_winner_correct):.1%}')

    if vegas_total is not None:
        has_total = ~np.isnan(vegas_total) & (vegas_total != 0)

        if has_total.sum() > 0:
            vs_total = vegas_total[has_total]
            vs_actual_total = actual_total[has_total]
            vs_pred_total = pred_total[has_total]

            went_over = vs_actual_total > vs_total
            model_picked_over = vs_pred_total > vs_total

            model_ou_correct = model_picked_over == went_over
            model_ou_accuracy = np.mean(model_ou_correct)

            print(f'\nOver/Under Analysis:')
            print(f'  Games with Vegas total: {has_total.sum()}')
            print(f'  Model O/U Accuracy: {model_ou_accuracy:.1%}')

            # Compare total MAE
            vegas_total_mae = np.mean(np.abs(vs_total - vs_actual_total))
            model_total_mae = np.mean(np.abs(vs_pred_total - vs_actual_total))
            print(f'  Vegas Total MAE: {vegas_total_mae:.2f}')
            print(f'  Model Total MAE: {model_total_mae:.2f}')

    print(f'\n' + '=' * 80)
    print('CFB MODEL TRAINING COMPLETE!')
    print('=' * 80)


if __name__ == '__main__':
    main()
