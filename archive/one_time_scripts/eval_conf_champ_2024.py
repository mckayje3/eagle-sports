"""
Evaluate model performance on 2024 Conference Championship Week
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import sqlite3


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
        self.home_score_head = nn.Sequential(nn.Linear(prev_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.away_score_head = nn.Sequential(nn.Linear(prev_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        features = self.feature_extractor(x)
        return torch.cat([self.home_score_head(features), self.away_score_head(features)], dim=1)


def main():
    # Load features
    df = pd.read_csv('cfb_2023_2024_deep_eagle_features.csv')
    week15 = df[(df['season'] == 2024) & (df['week'] == 15)].copy()

    print('=' * 80)
    print('CFB 2024 CONFERENCE CHAMPIONSHIP WEEK (Week 15) ANALYSIS')
    print('=' * 80)
    print(f'Games: {len(week15)}')

    # Get team names
    conn = sqlite3.connect('cfb_games.db')
    query = '''
    SELECT g.game_id, g.week, g.date,
           ht.name as home_team, at.name as away_team,
           g.home_score, g.away_score
    FROM games g
    JOIN teams ht ON g.home_team_id = ht.team_id
    JOIN teams at ON g.away_team_id = at.team_id
    WHERE g.season = 2024 AND g.week = 15
    ORDER BY g.date
    '''
    games_info = pd.read_sql_query(query, conn)
    conn.close()

    print('\n2024 Conference Championship Games:')
    for _, row in games_info.iterrows():
        print(f"  {row['away_team']} @ {row['home_team']}: {row['away_score']}-{row['home_score']}")

    # Train/test split - train on 2023 + 2024 weeks 1-14, test on week 15
    print('\n' + '=' * 80)
    print('TRAINING MODEL (2023 + 2024 weeks 1-14) TO TEST ON WEEK 15')
    print('=' * 80)

    train_df = df[(df['season'] == 2023) | ((df['season'] == 2024) & (df['week'] < 15))]
    test_df = week15

    print(f'Training games: {len(train_df)}')
    print(f'Test games (Week 15): {len(test_df)}')

    # Features
    exclude_cols = [
        'game_id', 'season', 'week', 'home_team_id', 'away_team_id',
        'home_score', 'away_score', 'point_spread', 'total_points', 'home_win'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[['home_score', 'away_score']].values
    y_test = test_df[['home_score', 'away_score']].values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = DeepEagleModel(input_dim=X_train.shape[1])
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print('Training model...')
    best_loss = float('inf')
    best_state = None
    for epoch in range(150):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test_t), torch.FloatTensor(y_test)).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # Get predictions
    with torch.no_grad():
        predictions = model(X_test_t).numpy()

    pred_home = predictions[:, 0]
    pred_away = predictions[:, 1]
    pred_spread = pred_home - pred_away
    pred_total = pred_home + pred_away

    actual_home = y_test[:, 0]
    actual_away = y_test[:, 1]
    actual_spread = actual_home - actual_away
    actual_total = actual_home + actual_away

    vegas_spread = test_df['odds_latest_spread'].values
    vegas_total = test_df['odds_latest_total'].values

    print('\n' + '=' * 80)
    print('2024 CONFERENCE CHAMPIONSHIP WEEK - GAME BY GAME')
    print('=' * 80)

    # Detailed results
    for i in range(len(test_df)):
        game_id = test_df.iloc[i]['game_id']
        game_info = games_info[games_info['game_id'] == game_id].iloc[0]

        vs = vegas_spread[i] if not np.isnan(vegas_spread[i]) else None
        vt = vegas_total[i] if not np.isnan(vegas_total[i]) else None

        print(f"\n{game_info['away_team']} @ {game_info['home_team']}")
        print(f"  Actual Score: {int(actual_away[i])}-{int(actual_home[i])} (Spread: {actual_spread[i]:+.0f}, Total: {int(actual_total[i])})")
        print(f"  Model Pred:   {pred_away[i]:.0f}-{pred_home[i]:.0f} (Spread: {pred_spread[i]:+.1f}, Total: {pred_total[i]:.0f})")
        if vs:
            print(f"  Vegas Line:   Spread: {vs:+.1f}, Total: {vt:.1f}")

            # ATS result
            home_covered = actual_spread[i] > -vs
            model_picked_cover = pred_spread[i] > -vs
            ats_result = "COVER" if model_picked_cover == home_covered else "MISS"

            # O/U result
            went_over = actual_total[i] > vt
            model_picked_over = pred_total[i] > vt
            ou_result = "HIT" if model_picked_over == went_over else "MISS"

            print(f"  Model ATS: {ats_result} | O/U: {ou_result}")

    # Summary
    print('\n' + '=' * 80)
    print('SUMMARY METRICS')
    print('=' * 80)

    has_spread = ~np.isnan(vegas_spread) & (vegas_spread != 0)
    has_total = ~np.isnan(vegas_total) & (vegas_total != 0)

    if has_spread.sum() > 0:
        vs = vegas_spread[has_spread]
        act_sp = actual_spread[has_spread]
        pred_sp = pred_spread[has_spread]

        # ATS
        home_covered = act_sp > -vs
        model_pick_cover = pred_sp > -vs
        model_ats = np.mean(model_pick_cover == home_covered)

        # Winner
        vegas_pick_home = vs < 0
        model_pick_home = pred_sp > 0
        actual_home_won = act_sp > 0

        vegas_winner_acc = np.mean(vegas_pick_home == actual_home_won)
        model_winner_acc = np.mean(model_pick_home == actual_home_won)

        n = has_spread.sum()
        print(f'Games with spread: {n}')
        print(f'')
        print(f'WINNER PREDICTION:')
        print(f'  Vegas: {vegas_winner_acc:.1%} ({int(vegas_winner_acc * n)}/{n})')
        print(f'  Model: {model_winner_acc:.1%} ({int(model_winner_acc * n)}/{n})')
        print(f'')
        print(f'ATS (Against The Spread):')
        print(f'  Model: {model_ats:.1%} ({int(model_ats * n)}/{n})')

    if has_total.sum() > 0:
        vt = vegas_total[has_total]
        act_tot = actual_total[has_total]
        pred_tot = pred_total[has_total]

        went_over = act_tot > vt
        model_pick_over = pred_tot > vt

        model_ou = np.mean(model_pick_over == went_over)
        n = has_total.sum()

        print(f'')
        print(f'OVER/UNDER:')
        print(f'  Model: {model_ou:.1%} ({int(model_ou * n)}/{n})')


if __name__ == '__main__':
    main()
