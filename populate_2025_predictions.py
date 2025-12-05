"""
Populate prediction cache with Deep Eagle predictions for 2025 season
Uses the proper DeepEagleFeatureExtractor to match model training
"""
import sqlite3
import torch
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import the actual feature extractor used for training
from deep_eagle_feature_extractor import DeepEagleFeatureExtractor


class DeepEagleModel(torch.nn.Module):
    """Deep Eagle neural network for score prediction - new architecture"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(DeepEagleModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.feature_extractor = torch.nn.Sequential(*layers)

        self.home_score_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

        self.away_score_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        home_score = self.home_score_head(features)
        away_score = self.away_score_head(features)
        return torch.cat([home_score, away_score], dim=1)


class DeepEagleModelOld(torch.nn.Module):
    """Deep Eagle neural network - old architecture (uses 'features' and 'home_head')"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(DeepEagleModelOld, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.features = torch.nn.Sequential(*layers)

        self.home_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

        self.away_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.features(x)
        home_score = self.home_head(features)
        away_score = self.away_head(features)
        return torch.cat([home_score, away_score], dim=1)


def generate_predictions_for_sport(sport, season=2025, weeks=None, backfill=False):
    """Generate predictions for a sport using proper feature extraction"""
    print(f"\n{'='*80}")
    print(f"GENERATING {sport.upper()} 2025 PREDICTIONS")
    print('='*80)

    db_path = f'{sport}_games.db'
    # For CFB, use the 2024 gameday model which has no target leak and good historical features
    # Trained on 2023-2024 data, works for 2025 predictions
    if sport == 'cfb':
        model_path = 'models/deep_eagle_cfb_2024_gameday.pt'
        scaler_path = 'models/deep_eagle_cfb_2024_gameday_scaler.pkl'
    else:
        model_path = f'models/deep_eagle_{sport}_2025.pt'
        scaler_path = f'models/deep_eagle_{sport}_2025_scaler.pkl'

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return []

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    feature_cols = checkpoint['feature_cols']

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    input_dim = len(feature_cols)
    # Check if model uses old architecture (features/home_head) or new (feature_extractor/home_score_head)
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    if 'features.0.weight' in state_dict_keys:
        # Old architecture
        model = DeepEagleModelOld(input_dim)
    else:
        # New architecture
        model = DeepEagleModel(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model with {input_dim} features")
    print(f"Features: {feature_cols[:5]}...")

    # Get games
    conn = sqlite3.connect(db_path)

    # Get ALL games for 2025 (we need to generate predictions for all)
    query = '''
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.date,
            g.home_team_id,
            g.away_team_id,
            ht.display_name as home_team,
            at.display_name as away_team,
            g.neutral_site,
            g.conference_game,
            g.temperature,
            g.wind_speed,
            g.is_dome,
            g.home_score,
            g.away_score,
            g.completed
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.season = ?
        AND g.home_team_id > 0 AND g.away_team_id > 0
    '''
    params = [season]

    if weeks:
        week_list = ','.join(map(str, weeks))
        query += f' AND g.week IN ({week_list})'

    query += ' ORDER BY g.week, g.date'

    games_df = pd.read_sql_query(query, conn, params=params)
    print(f"Found {len(games_df)} games")

    if len(games_df) == 0:
        conn.close()
        return []

    # Use the proper feature extractor
    extractor = DeepEagleFeatureExtractor(db_path, sport)
    all_predictions = []

    for week in sorted(games_df['week'].unique()):
        week_games = games_df[games_df['week'] == week]
        print(f"\nWeek {week}: {len(week_games)} games")

        for _, game in week_games.iterrows():
            try:
                # Create a game series that looks like what _extract_game_features expects
                game_data = pd.Series({
                    'game_id': game['game_id'],
                    'season': game['season'],
                    'week': game['week'],
                    'date': game['date'],
                    'home_team_id': game['home_team_id'],
                    'away_team_id': game['away_team_id'],
                    'home_score': game['home_score'] if game['completed'] else 0,
                    'away_score': game['away_score'] if game['completed'] else 0,
                    'neutral_site': game['neutral_site'],
                    'conference_game': game['conference_game'],
                    'temperature': game['temperature'],
                    'wind_speed': game['wind_speed'],
                    'is_dome': game['is_dome']
                })

                # Use the same feature extraction as training
                features = extractor._extract_game_features(game_data)

                if features is None:
                    print(f"  Skipping game {game['game_id']} - no features")
                    continue

                # Build feature vector in the correct order
                feature_vector = []
                for col in feature_cols:
                    if col in features:
                        val = features[col]
                        feature_vector.append(float(val) if val is not None else 0.0)
                    else:
                        feature_vector.append(0.0)

                X = np.array([feature_vector], dtype=np.float32)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                X = scaler.transform(X)

                X_tensor = torch.FloatTensor(X).to(device)

                with torch.no_grad():
                    pred = model(X_tensor).cpu().numpy()[0]

                pred_home = float(pred[0])
                pred_away = float(pred[1])
                pred_spread = pred_home - pred_away
                pred_total = pred_home + pred_away

                # Win probability (simple logistic)
                win_prob = 1 / (1 + np.exp(-pred_spread / 7))

                # Confidence based on data availability
                home_games = features.get('home_hist_games_played', 0)
                away_games = features.get('away_hist_games_played', 0)
                games_played = home_games + away_games
                base_conf = 0.70 + min(games_played / 20, 0.20)
                spread_conf = min(abs(pred_spread) / 40, 0.10)
                confidence = min(base_conf + spread_conf, 0.95)

                # Get odds from features
                vegas_spread = features.get('odds_latest_spread', 0) or features.get('odds_opening_spread', 0)
                vegas_total = features.get('odds_latest_total', 0) or features.get('odds_opening_total', 0)

                prediction = {
                    'game_id': game['game_id'],
                    'sport': sport.upper(),
                    'season': season,
                    'week': int(game['week']),
                    'game_date': game['date'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'predicted_home_score': round(pred_home, 1),
                    'predicted_away_score': round(pred_away, 1),
                    'predicted_spread': round(pred_spread, 1),
                    'predicted_total': round(pred_total, 1),
                    'home_win_probability': round(win_prob, 3),
                    'vegas_spread': vegas_spread if vegas_spread else 0,
                    'vegas_total': vegas_total if vegas_total else 0,
                    'game_completed': int(game['completed']),
                    'actual_home_score': game['home_score'] if game['completed'] else None,
                    'actual_away_score': game['away_score'] if game['completed'] else None,
                    'confidence': round(confidence, 3),
                    'spread_low': round(pred_spread - 5, 1),
                    'spread_high': round(pred_spread + 5, 1),
                    'total_low': round(pred_total - 8, 1),
                    'total_high': round(pred_total + 8, 1)
                }

                all_predictions.append(prediction)

            except Exception as e:
                print(f"  Error on game {game['game_id']}: {e}")
                import traceback
                traceback.print_exc()
                continue

    extractor.close()
    conn.close()

    print(f"\nGenerated {len(all_predictions)} predictions")
    return all_predictions


def save_predictions_to_cache(predictions):
    """Save predictions to users.db prediction_cache"""
    if not predictions:
        print("No predictions to save")
        return

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Group by sport and week for summary
    summary = {}

    for pred in predictions:
        sport = pred['sport']
        week = pred['week']
        key = f"{sport} Week {week}"
        summary[key] = summary.get(key, 0) + 1

        # Delete existing prediction for this game
        cursor.execute(
            'DELETE FROM prediction_cache WHERE game_id = ? AND sport = ?',
            (pred['game_id'], pred['sport'])
        )

        # Insert new prediction
        cursor.execute('''
            INSERT INTO prediction_cache (
                sport, game_id, season, week, game_date,
                home_team, away_team,
                predicted_home_score, predicted_away_score,
                predicted_spread, predicted_total,
                home_win_probability,
                vegas_spread, vegas_total,
                game_completed, actual_home_score, actual_away_score,
                created_at, confidence,
                spread_low, spread_high, total_low, total_high
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pred['sport'],
            pred['game_id'],
            pred['season'],
            pred['week'],
            pred['game_date'],
            pred['home_team'],
            pred['away_team'],
            pred['predicted_home_score'],
            pred['predicted_away_score'],
            pred['predicted_spread'],
            pred['predicted_total'],
            pred['home_win_probability'],
            pred['vegas_spread'],
            pred['vegas_total'],
            pred['game_completed'],
            pred['actual_home_score'],
            pred['actual_away_score'],
            datetime.now().isoformat(),
            pred['confidence'],
            pred['spread_low'],
            pred['spread_high'],
            pred['total_low'],
            pred['total_high']
        ))

    conn.commit()
    conn.close()

    print("\nSaved predictions to cache:")
    for key, count in sorted(summary.items()):
        print(f"  {key}: {count} games")


def push_to_github():
    """Commit and push users.db to GitHub"""
    import subprocess

    print(f"\n{'='*80}")
    print("PUSHING TO GITHUB")
    print('='*80)

    try:
        # Check if there are changes to users.db
        result = subprocess.run(
            ['git', 'status', '--porcelain', 'users.db'],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if not result.stdout.strip():
            print("No changes to users.db")
            return True

        # Add users.db
        subprocess.run(['git', 'add', 'users.db'], check=True,
                      cwd=os.path.dirname(os.path.abspath(__file__)))

        # Commit
        commit_msg = f"Update predictions cache - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True,
                      cwd=os.path.dirname(os.path.abspath(__file__)))

        # Push
        subprocess.run(['git', 'push', 'origin', 'main'], check=True,
                      cwd=os.path.dirname(os.path.abspath(__file__)))

        print("Successfully pushed to GitHub!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}")
        return False
    except Exception as e:
        print(f"Error pushing to GitHub: {e}")
        return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate 2025 predictions')
    parser.add_argument('--sport', choices=['nfl', 'cfb', 'both'], default='both')
    parser.add_argument('--weeks', type=str, help='Comma-separated list of weeks (e.g., 14,15)')
    parser.add_argument('--backfill', action='store_true', help='Include completed games')
    parser.add_argument('--all-weeks', action='store_true', help='Generate for all 2025 weeks')
    parser.add_argument('--push', action='store_true', help='Auto-push users.db to GitHub after generating')

    args = parser.parse_args()

    weeks = None
    if args.weeks:
        weeks = [int(w.strip()) for w in args.weeks.split(',')]
    elif args.all_weeks:
        # All possible weeks
        weeks = list(range(0, 18))

    all_predictions = []

    if args.sport in ['nfl', 'both']:
        predictions = generate_predictions_for_sport('nfl', 2025, weeks, args.backfill)
        all_predictions.extend(predictions)

    if args.sport in ['cfb', 'both']:
        predictions = generate_predictions_for_sport('cfb', 2025, weeks, args.backfill)
        all_predictions.extend(predictions)

    save_predictions_to_cache(all_predictions)

    print(f"\n{'='*80}")
    print("PREDICTION GENERATION COMPLETE")
    print(f"Total predictions: {len(all_predictions)}")
    print('='*80)

    # Auto-push if requested
    if args.push and all_predictions:
        push_to_github()


if __name__ == '__main__':
    main()
