"""
Populate prediction cache with Deep Eagle predictions for NBA 2024-25 season
Uses the proper NBAFeatureExtractor to match model training
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
from nba_feature_extractor import NBAFeatureExtractor


class DeepEagleModel(torch.nn.Module):
    """Deep Eagle neural network for score prediction"""

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


def generate_nba_predictions(season=2024, upcoming_only=False, backfill=False):
    """Generate predictions for NBA using proper feature extraction"""
    print(f"\n{'='*80}")
    print(f"GENERATING NBA {season}-{str(season+1)[-2:]} PREDICTIONS")
    print('='*80)

    db_path = 'nba_games.db'
    model_path = f'models/deep_eagle_nba_{season}.pt'
    scaler_path = f'models/deep_eagle_nba_{season}_scaler.pkl'

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
    model = DeepEagleModel(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model with {input_dim} features")
    print(f"Features: {feature_cols[:5]}...")

    # Get games
    conn = sqlite3.connect(db_path)

    # Get games for the season
    query = '''
        SELECT
            g.game_id,
            g.season,
            g.date,
            g.home_team_id,
            g.away_team_id,
            ht.display_name as home_team,
            at.display_name as away_team,
            g.home_score,
            g.away_score,
            g.completed,
            g.attendance
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.season = ?
        AND g.home_team_id > 0 AND g.away_team_id > 0
    '''
    params = [season]

    if upcoming_only:
        query += ' AND g.completed = 0'

    query += ' ORDER BY g.date'

    games_df = pd.read_sql_query(query, conn, params=params)
    print(f"Found {len(games_df)} games")

    if len(games_df) == 0:
        conn.close()
        return []

    # Use the proper feature extractor
    extractor = NBAFeatureExtractor(db_path)
    all_predictions = []

    for idx, game in games_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processing game {idx + 1}/{len(games_df)}...")

        try:
            # Create a game series that looks like what _extract_game_features expects
            game_data = pd.Series({
                'game_id': game['game_id'],
                'season': game['season'],
                'date': game['date'],
                'home_team_id': game['home_team_id'],
                'away_team_id': game['away_team_id'],
                'home_score': game['home_score'] if game['completed'] else 0,
                'away_score': game['away_score'] if game['completed'] else 0,
                'venue_name': None,
                'attendance': game['attendance']
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

            # Win probability (simple logistic, but adjusted for NBA scoring)
            win_prob = 1 / (1 + np.exp(-pred_spread / 10))

            # Confidence based on data availability
            home_games = features.get('home_hist_games_played', 0)
            away_games = features.get('away_hist_games_played', 0)
            games_played = home_games + away_games
            base_conf = 0.70 + min(games_played / 30, 0.20)  # NBA has 82 games
            spread_conf = min(abs(pred_spread) / 30, 0.10)
            confidence = min(base_conf + spread_conf, 0.95)

            # Get odds from features
            vegas_spread = features.get('odds_latest_spread', 0) or features.get('odds_opening_spread', 0)
            vegas_total = features.get('odds_latest_total', 220) or features.get('odds_opening_total', 220)

            # NBA uses dates, not weeks - calculate a "week" equivalent
            game_date = pd.to_datetime(game['date']).tz_localize(None) if pd.to_datetime(game['date']).tzinfo else pd.to_datetime(game['date'])
            season_start = pd.Timestamp(season, 10, 15)  # NBA season typically starts mid-October
            week_num = max(0, (game_date - season_start).days // 7)

            prediction = {
                'game_id': game['game_id'],
                'sport': 'NBA',
                'season': season,
                'week': week_num,  # Week since season start
                'game_date': game['date'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'predicted_home_score': round(pred_home, 1),
                'predicted_away_score': round(pred_away, 1),
                'predicted_spread': round(pred_spread, 1),
                'predicted_total': round(pred_total, 1),
                'home_win_probability': round(win_prob, 3),
                'vegas_spread': vegas_spread if vegas_spread else 0,
                'vegas_total': vegas_total if vegas_total else 220,
                'game_completed': int(game['completed']),
                'actual_home_score': game['home_score'] if game['completed'] else None,
                'actual_away_score': game['away_score'] if game['completed'] else None,
                'confidence': round(confidence, 3),
                'spread_low': round(pred_spread - 7, 1),  # NBA has higher variance
                'spread_high': round(pred_spread + 7, 1),
                'total_low': round(pred_total - 12, 1),
                'total_high': round(pred_total + 12, 1)
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
    for key, count in sorted(summary.items())[:20]:  # Show first 20
        print(f"  {key}: {count} games")
    if len(summary) > 20:
        print(f"  ... and {len(summary) - 20} more weeks")


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
        commit_msg = f"Update NBA predictions cache - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
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

    parser = argparse.ArgumentParser(description='Generate NBA predictions')
    parser.add_argument('--season', type=int, default=2024, help='Season year (e.g., 2024 for 2024-25)')
    parser.add_argument('--upcoming-only', action='store_true', help='Only generate for upcoming games')
    parser.add_argument('--backfill', action='store_true', help='Include completed games')
    parser.add_argument('--push', action='store_true', help='Auto-push users.db to GitHub after generating')

    args = parser.parse_args()

    predictions = generate_nba_predictions(args.season, args.upcoming_only, args.backfill)
    save_predictions_to_cache(predictions)

    print(f"\n{'='*80}")
    print("NBA PREDICTION GENERATION COMPLETE")
    print(f"Total predictions: {len(predictions)}")
    print('='*80)

    # Auto-push if requested
    if args.push and predictions:
        push_to_github()


if __name__ == '__main__':
    main()
