"""
Quick test to verify Deep-Eagle can work with CFB data
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlite3
import pandas as pd
import numpy as np

# Check PyTorch availability
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} installed")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch not installed")
    print("   Install with: pip install torch")
    sys.exit(1)

# Check Deep-Eagle availability
try:
    sys.path.insert(0, r'C:\Users\jbeast\documents\coding\deep')
    from core import LSTMModel, TimeSeriesDataset
    print("✅ Deep-Eagle core modules imported successfully")
except ImportError as e:
    print(f"❌ Deep-Eagle import failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("CFB DATA COMPATIBILITY CHECK")
print("=" * 80)

# Load CFB data from database
conn = sqlite3.connect('cfb_games.db')

# Get game data with team stats
query = '''
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.date,
        g.home_team_id,
        g.away_team_id,
        g.home_score,
        g.away_score,
        g.completed,
        (g.home_score > g.away_score) as home_win,
        (g.home_score - g.away_score) as point_differential,
        (g.home_score + g.away_score) as total_points
    FROM games g
    WHERE g.season IN (2024, 2025)
        AND g.completed = 1
    ORDER BY g.season, g.week, g.date
    LIMIT 100
'''

df = pd.read_sql_query(query, conn)
print(f"\n✅ Loaded {len(df)} completed games from database")
print(f"   Seasons: {df['season'].unique()}")
print(f"   Weeks: {df['week'].min()} - {df['week'].max()}")

# Check if we have enough data for sequence learning
min_games_needed = 15  # Deep-Eagle default sequence length
print(f"\n📊 Data Requirements:")
print(f"   Minimum games needed: {min_games_needed}")
print(f"   Games available: {len(df)}")
if len(df) >= min_games_needed * 2:
    print(f"   ✅ Sufficient data for training")
else:
    print(f"   ⚠️  Limited data - may need more games")

# Sample feature engineering (simplified)
print(f"\n🔧 Feature Engineering Test:")

# Create rolling averages (simplified version)
features = []
targets = []

for idx in range(15, len(df)):  # Start at game 15 to have 15-game history
    # Get last 15 games
    recent_games = df.iloc[idx-15:idx]

    # Calculate rolling stats
    roll_3_pts = recent_games.tail(3)['total_points'].mean()
    roll_5_pts = recent_games.tail(5)['total_points'].mean()
    roll_10_pts = recent_games.tail(10)['total_points'].mean()
    roll_3_diff = recent_games.tail(3)['point_differential'].mean()

    # Create feature vector
    feature_vec = [
        roll_3_pts,
        roll_5_pts,
        roll_10_pts,
        roll_3_diff
    ]

    features.append(feature_vec)
    targets.append(df.iloc[idx]['total_points'])  # Predict total points

features = np.array(features)
targets = np.array(targets)

print(f"   Features shape: {features.shape}")
print(f"   Targets shape: {targets.shape}")
print(f"   ✅ Feature engineering successful")

# Test Deep-Eagle dataset creation
print(f"\n🧪 Deep-Eagle Dataset Test:")
try:
    dataset = TimeSeriesDataset(
        data=features,
        targets=targets,
        sequence_length=5,  # Use last 5 games for this test
        forecast_horizon=1
    )
    print(f"   ✅ TimeSeriesDataset created")
    print(f"   Dataset length: {len(dataset)}")

    # Get a sample
    sample_x, sample_y = dataset[0]
    print(f"   Sample X shape: {sample_x.shape}")
    print(f"   Sample Y shape: {sample_y.shape}")

except Exception as e:
    print(f"   ❌ Dataset creation failed: {e}")
    conn.close()
    sys.exit(1)

# Test model creation
print(f"\n🏗️  Deep-Eagle Model Test:")
try:
    model = LSTMModel(
        input_dim=features.shape[1],
        hidden_dim=128,
        output_dim=1,
        num_layers=2,
        dropout=0.2
    )
    print(f"   ✅ LSTM model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    test_input = torch.randn(1, 5, features.shape[1])  # batch=1, seq=5, features
    with torch.no_grad():
        test_output = model(test_input)
    print(f"   ✅ Forward pass successful")
    print(f"   Output shape: {test_output.shape}")

except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    conn.close()
    sys.exit(1)

conn.close()

print("\n" + "=" * 80)
print("COMPATIBILITY SUMMARY")
print("=" * 80)
print("\n✅ Deep-Eagle is COMPATIBLE with CFB data")
print("\nNext Steps:")
print("  1. Create comprehensive feature engineering script")
print("  2. Add team-specific rolling stats")
print("  3. Include opponent strength, h2h history")
print("  4. Train full model on 2024+2025 data")
print("  5. Validate against Vegas lines")
print("\n" + "=" * 80)
