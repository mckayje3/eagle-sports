import torch

checkpoint = torch.load('models/deep_eagle_nfl_2025.pt', map_location='cpu')
print(f'Model expects {len(checkpoint["feature_cols"])} features')
print('\nDrive-related features in model:')
drive_features = [f for f in checkpoint['feature_cols'] if 'drive' in f.lower()]
print(f'Found {len(drive_features)} drive features:')
for f in drive_features:
    print(f'  - {f}')
