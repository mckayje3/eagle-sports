import pickle

data = pickle.load(open('cfb_model_v2_scaler.pkl', 'rb'))
print('Feature columns:')
for i, col in enumerate(data['feature_columns']):
    print(f'{i+1}. {col}')
