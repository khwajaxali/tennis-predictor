# test_raw_model.py
from predictor import TennisPredictor
import pandas as pd
import numpy as np

tp = TennisPredictor(models_path='./models', data_path='./data')

# Get stats
nadal = tp._get_player_stats("Rafael Nadal", "Clay")
djokovic = tp._get_player_stats("Novak Djokovic", "Clay")

# Build features without H2H
features = tp._build_feature_vector(nadal, djokovic, "Clay", "G", "F")
features['h2h_total'] = 0
features['h2h_diff'] = 0
features['h2h_win_rate'] = 0.5

X = pd.DataFrame([features])[tp.feature_cols].values
X_scaled = tp.scaler.transform(X)

xgb_prob = tp.xgb_model.predict_proba(X_scaled)[0, 1]
lgb_prob = tp.lgb_model.predict_proba(X_scaled)[0, 1]
raw_prob = 0.55 * xgb_prob + 0.45 * lgb_prob

print(f"Raw model prediction (no H2H): Nadal {raw_prob:.1%}")