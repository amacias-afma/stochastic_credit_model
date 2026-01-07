# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.calibration import CalibratedClassifierCV

# # --- 1. Synthetic Panel Data Generator ---
# def generate_synthetic_panel_data(n_loans=2000, max_months=60):
#     """
#     Simulates loan paths. 
#     State 0: Current, 1: Late, 2: Default (Absorbing), 3: Prepaid (Absorbing)
#     """
#     records = []
    
#     for loan_id in range(n_loans):
#         # Static Features
#         fico = np.random.normal(720, 50)
#         dti = np.random.normal(0.35, 0.1)
#         orig_rate = np.random.normal(0.04, 0.01) # 4%
        
#         current_state = 0 # Start as Current
        
#         for t in range(max_months):
#             # Dynamic Features (Stochastic Control variables)
#             market_rate = 0.04 + np.random.normal(0, 0.01) # Fluctuating market rate
#             rate_spread = orig_rate - market_rate # Incentive to prepay (Refi)
            
#             # Record observation (State t -> State t+1)
#             records.append({
#                 "fico": fico, "dti": dti, "loan_age": t, "rate_spread": rate_spread,
#                 "current_state": current_state, "target": None # To be filled
#             })
            
#             # --- Transition Logic (The "Ground Truth" Process) ---
#             if current_state in [2, 3]: # Absorbing states
#                 next_state = current_state
#             else:
#                 # Base probabilities [Stay, Late, Default, Prepay]
#                 probs = [0.93, 0.05, 0.005, 0.015]
                
#                 # Adjust intensities based on features
#                 if current_state == 1: # If already Late
#                     probs = [0.30, 0.50, 0.19, 0.01] # High risk of staying late/defaulting
                
#                 # FICO effect: Low FICO -> Higher Default
#                 probs[2] += max(0, (700 - fico) * 0.0005)
#                 # Rate Spread effect: High Spread -> Higher Prepayment
#                 if rate_spread > 0.01: probs[3] += rate_spread * 2.0
                
#                 # Normalize
#                 probs = np.array(probs).clip(min=0)
#                 probs /= probs.sum()
                
#                 next_state = np.random.choice([0, 1, 2, 3], p=probs)
            
#             records[-1]['target'] = next_state
#             current_state = next_state
            
#             if current_state in [2, 3]: break # End of path
                
#     return pd.DataFrame(records)

# # --- 2. Train the Estimator ---
# def train_model(data_type='synthetic'):
#     if data_type == 'synthetic':
#         print("Generating Synthetic Stochastic Data...")
#         df = generate_synthetic_panel_data()
#         X = df[['fico', 'dti', 'loan_age', 'rate_spread', 'current_state']]
#         y = df['target']
    
#     elif data_type == 'real':
#         print("Loading Real Data...")
#         X, y = load_real_data()
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
#     print("Training Multi-State XGBoost...")
#     # 'multi:softprob' gives us the probability distribution vector
#     model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, n_estimators=100)
#     model.fit(X_train, y_train)
    
#     y_pred = model.predict(X_test)
#     print("\nValidation Report:")
#     print(classification_report(y_test, y_pred, target_names=['Current', 'Late', 'Default', 'Prepaid']))
    
#     joblib.dump(model, 'stochastic_model.pkl')
#     print("Model saved to stochastic_model.pkl")

# def load_real_data():
#     df = pd.read_parquet('data/processed/lending_club_panel.parquet')
    
#     # Features for Stochastic Control
#     X = df[['fico', 'dti', 'loan_age', 'rate_spread', 'current_state', 'balance']]
#     y = df['target']
    
#     return X, y


# def train_calibrated_model(base_model, X, y):
#     """
#     Ensures P(Default=0.2) actually means 20% of such loans default.
#     """
#     # Use 'isotonic' for non-parametric calibration
#     calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
#     calibrated.fit(X, y)
#     return calibrated

# if __name__ == "__main__":
#     train_model(data_type='real')

import pandas as pd
import xgboost as xgb
import joblib
from src.validation import TemporalValidator

# Configuration
DATA_PATH = 'data/processed/lending_club_panel.parquet'
OOT_CUTOFF = '2016-01-01' # We train on 2012-2015, Test on 2016+

def train_professional_model():
    print("1. Loading Panel Data...")
    df = pd.read_parquet(DATA_PATH)
    
    # 2. Temporal Split (The Filtration)
    validator = TemporalValidator(df)
    train_df, test_df = validator.split_by_date(OOT_CUTOFF)
    
    features = ['fico', 'dti', 'loan_age', 'rate_spread', 'current_state', 'balance']
    target = 'target'
    
    # 3. Train on "The Past"
    print("3. Training XGBoost (Markov Transition Estimator)...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=20, 
        eval_metric='mlogloss'
    )
    
    # We use the Future set (test_df) only for Early Stopping, not for Gradient Descent
    model.fit(
        train_df[features], 
        train_df[target],
        eval_set=[(train_df[features], train_df[target]), (test_df[features], test_df[target])],
        verbose=50
    )
    
    # 4. Stability Analysis on "The Future"
    print("4. Running Stability Analysis...")
    stability_df = validator.evaluate_stability(model, test_df)
    
    # Save artifacts
    stability_df.to_csv('data/processed/oot_metrics.csv', index=False)
    joblib.dump(model, 'stochastic_model.pkl')
    print("   Metrics saved to data/processed/oot_metrics.csv")

if __name__ == "__main__":
    train_professional_model()