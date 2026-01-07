import numpy as np
import xgboost as xgb
import joblib
from sklearn.base import BaseEstimator

class StochasticHybridModel(BaseEstimator):
    def __init__(self):
        # 1. The Topological Router (Classifier)
        # Determines the Regime of the loan at time t
        # Class 0: Default (Absorbing -> Payment = Recovery)
        # Class 1: Continuing (Transient -> Payment = Scheduled + Partial)
        # Class 2: Prepayment (Absorbing -> Payment = Balance)
        self.router = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5
        )
        
        # 2. The Flux Expert (Regressor)
        # Conditional on Continuing (Class 1), predicts the Fraction of Balance paid.
        # We model the Log-Odds to ensure mathematical validity.
        self.expert = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=150,
            learning_rate=0.05
        )

    def fit(self, X, y_fraction):
        """
        X: Features DataFrame
        y_fraction: Actual Payment / Balance (Values in [0, 1])
        """
        # --- A. Define Regimes (Topology) ---
        labels = np.zeros(len(y_fraction), dtype=int)
        epsilon = 1e-4
        
        # Absorbing States
        labels[y_fraction <= epsilon] = 0           # Default
        labels[y_fraction >= (1.0 - epsilon)] = 2   # Prepay
        
        # Transient State (Scheduled + Partial)
        mask_cont = (y_fraction > epsilon) & (y_fraction < (1.0 - epsilon))
        labels[mask_cont] = 1 
        
        print(f"Training Router... Regimes: Def={sum(labels==0)}, Cont={sum(labels==1)}, Pre={sum(labels==2)}")
        self.router.fit(X, labels)
        
        # --- B. Train Flux Expert (Control) ---
        # We only train the regressor on loans that represent the continuous flow
        if mask_cont.sum() > 0:
            print(f"Training Expert on {mask_cont.sum()} continuing loans...")
            X_cont = X[mask_cont]
            y_cont = y_fraction[mask_cont]
            
            # Logit Transform: Map (0,1) -> (-inf, +inf)
            # This is critical for PhD rigor: standard regression can predict negative payments.
            y_safe = np.clip(y_cont, 1e-5, 1.0 - 1e-5)
            y_logit = np.log(y_safe / (1.0 - y_safe))
            
            self.expert.fit(X_cont, y_logit)
            
        return self

    def predict_components(self, X):
        """
        Returns parameters for the Stochastic Differential Equation (SDE) simulation.
        """
        # 1. Regime Probabilities [P_def, P_cont, P_pre]
        probs = self.router.predict_proba(X)
        
        # 2. Continuous Fraction Mean (Inverse Logit)
        logit_pred = self.expert.predict(X)
        pred_fraction = 1.0 / (1.0 + np.exp(-logit_pred))
        
        return probs, pred_fraction
    
    def save(self, path='hybrid_model.pkl'):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path='hybrid_model.pkl'):
        return joblib.load(path)
