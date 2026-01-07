import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

class TemporalValidator:
    def __init__(self, df, date_col='date', target_col='target'):
        """
        Manages Time-Series splits for Non-Stationary processes.
        """
        self.df = df.sort_values(date_col)
        self.date_col = date_col
        self.target_col = target_col
        
    def split_by_date(self, cutoff_date):
        """
        Splits panel into:
        - In-Time (Train): Observations BEFORE cutoff (The Past)
        - Out-of-Time (Test): Observations AFTER cutoff (The Future)
        """
        cutoff = pd.to_datetime(cutoff_date)
        
        # Ensure date format
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        
        train = self.df[self.df[self.date_col] < cutoff].copy()
        test = self.df[self.df[self.date_col] >= cutoff].copy()
        
        print(f"--- OOT Split (Cutoff: {cutoff.date()}) ---")
        print(f"Training Set: {len(train):,} obs ({train[self.date_col].min().date()} to {train[self.date_col].max().date()})")
        print(f"Testing Set:  {len(test):,} obs ({test[self.date_col].min().date()} to {test[self.date_col].max().date()})")
        
        return train, test

    def evaluate_stability(self, model, test_data):
        """
        Calculates performance metrics grouped by month to detect Regime Shifts.
        """
        results = []
        
        # Group by Month to create the Time Series of performance
        grouped = test_data.groupby(pd.Grouper(key=self.date_col, freq='M'))
        
        for date, group in grouped:
            if len(group) < 100: continue
            
            X = group[['fico', 'dti', 'loan_age', 'rate_spread', 'current_state', 'balance']]
            y_true = group[self.target_col]
            
            # Predict Probabilities
            y_prob = model.predict_proba(X)
            
            # 1. Log Loss (Likelihood Fit)
            loss = log_loss(y_true, y_prob, labels=[0,1,2,3])
            
            # 2. Default AUC (Discrimination Stability)
            # We isolate Class 2 (Default) to see if the model correctly ranks risk
            y_def_binary = (y_true == 2).astype(int)
            
            try:
                if y_def_binary.sum() > 5: # Need at least a few defaults to calc AUC
                    auc = roc_auc_score(y_def_binary, y_prob[:, 2])
                else:
                    auc = np.nan
            except:
                auc = np.nan

            # 3. Calibration (Flux Accuracy)
            # Crucial for your project: Is the Expected Default Rate equal to the Actual?
            expected_dr = y_prob[:, 2].mean()
            actual_dr = y_def_binary.mean()
            
            results.append({
                'date': date,
                'log_loss': loss,
                'auc_default': auc,
                'expected_dr': expected_dr,
                'actual_dr': actual_dr,
                'obs_count': len(group)
            })
            
        return pd.DataFrame(results)