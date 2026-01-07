import pandas as pd
import numpy as np
import requests
import io
from scipy import stats

class MarketEnvironment:
    def __init__(self, tenor='30Y'):
        # FRED Series ID: 30-Year Fixed Rate Mortgage Average
        # We use direct CSV download for robustness (pandas_datareader can be flaky)
        self.url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE{tenor[:-1]}US"
        self.data = None

    def fetch_data(self):
        """Downloads historical data directly from St. Louis Fed (FRED)."""
        print(f"Fetching {self.url}...")
        try:
            resp = requests.get(self.url)
            resp.raise_for_status()
            
            # Parse CSV

            # df = pd.read_csv(io.StringIO(resp.content.decode('utf-8')))
            # print(df.head())

            df = pd.read_csv(io.StringIO(resp.content.decode('utf-8')), parse_dates=['observation_date'])
            df.set_index('observation_date', inplace=True)
            
            # Data comes as percent (e.g., 6.5). Convert to decimal (0.065)
            # Resample to End-of-Month to match loan performance frequency
            self.data = df.resample('ME').last() / 100.0
            self.data.columns = ['rate']
            
            return self.data.dropna()
        except Exception as e:
            print(f"Error downloading market data: {e}")
            return pd.DataFrame()

    def calibrate_vasicek(self):
        """
        Calibrates the Ornstein-Uhlenbeck (Vasicek) process parameters.
        
        SDE:       dr_t = kappa * (theta - r_t)dt + sigma * dW_t
        Discrete:  r_{t+1} - r_t = alpha + beta * r_t + epsilon
        
        Mapping OLS -> SDE:
        - kappa = -beta / dt
        - theta = -alpha / beta
        - sigma = std(epsilon) / sqrt(dt)
        """
        if self.data is None: self.fetch_data()
        
        rates = self.data['rate'].values
        dt = 1/12 # Monthly time step
        
        # Prepare Regression: Y = (Rate_Next - Rate_Curr), X = Rate_Curr
        r_curr = rates[:-1]
        r_next = rates[1:]
        dr = r_next - r_curr
        
        # Perform Linear Regression (OLS)
        # dr = intercept + slope * r_curr
        slope, intercept, r_value, p_value, std_err = stats.linregress(r_curr, dr)
        
        # Recover SDE Parameters
        kappa = -slope / dt
        theta = -intercept / slope
        
        # Calculate residuals to estimate volatility
        residuals = dr - (intercept + slope * r_curr)
        sigma = np.std(residuals) / np.sqrt(dt)
        
        return {
            "kappa": kappa,        # Speed of Mean Reversion
            "theta": theta,        # Long Term Mean Rate
            "sigma": sigma,        # Volatility
            "current_rate": rates[-1],
            "r_squared": r_value**2,
            "history": self.data
        }