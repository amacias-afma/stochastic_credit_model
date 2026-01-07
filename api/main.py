from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from src.financial_engine import FinancialEngine

app = FastAPI()
# we have a problem here.
model = joblib.load('stochastic_model.pkl')

class FluxRequest(BaseModel):
    fico: float
    dti: float
    orig_rate: float
    market_rate: float
    principal: float
    months: int = 36
    n_sims: int = 2000

@app.post("/calculate_stochastic_flux")
def get_flux(req: FluxRequest):
    # 1. Pre-compute Transition Matrices (Batch Inference)
    # We predict the environment dynamics for the full horizon at once
    rate_spread = req.orig_rate - req.market_rate
    
    # Create batch input: Rows for Current(0) and Late(1) for every month t
    batch_rows = []
    for t in range(req.months):
        base = {'fico': req.fico, 'dti': req.dti, 'loan_age': t, 'rate_spread': rate_spread}
        batch_rows.append({**base, 'current_state': 0})
        batch_rows.append({**base, 'current_state': 1})
    
    # Batch predict
    all_probs = model.predict_proba(pd.DataFrame(batch_rows))
    
    # Reconstruct Matrices T_t
    matrices = []
    for t in range(req.months):
        p_curr = all_probs[2*t]
        p_late = all_probs[2*t+1]
        T = np.vstack([
            p_curr,         # From Current
            p_late,         # From Late
            [0,0,1,0],      # From Default (Absorbing)
            [0,0,0,1]       # From Prepay (Absorbing)
        ])
        matrices.append(T)
    
    # 2. Run Monte Carlo
    engine = FinancialEngine(req.principal, req.orig_rate, req.months)
    cf_matrix = engine.run_monte_carlo(matrices, n_sims=req.n_sims)
    
    # 3. Calculate Statistics (The Flux Analysis)
    expected_flux = np.mean(cf_matrix, axis=0)
    
    # Confidence Intervals (VaR bounds)
    # 5th percentile (Downside Risk / Default Drag) 
    # 95th percentile (Upside Risk / Prepayment Spike)
    ci_lower = np.percentile(cf_matrix, 5, axis=0)
    ci_upper = np.percentile(cf_matrix, 95, axis=0)
    
    return {
        "months": list(range(1, req.months + 1)),
        "input_flux": [engine.pmt] * req.months, # The deterministic baseline
        "expected_flux": expected_flux.tolist(),
        "ci_lower": ci_lower.tolist(),
        "ci_upper": ci_upper.tolist()
    }

@app.post("/optimize_pricing")
def optimize_pricing(req: FluxRequest):
    """
    Solves the Stochastic Control problem:
    u* = argmax E[NPV(u)]
    """
    # Search Grid (e.g., 4% to 12%)
    trial_rates = np.arange(0.04, 0.12, 0.005)
    results = []
    
    for r in trial_rates:
        # 1. Apply Control
        req.orig_rate = float(r)
        
        # 2. Simulate System (Fast Mode)
        # Higher Rate -> Higher Income, but Higher Prepayment/Default Risk
        flux = calculate_stochastic_flux_internal(req, n_sims=500)
        
        # 3. Objective Function (Discounted Cash Flow)
        # Using a risk-free discount curve
        dcf = np.sum(flux['expected_flux'] * (1.03**(-np.array(flux['months'])/12)))
        results.append(dcf)
        
    best_idx = np.argmax(results)
    return {
        "optimal_rate": trial_rates[best_idx],
        "max_npv": results[best_idx],
        "convexity_curve": list(zip(trial_rates, results))
    }