import numpy as np
import pandas as pd

class HybridFinancialEngine:
    def __init__(self, principal, annual_rate, months):
        self.principal = principal
        self.monthly_rate = annual_rate / 12.0
        self.months = months

    def run_simulation(self, model, initial_features, n_sims=2000, recovery_rate=0.60):
        # 1. Initialize State
        # Replicate initial conditions for N particles
        balances = np.full(n_sims, self.principal, dtype=float)
        active_mask = np.ones(n_sims, dtype=bool)
        
        # Result Storage (Cash Flow Matrix)
        flux_matrix = np.zeros((n_sims, self.months))
        
        # Base Features (Vectorized)
        # We convert the single feature dict into a DataFrame repeated N times
        df_batch = pd.DataFrame([initial_features] * n_sims)
        
        for t in range(self.months):
            if not np.any(active_mask): break
            
            # Update Dynamic Features
            df_batch['loan_age'] = t
            df_batch['balance'] = balances
            
            # 2. Model Inference (Batch)
            # Get Probabilities and Expected Payment Fraction
            probs, pred_fraction = model.predict_components(df_batch)
            
            # 3. Stochastic Regime Sampling
            rng = np.random.rand(n_sims)
            
            # Define Regimes (Vectorized Masking)
            # Default: [0, p_def)
            is_def = (rng < probs[:, 0]) & active_mask
            # Prepay: [1-p_pre, 1]
            is_pre = (rng > (1.0 - probs[:, 2])) & active_mask
            # Continue: Everything else
            is_cont = active_mask & (~is_def) & (~is_pre)
            
            # 4. Calculate Cash Flows (The Impulse)
            step_flux = np.zeros(n_sims)
            
            # Case A: Default (Recovery)
            step_flux[is_def] = balances[is_def] * recovery_rate
            balances[is_def] = 0
            active_mask[is_def] = False
            
            # Case B: Prepayment (Full Balance)
            step_flux[is_pre] = balances[is_pre]
            balances[is_pre] = 0
            active_mask[is_pre] = False
            
            # Case C: Continuing (Partial Payment)
            if np.any(is_cont):
                # Add stochastic noise to the Expert's mean prediction
                # This simulates borrower heterogeneity (Sigma)
                noise = np.random.normal(0, 0.002, size=is_cont.sum())
                realized_frac = np.clip(pred_fraction[is_cont] + noise, 0, 0.99)
                
                # Total Cash Flow = Balance * Fraction
                payment = balances[is_cont] * realized_frac
                step_flux[is_cont] = payment
                
                # 5. Update Balances (Crucial Step)
                # We must subtract ONLY Principal, not Interest
                interest_accrued = balances[is_cont] * self.monthly_rate
                principal_pay = payment - interest_accrued
                
                # Floor principal pay at 0 (negative amortization check)
                balances[is_cont] -= np.maximum(principal_pay, 0)
                
                # Clean up tiny balances (dust)
                is_dust = (balances[is_cont] < 10)
                if np.any(is_dust):
                    # Find original indices
                    full_indices = np.where(is_cont)[0]
                    dust_indices = full_indices[is_dust]
                    
                    step_flux[dust_indices] += balances[dust_indices]
                    balances[dust_indices] = 0
                    active_mask[dust_indices] = False
            
            flux_matrix[:, t] = step_flux
            
        return flux_matrix

class FinancialEngine:
    def __init__(self, principal, annual_rate, months):
        self.principal = principal
        self.rate = annual_rate / 12.0
        self.months = int(months)
        
        # Calculate Deterministic "Input Flux" (Standard Annuity Formula)
        if self.rate > 0:
            self.pmt = self.principal * (self.rate * (1 + self.rate)**self.months) / ((1 + self.rate)**self.months - 1)
        else:
            self.pmt = self.principal / self.months
            
        # Generate the Contractual Balance Curve (Exposure at Default)
        self.schedule = self._generate_schedule()

    def _generate_schedule(self):
        """Generates the contractual Balance vector for t=0 to t=M."""
        balance = self.principal
        balances = []
        for _ in range(self.months):
            balances.append(balance) # Balance at START of month (for payoff calc)
            interest = balance * self.rate
            principal_pay = self.pmt - interest
            balance -= principal_pay
            if balance < 0: balance = 0
        return np.array(balances)

    def run_monte_carlo(self, transition_matrices, n_sims=2000, lgd=0.5):
        """
        Performs Vectorized Monte Carlo Simulation.
        Args:
            transition_matrices: List of (4x4) matrices for each time step t.
            n_sims: Number of particles (paths).
            lgd: Loss Given Default (e.g., 0.5 means 50% recovery).
        Returns:
            flux_distribution: (n_sims, months) matrix of cash flows.
        """
        horizon = len(transition_matrices)
        
        # 1. Initialize Particles
        # State 0: Current
        current_states = np.zeros(n_sims, dtype=int) 
        
        # Cash Flow Matrix [Simulation x Time]
        cf_matrix = np.zeros((n_sims, horizon))
        
        # Active Mask (Loans that haven't entered absorbing states 2 or 3 yet)
        active_mask = np.ones(n_sims, dtype=bool)
        
        for t in range(horizon):
            if not np.any(active_mask): break
            
            # Balance at start of month t
            bal_t = self.schedule[t]
            
            # --- Vectorized Transition Logic ---
            T = transition_matrices[t] 
            
            # Generate random numbers for all active sims
            rng = np.random.rand(n_sims)
            next_states = current_states.copy()
            
            # Inverse Transform Sampling for 'Current' (0) particles
            mask_0 = (current_states == 0) & active_mask
            if np.any(mask_0):
                cum_probs = np.cumsum(T[0]) # [p_curr, p_late, p_def, p_prep]
                next_states[mask_0] = np.searchsorted(cum_probs, rng[mask_0])
                
            # Inverse Transform Sampling for 'Late' (1) particles
            mask_1 = (current_states == 1) & active_mask
            if np.any(mask_1):
                cum_probs = np.cumsum(T[1])
                next_states[mask_1] = np.searchsorted(cum_probs, rng[mask_1])
            
            # --- Cash Flow Calculation (The Reward Function) ---
            
            # 1. Contractual Payment: Active and landed in Current (0)
            payers = (next_states == 0) & active_mask
            cf_matrix[payers, t] = self.pmt
            
            # 2. Prepayment Event: Transitioned to 3
            prepayers = (next_states == 3) & (current_states != 3)
            cf_matrix[prepayers, t] = bal_t # Pay full balance
            
            # 3. Default Event: Transitioned to 2
            defaulters = (next_states == 2) & (current_states != 2)
            cf_matrix[defaulters, t] = bal_t * (1 - lgd) # Recovery
            
            # Update States and Mask
            current_states = next_states
            # Deactivate particles in state 2 or 3
            active_mask = active_mask & (current_states < 2)
            
        return cf_matrix
    
    def generate_vasicek_paths(r0, kappa, theta, sigma, T, n_sims, dt=1/12):
        """
        Simulates Interest Rate paths: dr_t = kappa * (theta - r_t)dt + sigma * dW_t
        """
        n_steps = int(T)
        rates = np.zeros((n_sims, n_steps))
        rates[:, 0] = r0
        
        for t in range(1, n_steps):
            # Brownian Motion
            dW = np.random.normal(0, np.sqrt(dt), size=n_sims)
            # SDE (Euler-Maruyama)
            dr = kappa * (theta - rates[:, t-1]) * dt + sigma * dW
            rates[:, t] = rates[:, t-1] + dr
            
        return np.maximum(rates, 0.0) # Floor at 0