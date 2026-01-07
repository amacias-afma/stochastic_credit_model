import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- Configuration ---
INPUT_FILE = 'data/raw/accepted_2007_to_2018Q4.csv'  # Verify your path
OUTPUT_FILE = 'data/processed/lending_club_panel.parquet'
CHUNK_SIZE = 10000 

# Model States (ZOIB Topology: 3 Regimes)
S_DEFAULT = 0    # Absorbing: Pays 0
S_CONTINUE = 1   # Transient: Pays Scheduled + Partial
S_PREPAID = 2    # Absorbing: Pays 100%

# --- 1. Market Data Engine (FRED) ---
def get_market_rate_lookup():
    """
    Downloads historical 30-Year Fixed Mortgage Rates from FRED.
    Returns a Dictionary: {(Year, Month): Rate} for fast lookup.
    Example: {(2015, 1): 0.0366}
    """
    print("Downloading Market Data (MORTGAGE30US) from St. Louis Fed...")
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        
        # Read CSV directly from memory
        df = pd.read_csv(io.StringIO(resp.content.decode('utf-8')), parse_dates=['DATE'])
        
        # Resample to Monthly Average and convert percent (4.5) to decimal (0.045)
        df.set_index('DATE', inplace=True)
        # Note: 'ME' is Month End (Pandas 2.0+). Use 'M' for older versions.
        df_monthly = df.resample('ME').mean() / 100.0
        
        # Create O(1) Lookup Dictionary
        lookup = {}
        for date, row in df_monthly.iterrows():
            if pd.notnull(row['MORTGAGE30US']):
                lookup[(date.year, date.month)] = float(row['MORTGAGE30US'])
            
        print(f"   Loaded {len(lookup)} months of historical rates.")
        return lookup
    except Exception as e:
        print(f"   Warning: Could not download market data ({e}). Using flat 4.5% fallback.")
        return None

# --- 2. Financial Math ---
def calculate_amortization(principal, rate, term, age):
    """Calculates theoretical remaining balance (Exposure)."""
    if rate <= 0 or term <= 0: return principal
    monthly_r = rate / 12.0
    
    # Boundary Check
    if age >= term: return 0.0
    
    # Standard Annuity Remaining Balance Formula
    numerator = (1 + monthly_r)**term - (1 + monthly_r)**age
    denominator = (1 + monthly_r)**term - 1
    
    # Prevent division by zero
    if denominator == 0: return 0.0
    
    balance = principal * (numerator / denominator)
    return max(0.0, balance)

# --- 3. ETL Logic ---
def process_chunk(chunk, rate_lookup):
    """
    Explodes static loan rows into monthly state observations.
    """
    # Filter valid rows
    chunk = chunk.dropna(subset=['issue_d', 'last_pymnt_d', 'loan_status', 'term', 'loan_amnt', 'int_rate']).copy()
    
    panel_rows = []
    
    for _, row in chunk.iterrows():
        try:
            # A. Parse Metadata
            start_date = datetime.strptime(row['issue_d'], "%b-%Y")
            end_date = datetime.strptime(row['last_pymnt_d'], "%b-%Y")
            
            if end_date < start_date: continue
            
            term_str = str(row['term']).strip()
            term_months = int(term_str.split()[0]) # " 36 months" -> 36
            
            # Features
            fico = float(row.get('fico_range_low', 700))
            dti = float(row.get('dti', 20))
            
            # Rate cleaning: " 10.5%" -> 0.105
            raw_rate = row['int_rate']
            # Handle mixed types (string or float)
            if isinstance(raw_rate, str):
                rate = float(raw_rate.strip().strip('%')) / 100.0
            else:
                rate = float(raw_rate) / 100.0
            
            principal = float(row['loan_amnt'])
            
            # B. Determine Path & Final State
            months_active = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            if months_active < 1: months_active = 1
            
            status = row['loan_status']
            final_state = S_CONTINUE # Default assumption
            
            # Check Absorbing States
            if status in ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"]:
                final_state = S_DEFAULT
            elif status in ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]:
                # If paid off >2 months early -> Prepayment (Option Exercise)
                maturity_date = start_date + relativedelta(months=term_months)
                if end_date < (maturity_date - relativedelta(months=2)):
                    final_state = S_PREPAID
                else:
                    final_state = S_PREPAID # Natural maturity logic (treated as successful exit)
            
            # Construct State Sequence
            loan_states = [S_CONTINUE] * (months_active + 1)
            loan_states[-1] = final_state
            
            # C. Generate Monthly Records
            curr_date = start_date
            
            # Iterate t from 0 to T-1
            for t in range(len(loan_states) - 1):
                state_curr = loan_states[t]
                state_next = loan_states[t+1] # <--- BUG FIX: Explicitly defined here
                
                # 1. Financials
                bal_curr = calculate_amortization(principal, rate, term_months, t)
                bal_next = calculate_amortization(principal, rate, term_months, t+1)
                
                # 2. Market Context (Dynamic Spread)
                market_rate = 0.045 # Default
                if rate_lookup:
                    market_rate = rate_lookup.get((curr_date.year, curr_date.month), 0.045)
                
                # Rate Spread: (My Rate - Market Rate)
                # Positive Spread = Incentive to Prepay (Market is cheaper than my loan)
                rate_spread = rate - market_rate
                
                # 3. Continuous Target (Y in [0, 1]) for ZOIB Model
                if state_next == S_DEFAULT:
                    y_fraction = 0.0
                elif state_next == S_PREPAID:
                    y_fraction = 1.0
                else:
                    # Continuing: Assume Scheduled Payment
                    # Fraction = (Principal + Interest) / Balance
                    monthly_interest = bal_curr * (rate / 12.0)
                    sched_principal = bal_curr - bal_next
                    total_pay = sched_principal + monthly_interest
                    
                    if bal_curr > 1.0:
                        y_fraction = total_pay / bal_curr
                    else:
                        y_fraction = 1.0 # Cleanup dust
                
                # Clip safely to strict [0, 1] interval
                y_fraction = float(np.clip(y_fraction, 0.0, 1.0))
                
                panel_rows.append({
                    'loan_id': str(row.get('id', '')),
                    'date': curr_date,
                    'loan_age': t,
                    'fico': fico,
                    'dti': dti,
                    'rate': rate,
                    'market_rate': market_rate,
                    'rate_spread': rate_spread,   # <--- The PhD Feature
                    'balance': bal_curr,
                    'current_state': state_curr,
                    'target_state': state_next,   # For Classifier (0,1,2)
                    'target_fraction': y_fraction # For Regressor [0,1]
                })
                
                curr_date += relativedelta(months=1)
                
                # Stop if absorbed
                if state_curr in [S_DEFAULT, S_PREPAID]: break
                
        except Exception as e:
            continue
            
    return pd.DataFrame(panel_rows)

def main():
    # 1. Load Market Data
    rate_lookup = get_market_rate_lookup()
    
    print(f"Reading {INPUT_FILE}...")
    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)
    
    first_chunk = True
    total_rows = 0
    
    for i, chunk in enumerate(reader):
        print(f"Processing Chunk {i+1}...", end='\r')
        
        df_panel = process_chunk(chunk, rate_lookup)
        
        if not df_panel.empty:
            if first_chunk:
                # First write: Create file
                df_panel.to_parquet(OUTPUT_FILE, engine='fastparquet', index=False)
                first_chunk = False
            else:
                # Append write (Requires fastparquet)
                try:
                    df_panel.to_parquet(OUTPUT_FILE, engine='fastparquet', append=True, index=False)
                except Exception as e:
                    print(f"\nError appending parquet (install fastparquet): {e}")
                    break
        
        total_rows += len(df_panel)
        
        # STOPPER: Remove this 'if' block to process the full dataset (2M+ loans)
        if i == 30: 
            print("\nStopping early for test run...")
            break
            
    print(f"\nDone! Generated {total_rows} monthly observations.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# import pandas as pd
# import numpy as np
# from datetime import datetime
# from dateutil.relativedelta import relativedelta

# # --- Configuration ---
# INPUT_FILE = 'data/raw/accepted_2007_to_2018Q4.csv'
# OUTPUT_FILE = 'data/processed/lending_club_panel.parquet'
# CHUNK_SIZE = 10000  # Process 10k loans at a time to manage RAM

# # Model States
# S_CURRENT = 0
# S_LATE = 1
# S_DEFAULT = 2  # Absorbing
# S_PREPAID = 3  # Absorbing

# def calculate_amortization(principal, rate, term, age):
#     """Calculates theoretical remaining balance (Exposure at Default)."""
#     if rate <= 0 or term <= 0: return principal
#     monthly_r = rate / 12.0
#     # Annuity Formula for Remaining Balance
#     balance = principal * ((1 + monthly_r)**term - (1 + monthly_r)**age) / ((1 + monthly_r)**term - 1)
#     return max(0, balance)

# def process_chunk(chunk):
#     """
#     Explodes static loan rows into monthly state observations with Targets.
#     """
#     # 1. Filter usable rows
#     chunk = chunk.dropna(subset=['issue_d', 'last_pymnt_d', 'loan_status', 'term']).copy()
    
#     panel_rows = []
    
#     for _, row in chunk.iterrows():
#         try:
#             # Parse Dates
#             start_date = datetime.strptime(row['issue_d'], "%b-%Y")
#             end_date = datetime.strptime(row['last_pymnt_d'], "%b-%Y")
#             if end_date < start_date: continue
            
#             # Parse Terms
#             term_months = int(row['term'].strip().split()[0]) # " 36 months" -> 36
#             maturity_date = start_date + relativedelta(months=term_months)
            
#             # Static Features
#             fico = row.get('fico_range_low', 700)
#             dti = row.get('dti', 20)
#             rate = float(str(row['int_rate']).strip('%')) / 100 if isinstance(row['int_rate'], str) else row['int_rate'] / 100
#             principal = row.get('loan_amnt', 0)
            
#             # --- Path Reconstruction ---
            
#             # 1. Determine Timeline Length
#             # How many months was the loan active?
#             months_active = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
#             if months_active == 0: months_active = 1
            
#             # 2. Determine Final Fate
#             status = row['loan_status']
#             final_state = S_CURRENT # Default censorship
            
#             if status in ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]:
#                 # If paid off >3 months early, it's Prepayment. Else Natural Maturity.
#                 if end_date < (maturity_date - relativedelta(months=3)):
#                     final_state = S_PREPAID
#                 else:
#                     final_state = S_PREPAID # Treated as Exit for survival analysis
            
#             elif status in ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"]:
#                 final_state = S_DEFAULT

#             # 3. Generate Monthly Sequence
#             # We build a list of states for this specific loan
#             loan_states = [S_CURRENT] * (months_active + 1)
            
#             # Apply Final State
#             loan_states[-1] = final_state
            
#             # Apply "Hidden" Delinquency Logic
#             # If Defaulted, overwrite the 4 months prior to be "Late"
#             if final_state == S_DEFAULT:
#                 for k in range(1, 5):
#                     if len(loan_states) > k:
#                         loan_states[-(k+1)] = S_LATE

#             # 4. Create Panel Rows (Input + Target)
#             curr_date = start_date
            
#             # We iterate up to len-1 because we need a Next State for the Target
#             for t in range(len(loan_states) - 1):
#                 state_t = loan_states[t]
#                 state_t_plus_1 = loan_states[t+1]
                
#                 # # Calculate Flux (Balance)
#                 # bal = calculate_amortization(principal, rate, term_months, t)
#                 # Calculate theoretical balances to determine the required payment
#                 bal_curr = calculate_amortization(principal, rate, term_months, t)
#                 bal_next = calculate_amortization(principal, rate, term_months, t+1)
#                 # Determine the Continuous Target (Y)
#                 if state_next == S_DEFAULT:
#                     y_fraction = 0.0
#                 elif state_next == S_PREPAID:
#                     y_fraction = 1.0
#                 else:
#                     # Continuing State
#                     # Payment = Principal_Reduc + Interest
#                     interest_payment = bal_curr * (rate / 12.0)
#                     principal_payment = bal_curr - bal_next
#                     total_payment = principal_payment + interest_payment
                    
#                     # Target = Fraction of current balance
#                     if bal_curr > 0:
#                         y_fraction = total_payment / bal_curr
#                     else:
#                         y_fraction = 1.0

#                 panel_rows.append({
#                     'loan_id': str(row['id']),
#                     'date': curr_date,
#                     'loan_age': t,
#                     'fico': fico,
#                     'dti': dti,
#                     'rate': rate,
#                     'balance': bal,
#                     'rate_spread': rate - 0.04, # Simple Market Rate Proxy
#                     'current_state': state_t,
#                     'balance': bal_curr,
#                     'target_fraction': float(np.clip(y_fraction, 0, 1))
#                 })
                
#                 curr_date += relativedelta(months=1)
                
#                 # Optimization: If we hit absorbing state, stop generating
#                 if state_t in [S_DEFAULT, S_PREPAID]: break

#         except Exception as e:
#             continue

#     return pd.DataFrame(panel_rows)

# def main():
#     print(f"Reading {INPUT_FILE}...")
    
#     # Read CSV in chunks
#     reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)
    
#     first_chunk = True
#     total_rows = 0
    
#     for i, chunk in enumerate(reader):
#         print(f"Processing Chunk {i+1}...", end='\r')
        
#         df_panel = process_chunk(chunk)
        
#         if not df_panel.empty:
#             # Append to Parquet
#             if first_chunk:
#                 df_panel.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)
#                 first_chunk = False
#             else:
#                 # Append requires reading table, so usually keeping separate files 
#                 # or using fastparquet append is better.
#                 # For simplicity here: we overwrite in 'append' mode if library supports, 
#                 # but standard pandas to_parquet doesn't support append easily.
#                 # BETTER APPROACH: Save 1 file per chunk, or accumulate in list if RAM allows.
#                 # Here we will append to a specific file structure if using fastparquet,
#                 # Or simply:
#                 df_panel.to_parquet(OUTPUT_FILE, engine='fastparquet', append=True)
        
#         total_rows += len(df_panel)
        
#         # STOPPER: Remove this to process full 2M loans
#         if i == 10: 
#             print("\nStopping early for test run...")
#             break
            
#     print(f"\nDone! Generated {total_rows} monthly observations.")
#     print(f"Saved to {OUTPUT_FILE}")

# if __name__ == "__main__":
#     main()