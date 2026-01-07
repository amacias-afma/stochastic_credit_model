import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.financial_engine import HybridFinancialEngine

def render_simulation_page(model, market_params):
    st.header("⚡ Stochastic Flux Simulation")
    st.markdown("Zero-One Inflated Impulse Control (ZOIB)")

    # --- SIDEBAR: Defined LOCALLY so it only appears on this page ---
    with st.sidebar:
        st.header("🎮 Scenario Controls")
        st.markdown("---")
        
        st.subheader("Loan Profile")
        principal = st.number_input("Principal ($)", 100000, 2000000, 300000, step=10000)
        fico = st.slider("FICO Score", 300, 850, 720)
        dti = st.slider("DTI Ratio", 0.1, 0.6, 0.35)
        
        st.subheader("Market Environment")
        # Use real calibrated rate as default
        default_rate = float(market_params.get('current_rate', 0.065))
        contract_rate = st.number_input("Contract Rate", 0.01, 0.15, 0.065, step=0.001, format="%.3f")
        sim_market_rate = st.slider("Market Rate (Sim)", 0.01, 0.15, default_rate, step=0.001, format="%.3f")
        
        # Calculate Spread
        spread = contract_rate - sim_market_rate
        st.metric("Incentive Spread", f"{spread:.2%}", 
                  help="Positive = Market is cheaper (Refi Incentive)")
        
        st.markdown("---")
        horizon = st.slider("Horizon (Months)", 12, 120, 36)
        n_sims = st.select_slider("Monte Carlo Paths", options=[500, 1000, 2000, 5000], value=1000)
        
        run_btn = st.button("🚀 Run Simulation", type="primary")

    # --- MAIN CONTENT ---
    if model is None:
        st.error("⚠️ Model file not found. Please run `src/etl_lendingclub.py` then `src/hybrid_model.py`.")
        return

    if run_btn:
        with st.spinner(f"Simulating {n_sims} stochastic paths..."):
            # 1. Initialize Engine
            engine = HybridFinancialEngine(principal, annual_rate=contract_rate, months=horizon)
            
            # 2. Build Feature Vector
            features = {
                'fico': fico, 
                'dti': dti, 
                'rate_spread': spread, 
                'balance': principal
            }
            
            # 3. Execute Vectorized Simulation
            flux = engine.run_simulation(model, features, n_sims=n_sims)
            
            # 4. Visualization & Analysis
            mean_flux = np.mean(flux, axis=0)
            p05 = np.percentile(flux, 5, axis=0)
            p95 = np.percentile(flux, 95, axis=0)
            x_axis = np.arange(1, horizon + 1)
            
            # Plot
            fig = go.Figure()
            # Cone of Uncertainty (90% CI)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([p95, p05[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% Confidence Interval'
            ))
            # Expected Path
            fig.add_trace(go.Scatter(x=x_axis, y=mean_flux, name='Expected Flux', 
                                   line=dict(width=3, color='#0068C9')))
            
            fig.update_layout(
                title="Stochastic Cash Flow Projection (ZOIB)",
                xaxis_title="Month",
                yaxis_title="Cash Flow ($)",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            total_expected = np.sum(mean_flux)
            c1, c2 = st.columns(2)
            c1.metric("Total Expected Cash Flow", f"${total_expected:,.2f}")
            c2.metric("Avg Monthly Payment", f"${np.mean(mean_flux):,.2f}")
            
    else:
        st.info("👈 Use the sidebar to configure the scenario and click 'Run Simulation'")
